import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
import modal
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import math
from functools import partial
from collections import deque

# modal configuration
N_GPUS = 8
GPU_TYPE = "H200"
app = modal.App("grpo-multi-gpu")

# hyperparams
CONTEXT_LEN = 32
HORIZON = 1
BATCH = 16384
MICRO_BATCH = 256
GRAD_ACCUM = BATCH // (MICRO_BATCH * N_GPUS)
EPOCHS = 1
DATASET_SIZE = 1_115_394
ITERS_PER_EPOCH = DATASET_SIZE // BATCH
TOTAL_ITERS = ITERS_PER_EPOCH * EPOCHS
LR = 3e-3
BETA_KL = 0.1
KL_WARM = int(DATASET_SIZE * 0.8)
KL_FREE_FRACTION = 0.1

N_LAYER = 6
N_HEAD = 6
N_EMB = 384

BUCKET_SIZE_MB = 30

USE_FLASH_ATTN = True
USE_EXHAUSTIVE = True
K_SAMPLES = 32
TEMPERATURE = 1.0
CLIP_RATIO = 0.5
ENTROPY_COEF = 0.08
MIN_VARIANCE = 0.1
USE_CONFIDENCE_SCALING = True
CONFIDENCE_WEIGHT = 0.7
CONFIDENCE_CLIP = 2.0
ENABLE_CONFIDENCE_PENALTY = True

USE_CONTINUOUS_BATCHING = False
MAX_BATCH_TOKENS = 8192
DYNAMIC_BATCH_SIZE = True

USE_PAGED_ATTENTION = False
PAGE_SIZE = 1024
MAX_PAGES_PER_SEQ = 8

# Learning rate scheduling
USE_LR_DECAY = True
LR_DECAY_TYPE = "cosine"  # Options: "cosine", "linear", "exponential"
MIN_LR = 1e-5
WARMUP_ITERS = 100

def norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm implementation using PyTorch built-in"""
    return F.rms_norm(x, (x.size(-1),))

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper for rotary embeddings"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

class RotaryCache(nn.Module):
    """Pre-computed rotary position embeddings"""
    def __init__(self, head_dim: int, max_len: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))
        t = torch.arange(max_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        sin, cos = freqs.sin(), freqs.cos()
        self.register_buffer("sin_base", sin, persistent=False)
        self.register_buffer("cos_base", cos, persistent=False)

    def forward(self, seq_len: int):
        sin = self.sin_base[:seq_len].repeat_interleave(2, dim=-1)
        cos = self.cos_base[:seq_len].repeat_interleave(2, dim=-1)
        return sin[None, None, :, :], cos[None, None, :, :]

class KVCache(nn.Module):
    """
    KV cache for efficient inference - caches past key and values during generation.
    Based on Meta's implementation for torchtune.
    """
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "cache_pos", torch.arange(0, cache_shape[2]), persistent=False
        )
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def reset(self) -> None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos -= self.size

    @property
    def size(self) -> int:
        return self.cache_pos[0].item()

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache with new k_val, v_val and return the updated cache.
        Args:
            k_val: Current key tensor with shape [B, H, S, D]
            v_val: Current value tensor with shape [B, H, S, D]
        Returns:
            Updated key and value cache tensors
        """
        bsz, _, seq_len, _ = k_val.shape
        if bsz > self.k_cache.shape[0]:
            raise ValueError(
                f"Cache batch size is {self.k_cache.shape[0]} but got {bsz}"
            )

        assert (self.cache_pos[0] + seq_len) <= self.k_cache.shape[2]
        
        k_out = self.k_cache
        v_out = self.v_cache

        k_out[:, :, self.cache_pos[:seq_len]] = k_val
        v_out[:, :, self.cache_pos[:seq_len]] = v_val

        # Update position tracker
        self.cache_pos.add_(seq_len)

        return k_out, v_out

class ReLUSquared(nn.Module):
    """ReLU squared activation - faster than GELU, better than plain ReLU"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()

class OptimizedAttention(nn.Module):
    """Multi-head attention with Flash Attention support and RoPE"""
    def __init__(self, n_emb: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.n_emb = n_emb
        self.head_dim = n_emb // n_head
        
        # Fused QKV projection for efficiency
        self.qkv = nn.Linear(n_emb, 3 * n_emb, bias=False)
        self.o_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Rotary embeddings
        max_seq = CONTEXT_LEN + HORIZON
        self.rope = RotaryCache(self.head_dim, max_seq)
        
        # Try to use Flash Attention
        self.use_flash_attn = False
        if USE_FLASH_ATTN:
            try:
                from flash_attn import flash_attn_qkvpacked_func
                self.flash_attn_func = flash_attn_qkvpacked_func
                self.use_flash_attn = True
            except ImportError:
                print("Flash Attention not available, using standard attention")
        
        self.use_paged_attention = False
        
        # KV cache for inference (not used during training)
        self.kv_cache = None
        self.cache_enabled = False
    
    def init_kv_cache(self, batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16):
        """Initialize KV cache for inference"""
        self.kv_cache = KVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.n_head,
            head_dim=self.head_dim,
            dtype=dtype
        )
        self.cache_enabled = True
    
    def reset_kv_cache(self):
        """Reset the KV cache"""
        if self.kv_cache is not None:
            self.kv_cache.reset()
    
    def disable_kv_cache(self):
        """Disable KV cache (for training)"""
        self.cache_enabled = False
        self.kv_cache = None
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        
        # Use KV cache if enabled and requested
        if use_cache and self.cache_enabled and self.kv_cache is not None:
            return self._forward_with_cache(x, mask)
        
        # Standard forward pass (for training)
        # Compute QKV in one go
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        
        if self.use_flash_attn:
            # Flash Attention path
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            
            # Apply RoPE
            sin, cos = self.rope(T)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
            
            # Apply QK normalization for stable training
            q, k = norm(q), norm(k)
            
            # Pack for Flash Attention
            packed = torch.stack((q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)), dim=2)
            
            # Ensure correct dtype
            dtype0 = packed.dtype
            if dtype0 not in (torch.float16, torch.bfloat16):
                packed = packed.to(torch.bfloat16)
                
            # Flash Attention
            out = self.flash_attn_func(packed, causal=True, dropout_p=0.1 if self.training else 0.0)
            out = out.to(dtype0).reshape(B, T, C)
        else:
            # Standard attention path
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            
            # Apply RoPE
            sin, cos = self.rope(T)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
            
            # QK normalization
            q, k = norm(q), norm(k)
            
            # Scaled dot-product attention with causal mask
            # Note: is_causal=True automatically applies causal masking
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(out)
    
    def _forward_with_cache(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using KV cache for efficient inference"""
        B, T, C = x.shape
        
        # Compute QKV
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Apply RoPE to current position
        cache_size = self.kv_cache.size
        sin, cos = self.rope(cache_size + T)
        # Only apply to new positions
        sin_new = sin[:, :, cache_size:cache_size+T, :]
        cos_new = cos[:, :, cache_size:cache_size+T, :]
        
        q = (q * cos_new) + (_rotate_half(q) * sin_new)
        k = (k * cos_new) + (_rotate_half(k) * sin_new)
        
        # Normalize
        q, k = norm(q), norm(k)
        
        # Update KV cache
        k_cache, v_cache = self.kv_cache.update(k, v)
        
        # Compute attention with cached keys/values
        # Get only the valid portion of cache
        valid_cache_size = self.kv_cache.size
        k_valid = k_cache[:, :, :valid_cache_size, :]
        v_valid = v_cache[:, :, :valid_cache_size, :]
        
        # Standard attention computation
        out = F.scaled_dot_product_attention(q, k_valid, v_valid, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(out)
    

class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture"""
    def __init__(self, n_emb: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attn = OptimizedAttention(n_emb, n_head, dropout)
        
        # Feed-forward network with ReLU squared
        self.ffn = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb, bias=False),
            ReLUSquared(),
            nn.Linear(4 * n_emb, n_emb, bias=False),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache: bool = False) -> torch.Tensor:
        # Pre-norm architecture with residual connections
        x = x + self.attn(norm(x), mask, use_cache=use_cache)
        x = x + self.ffn(norm(x))
        return x

class GPT(nn.Module):
    """GPT model optimized for multi-GPU training"""
    def __init__(self, vocab_size: int, n_layer: int = N_LAYER, n_head: int = N_HEAD, 
                 n_emb: int = N_EMB, context_len: int = CONTEXT_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.n_layer = n_layer
        
        # Token embeddings
        self.wte = nn.Embedding(vocab_size, n_emb)
        self.drop = nn.Dropout(0.1)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(n_emb, n_head, dropout=0.1)
            for _ in range(n_layer)
        ])
        
        # Output head with weight tying
        self.head = nn.Linear(n_emb, vocab_size, bias=False)
        # Weight tying - delete the head weight first to avoid issues
        del self.head.weight
        self.head.weight = self.wte.weight  # Share the embedding weights
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Pre-compute causal mask
        self.register_buffer("causal_mask", torch.triu(
            torch.ones(context_len + HORIZON, context_len + HORIZON), diagonal=1
        ).bool())
    
    def _init_weights(self, module):
        """Initialize weights with appropriate scaling"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, T = idx.shape
        
        # Token embeddings
        tok_emb = self.wte(idx)
        x = self.drop(tok_emb)
        
        # Get causal mask
        mask = self.causal_mask[:T, :T] if T <= self.context_len + HORIZON else None
        
        # Forward through transformer layers
        for layer in self.layers:
            x = layer(x, mask, use_cache=use_cache)
        
        # Final norm and output projection
        x = norm(x)
        logits = self.head(x)
        
        return logits
    
    def init_kv_caches(self, batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16):
        """Initialize KV caches for all attention layers"""
        for layer in self.layers:
            layer.attn.init_kv_cache(batch_size, max_seq_len, dtype)
    
    def reset_kv_caches(self):
        """Reset all KV caches"""
        for layer in self.layers:
            layer.attn.reset_kv_cache()
    
    def disable_kv_caches(self):
        """Disable all KV caches"""
        for layer in self.layers:
            layer.attn.disable_kv_cache()
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, 
                 top_k: Optional[int] = None, use_cache: bool = True) -> torch.Tensor:
        """
        Generate tokens using the model with optional KV caching.
        
        Args:
            idx: Starting context tokens [B, T]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If specified, only sample from top k tokens
            use_cache: Whether to use KV cache for efficient generation
        
        Returns:
            Generated tokens including the initial context
        """
        device = idx.device
        B, T = idx.shape
        
        # Initialize KV cache if requested
        if use_cache:
            self.init_kv_caches(B, T + max_new_tokens, dtype=idx.dtype)
        
        # Generate tokens
        generated = idx
        for _ in range(max_new_tokens):
            # Get logits for next token
            if use_cache and generated.shape[1] > T:
                # Only feed the new token(s) when using cache
                logits = self(generated[:, -1:], use_cache=True)
            else:
                # Feed full sequence (first iteration or no cache)
                # Crop to context length if needed
                idx_cond = generated if generated.shape[1] <= self.context_len else generated[:, -self.context_len:]
                logits = self(idx_cond, use_cache=use_cache)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat((generated, idx_next), dim=1)
        
        # Clean up cache
        if use_cache:
            self.disable_kv_caches()
        
        return generated

# ─── Reference Model for GRPO ───────────────────────────────────────────────
class OnTheFlyNGramRef(nn.Module):
    """N-gram reference model combining bigram, trigram, and fourgram predictions"""
    def __init__(self, text: str, stoi: Dict[str, int], vocab_size: int, smoothing: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        
        # Initialize count tensors with smoothing
        self.bigram_counts = torch.ones(vocab_size, vocab_size) * smoothing
        self.trigram_counts = torch.ones(vocab_size, vocab_size, vocab_size) * smoothing
        self.fourgram_counts = torch.ones(vocab_size, vocab_size, vocab_size, vocab_size) * smoothing
        
        # Convert text to indices
        text_indices = [stoi[c] for c in text]
        
        # Count bigrams
        for i in range(len(text_indices) - 1):
            self.bigram_counts[text_indices[i], text_indices[i+1]] += 1
            
        # Count trigrams
        for i in range(len(text_indices) - 2):
            self.trigram_counts[text_indices[i], text_indices[i+1], text_indices[i+2]] += 1
            
        # Count fourgrams
        for i in range(len(text_indices) - 3):
            self.fourgram_counts[text_indices[i], text_indices[i+1], text_indices[i+2], text_indices[i+3]] += 1
        
        # Compute log probabilities
        bigram_probs = self.bigram_counts / self.bigram_counts.sum(dim=-1, keepdim=True)
        self.register_buffer("bigram_log_probs", torch.log(bigram_probs + 1e-8).clamp(min=-20.0))
        
        trigram_probs = self.trigram_counts / self.trigram_counts.sum(dim=-1, keepdim=True)
        self.register_buffer("trigram_log_probs", torch.log(trigram_probs + 1e-8).clamp(min=-20.0))
        
        fourgram_probs = self.fourgram_counts / self.fourgram_counts.sum(dim=-1, keepdim=True)
        self.register_buffer("fourgram_log_probs", torch.log(fourgram_probs + 1e-8).clamp(min=-20.0))
        
    def forward(self, idx: torch.Tensor, return_components: bool = False) -> torch.Tensor:
        B, T = idx.shape
        device = idx.device
        
        # Initialize logits for each n-gram type
        bigram_logits = torch.zeros(B, T, self.vocab_size, device=device)
        trigram_logits = torch.zeros(B, T, self.vocab_size, device=device)
        fourgram_logits = torch.zeros(B, T, self.vocab_size, device=device)
        
        # Uniform logits for fallback
        uniform_logits = torch.zeros(self.vocab_size, device=device)
        
        for t in range(T):
            if t == 0:
                # First position: uniform distribution
                bigram_logits[:, t, :] = uniform_logits
                trigram_logits[:, t, :] = uniform_logits
                fourgram_logits[:, t, :] = uniform_logits
            elif t == 1:
                # Second position: only bigram available
                prev_char = idx[:, t-1]
                bigram_logits[:, t, :] = self.bigram_log_probs[prev_char]
                trigram_logits[:, t, :] = uniform_logits
                fourgram_logits[:, t, :] = uniform_logits
            elif t == 2:
                # Third position: bigram and trigram available
                prev_char = idx[:, t-1]
                prev_prev_char = idx[:, t-2]
                bigram_logits[:, t, :] = self.bigram_log_probs[prev_char]
                trigram_logits[:, t, :] = self.trigram_log_probs[prev_prev_char, prev_char]
                fourgram_logits[:, t, :] = uniform_logits
            else:
                # Fourth position and beyond: all n-grams available
                prev_char = idx[:, t-1]
                prev_prev_char = idx[:, t-2]
                prev_prev_prev_char = idx[:, t-3]
                bigram_logits[:, t, :] = self.bigram_log_probs[prev_char]
                trigram_logits[:, t, :] = self.trigram_log_probs[prev_prev_char, prev_char]
                fourgram_logits[:, t, :] = self.fourgram_log_probs[prev_prev_prev_char, prev_prev_char, prev_char]
        
        # Combine n-gram predictions with equal weights
        weights = torch.tensor([1.0, 1.0, 1.0], device=device)
        weights = weights / weights.sum()
        
        # Stack and combine logits
        stacked_logits = torch.stack([bigram_logits, trigram_logits, fourgram_logits], dim=0)
        log_weights = torch.log(weights + 1e-10).view(-1, 1, 1, 1)
        weighted_logits = stacked_logits + log_weights
        combined_logits = torch.logsumexp(weighted_logits, dim=0)
        
        if return_components:
            return combined_logits, {
                'bigram': bigram_logits,
                'trigram': trigram_logits,
                'fourgram': fourgram_logits
            }
        
        return combined_logits

# ─── Distributed Setup Functions ────────────────────────────────────────────
def setup_distributed():
    """Initialize distributed training environment"""
    # Get environment variables set by torchrun
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set device - Modal guarantees GPUs
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize process group
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        
    return rank, world_size, local_rank, device

def cleanup_distributed():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_buckets(params: List[torch.nn.Parameter], bucket_size_mb: float = 25) -> List[List[torch.nn.Parameter]]:
    """
    Group parameters into buckets for gradient bucketing (from modded-nanogpt)
    This improves communication efficiency by batching gradients
    """
    buckets = []
    current_bucket = []
    current_size = 0
    
    for param in params:
        param_size = param.numel() * param.element_size() / 1024 / 1024  # Size in MB
        
        if current_size + param_size > bucket_size_mb and current_bucket:
            buckets.append(current_bucket)
            current_bucket = []
            current_size = 0
            
        current_bucket.append(param)
        current_size += param_size
    
    if current_bucket:
        buckets.append(current_bucket)
    
    return buckets

def setup_ddp_with_gradient_bucketing(model: nn.Module, device: torch.device, bucket_size_mb: float = 25) -> DDP:
    """
    Setup DistributedDataParallel with gradient bucketing for efficient communication
    """
    # Create parameter buckets
    params = list(model.parameters())
    buckets = create_buckets(params, bucket_size_mb)
    
    # Setup DDP with bucketing
    ddp_model = DDP(
        model,
        device_ids=[device.index],
        bucket_cap_mb=bucket_size_mb,
        gradient_as_bucket_view=True,  # Memory optimization
        static_graph=True,  # Enable if model structure doesn't change
    )
    
    # Register gradient compression hook for even more efficient communication
    if dist.is_initialized() and dist.get_world_size() > 1:
        ddp_model.register_comm_hook(
            state=None,
            hook=comm_hooks.fp16_compress_hook  # FP16 gradient compression
        )
    
    return ddp_model

def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all processes and compute mean"""
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor

def print_rank0(msg: str, rank: int):
    """Print only on rank 0 to avoid duplicate outputs"""
    if rank == 0:
        print(msg)

# ─── Data Loading for Distributed Training ──────────────────────────────────
# Step 4: Implement optimized data loading for distributed GRPO

def ensure_dataset(filepath: str = "input.txt") -> None:
    """Download TinyShakespeare dataset if not present"""
    if not Path(filepath).exists():
        import requests
        DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        text = requests.get(DATA_URL, timeout=10).text
        Path(filepath).write_text(text, encoding="utf-8")
        
def build_vocab(filepath: str = "input.txt") -> Tuple:
    """Build character-level vocabulary from text"""
    text = Path(filepath).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    
    def encode(s: str) -> torch.Tensor:
        return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    
    def decode(t: torch.Tensor) -> str:
        return "".join(itos[int(i)] for i in t)
    
    return encode, decode, len(chars), stoi, itos, text

def build_bigram_counts(text: str, stoi: Dict[str, int]) -> Dict[Tuple[int, int], int]:
    """Build bigram counts for reference model"""
    from collections import defaultdict
    counts = defaultdict(int)
    for i in range(len(text) - 1):
        c1 = stoi[text[i]]
        c2 = stoi[text[i + 1]]
        counts[(c1, c2)] += 1
    return counts

class DistributedContextualTextLoader:
    """
    Distributed data loader for GRPO training
    - Each GPU gets the full dataset but starts at different positions
    - Provides contextual windows for next-token prediction
    - Matches train.py behavior: no sharding, just different starting positions
    """
    def __init__(self, text: str, encode_fn, batch_size: int, horizon: int, 
                 context_len: int, rank: int, world_size: int, device: torch.device,
                 use_pinned_memory: bool = False, prefetch_factor: int = 1):
        # Encode full text once and move to device
        self.data = encode_fn(text).to(device)  # Move entire dataset to GPU since it's small
        self.batch_size = batch_size
        self.horizon = horizon
        self.context_len = context_len
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        # Following train.py pattern: each GPU gets full data but different starting position
        # This provides diversity while allowing overlapping context windows
        total_len = len(self.data)
        
        # Start position staggered by rank to ensure diversity
        # Each rank starts at a different position in the dataset
        offset_per_rank = (total_len - context_len - batch_size * horizon) // max(1, world_size)
        self.pos = context_len + (rank * offset_per_rank)
        
        # Ensure we don't start too close to the end
        max_start = total_len - batch_size * horizon - 1
        self.pos = min(self.pos, max_start)
        
    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch - exactly like train.py"""
        # Wrap around if we don't have enough data left
        if self.pos + self.batch_size * self.horizon + 1 > len(self.data):
            self.pos = self.context_len
            
        contexts = []
        targets = []
        
        for b in range(self.batch_size):
            # Get context window using sliding approach (same as train.py)
            ctx_start = self.pos - self.context_len + b * self.horizon
            ctx_end = self.pos + b * self.horizon
            
            context = self.data[ctx_start:ctx_end]
            target = self.data[ctx_end:ctx_end + self.horizon]
            
            contexts.append(context)
            targets.append(target)
        
        self.pos += self.batch_size * self.horizon
        
        # Data is already on device, so just stack
        return torch.stack(contexts), torch.stack(targets)
    
    def state_dict(self) -> Dict:
        """Save loader state for checkpointing"""
        return {
            'pos': self.pos,
            'rank': self.rank
        }
    
    def load_state_dict(self, state: Dict):
        """Restore loader state from checkpoint"""
        self.pos = state['pos']
        # Verify rank consistency
        if state['rank'] != self.rank:
            print(f"Warning: Loading state from rank {state['rank']} to rank {self.rank}")

class DistributedDataGenerator:
    """
    Simple data generator for non-contextual scenarios
    Each GPU gets a non-overlapping subset of the data
    """
    def __init__(self, data: torch.Tensor, batch_size: int, rank: int, world_size: int):
        self.data = data
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        
        # Calculate shard boundaries
        total_len = len(data)
        shard_size = total_len // world_size
        self.start_idx = rank * shard_size
        self.end_idx = (rank + 1) * shard_size if rank < world_size - 1 else total_len
        self.shard_data = data[self.start_idx:self.end_idx]
        
        self.pos = 0
        
    def next_batch(self) -> torch.Tensor:
        """Get next batch for this rank"""
        if self.pos + self.batch_size > len(self.shard_data):
            self.pos = 0
            
        batch = self.shard_data[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        
        return batch

# ─── GRPO-Specific Components ───────────────────────────────────────────────
# Step 3: GRPO algorithm components adapted from train.py


@torch.no_grad()
def generate_exhaustive_single_char(model: nn.Module, contexts: torch.Tensor, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ALL possible next characters (exhaustive exploration)
    Returns both the characters and their log probabilities
    """
    B = contexts.shape[0]
    device = contexts.device
    
    # Get model predictions for the contexts
    ctx_window = contexts[:, -model.context_len:] if contexts.shape[1] > model.context_len else contexts
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(ctx_window)[:, -1, :]  # [B, V]
    
    log_probs = F.log_softmax(logits, dim=-1)  # [B, V]
    
    # Create all possible next characters
    all_chars = torch.arange(vocab_size, device=device).unsqueeze(0).expand(B, -1)  # [B, V]
    
    return all_chars, log_probs

@torch.no_grad()
def generate_with_temperature(model: nn.Module, contexts: torch.Tensor, horizon: int, k_samples: int, 
                            vocab_size: int, temperature: float = 1.0) -> torch.Tensor:
    """
    Generate K samples with temperature-based sampling
    For multi-character generation
    """
    B = contexts.shape[0]
    
    # Expand contexts for K samples
    ctx = contexts.repeat_interleave(k_samples, dim=0)  # [B*K, context_len]
    
    # Generate with temperature
    for _ in range(horizon):
        ctx_window = ctx[:, -model.context_len:] if ctx.shape[1] > model.context_len else ctx
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(ctx_window)[:, -1, :] / temperature
        
        # Mask out padding tokens if vocab was padded
        logits[:, vocab_size:] = -1e9
        
        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, 1)
        ctx = torch.cat([ctx, next_char], dim=1)
    
    # Reshape to [B, K, horizon]
    generated = ctx[:, -horizon:].reshape(B, k_samples, horizon)
    return generated

def compute_exhaustive_rewards(all_chars: torch.Tensor, ref_char: torch.Tensor, ref_model: nn.Module, 
                             ctx: torch.Tensor, vocab_size: int, model_log_probs: Optional[torch.Tensor] = None,
                             old_probs: Optional[torch.Tensor] = None, 
                             use_confidence_scaling: bool = True,
                             confidence_weight: float = 0.5,
                             confidence_clip: float = 2.0) -> torch.Tensor:
    """
    Compute rewards for exhaustive exploration with confidence scaling
    Simplified to match train_batching_optimized.py approach
    """
    B = ctx.shape[0]
    device = ctx.device
    
    # Initialize base rewards (0 everywhere, 1.0 for correct character)
    base_rewards = torch.zeros(B, vocab_size, device=device)
    
    # Exact match reward
    for b in range(B):
        base_rewards[b, ref_char[b]] = 1.0
    
    # Add partial rewards from reference model
    with torch.no_grad():
        ref_logits = ref_model(ctx)[:, -1, :]
        ref_probs = F.softmax(ref_logits, dim=-1)
        partial_rewards = ref_probs * 0.1  # Small partial reward
        base_rewards = base_rewards + partial_rewards
    
    # Model confidence bonus (if model log probs provided)
    if model_log_probs is not None:
        model_probs = torch.exp(model_log_probs)
        correct_char_bonus = torch.zeros_like(base_rewards)
        for b in range(B):
            correct_char_bonus[b, ref_char[b]] = model_probs[b, ref_char[b]] * 2.0
        base_rewards = base_rewards + correct_char_bonus
    
    rewards = base_rewards
    
    # Confidence scaling (same as optimized version)
    if use_confidence_scaling and old_probs is not None:
        confidence = old_probs[torch.arange(B), ref_char]
        confidence_scale = 1.0 + confidence_weight * confidence.clamp(0, 1)
        confidence_scale = confidence_scale.clamp(1.0, confidence_clip)
        rewards[torch.arange(B), ref_char] *= confidence_scale
        
        # Confidence penalty for wrong but confident predictions
        if ENABLE_CONFIDENCE_PENALTY:
            top_k = min(10, vocab_size)
            top_probs, top_indices = old_probs.topk(top_k, dim=-1)
            wrong_mask = top_indices != ref_char.unsqueeze(1)
            confident_mask = top_probs > 0.5
            penalty_mask = wrong_mask & confident_mask
            penalties = -0.1 * top_probs * penalty_mask.float()
            rewards.scatter_add_(1, top_indices, penalties)
    
    # Normalize rewards to have zero mean per batch
    rewards = rewards - rewards.mean(dim=1, keepdim=True)
    
    return rewards

def compute_rewards_multi_char(gen: torch.Tensor, ref: torch.Tensor, ref_logits: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """
    Compute rewards for multi-character generation with detailed metrics
    Matches train_batching_optimized.py implementation
    """
    # Exact match reward
    exact_match = (gen == ref).float()
    
    # Partial credit based on reference probability
    with torch.no_grad():
        ref_probs = F.softmax(ref_logits, dim=-1)
        # Get probability of generated token under reference
        B, T = gen.shape
        gen_probs = ref_probs.gather(2, gen.unsqueeze(-1)).squeeze(-1)
        # Partial reward from reference model
        partial_reward = gen_probs
    
    # Combined reward (matching train_batching_optimized.py)
    reward = exact_match + partial_reward
    
    metrics = {
        "accuracy": exact_match.mean().item(),
        "avg_gen_prob": gen_probs.mean().item(),
        "reward_mean": reward.mean().item(),
        "reward_std": reward.std().item(),
    }
    
    return reward, metrics

class AdaptiveKLController:
    """Adaptive KL coefficient based on current KL divergence"""
    def __init__(self, target_kl: float, horizon: int = 10):
        self.target_kl = target_kl
        self.horizon = horizon
        self.kl_history = deque(maxlen=horizon)
        
    def update(self, current_kl: float, current_coef: float) -> float:
        self.kl_history.append(current_kl)
        
        if len(self.kl_history) < self.horizon:
            return current_coef
        
        avg_kl = np.mean(self.kl_history)
        
        if avg_kl > 2 * self.target_kl:
            new_coef = current_coef * 1.5
        elif avg_kl > 1.5 * self.target_kl:
            new_coef = current_coef * 1.1
        elif avg_kl < 0.5 * self.target_kl:
            new_coef = current_coef * 0.7
        elif avg_kl < 0.75 * self.target_kl:
            new_coef = current_coef * 0.9
        else:
            new_coef = current_coef
        
        return max(1e-5, min(1.0, new_coef))

class ConfidenceMonitor:
    """Monitor model confidence and calibration"""
    def __init__(self, calibration_threshold: float = 0.15, confidence_collapse_threshold: float = 0.8):
        self.calibration_threshold = calibration_threshold
        self.confidence_collapse_threshold = confidence_collapse_threshold
        self.confidence_history = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        self.gradient_history = deque(maxlen=50)
        
    def update(self, confidence: float, accuracy: float, gradient_norm: Optional[float] = None):
        self.confidence_history.append(confidence)
        self.accuracy_history.append(accuracy)
        if gradient_norm is not None:
            self.gradient_history.append(gradient_norm)
    
    def check_calibration(self) -> Tuple[bool, float]:
        if len(self.confidence_history) < 50:
            return True, 0.0
        
        avg_confidence = np.mean(self.confidence_history)
        avg_accuracy = np.mean(self.accuracy_history)
        calibration_error = abs(avg_confidence - avg_accuracy)
        
        is_calibrated = calibration_error < self.calibration_threshold
        return is_calibrated, calibration_error
    
    def check_confidence_collapse(self) -> Tuple[bool, float]:
        if len(self.confidence_history) < 20:
            return False, 0.0
        
        recent_confidence = list(self.confidence_history)[-20:]
        high_conf_ratio = sum(c > self.confidence_collapse_threshold for c in recent_confidence) / len(recent_confidence)
        
        return high_conf_ratio > 0.7, high_conf_ratio

class CurriculumManager:
    """Manages curriculum learning stages and progression"""
    def __init__(self, 
                 n_stages: int = 4,
                 ema_alpha: float = 0.95,
                 promotion_threshold: float = 0.05,
                 promotion_patience: int = 100,
                 min_iterations_per_stage: int = 500):
        self.n_stages = n_stages
        self.current_stage = 1
        self.ema_alpha = ema_alpha
        self.promotion_threshold = promotion_threshold
        self.promotion_patience = promotion_patience
        self.min_iterations_per_stage = min_iterations_per_stage
        
        # EMA tracking for key metrics
        self.ema_metrics = {
            'accuracy': 0.0,
            'reward_mean': 0.0,
            'kl_divergence': 0.0,
            'loss': float('inf')
        }
        
        # Stage progression tracking
        self.iterations_in_stage = 0
        self.plateau_counter = 0
        self.best_ema_reward = -float('inf')
        self.stage_start_metrics = {}
        
        # Stage-specific configurations
        self.stage_configs = {
            1: {  # Character patterns
                'name': 'Character Bigrams',
                'ngram_weights': [0.7, 0.2, 0.1],  # bigram, trigram, fourgram
                'reward_scale': 1.0,
                'kl_weight': 0.05,
                'confidence_weight': 0.3,
                'horizon_max': 4,
                'focus': 'Learning basic character patterns and transitions'
            },
            2: {  # Word formation
                'name': 'Trigrams & Words',
                'ngram_weights': [0.3, 0.5, 0.2],
                'reward_scale': 1.2,
                'kl_weight': 0.08,
                'confidence_weight': 0.5,
                'horizon_max': 6,
                'focus': 'Forming valid character sequences and simple words'
            },
            3: {  # Lexical validity
                'name': 'Lexical Validity',
                'ngram_weights': [0.2, 0.4, 0.4],
                'reward_scale': 1.5,
                'kl_weight': 0.1,
                'confidence_weight': 0.7,
                'horizon_max': 8,
                'lexical_bonus': True,
                'focus': 'Generating valid Shakespearean words and phrases'
            },
            4: {  # Full optimization
                'name': 'Full Optimization',
                'ngram_weights': [0.15, 0.35, 0.5],
                'reward_scale': 2.0,
                'kl_weight': 0.1,
                'confidence_weight': 0.7,
                'horizon_max': 8,
                'lexical_bonus': True,
                'semantic_bonus': True,
                'focus': 'Optimizing for perplexity with all features'
            }
        }
        
    def update_metrics(self, metrics: Dict[str, float]):
        """Update EMA metrics"""
        for key in self.ema_metrics:
            if key in metrics:
                if self.iterations_in_stage == 0:
                    self.ema_metrics[key] = metrics[key]
                else:
                    self.ema_metrics[key] = (self.ema_alpha * self.ema_metrics[key] + 
                                            (1 - self.ema_alpha) * metrics[key])
        
        self.iterations_in_stage += 1
        
    def check_promotion(self) -> Tuple[bool, str]:
        """Check if ready to promote to next stage"""
        if self.current_stage >= self.n_stages:
            return False, "Already at final stage"
            
        # Check minimum iterations
        if self.iterations_in_stage < self.min_iterations_per_stage:
            return False, f"Need {self.min_iterations_per_stage - self.iterations_in_stage} more iterations"
        
        # Check for improvement plateau
        current_reward = self.ema_metrics['reward_mean']
        improvement = abs(current_reward - self.best_ema_reward) / (abs(self.best_ema_reward) + 1e-6)
        
        if improvement < self.promotion_threshold:
            self.plateau_counter += 1
        else:
            self.plateau_counter = 0
            self.best_ema_reward = current_reward
            
        if self.plateau_counter >= self.promotion_patience:
            return True, f"Plateau detected: {self.plateau_counter} iterations without {self.promotion_threshold:.1%} improvement"
            
        # Stage-specific promotion criteria
        stage_config = self.get_stage_config()
        if self.current_stage == 1 and self.ema_metrics['accuracy'] > 0.3:
            return True, "Stage 1 accuracy threshold reached (>30%)"
        elif self.current_stage == 2 and self.ema_metrics['accuracy'] > 0.5:
            return True, "Stage 2 accuracy threshold reached (>50%)"
        elif self.current_stage == 3 and self.ema_metrics['accuracy'] > 0.65:
            return True, "Stage 3 accuracy threshold reached (>65%)"
            
        return False, f"Continuing stage {self.current_stage}"
        
    def promote(self):
        """Promote to next stage"""
        if self.current_stage < self.n_stages:
            # Save current stage end metrics
            self.stage_start_metrics[self.current_stage] = self.ema_metrics.copy()
            
            # Move to next stage
            self.current_stage += 1
            self.iterations_in_stage = 0
            self.plateau_counter = 0
            self.best_ema_reward = self.ema_metrics['reward_mean']
            
            return True
        return False
        
    def get_stage_config(self) -> Dict:
        """Get current stage configuration"""
        return self.stage_configs[self.current_stage]
        
    def get_reward_scale(self) -> float:
        """Get reward scaling factor for current stage"""
        return self.stage_configs[self.current_stage]['reward_scale']
        
    def get_horizon(self, base_horizon: int) -> int:
        """Get horizon for current stage"""
        max_horizon = self.stage_configs[self.current_stage]['horizon_max']
        return min(base_horizon, max_horizon)
        
    def get_ngram_weights(self) -> List[float]:
        """Get n-gram weights for current stage"""
        return self.stage_configs[self.current_stage]['ngram_weights']
        
    def get_state_dict(self) -> Dict:
        """Get state for checkpointing"""
        return {
            'current_stage': self.current_stage,
            'iterations_in_stage': self.iterations_in_stage,
            'ema_metrics': self.ema_metrics,
            'plateau_counter': self.plateau_counter,
            'best_ema_reward': self.best_ema_reward,
            'stage_start_metrics': self.stage_start_metrics
        }
        
    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint"""
        self.current_stage = state_dict['current_stage']
        self.iterations_in_stage = state_dict['iterations_in_stage']
        self.ema_metrics = state_dict['ema_metrics']
        self.plateau_counter = state_dict['plateau_counter']
        self.best_ema_reward = state_dict['best_ema_reward']
        self.stage_start_metrics = state_dict['stage_start_metrics']

def compute_ppo_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor, 
                    clip_ratio: float = 0.5) -> torch.Tensor:
    """
    Compute PPO clipped objective
    """
    ratio = torch.exp(logp_new - logp_old)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    
    pol_loss1 = -advantages * ratio
    pol_loss2 = -advantages * clipped_ratio
    pol_loss = torch.max(pol_loss1, pol_loss2).mean()
    
    return pol_loss, ratio

def normalize_advantages(advantages: torch.Tensor, min_variance: float = 0.1) -> torch.Tensor:
    """
    Normalize advantages with minimum variance threshold
    """
    adv_mean = advantages.mean(dim=1, keepdim=True)
    adv_std = torch.maximum(advantages.std(dim=1, keepdim=True), torch.tensor(min_variance, device=advantages.device))
    normalized_adv = (advantages - adv_mean) / adv_std
    return normalized_adv

# ─── Memory and Performance Optimizations ───────────────────────────────────
def setup_cuda_optimizations():
    """Configure CUDA settings for optimal performance"""
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set float32 matmul precision
    torch.set_float32_matmul_precision("high")
    
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    
    # Optimize memory allocation
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Don't set NCCL environment variables - let Modal/torchrun handle it

# ─── Model Synchronization Utilities ────────────────────────────────────────
def sync_model_across_ranks(model: nn.Module, src_rank: int = 0):
    """
    Synchronize model parameters across all ranks
    Used for syncing actor_old with actor
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
        
    for param in model.parameters():
        dist.broadcast(param.data, src=src_rank)

def average_gradients_across_ranks(model: nn.Module):
    """
    Manually average gradients across all ranks
    Useful for models not wrapped in DDP
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
        
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size

# ─── Checkpoint Management for Distributed Training ──────────────────────────
def save_checkpoint_rank0(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    checkpoint_path: str,
    rank: int,
    additional_state: Optional[Dict] = None
):
    """Save checkpoint only on rank 0"""
    if rank == 0:
        checkpoint = {
            "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        }
        if additional_state:
            checkpoint.update(additional_state)
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint_all_ranks(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
    rank: int
) -> Dict:
    """Load checkpoint on all ranks with proper synchronization"""
    # Rank 0 loads first
    if rank == 0:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = None
    
    # Broadcast checkpoint existence
    checkpoint_exists = torch.tensor([1 if checkpoint is not None else 0], device=device)
    if dist.is_initialized():
        dist.broadcast(checkpoint_exists, src=0)
    
    if checkpoint_exists.item() == 0:
        return {}
    
    # Load on all ranks
    if rank != 0:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint

# ─── Modal Image and Volume Setup ───────────────────────────────────────────
flash_attn_wheel = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch==2.5.0", "tqdm", "wandb", "requests", "matplotlib", "nvidia-ml-py3")
    .pip_install(flash_attn_wheel)
)

volume = modal.Volume.from_name("grpo-data", create_if_missing=True)

# ─── Main Entry Point (Placeholder for now) ─────────────────────────────────
@app.function(
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    volumes={"/data": volume},
    timeout=60 * 60 * 6,
    image=image,
    secrets=[modal.Secret.from_name("wandb")],
)
def train_distributed():
    """Launch distributed training with torchrun - following modal_nanogpt_simple.py pattern"""
    import subprocess
    
    print(f"Launching distributed training on {N_GPUS} {GPU_TYPE} GPUs")
    
    # Download dataset to current directory like train.py
    ensure_dataset("input.txt")
    
    # Copy this script to a temporary location
    script_content = Path(__file__).read_text()
    temp_script = "/tmp/train_multi_gpu.py"
    Path(temp_script).write_text(script_content)
    
    # Run distributed training - simple like modal_nanogpt_simple.py
    subprocess.run([
        "torchrun", f"--nproc-per-node={N_GPUS}", temp_script
    ], check=True)
    
    return "Distributed training test completed successfully!"


def test_distributed_setup():
    """Test function to verify distributed setup works"""
    try:
        rank, world_size, local_rank, device = setup_distributed()
        
        print_rank0(f"\n=== Testing Distributed Setup ===", rank)
        print_rank0(f"Initialized distributed training:", rank)
        print_rank0(f"  World size: {world_size}", rank)
        print_rank0(f"  Rank: {rank}, Local rank: {local_rank}", rank)
        print_rank0(f"  Device: {device}", rank)
        
        # Setup CUDA optimizations (without NCCL env vars)
        setup_cuda_optimizations()
        
        # Test tensor synchronization
        if world_size > 1:
            test_tensor = torch.tensor([rank], dtype=torch.float32, device=device)
            print(f"Rank {rank} before all_reduce: {test_tensor.item()}")
            
            all_reduce_mean(test_tensor)
            print(f"Rank {rank} after all_reduce (should be average): {test_tensor.item()}")
            
            # Test broadcast
            if rank == 0:
                broadcast_tensor = torch.tensor([42.0], device=device)
            else:
                broadcast_tensor = torch.zeros(1, device=device)
                
            dist.broadcast(broadcast_tensor, src=0)
            print(f"Rank {rank} broadcast result (should be 42): {broadcast_tensor.item()}")
        
        # ALL RANKS must create the model for DDP
        print_rank0("\n=== Creating Model on All Ranks ===", rank)
        vocab_size = 65  # TinyShakespeare vocabulary
        
        # Create model on ALL ranks
        model = GPT(vocab_size).to(device)
        if rank == 0:
            print(f"Created GPT model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test DDP wrapper on ALL ranks if multi-GPU
        if world_size > 1:
            print_rank0("\n=== Testing DDP Wrapper ===", rank)
            ddp_model = setup_ddp_with_gradient_bucketing(model, device)
            print_rank0("Successfully wrapped model in DDP with gradient bucketing", rank)
            
            # Synchronize before testing
            if dist.is_initialized():
                dist.barrier()
        
        # Test data loading on ALL ranks
        print_rank0("\n=== Testing Distributed Data Loading ===", rank)
        
        # Load dataset and vocab
        ensure_dataset()
        encode, decode, vocab_size_actual, stoi, itos, text = build_vocab()
        
        # Test distributed data loader
        loader = DistributedContextualTextLoader(
            text=text,
            encode_fn=encode,
            batch_size=MICRO_BATCH,
            horizon=HORIZON,
            context_len=CONTEXT_LEN,
            rank=rank,
            world_size=world_size,
            device=device
        )
        
        # Each rank gets full data but different starting position
        ctx, tgt = loader.next()
        print(f"Rank {rank} - Data loader test: ctx shape {ctx.shape}, tgt shape {tgt.shape}")
        print(f"Rank {rank} - Starting position: {loader.pos}")
        
        # Just print local info - avoid all_gather which might be causing NCCL issues
        print(f"Rank {rank} - Dataset length: {len(loader.data)}, Start position: {loader.pos}")
        
        # Only rank 0 performs model testing
        if rank == 0:
            print("\n=== Testing Model Forward Pass ===")
            batch_size = 4
            seq_len = CONTEXT_LEN
            
            # Test forward pass
            dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            with torch.no_grad():
                if world_size > 1:
                    output = ddp_model(dummy_input)
                else:
                    output = model(dummy_input)
            print(f"Model output shape: {output.shape} (expected: {(batch_size, seq_len, vocab_size)})")
            
            # Test OnTheFlyNGramRef model
            ref_model = OnTheFlyNGramRef(text, stoi, vocab_size).to(device)
            ref_output = ref_model(dummy_input)
            print(f"Reference model output shape: {ref_output.shape}")
            
            # Check Flash Attention
            attn_module = model.layers[0].attn
            print(f"Flash Attention enabled: {attn_module.use_flash_attn}")
            
            # Test GRPO reward computation
            print("\n=== Testing GRPO Components ===")
            all_chars, log_probs = generate_exhaustive_single_char(model, ctx, vocab_size)
            print(f"Exhaustive generation: all_chars shape {all_chars.shape}, log_probs shape {log_probs.shape}")
            
            ref_char = tgt[:, 0]  # First character of target
            rewards = compute_exhaustive_rewards(
                all_chars, ref_char, ref_model, ctx, vocab_size,
                model_log_probs=log_probs
            )
            print(f"Rewards shape: {rewards.shape}, mean: {rewards.mean():.3f}, std: {rewards.std():.3f}")
        
        print_rank0("\nAll tests complete!", rank)
        
    except Exception as e:
        print(f"Error on rank {os.environ.get('RANK', 0)}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        cleanup_distributed()

def train_grpo():
    """Main GRPO training loop with multi-GPU support"""
    # Setup distributed training
    rank, world_size, local_rank, device = setup_distributed()
    
    # Setup CUDA optimizations
    setup_cuda_optimizations()
    
    # Load dataset
    ensure_dataset()
    encode, decode, vocab_size, stoi, itos, text = build_vocab()
    
    print_rank0(f"\n=== Starting GRPO Training ===", rank)
    print_rank0(f"World size: {world_size} GPUs", rank)
    print_rank0(f"Vocabulary size: {vocab_size}", rank)
    print_rank0(f"Dataset size: {len(text):,} characters", rank)
    print_rank0(f"Micro batch size: {MICRO_BATCH} samples per GPU", rank)
    print_rank0(f"Gradient accumulation steps: {GRAD_ACCUM}", rank)
    print_rank0(f"Total batch size: {BATCH} samples", rank)
    print_rank0(f"Effective batch per GPU: {MICRO_BATCH * GRAD_ACCUM} samples", rank)
    
    # Initialize models
    ref_model = OnTheFlyNGramRef(text, stoi, vocab_size).to(device).eval()
    
    # Synchronize before model creation to ensure all ranks have same initialization
    if world_size > 1 and dist.is_initialized():
        dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
    
    # Create actor model with deterministic initialization
    torch.manual_seed(42)  # SAME seed for all ranks to ensure identical models
    torch.cuda.manual_seed(42)
    
    actor = GPT(vocab_size).to(device)
    actor_old = GPT(vocab_size).to(device)
    
    # IMPORTANT: Initialize old_actor with actor's weights
    actor_old.load_state_dict(actor.state_dict())
    actor_old.eval()
    
    # Synchronize after model creation
    if world_size > 1 and dist.is_initialized():
        dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
    
    # Wrap actor in DDP for multi-GPU training
    if world_size > 1:
        actor = setup_ddp_with_gradient_bucketing(actor, device, BUCKET_SIZE_MB)
        # Note: actor_old is NOT wrapped in DDP - we'll sync it manually
    
    # Create optimizer (only for actor)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    
    # Learning rate scheduler
    scheduler = None
    if USE_LR_DECAY:
        if LR_DECAY_TYPE == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=TOTAL_ITERS - WARMUP_ITERS, eta_min=MIN_LR
            )
        elif LR_DECAY_TYPE == "linear":
            def linear_schedule(step):
                if step < WARMUP_ITERS:
                    return step / WARMUP_ITERS
                else:
                    progress = (step - WARMUP_ITERS) / (TOTAL_ITERS - WARMUP_ITERS)
                    return (1 - progress) * (1 - MIN_LR/LR) + MIN_LR/LR
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_schedule)
        elif LR_DECAY_TYPE == "exponential":
            decay_rate = (MIN_LR / LR) ** (1 / (TOTAL_ITERS - WARMUP_ITERS))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    
    # Create data loader
    loader = DistributedContextualTextLoader(
        text=text,
        encode_fn=encode,
        batch_size=MICRO_BATCH,
        horizon=HORIZON,
        context_len=CONTEXT_LEN,
        rank=rank,
        world_size=world_size,
        device=device
    )
    
    # Initialize controllers
    kl_controller = AdaptiveKLController(target_kl=0.02)
    confidence_monitor = ConfidenceMonitor() if USE_CONFIDENCE_SCALING else None
    curriculum_manager = CurriculumManager()
    
    # Training state
    chars_seen = 0
    current_kl_coef = BETA_KL
    
    # Initialize wandb (only on rank 0)
    if rank == 0:
        import wandb
        wandb.init(
            project="grpo-multi-gpu",
            name=f"grpo-{N_GPUS}gpu-{N_LAYER}L-{N_EMB}D",
            config={
                "model": {
                    "vocab_size": vocab_size,
                    "n_layer": N_LAYER,
                    "n_head": N_HEAD,
                    "n_emb": N_EMB,
                    "context_len": CONTEXT_LEN,
                    "param_count": sum(p.numel() for p in actor.parameters()),
                },
                "training": {
                    "batch_size": BATCH,
                    "micro_batch_size": MICRO_BATCH,
                    "grad_accum_steps": GRAD_ACCUM,
                    "learning_rate": LR,
                    "epochs": EPOCHS,
                    "total_iters": TOTAL_ITERS,
                    "world_size": world_size,
                },
                "grpo": {
                    "use_exhaustive": USE_EXHAUSTIVE,
                    "k_samples": K_SAMPLES,
                    "clip_ratio": CLIP_RATIO,
                    "entropy_coef": ENTROPY_COEF,
                    "beta_kl": BETA_KL,
                    "confidence_scaling": USE_CONFIDENCE_SCALING,
                    "confidence_weight": CONFIDENCE_WEIGHT,
                },
                "distributed": {
                    "n_gpus": N_GPUS,
                    "gpu_type": GPU_TYPE,
                    "bucket_size_mb": BUCKET_SIZE_MB,
                    "gradient_compression": "fp16",
                },
            }
        )
    
    print_rank0(f"Starting training for {TOTAL_ITERS} iterations...", rank)
    
    # Full GRPO training loop
    from collections import defaultdict
    from tqdm import tqdm
    
    # AMP scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    # Metrics tracking
    all_metrics = defaultdict(list)
    
    for it in tqdm(range(1, TOTAL_ITERS + 1), desc="Training", disable=(rank != 0)):
        actor.train()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0.0
        iter_metrics = defaultdict(list)
        
        # Gradient accumulation loop
        for micro_step in range(GRAD_ACCUM):
            # Get batch
            ctx, ref_tok = loader.next()
            
            # Generate from old policy (matching train_batching_optimized.py exactly)
            old_probs = None
            if USE_CONFIDENCE_SCALING:
                with torch.no_grad():
                    old_logits = actor_old(ctx[:, -actor_old.module.context_len if hasattr(actor_old, 'module') else -actor_old.context_len:])[:, -1, :]
                    old_probs = F.softmax(old_logits, dim=-1)
            
            if USE_EXHAUSTIVE and HORIZON == 1:
                # Despite the name, train_batching_optimized.py uses K-sample generation here
                with torch.no_grad():
                    # Generate K samples using temperature sampling (NOT exhaustive)
                    G = generate_with_temperature(actor_old, ctx, HORIZON, K_SAMPLES, vocab_size, temperature=TEMPERATURE)
                
                # Track model accuracy
                with torch.no_grad():
                    model_logits = actor_old(ctx[:, -actor_old.module.context_len if hasattr(actor_old, 'module') else -actor_old.context_len:])[:, -1, :]
                    model_pred = model_logits.argmax(dim=-1)
                    model_accuracy = (model_pred == ref_tok[:, 0]).float().mean().item()
                    iter_metrics["accuracy"].append(model_accuracy)
                    
                    # N-gram accuracy tracking
                    ref_full = torch.cat([ctx, ref_tok], dim=1)
                    _, components = ref_model(ref_full[:, -HORIZON-1:], return_components=True)
                    
                    bigram_logits = components['bigram'][:, -1, :]
                    bigram_pred = bigram_logits.argmax(dim=-1)
                    bigram_correct = (bigram_pred == ref_tok[:, 0]).float().mean().item()
                    iter_metrics["bigram_accuracy"].append(bigram_correct)
                    
                    if CONTEXT_LEN >= 2:
                        trigram_logits = components['trigram'][:, -1, :]
                        trigram_pred = trigram_logits.argmax(dim=-1)
                        trigram_correct = (trigram_pred == ref_tok[:, 0]).float().mean().item()
                        iter_metrics["trigram_accuracy"].append(trigram_correct)
                    
                    if CONTEXT_LEN >= 3:
                        fourgram_logits = components['fourgram'][:, -1, :]
                        fourgram_pred = fourgram_logits.argmax(dim=-1)
                        fourgram_correct = (fourgram_pred == ref_tok[:, 0]).float().mean().item()
                        iter_metrics["fourgram_accuracy"].append(fourgram_correct)
                
                # Compute rewards for each sample
                R = torch.zeros_like(G, dtype=torch.float32)
                
                for k in range(K_SAMPLES):
                    gen = G[:, k, :]
                    
                    with torch.no_grad():
                        full_seq = torch.cat([ctx, gen], dim=1)
                        ref_logits = ref_model(full_seq[:, -HORIZON-1:])[:, -HORIZON:]
                    
                    reward, metrics = compute_rewards_multi_char(gen, ref_tok, ref_logits)
                    R[:, k, :] = reward
                    
                    # Store metrics from reward computation
                    for key, val in metrics.items():
                        iter_metrics[key].append(val)
                
                # Compute advantages
                base = R.mean(dim=1, keepdim=True)
                adv = R - base
                adv_std = torch.maximum(adv.std(dim=1, keepdim=True), torch.tensor(MIN_VARIANCE, device=device))
                adv = adv / adv_std
                
                # Store metrics
                iter_metrics["reward_mean"].append(R.mean().item())
                iter_metrics["reward_std"].append(R.std().item())
                iter_metrics["advantage_std"].append(adv.std().item())
                
                # Flatten for policy update
                flat_G = G.reshape(-1, HORIZON)
                flat_adv = adv.reshape(-1, HORIZON)
            
            # Expand contexts for K samples
            ctx_for_loss = ctx.repeat_interleave(K_SAMPLES, dim=0)
            full_seq = torch.cat([ctx_for_loss, flat_G], dim=1)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Get new policy logits
                context_len = actor.module.context_len if hasattr(actor, 'module') else actor.context_len
                input_seq = full_seq[:, -context_len:]
                new_logits = actor(input_seq)[:, -HORIZON:]
                
                # Policy loss (PPO-style)
                new_dist = torch.distributions.Categorical(logits=new_logits)
                logp_new = new_dist.log_prob(flat_G.squeeze(-1))
                
                # Get old policy log probs
                with torch.no_grad():
                    old_logits = actor_old(input_seq)[:, -HORIZON:]
                    old_dist = torch.distributions.Categorical(logits=old_logits)
                    logp_old = old_dist.log_prob(flat_G.squeeze(-1))
                
                # PPO loss (matching train_batching_optimized.py)
                ratio = torch.exp(logp_new - logp_old)
                clipped_ratio = torch.clamp(ratio, 1-CLIP_RATIO, 1+CLIP_RATIO)
                
                pol_loss1 = -flat_adv.squeeze(-1) * ratio
                pol_loss2 = -flat_adv.squeeze(-1) * clipped_ratio
                pol_loss = torch.max(pol_loss1, pol_loss2).mean()
                
                # KL penalty (matching train_batching_optimized.py approach)
                ref_logits = ref_model(full_seq[:, -HORIZON-1:])[:, -HORIZON:]
                
                # Compute KL using the simple approach from optimized version
                true_probs = torch.zeros_like(new_logits)
                true_probs.scatter_(2, flat_G.unsqueeze(-1), 1.0)
                
                new_probs = F.softmax(new_logits, dim=-1)
                true_char_probs = new_probs.gather(2, flat_G.unsqueeze(-1)).squeeze(-1)
                kl_divergence = -torch.log(true_char_probs + 1e-8).mean()
                
                kl = torch.clamp(kl_divergence, max=5.0)
                
                # Adaptive KL coefficient
                if kl_controller and micro_step == 0:
                    current_kl_coef = kl_controller.update(kl.item(), current_kl_coef)
                
                # Entropy bonus (matching train_batching_optimized.py) - removed since not actually used
                # Note: train_batching_optimized.py doesn't include entropy in the loss despite having ENTROPY_COEF
                
                # Total loss (matching train_batching_optimized.py)
                loss = (pol_loss + current_kl_coef * kl) / GRAD_ACCUM
            
            # Backward pass
            scaler.scale(loss).backward()
            total_loss += loss.item()
            
            # Store metrics
            iter_metrics["pol_loss"].append(pol_loss.item())
            iter_metrics["kl"].append(kl.item())
            iter_metrics["ratio"].append(ratio.mean().item())
            iter_metrics["ratio_max"].append(ratio.max().item())
            
            # Debug print loss values
            if it == 1 and micro_step == 0 and rank == 0:
                print(f"Debug - Policy loss: {pol_loss.item():.4f}, KL: {kl.item():.4f}")
        
        # Gradient clipping and optimization step
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Update character count
        chars_seen += BATCH * HORIZON
        
        # Aggregate metrics across GPUs
        if world_size > 1:
            # Gather metrics from all ranks
            metric_tensors = {}
            for key, values in iter_metrics.items():
                if values:  # Only process if we have values
                    metric_tensor = torch.tensor(np.mean(values), device=device, dtype=torch.float32)
                    all_reduce_mean(metric_tensor)
                    metric_tensors[key] = metric_tensor.item()
                else:
                    metric_tensors[key] = 0.0
        else:
            metric_tensors = {k: np.mean(v) if v else 0.0 for k, v in iter_metrics.items()}
        
        # Update curriculum manager with metrics
        curriculum_metrics = {
            'accuracy': metric_tensors.get('accuracy', 0.0),
            'reward_mean': metric_tensors.get('reward_mean', 0.0),
            'kl_divergence': metric_tensors.get('kl', 0.0),
            'loss': total_loss * GRAD_ACCUM
        }
        curriculum_manager.update_metrics(curriculum_metrics)
        
        # Check for curriculum promotion
        should_promote, promotion_reason = curriculum_manager.check_promotion()
        if should_promote:
            curriculum_manager.promote()
            stage_config = curriculum_manager.get_stage_config()
            if rank == 0:
                print(f"\n[Curriculum] Promoted to Stage {curriculum_manager.current_stage}: {stage_config['name']}")
                print(f"[Curriculum] Reason: {promotion_reason}")
                print(f"[Curriculum] Focus: {stage_config['focus']}")
        
        # Log to wandb (only rank 0)
        if rank == 0 and it % 2 == 0:
            wandb.log({
                "loss/total": total_loss * GRAD_ACCUM,
                "loss/policy": metric_tensors.get("pol_loss", 0),
                "loss/kl": metric_tensors.get("kl", 0),
                "loss/kl_coef": current_kl_coef,
                "metrics/accuracy": metric_tensors.get("accuracy", 0),
                "metrics/reward_mean": metric_tensors.get("reward_mean", 0),
                "metrics/reward_std": metric_tensors.get("reward_std", 0),
                "metrics/advantage_std": metric_tensors.get("advantage_std", 0),
                "metrics/ratio_mean": metric_tensors.get("ratio", 1.0),
                "metrics/ratio_max": metric_tensors.get("ratio_max", 1.0),
                "training/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "training/chars_seen": chars_seen,
                "training/iteration": it,
                "training/epoch": it / ITERS_PER_EPOCH,
                "curriculum/stage": curriculum_manager.current_stage,
                "curriculum/iterations_in_stage": curriculum_manager.iterations_in_stage,
                "curriculum/ema_accuracy": curriculum_manager.ema_metrics['accuracy'],
                "curriculum/ema_reward": curriculum_manager.ema_metrics['reward_mean'],
                "curriculum/plateau_counter": curriculum_manager.plateau_counter,
            })
        
        # Update old_actor periodically
        if it % 5 == 0:
            if rank == 0:
                print(f"Iter {it}: loss={total_loss*GRAD_ACCUM:.4f}, acc={metric_tensors.get('accuracy', 0):.3f}, kl={metric_tensors.get('kl', 0):.4f}")
            
            # Update old actor
            actor_state = actor.module.state_dict() if hasattr(actor, 'module') else actor.state_dict()
            actor_old.load_state_dict(actor_state)
            
            # Synchronize old_actor across all ranks
            if world_size > 1:
                sync_model_across_ranks(actor_old, src_rank=0)
            
            actor_old.eval()
            
        
        # Save checkpoint periodically
        if it % 100 == 0:
            save_checkpoint_rank0(
                model=actor,
                optimizer=optimizer,
                iteration=it,
                checkpoint_path=f"/data/checkpoint_iter{it}.pt",
                rank=rank,
                additional_state={
                    "chars_seen": chars_seen,
                    "kl_coef": current_kl_coef,
                    "scaler_state": scaler.state_dict(),
                    "curriculum_state": curriculum_manager.get_state_dict(),
                }
            )
        
        # Confidence monitoring
        if USE_CONFIDENCE_SCALING and confidence_monitor and rank == 0:
            confidence_monitor.update(
                confidence=metric_tensors.get("accuracy", 0),
                accuracy=metric_tensors.get("accuracy", 0),
                gradient_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            
            if it % 20 == 0:
                is_calibrated, calib_error = confidence_monitor.check_calibration()
                has_collapse, collapse_ratio = confidence_monitor.check_confidence_collapse()
                
                wandb.log({
                    "confidence/calibration_error": calib_error,
                    "confidence/collapse_ratio": collapse_ratio,
                    "confidence/is_calibrated": float(is_calibrated),
                })
                
                if not is_calibrated and rank == 0:
                    print(f"Warning: Model miscalibrated, error={calib_error:.3f}")
                if has_collapse and rank == 0:
                    print(f"Warning: Confidence collapse detected, ratio={collapse_ratio:.3f}")
    
    print_rank0("\nTraining completed!", rank)
    
    # Final checkpoint
    save_checkpoint_rank0(
        model=actor,
        optimizer=optimizer,
        iteration=TOTAL_ITERS,
        checkpoint_path=f"/data/checkpoint_final.pt",
        rank=rank,
        additional_state={
            "chars_seen": chars_seen,
            "kl_coef": current_kl_coef,
            "scaler_state": scaler.state_dict(),
        }
    )
    
    # Finish wandb run
    if rank == 0:
        wandb.finish()
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    # This section runs when launched by torchrun
    train_grpo()