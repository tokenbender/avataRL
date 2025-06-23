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
N_GPUS = 8
GPU_TYPE = "H100"
app = modal.App("perplexity_fix_2")
CONTEXT_LEN = 32
HORIZON = 8
BATCH = 16384
MICRO_BATCH = 256
GRAD_ACCUM = BATCH // (MICRO_BATCH * N_GPUS)
EPOCHS = 3.0
DATASET_SIZE = 1_115_394
ITERS_PER_EPOCH = DATASET_SIZE // BATCH
TOTAL_ITERS = int(ITERS_PER_EPOCH * EPOCHS)
LR = 3e-3
BETA_KL = 0.1
KL_WARM = int(DATASET_SIZE * 0.8)
KL_FREE_FRACTION = 0.1
N_LAYER = 6
N_HEAD = 6
N_EMB = 384
SAVE_INTERMEDIATE_CHECKPOINTS = False
SAVE_FINAL_CHECKPOINT = True
CHECKPOINT_INTERVAL = 1000
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
USE_LR_DECAY = True
LR_DECAY_TYPE = "cosine"
MIN_LR = 1e-5
WARMUP_ITERS = max(1, int(TOTAL_ITERS * 0.02))
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
        device: torch.device = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype, device=device), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype, device=device), persistent=False
        )
        self.register_buffer(
            "cache_pos", torch.tensor(0, device=device), persistent=False
        )
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
    def reset(self) -> None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos.zero_()
    @property
    def size(self) -> int:
        return self.cache_pos.item()
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
        assert (self.cache_pos + seq_len) <= self.k_cache.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        cache_start = self.cache_pos.item()
        cache_end = cache_start + seq_len
        k_out[:, :, cache_start:cache_end] = k_val
        v_out[:, :, cache_start:cache_end] = v_val
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
        self.qkv = nn.Linear(n_emb, 3 * n_emb, bias=False)
        self.o_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Use a larger max_seq to support generation
        max_seq = max(CONTEXT_LEN + HORIZON, 512)  # Support up to 512 tokens for generation
        self.rope = RotaryCache(self.head_dim, max_seq)
        self.use_flash_attn = False
        if USE_FLASH_ATTN:
            try:
                from flash_attn import flash_attn_qkvpacked_func
                self.flash_attn_func = flash_attn_qkvpacked_func
                self.use_flash_attn = True
            except ImportError:
                print("Flash Attention not available, using standard attention")
        self.use_paged_attention = False
        self.kv_cache = None
        self.cache_enabled = False
    def init_kv_cache(self, batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16, device: torch.device = None):
        """Initialize KV cache for inference"""
        self.kv_cache = KVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.n_head,
            head_dim=self.head_dim,
            dtype=dtype,
            device=device
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
        if use_cache and self.cache_enabled and self.kv_cache is not None:
            return self._forward_with_cache(x, mask)
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        if self.use_flash_attn:
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            sin, cos = self.rope(T)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
            q, k = norm(q), norm(k)
            packed = torch.stack((q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)), dim=2)
            dtype0 = packed.dtype
            if dtype0 not in (torch.float16, torch.bfloat16):
                packed = packed.to(torch.bfloat16)
            out = self.flash_attn_func(packed, causal=True, dropout_p=0.1 if self.training else 0.0)
            out = out.to(dtype0).reshape(B, T, C)
        else:
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            sin, cos = self.rope(T)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
            q, k = norm(q), norm(k)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)
    def _forward_with_cache(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass using KV cache for efficient inference"""
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        cache_size = self.kv_cache.size
        sin, cos = self.rope(cache_size + T)
        sin_new = sin[:, :, cache_size:cache_size+T, :]
        cos_new = cos[:, :, cache_size:cache_size+T, :]
        q = (q * cos_new) + (_rotate_half(q) * sin_new)
        k = (k * cos_new) + (_rotate_half(k) * sin_new)
        q, k = norm(q), norm(k)
        k_cache, v_cache = self.kv_cache.update(k, v)
        valid_cache_size = self.kv_cache.size
        
        # For the first token(s), we need the updated cache that includes current k,v
        # The cache was just updated, so we use the full valid portion
        k_valid = k_cache[:, :, :valid_cache_size, :]
        v_valid = v_cache[:, :, :valid_cache_size, :]
        
        # Compute attention - no causal mask needed as we're attending to all cached tokens
        # Check if shapes are compatible
        if q.shape[2] == 0 or k_valid.shape[2] == 0:
            print(f"WARNING: Empty sequence in attention!")
            print(f"  q shape: {q.shape}")
            print(f"  k_valid shape: {k_valid.shape}")
            print(f"  v_valid shape: {v_valid.shape}")
            print(f"  valid_cache_size: {valid_cache_size}")
            print(f"  cache_size before update: {cache_size}")
            # Return zeros if we have empty sequences
            return self.o_proj(torch.zeros(B, T, C, device=x.device, dtype=x.dtype))
        
        out = F.scaled_dot_product_attention(q, k_valid, v_valid, is_causal=False)
        
        # Debug check
        if out.numel() == 0:
            print(f"WARNING: Empty attention output!")
            print(f"  q shape: {q.shape}")
            print(f"  k_valid shape: {k_valid.shape}")
            print(f"  v_valid shape: {v_valid.shape}")
            print(f"  valid_cache_size: {valid_cache_size}")
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)
class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture"""
    def __init__(self, n_emb: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attn = OptimizedAttention(n_emb, n_head, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb, bias=False),
            ReLUSquared(),
            nn.Linear(4 * n_emb, n_emb, bias=False),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache: bool = False) -> torch.Tensor:
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
        self.wte = nn.Embedding(vocab_size, n_emb)
        self.drop = nn.Dropout(0.1)
        self.layers = nn.ModuleList([
            TransformerBlock(n_emb, n_head, dropout=0.1)
            for _ in range(n_layer)
        ])
        self.head = nn.Linear(n_emb, vocab_size, bias=False)
        del self.head.weight
        self.head.weight = self.wte.weight  
        self.apply(self._init_weights)
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
        tok_emb = self.wte(idx)
        x = self.drop(tok_emb)
        mask = self.causal_mask[:T, :T] if T <= self.context_len + HORIZON else None
        for layer in self.layers:
            x = layer(x, mask, use_cache=use_cache)
        x = norm(x)
        logits = self.head(x)
        return logits
    def init_kv_caches(self, batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16, device: torch.device = None):
        """Initialize KV caches for all attention layers"""
        for layer in self.layers:
            layer.attn.init_kv_cache(batch_size, max_seq_len, dtype, device)
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
                 top_k: Optional[int] = None, use_cache: bool = True, 
                 repetition_penalty: float = 2.0, repetition_window: int = 128) -> torch.Tensor:
        """
        Generate tokens using the model with optional KV caching and repetition penalty.
        Args:
            idx: Starting context tokens [B, T]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If specified, only sample from top k tokens
            use_cache: Whether to use KV cache for efficient generation
            repetition_penalty: Penalty factor for repeated tokens (>1.0 reduces repetition)
            repetition_window: Window size to check for repetitions
        Returns:
            Generated tokens including the initial context
        """
        device = idx.device
        B, T = idx.shape
        if use_cache:
            # Use the model's dtype (from embeddings) not the input indices dtype
            model_dtype = self.wte.weight.dtype
            self.init_kv_caches(B, T + max_new_tokens, dtype=model_dtype, device=device)
        generated = idx
        for _ in range(max_new_tokens):
            if use_cache and generated.shape[1] > T:
                logits = self(generated[:, -1:], use_cache=True)
            else:
                idx_cond = generated if generated.shape[1] <= self.context_len else generated[:, -self.context_len:]
                logits = self(idx_cond, use_cache=use_cache)
            logits = logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                window_start = max(0, generated.shape[1] - repetition_window)
                recent_tokens = generated[:, window_start:]
                for b in range(B):
                    token_counts = {}
                    for token in recent_tokens[b].tolist():
                        token_counts[token] = token_counts.get(token, 0) + 1
                    for token, count in token_counts.items():
                        if count == 1:
                            logits[b, token] /= repetition_penalty
                        elif count == 2:
                            logits[b, token] /= (repetition_penalty * 2.0)
                        else:
                            logits[b, token] /= (repetition_penalty * count * 2.0)
                    if generated.shape[1] >= 3:
                        last_tokens = generated[b, -3:].tolist()
                        if len(set(last_tokens)) == 1:  
                            logits[b, last_tokens[0]] = -float('inf')  
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, idx_next), dim=1)
        if use_cache:
            self.disable_kv_caches()
        return generated
class OnTheFlyNGramRef(nn.Module):
    """Bigram reference model for character-level language modeling"""
    def __init__(self, text: str, stoi: Dict[str, int], vocab_size: int, smoothing: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.bigram_counts = torch.ones(vocab_size, vocab_size) * smoothing
        text_indices = [stoi[c] for c in text]
        for i in range(len(text_indices) - 1):
            self.bigram_counts[text_indices[i], text_indices[i+1]] += 1
        bigram_probs = self.bigram_counts / self.bigram_counts.sum(dim=-1, keepdim=True)
        self.register_buffer("bigram_log_probs", torch.log(bigram_probs + 1e-8).clamp(min=-20.0))
    def forward(self, idx: torch.Tensor, return_components: bool = False) -> torch.Tensor:
        B, T = idx.shape
        device = idx.device
        bigram_logits = torch.zeros(B, T, self.vocab_size, device=device)
        uniform_logits = torch.zeros(self.vocab_size, device=device)
        for t in range(T):
            if t == 0:
                bigram_logits[:, t, :] = uniform_logits
            else:
                prev_char = idx[:, t-1]
                bigram_logits[:, t, :] = self.bigram_log_probs[prev_char]
        if return_components:
            return bigram_logits, {
                'bigram': bigram_logits
            }
        return bigram_logits
def load_shakespeare_reference_model(checkpoint_path: str, vocab_size: int, device: torch.device, rank: int) -> Optional[GPT]:
    """Load pretrained Shakespeare model for KL divergence reference"""
    try:
        if rank == 0:
            print(f"[Reference Model] Loading pretrained Shakespeare model from {checkpoint_path}...")
        if checkpoint_path.startswith('http'):
            import urllib.request
            local_path = "/tmp/shakespeare_ckpt.pt"
            if rank == 0:
                print(f"[Reference Model] Downloading from {checkpoint_path}...")
                urllib.request.urlretrieve(checkpoint_path, local_path)
            if dist.is_initialized():
                dist.barrier()
            checkpoint_path = local_path
        else:
            if not os.path.exists(checkpoint_path):
                alternative_paths = [
                    "/data/checkpoints/shakespeare/ckpt.pt",
                    "/data/checkpoints/shakespeare/checkpoint.pt",
                    "/data/checkpoints/shakespeare/model.pt",
                    "/data/checkpoints/shakespeare/shakespeare_char_model.pt"
                ]
                found = False
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        checkpoint_path = alt_path
                        found = True
                        if rank == 0:
                            print(f"[Reference Model] Found checkpoint at {alt_path}")
                        break
                if not found:
                    raise FileNotFoundError(f"Shakespeare checkpoint not found at {checkpoint_path} or alternative paths")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if 'model_args' in checkpoint:
            model_args = checkpoint['model_args']
            n_layer = model_args.get('n_layer', 6)
            n_head = model_args.get('n_head', 6)
            n_embd = model_args.get('n_embd', 384)
            block_size = model_args.get('block_size', 256)
        else:
            n_layer = 6
            n_head = 6
            n_embd = 384
            block_size = 256
        ref_model = GPT(
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_embd,
            context_len=block_size
        ).to(device)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k == 'transformer.wte.weight':
                new_state_dict['wte.weight'] = v
                new_state_dict['head.weight'] = v  
            elif 'transformer.h.' in k and '.attn.c_attn.weight' in k:
                layer_idx = k.split('.')[2]
                new_state_dict[f'blocks.{layer_idx}.attn.qkv_proj.weight'] = v
            elif 'transformer.h.' in k and '.attn.c_proj.weight' in k:
                layer_idx = k.split('.')[2]
                new_state_dict[f'blocks.{layer_idx}.attn.out_proj.weight'] = v
            elif 'transformer.h.' in k and '.mlp.c_fc.weight' in k:
                layer_idx = k.split('.')[2]
                new_state_dict[f'blocks.{layer_idx}.mlp.fc1.weight'] = v
            elif 'transformer.h.' in k and '.mlp.c_proj.weight' in k:
                layer_idx = k.split('.')[2]
                new_state_dict[f'blocks.{layer_idx}.mlp.fc2.weight'] = v
            elif k == 'transformer.lm_head.weight' or k == 'lm_head.weight':
                new_state_dict['head.weight'] = v
                new_state_dict['wte.weight'] = v  
        ref_model.load_state_dict(new_state_dict, strict=False)
        ref_model.eval()
        if rank == 0:
            print(f"[Reference Model] Successfully loaded Shakespeare reference model")
            print(f"[Reference Model] Model config: layers={n_layer}, heads={n_head}, embed={n_embd}, context={block_size}")
        return ref_model
    except Exception as e:
        if rank == 0:
            print(f"[Reference Model] Failed to load pretrained model: {e}")
            print("[Reference Model] Continuing without reference model KL penalty")
        return None
def compute_shakespeare_kl_penalty(
    generated_tokens: torch.Tensor,    
    new_logits: torch.Tensor,          
    shakespeare_model: GPT,            
    stage: int,                        
    device: torch.device
) -> torch.Tensor:
    """
    Compute KL divergence between current model and pretrained Shakespeare model
    with curriculum-based top-k filtering that gradually tightens
    Returns: [B] KL penalty per sequence
    """
    B, T, V = new_logits.shape
    stage_topk_config = {
        0: V,      
        1: V,      
        2: 50,     
        3: 20,     
        4: 10,     
    }
    top_k = stage_topk_config.get(stage, V)
    with torch.no_grad():
        shakespeare_logits = shakespeare_model(generated_tokens)
        shakespeare_log_probs = F.log_softmax(shakespeare_logits, dim=-1)
        new_log_probs = F.log_softmax(new_logits, dim=-1)
        if top_k < V:
            topk_values, topk_indices = torch.topk(shakespeare_log_probs, k=top_k, dim=-1)
            mask = torch.zeros_like(shakespeare_log_probs, dtype=torch.bool)
            mask.scatter_(2, topk_indices, True)
            shakespeare_log_probs_masked = shakespeare_log_probs.clone()
            shakespeare_log_probs_masked[~mask] = float('-inf')
            new_log_probs_masked = new_log_probs.clone()
            new_log_probs_masked[~mask] = float('-inf')
            shakespeare_probs = F.softmax(shakespeare_log_probs_masked, dim=-1)
            new_probs = F.softmax(new_log_probs_masked, dim=-1)
        else:
            shakespeare_probs = torch.exp(shakespeare_log_probs)
            new_probs = torch.exp(new_log_probs)
        kl_div = F.kl_div(
            torch.log(new_probs + 1e-8),
            shakespeare_probs,
            reduction='none'
        ).sum(dim=-1)  
        kl_penalty = kl_div.mean(dim=1)  
        stage_scale = {
            0: 0.0,
            1: 0.0,
            2: 0.5,   
            3: 0.8,   
            4: 1.0,   
        }
        scale = stage_scale.get(stage, 1.0)
        kl_penalty = kl_penalty * scale
    return kl_penalty
def setup_distributed():
    """Initialize distributed training environment"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
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
        param_size = param.numel() * param.element_size() / 1024 / 1024  
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
    params = list(model.parameters())
    buckets = create_buckets(params, bucket_size_mb)
    ddp_model = DDP(
        model,
        device_ids=[device.index],
        bucket_cap_mb=bucket_size_mb,
        gradient_as_bucket_view=True,  
        static_graph=True,  
    )
    if dist.is_initialized() and dist.get_world_size() > 1:
        ddp_model.register_comm_hook(
            state=None,
            hook=comm_hooks.fp16_compress_hook  
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
        self.data = encode_fn(text).to(device)  
        self.batch_size = batch_size
        self.horizon = horizon
        self.context_len = context_len
        self.rank = rank
        self.world_size = world_size
        self.device = device
        total_len = len(self.data)
        offset_per_rank = (total_len - context_len - batch_size * horizon) // max(1, world_size)
        self.pos = context_len + (rank * offset_per_rank)
        max_start = total_len - batch_size * horizon - 1
        self.pos = min(self.pos, max_start)
    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch - exactly like train.py"""
        if self.pos + self.batch_size * self.horizon + 1 > len(self.data):
            self.pos = self.context_len
        contexts = []
        targets = []
        for b in range(self.batch_size):
            ctx_start = self.pos - self.context_len + b * self.horizon
            ctx_end = self.pos + b * self.horizon
            context = self.data[ctx_start:ctx_end]
            target = self.data[ctx_end:ctx_end + self.horizon]
            contexts.append(context)
            targets.append(target)
        self.pos += self.batch_size * self.horizon
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
@torch.no_grad()
def generate_exhaustive_single_char(model: nn.Module, contexts: torch.Tensor, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate ALL possible next characters (exhaustive exploration)
    Returns both the characters and their log probabilities
    """
    B = contexts.shape[0]
    device = contexts.device
    ctx_window = contexts[:, -model.context_len:] if contexts.shape[1] > model.context_len else contexts
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(ctx_window)[:, -1, :]  
    log_probs = F.log_softmax(logits, dim=-1)  
    all_chars = torch.arange(vocab_size, device=device).unsqueeze(0).expand(B, -1)  
    return all_chars, log_probs
@torch.no_grad()
def generate_with_temperature(model: nn.Module, contexts: torch.Tensor, horizon: int, k_samples: int, 
                            vocab_size: int, temperature: float = 1.0) -> torch.Tensor:
    """
    Generate K samples with temperature-based sampling
    For multi-character generation
    """
    B = contexts.shape[0]
    ctx = contexts.repeat_interleave(k_samples, dim=0)  
    for _ in range(horizon):
        ctx_window = ctx[:, -model.context_len:] if ctx.shape[1] > model.context_len else ctx
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(ctx_window)[:, -1, :] / temperature
        logits[:, vocab_size:] = -1e9
        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, 1)
        ctx = torch.cat([ctx, next_char], dim=1)
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
    base_rewards = torch.zeros(B, vocab_size, device=device)
    for b in range(B):
        base_rewards[b, ref_char[b]] = 1.0
    with torch.no_grad():
        ref_logits = ref_model(ctx)[:, -1, :]
        ref_probs = F.softmax(ref_logits, dim=-1)
        partial_rewards = ref_probs * 0.1  
        base_rewards = base_rewards + partial_rewards
    if model_log_probs is not None:
        model_probs = torch.exp(model_log_probs)
        correct_char_bonus = torch.zeros_like(base_rewards)
        for b in range(B):
            correct_char_bonus[b, ref_char[b]] = model_probs[b, ref_char[b]] * 2.0
        base_rewards = base_rewards + correct_char_bonus
    rewards = base_rewards
    if use_confidence_scaling and old_probs is not None:
        confidence = old_probs[torch.arange(B), ref_char]
        confidence_scale = 1.0 + confidence_weight * confidence.clamp(0, 1)
        confidence_scale = confidence_scale.clamp(1.0, confidence_clip)
        rewards[torch.arange(B), ref_char] *= confidence_scale
        if ENABLE_CONFIDENCE_PENALTY:
            top_k = min(10, vocab_size)
            top_probs, top_indices = old_probs.topk(top_k, dim=-1)
            wrong_mask = top_indices != ref_char.unsqueeze(1)
            confident_mask = top_probs > 0.5
            penalty_mask = wrong_mask & confident_mask
            penalties = -0.1 * top_probs * penalty_mask.float()
            rewards.scatter_add_(1, top_indices, penalties)
    rewards = rewards - rewards.mean(dim=1, keepdim=True)
    return rewards
def calculate_repetition_penalty(tokens: torch.Tensor, window_size: int = 16, 
                               ngram_weights: Dict[int, float] = None,
                               device: str = 'cuda') -> torch.Tensor:
    """
    Calculate repetition penalty for generated sequences using n-gram detection
    Args:
        tokens: [B, T] generated token sequences
        window_size: size of sliding window for repetition detection
        ngram_weights: weights for different n-gram levels (default: {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1})
        device: device for computation
    Returns:
        penalty: [B] repetition penalty per sequence (0-1, higher = more repetition)
    """
    if ngram_weights is None:
        ngram_weights = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
    B, T = tokens.shape
    penalties = torch.zeros(B, device=device)
    for n in range(1, min(5, T + 1)):  
        if n not in ngram_weights or ngram_weights[n] == 0:
            continue
        ngram_penalties = torch.zeros(B, device=device)
        for b in range(B):
            seq = tokens[b]
            for start in range(max(0, T - window_size)):
                end = min(T, start + window_size)
                window = seq[start:end]
                if len(window) < n:
                    continue
                ngram_counts = {}
                for i in range(len(window) - n + 1):
                    ngram = tuple(window[i:i+n].tolist())
                    ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
                window_penalty = 0.0
                for ngram, count in ngram_counts.items():
                    if count > 1:
                        if count == 2:
                            penalty = 0.1
                        else:
                            penalty = 1.0 - math.exp(-0.5 * (count - 1))
                        window_penalty += penalty
                num_ngrams = len(window) - n + 1
                if num_ngrams > 0:
                    window_penalty /= num_ngrams
                ngram_penalties[b] = max(ngram_penalties[b], window_penalty)
        penalties += ngram_weights[n] * ngram_penalties
    return penalties.clamp(0.0, 1.0)
def compute_rewards_multi_char(gen: torch.Tensor, ref: torch.Tensor, ref_logits: torch.Tensor, 
                             curriculum_manager: Optional['CurriculumManager'] = None,
                             sequences: Optional[torch.Tensor] = None,
                             ngram_scores: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
    """
    Compute rewards for multi-character generation with detailed metrics
    Includes advanced repetition penalty for stage 2+
    """
    exact_match = (gen == ref).float()
    with torch.no_grad():
        B, T = gen.shape
        if ref_logits is not None:
            ref_probs = F.softmax(ref_logits, dim=-1)
            gen_probs = ref_probs.gather(2, gen.unsqueeze(-1)).squeeze(-1)
            partial_reward = gen_probs
        else:
            # No partial rewards without reference model
            partial_reward = torch.zeros_like(exact_match)
            gen_probs = torch.zeros_like(exact_match)  # Placeholder for metrics
    base_reward = exact_match + partial_reward
    repetition_penalties = torch.zeros(B, device=gen.device)
    if curriculum_manager is not None and curriculum_manager.current_stage >= 2:
        repetition_penalties = calculate_repetition_penalty(
            gen, 
            window_size=16,
            device=gen.device
        )
        penalty_weight = curriculum_manager.get_repetition_penalty_weight()
        rep_penalty_expanded = repetition_penalties.unsqueeze(1).expand_as(base_reward)
        zero_reward_mask = rep_penalty_expanded > 0.1
        base_reward = base_reward * (~zero_reward_mask).float()
        minor_rep_mask = (rep_penalty_expanded > 0) & (rep_penalty_expanded <= 0.1)
        base_reward = base_reward * torch.where(minor_rep_mask, 
                                                 0.2,  
                                                 1.0)
    if curriculum_manager is not None and sequences is not None and ngram_scores is not None:
        reward = curriculum_manager.compute_curriculum_rewards(base_reward.mean(dim=1), ngram_scores, sequences)
        reward = reward.unsqueeze(-1).expand_as(base_reward)
    else:
        reward = base_reward
    repetition_stats = {
        "repetition_penalty_mean": repetition_penalties.mean().item(),
        "repetition_penalty_max": repetition_penalties.max().item(),
        "repetition_penalty_std": repetition_penalties.std().item() if B > 1 else 0.0,
    }
    repetition_rate = 0.0
    if T > 1:
        immediate_reps = (gen[:, 1:] == gen[:, :-1]).float().mean().item()
        repetition_rate = immediate_reps
    metrics = {
        "accuracy": exact_match.mean().item(),
        "avg_gen_prob": gen_probs.mean().item(),
        "reward_mean": reward.mean().item(),
        "reward_std": reward.std().item(),
        "repetition_rate": repetition_rate,
        **repetition_stats
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
class CharacterDifficultyAnalyzer:
    """Analyzes character-level difficulty metrics for curriculum learning"""
    def __init__(self, text: str, stoi: Dict[str, int], vocab_size: int, device: torch.device):
        self.vocab_size = vocab_size
        self.device = device
        text_indices = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
        self.char_counts = torch.zeros(vocab_size, device=device)
        self.char_counts.index_add_(0, text_indices, torch.ones_like(text_indices, dtype=torch.float32))
        self.char_freq = self.char_counts / self.char_counts.sum()
        self.rare_threshold = torch.quantile(self.char_freq, 0.2)
        self.is_rare_char = self.char_freq < self.rare_threshold
        bigram_counts = torch.zeros(vocab_size, vocab_size, device=device, dtype=torch.float32)
        if len(text_indices) > 1:
            text_indices_shifted = text_indices[1:]
            text_indices_base = text_indices[:-1]
            indices = torch.stack([text_indices_base, text_indices_shifted], dim=0)
            bigram_counts.index_put_(tuple(indices), 
                                    torch.ones(len(text)-1, device=device), 
                                    accumulate=True)
        row_sums = bigram_counts.sum(dim=1, keepdim=True)
        self.bigram_probs = bigram_counts / (row_sums + 1e-8)
        self.log_char_freq = torch.log(self.char_freq + 1e-8)
        self.char_entropy = -torch.sum(self.char_freq * self.log_char_freq)
    def compute_batch_difficulty(self, sequences: torch.Tensor) -> torch.Tensor:
        """Compute difficulty scores for a batch of sequences"""
        B, T = sequences.shape
        char_freqs = self.char_freq[sequences]
        rare_mask = self.is_rare_char[sequences]
        char_freq_scores = char_freqs.mean(dim=1)
        rare_ratios = rare_mask.float().mean(dim=1)
        if T > 1:
            seq_flat = sequences.view(-1)
            idx1 = seq_flat[:-B].view(B, T-1)
            idx2 = seq_flat[B:].view(B, T-1)
            bigram_probs = self.bigram_probs[idx1, idx2]
            bigram_scores = bigram_probs.mean(dim=1)
        else:
            bigram_scores = torch.zeros(B, device=sequences.device)
        return (1.0 - (0.4 * char_freq_scores + 
                      0.4 * bigram_scores + 
                      0.2 * (1.0 - rare_ratios))).clamp(0.0, 1.0)
class CurriculumManager:
    """Manages curriculum learning stages and progression with word-based rewards"""
    def __init__(self, device: torch.device, vocab_size: int = 65,
                 ema_alpha: float = 0.95,
                 promotion_threshold: float = 0.05,
                 promotion_patience: int = 100,
                 min_iterations_per_stage: int = 500,
                 total_iterations: Optional[int] = None):
        self.device = device
        self.vocab_size = vocab_size
        self.current_stage = 0  
        self.ema_alpha = ema_alpha
        self.promotion_threshold = promotion_threshold
        if total_iterations is not None:
            self.min_iterations_per_stage = max(1, int(total_iterations * 0.05))  
            # Updated: Increased patience to 10% of total iterations for EMA-based promotion
            self.promotion_patience = max(2, int(total_iterations * 0.10))  
        else:
            self.promotion_patience = promotion_patience
            self.min_iterations_per_stage = min_iterations_per_stage
        self.ema_metrics = {
            'accuracy': 0.0,
            'reward_mean': 0.0,
            'kl_divergence': 0.0,
            'loss': float('inf')
        }
        self.iterations_in_stage = 0
        self.best_ema_reward = -float('inf')
        self.stage_start_metrics = {}
        boundary_chars = " .,!?:;\n'-\""
        self.boundary_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        for char in boundary_chars:
            if ord(char) < vocab_size:
                self.boundary_mask[ord(char)] = True
        self._compile_stage_configs()
        self.word_count_buffer = torch.zeros(256, 5, device=device)
        self.n_stages = 5  
        self.repetition_penalty_weights = {
            0: 0.0,   
            1: 0.0,   
            2: 0.5,   
            3: 0.7,   
            4: 0.9,   
        }
        self.ref_model_kl_weights = {
            0: 0.0,    
            1: 0.0,    
            2: 0.0,    
            3: 0.05,   
            4: 0.1,    
        }
        self.gpt2_kl_weights = {
            0: 0.05,    
            1: 0.05,    
            2: 0.05,    
            3: 0.05,   
            4: 0.1,    
        }
        self.perplexity_improvement_thresholds = {
            0: 0.70,  
            1: 0.50,  
            2: 0.35,  
            3: 0.25,  
            4: float('inf')  
        }
        self.initial_perplexity = None
        self.baseline_perplexity_per_stage = {}  
        self.loss_history = deque(maxlen=50)
        self.reward_plateau_counter = 0
        self.current_perplexity = None  # Will be set from initial measurement
        
        # EMA history tracking for the new trigger mechanism
        self.ema_history_window_size = max(1, int(self.promotion_patience))  # 10% of total iterations
        self.ema_reward_history = deque(maxlen=self.ema_history_window_size)
        self.ema_improvement_threshold = 0.05  # 5% improvement threshold
    def _compile_stage_configs(self):
        """Pre-compile all stage configurations as tensors"""
        self.stage_configs = {
            0: {  
                'name': 'Bigram Foundation',
                'ngram_weights': torch.tensor([1.0, 0.0, 0.0], device=self.device),
                'word_weights': torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=self.device),
                'reward_scale': 1.0,
                'kl_weight': 0.03,
                'confidence_weight': 0.2,
                'horizon_max': 8,
                'focus': 'Learn character transitions only'
            },
            1: {  
                'name': '2-3 Letter Words',
                'ngram_weights': torch.tensor([0.5, 0.0, 0.0], device=self.device),
                'word_weights': torch.tensor([0.1, 0.27, 0.0, 0.0, 0.0], device=self.device),
                'reward_scale': 1.0,
                'kl_weight': 0.05,
                'confidence_weight': 0.3,
                'horizon_max': 8,
                'focus': 'Learn simple 2-3 letter words while maintaining bigram patterns'
            },
            2: {  
                'name': '2-4 Letter Words',
                'ngram_weights': torch.tensor([0.5, 0.0, 0.0], device=self.device),
                'word_weights': torch.tensor([0.06, 0.16, 0.28, 0.0, 0.0], device=self.device),
                'reward_scale': 1.2,
                'kl_weight': 0.08,
                'confidence_weight': 0.5,
                'horizon_max': 8,
                'focus': 'Equal emphasis on patterns and word formation'
            },
            3: {  
                'name': 'Full Word Spectrum',
                'ngram_weights': torch.tensor([0.5, 0.0, 0.0], device=self.device),
                'word_weights': torch.tensor([0.15, 0.41, 0.73, 0.21, 0.0], device=self.device),
                'reward_scale': 1.5,
                'kl_weight': 0.1,
                'confidence_weight': 0.7,
                'horizon_max': 8,
                'focus': 'Prioritize word formation and vocabulary expansion'
            },
            4: {  
                'name': 'Full Optimization',
                'ngram_weights': torch.tensor([0.5, 0.0, 0.0], device=self.device),
                'word_weights': torch.tensor([0.1, 0.27, 0.48, 1.35, 3.5], device=self.device),
                'reward_scale': 2.0,
                'kl_weight': 0.1,
                'confidence_weight': 0.7,
                'horizon_max': 8,
                'focus': 'Heavy emphasis on complete words and semantic coherence'
            }
        }
        for stage, config in self.stage_configs.items():
            config['ngram_sum'] = config['ngram_weights'].sum().item()
            config['word_sum'] = config['word_weights'].sum().item()
            config['total_sum'] = config['ngram_sum'] + config['word_sum']
    def calculate_perplexity(self, loss: float) -> float:
        """Calculate perplexity from cross-entropy loss
        
        Args:
            loss: Cross-entropy loss (negative log likelihood)
        
        Returns:
            Perplexity value (exp(loss))
        """
        # Cap loss to prevent overflow
        capped_loss = min(loss, 20.0)
        return torch.exp(torch.tensor(capped_loss)).item()
    def _calculate_loss_improvement_rate(self) -> float:
        """Calculate rate of loss improvement over recent window"""
        if len(self.loss_history) < 20:
            return 1.0  
        recent_avg = np.mean(list(self.loss_history)[-10:])
        previous_avg = np.mean(list(self.loss_history)[-20:-10])
        if previous_avg == 0:
            return 0.0
        improvement_rate = (previous_avg - recent_avg) / previous_avg
        return improvement_rate
    def update_metrics(self, metrics: Dict[str, float]):
        """Update EMA metrics"""
        for key in self.ema_metrics:
            if key in metrics:
                if self.iterations_in_stage == 0:
                    self.ema_metrics[key] = metrics[key]
                else:
                    self.ema_metrics[key] = (self.ema_alpha * self.ema_metrics[key] + 
                                            (1 - self.ema_alpha) * metrics[key])
        if 'loss' in metrics:
            self.loss_history.append(metrics['loss'])
        if 'cross_entropy' in self.ema_metrics:
            self.current_perplexity = self.calculate_perplexity(self.ema_metrics['cross_entropy'])
        else:
            # Keep perplexity as inf if cross_entropy not available yet
            # This prevents using the wrong loss metric for perplexity
            pass
        
        # Track EMA reward history for the new trigger mechanism
        self.ema_reward_history.append(self.ema_metrics['reward_mean'])
        
        self.iterations_in_stage += 1
    def check_promotion(self) -> Tuple[bool, str]:
        """Check if ready to promote to next stage"""
        if self.current_stage >= self.n_stages - 1:
            return False, "Already at final stage"
        if self.iterations_in_stage < self.min_iterations_per_stage:
            return False, f"min_iterations_not_met"
        if self.initial_perplexity is None:
            if self.current_perplexity < float('inf'):
                self.initial_perplexity = self.current_perplexity
                self.baseline_perplexity_per_stage[0] = self.current_perplexity
        if self.initial_perplexity is not None:
            current_perplexity = self.current_perplexity
            improvement_threshold = self.perplexity_improvement_thresholds[self.current_stage]
            target_perplexity = self.initial_perplexity * improvement_threshold
            if current_perplexity <= target_perplexity:
                improvement_percent = (1.0 - current_perplexity/self.initial_perplexity) * 100
                return True, f"perplexity_threshold_met ({current_perplexity:.2f} <= {target_perplexity:.2f}, {improvement_percent:.1f}% improvement)"
        
        # New EMA-based trigger: Check if current EMA crosses 5% above the 20%-ago EMA
        if len(self.ema_reward_history) >= self.ema_history_window_size:
            # Get the EMA from 20% iterations ago (oldest in the deque)
            past_ema_reward = self.ema_reward_history[0]
            current_ema_reward = self.ema_metrics['reward_mean']
            
            # Calculate improvement percentage
            if abs(past_ema_reward) > 1e-8:  # Avoid division by zero
                improvement = (current_ema_reward - past_ema_reward) / abs(past_ema_reward)
                
                # Trigger promotion if improvement exceeds 5%
                if improvement >= self.ema_improvement_threshold:
                    improvement_percent = improvement * 100
                    return True, f"ema_improvement_trigger ({current_ema_reward:.4f} is {improvement_percent:.1f}% above past EMA {past_ema_reward:.4f})"
        
        # Loss plateau detection removed due to issues with negative losses in RL
        # The other triggers (perplexity, EMA reward, reward plateau) are more reliable
        reward_improvement = (self.ema_metrics['reward_mean'] - self.best_ema_reward) / (abs(self.best_ema_reward) + 1e-8)
        if reward_improvement < self.promotion_threshold:
            self.reward_plateau_counter += 1
            if self.reward_plateau_counter >= self.promotion_patience:
                return True, "reward_plateau_detected"
        else:
            self.reward_plateau_counter = 0
            self.best_ema_reward = max(self.best_ema_reward, self.ema_metrics['reward_mean'])
        return False, "continue_training"
    def promote(self):
        """Promote to next stage"""
        if self.current_stage < self.n_stages - 1:
            self.stage_start_metrics[self.current_stage] = self.ema_metrics.copy()
            self.baseline_perplexity_per_stage[self.current_stage + 1] = self.current_perplexity
            self.current_stage += 1
            self.iterations_in_stage = 0
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
            'best_ema_reward': self.best_ema_reward,
            'stage_start_metrics': self.stage_start_metrics
        }
    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint"""
        self.current_stage = state_dict['current_stage']
        self.iterations_in_stage = state_dict['iterations_in_stage']
        self.ema_metrics = state_dict['ema_metrics']
        self.best_ema_reward = state_dict['best_ema_reward']
        self.stage_start_metrics = state_dict['stage_start_metrics']
    def detect_words_gpu(self, sequences: torch.Tensor) -> torch.Tensor:
        """GPU-native word detection"""
        B, T = sequences.shape
        is_boundary = self.boundary_mask[sequences]
        is_boundary[:, 0] = True
        if T > 1:
            is_boundary[:, -1] = True
        word_counts = self.word_count_buffer[:B, :].zero_()
        for b in range(B):
            boundaries = torch.where(is_boundary[b])[0]
            if len(boundaries) > 1:
                word_lengths = boundaries[1:] - boundaries[:-1]
                for length in range(2, 7):
                    word_counts[b, length-2] = (word_lengths == length).sum()
        return word_counts
    def get_repetition_penalty_weight(self) -> float:
        """Get repetition penalty weight for current stage"""
        return self.repetition_penalty_weights.get(self.current_stage, 0.0)
    def get_ref_model_kl_weight(self) -> float:
        """Get reference model KL penalty weight for current stage"""
        return self.ref_model_kl_weights.get(self.current_stage, 0.0)
    def get_gpt2_kl_weight(self) -> float:
        """Get GPT-2 KL penalty weight for current stage"""
        return self.gpt2_kl_weights.get(self.current_stage, 0.0)
    def compute_curriculum_rewards(self, base_rewards: torch.Tensor,
                                 ngram_scores: torch.Tensor,
                                 sequences: torch.Tensor) -> torch.Tensor:
        """Compute combined n-gram and word rewards (bigram only)"""
        word_counts = self.detect_words_gpu(sequences)
        stage_config = self.get_stage_config()
        ngram_weights = stage_config['ngram_weights']
        word_weights = stage_config['word_weights']
        bigram_score = ngram_scores[:, 0:1]  
        ngram_reward = bigram_score * ngram_weights[0]
        word_reward = (word_counts * word_weights).sum(dim=1)
        ngram_sum = ngram_weights[0]  
        word_sum = stage_config['word_sum']
        if ngram_sum > 0:
            ngram_reward = ngram_reward.squeeze(1) / ngram_sum
        else:
            ngram_reward = ngram_reward.squeeze(1)
        if word_sum > 0:
            word_reward /= word_sum
        combined_reward = ngram_reward + word_reward
        return base_rewards * combined_reward * stage_config['reward_scale']
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
def setup_cuda_optimizations():
    """Configure CUDA settings for optimal performance"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.95)
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
    if rank == 0:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    else:
        checkpoint = None
    checkpoint_exists = torch.tensor([1 if checkpoint is not None else 0], device=device)
    if dist.is_initialized():
        dist.broadcast(checkpoint_exists, src=0)
    if checkpoint_exists.item() == 0:
        return {}
    if rank != 0:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
flash_attn_wheel = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch==2.5.0", "tqdm", "wandb", "requests", "matplotlib", "nvidia-ml-py3")
    .pip_install(flash_attn_wheel)
)
volume = modal.Volume.from_name("grpo-data", create_if_missing=True)
shakespeare_volume = modal.Volume.from_name("nanogpt-data", create_if_missing=False)
@app.function(
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    volumes={
        "/data": shakespeare_volume,  # Mount nanogpt-data as /data to access checkpoints
        "/grpo": volume
    },
    timeout=60 * 60 * 6,
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_distributed():
    """Launch distributed training with torchrun - following modal_nanogpt_simple.py pattern"""
    import subprocess
    print(f"Launching distributed training on {N_GPUS} {GPU_TYPE} GPUs")
    ensure_dataset("input.txt")
    script_content = Path(__file__).read_text()
    temp_script = "/tmp/train_multi_gpu.py"
    Path(temp_script).write_text(script_content)
    # Stream output in real-time instead of capturing
    result = subprocess.run([
        "torchrun", f"--nproc-per-node={N_GPUS}", temp_script
    ])
    
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args)
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
        setup_cuda_optimizations()
        if world_size > 1:
            test_tensor = torch.tensor([rank], dtype=torch.float32, device=device)
            print(f"Rank {rank} before all_reduce: {test_tensor.item()}")
            all_reduce_mean(test_tensor)
            print(f"Rank {rank} after all_reduce (should be average): {test_tensor.item()}")
            if rank == 0:
                broadcast_tensor = torch.tensor([42.0], device=device)
            else:
                broadcast_tensor = torch.zeros(1, device=device)
            dist.broadcast(broadcast_tensor, src=0)
            print(f"Rank {rank} broadcast result (should be 42): {broadcast_tensor.item()}")
        print_rank0("\n=== Creating Model on All Ranks ===", rank)
        vocab_size = 65  
        model = GPT(vocab_size).to(device)
        if rank == 0:
            print(f"Created GPT model with {sum(p.numel() for p in model.parameters()):,} parameters")
        if world_size > 1:
            print_rank0("\n=== Testing DDP Wrapper ===", rank)
            ddp_model = setup_ddp_with_gradient_bucketing(model, device)
            print_rank0("Successfully wrapped model in DDP with gradient bucketing", rank)
            if dist.is_initialized():
                dist.barrier()
        print_rank0("\n=== Testing Distributed Data Loading ===", rank)
        ensure_dataset()
        encode, decode, vocab_size_actual, stoi, itos, text = build_vocab()
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
        ctx, tgt = loader.next()
        print(f"Rank {rank} - Data loader test: ctx shape {ctx.shape}, tgt shape {tgt.shape}")
        print(f"Rank {rank} - Starting position: {loader.pos}")
        print(f"Rank {rank} - Dataset length: {len(loader.data)}, Start position: {loader.pos}")
        if rank == 0:
            print("\n=== Testing Model Forward Pass ===")
            batch_size = 4
            seq_len = CONTEXT_LEN
            dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            with torch.no_grad():
                if world_size > 1:
                    output = ddp_model(dummy_input)
                else:
                    output = model(dummy_input)
            print(f"Model output shape: {output.shape} (expected: {(batch_size, seq_len, vocab_size)})")
            # Bigram reference model disabled - using only Shakespeare model KL
            # ref_model = OnTheFlyNGramRef(text, stoi, vocab_size).to(device)
            # ref_output = ref_model(dummy_input)
            # print(f"Reference model output shape: {ref_output.shape}")
            attn_module = model.layers[0].attn
            print(f"Flash Attention enabled: {attn_module.use_flash_attn}")
            print("\n=== Testing GRPO Components ===")
            all_chars, log_probs = generate_exhaustive_single_char(model, ctx, vocab_size)
            print(f"Exhaustive generation: all_chars shape {all_chars.shape}, log_probs shape {log_probs.shape}")
            ref_char = tgt[:, 0]  
            # Skipping exhaustive rewards test - ref_model disabled
            # rewards = compute_exhaustive_rewards(
            #     all_chars, ref_char, ref_model, ctx, vocab_size,
            #     model_log_probs=log_probs
            # )
            rewards = torch.zeros(all_chars.shape[0], all_chars.shape[1], device=device)
            print(f"Rewards shape: {rewards.shape}, mean: {rewards.mean():.3f}, std: {rewards.std():.3f}")
        print_rank0("\nAll tests complete!", rank)
    except Exception as e:
        print(f"Error on rank {os.environ.get('RANK', 0)}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()
class CharTokenAligner:
    """Aligns character sequences to GPT-2 tokens with caching"""
    def __init__(self, vocab_size: int, itos: Dict[int, str], device: torch.device, cache_size: int = 10000):
        self.vocab_size = vocab_size
        self.itos = itos
        self.device = device
        self.cache = {}
        self.cache_size = cache_size
        self.tokenizer = None
        self.gpt2_vocab_size = None
    def _init_tokenizer(self):
        """Lazy initialization of tokenizer"""
        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.gpt2_vocab_size = self.tokenizer.vocab_size
        except ImportError:
            print("Warning: transformers not available, GPT-2 KL penalty disabled")
            return False
        return True
    def align_sequence(self, char_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """Convert character indices to token IDs with caching"""
        if self.tokenizer is None:
            if not self._init_tokenizer():
                return None
        chars = ''.join([self.itos[idx.item()] for idx in char_indices])
        if chars in self.cache:
            return self.cache[chars]
        tokens = self.tokenizer.encode(chars, return_tensors='pt').to(self.device)
        if len(self.cache) < self.cache_size:
            self.cache[chars] = tokens
        return tokens
def load_shakespeare_checkpoint(device: torch.device, checkpoint_url: str = None) -> Optional[nn.Module]:
    """Load Shakespeare-trained GPT-2 style model from checkpoint"""
    try:
        import requests
        import tempfile
        if checkpoint_url is None:
            checkpoint_url = "https://modal.com/api/volumes/tokenbender/main/nanogpt-data/files/content?path=checkpoints%2Fshakespeare%2Fckpt.pt"
        print(f"[gpt2] Downloading Shakespeare checkpoint...")
        response = requests.get(checkpoint_url, stream=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        print(f"[gpt2] Loading checkpoint from {tmp_path}...")
        checkpoint = torch.load(tmp_path, map_location=device, weights_only=True)
        model_args = checkpoint.get('model_args', {})
        from dataclasses import dataclass
        @dataclass
        class GPTConfig:
            block_size: int = 1024
            vocab_size: int = 50304  
            n_layer: int = 12
            n_head: int = 12
            n_embd: int = 768
            dropout: float = 0.0
            bias: bool = True
        config = GPTConfig(**model_args)
        model = GPT(
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_emb=config.n_embd,
            context_len=config.block_size
        ).to(device)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        os.unlink(tmp_path)
        print(f"[gpt2] Successfully loaded Shakespeare model")
        return model
    except Exception as e:
        print(f"[gpt2] Failed to load Shakespeare checkpoint: {e}")
        return None
def compute_gpt2_kl_penalty(
    char_logits: torch.Tensor,      
    char_sequences: torch.Tensor,   
    gpt2_model: nn.Module,          
    aligner: CharTokenAligner,      
    stage: int,                     
    device: torch.device
) -> torch.Tensor:
    """
    Compute KL divergence penalty using pretrained model as reference
    Returns: [B] KL penalty per sequence
    """
    B, T, V = char_logits.shape
    kl_penalties = torch.zeros(B, device=device)
    if gpt2_model is None:
        return kl_penalties
    with torch.no_grad():
        for b in range(B):
            char_seq = char_sequences[b]
            token_ids = aligner.align_sequence(char_seq)
            if token_ids is None:
                continue
            if token_ids.shape[1] == 0:
                continue
            try:
                ref_outputs = gpt2_model(token_ids)
                if hasattr(ref_outputs, 'logits'):
                    ref_logits = ref_outputs.logits  
                else:
                    ref_logits = ref_outputs  
                ref_char_probs = torch.ones(T, V, device=device) / V
                char_probs = F.softmax(char_logits[b], dim=-1)
                kl = F.kl_div(
                    torch.log(ref_char_probs + 1e-8),
                    char_probs,
                    reduction='none'
                ).sum(dim=-1).mean()
                kl_penalties[b] = kl.clamp(0.0, 5.0)  
            except Exception as e:
                print(f"[gpt2] Error computing KL for sequence {b}: {e}")
                continue
    return kl_penalties
def train_grpo():
    """Main GRPO training loop with multi-GPU support"""
    rank, world_size, local_rank, device = setup_distributed()
    setup_cuda_optimizations()
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
    character_analyzer = CharacterDifficultyAnalyzer(text, stoi, vocab_size, device)
    # Bigram reference model disabled - using only Shakespeare model KL
    # ref_model = OnTheFlyNGramRef(text, stoi, vocab_size).to(device).eval()
    ref_model = None  # Placeholder to avoid errors
    if world_size > 1 and dist.is_initialized():
        dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
    torch.manual_seed(42)  
    torch.cuda.manual_seed(42)
    actor = GPT(vocab_size).to(device)
    actor_old = GPT(vocab_size).to(device)
    actor_old.load_state_dict(actor.state_dict())
    actor_old.eval()
    if world_size > 1 and dist.is_initialized():
        dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
    if world_size > 1:
        actor = setup_ddp_with_gradient_bucketing(actor, device, BUCKET_SIZE_MB)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
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
    # KL controller disabled - using only Shakespeare model KL
    # kl_controller = AdaptiveKLController(target_kl=0.02)
    kl_controller = None
    confidence_monitor = ConfidenceMonitor() if USE_CONFIDENCE_SCALING else None
    # Adjusted EMA/plateau compatibility: Using 5% promotion threshold for better stability
    curriculum_manager = CurriculumManager(device=device, vocab_size=vocab_size, total_iterations=TOTAL_ITERS,
                                         promotion_threshold=0.05, ema_alpha=0.95)
    shakespeare_model = None
    char_token_aligner = None
    if curriculum_manager.current_stage >= 3 or any(curriculum_manager.gpt2_kl_weights[s] > 0 for s in range(3, 5)):
        shakespeare_model = load_shakespeare_reference_model("/data/checkpoints/shakespeare/ckpt.pt", vocab_size, device, rank)
        if shakespeare_model is not None:
            char_token_aligner = CharTokenAligner(vocab_size, itos, device)
            print_rank0("[GPT-2 KL] Initialized Shakespeare reference model for KL penalty", rank)
    chars_seen = 0
    # current_kl_coef = BETA_KL  # Disabled - using only Shakespeare model KL
    current_kl_coef = 0.0
    if rank == 0:
        import wandb
        wandb.init(
            project="avataRL_12",
            name="perplexity_fix_2",
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
    from collections import defaultdict
    from tqdm import tqdm
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    all_metrics = defaultdict(list)
    for it in tqdm(range(1, TOTAL_ITERS + 1), desc="Training", disable=(rank != 0)):
        actor.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        iter_metrics = defaultdict(list)
        
        # Calculate initial cross-entropy for perplexity before first iteration
        if it == 1:
            with torch.no_grad():
                # Get a sample batch to calculate initial perplexity
                ctx_sample, ref_tok_sample = loader.next()
                initial_logits = actor(ctx_sample)[:, -HORIZON:]
                initial_ce = F.cross_entropy(initial_logits.reshape(-1, vocab_size), ref_tok_sample.reshape(-1))
                iter_metrics["cross_entropy"].append(initial_ce.item())
                initial_perplexity = torch.exp(initial_ce).item()
                print_rank0(f"Initial perplexity: {initial_perplexity:.2f} (expected ~65 for random init)", rank)
                
                # Set the initial perplexity in curriculum manager
                curriculum_manager.current_perplexity = initial_perplexity
                curriculum_manager.initial_perplexity = initial_perplexity
                curriculum_manager.baseline_perplexity_per_stage[0] = initial_perplexity
                
                # Also initialize the EMA metric to avoid inf
                curriculum_manager.ema_metrics['cross_entropy'] = initial_ce.item()
        for micro_step in range(GRAD_ACCUM):
            ctx, ref_tok = loader.next()
            old_probs = None
            if USE_CONFIDENCE_SCALING:
                with torch.no_grad():
                    old_logits = actor_old(ctx[:, -actor_old.module.context_len if hasattr(actor_old, 'module') else -actor_old.context_len:])[:, -1, :]
                    old_probs = F.softmax(old_logits, dim=-1)
            if USE_EXHAUSTIVE:
                with torch.no_grad():
                    G = generate_with_temperature(actor_old, ctx, HORIZON, K_SAMPLES, vocab_size, temperature=TEMPERATURE)
                with torch.no_grad():
                    model_logits = actor_old(ctx[:, -actor_old.module.context_len if hasattr(actor_old, 'module') else -actor_old.context_len:])[:, -1, :]
                    model_pred = model_logits.argmax(dim=-1)
                    model_accuracy = (model_pred == ref_tok[:, 0]).float().mean().item()
                    iter_metrics["accuracy"].append(model_accuracy)
                    # Bigram accuracy calculation disabled - using only Shakespeare model KL
                    # ref_full = torch.cat([ctx, ref_tok], dim=1)
                    # _, components = ref_model(ref_full[:, -HORIZON-1:], return_components=True)
                    # bigram_logits = components['bigram'][:, -1, :]
                    # bigram_pred = bigram_logits.argmax(dim=-1)
                    # bigram_correct = (bigram_pred == ref_tok[:, 0]).float().mean().item()
                    # iter_metrics["bigram_accuracy"].append(bigram_correct)
                    iter_metrics["bigram_accuracy"].append(0.0)  # Placeholder for metrics
                B = ctx.shape[0]  
                R = torch.zeros_like(G, dtype=torch.float32)
                for k in range(K_SAMPLES):
                    gen = G[:, k, :]
                    # Bigram reference model disabled
                    # with torch.no_grad():
                    #     full_seq = torch.cat([ctx, gen], dim=1)
                    #     ref_logits = ref_model(full_seq[:, -HORIZON-1:])[:, -HORIZON:]
                    ref_logits = None  # Will be handled in compute_rewards_multi_char
                    if micro_step == 0 and k == 0:
                        # Bigram reference model disabled
                        # with torch.no_grad():
                        #     ref_full = torch.cat([ctx, ref_tok], dim=1)
                        #     _, components = ref_model(ref_full[:, -HORIZON-1:], return_components=True)
                        #     bigram_acc = (components['bigram'][:, -1, :].argmax(dim=-1) == ref_tok[:, 0]).float().mean()
                        #     ngram_scores = torch.stack([bigram_acc, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)]).unsqueeze(0).expand(B, -1)
                        ngram_scores = torch.zeros(B, 3, device=device)  # Placeholder
                    reward, metrics = compute_rewards_multi_char(gen, ref_tok, ref_logits,
                                                                curriculum_manager, ctx, ngram_scores)
                    R[:, k, :] = reward
                    for key, val in metrics.items():
                        iter_metrics[key].append(val)
                base = R.mean(dim=1, keepdim=True)
                adv = R - base
                adv_std = torch.maximum(adv.std(dim=1, keepdim=True), torch.tensor(MIN_VARIANCE, device=device))
                adv = adv / adv_std
                iter_metrics["reward_mean"].append(R.mean().item())
                iter_metrics["reward_std"].append(R.std().item())
                iter_metrics["advantage_std"].append(adv.std().item())
                flat_G = G.reshape(-1, HORIZON)
                flat_adv = adv.reshape(-1, HORIZON)
            ctx_for_loss = ctx.repeat_interleave(K_SAMPLES, dim=0)
            full_seq = torch.cat([ctx_for_loss, flat_G], dim=1)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                context_len = actor.module.context_len if hasattr(actor, 'module') else actor.context_len
                input_seq = full_seq[:, -context_len:]
                new_logits = actor(input_seq)[:, -HORIZON:]
                new_dist = torch.distributions.Categorical(logits=new_logits)
                logp_new = new_dist.log_prob(flat_G.squeeze(-1))
                with torch.no_grad():
                    old_logits = actor_old(input_seq)[:, -HORIZON:]
                    old_dist = torch.distributions.Categorical(logits=old_logits)
                    logp_old = old_dist.log_prob(flat_G.squeeze(-1))
                ratio = torch.exp(logp_new - logp_old)
                clipped_ratio = torch.clamp(ratio, 1-CLIP_RATIO, 1+CLIP_RATIO)
                pol_loss1 = -flat_adv.squeeze(-1) * ratio
                pol_loss2 = -flat_adv.squeeze(-1) * clipped_ratio
                pol_loss = torch.max(pol_loss1, pol_loss2).mean()
                # Bigram reference model KL penalty disabled - using only Shakespeare model KL
                # ref_logits = ref_model(full_seq[:, -HORIZON-1:])[:, -HORIZON:]
                # true_probs = torch.zeros_like(new_logits)
                # true_probs.scatter_(2, flat_G.unsqueeze(-1), 1.0)
                # new_probs = F.softmax(new_logits, dim=-1)
                # true_char_probs = new_probs.gather(2, flat_G.unsqueeze(-1)).squeeze(-1)
                # kl_divergence = -torch.log(true_char_probs + 1e-8).mean()
                # kl = torch.clamp(kl_divergence, max=5.0)
                # if kl_controller and micro_step == 0:
                #     current_kl_coef = kl_controller.update(kl.item(), current_kl_coef)
                kl = torch.tensor(0.0, device=device)  # Placeholder for metrics
                entropy = new_dist.entropy().mean()
                entropy_bonus = ENTROPY_COEF  
                iter_metrics["entropy"].append(entropy.item())
                gpt2_kl_penalty_term = torch.tensor(0.0, device=device)
                if shakespeare_model is not None:
                    gpt2_kl_weight = curriculum_manager.get_gpt2_kl_weight()
                    if gpt2_kl_weight > 0:
                        shakespeare_kl = compute_shakespeare_kl_penalty(
                            flat_G,              
                            new_logits,          
                            shakespeare_model,   
                            curriculum_manager.current_stage,  
                            device
                        )
                        gpt2_kl_penalty_term = gpt2_kl_weight * shakespeare_kl.mean()
                        iter_metrics["gpt2_kl"].append(shakespeare_kl.mean().item())
                loss = (pol_loss - entropy_bonus * entropy + gpt2_kl_penalty_term) / GRAD_ACCUM
            
            # Calculate cross-entropy loss for perplexity (not used for training)
            # Fixed: Calculate CE on true data distribution, not generated sequences
            with torch.no_grad():
                # Get model predictions on true context
                true_logits = actor(ctx)[:, -HORIZON:]
                # Calculate CE against true reference tokens
                ce_loss = F.cross_entropy(true_logits.reshape(-1, vocab_size), ref_tok.reshape(-1))
                iter_metrics["cross_entropy"].append(ce_loss.item())
                
                # Sanity check: perplexity should be > 2.0 for any reasonable model
                # Random init should give ~65 for 65-char vocabulary
                perplexity = torch.exp(ce_loss).item()
                if perplexity < 2.0:
                    print_rank0(f"WARNING: Suspiciously low perplexity {perplexity:.4f} at iteration {it}. "
                              f"Expected >2.0 (random init ~65). Check calculation.", rank)
                
                # Debug print for first few iterations
                if it <= 3 and rank == 0:
                    print(f"[DEBUG] Iteration {it}: CE Loss = {ce_loss.item():.4f}, Perplexity = {perplexity:.4f}")
            
            scaler.scale(loss).backward()
            total_loss += loss.item()
            iter_metrics["pol_loss"].append(pol_loss.item())
            iter_metrics["kl"].append(kl.item())
            iter_metrics["entropy"].append(entropy.item())
            iter_metrics["ratio"].append(ratio.mean().item())
            iter_metrics["ratio_max"].append(ratio.max().item())
            if it == 1 and micro_step == 0 and rank == 0:
                print(f"Debug - Policy loss: {pol_loss.item():.4f}, KL: {kl.item():.4f}")
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        chars_seen += BATCH * HORIZON
        if world_size > 1:
            metric_tensors = {}
            for key, values in iter_metrics.items():
                if values:  
                    metric_tensor = torch.tensor(np.mean(values), device=device, dtype=torch.float32)
                    all_reduce_mean(metric_tensor)
                    metric_tensors[key] = metric_tensor.item()
                else:
                    metric_tensors[key] = 0.0
        else:
            metric_tensors = {k: np.mean(v) if v else 0.0 for k, v in iter_metrics.items()}
        curriculum_metrics = {
            'accuracy': metric_tensors.get('accuracy', 0.0),
            'reward_mean': metric_tensors.get('reward_mean', 0.0),
            'kl_divergence': metric_tensors.get('kl', 0.0),
            'loss': total_loss * GRAD_ACCUM,
            'cross_entropy': metric_tensors.get('cross_entropy', 0.0)
        }
        curriculum_manager.update_metrics(curriculum_metrics)
        should_promote, promotion_reason = curriculum_manager.check_promotion()
        if should_promote:
            curriculum_manager.promote()
            stage_config = curriculum_manager.get_stage_config()
            if rank == 0:
                print(f"\n[Curriculum] Promoted to Stage {curriculum_manager.current_stage}: {stage_config['name']}")
                print(f"[Curriculum] Reason: {promotion_reason}")
                print(f"[Curriculum] Focus: {stage_config['focus']}")
            if curriculum_manager.current_stage >= 3 and shakespeare_model is None:
                shakespeare_model = load_shakespeare_reference_model("/data/checkpoints/shakespeare/ckpt.pt", vocab_size, device, rank)
                if shakespeare_model is not None:
                    char_token_aligner = CharTokenAligner(vocab_size, itos, device)
                    print_rank0("[GPT-2 KL] Loaded Shakespeare reference model for stage 3+", rank)
        if rank == 0 and it % 2 == 0:
            wandb.log({
                "loss/total": total_loss * GRAD_ACCUM,
                "loss/policy": metric_tensors.get("pol_loss", 0),
                "loss/kl": metric_tensors.get("kl", 0),
                "loss/entropy": metric_tensors.get("entropy", 0),
                "loss/kl_coef": current_kl_coef,
                "metrics/accuracy": metric_tensors.get("accuracy", 0),
                "metrics/reward_mean": metric_tensors.get("reward_mean", 0),
                "metrics/reward_std": metric_tensors.get("reward_std", 0),
                "metrics/advantage_std": metric_tensors.get("advantage_std", 0),
                "metrics/ratio_mean": metric_tensors.get("ratio", 1.0),
                "metrics/ratio_max": metric_tensors.get("ratio_max", 1.0),
                "metrics/repetition_rate": metric_tensors.get("repetition_rate", 0.0),
                "metrics/repetition_penalty_mean": metric_tensors.get("repetition_penalty_mean", 0.0),
                "metrics/gpt2_kl": metric_tensors.get("gpt2_kl", 0.0),
                "training/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "training/chars_seen": chars_seen,
                "training/iteration": it,
                "training/epoch": it / ITERS_PER_EPOCH,
                "curriculum/stage": curriculum_manager.current_stage,
                "curriculum/iterations_in_stage": curriculum_manager.iterations_in_stage,
                "curriculum/ema_accuracy": curriculum_manager.ema_metrics['accuracy'],
                "curriculum/ema_reward": curriculum_manager.ema_metrics['reward_mean'],
                "curriculum/perplexity": curriculum_manager.current_perplexity if curriculum_manager.current_perplexity is not None else float('inf'),
                "curriculum/perplexity_threshold": curriculum_manager.initial_perplexity * curriculum_manager.perplexity_improvement_thresholds[curriculum_manager.current_stage] if curriculum_manager.initial_perplexity is not None else float('inf'),
                "curriculum/loss_improvement_rate": curriculum_manager._calculate_loss_improvement_rate(),
            })
        # Log every iteration for better visibility
        if rank == 0:
            print(f"Iter {it}: loss={total_loss*GRAD_ACCUM:.4f}, acc={metric_tensors.get('accuracy', 0):.3f}, "
                  f"kl={metric_tensors.get('kl', 0):.4f}, gpt2_kl={metric_tensors.get('gpt2_kl', 0):.4f}, "
                  f"stage={curriculum_manager.current_stage}, chars={chars_seen:,}", flush=True)
        
        if it % 5 == 0:
            actor_state = actor.module.state_dict() if hasattr(actor, 'module') else actor.state_dict()
            actor_old.load_state_dict(actor_state)
            if world_size > 1:
                sync_model_across_ranks(actor_old, src_rank=0)
            actor_old.eval()
        if SAVE_INTERMEDIATE_CHECKPOINTS and it % CHECKPOINT_INTERVAL == 0:
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
    if SAVE_FINAL_CHECKPOINT:
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
                "curriculum_state": curriculum_manager.get_state_dict(),
            }
        )
        print_rank0(f"\nFinal checkpoint saved to /data/checkpoint_final.pt", rank)
    if rank == 0:
        print_rank0("\n=== Generating Sample Text ===", rank)
        print_rank0(f"Model trained for {TOTAL_ITERS} iterations through {curriculum_manager.current_stage + 1} curriculum stages", rank)
        print_rank0(f"Final perplexity: {curriculum_manager.current_perplexity:.2f}", rank)
        actor_eval = actor.module if hasattr(actor, 'module') else actor
        actor_eval.eval()
        sample_prompts = [
            "To be or not to be",
            "O Romeo, Romeo",
            "All the world's a stage",
            "What is in a name?",
            "The quality of mercy",
            "Now is the winter",
            "Friends, Romans, countrymen",
            "If music be the food",
        ]
        print_rank0("\nGenerating text samples with different prompts:", rank)
        print_rank0("=" * 80, rank)
        with torch.no_grad():
            for i, prompt in enumerate(sample_prompts):
                prompt_encoded = encode(prompt).unsqueeze(0).to(device)
                generated = actor_eval.generate(
                    prompt_encoded,
                    max_new_tokens=100,  
                    temperature=0.8,     
                    top_k=40,           
                    use_cache=True,    
                    repetition_penalty=2.0,  
                    repetition_window=128     
                )
                generated_text = decode(generated[0])
                print_rank0(f"\nSample {i+1}:", rank)
                print_rank0(f"Prompt: '{prompt}'", rank)
                print_rank0(f"Generated: {generated_text}", rank)
                print_rank0("-" * 40, rank)
                if i < 4:  
                    wandb.log({
                        f"samples/prompt_{i}": prompt,
                        f"samples/generated_{i}": generated_text,
                    })
        print_rank0("\nUnconditional generation (from empty prompt):", rank)
        print_rank0("=" * 80, rank)
        for i in range(3):
            start_token = encode("\n").unsqueeze(0).to(device)
            generated = actor_eval.generate(
                start_token,
                max_new_tokens=150,  
                temperature=0.8,
                top_k=40,
                use_cache=True,    
                repetition_penalty=2.0,  
                repetition_window=128     
            )
            generated_text = decode(generated[0])
            print_rank0(f"\nUnconditional Sample {i+1}:", rank)
            print_rank0(generated_text, rank)
            print_rank0("-" * 40, rank)
            wandb.log({
                f"samples/unconditional_{i}": generated_text,
            })
        print_rank0("\n" + "=" * 80, rank)
        print_rank0("Sample generation complete!", rank)
    if rank == 0:
        wandb.finish()
    cleanup_distributed()
if __name__ == "__main__":
    train_grpo()