"""Core training logic for avataRL without Modal dependencies."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import math
from functools import partial
from collections import deque, defaultdict
from tqdm import tqdm

# Import configuration
from config import *


class DistributedContextualTextLoader:
    """
    GPU-resident data loader for GRPO training
    - Data is loaded ONCE to GPU and stays there
    - Each GPU gets the full dataset but starts at different positions
    - Provides contextual windows for next-token prediction
    """
    def __init__(self, text: str, encode_fn, batch_size: int, horizon: int, 
                 context_len: int, rank: int, world_size: int, device: torch.device):
        # Load and encode data ONCE to GPU
        self.data = encode_fn(text).to(device)
        self.batch_size = batch_size
        self.horizon = horizon
        self.context_len = context_len
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        # Calculate starting position for this rank
        total_len = len(self.data)
        offset_per_rank = (total_len - context_len - batch_size * horizon) // max(1, world_size)
        self.pos = context_len + (rank * offset_per_rank)
        
        # Ensure we don't go past the end
        max_start = total_len - batch_size * horizon - 1
        self.pos = min(self.pos, max_start)
    
    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch - data stays on GPU"""
        # Wrap around if we reach the end
        if self.pos + self.batch_size * self.horizon + 1 > len(self.data):
            self.pos = self.context_len
        
        contexts = []
        targets = []
        
        for b in range(self.batch_size):
            ctx_start = self.pos - self.context_len + b * self.horizon
            ctx_end = self.pos + b * self.horizon
            
            # All operations on GPU-resident data
            context = self.data[ctx_start:ctx_end]
            target = self.data[ctx_end:ctx_end + self.horizon]
            
            contexts.append(context)
            targets.append(target)
        
        # Move position forward
        self.pos += self.batch_size * self.horizon
        
        # Stack on GPU
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
        """Forward pass with KV caching for efficient generation"""
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.transpose(1, 2).unbind(dim=2)
        cache_pos = self.kv_cache.size
        sin, cos = self.rope(cache_pos + T)
        sin = sin[:, :, cache_pos:]
        cos = cos[:, :, cache_pos:]
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        k_cache, v_cache = self.kv_cache.update(k, v)
        effective_T = cache_pos + T
        k_for_attn = k_cache[:, :, :effective_T]
        v_for_attn = v_cache[:, :, :effective_T]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k_for_attn.transpose(-2, -1)) * scale
        causal_mask = torch.triu(
            torch.ones(T, effective_T, device=x.device), 
            diagonal=cache_pos + 1 - T
        ).bool()
        attn = attn.masked_fill(causal_mask, -1e10)
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v_for_attn)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_out)


class MLP(nn.Module):
    """Optimized MLP with fused operations"""
    def __init__(self, n_emb: int, dropout: float = 0.1):
        super().__init__()
        self.gate_up_proj = nn.Linear(n_emb, 2 * 4 * n_emb, bias=False)
        self.down_proj = nn.Linear(4 * n_emb, n_emb, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = ReLUSquared()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.dropout(self.down_proj(self.act(gate) * up))


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization and parallel computation"""
    def __init__(self, n_emb: int, n_head: int):
        super().__init__()
        self.norm1 = nn.Identity()
        self.attn = OptimizedAttention(n_emb, n_head)
        self.norm2 = nn.Identity()
        self.mlp = MLP(n_emb)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, use_cache: bool = False) -> torch.Tensor:
        norm_x = norm(x)
        x = x + self.attn(norm_x, mask, use_cache=use_cache)
        norm_x = norm(x)
        x = x + self.mlp(norm_x)
        return x


class GPT(nn.Module):
    """GPT model with modern architecture"""
    def __init__(self):
        super().__init__()
        self.vocab_size = VOCAB_SIZE
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMB)
        self.blocks = nn.ModuleList([TransformerBlock(N_EMB, N_HEAD) for _ in range(N_LAYER)])
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(N_EMB, VOCAB_SIZE, bias=False)
        self.tok_emb.weight = self.lm_head.weight
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            if USE_EXHAUSTIVE:
                loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), reduction='none')
            else:
                loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
    
    def init_kv_caches(self, batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16, device: torch.device = None):
        """Initialize KV caches for all attention layers"""
        for block in self.blocks:
            block.attn.init_kv_cache(batch_size, max_seq_len, dtype, device)
    
    def reset_kv_caches(self):
        """Reset all KV caches"""
        for block in self.blocks:
            block.attn.reset_kv_cache()
    
    def disable_kv_caches(self):
        """Disable all KV caches (for training)"""
        for block in self.blocks:
            block.attn.disable_kv_cache()


def get_lr(it: int) -> float:
    """Learning rate schedule with warmup and decay"""
    if not USE_LR_DECAY:
        return LR
    if it < WARMUP_ITERS:
        return LR * it / WARMUP_ITERS
    if it > TOTAL_ITERS:
        return MIN_LR
    if LR_DECAY_TYPE == "cosine":
        decay_ratio = (it - WARMUP_ITERS) / (TOTAL_ITERS - WARMUP_ITERS)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return MIN_LR + coeff * (LR - MIN_LR)
    elif LR_DECAY_TYPE == "linear":
        decay_ratio = (it - WARMUP_ITERS) / (TOTAL_ITERS - WARMUP_ITERS)
        return LR - (LR - MIN_LR) * decay_ratio
    else:
        return LR


def get_kl_weight(sample_count: int) -> float:
    """KL divergence weighting schedule with free bits"""
    if sample_count < KL_WARM:
        kl_weight = BETA_KL * sample_count / KL_WARM
    else:
        kl_weight = BETA_KL
    return kl_weight


def prepare_tinyshakespeare() -> Tuple[str, Dict[str, int], Dict[int, str]]:
    """Load and prepare the TinyShakespeare dataset"""
    # Check Modal volume path first, then local
    data_paths = ["/data/input.txt", "input.txt", "tinyshakespeare.txt"]
    data_path = None
    
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        # Download if not found
        data_path = "input.txt"
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        urllib.request.urlretrieve(url, data_path)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    assert vocab_size == VOCAB_SIZE, f"Vocab size mismatch: {vocab_size} vs {VOCAB_SIZE}"
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return text, char_to_idx, idx_to_char


def compute_rewards(
    sequences: torch.Tensor,
    model: nn.Module,
    char_to_idx: Dict[str, int],
    device: torch.device
) -> torch.Tensor:
    """
    Enhanced reward function with multi-component scoring.
    Now includes balanced perplexity, repetition penalty, and coherence metrics.
    """
    B, L = sequences.shape
    rewards = torch.zeros(B, device=device)
    
    # Get model predictions for perplexity calculation
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(sequences[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Calculate perplexity-based reward
        targets = sequences[:, 1:].unsqueeze(-1)
        selected_log_probs = log_probs.gather(-1, targets).squeeze(-1)
        
        # Use mean log probability to avoid length bias
        mean_log_prob = selected_log_probs.mean(dim=1)
        perplexity_reward = torch.exp(mean_log_prob)  # Higher is better (closer to 1)
        
        # Normalize perplexity reward to reasonable range
        perplexity_reward = torch.clamp(perplexity_reward, 0.0, 1.0)
    
    # Character diversity penalty
    for i in range(B):
        seq = sequences[i].tolist()
        unique_chars = len(set(seq))
        diversity_score = unique_chars / len(seq)
        rewards[i] += diversity_score * 0.3  # Weight diversity component
    
    # Repetition penalty for consecutive duplicates
    for i in range(B):
        seq = sequences[i].tolist()
        consecutive_repeats = sum(1 for j in range(1, len(seq)) if seq[j] == seq[j-1])
        repeat_penalty = consecutive_repeats / len(seq)
        rewards[i] -= repeat_penalty * 0.5  # Penalize repetition
    
    # Add perplexity component
    rewards += perplexity_reward * 0.5
    
    # Ensure rewards are centered around 0 with reasonable variance
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    return rewards


def compute_confidence_scores(
    sequences: torch.Tensor,
    model: nn.Module,
    device: torch.device
) -> torch.Tensor:
    """Compute confidence scores based on model's prediction entropy"""
    B, L = sequences.shape
    confidence_scores = torch.zeros(B, device=device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(sequences[:, :-1])
        probs = F.softmax(logits, dim=-1)
        # Use negative entropy as confidence (low entropy = high confidence)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        # Average entropy across sequence, normalize
        avg_entropy = entropy.mean(dim=1)
        max_entropy = math.log(model.vocab_size)
        confidence_scores = 1.0 - (avg_entropy / max_entropy)
    
    return torch.clamp(confidence_scores, 0.1, 1.0)  # Avoid zero confidence


def train_step(
    model: nn.Module,
    ref_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    input_ids: torch.Tensor,
    sample_count: int,
    char_to_idx: Dict[str, int],
    device: torch.device,
    accumulation_step: int,
    total_accumulation_steps: int
) -> Dict[str, float]:
    """
    Single training step with GRPO algorithm.
    
    Args:
        model: Current policy model
        ref_model: Reference model for KL computation
        optimizer: Optimizer
        input_ids: Input token IDs [batch_size, context_len]
        sample_count: Total samples seen so far (for KL scheduling)
        char_to_idx: Character to index mapping
        device: Device to run on
        accumulation_step: Current gradient accumulation step
        total_accumulation_steps: Total gradient accumulation steps
    
    Returns:
        Dictionary of metrics
    """
    B, T = input_ids.shape
    
    # Generate K samples for each input
    all_sequences = []
    all_log_probs = []
    
    model.eval()  # Set to eval for generation
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(K_SAMPLES):
                sequences = input_ids.clone()
                log_probs = torch.zeros(B, HORIZON, device=device)
                
                for t in range(HORIZON):
                    if t > 0:
                        context = sequences[:, -(CONTEXT_LEN-1):]
                        new_token = generated_token.unsqueeze(1)
                        sequences = torch.cat([context, new_token], dim=1)
                    
                    logits, _ = model(sequences)
                    logits = logits[:, -1, :] / TEMPERATURE
                    probs = F.softmax(logits, dim=-1)
                    
                    generated_token = torch.multinomial(probs, 1).squeeze(-1)
                    token_log_probs = torch.log(probs.gather(-1, generated_token.unsqueeze(-1)) + 1e-10).squeeze(-1)
                    log_probs[:, t] = token_log_probs
                    
                    if t == HORIZON - 1:
                        final_sequences = torch.cat([
                            sequences, 
                            generated_token.unsqueeze(1)
                        ], dim=1)
                
                all_sequences.append(final_sequences)
                all_log_probs.append(log_probs)
    
    all_sequences = torch.stack(all_sequences)  # [K, B, T+H]
    all_log_probs = torch.stack(all_log_probs)  # [K, B, H]
    
    # Compute rewards for all samples
    rewards_list = []
    for k in range(K_SAMPLES):
        rewards = compute_rewards(all_sequences[k], model, char_to_idx, device)
        rewards_list.append(rewards)
    all_rewards = torch.stack(rewards_list)  # [K, B]
    
    # Compute baseline as mean across samples for each sequence
    baseline = all_rewards.mean(dim=0, keepdim=True)  # [1, B]
    
    # Compute advantages
    advantages = all_rewards - baseline  # [K, B]
    
    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std
    
    # Compute confidence scores if enabled
    if USE_CONFIDENCE_SCALING:
        confidence_scores = []
        for k in range(K_SAMPLES):
            scores = compute_confidence_scores(all_sequences[k], model, device)
            confidence_scores.append(scores)
        confidence_scores = torch.stack(confidence_scores)  # [K, B]
        
        # Scale advantages by confidence
        scaled_advantages = advantages * torch.pow(confidence_scores, CONFIDENCE_WEIGHT)
        scaled_advantages = torch.clamp(scaled_advantages, -CONFIDENCE_CLIP, CONFIDENCE_CLIP)
    else:
        scaled_advantages = advantages
    
    # Select sequences based on advantages
    if USE_EXHAUSTIVE:
        # Use all sequences with importance weighting
        selected_mask = torch.ones_like(scaled_advantages, dtype=torch.bool)
        importance_weights = F.softmax(scaled_advantages.flatten(), dim=0).reshape(K_SAMPLES, B)
    else:
        # Select top sequences based on advantages
        k_select = max(1, K_SAMPLES // 4)
        _, top_indices = scaled_advantages.topk(k_select, dim=0)
        selected_mask = torch.zeros_like(scaled_advantages, dtype=torch.bool)
        selected_mask.scatter_(0, top_indices, True)
        importance_weights = None
    
    # Now compute losses for selected sequences using BATCHED operations
    model.train()
    kl_weight = get_kl_weight(sample_count)
    
    # Flatten sequences for batched processing
    # Instead of K*B individual forward passes, do one batched pass
    flat_sequences = all_sequences.reshape(-1, all_sequences.shape[-1])  # [K*B, T+H]
    flat_log_probs = all_log_probs.reshape(-1, HORIZON)  # [K*B, H]
    flat_advantages = scaled_advantages.reshape(-1)  # [K*B]
    flat_mask = selected_mask.reshape(-1)  # [K*B]
    
    # Get indices of selected sequences
    selected_indices = torch.where(flat_mask)[0]
    num_selected = len(selected_indices)
    
    if num_selected == 0:
        # No sequences selected, return zero metrics
        return {
            'loss': 0.0,
            'pg_loss': 0.0,
            'kl_loss': 0.0,
            'entropy_loss': 0.0,
            'reward_mean': all_rewards.mean().item(),
            'reward_std': all_rewards.std().item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'num_selected': 0,
            'kl_weight': kl_weight,
        }
    
    # Extract selected sequences for batched processing
    selected_sequences = flat_sequences[selected_indices]  # [num_selected, T+H]
    selected_old_log_probs = flat_log_probs[selected_indices]  # [num_selected, H]
    selected_advantages = flat_advantages[selected_indices]  # [num_selected]
    
    # Prepare inputs and targets for batched forward pass
    contexts = selected_sequences[:, :T]  # [num_selected, T]
    generated = selected_sequences[:, T:T+HORIZON]  # [num_selected, H]
    
    # Construct full inputs and targets
    full_inputs = torch.cat([contexts[:, :-1], generated], dim=1)  # [num_selected, T-1+H]
    full_targets = torch.cat([contexts[:, 1:], generated], dim=1)  # [num_selected, T-1+H]
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        # Single batched forward pass through current model
        logits, _ = model(full_inputs, full_targets)
        
        # Extract logits for generated tokens
        gen_start_idx = T - 1  # Where generated tokens start in the output
        gen_logits = logits[:, gen_start_idx:gen_start_idx+HORIZON]  # [num_selected, H, vocab_size]
        
        # Compute log probs
        gen_log_probs = F.log_softmax(gen_logits, dim=-1)  # [num_selected, H, vocab_size]
        
        # Gather log probs for actually generated tokens
        generated_expanded = generated.unsqueeze(-1)  # [num_selected, H, 1]
        selected_log_probs = gen_log_probs.gather(-1, generated_expanded).squeeze(-1)  # [num_selected, H]
        
        # PPO-style clipping
        ratio = torch.exp(selected_log_probs - selected_old_log_probs)  # [num_selected, H]
        clip_ratio = torch.clamp(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO)
        
        # Expand advantages for broadcasting
        advantages_expanded = selected_advantages.unsqueeze(-1)  # [num_selected, 1]
        
        # Policy gradient loss
        pg_loss_per_token = -torch.min(
            ratio * advantages_expanded,
            clip_ratio * advantages_expanded
        )  # [num_selected, H]
        pg_loss = pg_loss_per_token.mean()
        
        # KL divergence with reference model (batched)
        with torch.no_grad():
            ref_logits, _ = ref_model(full_inputs)
            ref_gen_logits = ref_logits[:, gen_start_idx:gen_start_idx+HORIZON]
            ref_log_probs = F.log_softmax(ref_gen_logits, dim=-1)
        
        # KL divergence per sample
        # Note: F.kl_div expects log probabilities for input and probabilities for target
        kl_div = F.kl_div(
            gen_log_probs,  # [num_selected, H, vocab_size] - log probs
            ref_log_probs.exp(),  # [num_selected, H, vocab_size] - probs
            reduction='none'
        ).sum(dim=-1).mean(dim=-1)  # Sum over vocab, mean over horizon -> [num_selected]
        
        # Apply KL free bits
        kl_loss = torch.maximum(kl_div - KL_FREE_FRACTION, torch.tensor(0.0, device=device)).mean()
        
        # Entropy regularization
        entropy = -(gen_log_probs.exp() * gen_log_probs).sum(dim=-1).mean()
        entropy_loss = -ENTROPY_COEF * entropy
        
        # Apply importance weights if using exhaustive sampling
        if importance_weights is not None:
            # Map back to original K, B indices
            importance_weights_flat = importance_weights.reshape(-1)[selected_indices]
            total_loss = (importance_weights_flat * (pg_loss + kl_weight * kl_loss + entropy_loss)).mean()
        else:
            total_loss = pg_loss + kl_weight * kl_loss + entropy_loss
        
        # Scale by accumulation steps
        total_loss = total_loss / total_accumulation_steps
        
        # Single backward pass for all selected sequences
        scaler.scale(total_loss).backward()
    
    # Only step optimizer on last accumulation step
    if accumulation_step == total_accumulation_steps - 1:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    metrics = {
        'loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        'pg_loss': pg_loss.item() if isinstance(pg_loss, torch.Tensor) else pg_loss,
        'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
        'entropy_loss': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
        'reward_mean': all_rewards.mean().item(),
        'reward_std': all_rewards.std().item(),
        'advantage_mean': advantages.mean().item(),
        'advantage_std': advantages.std().item(),
        'num_selected': num_selected,
        'kl_weight': kl_weight,
    }
    
    if USE_CONFIDENCE_SCALING:
        metrics['confidence_mean'] = confidence_scores.mean().item()
        metrics['confidence_std'] = confidence_scores.std().item()
    
    return metrics


def evaluate_model(
    model: nn.Module,
    text: str,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    device: torch.device,
    eval_iters: int = 50
) -> Dict[str, float]:
    """Evaluate model perplexity and generate samples"""
    model.eval()
    losses = []
    
    # Calculate perplexity on random sequences
    with torch.no_grad():
        for _ in range(eval_iters):
            # Get random slice of data
            start_idx = torch.randint(0, len(text) - CONTEXT_LEN - 1, (1,)).item()
            chunk = text[start_idx:start_idx + CONTEXT_LEN + 1]
            indices = [char_to_idx[ch] for ch in chunk]
            x = torch.tensor(indices[:-1], device=device).unsqueeze(0)
            y = torch.tensor(indices[1:], device=device).unsqueeze(0)
            
            _, loss = model(x, y)
            losses.append(loss.mean().item())
    
    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    
    # Generate a sample
    context = "KING HENRY VI:"
    context_indices = [char_to_idx[ch] for ch in context]
    x = torch.tensor(context_indices, device=device).unsqueeze(0)
    
    model.init_kv_caches(batch_size=1, max_seq_len=len(context) + 100, device=device)
    
    generated_indices = []
    for _ in range(100):
        with torch.no_grad():
            logits, _ = model(x, use_cache=True)
            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            generated_indices.append(next_idx.item())
            x = next_idx
    
    model.reset_kv_caches()
    model.disable_kv_caches()
    
    generated_text = ''.join([idx_to_char[idx] for idx in generated_indices])
    full_text = context + generated_text
    
    return {
        'perplexity': perplexity,
        'loss': avg_loss,
        'sample': full_text
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    metrics: Dict[str, float],
    checkpoint_dir: str = "checkpoints"
):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'n_layer': N_LAYER,
            'n_head': N_HEAD,
            'n_emb': N_EMB,
            'vocab_size': VOCAB_SIZE,
            'context_len': CONTEXT_LEN,
        }
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def main(rank: int = 0, world_size: int = 1):
    """Main training function"""
    # Setup distributed training if needed
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.95)
    
    # Load dataset
    text, char_to_idx, idx_to_char = prepare_tinyshakespeare()
    
    # Create encode function for the data loader
    encode = lambda s: torch.tensor([char_to_idx[ch] for ch in s], dtype=torch.long)
    
    # Initialize GPU-resident data loader
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
    
    # Initialize models
    model = GPT().to(device)
    ref_model = GPT().to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Setup DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[rank], bucket_cap_mb=BUCKET_SIZE_MB)
        model.register_comm_hook(dist.group.WORLD, comm_hooks.bf16_compress_hook)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    
    # Setup gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    # Training loop
    model.train()
    sample_count = 0
    recent_losses = deque(maxlen=100)
    metrics = {}  # Initialize metrics dict for checkpointing
    
    # Initialize wandb on rank 0
    if rank == 0:
        import wandb
        wandb.init(
            project="avataRL_12",
            name="perplexity_fix_2",
            config={
                "model": {
                    "vocab_size": VOCAB_SIZE,
                    "n_layer": N_LAYER,
                    "n_head": N_HEAD,
                    "n_emb": N_EMB,
                    "context_len": CONTEXT_LEN,
                    "param_count": sum(p.numel() for p in model.parameters()),
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
        print(f"Starting training with {world_size} GPUs...", flush=True)
        print(f"Total iterations: {TOTAL_ITERS}", flush=True)
        print(f"Gradient accumulation steps: {GRAD_ACCUM}", flush=True)
    
    chars_seen = 0
    
    # Use tqdm for progress tracking on rank 0
    for it in tqdm(range(1, TOTAL_ITERS + 1), desc="Training", disable=(rank != 0)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        # Prepare batch across accumulation steps
        for accum_step in range(GRAD_ACCUM):
            # Get batch from GPU-resident data loader
            contexts, _ = loader.next()  # targets not needed for GRPO
            
            # Update learning rate
            lr = get_lr(it)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Training step with contexts as input
            metrics = train_step(
                model.module if world_size > 1 else model,
                ref_model,
                optimizer,
                scaler,
                contexts,  # Already on GPU
                sample_count,
                char_to_idx,
                device,
                accum_step,
                GRAD_ACCUM
            )
            
            sample_count += contexts.shape[0] * world_size
            chars_seen += contexts.shape[0] * CONTEXT_LEN * world_size
            recent_losses.append(metrics['loss'])
        
        # Log metrics every iteration like the bible
        if rank == 0:
            print(f"Iter {it}: loss={metrics['loss']:.4f}, acc={metrics.get('accuracy', 0):.3f}, "
                  f"kl={metrics.get('kl_loss', 0):.4f}, gpt2_kl={metrics.get('gpt2_kl', 0):.4f}, "
                  f"stage=0, chars={chars_seen:,}", flush=True)
        
        # WandB logging
        if rank == 0 and it % 2 == 0:
            wandb.log({
                "iteration": it,
                "loss/total": metrics['loss'],
                "loss/policy": metrics['pg_loss'],
                "loss/kl": metrics['kl_loss'],
                "loss/entropy": metrics['entropy_loss'],
                "rewards/mean": metrics['reward_mean'],
                "rewards/std": metrics['reward_std'],
                "advantages/mean": metrics['advantage_mean'],
                "advantages/std": metrics['advantage_std'],
                "training/lr": lr,
                "training/kl_weight": metrics['kl_weight'],
                "training/num_selected": metrics['num_selected'],
                "training/chars_seen": chars_seen,
            })
            
            if USE_CONFIDENCE_SCALING:
                wandb.log({
                    "confidence/mean": metrics['confidence_mean'],
                    "confidence/std": metrics['confidence_std'],
                })
        
        # Evaluation
        if rank == 0 and it % 100 == 0:
            eval_metrics = evaluate_model(
                model.module if world_size > 1 else model,
                text, char_to_idx, idx_to_char, device
            )
            print(f"\n[Eval] Perplexity: {eval_metrics['perplexity']:.2f}")
            print(f"Sample: {eval_metrics['sample'][:200]}...\n")
        
        # Checkpointing
        if rank == 0 and SAVE_INTERMEDIATE_CHECKPOINTS and it % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                model.module if world_size > 1 else model,
                optimizer, it, metrics
            )
    
    # Final checkpoint
    if rank == 0 and SAVE_FINAL_CHECKPOINT:
        save_checkpoint(
            model.module if world_size > 1 else model,
            optimizer, it, metrics, "checkpoints/final"
        )
    
    # Finish wandb run
    if rank == 0:
        wandb.finish()
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    
    main(args.rank, args.world_size)