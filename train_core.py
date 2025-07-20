"""Core training logic for avataRL without Modal dependencies."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import math
from functools import partial
from collections import deque, defaultdict
from tqdm import tqdm

# Import configuration
from config import *
import numpy as np

# Global tokenizer cache
_tokenizer = None


def check_tokenizer_exists(vocab_size: int = VOCAB_SIZE) -> bool:
    """Check if the BPE tokenizer exists and provide instructions if not."""
    # Check Modal volume path first, then local paths
    tokenizer_paths = [
        f"/data/{TOKENIZER_DIR}/shakespeare-bpe-{vocab_size}.json",
        os.path.join(DATA_ROOT, TOKENIZER_DIR, f"shakespeare-bpe-{vocab_size}.json")
    ]
    
    for path in tokenizer_paths:
        if os.path.exists(path):
            return True
    
    print(f"\n{'='*60}")
    print(f"ERROR: BPE tokenizer not found!")
    print(f"{'='*60}")
    print(f"Expected tokenizer at one of:")
    for path in tokenizer_paths:
        print(f"  - {path}")
    print(f"\nTo create the tokenizer:")
    print(f"  1. Install tokenizers: pip install tokenizers")
    print(f"  2. Run: python create_tokenizer_example.py --vocab-size {vocab_size}")
    print(f"\nOr if using Modal:")
    print(f"  modal run train_tokenizer_modal.py::train_bpe_tokenizer")
    print(f"{'='*60}\n")
    return False


def load_bpe_tokenizer(vocab_size: int = VOCAB_SIZE):
    """Load BPE tokenizer from Modal volume or local path."""
    global _tokenizer
    
    if _tokenizer is not None:
        return _tokenizer
    
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise ImportError(
            "tokenizers library not installed. "
            "Install with: pip install tokenizers"
        )
    
    # Check Modal volume path first, then local paths
    tokenizer_paths = [
        f"/data/{TOKENIZER_DIR}/shakespeare-bpe-{vocab_size}.json",
        os.path.join(DATA_ROOT, TOKENIZER_DIR, f"shakespeare-bpe-{vocab_size}.json")
    ]
    
    tokenizer_path = None
    for path in tokenizer_paths:
        if os.path.exists(path):
            tokenizer_path = path
            break
    
    if tokenizer_path is None:
        if not check_tokenizer_exists(vocab_size):
            raise FileNotFoundError(
                f"BPE tokenizer not found for vocab_size={vocab_size}"
            )
    
    _tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Loaded BPE tokenizer from {tokenizer_path}")
    print(f"Vocabulary size: {_tokenizer.get_vocab_size()}")
    
    return _tokenizer


def load_pretokenized_data(vocab_size: int = VOCAB_SIZE, split: str = "train") -> np.ndarray:
    """Load pre-tokenized binary data from Modal volume or local path.
    
    Args:
        vocab_size: BPE vocabulary size
        split: 'train' or 'val'
    
    Returns:
        Memory-mapped numpy array of token IDs (uint16 format)
    """
    # Format the tokenized data directory name
    data_dir_name = TOKENIZED_DATA_DIR.format(vocab_size=vocab_size)
    
    # Check Modal volume path first, then local paths
    data_paths = [
        f"/data/{data_dir_name}/{split}.bin",
        os.path.join(DATA_ROOT, data_dir_name, f"{split}.bin")
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print(f"\n{'='*60}")
        print(f"ERROR: Pre-tokenized data not found!")
        print(f"{'='*60}")
        print(f"Expected {split}.bin at one of:")
        for path in data_paths:
            print(f"  - {path}")
        print(f"\nTo create tokenized data:")
        print(f"  1. Ensure you have the BPE tokenizer (vocab_size={vocab_size})")
        print(f"  2. Run tokenization script or use Modal:")
        print(f"     modal run prepare_data.py")
        print(f"{'='*60}\n")
        raise FileNotFoundError(
            f"Pre-tokenized {split} data not found for vocab_size={vocab_size}"
        )
    
    # Load as memory-mapped array for efficiency
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    print(f"Loaded {split} data from {data_path}")
    print(f"Total tokens: {len(data):,}")
    
    return data


class TokenizedDataLoader:
    """
    Memory-mapped loader for pre-tokenized binary data.
    
    This loader uses numpy memory mapping for efficient data access without
    loading the entire dataset into memory. It handles distributed training
    by partitioning data across ranks and provides contextual windows for
    GRPO training.
    """
    def __init__(self, data_path: str, batch_size: int, context_len: int,
                 horizon: int, rank: int, world_size: int, device: torch.device):
        # Load data as memory-mapped array
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.batch_size = batch_size
        self.context_len = context_len
        self.horizon = horizon
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        # Calculate starting position for this rank
        total_len = len(self.data)
        self.max_start_pos = total_len - context_len - batch_size * horizon - 1
        
        if self.max_start_pos <= 0:
            raise ValueError(
                f"Dataset too small ({total_len} tokens) for context_len={context_len}, "
                f"batch_size={batch_size}, horizon={horizon}"
            )
        
        # Distribute starting positions across ranks
        positions_per_rank = max(1, self.max_start_pos // max(1, world_size))
        self.pos = context_len + (rank * positions_per_rank) % self.max_start_pos
        
        print(f"TokenizedDataLoader initialized for rank {rank}/{world_size}")
        print(f"Data size: {total_len:,} tokens, starting position: {self.pos}")
    
    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of contexts and targets."""
        # Wrap around if we reach the end
        if self.pos + self.batch_size * self.horizon + 1 > len(self.data):
            self.pos = self.context_len
        
        contexts = []
        targets = []
        
        for b in range(self.batch_size):
            ctx_start = self.pos - self.context_len + b * self.horizon
            ctx_end = self.pos + b * self.horizon
            
            # Load from memory-mapped array and convert to tensor
            context = self.data[ctx_start:ctx_end].astype(np.int64)
            target = self.data[ctx_end:ctx_end + self.horizon].astype(np.int64)
            
            contexts.append(torch.from_numpy(context))
            targets.append(torch.from_numpy(target))
        
        # Move position forward
        self.pos += self.batch_size * self.horizon
        
        # Stack and move to device
        contexts = torch.stack(contexts).to(self.device)
        targets = torch.stack(targets).to(self.device)
        
        return contexts, targets
    
    def state_dict(self) -> Dict:
        """Save loader state for checkpointing."""
        return {
            'pos': self.pos,
            'rank': self.rank
        }
    
    def load_state_dict(self, state: Dict):
        """Restore loader state from checkpoint."""
        self.pos = state['pos']
        if state['rank'] != self.rank:
            print(f"Warning: Loading state from rank {state['rank']} to rank {self.rank}")


class DistributedContextualTextLoader:
    """
    GPU-resident data loader for GRPO training
    - Data is loaded ONCE to GPU and stays there
    - Each GPU gets the full dataset but starts at different positions
    - Provides contextual windows for next-token prediction
    """
    def __init__(self, data: Union[str, np.ndarray], encode_fn=None, batch_size: int = None, 
                 horizon: int = None, context_len: int = None, rank: int = 0, 
                 world_size: int = 1, device: torch.device = None):
        # Handle both text and pre-tokenized data
        if isinstance(data, np.ndarray):
            # Pre-tokenized path: convert to tensor and load to GPU
            self.data = torch.from_numpy(data.astype(np.int64)).to(device)
        else:
            # Legacy text path (for compatibility)
            if encode_fn is None:
                raise ValueError("encode_fn must be provided when data is text")
            self.data = encode_fn(data).to(device)
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


class CurriculumManager:
    """
    Manages curriculum learning with staged training based on perplexity tracking.
    Progressively increases task complexity through 5 stages.
    """
    def __init__(self, device: torch.device, vocab_size: int = VOCAB_SIZE,
                 ema_alpha: float = 0.95,
                 promotion_threshold: float = 0.05,
                 promotion_patience: int = 100,
                 min_iterations_per_stage: int = 500,
                 total_iterations: Optional[int] = None):
        self.device = device
        self.vocab_size = vocab_size
        self.current_stage = 0
        self.stage_iterations = 0
        self.total_iterations_so_far = 0
        
        # EMA tracking
        self.ema_alpha = ema_alpha
        self.ema_perplexity = None
        self.ema_accuracy = None
        self.ema_reward = None
        
        # Promotion criteria
        self.promotion_threshold = promotion_threshold
        self.promotion_patience = promotion_patience
        self.min_iterations_per_stage = min_iterations_per_stage
        self.patience_counter = 0
        self.best_ema_perplexity = float('inf')
        
        # History tracking
        self.perplexity_history = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
        # Stage configurations
        self.stages = self._setup_stages()
        
        # Adaptive timing
        if total_iterations:
            self.min_iterations_per_stage = max(int(total_iterations * 0.05), 100)
            self.promotion_patience = max(int(total_iterations * 0.1 / 5), 50)
        
        # Word boundaries for word detection
        self.word_boundaries = {' ', '\n', '.', ',', '!', '?', ';', ':', '"', "'"}
        
        # Stage-specific KL weights
        self.kl_weights = {
            0: 0.03,
            1: 0.05,
            2: 0.08,
            3: 0.1,
            4: 0.1
        }
        
        # Repetition penalty weights by stage
        self.repetition_weights = {
            0: 0.0,
            1: 0.3,
            2: 0.5,
            3: 0.7,
            4: 0.9
        }
    
    def _setup_stages(self) -> Dict:
        """Define curriculum stages with specific objectives."""
        return {
            0: {
                'name': 'Bigram Foundation',
                'description': 'Learn token transitions',
                'word_weights': [0.0, 0.0, 0.0, 0.0, 0.0],
                'ngram_weight': 1.0,
                'perplexity_threshold': 0.2,
                'min_iterations': 500
            },
            1: {
                'name': '2-3 Letter Words',
                'description': 'Form simple words',
                'word_weights': [0.1, 0.27, 0.0, 0.0, 0.0],
                'ngram_weight': 0.7,
                'perplexity_threshold': 0.15,
                'min_iterations': 700
            },
            2: {
                'name': '2-4 Letter Words',
                'description': 'Expand vocabulary',
                'word_weights': [0.06, 0.16, 0.28, 0.0, 0.0],
                'ngram_weight': 0.5,
                'perplexity_threshold': 0.1,
                'min_iterations': 1000
            },
            3: {
                'name': 'Full Word Spectrum',
                'description': 'Prioritize word formation',
                'word_weights': [0.15, 0.41, 0.73, 0.21, 0.0],
                'ngram_weight': 0.3,
                'perplexity_threshold': 0.08,
                'min_iterations': 1500
            },
            4: {
                'name': 'Full Optimization',
                'description': 'Complete words emphasis',
                'word_weights': [0.1, 0.27, 0.48, 1.35, 3.5],
                'ngram_weight': 0.2,
                'perplexity_threshold': 0.05,
                'min_iterations': 2000
            }
        }
    
    def calculate_perplexity(self, loss: float) -> float:
        """Calculate perplexity from loss with overflow protection."""
        capped_loss = min(loss, 20.0)
        return math.exp(capped_loss)
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update EMA metrics and check for stage promotion."""
        self.stage_iterations += 1
        self.total_iterations_so_far += 1
        
        # Calculate perplexity
        perplexity = self.calculate_perplexity(metrics.get('cross_entropy', metrics['loss']))
        accuracy = metrics.get('accuracy', 0.0)
        reward = metrics.get('reward_mean', 0.0)
        
        # Initialize or update EMAs
        if self.ema_perplexity is None:
            self.ema_perplexity = perplexity
            self.ema_accuracy = accuracy
            self.ema_reward = reward
        else:
            self.ema_perplexity = self.ema_alpha * self.ema_perplexity + (1 - self.ema_alpha) * perplexity
            self.ema_accuracy = self.ema_alpha * self.ema_accuracy + (1 - self.ema_alpha) * accuracy
            self.ema_reward = self.ema_alpha * self.ema_reward + (1 - self.ema_alpha) * reward
        
        # Track history
        self.perplexity_history.append(perplexity)
        self.accuracy_history.append(accuracy)
        self.reward_history.append(reward)
        
        # Check promotion
        if self._should_promote():
            self._promote_stage()
    
    def _should_promote(self) -> bool:
        """Check if criteria met for stage promotion."""
        if self.current_stage >= 4:
            return False
        
        if self.stage_iterations < self.min_iterations_per_stage:
            return False
        
        stage_config = self.stages[self.current_stage]
        
        # Check perplexity improvement
        if self.best_ema_perplexity == float('inf'):
            self.best_ema_perplexity = self.ema_perplexity
        
        improvement = (self.best_ema_perplexity - self.ema_perplexity) / self.best_ema_perplexity
        
        if improvement >= stage_config['perplexity_threshold']:
            return True
        
        # Check EMA trigger (5% improvement)
        if len(self.perplexity_history) >= 50:
            old_perplexity = sum(list(self.perplexity_history)[:10]) / 10
            new_perplexity = sum(list(self.perplexity_history)[-10:]) / 10
            if (old_perplexity - new_perplexity) / old_perplexity >= 0.05:
                return True
        
        # Check plateau
        if self.ema_perplexity < self.best_ema_perplexity:
            self.best_ema_perplexity = self.ema_perplexity
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.promotion_patience
    
    def _promote_stage(self):
        """Promote to next curriculum stage."""
        self.current_stage = min(self.current_stage + 1, 4)
        self.stage_iterations = 0
        self.patience_counter = 0
        self.best_ema_perplexity = self.ema_perplexity
        
        print(f"\n🎓 Promoted to Stage {self.current_stage}: {self.stages[self.current_stage]['name']}")
        print(f"   EMA Perplexity: {self.ema_perplexity:.4f}")
        print(f"   Total iterations: {self.total_iterations_so_far}\n")
    
    def get_stage_config(self) -> Dict:
        """Get current stage configuration."""
        return self.stages[self.current_stage]
    
    def get_kl_weight(self) -> float:
        """Get KL weight for current stage."""
        return self.kl_weights[self.current_stage]
    
    def get_repetition_weight(self) -> float:
        """Get repetition penalty weight for current stage."""
        return self.repetition_weights[self.current_stage]
    
    def get_state_dict(self) -> Dict:
        """Save curriculum state for checkpointing."""
        return {
            'current_stage': self.current_stage,
            'stage_iterations': self.stage_iterations,
            'total_iterations_so_far': self.total_iterations_so_far,
            'ema_perplexity': self.ema_perplexity,
            'ema_accuracy': self.ema_accuracy,
            'ema_reward': self.ema_reward,
            'best_ema_perplexity': self.best_ema_perplexity,
            'patience_counter': self.patience_counter,
            'perplexity_history': list(self.perplexity_history),
            'accuracy_history': list(self.accuracy_history),
            'reward_history': list(self.reward_history)
        }
    
    def load_state_dict(self, state: Dict):
        """Restore curriculum state from checkpoint."""
        self.current_stage = state['current_stage']
        self.stage_iterations = state['stage_iterations']
        self.total_iterations_so_far = state['total_iterations_so_far']
        self.ema_perplexity = state['ema_perplexity']
        self.ema_accuracy = state['ema_accuracy']
        self.ema_reward = state['ema_reward']
        self.best_ema_perplexity = state['best_ema_perplexity']
        self.patience_counter = state['patience_counter']
        self.perplexity_history = deque(state['perplexity_history'], maxlen=100)
        self.accuracy_history = deque(state['accuracy_history'], maxlen=100)
        self.reward_history = deque(state['reward_history'], maxlen=100)


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
    def __init__(self, vocab_size: Optional[int] = None):
        super().__init__()
        # Use provided vocab_size or fall back to config
        self.vocab_size = vocab_size if vocab_size is not None else VOCAB_SIZE
        self.tok_emb = nn.Embedding(self.vocab_size, N_EMB)
        self.blocks = nn.ModuleList([TransformerBlock(N_EMB, N_HEAD) for _ in range(N_LAYER)])
        self.final_norm = nn.Identity()
        self.lm_head = nn.Linear(N_EMB, self.vocab_size, bias=False)
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


def compute_rewards(
    sequences: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    curriculum_manager: Optional[CurriculumManager] = None
) -> torch.Tensor:
    """
    Enhanced reward function with multi-component scoring.
    Now includes balanced perplexity, repetition penalty, and coherence metrics.
    Optionally uses curriculum learning for staged rewards.
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
    
    # Token diversity penalty
    for i in range(B):
        seq = sequences[i].tolist()
        unique_tokens = len(set(seq))
        diversity_score = unique_tokens / len(seq)
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
    device: torch.device,
    accumulation_step: int,
    total_accumulation_steps: int,
    curriculum_manager: Optional[CurriculumManager] = None
) -> Dict[str, float]:
    """
    Single training step with GRPO algorithm.
    
    Args:
        model: Current policy model
        ref_model: Reference model for KL computation
        optimizer: Optimizer
        input_ids: Input token IDs [batch_size, context_len]
        sample_count: Total samples seen so far (for KL scheduling)
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
        rewards = compute_rewards(all_sequences[k], model, device, curriculum_manager)
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
    # Use curriculum KL weight if available, otherwise use global KL weight
    if curriculum_manager is not None:
        kl_weight = curriculum_manager.get_kl_weight()
    else:
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
    
    # Compute cross-entropy loss for curriculum tracking
    with torch.no_grad():
        # Use the full targets for cross-entropy computation
        ce_loss = F.cross_entropy(gen_logits.reshape(-1, gen_logits.size(-1)), 
                                  generated.reshape(-1), reduction='mean')
    
    metrics = {
        'loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        'pg_loss': pg_loss.item() if isinstance(pg_loss, torch.Tensor) else pg_loss,
        'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
        'entropy_loss': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
        'cross_entropy': ce_loss.item(),  # Add cross-entropy for perplexity tracking
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
    val_data: np.ndarray,
    tokenizer,
    device: torch.device,
    eval_iters: int = 50
) -> Dict[str, float]:
    """Evaluate model perplexity and generate samples"""
    model.eval()
    losses = []
    
    # Calculate perplexity on random sequences
    with torch.no_grad():
        for _ in range(eval_iters):
            # Get random slice of validation data
            start_idx = torch.randint(0, len(val_data) - CONTEXT_LEN - 1, (1,)).item()
            indices = val_data[start_idx:start_idx + CONTEXT_LEN + 1]
            x = torch.tensor(indices[:-1], device=device, dtype=torch.long).unsqueeze(0)
            y = torch.tensor(indices[1:], device=device, dtype=torch.long).unsqueeze(0)
            
            _, loss = model(x, y)
            losses.append(loss.mean().item())
    
    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    
    # Generate a sample
    context = "First Citizen:\nBefore we proceed any further, hear me speak."
    context_tokens = tokenizer.encode(context).ids
    x = torch.tensor(context_tokens, device=device, dtype=torch.long).unsqueeze(0)
    
    model.init_kv_caches(batch_size=1, max_seq_len=len(context_tokens) + 100, device=device)
    
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
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_indices)
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
    checkpoint_dir: str = "checkpoints",
    scaler: Optional[torch.amp.GradScaler] = None,
    curriculum_manager: Optional[CurriculumManager] = None
):
    """Save model checkpoint with all training state"""
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
    
    # Add scaler state if available
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Add curriculum state if available
    if curriculum_manager is not None:
        checkpoint['curriculum_state'] = curriculum_manager.get_state_dict()
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    curriculum_manager: Optional[CurriculumManager] = None,
    strict: bool = True
) -> Dict:
    """Load model checkpoint and validate configuration.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scaler: Optional gradient scaler to load state into
        curriculum_manager: Optional curriculum manager to load state into
        strict: Whether to strictly enforce matching keys in state_dict
    
    Returns:
        Dictionary containing checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Validate configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        
        # Check vocab_size compatibility
        if 'vocab_size' in config:
            checkpoint_vocab_size = config['vocab_size']
            model_vocab_size = model.vocab_size if hasattr(model, 'vocab_size') else VOCAB_SIZE
            
            if checkpoint_vocab_size != model_vocab_size:
                raise ValueError(
                    f"Vocab size mismatch: checkpoint has vocab_size={checkpoint_vocab_size}, "
                    f"but model expects vocab_size={model_vocab_size}. "
                    f"Please ensure you're using the correct tokenizer and model configuration."
                )
        
        # Validate other critical parameters
        if config.get('n_layer') != N_LAYER:
            print(f"Warning: Layer count mismatch - checkpoint: {config.get('n_layer')}, config: {N_LAYER}")
        if config.get('n_head') != N_HEAD:
            print(f"Warning: Head count mismatch - checkpoint: {config.get('n_head')}, config: {N_HEAD}")
        if config.get('n_emb') != N_EMB:
            print(f"Warning: Embedding size mismatch - checkpoint: {config.get('n_emb')}, config: {N_EMB}")
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scaler state if provided
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Load curriculum state if provided
    if curriculum_manager is not None and 'curriculum_state' in checkpoint:
        curriculum_manager.load_state_dict(checkpoint['curriculum_state'])
    
    print(f"Successfully loaded checkpoint from iteration {checkpoint.get('iteration', 'unknown')}")
    
    return {
        'iteration': checkpoint.get('iteration', 0),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {})
    }


def main(rank: int = 0, world_size: int = 1):
    """Main training function"""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
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
    
    # Load pre-tokenized data
    train_data = load_pretokenized_data(VOCAB_SIZE, "train")
    val_data = load_pretokenized_data(VOCAB_SIZE, "val")
    
    # Load BPE tokenizer for decoding
    tokenizer = load_bpe_tokenizer(VOCAB_SIZE)
    
    # Initialize GPU-resident data loader with pre-tokenized data
    loader = DistributedContextualTextLoader(
        data=train_data,  # Pass pre-tokenized numpy array directly
        encode_fn=None,   # Not needed for pre-tokenized data
        batch_size=MICRO_BATCH,
        horizon=HORIZON,
        context_len=CONTEXT_LEN,
        rank=rank,
        world_size=world_size,
        device=device
    )
    
    # Initialize models with BPE vocab size
    model = GPT(vocab_size=VOCAB_SIZE).to(device)
    ref_model = GPT(vocab_size=VOCAB_SIZE).to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Setup DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[rank], bucket_cap_mb=BUCKET_SIZE_MB)
        model.register_comm_hook(dist.group.WORLD, comm_hooks.bf16_compress_hook)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    
    # Setup gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    # Initialize curriculum manager with BPE vocab size
    curriculum_manager = CurriculumManager(
        device=device,
        vocab_size=VOCAB_SIZE,
        total_iterations=TOTAL_ITERS,
        promotion_threshold=0.05,
        ema_alpha=0.95
    )
    
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
                    "vocab_size": VOCAB_SIZE,  # Use actual vocab size
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
    
    tokens_seen = 0
    
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
                device,
                accum_step,
                GRAD_ACCUM,
                curriculum_manager
            )
            
            sample_count += contexts.shape[0] * world_size
            tokens_seen += contexts.shape[0] * CONTEXT_LEN * world_size
            recent_losses.append(metrics['loss'])
        
        # Update curriculum metrics after each iteration
        if curriculum_manager is not None:
            curriculum_manager.update_metrics(metrics)
        
        # Log metrics every iteration like the bible
        if rank == 0:
            current_stage = curriculum_manager.current_stage if curriculum_manager else 0
            print(f"Iter {it}: loss={metrics['loss']:.4f}, acc={metrics.get('accuracy', 0):.3f}, "
                  f"kl={metrics.get('kl_loss', 0):.4f}, gpt2_kl={metrics.get('gpt2_kl', 0):.4f}, "
                  f"stage={current_stage}, tokens={tokens_seen:,}", flush=True)
        
        # WandB logging
        if rank == 0 and it % 2 == 0:
            log_dict = {
                "iteration": it,
                "loss/total": metrics['loss'],
                "loss/policy": metrics['pg_loss'],
                "loss/kl": metrics['kl_loss'],
                "loss/entropy": metrics['entropy_loss'],
                "loss/cross_entropy": metrics.get('cross_entropy', 0),
                "rewards/mean": metrics['reward_mean'],
                "rewards/std": metrics['reward_std'],
                "advantages/mean": metrics['advantage_mean'],
                "advantages/std": metrics['advantage_std'],
                "training/lr": lr,
                "training/kl_weight": metrics['kl_weight'],
                "training/num_selected": metrics['num_selected'],
                "training/tokens_seen": tokens_seen,
            }
            
            # Add curriculum metrics
            if curriculum_manager is not None:
                log_dict.update({
                    "curriculum/stage": curriculum_manager.current_stage,
                    "curriculum/perplexity": curriculum_manager.calculate_perplexity(metrics.get('cross_entropy', metrics['loss'])),
                    "curriculum/ema_perplexity": curriculum_manager.ema_perplexity if curriculum_manager.ema_perplexity is not None else 0,
                    "curriculum/stage_iterations": curriculum_manager.stage_iterations,
                })
            
            if USE_CONFIDENCE_SCALING:
                log_dict.update({
                    "confidence/mean": metrics['confidence_mean'],
                    "confidence/std": metrics['confidence_std'],
                })
            
            wandb.log(log_dict)
        
        # Evaluation
        if rank == 0 and it % 100 == 0:
            eval_metrics = evaluate_model(
                model.module if world_size > 1 else model,
                val_data, tokenizer, device
            )
            print(f"\n[Eval] Perplexity: {eval_metrics['perplexity']:.2f}")
            print(f"Sample: {eval_metrics['sample'][:200]}...\n")
        
        # Checkpointing
        if rank == 0 and SAVE_INTERMEDIATE_CHECKPOINTS and it % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                model.module if world_size > 1 else model,
                optimizer, it, metrics,
                scaler=scaler,
                curriculum_manager=curriculum_manager
            )
    
    # Final checkpoint
    if rank == 0 and SAVE_FINAL_CHECKPOINT:
        save_checkpoint(
            model.module if world_size > 1 else model,
            optimizer, it, metrics, "checkpoints/final",
            scaler=scaler,
            curriculum_manager=curriculum_manager
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