#!/usr/bin/env python

# Hyperparameters
CONTEXT_LEN = 8                   # Sliding window context
HORIZON = 1                       # Single character prediction for exhaustive exploration
BATCH = 4096                      # Large batch via accumulation
MICRO_BATCH = 512                 # Moderate micro-batch 
GRAD_ACCUM = BATCH // MICRO_BATCH # 8 gradient accumulation steps
EPOCHS = 3                        # Number of epochs to train
DATASET_SIZE = 1_115_394          # Exact size of TinyShakespeare in characters
ITERS_PER_EPOCH = DATASET_SIZE // BATCH  # ~2,178 iterations per epoch
TOTAL_ITERS = ITERS_PER_EPOCH * EPOCHS   # Total training iterations
LR = 3e-5                         # Learning rate
BETA_KL = 1e-3                    # KL divergence coefficient
KL_WARM = 50_000                  # KL warmup tokens
NEG_REWARD = False                # No negative rewards
GPU_TYPE = "H100"                 
CLIP_RATIO = 0.5                  # PPO clip ratio
K_SAMPLES = 65                    # We explore ALL 65 characters exhaustively
ENTROPY_COEF = 0.08               # Entropy bonus coefficient
TEMPERATURE = 1.2                 # Sampling temperature (not used in exhaustive)
MIN_VARIANCE = 0.1                # Minimum advantage variance
USE_EXHAUSTIVE = True             # Use exhaustive exploration
USE_CONFIDENCE_SCALING = True     # Enable confidence scaling rewards
CONFIDENCE_WEIGHT = 0.5           # Conservative confidence scaling weight
CONFIDENCE_CLIP = 2.0             # Maximum confidence scaling factor
ENABLE_CONFIDENCE_PENALTY = False # Phase 2: Enable penalties for overconfident wrong predictions

# Model architecture
N_LAYER = 6                       
N_HEAD = 8                        
N_EMB = 512                       # Wider for better GPU util

# Optimization settings
USE_FLASH_ATTN = True             # Keep Flash Attention
USE_TORCH_COMPILE = False         # Disable torch.compile to avoid conflicts
USE_CHUNKED_LOSS = False          # Use standard loss computation
PACKED_DATASET = False            # Use standard data loading
USE_8BIT_OPTIMIZER = False        # Use standard AdamW
NUM_CUDA_STREAMS = 1              # Single stream for simplicity

# Advanced settings
ADAPTIVE_KL = True                
KL_TARGET = 0.02                  
GRAD_CLIP = 1.0                   
LOG_INTERVAL = 5                  
EVAL_INTERVAL = 20                
SAMPLE_INTERVAL = 20              

# Mixed precision
USE_AMP = True                    
AMP_DTYPE = "bfloat16"           

# Imports
import requests, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import modal, wandb, copy
from collections import defaultdict, deque
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from torch.cuda.amp import autocast, GradScaler
from functools import partial
import warnings

stub = modal.App("exhaustive-confidence-scaling-v2-h200-phase1")

# Modal image with Flash Attention
flash_attn_wheel = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch==2.5.0", "tqdm", "wandb", "requests", "matplotlib", "nvidia-ml-py3")
    .pip_install(flash_attn_wheel)
)

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt")


# RoPE Helper functions
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryCache(nn.Module):
    def __init__(self, head_dim: int, max_len: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))
        t = torch.arange(max_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (max_len, d/2)
        sin, cos = freqs.sin(), freqs.cos()
        self.register_buffer("sin_base", sin, persistent=False)
        self.register_buffer("cos_base", cos, persistent=False)

    def forward(self, seq_len: int):
        sin = self.sin_base[:seq_len].repeat_interleave(2, dim=-1)
        cos = self.cos_base[:seq_len].repeat_interleave(2, dim=-1)
        return sin[None, None, :, :], cos[None, None, :, :]


class ContextualTextLoader:
    """Provides longer context windows for better prediction"""
    def __init__(self, text, enc, B, T, context_len, device='cuda'):
        self.data = enc(text)
        self.B, self.T, self.context_len = B, T, context_len
        self.device = device
        self.pos = context_len  # Start after we have enough context
        
    def next(self):
        # Ensure we have enough context
        if self.pos + self.B*self.T + 1 > len(self.data):
            self.pos = self.context_len
            
        contexts = []
        targets = []
        
        for b in range(self.B):
            # Get context window using sliding approach
            ctx_start = self.pos - self.context_len + b*self.T
            ctx_end = self.pos + b*self.T
            
            context = self.data[ctx_start:ctx_end]
            target = self.data[ctx_end:ctx_end + self.T]
            
            contexts.append(context)
            targets.append(target)
        
        self.pos += self.B * self.T
        
        return (torch.stack(contexts).to(self.device), 
                torch.stack(targets).to(self.device))

class OptimizedAttention(nn.Module):
    """TorchTune-style optimized attention"""
    def __init__(self, n_emb, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.n_emb = n_emb
        self.head_dim = n_emb // n_head
        
        # Fused QKV projection for efficiency
        self.qkv = nn.Linear(n_emb, 3 * n_emb, bias=False)
        self.o_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.dropout = nn.Dropout(dropout)
        max_seq = CONTEXT_LEN + HORIZON
        self.rope = RotaryCache(self.head_dim, max_seq)
        # Try to use Flash Attention
        self.use_flash_attn = False
        if USE_FLASH_ATTN:
            try:
                from flash_attn import flash_attn_qkvpacked_func
                self.flash_attn_func = flash_attn_qkvpacked_func
                self.use_flash_attn = True
            except:
                pass
    
                
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Compute QKV in one go
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        
        q, k, v = qkv.unbind(dim=2)  # each (B, T, h, d)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, h, T, d)

        sin, cos = self.rope(T)
        q, k = (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)
        q, k = norm(q), norm(k)

        if self.use_flash_attn:
            packed = torch.stack((q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)), dim=2)  # (B, T, 3, h, d)
            dtype0 = packed.dtype
            if dtype0 not in (torch.float16, torch.bfloat16):
                packed = packed.to(torch.bfloat16)
            out = self.flash_attn_func(packed, causal=True, dropout_p=0.1 if self.training else 0.0)  # (B, T, h, d)
            out = out.to(dtype0).reshape(B, T, C)
        else:
            # Standard attention with memory-efficient implementation
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Scaled dot-product attention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, C) 
            out = self.o_proj(y)
        return out

def norm(x):
    """RMSNorm implementation - compatible with PyTorch 2.3"""
    # RMS normalization: x / sqrt(mean(x^2) + eps)
    return F.rms_norm(x, (x.size(-1),))


class TransformerBlock(nn.Module):
    def __init__(self, n_emb, n_head, dropout=0.1):
        super().__init__()
        self.attn = OptimizedAttention(n_emb, n_head, dropout)
        
        # Fused FFN operations
        self.ffn = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb, bias=False),
            nn.GELU(approximate='tanh'),  # Faster GELU approximation
            nn.Linear(4 * n_emb, n_emb, bias=False),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Pre-norm architecture with RMSNorm
        x = x + self.attn(norm(x), mask)
        x = x + self.ffn(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, V, n_layer=N_LAYER, n_head=N_HEAD, n_emb=N_EMB, 
                 context_len=CONTEXT_LEN):
        super().__init__()
        self.context_len = context_len
        self.n_layer = n_layer
        
        # Embeddings
        self.wte = nn.Embedding(V, n_emb)
        # self.wpe = nn.Embedding(context_len + HORIZON, n_emb)
        self.drop = nn.Dropout(0.1)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(n_emb, n_head, dropout=0.1)
            for _ in range(n_layer)
        ])
        
        self.head = nn.Linear(n_emb, V, bias=False)
        self.head.weight = self.wte.weight  # Weight tying
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Pre-compute causal mask on correct device
        self.register_buffer("causal_mask", torch.triu(
            torch.ones(context_len + HORIZON, context_len + HORIZON), diagonal=1
        ).bool())
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, use_cache=False):
        B, T = idx.shape
        
        
        # Embeddings
        tok_emb = self.wte(idx)
        
        x = self.drop(tok_emb)
        
        # Get mask
        mask = self.causal_mask[:T, :T]
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = norm(x)  # RMSNorm instead of LayerNorm
        logits = self.head(x)
        
        return logits

def ensure_dataset(p="input.txt"):
    if not Path(p).exists():
        Path(p).write_text(requests.get(DATA_URL, timeout=10).text, encoding="utf-8")

def build_vocab(p="input.txt"):
    text = Path(p).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi  = {ch:i for i,ch in enumerate(chars)}
    itos  = {i:ch for ch,i in stoi.items()}
    def enc(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    def dec(t): return "".join(itos[int(i)] for i in t)
    return enc, dec, len(chars), stoi, itos, text

class BigramRef(nn.Module):
    def __init__(self, bigram_counts, V, smoothing=1.0):
        super().__init__()
        self.V = V
        self.probs = torch.ones(V, V) * smoothing
        
        for (c1, c2), count in bigram_counts.items():
            self.probs[c1, c2] += count
            
        self.probs = self.probs / self.probs.sum(dim=1, keepdim=True)
        self.register_buffer("log_probs", torch.log(self.probs + 1e-10))
        
    def forward(self, idx):
        B, T = idx.shape
        logits = torch.zeros(B, T, self.V, device=idx.device)
        logits[:, 0, :] = 0.0
        
        for t in range(1, T):
            prev_char = idx[:, t-1]
            logits[:, t, :] = self.log_probs[prev_char, :]
            
        return logits

def build_bigram_counts(text, stoi):
    counts = defaultdict(int)
    for i in range(len(text) - 1):
        c1 = stoi[text[i]]
        c2 = stoi[text[i+1]]
        counts[(c1, c2)] += 1
    return counts

def build_char_frequencies(text, stoi, V):
    """Build character frequency distribution from Shakespeare text"""
    char_counts = torch.zeros(V)
    for char in text:
        char_counts[stoi[char]] += 1
    char_freq = char_counts / char_counts.sum()
    return char_freq

@torch.no_grad()
def generate_with_temperature(model, contexts, horizon, K, temperature=1.0):
    """
    Generate K samples with temperature-based sampling
    Optimized version using CUDA graphs when possible
    """
    B = contexts.shape[0]
    device = contexts.device
    
    # Expand contexts for K samples
    ctx = contexts.repeat_interleave(K, dim=0)  # [B*K, context_len]
    
    # Generate with temperature
    for _ in range(horizon):
        ctx_window = ctx[:, -model.context_len:] if ctx.shape[1] > model.context_len else ctx
        
        with torch.amp.autocast('cuda',dtype = torch.bfloat16):
            logits = model(ctx_window)[:, -1, :] / temperature
        
        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, 1)
        ctx = torch.cat([ctx, next_char], dim=1)
    
    # Reshape to [B, K, horizon]
    generated = ctx[:, -horizon:].reshape(B, K, horizon)
    return generated

@torch.no_grad()
def generate_exhaustive_single_char(model, contexts, V):
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
    all_chars = torch.arange(V, device=device).unsqueeze(0).expand(B, -1)  # [B, V]
    
    return all_chars, log_probs

def compute_rewards(gen, ref, ref_logits):
    # Exact match reward (scaled up)
    exact_match = (gen == ref).float()
    
    # Partial credit based on reference probability
    with torch.no_grad():
        ref_probs = F.softmax(ref_logits, dim=-1)
        # Get probability of generated token under reference
        B, T = gen.shape
        gen_probs = ref_probs.gather(2, gen.unsqueeze(-1)).squeeze(-1)
        # Log probability as partial reward
        partial_reward = torch.log(gen_probs + 1e-10) / 10.0  # Scale down
    
    # Combined reward
    reward = exact_match + 0.1 * partial_reward
    
    return reward

def compute_rewards_batch(gen, ref, ref_logits):
    reward = compute_rewards(gen, ref, ref_logits)
    exact_match = (gen == ref).float()
    
    with torch.no_grad():
        ref_probs = F.softmax(ref_logits, dim=-1)
        gen_probs = ref_probs.gather(2, gen.unsqueeze(-1)).squeeze(-1)
    
    metrics = {
        "accuracy": exact_match.mean().item(),
        "avg_gen_prob": gen_probs.mean().item(),
        "reward_mean": reward.mean().item(),
        "reward_std": reward.std().item(),
    }
    
    return reward, exact_match, metrics

def compute_exhaustive_rewards(all_chars, ref_char, ref_model, ctx, V, model_log_probs=None, old_probs=None):
    B = ctx.shape[0]
    device = ctx.device
    
    # Get reference model's distribution
    with torch.no_grad():
        ref_logits = ref_model(ctx)[:, -1, :]  # [B, V]
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)  # [B, V]
        
    # Base reward: negative log probability under reference model
    base_rewards = -ref_log_probs  # [B, V]
    
    # Scale rewards to increase variance (3x through scaling factor)
    VARIANCE_SCALE = 3.0
    base_rewards = base_rewards * VARIANCE_SCALE
    
    # Model improvement bonus
    if model_log_probs is not None:
        improvement = model_log_probs - ref_log_probs
        improvement_bonus = torch.clamp(improvement, min=-1.0, max=2.0)
        base_rewards = base_rewards + improvement_bonus
    
    # Exact match bonus
    exact_match_bonus = torch.zeros_like(base_rewards)
    for b in range(B):
        exact_match_bonus[b, ref_char[b]] = 5.0 * VARIANCE_SCALE
    
    # Combine all rewards
    rewards = base_rewards + exact_match_bonus
    
    # Confidence scaling
    if USE_CONFIDENCE_SCALING and old_probs is not None:
        confidence = old_probs[torch.arange(B), ref_char]
        confidence_scale = 1.0 + CONFIDENCE_WEIGHT * confidence.clamp(0, 1)
        confidence_scale = confidence_scale.clamp(1.0, CONFIDENCE_CLIP)
        rewards[torch.arange(B), ref_char] *= confidence_scale
        
        # Phase 2: Penalize overconfident wrong predictions
        if ENABLE_CONFIDENCE_PENALTY:
            top_k = min(10, V)
            top_probs, top_indices = old_probs.topk(top_k, dim=-1)
            wrong_mask = top_indices != ref_char.unsqueeze(1)
            confident_mask = top_probs > 0.5
            penalty_mask = wrong_mask & confident_mask
            penalties = -0.1 * top_probs * penalty_mask.float()
            rewards.scatter_add_(1, top_indices, penalties)
    
    # Normalize rewards to have zero mean per batch
    rewards = rewards - rewards.mean(dim=1, keepdim=True)
    
    return rewards

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

class AdaptiveKLController:
    def __init__(self, target_kl, horizon=10):
        self.target_kl = target_kl
        self.horizon = horizon
        self.kl_history = deque(maxlen=horizon)
        
    def update(self, current_kl, current_coef):
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
    def __init__(self, calibration_threshold=0.15, confidence_collapse_threshold=0.8):
        self.calibration_threshold = calibration_threshold
        self.confidence_collapse_threshold = confidence_collapse_threshold
        self.confidence_history = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        self.gradient_history = deque(maxlen=50)
        
    def update(self, confidence, accuracy, gradient_norm=None):
        self.confidence_history.append(confidence)
        self.accuracy_history.append(accuracy)
        if gradient_norm is not None:
            self.gradient_history.append(gradient_norm)
    
    def check_calibration(self):
        if len(self.confidence_history) < 50:
            return True, 0.0
        
        avg_confidence = np.mean(self.confidence_history)
        avg_accuracy = np.mean(self.accuracy_history)
        calibration_error = abs(avg_confidence - avg_accuracy)
        
        is_calibrated = calibration_error < self.calibration_threshold
        return is_calibrated, calibration_error
    
    def check_confidence_collapse(self):
        if len(self.confidence_history) < 20:
            return False, 0.0
        
        recent_confidence = list(self.confidence_history)[-20:]
        high_conf_ratio = sum(c > self.confidence_collapse_threshold for c in recent_confidence) / len(recent_confidence)
        
        return high_conf_ratio > 0.7, high_conf_ratio
    
    def check_gradient_stability(self):
        if len(self.gradient_history) < 10:
            return True, 0.0
        
        recent_grads = list(self.gradient_history)[-10:]
        grad_variance = np.var(recent_grads)
        grad_trend = np.polyfit(range(len(recent_grads)), recent_grads, 1)[0]
        
        is_stable = grad_variance < 10.0 and grad_trend < 0.1
        return is_stable, grad_variance

@stub.function(
    gpu=GPU_TYPE, 
    image=image, 
    timeout=60*60*6,
    secrets=[modal.Secret.from_name("wandb")]
)
def train_remote():
    # Set all optimizations
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use more GPU memory
    
    DEV = "cuda"
    ensure_dataset()
    ENC, DEC, V, stoi, itos, text = build_vocab()
    
    print(f"Starting GRPO V2 training with Flash Attention")
    print(f"Vocabulary size: {V}")
    print(f"Device: {DEV}")
    print(f"Batch size: {BATCH} (micro-batch: {MICRO_BATCH}, grad accum: {GRAD_ACCUM})")
    print(f"Model size: {N_LAYER} layers, {N_EMB} embedding dim, {N_HEAD} heads")
    print(f"Using Flash Attention: {USE_FLASH_ATTN}")
    print(f"Using Exhaustive Exploration: {USE_EXHAUSTIVE} (exploring all {V} characters)")
    print(f"Training for {EPOCHS} epochs ({TOTAL_ITERS:,} iterations, {ITERS_PER_EPOCH:,} iters/epoch)")
    
    # Initialize models
    bigram_counts = build_bigram_counts(text, stoi)
    ref = BigramRef(bigram_counts, V).to(DEV).eval()
    
    actor = GPT(V).to(DEV)
    actor_old = GPT(V).to(DEV)
    actor_old.load_state_dict(actor.state_dict())
    actor_old.eval()
    
    # Count parameters
    param_count = sum(p.numel() for p in actor.parameters())
    print(f"Total parameters: {param_count:,}")
    
    # Standard optimizer
    opt = torch.optim.AdamW(actor.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    
    # Mixed precision
    scaler = GradScaler(enabled=USE_AMP)
    
    # Data loader
    loader = ContextualTextLoader(text, ENC, MICRO_BATCH, HORIZON, CONTEXT_LEN, device=DEV)
    kl_controller = AdaptiveKLController(KL_TARGET) if ADAPTIVE_KL else None
    
    # Initialize confidence monitor
    confidence_monitor = ConfidenceMonitor() if USE_CONFIDENCE_SCALING else None
    
    # Initialize wandb
    run = wandb.init(
        project="avataRL",
        name="exhaustive-confidence-scaling-phase1-v2-h200",
        config={
            "model": {
                "vocab_size": V,
                "context_len": CONTEXT_LEN,
                "horizon": HORIZON,
                "architecture": "dense-v2-flash-attention",
                "n_layer": N_LAYER,
                "n_head": N_HEAD,
                "n_emb": N_EMB,
                "param_count": param_count,
            },
            "training": {
                "batch_size": BATCH,
                "micro_batch_size": MICRO_BATCH,
                "grad_accum_steps": GRAD_ACCUM,
                "learning_rate": LR,
                "k_samples": K_SAMPLES,
                "exhaustive_exploration": USE_EXHAUSTIVE,
                "exploration_method": "exhaustive_all_chars" if USE_EXHAUSTIVE else "sampling",
                "confidence_scaling": USE_CONFIDENCE_SCALING,
                "confidence_weight": CONFIDENCE_WEIGHT,
                "confidence_clip": CONFIDENCE_CLIP,
                "confidence_penalty_enabled": ENABLE_CONFIDENCE_PENALTY,
            },
            "optimizations": {
                "flash_attention": USE_FLASH_ATTN,
                "mixed_precision": USE_AMP,
                "amp_dtype": AMP_DTYPE,
            }
        }
    )
    
    # Single stream for simplicity
    stream = torch.cuda.Stream()
    
    chars_seen = 0
    current_kl_coef = BETA_KL
    
    # Training loop
    for it in tqdm(range(1, TOTAL_ITERS+1), desc="Training"):
        actor.train()
        opt.zero_grad(set_to_none=True)
        
        total_loss = 0.0
        all_metrics = defaultdict(list)
        all_R = []  # Store rewards for logging
        all_adv = []  # Store advantages for logging
        
        for accum_step in range(GRAD_ACCUM):
            # Load data
            ctx, ref_tok = loader.next()
            
            if USE_EXHAUSTIVE and HORIZON == 1:
                with torch.no_grad():
                    all_chars, old_log_probs = generate_exhaustive_single_char(actor_old, ctx, V)
                    
                ref_char = ref_tok[:, 0]
                old_probs = torch.exp(old_log_probs) if USE_CONFIDENCE_SCALING else None
                R = compute_exhaustive_rewards(all_chars, ref_char, ref, ctx, V, 
                                             model_log_probs=old_log_probs,
                                             old_probs=old_probs)
                
                top_choice = old_log_probs.argmax(dim=-1)
                accuracy = (top_choice == ref_char).float().mean().item()
                all_metrics["accuracy"].append(accuracy)
                
                all_metrics["reward_variance"].append(R.var().item())
                all_metrics["reward_spread"].append((R.max() - R.min()).mean().item())
                
                if hasattr(actor_old, '_prev_log_probs'):
                    prob_change = (old_log_probs - actor_old._prev_log_probs).abs().mean().item()
                    all_metrics["prob_change"].append(prob_change)
                actor_old._prev_log_probs = old_log_probs.detach()
                
                base = R.mean(dim=1, keepdim=True)
                adv = R - base
                adv_std = torch.maximum(adv.std(dim=1, keepdim=True), torch.tensor(MIN_VARIANCE, device=DEV))
                adv = adv / adv_std
                
                all_R.append(R.unsqueeze(2))
                all_adv.append(adv.unsqueeze(2))
                
                flat_G = all_chars.reshape(-1, 1)
                flat_adv = adv.reshape(-1, 1)
                
            else:
                with torch.no_grad():
                    G = generate_with_temperature(actor_old, ctx, HORIZON, K_SAMPLES, temperature=TEMPERATURE)
                
                R = torch.zeros_like(G, dtype=torch.float32)
                
                for k in range(K_SAMPLES):
                    gen = G[:, k, :]
                    
                    with torch.no_grad():
                        full_seq = torch.cat([ctx, gen], dim=1)
                        ref_logits = ref(full_seq[:, -HORIZON-1:])[:, -HORIZON:]
                    
                    reward, accuracy, reward_metrics = compute_rewards_batch(gen, ref_tok, ref_logits)
                    R[:, k, :] = reward
                    
                    for key, val in reward_metrics.items():
                        all_metrics[key].append(val)
                
                base = R.mean(dim=1, keepdim=True)
                adv = R - base
                adv_std = torch.maximum(adv.std(dim=1, keepdim=True), torch.tensor(MIN_VARIANCE, device=DEV))
                adv = adv / adv_std
                
                all_R.append(R)
                all_adv.append(adv)
                
                flat_G = G.reshape(-1, HORIZON)
                flat_adv = adv.reshape(-1, HORIZON)
            
            if USE_EXHAUSTIVE and HORIZON == 1:
                ctx_for_loss = ctx.repeat_interleave(V, dim=0)
            else:
                ctx_for_loss = ctx.repeat_interleave(K_SAMPLES, dim=0)
            full_seq = torch.cat([ctx_for_loss, flat_G], dim=1)
            
            with torch.amp.autocast('cuda',dtype = torch.bfloat16):
                input_seq = full_seq[:, -actor.context_len:]
                
                new_logits = actor(input_seq)
                new_logits = new_logits[:, -HORIZON:]
                
                # Policy loss
                new_dist = torch.distributions.Categorical(logits=new_logits)
                logp_new = new_dist.log_prob(flat_G)
                
                with torch.no_grad():
                    old_logits = actor_old(input_seq)
                    old_logits = old_logits[:, -HORIZON:]
                    old_dist = torch.distributions.Categorical(logits=old_logits)
                    logp_old = old_dist.log_prob(flat_G)
                
                ratio = torch.exp(logp_new - logp_old)
                clipped_ratio = torch.clamp(ratio, 1-CLIP_RATIO, 1+CLIP_RATIO)
                
                pol_loss1 = -flat_adv * ratio
                pol_loss2 = -flat_adv * clipped_ratio
                pol_loss = torch.max(pol_loss1, pol_loss2).mean()
                
                # Entropy and KL
                entropy = new_dist.entropy().mean()
                
                ref_logits = ref(full_seq[:, -HORIZON-1:])[:, -HORIZON:]
                ref_dist = torch.distributions.Categorical(logits=ref_logits)
                kl = torch.distributions.kl.kl_divergence(new_dist, ref_dist).mean()
                
                if ADAPTIVE_KL and kl_controller and accum_step == 0:
                    current_kl_coef = kl_controller.update(kl.item(), current_kl_coef)
                else:
                    current_kl_coef = min(BETA_KL, BETA_KL * chars_seen / max(1, KL_WARM))
                
                # Total loss
                loss = (pol_loss + current_kl_coef * kl - ENTROPY_COEF * entropy) / GRAD_ACCUM
            
            # Backward
            scaler.scale(loss).backward()
            total_loss += loss.item()
            
            all_metrics["kl"].append(kl.item())
            all_metrics["entropy"].append(entropy.item())
            all_metrics["pol_loss"].append(pol_loss.item())
            all_metrics["ratio"].append(ratio.mean().item())
            all_metrics["ratio_max"].append(ratio.max().item())
        
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        
        chars_seen += BATCH * HORIZON
        
        # Update old policy every 5 iterations
        if it % 5 == 0:
            actor_old.load_state_dict(actor.state_dict())
            actor_old.eval()
        
        if it % LOG_INTERVAL == 0:
            R_all = torch.cat(all_R, dim=0) if all_R else R
            adv_all = torch.cat(all_adv, dim=0) if all_adv else adv
            
            current_epoch = it / ITERS_PER_EPOCH
            
            wandb.log({
                "reward": R_all.mean().item(),
                "reward_max": R_all.max().item(),
                "reward_min": R_all.min().item(),
                "advantage_mean": adv_all.mean().item(),
                "advantage_std": adv_all.std().item(),
                "kl": all_metrics.get("kl", [0])[-1] if all_metrics.get("kl") else 0,
                "kl_coef": current_kl_coef,
                "ratio": all_metrics.get("ratio", [1.0])[-1] if all_metrics.get("ratio") else 1.0,
                "ratio_max": all_metrics.get("ratio_max", [1.0])[-1] if all_metrics.get("ratio_max") else 1.0,
                "entropy": all_metrics.get("entropy", [0])[-1] if all_metrics.get("entropy") else 0,
                "pol_loss": all_metrics.get("pol_loss", [0])[-1] if all_metrics.get("pol_loss") else 0,
                "total_loss": total_loss * GRAD_ACCUM,
                "accuracy": np.mean(all_metrics.get("accuracy", [0])),
                "reward_variance": np.mean(all_metrics.get("reward_variance", [0])),
                "reward_spread": np.mean(all_metrics.get("reward_spread", [0])),
                "prob_change": np.mean(all_metrics.get("prob_change", [0])) if all_metrics.get("prob_change") else 0,
                "chars": chars_seen,
                "epoch": current_epoch,
            }, step=chars_seen)
            
            if USE_CONFIDENCE_SCALING and old_probs is not None:
                correct_confidence = old_probs[torch.arange(MICRO_BATCH), ref_char].mean().item()
                confidence_scale_avg = (1.0 + CONFIDENCE_WEIGHT * old_probs[torch.arange(MICRO_BATCH), ref_char].clamp(0, 1)).mean().item()
                wandb.log({
                    "confidence/mean": correct_confidence,
                    "confidence/reward_boost": confidence_scale_avg - 1.0,
                    "gradient/norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                }, step=chars_seen)
                
                if confidence_monitor:
                    avg_accuracy = np.mean(all_metrics.get("accuracy", [0]))
                    confidence_monitor.update(correct_confidence, avg_accuracy, grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                    
                    is_calibrated, calibration_error = confidence_monitor.check_calibration()
                    has_collapse, collapse_ratio = confidence_monitor.check_confidence_collapse()
                    is_stable, grad_variance = confidence_monitor.check_gradient_stability()
                    
                    wandb.log({
                        "monitoring/calibration_error": calibration_error,
                        "monitoring/confidence_collapse_ratio": collapse_ratio,
                        "monitoring/gradient_variance": grad_variance,
                        "monitoring/is_calibrated": float(is_calibrated),
                        "monitoring/has_collapse": float(has_collapse),
                        "monitoring/is_stable": float(is_stable),
                    }, step=chars_seen)
                    
                    if not is_calibrated:
                        print(f"WARNING: Model miscalibrated! Error: {calibration_error:.3f}")
                    if has_collapse:
                        print(f"WARNING: Confidence collapse detected! Ratio: {collapse_ratio:.3f}")
                    if not is_stable:
                        print(f"WARNING: Gradient instability! Variance: {grad_variance:.3f}")
        
        # Generate samples periodically
        if it % SAMPLE_INTERVAL == 0:
            with torch.no_grad():
                # Generate at different temperatures
                test_ctx = ctx[:3]
                samples = []
                
                for i, temp in enumerate([0.8, 1.0, 1.2]):
                    if i < test_ctx.shape[0]:
                        gen = generate_with_temperature(
                            actor, test_ctx[i:i+1], 150, 1, temperature=temp
                        )[0, 0]
                        
                        sample_text = DEC(gen)
                        context_text = DEC(test_ctx[i, -50:])
                        
                        samples.append(f"[Temp={temp}] Context: ...{context_text}\nGenerated: {sample_text}")
                
                wandb.log({
                    "samples": wandb.Html(
                        "<pre>" + "\n\n".join(samples) + "</pre>"
                    )
                }, step=chars_seen)
        
        # Evaluation
        if it % EVAL_INTERVAL == 0:
            actor.eval()
            with torch.no_grad():
                val_ctx, val_ref = loader.next()
                
                with torch.amp.autocast('cuda',dtype = torch.bfloat16):
                    val_logits = actor(val_ctx)
                    val_logits = val_logits[:, -HORIZON:, :]
                    
                    val_loss = F.cross_entropy(
                        val_logits.reshape(-1, V),
                        val_ref.reshape(-1)
                    )
                
                perplexity = torch.exp(val_loss)
                
                wandb.log({
                    "eval/perplexity": perplexity.item(),
                    "eval/loss": val_loss.item(),
                }, step=chars_seen)
    
    # Final summary
    wandb.summary.update({
        "final_accuracy": avg_metrics.get("accuracy", 0) if 'avg_metrics' in locals() else 0,
        "final_perplexity": perplexity.item() if 'perplexity' in locals() else None,
        "total_chars_seen": chars_seen,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    })
    
    run.finish()

@stub.local_entrypoint()
def main():
    train_remote.remote()

if __name__ == "__main__":
    main()
