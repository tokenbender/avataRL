#!/usr/bin/env python
"""
GRPO Dense GPT – V2 with Flash‑Attention with  RoPE & QK‑Norm
===========================================================

"""

# ─── hyper-params ────────────────────────────────────────────────────────
CONTEXT_LEN = 32                  # Sliding window context
HORIZON = 8                       # Characters to generate
BATCH = 512                       # Large batch via accumulation
MICRO_BATCH = 64                  # Moderate micro-batch 
GRAD_ACCUM = BATCH // MICRO_BATCH # 8 gradient accumulation steps
TOTAL_ITERS = 500                 # Training iterations
LR = 3e-5                         # Learning rate
BETA_KL = 1e-3                    # KL divergence coefficient
KL_WARM = 50_000                  # KL warmup tokens
NEG_REWARD = False                # No negative rewards
GPU_TYPE = "H100"                 
CLIP_RATIO = 0.5                  # PPO clip ratio
K_SAMPLES = 4                     # Samples per context
ENTROPY_COEF = 0.01               # Entropy bonus coefficient
TEMPERATURE = 1.2                 # Sampling temperature
MIN_VARIANCE = 0.1                # Minimum advantage variance                

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
# USE_FLASH_ATTN is set by the auto-configuration above      




# ─── imports ─────────────────────────────────────────────────────────────
import math
import requests
from pathlib import Path
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import wandb
import modal

stub = modal.App("grpo-dense-v2-rope-qknorm")

# Flash‑Attention wheel (CUDA 12.3 + Torch 2.3). Change if needed.
FLASH_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
    "flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "torch==2.3.0",
        "tqdm",
        "wandb",
        "requests",
        "matplotlib",
        "nvidia-ml-py3",
    )
    .pip_install(FLASH_WHEEL)
)

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

# ─── RoPE helpers ────────────────────────────────────────────────────────

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


# ─── Data loader (unchanged) ─────────────────────────────────────────────
class ContextualTextLoader:
    def __init__(self, text, enc, B, T, context_len, device="cuda"):
        self.data = enc(text)
        self.B, self.T, self.context_len = B, T, context_len
        self.device = device
        self.pos = context_len

    def next(self):
        if self.pos + self.B * self.T + 1 > len(self.data):
            self.pos = self.context_len
        ctx, tgt = [], []
        for b in range(self.B):
            s = self.pos - self.context_len + b * self.T
            e = self.pos + b * self.T
            ctx.append(self.data[s:e])
            tgt.append(self.data[e : e + self.T])
        self.pos += self.B * self.T
        return torch.stack(ctx).to(self.device), torch.stack(tgt).to(self.device)


# ─── Attention with RoPE + QK‑Norm ───────────────────────────────────────
class OptimizedAttention(nn.Module):
    def __init__(self, n_emb: int, n_head: int, dropout: float = 0.1, max_seq: int = CONTEXT_LEN + HORIZON):
        super().__init__()
        assert n_emb % n_head == 0
        self.n_head = n_head
        self.head_dim = n_emb // n_head
        self.qkv = nn.Linear(n_emb, 3 * n_emb, bias=False)
        self.o_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryCache(self.head_dim, max_seq)

        self.use_flash = False
        if USE_FLASH_ATTN:
            try:
                from flash_attn import flash_attn_qkvpacked_func

                self.flash_func = flash_attn_qkvpacked_func
                self.use_flash = True
            except Exception:
                pass

    @staticmethod
    def _qk_norm(q: torch.Tensor, k: torch.Tensor, eps: float = 1e-6):
        q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
        k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps)
        return q, k

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, h, d)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, h, T, d)

        sin, cos = self.rope(T)
        q, k = (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)
        q, k = self._qk_norm(q, k) # added qk norm for stable gradients and efficient training

        if self.use_flash:
            packed = torch.stack((q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)), dim=2)  # (B, T, 3, h, d)
            dtype0 = packed.dtype
            if dtype0 not in (torch.float16, torch.bfloat16):
                packed = packed.to(torch.bfloat16)
            out = self.flash_func(packed, causal=True, dropout_p=0.1 if self.training else 0.0)  # (B, T, h, d)
            out = out.to(dtype0).reshape(B, T, C)  # FIX: merge head dim
        else:
            scores = torch.matmul(q, k.transpose(-2, -1))  # (B, h, T, T)
            if mask is not None:
                scores = scores.masked_fill(mask[None, None, :T, :T], float("-inf"))
            attn = self.drop(F.softmax(scores, dim=-1))
            out = torch.matmul(attn, v)  # (B, h, T, d)
            out = out.transpose(1, 2).reshape(B, T, C)

        return self.o_proj(out)


# ─── Transformer block ─────────────────────────────────────────────────--
class TransformerBlock(nn.Module):
    def __init__(self, n_emb: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attn = OptimizedAttention(n_emb, n_head, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * n_emb, n_emb, bias=False),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _rms(x: torch.Tensor, eps: float = 1e-5):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        x = x + self.attn(self._rms(x), mask)
        x = x + self.ffn(self._rms(x))
        return x


# ─── GPT wrapper ─────────────────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int = N_LAYER, n_head: int = N_HEAD, n_emb: int = N_EMB, context_len: int = CONTEXT_LEN + HORIZON):
        super().__init__()
        self.context_len = context_len
        self.wte = nn.Embedding(vocab_size, n_emb)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([TransformerBlock(n_emb, n_head) for _ in range(n_layer)])
        self.head = nn.Linear(n_emb, vocab_size, bias=False)
        self.head.weight = self.wte.weight  # weight tying
        self.apply(self._init)
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), 1).bool(), persistent=False)

    @staticmethod
    def _init(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    @staticmethod
    def _rms(x: torch.Tensor, eps: float = 1e-5):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        x = self.drop(self.wte(idx))
        m = self.mask[:T, :T]
        for blk in self.blocks:
            x = blk(x, m)
        x = self._rms(x)
        return self.head(x)


\

# ─── Helper functions ─────────────────────────────────────────────────────
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

# ─── Temperature-based generation (from v2) ──────────────────────────────
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
        
        with autocast(enabled=USE_AMP, dtype=torch.bfloat16 if AMP_DTYPE == "bfloat16" else torch.float16):
            logits = model(ctx_window)[:, -1, :] / temperature
        
        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, 1)
        ctx = torch.cat([ctx, next_char], dim=1)
    
    # Reshape to [B, K, horizon]
    generated = ctx[:, -horizon:].reshape(B, K, horizon)
    return generated

# ─── Improved reward computation (from v2) ───────────────────────────────
def compute_rewards(gen, ref, ref_logits):
    """
    Compute rewards with better scaling (adapted from v2)
    """
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
    """Compute rewards with detailed tracking"""
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

def compute_entropy(logits):
    """Compute entropy to measure model uncertainty"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

class AdaptiveKLController:
    """Adaptive KL coefficient based on current KL divergence"""
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

# ─── Ultra-optimized training loop ───────────────────────────────────────
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
    
    # Initialize wandb
    run = wandb.init(
        project="gpt2-grpo-v2",
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
            
            # Generate samples with temperature-based sampling
            with torch.no_grad():
                G = generate_with_temperature(actor_old, ctx, HORIZON, K_SAMPLES, temperature=TEMPERATURE)
            
            # Calculate rewards
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
            
            # Compute advantages
            base = R.mean(dim=1, keepdim=True)
            adv = R - base
            adv_std = torch.maximum(adv.std(dim=1, keepdim=True), torch.tensor(MIN_VARIANCE, device=DEV))
            adv = adv / adv_std
            
            # Store for logging
            all_R.append(R)
            all_adv.append(adv)
            
            # Flatten
            flat_G = G.reshape(-1, HORIZON)
            flat_adv = adv.reshape(-1, HORIZON)
            
            # Forward pass
            ctx_for_loss = ctx.repeat_interleave(K_SAMPLES, dim=0)
            full_seq = torch.cat([ctx_for_loss, flat_G], dim=1)
            
            with autocast(enabled=USE_AMP, dtype=torch.bfloat16):
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
                
                # Update KL coefficient
                if ADAPTIVE_KL and kl_controller and accum_step == 0:
                    current_kl_coef = kl_controller.update(kl.item(), current_kl_coef)
                else:
                    current_kl_coef = min(BETA_KL, BETA_KL * chars_seen / max(1, KL_WARM))
                
                # Total loss
                loss = (pol_loss + current_kl_coef * kl - ENTROPY_COEF * entropy) / GRAD_ACCUM
            
            # Backward
            scaler.scale(loss).backward()
            total_loss += loss.item()
            
            # Track metrics
            all_metrics["kl"].append(kl.item())
            all_metrics["entropy"].append(entropy.item())
            all_metrics["pol_loss"].append(pol_loss.item())
            all_metrics["ratio"].append(ratio.mean().item())
            all_metrics["ratio_max"].append(ratio.max().item())
        
        # Optimizer step (potentially fused)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        
        chars_seen += BATCH * HORIZON
        
        # Update old policy every 5 iterations
        if it % 5 == 0:
            actor_old.load_state_dict(actor.state_dict())
            actor_old.eval()
        
        # Logging (reference style + accuracy)
        if it % LOG_INTERVAL == 0:
            # Aggregate rewards and advantages from all accumulation steps
            R_all = torch.cat(all_R, dim=0) if all_R else R
            adv_all = torch.cat(all_adv, dim=0) if all_adv else adv
            
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
                "accuracy": np.mean(all_metrics.get("accuracy", [0])),  # Our addition
                "chars": chars_seen,
            }, step=chars_seen)
        
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
                
                with autocast(enabled=USE_AMP, dtype=torch.bfloat16):
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

# ─── Local entry point ────────────────────────────────────────────────────
@stub.local_entrypoint()
def main():
    train_remote.remote()

if __name__ == "__main__":
    main()
