#!/usr/bin/env python

PROJECT_NAME = "avataRL-optimized-grpo"
EXPERIMENT_NAME = "continuous-batch-paged-attn-grpo"

CONTEXT_LEN =32
HORIZON = 1
BATCH = 4096
MICRO_BATCH = 512
GRAD_ACCUM = BATCH // MICRO_BATCH
EPOCHS = 1
DATASET_SIZE = 1_115_394
ITERS_PER_EPOCH = DATASET_SIZE // BATCH
TOTAL_ITERS = int(ITERS_PER_EPOCH * EPOCHS)
LR = 3e-3
BETA_KL = 0.1
KL_WARM = int(DATASET_SIZE * 0.8)
KL_FREE_FRACTION = 0.1
GPU_TYPE = "H200"
CLIP_RATIO = 0.5
K_SAMPLES = 64
TEMPERATURE = 1.0
EVAL_TEMPERATURE = 1.0
MIN_VARIANCE = 0.1
USE_EXHAUSTIVE = True
USE_CONFIDENCE_SCALING = True
CONFIDENCE_WEIGHT = 0.7
CONFIDENCE_CLIP = 2.0
ENABLE_CONFIDENCE_PENALTY = True


N_LAYER = 6
N_HEAD = 6
N_EMB = 384

USE_FLASH_ATTN = True
USE_TORCH_COMPILE = False
USE_CHUNKED_LOSS = False
PACKED_DATASET = False
USE_8BIT_OPTIMIZER = False
NUM_CUDA_STREAMS = 1

DROPOUT = 0.1
WEIGHT_DECAY = 0.01

USE_LR_DECAY = True
LR_DECAY_TYPE = "cosine"
MIN_LR = 1e-5
WARMUP_ITERS = 100

ADAPTIVE_KL = True
KL_TARGET = 0.01
GRAD_CLIP = 1.0
LOG_INTERVAL = 2
EVAL_INTERVAL = int(ITERS_PER_EPOCH * 0.1)
SAMPLE_INTERVAL = int(ITERS_PER_EPOCH * 0.1)             

USE_AMP = True
AMP_DTYPE = "bfloat16"           

USE_CONTINUOUS_BATCHING = True
MAX_BATCH_TOKENS = 4096  # Maximum tokens in a batch
DYNAMIC_BATCH_SIZE = True  # Adjust batch size dynamically based on sequence lengths

USE_PAGED_ATTENTION = True
PAGE_SIZE = 512  # Size of each attention page
MAX_PAGES_PER_SEQ = 8  # Maximum pages per sequence

import requests, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import modal, wandb
from collections import defaultdict, deque
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from torch.cuda.amp import autocast, GradScaler
import gc


stub = modal.App(f"{PROJECT_NAME}-{EXPERIMENT_NAME}")

flash_attn_wheel = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch==2.5.0", "tqdm", "wandb", "requests", "matplotlib", "nvidia-ml-py3")
    .pip_install(flash_attn_wheel)
)

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryCache(nn.Module):
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


class ContextualTextLoader:
    def __init__(self, text, enc, B, T, context_len, device='cuda'):
        self.data = enc(text)
        self.B, self.T, self.context_len = B, T, context_len
        self.device = device
        self.pos = context_len
        
    def next(self):
        if self.pos + self.B*self.T + 1 > len(self.data):
            self.pos = self.context_len
            
        contexts = []
        targets = []
        
        for b in range(self.B):
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
    def __init__(self, n_emb, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.n_emb = n_emb
        self.head_dim = n_emb // n_head
        
        self.qkv = nn.Linear(n_emb, 3 * n_emb, bias=False)
        self.o_proj = nn.Linear(n_emb, n_emb, bias=False)
        self.dropout = nn.Dropout(dropout)
        max_seq = CONTEXT_LEN + HORIZON
        self.rope = RotaryCache(self.head_dim, max_seq)
        self.use_flash_attn = False
        if USE_FLASH_ATTN:
            try:
                from flash_attn import flash_attn_qkvpacked_func
                self.flash_attn_func = flash_attn_qkvpacked_func
                self.use_flash_attn = True
            except:
                pass
        
        self.use_paged_attention = USE_PAGED_ATTENTION if 'USE_PAGED_ATTENTION' in globals() else False
        if self.use_paged_attention:
            self.page_size = PAGE_SIZE if 'PAGE_SIZE' in globals() else 512
            self.kv_cache = None
    
                
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        sin, cos = self.rope(T)
        q, k = (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)
        q, k = norm(q), norm(k)

        if self.use_flash_attn:
            packed = torch.stack((q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)), dim=2)
            dtype0 = packed.dtype
            if dtype0 not in (torch.float16, torch.bfloat16):
                packed = packed.to(torch.bfloat16)
            out = self.flash_attn_func(packed, causal=True, dropout_p=self.dropout.p if self.training else 0.0)
            out = out.to(dtype0).reshape(B, T, C)
        elif self.use_paged_attention:
            out = self._paged_attention(q, k, v, mask)
            out = self.o_proj(out)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, C) 
            out = self.o_proj(y)
        return out
    
    def _paged_attention(self, q, k, v, mask=None):
        B, H, T, D = q.shape
        
        num_pages = (T + self.page_size - 1) // self.page_size
        
        output = torch.zeros(B, H, T, D, device=q.device, dtype=q.dtype)
        
        for page_idx in range(num_pages):
            start_idx = page_idx * self.page_size
            end_idx = min((page_idx + 1) * self.page_size, T)
            page_len = end_idx - start_idx
            
            q_page = q[:, :, start_idx:end_idx, :]
            
            k_context = k[:, :, :end_idx, :]
            v_context = v[:, :, :end_idx, :]
            
            scores = torch.matmul(q_page, k_context.transpose(-2, -1)) / math.sqrt(D)
            
            if mask is not None:
                mask_page = mask[start_idx:end_idx, :end_idx]
                scores = scores.masked_fill(mask_page.unsqueeze(0).unsqueeze(0), float('-inf'))
            else:
                causal_mask = torch.triu(torch.ones(page_len, end_idx, device=q.device), 
                                       diagonal=start_idx+1-end_idx).bool()
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            
            if self.training and self.dropout.p > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout.p)
            
            page_output = torch.matmul(attn_weights, v_context)
            output[:, :, start_idx:end_idx, :] = page_output
        
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return output

def norm(x):
    """RMSNorm implementation - compatible with PyTorch 2.3"""
    # Add epsilon for numerical stability
    eps = 1e-8
    # Manual RMSNorm with stability checks
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms


class TransformerBlock(nn.Module):
    def __init__(self, n_emb, n_head, dropout=0.1):
        super().__init__()
        self.attn = OptimizedAttention(n_emb, n_head, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb, bias=False),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * n_emb, n_emb, bias=False),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        x = x + self.attn(norm(x), mask)
        x = x + self.ffn(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, V, n_layer=N_LAYER, n_head=N_HEAD, n_emb=N_EMB, 
                 context_len=CONTEXT_LEN, dropout=DROPOUT):
        super().__init__()
        self.context_len = context_len
        self.n_layer = n_layer
        self.dropout = dropout
        
        self.wte = nn.Embedding(V, n_emb)
        self.drop = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(n_emb, n_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        
        self.head = nn.Linear(n_emb, V, bias=False)
        self.head.weight = self.wte.weight
        
        self.apply(self._init_weights)
        
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
        
        tok_emb = self.wte(idx)
        
        if torch.isnan(tok_emb).any():
            print(f"NaN detected in token embeddings! Shape: {tok_emb.shape}")
            print(f"idx values: min={idx.min()}, max={idx.max()}")
            raise ValueError("NaN in embeddings")
        
        x = self.drop(tok_emb)
        
        mask = self.causal_mask[:T, :T]
        
        for i, layer in enumerate(self.layers):
            x_prev = x
            x = layer(x, mask)
            
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"NaN/Inf detected after layer {i}! Shape: {x.shape}")
                print(f"Previous layer output had NaN: {torch.isnan(x_prev).any()}")
                raise ValueError(f"NaN/Inf in layer {i} output")
        
        x = norm(x)
        
        if torch.isnan(x).any():
            print(f"NaN detected after final norm! Shape: {x.shape}")
            raise ValueError("NaN after normalization")
        
        logits = self.head(x)
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"NaN/Inf detected in final logits! Shape: {logits.shape}")
            raise ValueError("NaN/Inf in logits")
        
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        
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

class OnTheFlyNGramRef(nn.Module):
    def __init__(self, text, stoi, V, smoothing=1.0):
        super().__init__()
        self.V = V
        self.smoothing = smoothing
        
        self.bigram_counts = torch.ones(V, V) * smoothing
        self.trigram_counts = torch.ones(V, V, V) * smoothing
        self.fourgram_counts = torch.ones(V, V, V, V) * smoothing
        
        text_indices = [stoi[c] for c in text]
        
        for i in range(len(text_indices) - 1):
            self.bigram_counts[text_indices[i], text_indices[i+1]] += 1
            
        for i in range(len(text_indices) - 2):
            self.trigram_counts[text_indices[i], text_indices[i+1], text_indices[i+2]] += 1
            
        for i in range(len(text_indices) - 3):
            self.fourgram_counts[text_indices[i], text_indices[i+1], text_indices[i+2], text_indices[i+3]] += 1
        
        bigram_probs = self.bigram_counts / self.bigram_counts.sum(dim=-1, keepdim=True)
        self.register_buffer("bigram_log_probs", torch.log(bigram_probs + 1e-8).clamp(min=-20.0))
        
        trigram_probs = self.trigram_counts / self.trigram_counts.sum(dim=-1, keepdim=True)
        self.register_buffer("trigram_log_probs", torch.log(trigram_probs + 1e-8).clamp(min=-20.0))
        
        fourgram_probs = self.fourgram_counts / self.fourgram_counts.sum(dim=-1, keepdim=True)
        self.register_buffer("fourgram_log_probs", torch.log(fourgram_probs + 1e-8).clamp(min=-20.0))
        
    def forward(self, idx, return_components=False):
        B, T = idx.shape
        device = idx.device
        
        bigram_logits = torch.zeros(B, T, self.V, device=device)
        trigram_logits = torch.zeros(B, T, self.V, device=device)
        fourgram_logits = torch.zeros(B, T, self.V, device=device)
        
        uniform_logits = torch.zeros(self.V, device=device)
        
        for t in range(T):
            if t == 0:
                bigram_logits[:, t, :] = uniform_logits
                trigram_logits[:, t, :] = uniform_logits
                fourgram_logits[:, t, :] = uniform_logits
            elif t == 1:
                prev_char = idx[:, t-1]
                bigram_logits[:, t, :] = self.bigram_log_probs[prev_char]
                trigram_logits[:, t, :] = uniform_logits
                fourgram_logits[:, t, :] = uniform_logits
            elif t == 2:
                prev_char = idx[:, t-1]
                prev_prev_char = idx[:, t-2]
                bigram_logits[:, t, :] = self.bigram_log_probs[prev_char]
                trigram_logits[:, t, :] = self.trigram_log_probs[prev_prev_char, prev_char]
                fourgram_logits[:, t, :] = uniform_logits
            else:
                prev_char = idx[:, t-1]
                prev_prev_char = idx[:, t-2]
                prev_prev_prev_char = idx[:, t-3]
                bigram_logits[:, t, :] = self.bigram_log_probs[prev_char]
                trigram_logits[:, t, :] = self.trigram_log_probs[prev_prev_char, prev_char]
                fourgram_logits[:, t, :] = self.fourgram_log_probs[prev_prev_prev_char, prev_prev_char, prev_char]
        
        weights = torch.tensor([1.0, 1.0, 1.0], device=device)
        weights = weights / weights.sum()
        
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

def build_bigram_counts(text, stoi):
    counts = defaultdict(int)
    for i in range(len(text) - 1):
        c1 = stoi[text[i]]
        c2 = stoi[text[i+1]]
        counts[(c1, c2)] += 1
    return counts

def build_char_frequencies(text, stoi, V):
    char_counts = torch.zeros(V)
    for char in text:
        char_counts[stoi[char]] += 1
    char_freq = char_counts / char_counts.sum()
    return char_freq


class ContinuousBatchGenerator:
    def __init__(self, model, context_len, vocab_size, device="cuda"):
        self.model = model
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.device = device
        
        self.max_concurrent = 8192
        self.kv_cache = None
        
    @torch.no_grad()
    def generate_continuous_batch(self, contexts, num_samples_per_context, temperature=1.0, max_tokens=1):
        B = contexts.shape[0]
        total_sequences = B * num_samples_per_context
        
        if USE_CONTINUOUS_BATCHING and DYNAMIC_BATCH_SIZE:
            seq_len = contexts.shape[1] + max_tokens
            memory_per_seq = seq_len * N_EMB * 4 * 3
            
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            max_sequences = min(total_sequences, int(available_memory * 0.95 / memory_per_seq))
            
            if max_sequences < total_sequences:
                return self._generate_chunked(contexts, num_samples_per_context, temperature, max_tokens, max_sequences)
        
        sequences = contexts.repeat_interleave(num_samples_per_context, dim=0)
        
        done = torch.zeros(total_sequences, dtype=torch.bool, device=self.device)
        generated_tokens = torch.zeros(total_sequences, max_tokens, dtype=torch.long, device=self.device)
        
        for step in range(max_tokens):
            active_mask = ~done
            if not active_mask.any():
                break
                
            active_sequences = sequences[active_mask]
            
            if active_sequences.shape[1] > self.context_len:
                active_sequences = active_sequences[:, -self.context_len:]
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = self.model(active_sequences)[:, -1, :] / temperature
            
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, 1).squeeze(-1)
            
            generated_tokens[active_mask, step] = next_tokens
            
            sequences = torch.cat([
                sequences,
                torch.zeros(total_sequences, 1, dtype=torch.long, device=self.device)
            ], dim=1)
            sequences[active_mask, -1] = next_tokens
            
            if max_tokens == 1:
                done.fill_(True)
        
        return generated_tokens.reshape(B, num_samples_per_context, max_tokens)
    
    def _generate_chunked(self, contexts, num_samples_per_context, temperature, max_tokens, chunk_size):
        B = contexts.shape[0]
        all_generated = []
        
        for i in range(0, B, chunk_size // num_samples_per_context):
            end_idx = min(i + chunk_size // num_samples_per_context, B)
            chunk_contexts = contexts[i:end_idx]
            
            chunk_generated = self.generate_continuous_batch(
                chunk_contexts, num_samples_per_context, temperature, max_tokens
            )
            all_generated.append(chunk_generated)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(all_generated, dim=0)

@torch.no_grad()
def generate_with_temperature(model, contexts, horizon, K, temperature=1.0):
    B = contexts.shape[0]
    device = contexts.device
    
    ctx = contexts.repeat_interleave(K, dim=0)
    
    for _ in range(horizon):
        ctx_window = ctx[:, -model.context_len:] if ctx.shape[1] > model.context_len else ctx
        
        with torch.amp.autocast('cuda',dtype = torch.bfloat16):
            logits = model(ctx_window)[:, -1, :] / temperature
        
        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, 1)
        ctx = torch.cat([ctx, next_char], dim=1)
    
    generated = ctx[:, -horizon:].reshape(B, K, horizon)
    return generated

@torch.no_grad()
def generate_exhaustive_single_char(model, contexts, V):
    B = contexts.shape[0]
    device = contexts.device
    
    ctx_window = contexts[:, -model.context_len:] if contexts.shape[1] > model.context_len else contexts
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(ctx_window)[:, -1, :]
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    all_chars = torch.arange(V, device=device).unsqueeze(0).expand(B, -1)
    
    return all_chars, log_probs

def compute_rewards(gen, ref, ref_logits):
    exact_match = (gen == ref).float()
    
    with torch.no_grad():
        ref_probs = F.softmax(ref_logits, dim=-1)
        B, T = gen.shape
        gen_probs = ref_probs.gather(2, gen.unsqueeze(-1)).squeeze(-1)
        partial_reward = gen_probs
    
    reward = exact_match + partial_reward
    
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
    
    base_rewards = torch.zeros(B, V, device=device)
    
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
    
    if USE_CONFIDENCE_SCALING and old_probs is not None:
        confidence = old_probs[torch.arange(B), ref_char]
        confidence_scale = 1.0 + CONFIDENCE_WEIGHT * confidence.clamp(0, 1)
        confidence_scale = confidence_scale.clamp(1.0, CONFIDENCE_CLIP)
        rewards[torch.arange(B), ref_char] *= confidence_scale
        
        if ENABLE_CONFIDENCE_PENALTY:
            top_k = min(10, V)
            top_probs, top_indices = old_probs.topk(top_k, dim=-1)
            wrong_mask = top_indices != ref_char.unsqueeze(1)
            confident_mask = top_probs > 0.5
            penalty_mask = wrong_mask & confident_mask
            penalties = -0.1 * top_probs * penalty_mask.float()
            rewards.scatter_add_(1, top_indices, penalties)
    
    rewards = rewards - rewards.mean(dim=1, keepdim=True)
    
    return rewards


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
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.98)
    
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
    print(f"Using On-the-fly N-grams")
    print(f"Training for {EPOCHS} epochs ({TOTAL_ITERS:,} iterations, {ITERS_PER_EPOCH:,} iters/epoch)")
    print(f"Using Continuous Batching: {USE_CONTINUOUS_BATCHING}")
    print(f"Dynamic Batch Size: {DYNAMIC_BATCH_SIZE}")
    print(f"Using Paged Attention: {USE_PAGED_ATTENTION} (page size: {PAGE_SIZE})")
    
    print("Building on-the-fly n-gram reference model...")
    ref = OnTheFlyNGramRef(text, stoi, V).to(DEV).eval()
    print("N-gram reference model initialized")
    
    actor = GPT(V).to(DEV)
    actor_old = GPT(V).to(DEV)
    
    continuous_generator = None
    if USE_CONTINUOUS_BATCHING:
        print("Initializing continuous batch generator...")
        continuous_generator = ContinuousBatchGenerator(actor_old, CONTEXT_LEN, V, device=DEV)
        print("Continuous batch generator initialized")
    
    
    actor_old.load_state_dict(actor.state_dict())
    actor_old.eval()
    
    param_count = sum(p.numel() for p in actor.parameters())
    print(f"Total parameters: {param_count:,}")
    
    opt = torch.optim.AdamW(actor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    
    scheduler = None
    if USE_LR_DECAY:
        if LR_DECAY_TYPE == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=TOTAL_ITERS - WARMUP_ITERS, eta_min=MIN_LR
            )
        elif LR_DECAY_TYPE == "linear":
            def linear_schedule(step):
                if step < WARMUP_ITERS:
                    return step / WARMUP_ITERS
                else:
                    progress = (step - WARMUP_ITERS) / (TOTAL_ITERS - WARMUP_ITERS)
                    return (1 - progress) * (1 - MIN_LR/LR) + MIN_LR/LR
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=linear_schedule)
        elif LR_DECAY_TYPE == "exponential":
            decay_rate = (MIN_LR / LR) ** (1 / (TOTAL_ITERS - WARMUP_ITERS))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay_rate)
    
    scaler = GradScaler(enabled=USE_AMP)
    
    loader = ContextualTextLoader(text, ENC, MICRO_BATCH, HORIZON, CONTEXT_LEN, device=DEV)
    kl_controller = AdaptiveKLController(KL_TARGET) if ADAPTIVE_KL else None
    
    confidence_monitor = ConfidenceMonitor() if USE_CONFIDENCE_SCALING else None
    
    run = wandb.init(
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
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
                "use_on_the_fly_ngrams": True,
            },
            "optimizations": {
                "flash_attention": USE_FLASH_ATTN,
                "mixed_precision": USE_AMP,
                "amp_dtype": AMP_DTYPE,
                "continuous_batching": USE_CONTINUOUS_BATCHING,
                "dynamic_batch_size": DYNAMIC_BATCH_SIZE,
                "max_batch_tokens": MAX_BATCH_TOKENS,
                "paged_attention": USE_PAGED_ATTENTION,
                "page_size": PAGE_SIZE,
            }
        }
    )
    
    stream = torch.cuda.Stream()
    
    chars_seen = 0
    current_kl_coef = 0.0
    
    for it in tqdm(range(1, TOTAL_ITERS+1), desc="Training"):
        actor.train()
        opt.zero_grad(set_to_none=True)
        
        total_loss = 0.0
        all_metrics = defaultdict(list)
        
        all_R = []
        all_adv = []
        
        for accum_step in range(GRAD_ACCUM):
            ctx, ref_tok = loader.next()
            
            old_probs = None
            if USE_CONFIDENCE_SCALING:
                with torch.no_grad():
                    old_logits = actor_old(ctx[:, -actor_old.context_len:])[:, -1, :]
                    old_probs = F.softmax(old_logits, dim=-1)
            
            if USE_EXHAUSTIVE and HORIZON == 1:
                with torch.no_grad():
                    if USE_CONTINUOUS_BATCHING and continuous_generator is not None:
                        G = continuous_generator.generate_continuous_batch(
                            ctx, K_SAMPLES, temperature=TEMPERATURE, max_tokens=HORIZON
                        )
                    else:
                        G = generate_with_temperature(actor_old, ctx, HORIZON, K_SAMPLES, temperature=TEMPERATURE)
                
                with torch.no_grad():
                    model_logits = actor_old(ctx[:, -actor_old.context_len:])[:, -1, :]
                    model_pred = model_logits.argmax(dim=-1)
                    model_accuracy = (model_pred == ref_tok[:, 0]).float().mean().item()
                    all_metrics["accuracy"].append(model_accuracy)
                    
                    ref_full = torch.cat([ctx, ref_tok], dim=1)
                    _, components = ref(ref_full[:, -HORIZON-1:], return_components=True)
                    
                    bigram_logits = components['bigram'][:, -1, :]
                    bigram_pred = bigram_logits.argmax(dim=-1)
                    bigram_correct = (bigram_pred == ref_tok[:, 0]).float().mean().item()
                    all_metrics["bigram_accuracy"].append(bigram_correct)
                    
                    if CONTEXT_LEN >= 2:
                        trigram_logits = components['trigram'][:, -1, :]
                        trigram_pred = trigram_logits.argmax(dim=-1)
                        trigram_correct = (trigram_pred == ref_tok[:, 0]).float().mean().item()
                        all_metrics["trigram_accuracy"].append(trigram_correct)
                    
                    if CONTEXT_LEN >= 3:
                        fourgram_logits = components['fourgram'][:, -1, :]
                        fourgram_pred = fourgram_logits.argmax(dim=-1)
                        fourgram_correct = (fourgram_pred == ref_tok[:, 0]).float().mean().item()
                        all_metrics["fourgram_accuracy"].append(fourgram_correct)
                
                R = torch.zeros_like(G, dtype=torch.float32)
                
                for k in range(K_SAMPLES):
                    gen = G[:, k, :]
                    
                    with torch.no_grad():
                        full_seq = torch.cat([ctx, gen], dim=1)
                        ref_logits = ref(full_seq[:, -HORIZON-1:])[:, -HORIZON:]
                    
                    reward, _, _ = compute_rewards_batch(gen, ref_tok, ref_logits)
                    R[:, k, :] = reward
                
                base = R.mean(dim=1, keepdim=True)
                adv = R - base
                adv_std = torch.maximum(adv.std(dim=1, keepdim=True), torch.tensor(MIN_VARIANCE, device=DEV))
                adv = adv / adv_std
                
                all_R.append(R)
                all_adv.append(adv)
                
                flat_G = G.reshape(-1, HORIZON)
                flat_adv = adv.reshape(-1, HORIZON)
                
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
            
            ctx_for_loss = ctx.repeat_interleave(K_SAMPLES, dim=0)
            full_seq = torch.cat([ctx_for_loss, flat_G], dim=1)
            
            with torch.amp.autocast('cuda',dtype = torch.bfloat16):
                input_seq = full_seq[:, -actor.context_len:]
                
                if torch.isnan(input_seq).any():
                    print(f"ERROR: NaN in input_seq at iteration {it}, accum_step {accum_step}")
                    print(f"Input shape: {input_seq.shape}")
                    raise ValueError("NaN detected in input sequence")
                
                new_logits = actor(input_seq)
                new_logits = new_logits[:, -HORIZON:]
                
                if torch.isnan(new_logits).any() or torch.isinf(new_logits).any():
                    print(f"WARNING: NaN/Inf in new_logits at iteration {it}, accum_step {accum_step}")
                    print(f"Logits shape: {new_logits.shape}")
                    print(f"Contains NaN: {torch.isnan(new_logits).any()}, Contains Inf: {torch.isinf(new_logits).any()}")
                    
                    for name, param in actor.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            print(f"NaN/Inf in parameter: {name}")
                    
                    new_logits = torch.nan_to_num(new_logits, nan=0.0, posinf=50.0, neginf=-50.0)
                    new_logits = torch.clamp(new_logits, min=-100.0, max=100.0)
                
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
                
                
                ref_logits = ref(full_seq[:, -HORIZON-1:])[:, -HORIZON:]
                
                if torch.isnan(ref_logits).any() or torch.isinf(ref_logits).any():
                    print(f"ERROR: NaN/Inf in ref_logits at iteration {it}")
                    print(f"Ref logits shape: {ref_logits.shape}")
                    if hasattr(ref, 'bigram_scores'):
                        print(f"Bigram scores has NaN: {torch.isnan(ref.bigram_scores).any()}")
                    if hasattr(ref, 'trigram_scores'):
                        print(f"Trigram scores has NaN: {torch.isnan(ref.trigram_scores).any()}")
                    if hasattr(ref, 'fourgram_scores'):
                        print(f"Fourgram scores has NaN: {torch.isnan(ref.fourgram_scores).any()}")
                    raise ValueError("NaN/Inf detected in reference model output")
                
                true_probs = torch.zeros_like(new_logits)
                true_probs.scatter_(2, flat_G.unsqueeze(-1), 1.0)
                
                new_probs = F.softmax(new_logits, dim=-1)
                true_char_probs = new_probs.gather(2, flat_G.unsqueeze(-1)).squeeze(-1)
                kl_divergence = -torch.log(true_char_probs + 1e-8).mean()
                
                kl = torch.clamp(kl_divergence, max=5.0)
                
                if ADAPTIVE_KL and kl_controller and accum_step == 0:
                    current_kl_coef = kl_controller.update(kl.item(), current_kl_coef)
                else:
                    kl_free_iters = int(TOTAL_ITERS * KL_FREE_FRACTION)
                    
                    if it < kl_free_iters:
                        current_kl_coef = 0.0
                    else:
                        chars_after_kl_free = chars_seen - (kl_free_iters * BATCH * HORIZON)
                        total_chars = (TOTAL_ITERS - kl_free_iters) * BATCH * HORIZON
                        
                        if chars_after_kl_free < KL_WARM:
                            warmup_progress = chars_after_kl_free / KL_WARM
                            cosine_factor = 0.5 * (1 - math.cos(math.pi * warmup_progress))
                            current_kl_coef = BETA_KL * cosine_factor
                        else:
                            remaining_chars = chars_after_kl_free - KL_WARM
                            remaining_total = total_chars - KL_WARM
                            if remaining_total > 0:
                                extended_progress = remaining_chars / remaining_total
                                cosine_factor = 0.5 * (1 + math.cos(math.pi * (1 + extended_progress)))
                                current_kl_coef = BETA_KL * cosine_factor
                            else:
                                current_kl_coef = BETA_KL
                
                loss = (pol_loss + current_kl_coef * kl) / GRAD_ACCUM
            
            scaler.scale(loss).backward()
            total_loss += loss.item()
            
            all_metrics["kl"].append(kl.item())
            all_metrics["pol_loss"].append(pol_loss.item())
            all_metrics["ratio"].append(ratio.mean().item())
            all_metrics["ratio_max"].append(ratio.max().item())
        
        scaler.unscale_(opt)
        
        has_nan_grad = False
        for name, param in actor.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"NaN/Inf gradient detected in {name}")
                    has_nan_grad = True
        
        if has_nan_grad:
            print(f"Skipping optimizer step due to NaN gradients at iteration {it}")
            opt.zero_grad(set_to_none=True)
            scaler.update()
            continue
        
        grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
        
        if grad_norm > 100.0:
            print(f"Warning: Large gradient norm {grad_norm} at iteration {it}")
        
        scaler.step(opt)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        chars_seen += BATCH * HORIZON
        
        if it % 5 == 0:
            actor_old.load_state_dict(actor.state_dict())
            actor_old.eval()
            if USE_CONTINUOUS_BATCHING and continuous_generator is not None:
                continuous_generator.model = actor_old
        
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
                "pol_loss": all_metrics.get("pol_loss", [0])[-1] if all_metrics.get("pol_loss") else 0,
                "accuracy": np.mean(all_metrics.get("accuracy", [0])),
                "chars": chars_seen,
                "epoch": current_epoch,
                "learning_rate": opt.param_groups[0]['lr'],
            }, step=chars_seen)
            
            
            if USE_CONFIDENCE_SCALING and old_probs is not None:
                ref_first_char = ref_tok[:, 0]
                correct_confidence = old_probs[torch.arange(MICRO_BATCH), ref_first_char].mean().item()
                confidence_scale_avg = (1.0 + CONFIDENCE_WEIGHT * old_probs[torch.arange(MICRO_BATCH), ref_first_char].clamp(0, 1)).mean().item()
                wandb.log({
                    "confidence/mean": correct_confidence,
                    "confidence/reward_boost": confidence_scale_avg - 1.0,
                }, step=chars_seen)
        
        if it % SAMPLE_INTERVAL == 0:
            with torch.no_grad():
                eval_ctx, eval_ref = loader.next()
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    eval_logits = actor(eval_ctx)[:, -1, :]
                
                predicted_chars = eval_logits.argmax(dim=-1)
                correct = (predicted_chars == eval_ref[:, 0]).float()
                top1_accuracy = correct.mean().item()
                
                _, top5_preds = eval_logits.topk(5, dim=-1)
                top5_correct = (top5_preds == eval_ref[:, 0].unsqueeze(1)).any(dim=1).float()
                top5_accuracy = top5_correct.mean().item()
                
                probs = F.softmax(eval_logits, dim=-1)
                pred_confidence = probs.gather(1, predicted_chars.unsqueeze(1)).squeeze().mean().item()
                
                ref_logits = ref(eval_ctx)[:, -1, :]
                ref_predicted = ref_logits.argmax(dim=-1)
                ref_correct = (ref_predicted == eval_ref[:, 0]).float().mean().item()
                
                sampling_correct_count = 0
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    sample_logits = eval_logits / TEMPERATURE
                    sample_probs = F.softmax(sample_logits, dim=-1)
                    
                    for _ in range(K_SAMPLES):
                        sampled_chars = torch.multinomial(sample_probs, 1).squeeze(-1)
                        sampling_correct_count += (sampled_chars == eval_ref[:, 0]).float().sum().item()
                
                sampling_based_accuracy = sampling_correct_count / (MICRO_BATCH * K_SAMPLES)
                
                num_samples = 50
                context_len_for_eval = CONTEXT_LEN
                gen_len = HORIZON
                
                eval_contexts = []
                start_positions = []
                for _ in range(num_samples):
                    start_pos = torch.randint(context_len_for_eval, len(loader.data) - gen_len, (1,)).item()
                    context = loader.data[start_pos - context_len_for_eval:start_pos]
                    eval_contexts.append(context)
                    start_positions.append(start_pos)
                
                eval_contexts = torch.stack(eval_contexts).to(DEV)
                
                sample_data = []
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    for i in range(num_samples):
                        ctx = eval_contexts[i:i+1]
                        start_pos = start_positions[i]
                        
                        
                        generated = []
                        for _ in range(gen_len):
                            if len(generated) > 0:
                                ctx_with_gen = torch.cat([ctx, torch.tensor(generated, device=DEV).unsqueeze(0)], dim=1)
                                logits = actor(ctx_with_gen[:, -actor.context_len:])[:, -1, :] / EVAL_TEMPERATURE
                            else:
                                logits = actor(ctx[:, -actor.context_len:])[:, -1, :] / EVAL_TEMPERATURE
                            
                            probs = F.softmax(logits, dim=-1)
                            next_char = torch.multinomial(probs, 1).squeeze(-1)
                            generated.append(next_char.item())
                        
                        context_text = DEC(ctx[0].tolist())
                        generated_text = DEC(generated)
                        true_start = start_pos
                        true_text = DEC(loader.data[true_start:true_start + gen_len].tolist())
                        
                        correct = generated_text == true_text
                        sample_data.append([i+1, context_text, generated_text, true_text, "✓" if correct else "✗"])
                
                num_correct_sequences = sum(1 for row in sample_data if row[4] == "✓")
                sequence_accuracy = num_correct_sequences / num_samples
                
                sample_table = wandb.Table(
                    columns=["Sample", f"Context ({context_len_for_eval} chars)", f"Generated ({gen_len} chars)", f"True ({gen_len} chars)", "Correct"],
                    data=sample_data
                )
                
                wandb.log({
                    "eval/next_char_accuracy": top1_accuracy,
                    "eval/top5_accuracy": top5_accuracy,
                    "eval/sampling_based_accuracy": sampling_based_accuracy,
                    "eval/prediction_confidence": pred_confidence,
                    "eval/ref_model_accuracy": ref_correct,
                    "eval/accuracy_improvement": top1_accuracy - ref_correct,
                    f"eval/{gen_len}char_sequence_accuracy": sequence_accuracy,
                    f"eval/{gen_len}char_correct_count": num_correct_sequences,
                    "text_samples": sample_table
                }, step=chars_seen)
        
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
                
                if USE_EXHAUSTIVE and HORIZON == 1:
                    all_chars, log_probs = generate_exhaustive_single_char(actor, val_ctx, V)
                    val_probs = torch.exp(log_probs)
                    rewards = compute_exhaustive_rewards(all_chars, val_ref[:, 0], ref, val_ctx, V, 
                                                       model_log_probs=log_probs,
                                                       old_probs=val_probs if USE_CONFIDENCE_SCALING else None)
                    
                    best_reward_chars = rewards.argmax(dim=-1)
                    reward_based_accuracy = (best_reward_chars == val_ref[:, 0]).float().mean().item()
                    
                    correct_rewards = rewards[torch.arange(val_ctx.shape[0]), val_ref[:, 0]]
                    avg_correct_reward = correct_rewards.mean().item()
                    
                    wandb.log({
                        "eval/reward_based_accuracy": reward_based_accuracy,
                        "eval/avg_correct_reward": avg_correct_reward,
                        "eval/reward_variance": rewards.var().item()
                    }, step=chars_seen)
                
                wandb.log({
                    "eval/perplexity": perplexity.item(),
                    "eval/loss": val_loss.item(),
                }, step=chars_seen)
    
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