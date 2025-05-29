#!/usr/bin/env python
"""
GRPO with Guided Exhaustive Search - Fixed Training Dynamics
- Entropy bonus to prevent collapse
- Larger clip ratio for better exploration
- Temperature-based sampling
- Better reward scaling
"""

# ─── hyper-params ────────────────────────────────────────────────────────
CONTEXT_LEN = 100                 # Sliding window context length
HORIZON = 70                      # Characters to generate
BATCH = 32                        # Batch size
EXHAUSTIVE_RATIO = 0.2           # Only 20% exhaustive
TOTAL_ITERS = 500                
LR = 3e-5                         # Increased learning rate
BETA_KL = 1e-3                    # Reduced KL penalty
KL_WARM = 50_000                  # Faster warmup
NEG_REWARD = False                
GPU_TYPE = "H100"                 
CLIP_RATIO = 0.5                  # Increased for more exploration
K_SAMPLES = 4                     # Fixed K for all generations
ENTROPY_COEF = 0.01               # Entropy bonus
TEMPERATURE = 1.2                 # Sampling temperature
MIN_VARIANCE = 0.1                # Minimum advantage variance

# ─── imports ─────────────────────────────────────────────────────────────
import requests, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import modal, wandb, copy
from collections import defaultdict
import numpy as np

stub = modal.App("grpo-guided-exhaustive-fixed-v2")
image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "torch==2.3.0", "tqdm", "wandb", "requests")
)

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt")

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

class ContextualTextLoader:
    """Provides longer context windows for better prediction"""
    def __init__(self, text, enc, B, T, context_len):
        self.data = enc(text)
        self.B, self.T, self.context_len = B, T, context_len
        self.pos = context_len  # Start after we have enough context
        
    def next(self):
        # Ensure we have enough context
        if self.pos + self.B*self.T + 1 > len(self.data):
            self.pos = self.context_len
            
        contexts = []
        targets = []
        
        for b in range(self.B):
            # Get context window
            ctx_start = self.pos - self.context_len + b*self.T
            ctx_end = self.pos + b*self.T
            
            context = self.data[ctx_start:ctx_end]
            target = self.data[ctx_end:ctx_end + self.T]
            
            contexts.append(context)
            targets.append(target)
        
        self.pos += self.B * self.T
        
        return torch.stack(contexts), torch.stack(targets)

# ─── Enhanced GPT with better initialization ─────────────────────────────
class ContextualGPT(nn.Module):
    def __init__(self, V, n_layer=6, n_head=8, n_emb=384, context_len=100):
        super().__init__()
        self.context_len = context_len
        self.wte = nn.Embedding(V, n_emb)
        self.wpe = nn.Embedding(context_len + HORIZON, n_emb)
        self.drop = nn.Dropout(0.1)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                n_emb, n_head, 4*n_emb, 
                dropout=0.1, activation="gelu", 
                batch_first=True
            ) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, V, bias=False)
        self.head.weight = self.wte.weight  # Weight tying
        
        # Better initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        
        tok_emb = self.drop(self.wte(idx))
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        
        # Create causal mask
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        
        for layer in self.layers:
            x = layer(x, src_mask=mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ─── Bigram reference (simpler than 3-gram) ──────────────────────────────
class BigramRef(nn.Module):
    def __init__(self, bigram_counts, V, smoothing=1.0):
        super().__init__()
        self.V = V
        
        # Build bigram probability table
        self.probs = torch.ones(V, V) * smoothing
        
        for (c1, c2), count in bigram_counts.items():
            self.probs[c1, c2] += count
            
        # Normalize
        self.probs = self.probs / self.probs.sum(dim=1, keepdim=True)
        
        # Convert to log space
        self.register_buffer("log_probs", torch.log(self.probs + 1e-10))
        
    def forward(self, idx):
        B, T = idx.shape
        logits = torch.zeros(B, T, self.V, device=idx.device)
        
        # First position: uniform
        logits[:, 0, :] = 0.0
        
        # Rest: use bigram
        for t in range(1, T):
            prev_char = idx[:, t-1]
            logits[:, t, :] = self.log_probs[prev_char, :]
            
        return logits

def build_bigram_counts(text, stoi):
    """Build bigram count table"""
    counts = defaultdict(int)
    
    for i in range(len(text) - 1):
        c1 = stoi[text[i]]
        c2 = stoi[text[i+1]]
        counts[(c1, c2)] += 1
        
    return counts

# ─── Temperature-based generation ─────────────────────────────────────────
@torch.no_grad()
def generate_with_temperature(model, contexts, horizon, K, temperature=1.0):
    """
    Generate K samples with temperature-based sampling
    """
    B = contexts.shape[0]
    device = contexts.device
    
    # Expand contexts for K samples
    ctx = contexts.repeat_interleave(K, dim=0)  # [B*K, context_len]
    
    # Generate with temperature
    for _ in range(horizon):
        logits = model(ctx[:, -model.context_len:])[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, 1)
        ctx = torch.cat([ctx, next_char], dim=1)
    
    # Reshape to [B, K, horizon]
    generated = ctx[:, -horizon:].reshape(B, K, horizon)
    return generated

def compute_entropy(logits):
    """Compute entropy to measure model uncertainty"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

# ─── Improved rewards ────────────────────────────────────────────────────
def compute_rewards(gen, ref, ref_logits):
    """
    Compute rewards with better scaling
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

# ─── Training loop ───────────────────────────────────────────────────────
@stub.function(
    gpu=GPU_TYPE, 
    image=image, 
    timeout=60*60*3,
    secrets=[modal.Secret.from_name("wandb")]
)
def train_remote():
    torch.set_float32_matmul_precision("high")
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dataset()
    ENC, DEC, V, stoi, itos, text = build_vocab()
    
    print(f"Vocabulary size: {V}")
    print(f"Context length: {CONTEXT_LEN}")
    print(f"Batch size: {BATCH}")
    print(f"K samples: {K_SAMPLES}")
    print(f"Temperature: {TEMPERATURE}")
    
    # Build bigram reference (simpler than 3-gram)
    print("Building bigram reference model...")
    bigram_counts = build_bigram_counts(text, stoi)
    ref = BigramRef(bigram_counts, V).to(DEV).eval()
    
    # Initialize models
    actor = ContextualGPT(V, context_len=CONTEXT_LEN).to(DEV)
    actor_old = copy.deepcopy(actor).eval()
    opt = torch.optim.AdamW(actor.parameters(), lr=LR, weight_decay=0.01)
    loader = ContextualTextLoader(text, ENC, BATCH, HORIZON, CONTEXT_LEN)
    
    run = wandb.init(
        project="gpt2-grpo-fixed-v2",
        config=dict(
            ref="bigram", context_len=CONTEXT_LEN, 
            horizon=HORIZON, beta=BETA_KL, batch=BATCH,
            k_samples=K_SAMPLES, temperature=TEMPERATURE,
            entropy_coef=ENTROPY_COEF
        )
    )
    
    chars_seen = 0
    
    for it in tqdm(range(1, TOTAL_ITERS+1), leave=False):
        ctx, ref_tok = loader.next()
        ctx, ref_tok = ctx.to(DEV), ref_tok.to(DEV)
        
        # Generate K samples with temperature
        G = generate_with_temperature(
            actor_old, ctx, HORIZON, K_SAMPLES, temperature=TEMPERATURE
        )  # [B, K, horizon]
        
        # Calculate rewards
        R = torch.zeros_like(G, dtype=torch.float32)
        
        for k in range(K_SAMPLES):
            gen = G[:, k, :]  # [B, horizon]
            
            # Get reference logits
            with torch.no_grad():
                full_seq = torch.cat([ctx, gen], dim=1)
                ref_logits = ref(full_seq[:, -HORIZON-1:])[:, -HORIZON:]
            
            R[:, k, :] = compute_rewards(gen, ref_tok, ref_logits)
        
        # Compute advantages with minimum variance
        base = R.mean(dim=1, keepdim=True)
        adv = R - base
        adv_std = adv.std(dim=1, keepdim=True)
        adv_std = torch.maximum(adv_std, torch.tensor(MIN_VARIANCE, device=DEV))
        adv = adv / adv_std
        
        # Flatten for loss computation
        flat_G = G.reshape(-1, HORIZON)
        flat_adv = adv.reshape(-1, HORIZON)
        
        # Forward pass through new policy
        ctx_for_loss = ctx.repeat_interleave(K_SAMPLES, dim=0)
        full_seq = torch.cat([ctx_for_loss, flat_G], dim=1)
        
        new_logits = actor(full_seq[:, -actor.context_len:])[:, -HORIZON:]
        new_dist = torch.distributions.Categorical(logits=new_logits)
        logp_new = new_dist.log_prob(flat_G)
        
        # Get old policy log probs
        with torch.no_grad():
            old_logits = actor_old(full_seq[:, -actor_old.context_len:])[:, -HORIZON:]
            old_dist = torch.distributions.Categorical(logits=old_logits)
            logp_old = old_dist.log_prob(flat_G)
        
        # PPO loss
        ratio = torch.exp(logp_new - logp_old)
        clipped_ratio = torch.clamp(ratio, 1-CLIP_RATIO, 1+CLIP_RATIO)
        
        pol_loss1 = -flat_adv * ratio
        pol_loss2 = -flat_adv * clipped_ratio
        pol_loss = torch.max(pol_loss1, pol_loss2).mean()
        
        # Entropy bonus
        entropy = new_dist.entropy().mean()
        
        # KL from reference
        ref_logits = ref(full_seq[:, -HORIZON-1:])[:, -HORIZON:]
        ref_dist = torch.distributions.Categorical(logits=ref_logits)
        kl = torch.distributions.kl.kl_divergence(new_dist, ref_dist).mean()
        
        # Total loss with KL warmup and entropy bonus
        kl_coef = min(BETA_KL, BETA_KL * chars_seen / max(1, KL_WARM))
        loss = pol_loss + kl_coef * kl - ENTROPY_COEF * entropy
        
        # Optimize
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
        
        chars_seen += BATCH * HORIZON
        
        # Update old policy more frequently
        if it % 5 == 0:
            actor_old = copy.deepcopy(actor).eval()
        
        # Logging
        if it % 5 == 0:
            wandb.log({
                "reward": R.mean().item(),
                "reward_max": R.max().item(),
                "reward_min": R.min().item(),
                "advantage_mean": adv.mean().item(),
                "advantage_std": adv.std().item(),
                "kl": kl.item(),
                "kl_coef": kl_coef,
                "ratio": ratio.mean().item(),
                "ratio_max": ratio.max().item(),
                "entropy": entropy.item(),
                "pol_loss": pol_loss.item(),
                "total_loss": loss.item(),
                "chars": chars_seen,
            }, step=chars_seen)
        
        # Generate samples
        if it % 20 == 0:
            with torch.no_grad():
                # Generate at different temperatures
                test_ctx = ctx[:3]
                samples = []
                
                for i, temp in enumerate([0.8, 1.0, 1.2]):
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
    
    run.finish()

# ─── Local entry point ────────────────────────────────────────────────────
@stub.local_entrypoint()
def main():
    train_remote.remote()

if __name__ == "__main__":
    main()