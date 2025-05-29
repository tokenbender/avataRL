#!/usr/bin/env python
"""
GRPO tiny-Shakespeare with EXHAUSTIVE rollouts (H100 Optimized)
Reference = frozen bigram counts
Launch: modal run grpo_bigram_exhaustive_h100.py

Key Features:
1. Exhaustive first character sampling (all 65 characters)
2. Optimized batch size for H100 80GB memory
3. Fixed all GRPO bugs from original implementation
4. Complete winning signal for every character at each position
"""

# ─── hyper-params optimized for H100 exhaustive search ──────────────────
HORIZON = 70                      # Characters per rollout
BATCH = 64                        # Reduced for 65x rollouts (64 * 65 = 4,160 sequences)
TOTAL_ITERS = 200                 # Fewer needed with exhaustive search
LR = 1e-5                         
BETA_KL = 1e-3                    
KL_WARM = BATCH * HORIZON * 10    # Warmup for 10 iterations
NEG_REWARD = False                
PARTIAL_LP = 0.0                  
GPU_TYPE = "H100"                 
CLIP_RATIO = 0.2                  

# ─── Modal plumbing ──────────────────────────────────────────────────────
import math, requests, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import modal, wandb, numpy as np, copy

stub = modal.App("grpo-char-bigram-exhaustive-h100")
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

class TextLoader:
    def __init__(self, text, enc, B, T):
        self.data = enc(text)
        self.B, self.T, self.pos = B, T, 0
    def next(self):
        span = self.B*self.T + 1
        if self.pos + span > len(self.data): self.pos = 0
        buf = self.data[self.pos:self.pos+span]; self.pos += span
        return buf[:-1].view(self.B,self.T), buf[1:].view(self.B,self.T)

# ─── Actor model (tiny GPT) ──────────────────────────────────────────────
class TinyGPT(nn.Module):
    def __init__(self, V, n_layer=6, n_head=6, n_emb=384, blk=256):
        super().__init__()
        self.wte = nn.Embedding(V,n_emb); self.wpe = nn.Embedding(blk,n_emb)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(n_emb,n_head,4*n_emb,activation="gelu",batch_first=True)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb,V,bias=False); self.head.weight = self.wte.weight
    def forward(self, idx):
        B,T = idx.shape; pos = torch.arange(T,device=idx.device)
        x = self.wte(idx)+self.wpe(pos)
        for l in self.layers: x = l(x)
        return self.head(self.ln_f(x))

# ─── Bigram reference (logits = log P) ───────────────────────────────────
class BigramRef(nn.Module):
    def __init__(self, counts):
        super().__init__()
        probs = (counts + 1) / (counts.sum(1,keepdims=True)+counts.size(1))
        self.register_buffer("logits", torch.log(probs))
    def forward(self, idx):
        last = idx.clone()
        B,T = last.shape
        out = self.logits[last]
        return out

# ─── rewards / generation ───────────────────────────────────────────────
def tok_reward(gen, ref, neg=False):
    m = (gen==ref).float(); return m*2-1 if neg else m
def add_partial(r,lp,c): return r if c==0 else r + c*lp.detach()

@torch.no_grad()
def generate_forced_first(m, ctx, first_char_idx, steps, V):
    """Generate with a specific first character, then continue normally"""
    B = ctx.shape[0]
    # Force the first character (broadcast across batch)
    first_char = torch.full((B, 1), first_char_idx, device=ctx.device, dtype=torch.long)
    ctx = torch.cat([ctx, first_char], 1)
    
    # Generate the remaining characters normally
    for _ in range(steps - 1):
        nxt = torch.multinomial(torch.softmax(m(ctx)[:,-1],-1),1)
        ctx = torch.cat([ctx,nxt],1)
    return ctx[:,-steps:]

@torch.no_grad()
def generate_exhaustive_rollouts(m, ctx0, V, steps):
    """Generate V rollouts, each starting with a different character"""
    B = ctx0.shape[0]
    all_gens = []
    
    # Try every possible first character
    for char_idx in range(V):
        gen = generate_forced_first(m, ctx0.clone(), char_idx, steps, V)
        all_gens.append(gen)
    
    # Stack: (B, V, HORIZON)
    return torch.stack(all_gens, dim=1)

# ─── training loop ───────────────────────────────────────────────────────
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
    ENC,DEC,V,stoi,itos,text = build_vocab()
    
    print(f"Vocabulary size: {V}")
    print(f"Batch size: {BATCH}")
    print(f"Total sequences per iteration: {BATCH * V}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # bigram counts
    big = torch.zeros(V,V,dtype=torch.float32)
    for a,b in zip(text, text[1:]): big[stoi[a],stoi[b]] += 1
    ref = BigramRef(big).to(DEV).eval()

    actor = TinyGPT(V).to(DEV)
    actor_old = copy.deepcopy(actor).eval()
    opt   = torch.optim.AdamW(actor.parameters(), lr=LR)
    loader= TextLoader(text, ENC, BATCH, HORIZON+1)

    run = wandb.init(project="gpt2-grpo-char-exhaustive-h100",
                     config=dict(ref="bigram", horizon=HORIZON, beta=BETA_KL,
                                batch=BATCH, vocab_size=V, mode="exhaustive"))
    chars_seen=0
    
    for it in tqdm(range(1,TOTAL_ITERS+1),leave=False):
        ctx0, ref_tok = loader.next()
        ctx0, ref_tok = ctx0[:,:1].to(DEV), ref_tok[:,:HORIZON].to(DEV)
        
        # Generate ALL V rollouts from OLD policy (exhaustive first character)
        with torch.no_grad():
            G = generate_exhaustive_rollouts(actor_old, ctx0, V, HORIZON)  # [B, V, T]
            
            # Calculate rewards for all rollouts
            R = []
            for v in range(V):
                gen = G[:, v, :]  # [B, T]
                
                # Get log probs under old policy
                old_logits = actor_old(gen)
                old_dist = torch.distributions.Categorical(logits=old_logits)
                old_logp = old_dist.log_prob(gen)
                
                # Calculate rewards
                r = tok_reward(gen, ref_tok, NEG_REWARD)
                r = add_partial(r, old_logp, PARTIAL_LP)
                R.append(r)
            
            R = torch.stack(R, 1)  # [B, V, T]
        
        # CORRECT Group-relative advantage calculation
        # Mean across ALL V characters (complete action space)
        base = R.mean(dim=1, keepdim=True)  # [B, 1, T]
        adv = R - base  # [B, V, T]
        
        # Normalize advantages per batch item
        adv_std = adv.std(dim=1, keepdim=True) + 1e-8
        adv = adv / adv_std
        
        # Flatten for loss calculation
        flatG = G.reshape(-1, HORIZON)    # [B*V, T]
        flatA = adv.reshape(-1, HORIZON)  # [B*V, T]
        
        # Calculate log probs under NEW policy
        new_logits = actor(flatG)
        new_dist = torch.distributions.Categorical(logits=new_logits)
        logp_new = new_dist.log_prob(flatG)
        
        # Calculate log probs under OLD policy (for ratio)
        with torch.no_grad():
            old_logits = actor_old(flatG)
            old_dist = torch.distributions.Categorical(logits=old_logits)
            logp_old = old_dist.log_prob(flatG)
        
        # Calculate ratio for PPO-style clipping
        ratio = torch.exp(logp_new - logp_old)
        clipped_ratio = torch.clamp(ratio, 1-CLIP_RATIO, 1+CLIP_RATIO)
        
        # Policy loss with clipping
        pol_loss1 = -flatA * ratio
        pol_loss2 = -flatA * clipped_ratio
        pol_loss = torch.max(pol_loss1, pol_loss2).mean()
        
        # CORRECT KL divergence calculation
        ref_logits = ref(flatG)
        ref_dist = torch.distributions.Categorical(logits=ref_logits)
        kl = torch.distributions.kl.kl_divergence(new_dist, ref_dist).mean()
        
        # KL warmup schedule
        kl_coef = min(BETA_KL, BETA_KL*chars_seen/max(1,KL_WARM))
        
        # Total loss
        loss = pol_loss + kl_coef*kl

        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0); opt.step()
        chars_seen += BATCH*HORIZON
        
        # Update old policy periodically
        if it % 10 == 0:
            actor_old = copy.deepcopy(actor).eval()

        if it%5==0:
            # Find which character gets highest average reward
            avg_reward_per_char = R.mean(dim=(0, 2))  # Average over batch and sequence
            best_char_idx = avg_reward_per_char.argmax().item()
            worst_char_idx = avg_reward_per_char.argmin().item()
            
            # Get character distribution for logging
            char_rewards = {itos[i]: avg_reward_per_char[i].item() for i in range(V)}
            top_5_chars = sorted(char_rewards.items(), key=lambda x: x[1], reverse=True)[:5]
            
            wandb.log({
                "reward": R.mean().item(),
                "kl": kl.item(),
                "kl_coef": kl_coef,
                "chars": chars_seen,
                "ratio": ratio.mean().item(),
                "best_first_char": itos[best_char_idx],
                "worst_first_char": itos[worst_char_idx],
                "best_char_reward": avg_reward_per_char[best_char_idx].item(),
                "worst_char_reward": avg_reward_per_char[worst_char_idx].item(),
                "reward_spread": (avg_reward_per_char.max() - avg_reward_per_char.min()).item(),
            }, step=chars_seen)
            
        if it%20==0:
            # Generate samples starting with different characters
            print(f"\nIteration {it} samples:")
            sample_chars = [best_char_idx, 
                          avg_reward_per_char.argsort(descending=True)[V//2].item(),  # Median
                          worst_char_idx]
            
            for idx in sample_chars:
                samp = DEC(generate_forced_first(actor, ctx0[:1], idx, 120, V)[0])
                reward_pct = avg_reward_per_char[idx].item() * 100
                wandb.log({f"sample_{itos[idx]}": wandb.Html(f"<pre>'{itos[idx]}' ({reward_pct:.1f}%): {samp}</pre>")}, 
                         step=chars_seen)

    run.finish()

# ─── local entry ─────────────────────────────────────────────────────────
@stub.local_entrypoint()
def main():
    train_remote.remote()

if __name__ == "__main__":
    main()