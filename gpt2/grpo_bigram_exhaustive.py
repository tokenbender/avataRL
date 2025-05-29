#!/usr/bin/env python
"""
GRPO tiny-Shakespeare with EXHAUSTIVE rollouts
Reference = frozen bigram counts
Launch: modal run grpo_bigram_exhaustive.py

Key change: Instead of sampling K=4 rollouts, we generate ALL 65 possible
continuations (one starting with each character in the vocabulary).
This gives us a complete picture of which character leads to best outcomes.
"""

# ─── hyper-params ────────────────────────────────────────────────────────
HORIZON = 70                      # Characters to generate per rollout
BATCH = 8                         # Reduced due to memory (65 rollouts per batch item)
TOTAL_ITERS, LR = 1_000, 1e-5
BETA_KL, KL_WARM = 1e-3, 200_000
NEG_REWARD, PARTIAL_LP = False, 0.0
GPU_TYPE = "A10G"                 # May need larger GPU for 65 rollouts

# ─── Modal plumbing ──────────────────────────────────────────────────────
import math, requests, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import modal, wandb, numpy as np, copy

stub = modal.App("grpo-char-bigram-exhaustive")
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
    def __init__(self, counts):                          # counts: (V,V)
        super().__init__()
        probs = (counts + 1) / (counts.sum(1,keepdims=True)+counts.size(1))
        self.register_buffer("logits", torch.log(probs)) # (V,V)
    def forward(self, idx):                               # idx: (B,T)
        last = idx.clone()
        B,T = last.shape
        out = self.logits[last]                           # (B,T,V)
        return out

# ─── rewards / generation ───────────────────────────────────────────────
def tok_reward(gen, ref, neg=False):
    m = (gen==ref).float(); return m*2-1 if neg else m
def add_partial(r,lp,c): return r if c==0 else r + c*lp.detach()

@torch.no_grad()
def generate_forced_first(m, ctx, first_char, steps):
    """Generate with a specific first character, then continue normally"""
    # Force the first character
    ctx = torch.cat([ctx, first_char.unsqueeze(0)], 1)
    
    # Generate the remaining characters
    for _ in range(steps - 1):
        nxt = torch.multinomial(torch.softmax(m(ctx)[:,-1],-1),1)
        ctx = torch.cat([ctx,nxt],1)
    return ctx[:,-steps:]

@torch.no_grad()
def generate_all_rollouts(m, ctx0, V, steps):
    """Generate V rollouts, each starting with a different character"""
    B = ctx0.shape[0]
    all_gens = []
    
    for char_idx in range(V):
        # Create batch where each item starts with the same character
        first_char = torch.full((B,), char_idx, device=ctx0.device, dtype=torch.long)
        gen = generate_forced_first(m, ctx0.clone(), first_char, steps)
        all_gens.append(gen)
    
    # Stack: (B, V, HORIZON)
    return torch.stack(all_gens, dim=1)

# ─── training loop ───────────────────────────────────────────────────────
@stub.function(
    gpu=GPU_TYPE, image=image, timeout=60*60*3,
    secrets=[modal.Secret.from_name("wandb")]
)
def train_remote():
    torch.set_float32_matmul_precision("high")
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dataset()
    ENC,DEC,V,stoi,itos,text = build_vocab()
    
    # bigram counts
    big = torch.zeros(V,V,dtype=torch.float32)
    for a,b in zip(text, text[1:]): big[stoi[a],stoi[b]] += 1
    ref = BigramRef(big).to(DEV).eval()

    actor = TinyGPT(V).to(DEV)
    opt   = torch.optim.AdamW(actor.parameters(), lr=LR)
    loader= TextLoader(text, ENC, BATCH, HORIZON+1)

    run = wandb.init(project="gpt2-grpo-char-exhaustive",
                     config=dict(ref="bigram",horizon=HORIZON,beta=BETA_KL,
                                k_samples=V, mode="exhaustive"))
    chars_seen=0
    for it in tqdm(range(1,TOTAL_ITERS+1),leave=False):
        ctx0, ref_tok = loader.next()
        ctx0, ref_tok = ctx0[:,:1].to(DEV), ref_tok[:,:HORIZON].to(DEV)
        
        # Generate ALL possible rollouts (one per character)
        G = generate_all_rollouts(actor, ctx0, V, HORIZON)  # (B, V, HORIZON)
        
        # Calculate rewards for all rollouts
        R = []
        for v in range(V):
            gen = G[:, v, :]  # (B, HORIZON)
            lp = torch.distributions.Categorical(logits=actor(gen)).log_prob(gen)
            r  = add_partial(tok_reward(gen,ref_tok,NEG_REWARD), lp, PARTIAL_LP)
            R.append(r)
        R = torch.stack(R, 1)  # (B, V, HORIZON)
        
        # Advantage calculation (compare all V options)
        base = R.mean(1,keepdim=True)  # Mean across all V characters
        adv = (R-base)
        adv = (adv-adv.mean())/(adv.std()+1e-8)

        # Flatten for loss calculation
        flatG, flatA = G.reshape(-1,HORIZON), adv.reshape(-1,HORIZON)
        logp_new = torch.distributions.Categorical(logits=actor(flatG)).log_prob(flatG)
        logp_ref = torch.distributions.Categorical(logits=ref(flatG)).log_prob(flatG)
        kl = (logp_new - logp_ref).mean()
        KL_WARMUP_TOKENS = 200000
        kl_coef = min(BETA_KL, BETA_KL*chars_seen/max(1,KL_WARMUP_TOKENS))
        loss = -(flatA*logp_new).mean() + kl_coef*kl

        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0); opt.step()
        chars_seen += BATCH*HORIZON

        if it%10==0:
            # Find which character gets highest average reward
            avg_reward_per_char = R.mean(dim=(0, 2))  # Average over batch and sequence
            best_char_idx = avg_reward_per_char.argmax().item()
            best_char = itos[best_char_idx]
            
            wandb.log({"reward":R.mean().item(),"kl":kl.item(),"kl_coef":kl_coef,
                       "chars":chars_seen, "best_first_char": best_char,
                       "best_char_reward": avg_reward_per_char[best_char_idx].item()},
                      step=chars_seen)
        if it%200==0:
            # Generate sample starting with best character
            samp = DEC(generate_forced_first(actor,ctx0[:1],
                                           torch.tensor([best_char_idx], device=DEV),120)[0])
            wandb.log({"sample":wandb.Html(f"<pre>{samp}</pre>")},step=chars_seen)

    run.finish()

if __name__ == "__main__":
    train_remote.remote()