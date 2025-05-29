#!/usr/bin/env python
"""
GRPO tiny-Shakespeare (FIXED VERSION)
Reference = frozen bigram counts
Launch: modal run grpo_bigram_fixed.py

Fixed issues based on proper GRPO algorithm:
1. Correct KL divergence calculation using torch.distributions.kl
2. Proper advantage normalization (group-relative)
3. Fixed distribution creation from logits
4. Added ratio clipping as per GRPO spec
"""

# ─── hyper-params ────────────────────────────────────────────────────────
HORIZON, BATCH, K_SAMPLES = 70, 32, 4
TOTAL_ITERS, LR           = 1_000, 1e-5
BETA_KL, KL_WARM          = 1e-3, 200_000
NEG_REWARD, PARTIAL_LP    = False, 0.0
GPU_TYPE                  = "H100"        # Premium GPU for faster training
CLIP_RATIO                = 0.2          # PPO-style clipping

# ─── Modal plumbing ──────────────────────────────────────────────────────
import math, requests, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import modal, wandb, numpy as np, copy

stub = modal.App("grpo-char-bigram-fixed")
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
def generate(m,ctx,steps):
    for _ in range(steps):
        nxt = torch.multinomial(torch.softmax(m(ctx)[:,-1],-1),1)
        ctx = torch.cat([ctx,nxt],1)
    return ctx[:,-steps:]

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
    # Keep a copy of old policy for ratio calculation
    actor_old = copy.deepcopy(actor).eval()
    opt   = torch.optim.AdamW(actor.parameters(), lr=LR)
    loader= TextLoader(text, ENC, BATCH, HORIZON+1)

    run = wandb.init(project="gpt2-grpo-char-fixed",
                     config=dict(ref="bigram",horizon=HORIZON,beta=BETA_KL))
    chars_seen=0
    for it in tqdm(range(1,TOTAL_ITERS+1),leave=False):
        ctx0, ref_tok = loader.next()
        ctx0, ref_tok = ctx0[:,:1].to(DEV), ref_tok[:,:HORIZON].to(DEV)
        G,R = [],[]
        
        # Generate K samples from OLD policy
        with torch.no_grad():
            for _ in range(K_SAMPLES):
                gen=generate(actor_old,ctx0.clone(),HORIZON)
                # Calculate log probs under old policy
                old_logits = actor_old(gen)
                old_dist = torch.distributions.Categorical(logits=old_logits)
                old_logp = old_dist.log_prob(gen)
                
                # Calculate rewards
                r = tok_reward(gen,ref_tok,NEG_REWARD)
                r = add_partial(r, old_logp, PARTIAL_LP)
                G.append(gen); R.append(r)
        
        G,R = torch.stack(G,1), torch.stack(R,1)
        
        # CORRECT Group-relative advantage calculation
        # Normalize PER BATCH ITEM, not across entire flattened tensor
        base = R.mean(dim=1,keepdim=True)  # Mean across K_SAMPLES
        adv = R - base
        # Normalize advantages per batch item
        adv_std = adv.std(dim=1,keepdim=True) + 1e-8
        adv = adv / adv_std

        # Flatten for loss calculation
        flatG = G.reshape(-1,HORIZON)
        flatA = adv.reshape(-1,HORIZON)
        
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

        if it%10==0:
            wandb.log({"reward":R.mean().item(),"kl":kl.item(),"kl_coef":kl_coef,
                       "chars":chars_seen, "ratio":ratio.mean().item()},step=chars_seen)
        if it%200==0:
            samp = DEC(generate(actor,ctx0[:1],120)[0])
            wandb.log({"sample":wandb.Html(f"<pre>{samp}</pre>")},step=chars_seen)

    run.finish()

# ─── local entry ─────────────────────────────────────────────────────────
@stub.local_entrypoint()
def main():
    train_remote.remote()

if __name__ == "__main__":
    main()