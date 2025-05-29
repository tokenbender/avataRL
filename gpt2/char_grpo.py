#!/usr/bin/env python
"""

 * GRPO (Group Relative Policy Optimisation) 
 * Character tokens → ~70-char horizon gives dense rewards
 * Optional:
       --neg_reward              (+1 / –1 token reward)
       --partial_logprob α       dense shaping term
 * KL annealing: 0 → beta_kl over kl_warmup_tokens characters
"""

import argparse, math, sys, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import wandb

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────── Data & vocab ──────────────────────────────────────
def build_vocab(path: str = "input.txt"):
    text  = Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi  = {ch: i for i, ch in enumerate(chars)}
    itos  = {i: ch for ch, i in stoi.items()}
    def enc(s: str) -> torch.Tensor:
        return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    def dec(t: torch.Tensor) -> str:
        return "".join(itos[int(i)] for i in t)
    return enc, dec, len(chars)

ENC, DEC, VOCAB_SIZE = build_vocab()

class TextLoader:
    """Deterministic cyclic mini-batch loader."""
    def __init__(self, path: str, batch: int, seq: int):
        data = ENC(Path(path).read_text(encoding="utf-8"))
        if len(data) < batch * seq + 1:
            raise ValueError("Dataset too small for chosen batch/seq length")
        self.data, self.B, self.T, self.pos = data, batch, seq, 0
    def next(self):
        span = self.B * self.T + 1
        if self.pos + span > len(self.data):
            self.pos = 0
        buf = self.data[self.pos:self.pos + span]; self.pos += span
        x = buf[:-1].view(self.B, self.T)  # context
        y = buf[1: ].view(self.B, self.T)  # next char
        return x, y

# ──────────────────────── Model ─────────────────────────────────────────────
@dataclass
class GPTCfg:
    block_size: int = 256        # must exceed horizon
    vocab_size:  int = VOCAB_SIZE
    n_layer:     int = 6
    n_head:      int = 6
    n_embed:     int = 384

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTCfg):
        super().__init__()
        self.n_head, self.d = cfg.n_head, cfg.n_embed // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embed, 3 * cfg.n_embed, bias=False)
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # [B,T,C] → [B,n_head,T,d]
        def reshape(t): return t.view(B, T, self.n_head, self.d).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, cfg: GPTCfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embed, 4 * cfg.n_embed),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * cfg.n_embed, cfg.n_embed),
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg: GPTCfg):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.n_embed), CausalSelfAttention(cfg)
        self.ln2, self.mlp  = nn.LayerNorm(cfg.n_embed), MLP(cfg)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTActor(nn.Module):
    def __init__(self, cfg: GPTCfg):
        super().__init__()
        self.wte  = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.wpe  = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.h    = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # weight-tying
    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.h: x = blk(x)
        return self.head(self.ln_f(x))

# ──────────────────────── Generation ────────────────────────────────────────
@torch.no_grad()
def generate(model: GPTActor, ctx: torch.Tensor, steps: int) -> torch.Tensor:
    model.eval()
    for _ in range(steps):
        logits = model(ctx)[:, -1]                  # last-token logits
        probs  = torch.softmax(logits, dim=-1)
        nxt    = torch.multinomial(probs, 1)
        ctx    = torch.cat([ctx, nxt], dim=1)
    return ctx[:, -steps:]                          # only new tokens

# ──────────────────────── Reward helpers ────────────────────────────────────
def tok_reward(gen, ref, neg=False):
    match = (gen == ref).float()
    return match * 2 - 1 if neg else match         # ±1  or  +1/0

def add_partial(rew, logp, coeff):
    return rew if coeff == 0 else rew + coeff * logp.detach()

# ──────────────────────── GRPO training ─────────────────────────────────────
@dataclass
class GRPOCfg:
    horizon:   int   = 70
    k_samples: int   = 4
    clip_eps:  float = 0.2      # kept for completeness (not used in basic GRPO)
    beta_kl:   float = 1e-3     # final KL weight
    lr:        float = 1e-5

BATCH_SIZE   = 32
TOTAL_ITERS  = 1000
LOG_EVERY    = 10
SAMPLE_EVERY = 200

def main():
    # ─── CLI ───────────────────────────────────────────────────────────────
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=70)
    ap.add_argument("--neg_reward", action="store_true")
    ap.add_argument("--partial_logprob", type=float, default=0.0)
    ap.add_argument("--beta_kl", type=float, default=1e-3)
    ap.add_argument("--kl_warmup_tokens", type=int, default=200_000,
                    help="linear ramp length for KL coefficient")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile() the model (PyTorch ≥ 2.0)")
    args = ap.parse_args()

    g = GPTCfg(block_size=max(256, args.horizon + 1))
    p = GRPOCfg(horizon=args.horizon, beta_kl=args.beta_kl)
    model = GPTActor(g).to(DEVICE)
    model = torch.compile(model)

    ref_policy = GPTActor(g).to(DEVICE).eval()   # frozen reference
    opt        = torch.optim.AdamW(model.parameters(), lr=p.lr)
    loader     = TextLoader("input.txt", BATCH_SIZE, p.horizon + 1)

    run = wandb.init(project="gpt2-grpo-char",
                     config={**g.__dict__, **p.__dict__,
                             "batch": BATCH_SIZE,
                             "neg_reward": args.neg_reward,
                             "partial": args.partial_logprob,
                             "kl_warmup_tokens": args.kl_warmup_tokens})
    chars_seen, kl_coef = 0, 0.0
    loop = tqdm(range(1, TOTAL_ITERS + 1), leave=False, desc="iters")

    for it in loop:
        # ── sample prompts ────────────────────────────────────────────────
        ctx0, ref_tok = loader.next()
        ctx0    = ctx0[:, :1].to(DEVICE)          # (B,1) start-char
        ref_tok = ref_tok[:, :p.horizon].to(DEVICE)

        group_gen, group_logp, group_rew = [], [], []
        for _k in range(p.k_samples):
            gen = generate(model, ctx0.clone(), p.horizon)         # (B,T)
            logits = model(gen)                                    # (B,T,V)
            dist   = torch.distributions.Categorical(logits.softmax(-1))
            logp   = dist.log_prob(gen)                            # (B,T)
            r_tok  = tok_reward(gen, ref_tok, args.neg_reward)
            r_tok  = add_partial(r_tok, logp, args.partial_logprob)
            group_gen.append(gen); group_logp.append(logp); group_rew.append(r_tok)

        gen  = torch.stack(group_gen, 1)   # (B,K,T)
        logp = torch.stack(group_logp, 1)  # (B,K,T)
        rew  = torch.stack(group_rew, 1)   # (B,K,T)

        base = rew.mean(dim=1, keepdim=True)      # baseline per-batch
        adv  = rew - base
        adv  = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalise

        flat_gen  = gen.reshape(-1, p.horizon)    # (B*K, T)
        flat_adv  = adv.reshape(-1, p.horizon)
        logits_new = model(flat_gen)
        logits_ref = ref_policy(flat_gen)

        dist_new = torch.distributions.Categorical(logits_new.softmax(-1))
        dist_ref = torch.distributions.Categorical(logits_ref.softmax(-1))
        logp_new = dist_new.log_prob(flat_gen)
        kl       = torch.distributions.kl.kl_divergence(dist_new, dist_ref)

        # ── KL annealing: linear 0 → beta over N tokens ───────────────────
        kl_coef = min(args.beta_kl,
                      args.beta_kl * chars_seen / max(1, args.kl_warmup_tokens))

        pol_loss = -(flat_adv * logp_new).mean()
        loss     = pol_loss + kl_coef * kl.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        chars_seen += BATCH_SIZE * p.horizon

        # ── Logging ───────────────────────────────────────────────────────
        if it % LOG_EVERY == 0:
            wandb.log({"reward": rew.mean().item(),
                       "policy_loss": pol_loss.item(),
                       "kl": kl.mean().item(),
                       "kl_coef": kl_coef,
                       "chars": chars_seen}, step=chars_seen)
        if it % SAMPLE_EVERY == 0:
            sample = DEC(generate(model, ctx0[:1], 120)[0])
            wandb.log({"sample": wandb.Html(f"<pre>{sample}</pre>")}, step=chars_seen)
            loop.set_postfix(R=f"{rew.mean():.3f}", KL=kl_coef)

    run.finish()

if __name__ == "__main__":
    main()



# # python grpo_char.py \
#     --horizon 70 \
#     --partial_logprob 0.05 \
#     --beta_kl 0.001 \
#     --kl_warmup_tokens 200000
