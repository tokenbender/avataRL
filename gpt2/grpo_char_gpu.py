#!/usr/bin/env python
"""
GRPO tiny-Shakespeare trainer (stand-alone)
==========================================

* Actor  : tiny GPT-style causal LM (random init)
* Reference (frozen, never updated):
    • "bigram" (default)  – P(char | previous char) from the dataset
    • "random"            – copy of actor at step 0 (KL starts at 0)
* Dense reward  : +1 for each correctly predicted char (optionally ±1)
* KL annealing  : 0 → beta_kl over kl_warmup_tokens characters

Run:   python grpo_tinyshakespeare.py --help   for all CLI flags
"""

# ─────────────── imports ────────────────────────────────────────────────
import argparse, math, copy, requests, sys
from pathlib import Path

import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ─────────────── helpers: dataset & vocab ───────────────────────────────
DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt")

def ensure_dataset(path="input.txt"):
    if not Path(path).exists():
        print("Downloading tiny-Shakespeare …")
        Path(path).write_text(requests.get(DATA_URL, timeout=10).text, encoding="utf-8")

def build_vocab(path="input.txt"):
    text  = Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi  = {ch:i for i,ch in enumerate(chars)}
    itos  = {i:ch for ch,i in stoi.items()}
    def enc(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    def dec(t): return "".join(itos[int(i)] for i in t)
    return enc, dec, len(chars), text

class TextLoader:
    def __init__(self, tokens: torch.Tensor, batch: int, seq: int):
        if len(tokens) < batch * seq + 1:
            raise ValueError("Dataset too small for chosen batch/seq length")
        self.data, self.B, self.T, self.pos = tokens, batch, seq, 0
    def next(self):
        span = self.B * self.T + 1
        if self.pos + span > len(self.data): self.pos = 0
        buf = self.data[self.pos : self.pos + span]; self.pos += span
        return buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)

# ─────────────── tiny GPT-like actor ────────────────────────────────────
class TinyGPT(nn.Module):
    def __init__(self, vocab, n_layer=6, n_head=6, n_emb=384, block_size=256):
        super().__init__()
        self.wte = nn.Embedding(vocab, n_emb)
        self.wpe = nn.Embedding(block_size, n_emb)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_emb, nhead=n_head, dim_feedforward=4*n_emb,
                activation="gelu", batch_first=True
            ) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab, bias=False)
        self.head.weight = self.wte.weight
    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln_f(x))

# ─────────────── bigram reference ───────────────────────────────────────
class BigramRef(nn.Module):
    def __init__(self, counts: torch.Tensor):
        super().__init__()
        probs = (counts + 1) / (counts.sum(1, keepdims=True) + counts.size(1))
        self.register_buffer("logits", torch.log(probs))   # (V,V)
    def forward(self, idx):                               # idx: (B,T)
        prev = idx                                         # (B,T)
        return self.logits[prev]                           # (B,T,V)

# ─────────────── utils: reward & generation ─────────────────────────────
def tok_reward(gen, ref, neg=False):
    match = (gen == ref).float()
    return match*2-1 if neg else match

def add_partial(rew, logp, coeff):
    return rew if coeff == 0 else rew + coeff * logp.detach()

@torch.no_grad()
def generate(model, ctx, steps):
    model.eval()
    for _ in range(steps):
        probs = torch.softmax(model(ctx)[:, -1], dim=-1)
        nxt   = torch.multinomial(probs, 1)
        ctx   = torch.cat([ctx, nxt], 1)
    return ctx[:, -steps:]

# ─────────────── main training loop ─────────────────────────────────────
def train(cfg):
    torch.set_float32_matmul_precision("high")
    DEV = torch.device("cuda" if (cfg.device=="auto" and torch.cuda.is_available())
                       else cfg.device)

    ensure_dataset(cfg.data_path)
    ENC, DEC, V, text = build_vocab(cfg.data_path)
    tokens = ENC(text)

    # actor + reference
    actor = TinyGPT(V).to(DEV)
    if cfg.ref_type == "random":
        ref = copy.deepcopy(actor).eval()
    else:  # bigram
        counts = torch.zeros(V, V)
        for a, b in zip(tokens[:-1], tokens[1:]):
            counts[a, b] += 1
        ref = BigramRef(counts).to(DEV).eval()

    opt = torch.optim.AdamW(actor.parameters(), lr=cfg.lr)
    loader = TextLoader(tokens, cfg.batch, cfg.horizon+1)

    # wandb
    if cfg.no_wandb or not WANDB_AVAILABLE:
        run = None
    else:
        run = wandb.init(project=cfg.wandb_project,
                         config=vars(cfg), reinit=True)

    chars_seen = 0
    pbar = tqdm(range(1, cfg.iters+1), desc="iters", leave=False)
    for it in pbar:
        ctx0, ref_tok = loader.next()
        ctx0, ref_tok = ctx0[:, :1].to(DEV), ref_tok[:, :cfg.horizon].to(DEV)

        group_gen, group_rew = [], []
        for _ in range(cfg.k_samples):
            gen    = generate(actor, ctx0.clone(), cfg.horizon)
            logits = actor(gen)
            dist   = torch.distributions.Categorical(logits.softmax(-1))
            logp   = dist.log_prob(gen)
            r_tok  = tok_reward(gen, ref_tok, cfg.neg_reward)
            r_tok  = add_partial(r_tok, logp, cfg.partial_logprob)
            group_gen.append(gen); group_rew.append(r_tok)

        gen  = torch.stack(group_gen, 1)
        rew  = torch.stack(group_rew, 1)
        base = rew.mean(dim=1, keepdim=True)
        adv  = (rew - base)
        adv  = (adv - adv.mean()) / (adv.std() + 1e-8)

        flat_gen = gen.reshape(-1, cfg.horizon)
        flat_adv = adv.reshape(-1, cfg.horizon)

        logits_new = actor(flat_gen)
        logits_ref = ref(flat_gen)
        dist_new   = torch.distributions.Categorical(logits_new.softmax(-1))
        dist_ref   = torch.distributions.Categorical(logits_ref.softmax(-1))
        logp_new   = dist_new.log_prob(flat_gen)
        kl         = torch.distributions.kl.kl_divergence(dist_new, dist_ref)

        kl_coef = min(cfg.beta_kl,
                      cfg.beta_kl * chars_seen / max(1, cfg.kl_warmup_tokens))
        pol_loss = -(flat_adv * logp_new).mean()
        loss     = pol_loss + kl_coef * kl.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()

        chars_seen += cfg.batch * cfg.horizon
        if run and it % cfg.log_every == 0:
            run.log({
                "reward": rew.mean().item(),
                "policy_loss": pol_loss.item(),
                "kl": kl.mean().item(),
                "kl_coef": kl_coef,
                "chars": chars_seen,
            }, step=chars_seen)

        if run and it % cfg.sample_every == 0:
            sample = DEC(generate(actor, ctx0[:1], 120)[0])
            run.log({"sample": wandb.Html(f"<pre>{sample}</pre>")}, step=chars_seen)
        pbar.set_postfix(R=rew.mean().item(), KL=kl_coef)

    if run: run.finish()

# ─────────────── CLI ────────────────────────────────────────────────────
def cli():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--horizon", type=int, default=70)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--k_samples", type=int, default=4)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta_kl", type=float, default=1e-3)
    p.add_argument("--kl_warmup_tokens", type=int, default=200_000)
    p.add_argument("--neg_reward", action="store_true", help="±1 vs 0/1 reward")
    p.add_argument("--partial_logprob", type=float, default=0.0)
    p.add_argument("--ref_type", choices=["bigram", "random"], default="bigram")
    p.add_argument("--data_path", default="input.txt")
    p.add_argument("--device", default="auto", help="'cpu', 'cuda', or 'auto'")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--sample_every", type=int, default=200)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="grpo-char")
    args = p.parse_args()
    if args.no_wandb and not WANDB_AVAILABLE:
        print("wandb not installed; continuing without logging.")
    train(args)

if __name__ == "__main__":
    cli()
