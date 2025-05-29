#!/usr/bin/env python


from dataclasses import dataclass
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
import tiktoken, wandb
torch.set_float32_matmul_precision('high')
# ─── hyper-params
@dataclass
class GPTCfg:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384

@dataclass
class PPOCfg:
    horizon = 32
    gamma = 1.0
    lam = 0.95
    clip_eps = 0.2
    ppo_epochs = 4
    minibatches = 4
    lr_policy = 3e-5
    lr_value = 1e-4

BATCH_SIZE = 64
TOTAL_ITERS = 3000
LOG_EVERY = 10
SAMPLE_EVERY = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── data
ENC = tiktoken.get_encoding("gpt2")
def enc(t): return torch.tensor(ENC.encode(t), dtype=torch.long)

class TextLoader:
    def __init__(self, path: str, B: int, T: int):
        self.tok = enc(open(path, "r", encoding="utf-8").read())
        if len(self.tok) < B * T + 1: raise ValueError("dataset too small")
        self.B, self.T, self.pos = B, T, 0
    def next_batch(self):
        span = self.B * self.T + 1
        if self.pos + span > len(self.tok): self.pos = 0
        buf = self.tok[self.pos : self.pos + span]; self.pos += span
        x = buf[:-1].view(self.B, self.T)
        y = buf[1: ].view(self.B, self.T)
        return x, y

# ─── model
class CausalSelfAttention(nn.Module):
    def __init__(self, c: GPTCfg):
        super().__init__()
        self.n_head, self.n_embed = c.n_head, c.n_embed
        self.qkv  = nn.Linear(c.n_embed, 3 * c.n_embed)
        self.proj = nn.Linear(c.n_embed, c.n_embed)
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, 2)
        q, k, v = [t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                   for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, c: GPTCfg):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c.n_embed, 4 * c.n_embed),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * c.n_embed, c.n_embed))
    def forward(self, x): return self.fc(x)

class Block(nn.Module):
    def __init__(self, c: GPTCfg):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(c.n_embed), CausalSelfAttention(c)
        self.ln2, self.mlp  = nn.LayerNorm(c.n_embed), MLP(c)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTCore(nn.Module):
    def __init__(self, c: GPTCfg):
        super().__init__()
        self.wte  = nn.Embedding(c.vocab_size, c.n_embed)
        self.wpe  = nn.Embedding(c.block_size, c.n_embed)
        self.h    = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embed)
        self.head = nn.Linear(c.n_embed, c.vocab_size, bias=False)
        self.head.weight = self.wte.weight
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.h: x = blk(x)
        h = self.ln_f(x)
        return self.head(h), h

class GPTActorCritic(nn.Module):
    def __init__(self, c: GPTCfg):
        super().__init__()
        self.core = GPTCore(c)
        self.v    = nn.Linear(c.n_embed, 1)
    def forward(self, idx, vals=False):
        logits, h = self.core(idx)
        return (logits, self.v(h).squeeze(-1)) if vals else logits

# ─── ops
def generate(model, ctx, steps):
    model.eval()
    for _ in range(steps):
        probs = torch.softmax(model(ctx)[:, -1], -1)
        nxt = torch.multinomial(probs, 1)
        ctx = torch.cat([ctx, nxt], 1)
    return ctx[:, -steps:]

def reward(gen, ref): return (gen == ref).float()

# ─── train
def main():
    g, p = GPTCfg(), PPOCfg()
    model = GPTActorCritic(g).to(DEVICE)
    model = torch.compile(model,mode = "default")
    opt_pi = torch.optim.AdamW(model.parameters(), lr=p.lr_policy)
    opt_v  = torch.optim.AdamW(model.v.parameters(), lr=p.lr_value)
    loader = TextLoader("input.txt", BATCH_SIZE, p.horizon + 1)

    run = wandb.init(project="gpt2-ppo-shakespeare",
                     config={**g.__dict__, **p.__dict__, "batch": BATCH_SIZE})
    wandb.watch(model, log="gradients", log_freq=50)

    tok_seen = 0
    loop = tqdm(range(1, TOTAL_ITERS + 1), desc="iters", leave=False)
    for it in loop:
        model.eval()
        x, _  = loader.next_batch()
        ctx0  = x[:, :1].to(DEVICE)
        ref   = x[:, 1:1+p.horizon].to(DEVICE)
        gen   = generate(model, ctx0, p.horizon)
        logits, vals = model(gen, vals=True)
        dist  = torch.distributions.Categorical(torch.softmax(logits, -1))
        logp  = dist.log_prob(gen)
        rew   = reward(gen, ref)

        B, T = rew.shape
        adv, ret = torch.zeros_like(rew), torch.zeros_like(rew)
        gae = torch.zeros(B, device=DEVICE)
        for t in reversed(range(T)):
            next_val = vals[:, t+1] if t+1 < T else 0
            delta = rew[:, t] + p.gamma * next_val - vals[:, t]
            gae   = delta + p.gamma * p.lam * gae
            adv[:, t], ret[:, t] = gae, gae + vals[:, t]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        logp_old = logp.detach().reshape(-1)
        adv      = adv.detach().reshape(-1)
        ret      = ret.detach().reshape(-1)
        obs      = gen.reshape(-1, 1)
        act      = gen.reshape(-1)

        model.train()
        idx = torch.randperm(len(adv), device=DEVICE)
        for _ in range(p.ppo_epochs):
            for mb in idx.split(len(idx)//p.minibatches):
                logits, v = model(obs[mb], vals=True)
                logits, v = logits.squeeze(1), v.squeeze(1)
                dist = torch.distributions.Categorical(torch.softmax(logits, -1))
                lp   = dist.log_prob(act[mb])
                ratio = (lp - logp_old[mb]).exp()
                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1-p.clip_eps, 1+p.clip_eps) * adv[mb]
                pol_loss = -torch.min(s1, s2).mean()
                val_loss = F.mse_loss(v, ret[mb])
                (pol_loss + .5*val_loss).backward()
                opt_pi.step(); opt_v.step()
                opt_pi.zero_grad(); opt_v.zero_grad()

        tok_seen += BATCH_SIZE * p.horizon
        if it % LOG_EVERY == 0:
            wandb.log({"reward": rew.mean().item(),
                       "p_loss": pol_loss.item(),
                       "v_loss": val_loss.item(),
                       "tokens": tok_seen}, step=tok_seen)
        if it % SAMPLE_EVERY == 0:
            s = ENC.decode(generate(model, ctx0[:1], 120)[0].tolist())
            wandb.log({"sample": wandb.Html(f"<pre>{s}</pre>")}, step=tok_seen)
            loop.set_postfix(R=f"{rew.mean():.3f}")

    run.finish()

if __name__ == "__main__":
    main()
