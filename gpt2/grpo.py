
# tiny-Shakespeare GRPO training script

# -------------------------------------------------------------
import sys
import argparse, math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm
import tiktoken, wandb
torch.set_float32_matmul_precision("high")

# ——— hyper-params ------------------------------------------------------------
@dataclass
class GPTCfg:
    block_size: int = 256
    vocab_size:  int = 50257
    n_layer:     int = 6
    n_head:      int = 6
    n_embed:     int = 384

@dataclass
class GRPOCfg:
    horizon     = 32
    k_samples   = 4          # group size
    clip_eps    = 0.2        # KL clip
    beta_kl     = 0.001       # KL coefficient
    lr          = 1e-5

BATCH_SIZE     = 32
TOTAL_ITERS    = 1000
LOG_EVERY      = 10
SAMPLE_EVERY   = 200
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# ——— helpers -----------------------------------------------------------------
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
        buf = self.tok[self.pos: self.pos + span]; self.pos += span
        x = buf[:-1].view(self.B, self.T)
        y = buf[1: ].view(self.B, self.T)
        return x, y

# ——— model -------------------------------------------------------------------
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

class GPTActor(nn.Module):
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
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.wte(idx) + self.wpe(pos)
        for blk in self.h: x = blk(x)
        return self.head(self.ln_f(x))

# ——— generation --------------------------------------------------------------
@torch.no_grad()
def generate(model, ctx, steps):
    model.eval()
    for _ in range(steps):
        probs = torch.softmax(model(ctx)[:, -1], -1)
        nxt   = torch.multinomial(probs, 1)
        ctx   = torch.cat([ctx, nxt], 1)
    return ctx[:, -steps:]

# ——— reward ------------------------------------------------------------------
def token_reward(gen, ref, neg_wrong=False):
    match = (gen == ref).float()
    if neg_wrong:
        return match*2 - 1          # +1 correct, −1 wrong
    return match                    # +1 / 0

def add_partial(rew, logp, coeff):
    if coeff == 0: return rew
    return rew + coeff * logp.detach()

# ——— training loop ------------------------------------------------------------
def main():
    # CLI flags ---------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--neg_reward", action="store_true",
                        help="give −1 reward for wrong tokens")
    parser.add_argument("--partial_logprob", type=float, default=0.,
                        help="α for partial reward based on log-prob (0=off)")
    args = parser.parse_args()

    g, p = GPTCfg(), GRPOCfg()
    model      = GPTActor(g).to(DEVICE)
    model  = torch.compile(model)
    ref_policy = GPTActor(g).to(DEVICE).eval()   # frozen reference
    opt        = torch.optim.AdamW(model.parameters(), lr=p.lr)

    loader = TextLoader("input.txt", BATCH_SIZE, p.horizon + 1)

    run = wandb.init(project="gpt2-grpo-shakespeare",
                     config={**g.__dict__, **p.__dict__,
                             "batch": BATCH_SIZE,
                             "neg_reward": args.neg_reward,
                             "partial": args.partial_logprob})
    tok_seen, kl_coef = 0, p.beta_kl
    loop = tqdm(range(1, TOTAL_ITERS + 1), desc="iters", leave=False)

    for it in loop:
        # ── sample K completions per prompt ---------------------------------
        ctx0, _ = loader.next_batch()
        ctx0    = ctx0[:, :1].to(DEVICE)                        # (B,1)
        ref_tok = _[:, :p.horizon].to(DEVICE)                  # (B,T)

        group_gen, group_logp, group_rew = [], [], []
        for _k in range(p.k_samples):
            gen   = generate(model, ctx0.clone(), p.horizon)   # (B,T)
            logits = model(gen)                                # (B,T,V)
            dist   = torch.distributions.Categorical(torch.softmax(logits, -1))
            logp   = dist.log_prob(gen)                        # (B,T)

            r_tok  = token_reward(gen, ref_tok, args.neg_reward)
            r_tok  = add_partial(r_tok, logp, args.partial_logprob)

            group_gen.append(gen); group_logp.append(logp); group_rew.append(r_tok)

        gen   = torch.stack(group_gen, 1)     # (B,K,T)
        logp  = torch.stack(group_logp, 1)    # (B,K,T)
        rew   = torch.stack(group_rew, 1)     # (B,K,T)

        # ── GRPO advantage: r − group mean -----------------------------------
        base      = rew.mean(dim=1, keepdim=True)              # (B,1,T)
        adv       = rew - base                                 # (B,K,T)
        adv_norm  = (adv - adv.mean()) / (adv.std() + 1e-8)    # stabilise

        # flatten for update
        flat_gen  = gen.reshape(-1, p.horizon)
        flat_adv  = adv_norm.reshape(-1, p.horizon)
        flat_logp = logp.reshape(-1, p.horizon)

        # logits for current & reference policy
        logits_new = model(flat_gen)                           # (N,T,V)
        logits_ref = ref_policy(flat_gen)

        dist_new = torch.distributions.Categorical(torch.softmax(logits_new, -1))
        dist_ref = torch.distributions.Categorical(torch.softmax(logits_ref, -1))

        logp_new = dist_new.log_prob(flat_gen)                 # (N,T)
        kl       = torch.distributions.kl.kl_divergence(dist_new, dist_ref)  # (N,T)

        # ——— GRPO loss -------------------------------------------------------
        pol_loss = -(flat_adv * logp_new).mean()
        kl_loss  = kl_coef * kl.mean()
        loss     = pol_loss + kl_loss
        # if torch.isnan(loss) or torch.isinf(loss):
        #     print("Detected NaN/Inf in loss; dumping stats...")
        #     # print max/min logits, gradients, etc.
        #     sys.exit(1)


        # optimise ------------------------------------------------------------
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        tok_seen += BATCH_SIZE * p.horizon

        # logging -------------------------------------------------------------
        if it % LOG_EVERY == 0:
            wandb.log({"reward": rew.mean().item(),
                       "policy_loss": pol_loss.item(),
                       "kl": kl.mean().item(),
                       "tokens": tok_seen}, step=tok_seen)
        if it % SAMPLE_EVERY == 0:
            sample = ENC.decode(generate(model, ctx0[:1], 120)[0].tolist())
            wandb.log({"sample": wandb.Html(f"<pre>{sample}</pre>")}, step=tok_seen)
            loop.set_postfix(R=f"{rew.mean():.3f}")

    run.finish()

if __name__ == "__main__":
    main()
