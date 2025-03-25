import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from dataclasses import dataclass


batch_size = 4
block_size = 32         
max_iters = 5000          
eval_interval = 100
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"


MASK_PROB = 0.15         
# GPT-2 uses a vocab size of 50257 by default; we'll add one extra token for [MASK]
MASK_TOKEN_ID = 50258     

def load_data_fromfile(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def tokenize(data):
    encoder = tiktoken.get_encoding('gpt2')
    tokens = encoder.encode(data)
    return tokens

class TextLoader:
    """
    Loads text from file, tokenizes it, and returns chunks of tokens.
    """
    def __init__(self, B, T):
        self.B = B
        self.T = T
        data = load_data_fromfile('input.txt')
        tokens = tokenize(data)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens.")
        print(f"Approximately {(len(self.tokens)) // (B*T)} batches per epoch.")
        self.cur_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        # Ensure we have a full chunk; if not, wrap around.
        if self.cur_pos + B * T > len(self.tokens):
            self.cur_pos = 0
        buffer = self.tokens[self.cur_pos : self.cur_pos + B * T]
        token_tensor = buffer.view(B, T)
        self.cur_pos += B * T
        return token_tensor


def mask_tokens(x, mask_token_id=MASK_TOKEN_ID, mask_prob=MASK_PROB):
    """
    x: (B, T) tensor of input token IDs.
    Returns:
      masked_x: (B, T) where ~mask_prob of tokens are replaced with the mask token.
      targets: original token tensor (used for loss computation).
    """
    B, T = x.shape
    targets = x.clone()
    # Create a mask: each position is masked with probability `mask_prob`
    rand_mask = torch.rand((B, T), device=x.device)
    mask_positions = rand_mask < mask_prob
    masked_x = x.clone()
    masked_x[mask_positions] = mask_token_id
    return masked_x, targets

@dataclass
class Config:
    block_size: int = 1024      # maximum sequence length (for positional embeddings)
    vocab_size: int = 50259     # GPT-2 vocab (50257) + 1 extra (e.g. <|endoftext|>) + 1 for [MASK]
    n_layer: int = 6            # number of transformer blocks
    n_head: int = 6             # number of attention heads
    n_embed: int = 384          # embedding dimension
    dropout: float = 0.2


class DenoisingSelfAttention(nn.Module):
    """
    Self-attention block without a causal mask. Instead of generating left-to-right,
    it attends to all tokens. This layer will be used for the denoising objective.
    """
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embed % config.n_head == 0, "Embedding dim must be divisible by num_heads."
        self.n_embed = config.n_embed
        self.n_head = config.n_head

        # Linear projection for Q, K, V together.
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)  # (B, T, 3*C)
        query, key, value = qkv.split(self.n_embed, dim=2)
        head_size = C // self.n_head
        query = query.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        key   = key.view(B, T, self.n_head, head_size).transpose(1, 2)
        value = value.view(B, T, self.n_head, head_size).transpose(1, 2)
        # Use PyTorch's built-in attention without causal mask.
        y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.attn = DenoisingSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTDenoiser(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embed),
            'wpe': nn.Embedding(config.block_size, config.n_embed),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embed)
        })
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # Weight tying between token embedding and the language model head.
        self.transformer['wte'].weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) masked input tokens.
        targets: (B, T) original tokens for computing loss.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, "Sequence length exceeds block_size."

        # Create positional indices and add positional embeddings.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        tok_emb = self.transformer['wte'](idx)   # (B, T, n_embed)
        pos_emb = self.transformer['wpe'](pos)     # (1, T, n_embed)
        x = tok_emb + pos_emb

        for block in self.transformer['h']:
            x = block(x)

        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


decoder = tiktoken.get_encoding("gpt2")

def decode_tokens(token_ids):
    """
    Converts a list or tensor of token IDs back to a text string.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    return decoder.decode(token_ids)

def visualize_masked_sequence(token_ids, mask_positions, mask_token_id=MASK_TOKEN_ID):
    """
    Visualize a sequence by showing [MASK] where mask_positions is True.
    token_ids: sequence of token IDs (list or tensor).
    mask_positions: a boolean list/tensor of the same length.
    """
    out_str = []
    for tid, m in zip(token_ids, mask_positions):
        if m:
            out_str.append("[MASK]")
        else:
            out_str.append(decoder.decode([tid]))
    return " ".join(out_str)


def sample_masked_sequence(
    model,
    seq_length=128,
    num_steps=5,
    temperature=1.0,
    top_k=None,
    top_p=None,
    device='cpu'
):
    """
    Iteratively refines a fully masked sequence into generated text.
    
    model: instance of GPTDenoiser.
    seq_length: desired output length.
    num_steps: number of refinement iterations.
    temperature: controls randomness.
    top_k, top_p: for filtering logits (optional).
    device: 'cuda' or 'cpu'.
    
    Returns:
        final_tokens: a tensor of token IDs (shape: (seq_length,))
    """
    # Start with a fully masked sequence.
    x = torch.full((1, seq_length), fill_value=MASK_TOKEN_ID, dtype=torch.long, device=device)
    # Keep track of which positions are still masked.
    still_masked = torch.ones_like(x, dtype=torch.bool)

    for step in range(num_steps):
        print(f"\n--- Iteration {step + 1} ---")
        print("Current sequence (masked positions shown as [MASK]):")
        # Visualize the current sequence.
        print(visualize_masked_sequence(x[0].tolist(), still_masked[0].tolist()))
        
        with torch.no_grad():
            logits, _ = model(x, targets=None)

        # For each position that is still masked, sample a token.
        for t in range(seq_length):
            if still_masked[0, t]:
                # Get logits for the current token position.
                logits_t = logits[0, t, :] / temperature
                probs = F.softmax(logits_t, dim=-1)
                # Optional top-k filtering.
                if top_k is not None:
                    v, ix = torch.topk(probs, top_k)
                    mask_filter = torch.zeros_like(probs).scatter_(0, ix, 1)
                    probs = probs * mask_filter
                    probs = probs / probs.sum()
                # Optional top-p (nucleus) filtering.
                if top_p is not None:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cumulative > top_p
                    # Shift the cutoff to retain at least one token.
                    cutoff = torch.cat([torch.zeros(1, device=device), cutoff[:-1]], dim=0)
                    remove_ix = sorted_indices[cutoff.bool()]
                    probs[remove_ix] = 0.0
                    probs = probs / probs.sum()
                sampled_token = torch.multinomial(probs, 1)
                x[0, t] = sampled_token

        # For this demo, once a token is sampled, we unmask it.
        still_masked.fill_(False)

    print("\n--- Final Generated Text ---")
    final_text = decode_tokens(x[0])
    print(final_text)
    return x[0]

# ---------------------------------------------------------
# Main: Training and Sampling
# ---------------------------------------------------------
if __name__ == "__main__":
    # Build configuration and instantiate the model.
    cfg = Config(
        block_size=block_size,
        vocab_size=50259,  # GPT2: 50257 + 1 extra (<|endoftext|>) + 1 for [MASK]
        n_layer=6,
        n_head=6,
        n_embed=384,
        dropout=0.2
    )
    model = GPTDenoiser(cfg).to(device)
    print("Total model parameters: {:.2f} Million".format(sum(p.numel() for p in model.parameters())/1e6))
    
    # Initialize the training data loader.
    train_loader = TextLoader(batch_size, block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("\n--- Starting Training ---")
    model.train()
    for step in range(max_iters):
        # Get the next batch and move it to the correct device.
        x = train_loader.next_batch().to(device)
        # Mask tokens for denoising.
        masked_x, targets = mask_tokens(x, mask_token_id=MASK_TOKEN_ID, mask_prob=MASK_PROB)
        # Forward pass.
        logits, loss = model(masked_x, targets=targets)
        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            print(f"Step {step:04d}, loss: {loss.item():.4f}")

    print("\n--- Training Complete ---")

    # Switch to evaluation mode for sampling.
    model.eval()
    # Sample and visualize iterative refinement.
    sample_masked_sequence(
        model, 
        seq_length=block_size, 
        num_steps=5, 
        temperature=1.0, 
        top_k=50,   # try adjusting top_k or top_p
        top_p=None,
        device=device
    )
