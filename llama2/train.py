#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
import tiktoken

#
# 1) Hyperparameters and data loading
#

# We'll treat one "epoch" as a fixed number of iterations over the data.
NUM_EPOCHS = 50
STEPS_PER_EPOCH = 5   # Increase if you want a fuller pass over the data each epoch.
LEARNING_RATE = 3e-4

# We'll do some small shapes for demonstration
BATCH_SIZE = 4
BLOCK_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data_fromfile(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def tokenize(data):
    """
    Uses tiktoken to encode text into GPT-2 BPE tokens.
    """
    encoder = tiktoken.get_encoding('gpt2')
    tokens = encoder.encode(data)
    return tokens

class TextLoader:
    """
    Loads data from 'input.txt', tokenizes it, and returns batches of shape (B, T).
    """
    def __init__(self, B, T, filename='input.txt'):
        self.B = B
        self.T = T
        data = load_data_fromfile(filename)
        tokens = tokenize(data)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens from '{filename}'")
        total_batches_per_epoch = len(self.tokens) // (B * T)
        print(f"Processing {total_batches_per_epoch} batches per epoch (approx.)")
        self.cur_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        # Grab a chunk of B*T + 1 tokens
        buffer = self.tokens[self.cur_pos : self.cur_pos + B*T + 1]
        if len(buffer) < B*T + 1:
            # Wrap around if we don't have enough tokens left
            self.cur_pos = 0
            buffer = self.tokens[self.cur_pos : self.cur_pos + B*T + 1]

        token_tensor = buffer[:-1].view(B, T)
        next_token_tensor = buffer[1:].view(B, T)
        # Move the cursor
        self.cur_pos += B*T
        return token_tensor, next_token_tensor


#
# 2) LLaMA-style Transformer code (with caches registered as buffers)
#

@dataclass
class ModelArgs:
    dim: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: Optional[int] = None
    vocab_size: int = 50257  # For GPT-2 BPE
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # For KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
):
    """
    Precompute the complex exponent for rotary embeddings.
    """
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    base = theta ** (theta_numerator / head_dim)
    theta_inv = 1.0 / base
    theta_inv = theta_inv.to(device)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta_inv).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """
    Apply the precomputed rotary embeddings to the Q or K vector x.
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)  # (1, Seq_Len, 1, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the key/value vectors to match the number of Q heads.
    """
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # --- Registering as buffers so they move to GPU with model.to(device)
        self.register_buffer(
            "cache_k",
            torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
            ),
            persistent=False
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(
                (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
            ),
            persistent=False
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        """
        x: (B, seq_len=1, dim) for single-token training
        start_pos: the position of the current token
        freqs_complex: the relevant slice of precomputed rotary embedding
        """
        batch_size, seq_len, _ = x.shape  # expecting seq_len == 1

        # Project to Q, K, V
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        # Reshape into (B, seq_len, n_head, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Rotary embeddings
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Update cache without tracking gradients
        with torch.no_grad():
            self.cache_k[:batch_size, start_pos : start_pos + seq_len].copy_(xk)
            self.cache_v[:batch_size, start_pos : start_pos + seq_len].copy_(xv)

        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Repeat to match the Q heads
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Prepare shapes for scaled dot-product
        xq = xq.transpose(1, 2)      # (B, H_Q, seq_len=1, head_dim)
        keys = keys.transpose(1, 2)    # (B, H_Q, seq_len_kv, head_dim)
        values = values.transpose(1, 2)  # (B, H_Q, seq_len_kv, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round up to multiple_of
        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of
        )

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = swish * x_v
        x = self.w2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1, "args.vocab_size must be set"
        self.args = args

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Precompute a large range of rotary embeddings
        self.register_buffer(
            "freqs_complex",
            precompute_theta_pos_frequencies(
                args.dim // args.n_heads,
                args.max_seq_len * 2,  # or bigger if you wish
                device="cpu"  # We'll move them later via .to(device)
            ),
            persistent=False
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        tokens: (B, 1)  -- only a single token at a time in this setup
        start_pos: integer position for these tokens
        """
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "This LLaMA code handles one token at a time (seq_len=1)."

        # Move the needed slice of freqs_complex to the correct device
        # (the entire buffer might be on GPU if model.to(device) is used).
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len].to(tokens.device)

        # Token embedding
        h = self.tok_embeddings(tokens)  # (B, 1, dim)

        # Pass through each transformer block
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        logits = self.output(h).float()  # (B, 1, vocab_size)
        return logits


#
# 3) Training Loop: We do 50 epochs, track loss, and then plot it.
#

def main():
    # Initialize data loader
    loader = TextLoader(BATCH_SIZE, BLOCK_SIZE, filename='input.txt')

    # Build model arguments
    args = ModelArgs()
    # We ensure everything is set to the device
    args.device = DEVICE
    args.max_batch_size = BATCH_SIZE
    args.max_seq_len = BLOCK_SIZE

    # Build the model
    model = Transformer(args)
    # Move model (and its buffers) to the specified device
    model.to(DEVICE)

    # Wrap model with torch.compile for faster training (requires PyTorch 2.0+)
    model = torch.compile(model)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # We'll collect the average loss at the end of each epoch
    all_losses = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0

        # We'll take STEPS_PER_EPOCH steps in each epoch
        for step in range(STEPS_PER_EPOCH):
            token_tensor, next_token_tensor = loader.next_batch()
            token_tensor = token_tensor.to(DEVICE)
            next_token_tensor = next_token_tensor.to(DEVICE)

            # Reset the attention cache so each batch is processed fresh.
            # Otherwise, keys/values from prior batch would interfere.
            for layer in model.layers:
                layer.attention.cache_k.zero_()
                layer.attention.cache_v.zero_()

            total_loss = 0.0

            # Process each position in block_size sequentially
            for pos in range(BLOCK_SIZE):
                # Single token input: shape (B, 1)
                input_tok = token_tensor[:, pos : pos + 1]  # (B, 1)
                logits = model(input_tok, start_pos=pos)   # (B, 1, vocab_size)

                # Cross-entropy loss to predict next token (if exists)
                if pos < BLOCK_SIZE - 1:
                    target_tok = next_token_tensor[:, pos]  # shape (B,)
                    logits_2d = logits.view(-1, logits.size(-1))  # (B, vocab_size)
                    loss = F.cross_entropy(logits_2d, target_tok)
                    total_loss += loss

            # We computed the sum of losses for each position except the last
            total_loss = total_loss / (BLOCK_SIZE - 1)

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        # Average loss in this epoch
        avg_epoch_loss = epoch_loss / STEPS_PER_EPOCH
        all_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_epoch_loss:.4f}")

    # After all epochs, plot the loss
    plt.figure(figsize=(7,4))
    plt.plot(range(1, NUM_EPOCHS+1), all_losses, marker='o')
    plt.title("Training Loss over 50 Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
