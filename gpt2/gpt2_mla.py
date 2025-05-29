import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return (x / (norm + self.eps)) * self.weight

class MultiHeadLatentAttention(nn.Module):
'''
Ropeless MLA
'''
    def __init__(self, config):
        super().__init__()
        assert config.v_head_dim is not None, "v_head_dim must be defined"
        assert config.q_lora_rank is not None, "q_lora_rank must be defined"
        assert config.kv_lora_rank is not None, "kv_lora_rank must be defined"
        assert config.nope_head_dim is not None, "nope_head_dim must be defined"
        self.config = config
        self.dim = config.d_model
        self.num_heads = config.n_heads
        self.v_head_dim = config.v_head_dim
        self.nope_head_dim = config.nope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.dropout = config.dropout
        self.value_dim = self.num_heads * self.v_head_dim
        self.nope_dim  = self.num_heads * self.nope_head_dim
        
        self.compress_q_linear = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.decompress_q_nope = nn.Linear(self.q_lora_rank, self.nope_dim, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank)
        
        self.compress_kv_linear = nn.Linear(self.dim, self.kv_lora_rank, bias=False)
        self.decompress_k_nope = nn.Linear(self.kv_lora_rank, self.nope_dim, bias=False)
        self.decompress_v_linear = nn.Linear(self.kv_lora_rank, self.value_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        
        self.proj = nn.Linear(self.value_dim, self.dim, bias=False)
        self.res_dropout = nn.Dropout(p=self.dropout)
        
    def forward(self, x, KV=None, prev_seq_length=0):
        batch_size, seq_len, _ = x.shape

        compressed_q = self.compress_q_linear(x)
        norm_q = self.q_norm(compressed_q)
        query_nope = self.decompress_q_nope(norm_q)

        compressed_kv = self.compress_kv_linear(x)
        norm_kv = self.kv_norm(compressed_kv)
        key_nope = self.decompress_k_nope(norm_kv)
        value = self.decompress_v_linear(norm_kv)

        # Heavy tensor manipulation: reshape and transpose for multi-head attention
        query = query_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1, 2)
        key   = key_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=self.dropout)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.value_dim)
        output = self.proj(attn_out)
        output = self.res_dropout(output)
        return output, None

def load_data_fromfile(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def tokenize(data):
    encoder = tiktoken.get_encoding('gpt2')
    return encoder.encode(data)

class TextLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        data = load_data_fromfile('input.txt')
        tokens = tokenize(data)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens, approx {len(self.tokens) // (B * T)} batches per epoch")
        self.cur_pos = 0
    def next_batch(self):
        B, T = self.B, self.T
        end_pos = self.cur_pos + B * T + 1
        buffer = self.tokens[self.cur_pos:end_pos]
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)
        self.cur_pos += B * T
        if self.cur_pos + B * T + 1 > len(self.tokens):
            self.cur_pos = 0
        return x, y

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    d_model: int = 768
    n_heads: int = 12
    dropout: float = 0.2
    v_head_dim: int = 64
    q_lora_rank: int = 192
    kv_lora_rank: int = 256
    nope_head_dim: int = 64

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.n_embed)
        self.attention = MultiHeadLatentAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    def forward(self, x, KV=None, prev_seq_length=0):
        x_ln = self.layer_norm_1(x)
        x_att, kv_out = self.attention(x_ln, KV=KV, prev_seq_length=prev_seq_length)
        x = x + x_att
        x_ln2 = self.layer_norm_2(x)
        x = x + self.mlp(x_ln2)
        return x, kv_out

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embed),
            'wpe': nn.Embedding(config.block_size, config.n_embed),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'layernorm_f': nn.LayerNorm(config.n_embed),
        })
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None, KV_caches=None, prev_seq_length=0):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        new_KV_caches = []
        for i, block in enumerate(self.transformer.h):
            layer_KV = KV_caches[i] if (KV_caches is not None) else None
            x, out_KV = block(x, KV=layer_KV, prev_seq_length=prev_seq_length)
            new_KV_caches.append(out_KV)
        x = self.transformer.layernorm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_KV_caches

# Training parameters
batch_size = 64
block_size = 256
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    gpt_config = Config(
        block_size=block_size,
        vocab_size=50257,
        n_layer=n_layer,
        d_model=n_embed,
        n_heads=n_head,
        dropout=dropout,
        v_head_dim=n_embed // n_head,
        q_lora_rank=192,
        kv_lora_rank=256,
        nope_head_dim=n_embed // n_head
    )
    model = GPT(gpt_config).to(device)
    B, T = 4, 32
    trainLoader = TextLoader(B, T)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        token_tensor, next_token_tensor = trainLoader.next_batch()
        token_tensor = token_tensor.to(device)
        next_token_tensor = next_token_tensor.to(device)
        optimizer.zero_grad()
        logits, loss, _ = model(token_tensor, targets=next_token_tensor)
        loss.backward()
        optimizer.step()
        print(f"Step {i}, Loss: {loss.item()}")
    print("Final loss:", loss.item())
