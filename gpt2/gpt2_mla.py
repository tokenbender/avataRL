import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass


batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

def load_data_fromfile(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def tokenize(data):
    # uses tiktoken
    encoder = tiktoken.get_encoding('gpt2')
    tokens = encoder.encode(data)
    return tokens

class TextLoader:
    """
    Simple batch loader for text data, tokenizes everything,
    then yields (input, target) pairs in chunks of (B, T).
    """
    def __init__(self, B, T):
        self.B = B
        self.T = T
        data = load_data_fromfile('input.txt')
        tokens = tokenize(data)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"processing {(len(self.tokens)) // (B*T)} batches per epoch")
        self.cur_pos = 0

    def next_batch(self):
        B,T = self.B, self.T
        end_pos = self.cur_pos + B*T + 1
        buffer = self.tokens[self.cur_pos:end_pos]
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)

        self.cur_pos += B*T
        if self.cur_pos + B*T + 1 > len(self.tokens):
            self.cur_pos = 0
        return x, y


@dataclass
class Config:
    block_size: int = 1024    # max sequence length
    vocab_size: int = 50257   # GPT-2 BPE vocab size
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

    # Additional fields for MLA usage
    d_model: int = 768
    n_heads: int = 12


class MLA(nn.Module):
    '''
        Ropeless implementation of MLA with KVCache
    '''
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model  # e.g. 384
        self.n_heads = config.n_heads  # e.g. 6

        # Dimension splits
        self.q_proj = self.d_model // 2            
        self.kv_proj = (2 * self.d_model) // 3     
        self.dh = self.d_model // self.n_heads     

    
        # "Down" from d_model → q_proj, then LN, then "Up" from q_proj → d_model.
        self.W_dq = nn.Parameter(0.01 * torch.randn((self.d_model, self.q_proj)))  # (384,192)
        self.W_uq = nn.Parameter(0.01 * torch.randn((self.q_proj, self.d_model)))  # (192,384)
        self.Q_layernorm = nn.LayerNorm(self.q_proj)

        
        # "Down" from d_model→2*kv_proj (e.g. 384→512), LN, then "Up" 512→2*d_model=768
        
        self.W_dkv = nn.Parameter(0.01 * torch.randn((self.d_model, 2 * self.kv_proj)))   # (384,512)
        self.W_ukv = nn.Parameter(0.01 * torch.randn((2 * self.kv_proj, 2 * self.d_model)))  # (512,768)
        self.KV_layernorm = nn.LayerNorm(2 * self.kv_proj)

        # final output projection
        self.W_o = nn.Parameter(0.01 * torch.randn((self.d_model, self.d_model)))  # (384,384)

    def forward(self, x, KV=None, prev_seq_length=0):
        """
        x: (B, T, d_model)
        KV: tuple of (k_cache, v_cache) if using caching
        prev_seq_length: number of tokens previously processed (for partial masking)
        """
        B, T, C = x.shape  # e.g. (B, T, 384)

        
        compressed_q = x @ self.W_dq              # (B, T, q_proj)
        compressed_q = self.Q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq              # (B, T, C )

        #KV path: down -> LN -> up, then split
        compressed_kv = x @ self.W_dkv            # (B, T, C)
        compressed_kv = self.KV_layernorm(compressed_kv)
        kv_out = compressed_kv @ self.W_ukv       # (B, T, d_model)
  
        K, V = torch.split(kv_out, C, dim=-1)     # K,V => (B, T, d_model)

        # reshape for multi-head attention
        Q_heads = Q.view(B, T, self.n_heads, self.dh).transpose(1, 2)  # (B, nHeads, T, dh)
        K_heads = K.view(B, T, self.n_heads, self.dh).transpose(1, 2)  
        V_heads = V.view(B, T, self.n_heads, self.dh).transpose(1, 2)  

        
        if KV is not None:
            k_cache, v_cache = KV
            K_heads = torch.cat([k_cache, K_heads], dim=2)  
            V_heads = torch.cat([v_cache, V_heads], dim=2)

        T_full = K_heads.size(2)  # total length (old + new)

        # create lower-triangular mask (shift by prev seq length)
        mask = torch.ones((T, T_full), device=x.device)
        mask = torch.tril(mask, diagonal=0 + prev_seq_length)
        sq_mask = (mask == 1)

        # dot-product attention
        out = F.scaled_dot_product_attention(
            Q_heads, K_heads, V_heads,
            attn_mask=sq_mask
        )
        
        out = out.transpose(1, 2).reshape(B, T, C)
        out = out @ self.W_o.T  # final linear layer

        # return output plus new caches
        return out, (K_heads, V_heads)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        # used by the original "nanoGPT" style code
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.n_embed)
        self.attention = MLA(config)   # MLA here
        self.layer_norm_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x, KV=None, prev_seq_length=0):
        """
        x: (B, T, n_embed)
        KV: optional (k_cache, v_cache)
        prev_seq_length: for partial mask offset
        """
        # Attention sub-layer
        x_ln = self.layer_norm_1(x)
        x_att, kv_out = self.attention(x_ln, KV=KV, prev_seq_length=prev_seq_length)
        x = x + x_att

        # Feed-forward sub-layer
        x_ln2 = self.layer_norm_2(x)
        x_mlp = self.mlp(x_ln2)
        x = x + x_mlp

        return x, kv_out


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            layernorm_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # share weights between wte and lm_head
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
        """
        idx: (B, T) token indices
        KV_caches: optional list of (K,V) for each layer (for incremental decoding)
        prev_seq_length: used if partially decoding
        """
        B, T = idx.size()
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)         # (B, T, n_embed)
        pos_emb = self.transformer.wpe(pos)         # (T, n_embed) -> broadcasts to (B,T,n_embed)
        x = tok_emb + pos_emb

        new_KV_caches = []

        # pass through each block
        for i, block in enumerate(self.transformer.h):
            layer_KV = KV_caches[i] if (KV_caches is not None) else None
            x, out_KV = block(
                x,
                KV=layer_KV,
                prev_seq_length=prev_seq_length
            )
            new_KV_caches.append(out_KV)

        x = self.transformer.layernorm_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, new_KV_caches


if __name__ == "__main__":
    # Create a GPT config for MLA
    gpt_config = Config(
        block_size=block_size,
        vocab_size=50257,
        n_layer=n_layer,
        n_head=n_head,
        n_embed=n_embed,
        d_model=n_embed,   # for MLA, matches n_embed
        n_heads=n_head
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
