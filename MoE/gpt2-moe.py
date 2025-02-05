import mlflow
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from dataclasses import dataclass

batch_size = 32
block_size = 256
max_iters = 1000
eval_interval = 100
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
num_experts = 8
top_k = 2
capacity_factor = 1.0

def load_data_fromfile(filename):
    with open(filename,'r',encoding='utf-8') as file:
        text = file.read()
    return text

def tokenize(data):
    encoder = tiktoken.get_encoding('gpt2')
    tokens = encoder.encode(data)
    return tokens

class TextLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        data = load_data_fromfile('input.txt')
        tokens = tokenize(data)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"processing {(len(self.tokens)) // (B*T)} batches per epoch")
        self.cur_pos = 0
    def next_batch(self):
        B,T = self.B, self.T
        buffer = self.tokens[self.cur_pos : self.cur_pos + B*T + 1]
        token_tensor = buffer[:-1].view(B,T)
        next_token_tensor = buffer[1:].view(B,T)
        self.cur_pos += B*T
        if self.cur_pos + B*T + 1 > len(self.tokens):
            self.cur_pos = 0
        return token_tensor, next_token_tensor

@dataclass
class config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    num_experts: int = num_experts
    top_k: int = top_k
    capacity_factor: float = capacity_factor

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.c_attention = nn.Linear(config.n_embed,3*config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))
    def forward(self,x):
        B,T,C = x.size()
        qkv = self.c_attention(x)
        query, key, value = qkv.split(self.n_embed, dim=2)
        key = key.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        query = query.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        value = value.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        y = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4*config.n_embed),
            nn.ReLU(),
            nn.Linear(4*config.n_embed, config.n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class NoisyTopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.top_k
        self.topkroute_linear = nn.Linear(config.n_embed, config.num_experts)
        self.noise_linear = nn.Linear(config.n_embed, config.num_experts)
    def forward(self, x):
        logits = self.topkroute_linear(x)
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = NoisyTopkRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.num_experts = config.num_experts
    def forward(self, x):
        B,T,C = x.shape
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, C)
        flat_gating_output = gating_output.view(-1, self.num_experts)
        tokens_per_batch = B*T*self.top_k
        expert_capacity = int((tokens_per_batch/self.num_experts)*self.capacity_factor)
        updates = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            selected_indices = torch.nonzero(flat_mask).squeeze(-1)
            limited_indices = selected_indices[:expert_capacity] if selected_indices.numel() > expert_capacity else selected_indices
            if limited_indices.numel() > 0:
                expert_input = flat_x[limited_indices]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[limited_indices,i].unsqueeze(1)
                weighted_output = expert_output*gating_scores
                updates.index_add_(0, limited_indices, weighted_output)
        final_output += updates.view(B,T,C)
        return final_output

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.n_embed)
        self.attention = CausalSelfAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.n_embed)
        self.moe = SparseMoE(config)
    def forward(self,x):
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.moe(self.layer_norm_2(x))
        return x

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embed),
            wpe = nn.Embedding(config.block_size,config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            layernorm_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self.__init__weights)
    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    def forward(self,idx,targets=None):
        B,T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.layernorm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def decode(tokens):
    enc = tiktoken.get_encoding('gpt2')
    return enc.decode(tokens)

if __name__ == "__main__":
    trainLoader = TextLoader(batch_size, block_size)
    model = GPT(config())
    model.to(device)
    model = torch.compile(model)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    with mlflow.start_run():
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("block_size", block_size)
        mlflow.log_param("max_iters", max_iters)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_experts", num_experts)
        mlflow.log_param("top_k", top_k)
        for i in range(max_iters):
            token_tensor, next_token_tensor = trainLoader.next_batch()
            token_tensor, next_token_tensor = token_tensor.to(device), next_token_tensor.to(device)
            optimiser.zero_grad()
            logits, loss = model(token_tensor,next_token_tensor)
            loss.backward()
            optimiser.step()
            if i % eval_interval == 0:
                mlflow.log_metric("train_loss", loss.item(), step=i)
                print(f"step: {i}, loss: {loss.item()}")
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    output = model.generate(context, max_new_tokens=500)
    print(decode(output[0].tolist()))
