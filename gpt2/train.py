from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F 
import tiktoken

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
    with open(filename,'r') as file:
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
        buffer = self.tokens[self.cur_pos : self.cur_pos + B*T+1]
        token_tensor = buffer[:-1].view(B,T)
        next_token_tensor = buffer[1:].view(B,T)

        # move to next batch
        self.cur_pos += B*T
        if self.cur_pos  + B*T + 1 > len(self.tokens):
            self.cur_pos = 0
        return token_tensor, next_token_tensor

@dataclass
class config:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embed: int = 768 # embedding dimension
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0 # should be divisible exactly
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        
        # K, Q, V for all heads 
        self.c_attention = nn.Linear(config.n_embed,3*config.n_embed)
        # output projection

        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # mask
        self.register_buffer('bias',torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size))


    def forward(self,x):
        ''' 
        nh -> number of heads
        hs -> head size
        C - > number of channels = nh*hs
        '''
        B,T,C = x.size() # batch_size, sequence_length, embedding dimensionality
        qkv = self.c_attention(x)
        query, key , value = qkv.split(self.n_embed, dim = 2)
        key = key.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        query = query.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        value = value.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        y = F.scaled_dot_product_attention(query, key, value, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        # fullly connected layer
        self.c_fc = nn.Linear(config.n_embed,4*config.n_embed) # multiplied by 4 to increase dimensionality for modelling complex patterns
        self.gelu = nn.GELU(approximate='tanh') # why tanh ?  -> smooth non linear activation, tanh so that we can have good efficiency without too much loss

        self.c_proj = nn.Linear(4*config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self,x):
        x = self.c_fc(x)
        # activation
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.layer_norm_1 = nn.LayerNorm(config.n_embed)
        self.attention = CausalSelfAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self,x): # return x
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x 
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config


        self.transformer = nn.ModuleDict(dict(
            # weights of token embedding
            wte = nn.Embedding(config.vocab_size,config.n_embed),
            # weights of positional embeddings
            wpe = nn.Embedding(config.block_size,config.n_embed),
            # hidden layers in the transformers
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # layer norm 
            layernorm_f = nn.LayerNorm(config.n_embed),
            ))  
        self.lm_head = nn.Linear(config.n_embed,config.vocab_size, bias= False)
        # weight sharing for wte and lm head
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # 2 * because two blocks added : attention and mlp
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
    def forward(self,idx,targets = None):
        B,T = idx.size()

        assert T <= self.config.block_size
        pos = torch.arange(0,T,dtype = torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos) # dims -> (_,T,n_embed) _ -> is broadcasted to match dims
        tok_emb = self.transformer.wte(idx) # dims -> (B,T,n_embed)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.layernorm_f(x)
        logits = self.lm_head(x) # dims -> (B,T,vocab_size) used to project the token into vocabulary space
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))  # reshaping (B,T,vocab_size) -> (B*T,vocab_size)
        # if targets is not None:

        return logits,loss
        

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT(config())
model.to(device)


B,T = 4,32
trainLoader = TextLoader(B,T)


# logits, loss  = model(token_tensor,next_token_tensor)
optimiser = torch.optim.AdamW(model.parameters(), lr = 3e-4)

for i in range(50):
    token_tensor, next_token_tensor  = trainLoader.next_batch()
    token_tensor, next_token_tensor = token_tensor.to(device), next_token_tensor.to(device)
    optimiser.zero_grad() #n always start with zero gradients
    logits, loss = model(token_tensor,next_token_tensor)
    loss.backward()
    optimiser.step()
    print(f"step : {i} , loss : {loss.item()}")

print(loss)


