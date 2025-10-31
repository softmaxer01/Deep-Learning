from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ln: layer norm
# ffn: feed forward network
# h: probabily hidden


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.c_attn = nn.Linear(config.d_model, 3*config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.register_buffer(
            "bias", torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length)
        )
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2) # split the 3*d_model into 3 parts
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (batch_dim,context_len,n_head,d_model/n_head=head_size)-->(B,n_head,Context_len,h_size)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # q shape: (B,nh,T,hs)@(B,nh,hs,T)==>(B,nh,T,T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B,nh,T,T)@(B,nh,T,hs)==>(B,nh,T,hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4*config.d_model)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.d_model, config.d_model)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTconfig:
    context_length: int = 256
    vocab_size: int = 65
    n_layer: int = 6 # number of blocks
    n_head: int = 6 # number of head
    d_model: int = 384 # embd dim

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.d_model),
                wpe=nn.Embedding(config.context_length, config.d_model),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                lnf=nn.LayerNorm(config.d_model)
            )
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.context_length, f"can't forward more than the context length"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.transformer['wpe'](pos)
        tok_emb = self.transformer['wte'](idx)
        x = tok_emb + pos_emb
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['lnf'](x)
        logits = self.lm_head(x)
        return logits


# gpt = GPT(GPTconfig())
# xb = torch.randint(0, 65, (64, 100), dtype=torch.long) 
# print(gpt(xb).shape)
# # output shape: (B,T,vocab_size)