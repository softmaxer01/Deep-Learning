import torch
import torch.nn as nn
from dataclasses import dataclass
import math
from torch.nn import functional as F
from config import BertConfig


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dp)
        pe = torch.zeros(config.seq_len, config.d_model, device=config.device)
        position = torch.arange(0, config.seq_len, dtype=torch.float, device=config.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2, device=config.device).float() * (-math.log(10000.0) / config.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.d_model%config.nh == 0
        self.config = config
        self.c_attn = nn.Linear(config.d_model,3*config.d_model)
        self.c_proj = nn.Linear(config.d_model,config.d_model)
        self.nh = config.nh
        
    def forward(self,x):
        batch,seq_len,d_model = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.config.d_model,dim = 2)
        k = k.view(batch, seq_len,self.nh, d_model//self.nh).transpose(1, 2)
        q = q.view(batch, seq_len, self.nh, d_model//self.nh).transpose(1, 2)
        v = v.view(batch, seq_len, self.nh, d_model//self.nh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        y = self.c_proj(y)
        return y

class ffn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(config.d_model, config.dff),
            nn.ReLU(),
            nn.Linear(config.dff, config.d_model),
            nn.Dropout(config.dp)
        )
    
    def forward(self, x):
        return self.layer(x)

class Residual(nn.Module):
    def __init__(self, sublayer, config):
        super().__init__()
        self.sublayer = sublayer(config)
        self.ln = nn.LayerNorm(config.d_model)
    
    def forward(self, x):
        return x + self.sublayer(self.ln(x))

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.Sequential(
            Residual(MultiHeadAttention, config=config),
            Residual(ffn, config=config)
        )
    
    def forward(self, x):
        return self.layer(x)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size,config.d_model)
        self.pos_encoding = PositionalEncoding(config)
        self.layers = nn.Sequential(*[EncoderBlock(config=config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.layers(x)
        x = self.ln(x)
        return x
