import torch
import torch.nn as nn
from dataclasses import dataclass
import math
import torch.nn.functional as F

@dataclass
class ModelConfig:
    d_model = 512
    n_h = 8
    seq_len = 512
    batch_size = 8
    dff = 2048
    dp = 0.1
    n_layers = 6
    vocab_size = 37000


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dp)
        
        pe = torch.zeros(config.seq_len, config.d_model)
        position = torch.arange(0, config.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalAttention(nn.Module):
    def __init__(self, config, decoder=True):
        super().__init__()
        assert config.d_model % config.n_h == 0
        self.config = config
        self.decoder = decoder
        self.c_attn = nn.Linear(config.d_model, config.d_model * 3)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dp)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.seq_len, config.seq_len)).view(1, 1, config.seq_len, config.seq_len)
        )

    def forward(self, x):
        batch, seq_len, d_model = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(d_model, dim=2)

        q = q.view(batch, seq_len, self.config.n_h, d_model // self.config.n_h).transpose(1, 2)
        k = k.view(batch, seq_len, self.config.n_h, d_model // self.config.n_h).transpose(1, 2)
        v = v.view(batch, seq_len, self.config.n_h, d_model // self.config.n_h).transpose(1, 2)

        attention_score = q @ k.transpose(-2, -1) * (1 / math.sqrt(k.size(-1)))

        if self.decoder:
            attention_score = attention_score.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf')
            )

        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)
        out = attention_score @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.c_proj(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_h == 0
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dp)

    def forward(self, q, k, v):
        batch, seq_len, d_model = q.size()

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(batch, seq_len, self.config.n_h, d_model // self.config.n_h).transpose(1, 2)
        k = k.view(batch, k.size(1), self.config.n_h, d_model // self.config.n_h).transpose(1, 2)
        v = v.view(batch, v.size(1), self.config.n_h, d_model // self.config.n_h).transpose(1, 2)

        attention_score = q @ k.transpose(-2, -1) * (1 / math.sqrt(k.size(-1)))
        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)
        out = attention_score @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.c_proj(out)
        return out


class Residual(nn.Module):
    def __init__(self, sublayer, config):
        super().__init__()
        self.sublayer = sublayer(config)
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, x):
        return x + self.sublayer(self.ln(x))


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


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.Sequential(
            Residual(lambda cfg: CausalAttention(cfg, decoder=False), config=config),
            Residual(ffn, config=config)
        )

    def forward(self, x):
        return self.layer(x)


class Encoder(nn.Module):
    def __init__(self, config, shared_embedding):
        super().__init__()
        self.config = config
        self.embedding = shared_embedding
        self.pos_encoding = PositionalEncoding(config)
        self.layers = nn.Sequential(*[EncoderBlock(config=config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.layers(x)
        x = self.ln(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mha = Residual(CausalAttention, config)
        self.ca = CrossAttention(config)
        self.feedforward = Residual(ffn, config)
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, dec_inp, enc_output):
        y = self.mha(dec_inp)
        out = self.ca(y, enc_output, enc_output)
        out = y + self.ln(out)
        out = self.feedforward(out)
        return out


class Decoder(nn.Module):
    def __init__(self, config, shared_embedding):
        super().__init__()
        self.config = config
        self.embedding = shared_embedding
        self.pos_encoding = PositionalEncoding(config)
        self.layers = nn.ModuleList([DecoderBlock(config=config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, dec_inp, enc_output):
        dec_inp = self.embedding(dec_inp)
        dec_inp = self.pos_encoding(dec_inp)
        for layer in self.layers:
            dec_inp = layer(dec_inp, enc_output)
        dec_inp = self.ln(dec_inp)
        return dec_inp


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.shared_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = Encoder(config, self.shared_embedding)
        self.decoder = Decoder(config, self.shared_embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared_embedding.weight
    
    def forward(self, src, tgt):
        enc_out = self.encoder(src)
        dec_out = self.decoder(tgt, enc_out)
        logits = self.lm_head(dec_out)
        return logits


config = ModelConfig()
model = Model(config)

src = torch.randint(0, config.vocab_size, (4, 512))
tgt = torch.randint(0, config.vocab_size, (4, 512))

output = model(src, tgt)
print("Model output shape:", output.shape)
print("Total parameters:", sum(p.numel() for p in model.parameters()))