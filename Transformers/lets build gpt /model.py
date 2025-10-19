import torch
import torch.nn as nn
import torch.nn.functional as f

# hyperparams
context_length = 8 # max context length or seq len sometime we will call it T also both are same
d_model = 384 # embedding dim
h = 6 # number of heads
Batch_size = 4
n_vocab = 65
n_layers = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SelfAttention(nn.Module):
    def __init__(self, head_size,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_size = head_size
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_length, context_length)))
    
    def forward(self, x):
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        # attention scores (B, T, T)
        wei = K @ Q.transpose(-2, -1)
        # apply causal mask for current sequence length
        B, T, _ = wei.shape
        mask = self.tril[:T, :T]  # it is just the current context length
        wei = wei.masked_fill(mask == 0, float("-inf"))
        wei = f.softmax(wei, dim=-1)
        # attention output (B, T, head_size)
        out = wei @ V
        return out        



class MultiHeadAttention(nn.Module):
    def __init__(self,h,head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size=head_size) for i in range(h)])
        self.proj = nn.Linear(h*head_size,d_model) # in case h*head_size != d_model
    
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.proj(out)
        return out
    


class FeedForwardNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = nn.Sequential(
            nn.Linear(d_model,d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4,d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        return self.layer(x)
    


class Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ma = MultiHeadAttention(h,(d_model//4))
        self.ffn = FeedForwardNetwork()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self,x):
        x = x + self.ma(self.ln1(x))
        x = x + self.ffn(self.ln1(x))
        return x
    

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embd = nn.Embedding(n_vocab,d_model)
        self.posembd = nn.Embedding(context_length,d_model)
        self.block = nn.Sequential(*[Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model,n_vocab)

    # for better weight initialization from karypathy
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self,xb):
        tok_emb = self.embd(xb)
        pos_emb = self.posembd(torch.arange(context_length))
        x = tok_emb + pos_emb
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.proj(x)
        return logits
    

# model = Model()
# xb = torch.randint(0,65,(Batch_size,context_length))
# print(model(xb).shape)
# #output shape: (Batch_size,context_length,vocab_size)