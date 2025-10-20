# this is the implemention of Vision transformer
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@dataclass
class ViTConfig:
    d_model = 768
    P = 32 
    H = 224 
    W = 224  
    C = 3
    context_length = N = (H*W)//P**2  # N = 49 patches
    n_head = 12
    layers = 12
    classes = 10  # ImageNet has 1000 classes (change to 10 for CIFAR-10)



class PatchEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.patch_size = config.P
        # Calculate input dimension: C * P * P
        patch_dim = config.C * config.P * config.P
        self.proj = nn.Linear(patch_dim, config.d_model, bias=True)
        self.cls = nn.Parameter(torch.randn(1, 1, config.d_model)) 
        self.pos_enc = PositionalEncoder(config)

    def forward(self, img):
        B, C, H, W = img.shape
        patches = self.get_patches(img, self.patch_size)  
        patches_proj = self.proj(patches)                 
        patches_proj = patches_proj + self.pos_enc()  # pos_enc already has batch dim
        cls_tokens = self.cls.expand(B, -1, -1)  
        out = torch.cat([cls_tokens, patches_proj], dim=1)  
        return out

    def get_patches(self, img, patch_size):
        B, C, H, W = img.shape
        assert H % patch_size == 0 and W % patch_size == 0
        patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        n_h, n_w = patches.size(2), patches.size(3)
        patches = patches.permute(0, 2, 3, 1, 4, 5) 
        patches = patches.contiguous().view(B, n_h*n_w, C*patch_size*patch_size)
        return patches


class PositionalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # N positions for patches
        self.pos_emb = nn.Parameter(torch.randn(1, config.N, config.d_model))
    
    def forward(self):
        return self.pos_emb  # Return with batch dimension  


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_head

        self.c_attn = nn.Linear(config.d_model, 3*config.d_model, bias=True)  # QKV projection with bias
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=True)    # Output projection with bias
        # self.register_buffer(
        #     "bias", torch.tril(torch.ones(config.N+1, config.N+1))
        #     .view(1, 1, config.N+1, config.N+1)
        # )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)  

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.c_proj(y)
        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4*config.d_model, bias=True)     # Add bias
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.d_model, config.d_model, bias=True)   # Add bias
    
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


class Linear_Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Linear(config.d_model, config.classes, bias=True)  # Add bias for classification head
    
    def forward(self, x):
        return self.layer(x)



class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc = PatchEncoder(config)
        self.blocks = nn.Sequential(*[Block(config=config) for i in range(config.layers)])
        self.linear = Linear_Layer(config)
    
    def forward(self, x):
        y = self.enc(x)
        y = self.blocks(y)
        z_L = y[:, 0, :]
        logits = self.linear(z_L)
        return logits





x = torch.randn((4, 3, 224, 224))  # ImageNet image size
model = ViT(ViTConfig)
print(model(x).shape)

num_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f"number of params: {num_trainable_params}")