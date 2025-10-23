import torch 
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, X):
        return self.embedding(X) * math.sqrt(self.d_model)

class Positional_Encoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)   # shape (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model: int = 512, hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.W1 = nn.Linear(self.d_model, self.hidden_dim, bias=True)
        self.W2 = nn.Linear(self.hidden_dim, self.d_model, bias=True)
    
    def forward(self, x):
        return self.W2(self.dropout(torch.relu(self.W1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_head: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        assert d_model % num_head == 0, "d_model is not divisible by num_head"
        self.head_dim = d_model // self.num_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
     
    @staticmethod
    def attention(query, key, value, mask, dropout=None):
        head_dim = query.shape[-1]
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)
        attention_score = torch.softmax(attention_score, dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score @ value), attention_score

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.num_head, self.head_dim).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_head, self.head_dim).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_head, self.head_dim).transpose(1, 2)
        
        x, self.attention_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_head * self.head_dim)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttention, feed_forward_block: FeedForward, features: int, dropout: float):
        super().__init__()
        self.self_attn = self_attention
        self.feed_forward = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, features: int):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, masked_mha_attn: MultiHeadAttention, mha_attn: MultiHeadAttention, 
                 feed_forward_block: FeedForward, features: int, dropout: float):
        super().__init__()
        self.masked_mha_attn = masked_mha_attn
        self.mha_attn = mha_attn
        self.ffn = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    
    def forward(self, y, enc_out, tgt_mask, src_mask=None):
        x = self.residual_connection[0](y, lambda y: self.masked_mha_attn(y, y, y, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.mha_attn(x, enc_out, enc_out, src_mask))
        x = self.residual_connection[2](x, self.ffn)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, features: int):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, tgt_mask, src_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return self.norm(x)

class Projection_layer(nn.Module):
    def __init__(self, d_model: int, num_vocab: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_vocab)
    
    def forward(self, x):
        # Return logits instead of applying softmax
        # Softmax is typically applied during loss calculation or inference
        return self.linear(x)

# Causal mask function moved to dataset.py to avoid duplication 

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_emb: InputEmbedding, 
                 tar_emb: InputEmbedding, src_pos: Positional_Encoding, 
                 tar_pos: Positional_Encoding, projection: Projection_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder  # Fixed typo: was "decoer"
        self.src_emb = src_emb
        self.tar_emb = tar_emb
        self.tar_pos = tar_pos
        self.src_pos = src_pos
        self.projection = projection

    def encode(self, src, src_mask):
        src = self.src_emb(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tar, tar_mask, enc_out, src_mask=None):  # Added src_mask parameter
        tar = self.tar_emb(tar)
        tar = self.tar_pos(tar)
        return self.decoder(tar, enc_out, tar_mask, src_mask)  # Fixed method name
    
    def project(self, x):
        return self.projection(x)

def build_transformer(src_vocab: int, tar_vocab: int, src_seq_len: int, tar_seq_len: int, 
                     d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, 
                     hidden_dim: int = 2048) -> Transformer:
    # Creating the embedding layers
    src_embedding = InputEmbedding(d_model, src_vocab)
    tar_embedding = InputEmbedding(d_model, tar_vocab)

    # Creating positional encoding layers
    src_pos = Positional_Encoding(d_model, src_seq_len, dropout)
    tar_pos = Positional_Encoding(d_model, tar_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        enc_attn = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, hidden_dim, dropout)
        encoder_block = EncoderBlock(enc_attn, feed_forward_block, d_model, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_attn = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attn = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, hidden_dim, dropout)
        decoder_block = DecoderBlock(decoder_attn, decoder_cross_attn, feed_forward_block, d_model, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks), d_model)
    decoder = Decoder(nn.ModuleList(decoder_blocks), d_model)
    projection_layer = Projection_layer(d_model, tar_vocab)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embedding, tar_embedding, src_pos, tar_pos, projection_layer)

    # Initialize parameters with Xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        
    return transformer