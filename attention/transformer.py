import torch
import torch.nn as nn

class Multiheaded_Attention(nn.Module):
    def __init__(self,num_embedding,head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.num_embedding = num_embedding
        self.w_q = nn.Linear(self.num_embedding,num_embedding,bias=False)
        self.w_k = nn.Linear(self.num_embedding,self.num_embedding,bias=False)
        self.w_v = nn.Linear(self.num_embedding,self.num_embedding,bias = False)
        self.w_o = nn.Linear(self.num_embedding,self.num_embedding,bias = False)
        self.softmax = nn.Softmax(dim=-1) 
    
    def forward(self,q,k,v):
        Qs = torch.split(q, self.head_dim, dim=-1)
        Ks = torch.split(k, self.head_dim, dim=-1)
        Vs = torch.split(v, self.head_dim, dim=-1)
        heads = []
        for q_head, k_head, v_head in zip(Qs, Ks, Vs):
            A = torch.matmul(q_head, k_head.transpose(-2,-1))/(self.head_dim)**0.5
            attention_score = self.softmax(A)
            heads.append(torch.matmul(attention_score, v_head))
        H = torch.cat(heads, dim=-1)
        return self.w_o(H)


class addnorm(nn.Module):
    def __init__(self,num_embedding):
        super().__init__()
        self.norm = nn.LayerNorm(num_embedding)

    def forward(self,x,y):
        return self.norm(x+y)


class FeedForward(nn.Module):
    def __init__(self, num_embedding, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * num_embedding  
        self.fc1 = nn.Linear(num_embedding, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_embedding)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class Encoder(nn.Module):
    def __init__(self,num_embedding,num_vocab,num_heads):
        super().__init__()
        self.num_embedding = num_embedding
        self.num_vocab = num_vocab
        self.num_heads = num_heads
        self.head_dim = num_embedding // num_heads        
        self.embedding = nn.Embedding(self.num_vocab,self.num_embedding)
        self.multihead_attn = Multiheaded_Attention(self.num_embedding,self.head_dim)
        self.feed_forward = FeedForward(self.num_embedding)
        self.add_norm1 = addnorm(self.num_embedding)
        self.add_norm2 = addnorm(self.num_embedding)
    
    def forward(self,V):
        emb_v = self.embedding(V)
        attn_output = self.multihead_attn(emb_v, emb_v, emb_v)
        x = self.add_norm1(emb_v, attn_output)
        ff_output = self.feed_forward(x)
        x = self.add_norm2(x, ff_output)
        return x

class Masked_Multiheaded_Attention(nn.Module):
    def __init__(self,num_embedding,head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.num_embedding = num_embedding
        self.w_q = nn.Linear(self.num_embedding,num_embedding,bias=False)
        self.w_k = nn.Linear(self.num_embedding,self.num_embedding,bias=False)
        self.w_v = nn.Linear(self.num_embedding,self.num_embedding,bias = False)
        self.w_o = nn.Linear(self.num_embedding,self.num_embedding,bias = False)
        self.softmax = nn.Softmax(dim=-1) 
    
    def forward(self,q,k,v):
        Qs = torch.split(q, self.head_dim, dim=-1)
        Ks = torch.split(k, self.head_dim, dim=-1)
        Vs = torch.split(v, self.head_dim, dim=-1)
        heads = []
        for q_head, k_head, v_head in zip(Qs, Ks, Vs):
            A = torch.matmul(q_head, k_head.transpose(-2,-1))/(self.head_dim)**0.5
            mask = torch.tril(torch.ones_like(A))
            result = A.masked_fill(mask == 0, float('-inf'))
            attention_score = self.softmax(result)
            heads.append(torch.matmul(attention_score, v_head))
        H = torch.cat(heads, dim=-1)
        return self.w_o(H)


class Decoder(nn.Module):
    def __init__(self,num_embedding,num_vocab,num_heads):
        super().__init__()
        self.num_embedding = num_embedding
        self.num_vocab = num_vocab
        self.num_heads = num_heads
        self.head_dim = num_embedding // num_heads        
        self.embedding = nn.Embedding(self.num_vocab,self.num_embedding)
        self.maskedmultihead_attn = Masked_Multiheaded_Attention(self.num_embedding,self.head_dim)
        self.multihead_attn = Multiheaded_Attention(self.num_embedding,self.head_dim)
        self.feed_forward = FeedForward(self.num_embedding)
        self.add_norm1 = addnorm(self.num_embedding)
        self.add_norm2 = addnorm(self.num_embedding)
        self.add_norm3 = addnorm(self.num_embedding)  
    
    def forward(self,V,enc_out):
        emb_v = self.embedding(V)
        masked_attn_output = self.maskedmultihead_attn(emb_v, emb_v, emb_v)
        x = self.add_norm1(emb_v, masked_attn_output)
        attn_output = self.multihead_attn(x,enc_out,enc_out)
        x1 = self.add_norm2(x, attn_output)
        ff_output = self.feed_forward(x1)
        x2 = self.add_norm3(x1, ff_output)
        return x2


class Seq2Seq(nn.Module):
    def __init__(self,enc_num_embedding,enc_num_vocab,enc_num_heads,de_num_embedding,de_num_vocab,de_num_heads):
        super().__init__()
        self.enc_num_embedding = enc_num_embedding
        self.enc_num_vocab = enc_num_vocab
        self.enc_num_heads = enc_num_heads
        self.de_num_embedding = de_num_embedding
        self.de_num_vocab = de_num_vocab
        self.de_num_heads = de_num_heads
        self.enc = Encoder(self.enc_num_embedding,self.enc_num_vocab,self.enc_num_heads)
        self.dec = Decoder(self.de_num_embedding,self.de_num_vocab,self.de_num_heads)
        self.linear = nn.LazyLinear(self.de_num_vocab)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self,X,Y):
        enc_out = self.enc(X) 
        return self.softmax(self.linear(self.dec(Y,enc_out)))

        

X = torch.randint(0,5,(6,))
Y = torch.randint(0,5,(6,))
model = Seq2Seq(16,5,4,16,5,4)
emb= model(X,Y)
print(emb)
print(emb.shape)
