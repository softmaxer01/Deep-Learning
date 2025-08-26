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



X = torch.randint(0,5,(6,))
enc = Encoder(16,5,4)
emb= enc(X)
print(emb.shape)
