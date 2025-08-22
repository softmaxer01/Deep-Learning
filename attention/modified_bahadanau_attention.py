import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,num_hiddens,embed_dim,vocab_size):
        super().__init__()
        self.num_hidden = num_hiddens
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.rnn = nn.RNN(self.embed_dim,self.num_hidden,bidirectional= True,batch_first=True)
        self.embedding = nn.Embedding(self.vocab_size,self.embed_dim)

    def forward(self,x):
        self.emb = self.embedding(x)
        output, last_hidden = self.rnn(self.emb)
        return output # dimension: (batch_size,seq_len,2*num_hiddens) because of the bidirectional rnn
    

class Bahdanau_Attention(nn.Module):
    def __init__(self,key_dim,q_dim,attention_dim):
        super().__init__()
        self.key_dim = key_dim
        self.q_dim = q_dim
        self.attention_dim = attention_dim

        self.W_encoder = nn.Linear(self.key_dim,self.attention_dim,bias= False)
        self.W_decoder = nn.Linear(self.q_dim,self.attention_dim,bias=False)
        self.v = nn.Linear(attention_dim,1,bias = False)
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self,key,q):
        batch_size = key.shape[0]
        seq_len = key.shape[1]

        self.U = self.W_encoder(key)
        self.V = self.W_decoder(q).unsqueeze(1).expand(-1,seq_len,-1)

        self.attention_score = self.v(torch.tanh(self.U + self.V)).squeeze(-1) # dim: batch_size,seq_len

        self.attention_weights = self.softmax(self.attention_score)

        self.context = torch.bmm(self.attention_weights.unsqueeze(1), key).squeeze(1)

        return self.context
    


X = torch.randint(0,4,(2,5))
Y = torch.randn(2,5)
enc = Encoder(10,6,4)
ba = Bahdanau_Attention(20,5,3)

print(f"coming: {ba(enc(X),Y).shape}")

print("expected: 2X20")

    
        

    