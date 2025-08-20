import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_hiddens, embed_dim, vocab_size):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.rnn = nn.RNN(self.embed_dim, self.num_hiddens, bidirectional=True, batch_first=True)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
    
    def forward(self, X):
        self.emb = self.embedding(X)
        output, h = self.rnn(self.emb)
        return output  # batch_size, seq_len, encoder_hidden_size

class bahdanau_Attention(nn.Module):
    def __init__(self, seq_len, encoder_hidden_size, decoder_hidden_size, attention_dim):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.seq_len = seq_len
        self.attention_dim = attention_dim
        # learnable params
        self.W_e2a = nn.Parameter(torch.randn(self.attention_dim, self.encoder_hidden_size))
        self.W_d2a = nn.Parameter(torch.randn(self.attention_dim, self.decoder_hidden_size))
        self.v_w = nn.Parameter(torch.randn(self.attention_dim, 1))
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, keys, q):
        self.batch_size = keys.shape[0]
        self.U = torch.matmul(keys, self.W_e2a.T)
        # after matmul we will get V shape as batch_size,attention_dim but we want it batch_size,seq_len,attention
        self.V = torch.matmul(q, self.W_d2a.T).unsqueeze(1).expand(-1, self.seq_len, -1)
        # added U and V then tanh then we have the value vector shape as attention_dimX1 and we add a extra dim as the batch_size dim
        self.e = torch.matmul(torch.tanh(self.U + self.V), self.v_w.unsqueeze(0).expand(self.batch_size, -1, -1)).squeeze(-1)
        self.a = self.softmax(self.e)  # batch_size, seq_len
        # for context we add a dim to the attention score it can broadcast over the encoder_dim and we can do the element wise mul ad get batch_size,seq_len,encoder_hidden_size
        # because we are adding dim at the end and adding over the seq_len dim so we get:
        self.context = torch.sum(self.a.unsqueeze(-1) * keys, dim=1)  # batch_size, encoder_hidden_size
        return self.context

class Decoder(nn.Module):
    def __init__(self, num_hidden, embed_dim, num_vocab, seq_len, encoder_hidden_size, attention_dim):
        super().__init__()
        self.num_hidden = num_hidden
        self.embed_dim = embed_dim
        self.num_vocab = num_vocab
        
        # Create attention block
        self.attention_block = bahdanau_Attention(seq_len, encoder_hidden_size, num_hidden, attention_dim)
        
        self.embedding = nn.Embedding(self.num_vocab, self.embed_dim)
        self.rnn = nn.RNN(self.embed_dim + encoder_hidden_size, self.num_hidden, batch_first=True)
        
        self.output_projection = nn.Linear(self.num_hidden, self.num_vocab)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, encoder_outputs, target_sentence=None, max_length=50, start_token=1):
        batch_size = encoder_outputs.shape[0]
        device = encoder_outputs.device
        
        hidden = torch.zeros(1, batch_size, self.num_hidden, device=device)
        
        if target_sentence is not None:
            return self._forward_training(encoder_outputs, target_sentence, hidden)
        else:
            return self._forward_inference(encoder_outputs, hidden, max_length, start_token)
    
    def _forward_training(self, encoder_outputs, target_sentence, hidden):
        batch_size, target_len = target_sentence.shape
        outputs = []
        
        for t in range(target_len):
            current_token = target_sentence[:, t:t+1]
            prev_hidden = hidden.squeeze(0)  
            context = self.attention_block(encoder_outputs, prev_hidden)
            embedded = self.embedding(current_token)
            context_expanded = context.unsqueeze(1) 
            rnn_input = torch.cat([embedded, context_expanded], dim=-1)  
            rnn_output, hidden = self.rnn(rnn_input, hidden) 
            current_hidden = rnn_output.squeeze(1) 
            logits = self.output_projection(current_hidden) 
            outputs.append(logits)
        return torch.stack(outputs, dim=1)
    
    def _forward_inference(self, encoder_outputs, hidden, max_length, start_token):
        batch_size = encoder_outputs.shape[0]
        device = encoder_outputs.device

        current_token = torch.full((batch_size, 1), start_token, device=device)
        outputs = []
        
        for t in range(max_length):
            prev_hidden = hidden.squeeze(0)  
            context = self.attention_block(encoder_outputs, prev_hidden) 
            embedded = self.embedding(current_token) 
            context_expanded = context.unsqueeze(1)  
            rnn_input = torch.cat([embedded, context_expanded], dim=-1)  
            rnn_output, hidden = self.rnn(rnn_input, hidden) 
            current_hidden = rnn_output.squeeze(1)  
            logits = self.output_projection(current_hidden)
            probs = self.softmax(logits)
            outputs.append(logits)
            current_token = torch.argmax(probs, dim=-1, keepdim=True) 
        return torch.stack(outputs, dim=1)

def create_seq2seq_model(src_vocab_size, tgt_vocab_size, embed_dim, encoder_hidden,decoder_hidden, seq_len, attention_dim):
    encoder_hidden_size = encoder_hidden * 2
    encoder = Encoder(encoder_hidden, embed_dim, src_vocab_size)
    decoder = Decoder(decoder_hidden, embed_dim, tgt_vocab_size, seq_len,encoder_hidden_size, attention_dim)
    return encoder, decoder
