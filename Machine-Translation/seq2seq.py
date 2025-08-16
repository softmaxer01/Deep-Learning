import torch
import torch.nn as nn
import spacy
from torchtext.datasets import Multi30k
import torch.optim as optim
from torchtext.data import Field,BucketIterator
import numpy
import random
from torch.utils.tensorboard import SummaryWriter


spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')


def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenizer_ger,lower=True,
               init_token='<sos>',eos_token='<eos>')

english = Field(tokenize=tokenizer_eng,lower=True,
               init_token='<sos>',eos_token='<eos>')


train_data,validation_data,test_data = Multi30k.splits(exts=('.de','.en'),fields=(german,english))


german.build_vocab(train_data,max_size = 10000,min_freq = 2)
english.build_vocab(train_data,max_size = 10000,min_freq = 2)



class Encoder(nn.Module):
    def __init__(self,input_size,embed_dim,hidden_size,num_layers,do):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(do)
        self.embedding = nn.Embedding(input_size,self.embed_dim)

        self.rnn = nn.LSTM(self.embed_dim,self.hidden_size,self.num_layers,dropout=do)

    def forward(self,X):
        # shape of X: (seq_len,batch_size)
        embedding = self.dropout(self.embedding(X))
        # shape of embedding: (seq_len,batch_size,embed_dim)
        outputs,(hidden,cell) = self.rnn(embedding)

        return hidden,cell

class Decoder(nn.Module):
    def __init__(self,input_size,embed_dim,hidden_size,
                 output_size,num_layers,do):
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        
        self.dropout = nn.Dropout(do)
        self.embedding = nn.Embedding(input_size,self.embed_dim)
        self.rnn = nn.LSTM(embed_dim,hidden_size,num_layers,dropout=do)

        self.fc = nn.Linear(hidden_size,output_size)


    def forward(self,x,hidden,cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        outputs,(hidden,cell) = self.rnn(embedding,(hidden,cell))
        prediction = self.fc(outputs.squeeze(0))

        return prediction, hidden, cell 

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self,src,tar,teacher_force_ratio = 0.5):
        batch_size = src.shape[1]
        # length of the target vocab
        tar_len = tar.shape[0]

        # vocab_size of target
        tar_vocab_size = len(english.vocab)
        outputs = torch.zeros(tar_len,batch_size,tar_vocab_size, device=src.device)
        hidden, cell = self.encoder(src)

        x = tar[0]

        for t in range(1,tar_len):
            output,hidden,cell = self.decoder(x,hidden,cell)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = tar[t] if random.random() < teacher_force_ratio else best_guess
        return outputs


"""
Flow:

Input to encoder LSTM:
shape: (seq_len, batch_size, embed_dim)
seq_len: number of tokens in source sentence (same after padding)
batch_size: number of sentences
embed_dim: size of word embedding

Output from encoder LSTM:
we only use the hidden and the cell
hidden shape: (num_layers, batch_size, hidden_size)
cell shape: (num_layers, batch_size, hidden_size)

Input to decoder LSTM (per time step):
we pass the batch of tokens from the same time step (i.e. ith token from each sentence)
x shape: (batch_size,)
unsqueeze → (1, batch_size)
after embedding → (1, batch_size, embed_dim)

Output from decoder LSTM (per time step):
after LSTM → (1, batch_size, hidden_size)
after linear layer → (batch_size, vocab_size)

Seq2Seq model:
first gets the last hidden & cell states from the encoder
feeds them into the decoder
decoder outputs, for each time step, the probabilities of the next word for the batch
final outputs shape: (tar_len, batch_size, vocab_size)

What it represents:
for each time step t → (batch_size, vocab_size) is the probability distribution of the next word
for each sentence in the batch at that time step

"""

#Hyper-parameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

load_model = False

input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding = 100
decoder_embedding = 100

hidden_size = 1024
encoder_do = 0.5
decoder_do = 0.5
num_layers = 2



train_itr, val_itr, test_itr = BucketIterator.splits(
    (train_data,validation_data,test_data),
    batch_size=batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src))

encoder_net = Encoder(input_size_encoder,encoder_embedding,hidden_size,num_layers,encoder_do)
decoder_net = Decoder(input_size_decoder,decoder_embedding,hidden_size,output_size,num_layers,decoder_do)


model = Seq2Seq(encoder_net,decoder_net)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_itr):
        inp_data = batch.src
        target = batch.trg

        output = model(inp_data,target)
        output = output[1:].reshape(-1,output.shape[2])
        target = target[1:].reshape(-1,)

        optimizer.zero_grad()
        loss = criterion(output,target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
        optimizer.step()
