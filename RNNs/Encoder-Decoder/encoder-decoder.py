import torch
import torch.nn as nn
from lstm import LSTM


class Encoder(nn.Module):
    def __init__(self, num_hiddens, num_vocab, embed_dim):
        super(Encoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_vocab = num_vocab
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(self.num_vocab, self.embed_dim)
        self.lstm = LSTM(self.embed_dim, self.num_hiddens, self.num_hiddens)

    def forward(self, feed_tokens):
        # feed_tokens shape: (seq_len, batch_size)
        emb = self.embedding(feed_tokens)  # (seq_len, batch_size, embed_dim)
        # LSTM returns (outputs, (final_h, final_c))
        _, (H, C) = self.lstm(emb)
        return H, C


class Decoder(nn.Module):
    def __init__(self, num_hiddens, embed_dim, vocab_size, H_C=None):
        super(Decoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstm = LSTM(self.embed_dim, self.num_hiddens, self.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.H_C = H_C

    def forward(self, target_tokens, ground_truth=None, training=True):
        # target_tokens shape: (seq_len, batch_size)
        emb = self.embedding(target_tokens)  # (seq_len, batch_size, embed_dim)
        
        # Forward pass through LSTM
        logits, _ = self.lstm(emb, self.H_C)
        
        if training and ground_truth is not None:
            # Convert list of logits to tensor and reshape for loss calculation
            # logits is a list of tensors, each of shape (batch_size, vocab_size)
            logits_stacked = torch.stack(logits)  # (seq_len, batch_size, vocab_size)
            logits_reshaped = logits_stacked.view(-1, self.vocab_size)  # (seq_len * batch_size, vocab_size)
            ground_truth_reshaped = ground_truth.view(-1)  # (seq_len * batch_size)
            loss = self.loss_fn(logits_reshaped, ground_truth_reshaped)
            return loss
        else:
            # Return logits for inference
            return logits


class seq2seq(nn.Module):
    def __init__(self, num_hiddens, num_vocab, embed_dim):
        super(seq2seq, self).__init__()
        self.encoder = Encoder(num_hiddens, num_vocab, embed_dim)
        self.decoder = Decoder(num_hiddens, embed_dim, num_vocab)

    def forward(self, src_tokens, tgt_tokens, ground_truth=None, training=True):
        # Encode source sequence
        H, C = self.encoder(src_tokens)
        
        # Set decoder's initial hidden state
        self.decoder.H_C = (H, C)
        
        if training and ground_truth is not None:
            # Training mode: compute loss
            loss = self.decoder(tgt_tokens, ground_truth, training=True)
            return loss
        else:
            # Inference mode: return predictions
            predictions = self.decoder(tgt_tokens, None, training=False)
            return predictions
