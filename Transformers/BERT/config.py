from dataclasses import dataclass
import torch

@dataclass
class BertConfig:
    d_model = 256
    nh = 8
    n_layers = 6
    batch = 32
    seq_len = 128
    dff = 1024
    dp = 0.1
    vocab_size = 30522
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class BertDatasetconfig:
    max_length = 128
