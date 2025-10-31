from dataclasses import dataclass
import torch

@dataclass
class BertConfig:
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_hidden_layers: int = 6
    batch_size: int = 32
    max_position_embeddings: int = 128
    intermediate_size: int = 1024
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    vocab_size: int = 30522
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class BertDatasetconfig:
    max_length = 128
