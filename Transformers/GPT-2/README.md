# GPT-2 Implementation

Implementation of GPT-2 (Generative Pre-trained Transformer 2) architecture from scratch.

## Overview

GPT-2 is a large-scale unsupervised language model that generates coherent text by predicting the next word in a sequence. It uses a transformer decoder architecture with causal (masked) self-attention.

## Architecture

### Key Components

1. **Token Embedding**: Converts input tokens to dense vectors
2. **Position Embedding**: Adds positional information to token embeddings
3. **Transformer Blocks**: Stack of masked self-attention and feed-forward layers
4. **Language Modeling Head**: Projects hidden states to vocabulary logits

### Model Configuration

```python
@dataclass
class GPTconfig:
    context_length: int = 256    # Maximum sequence length
    vocab_size: int = 65         # Vocabulary size
    n_layer: int = 6             # Number of transformer blocks
    n_head: int = 6              # Number of attention heads
    d_model: int = 384           # Hidden dimension
```

## Files

- **`gpt2model.py`**: Complete GPT-2 implementation
- **`test.ipynb`**: Jupyter notebook for testing and experimentation

## Key Features

### Causal Self-Attention
- Masked attention prevents looking at future tokens
- Multi-head attention for parallel processing
- Scaled dot-product attention mechanism

### Transformer Architecture
- Layer normalization (pre-norm style)
- Residual connections around attention and MLP blocks
- GELU activation in feed-forward networks
- Causal masking for autoregressive generation

### Components Breakdown

#### CausalSelfAttention
```python
class CausalSelfAttention(nn.Module):
    - Projects input to queries, keys, and values
    - Applies causal mask to prevent future information leakage
    - Multi-head attention with scaled dot-product
```

#### MLP (Feed-Forward Network)
```python
class MLP(nn.Module):
    - Two linear layers with GELU activation
    - 4x expansion in hidden dimension
    - Projects back to model dimension
```

#### Block (Transformer Block)
```python
class Block(nn.Module):
    - Layer normalization before attention and MLP
    - Residual connections around both components
    - Standard transformer decoder block
```

## Usage

```python
from gpt2model import GPT, GPTconfig

# Create model
config = GPTconfig()
model = GPT(config)

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)
```

## Architecture Details

### Attention Mechanism
- Causal masking ensures autoregressive property
- Multi-head attention allows focusing on different aspects
- Scaled attention prevents vanishing gradients

### Position Encoding
- Learned positional embeddings (not sinusoidal)
- Added to token embeddings before processing
- Allows model to understand sequence order

### Language Modeling
- Predicts next token given previous tokens
- Cross-entropy loss for training
- Greedy or sampling-based generation

## Key Innovations

1. **Scale**: Much larger than GPT-1 (1.5B parameters)
2. **Zero-shot Learning**: Performs tasks without fine-tuning
3. **Emergent Abilities**: Shows capabilities not explicitly trained for
4. **Transfer Learning**: Strong performance across various NLP tasks

## Training Considerations

- **Gradient Accumulation**: Handle large effective batch sizes
- **Learning Rate Scheduling**: Warmup and decay strategies
- **Regularization**: Dropout and weight decay
- **Mixed Precision**: Faster training with FP16

## Paper Reference

"Language Models are Unsupervised Multitask Learners"
- Authors: Radford et al.
- Organization: OpenAI
- Year: 2019

## Implementation Notes

- Uses GELU activation (tanh approximation)
- Pre-norm architecture (LayerNorm before attention/MLP)
- Causal attention mask prevents information leakage
- Learned position embeddings instead of sinusoidal
- No bias in final language modeling head
