# Transformers

This directory contains implementations of transformer-based models and architectures.

## Overview

Transformers are a neural network architecture that relies entirely on attention mechanisms to draw global dependencies between input and output. They have revolutionized natural language processing and are increasingly used in computer vision and other domains.

## Implemented Models

### GPT Implementation ("lets build gpt")
- **Directory**: `lets build gpt/`
- **Description**: Educational implementation of GPT (Generative Pre-trained Transformer)
- **Key Features**:
  - Decoder-only transformer architecture
  - Causal (masked) self-attention
  - Character-level tokenization
  - Training on small text corpus

## Project Structure

```
Transformers/
└── lets build gpt/
    ├── basic_self_attention.ipynb  # Self-attention tutorial
    ├── bigram_model.ipynb         # Simple bigram baseline
    ├── model.py                   # GPT implementation
    └── input.txt                  # Training text
```

## GPT Implementation Details

### Architecture Components

#### Self-Attention
```python
class SelfAttention(nn.Module):
    - Single attention head implementation
    - Causal masking for autoregressive generation
    - Scaled dot-product attention
```

#### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    - Multiple parallel attention heads
    - Concatenation and projection of head outputs
    - Improved representation capacity
```

#### Feed-Forward Network
```python
class FeedForwardNetwork(nn.Module):
    - Position-wise fully connected layers
    - ReLU activation with dropout
    - 4x hidden dimension expansion
```

#### Transformer Block
```python
class Block(nn.Module):
    - Multi-head attention + residual connection
    - Feed-forward network + residual connection
    - Layer normalization (pre-norm architecture)
```

### Model Configuration

```python
# Hyperparameters
context_length = 8      # Maximum sequence length
d_model = 384          # Embedding dimension
h = 6                  # Number of attention heads
n_layers = 6           # Number of transformer blocks
n_vocab = 65           # Vocabulary size
dropout = 0.2          # Dropout rate
```

## Key Features

### Causal Self-Attention
- **Masking**: Lower triangular mask prevents future information leakage
- **Autoregressive**: Model generates one token at a time
- **Parallel Training**: All positions computed simultaneously during training

### Position-wise Processing
- **No Recurrence**: Unlike RNNs, processes all positions in parallel
- **Position Encoding**: Uses learned positional embeddings
- **Permutation Invariant**: Attention is inherently position-agnostic

### Residual Connections
- **Skip Connections**: Help with gradient flow in deep networks
- **Layer Normalization**: Stabilizes training
- **Pre-norm Architecture**: LayerNorm before attention and FFN

## Educational Notebooks

### `basic_self_attention.ipynb`
- Step-by-step implementation of self-attention
- Visualization of attention patterns
- Understanding of attention mechanics

### `bigram_model.ipynb`
- Simple baseline model for comparison
- Character-level bigram language model
- Demonstrates the power of transformers vs simple models

## Usage Example

```python
from model import Model

# Initialize model
model = Model()

# Generate text
input_ids = torch.randint(0, n_vocab, (batch_size, context_length))
logits = model(input_ids)  # Shape: (batch_size, context_length, vocab_size)

# For generation (autoregressive)
generated = model.generate(start_tokens, max_length=100)
```

## Training Process

1. **Character-Level Tokenization**: Simple character-to-integer mapping
2. **Causal Language Modeling**: Predict next character given previous characters
3. **Cross-Entropy Loss**: Standard loss for language modeling
4. **Teacher Forcing**: Use ground truth during training

## Key Innovations of Transformers

### Attention Mechanism
- **Global Dependencies**: Can attend to any position in sequence
- **Parallel Computation**: All positions processed simultaneously
- **Interpretability**: Attention weights show what model focuses on

### Scalability
- **Parameter Scaling**: Performance improves with model size
- **Data Scaling**: Benefits from large datasets
- **Compute Scaling**: Efficient use of modern hardware (GPUs/TPUs)

### Transfer Learning
- **Pre-training**: Learn general language understanding
- **Fine-tuning**: Adapt to specific tasks
- **Few-shot Learning**: Perform tasks with minimal examples

## Advantages Over RNNs

1. **Parallelization**: No sequential dependency in computation
2. **Long-range Dependencies**: Direct connections between distant positions
3. **Training Speed**: Faster training due to parallelization
4. **Performance**: Better results on most NLP tasks

## Applications

### Natural Language Processing
- **Language Modeling**: GPT series models
- **Machine Translation**: Encoder-decoder transformers
- **Text Classification**: BERT-style models
- **Question Answering**: Reading comprehension tasks

### Computer Vision
- **Vision Transformer (ViT)**: Image classification
- **DETR**: Object detection
- **Image Generation**: DALL-E style models

### Other Domains
- **Protein Folding**: AlphaFold uses transformer components
- **Code Generation**: GitHub Copilot, CodeT5
- **Music Generation**: MuseNet, Jukebox

## Implementation Notes

- **Attention Scaling**: Divide by sqrt(head_dimension) for stability
- **Causal Masking**: Essential for autoregressive generation
- **Dropout**: Applied to attention weights and FFN outputs
- **Weight Initialization**: Proper initialization crucial for training
- **Gradient Clipping**: Helps with training stability

## Paper References

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Original transformer paper
   - Introduced self-attention mechanism

2. **"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019)
   - GPT-2 paper
   - Demonstrated scaling benefits

3. **"Improving Language Understanding by Generative Pre-Training"** (Radford et al., 2018)
   - Original GPT paper
   - Showed effectiveness of pre-training
