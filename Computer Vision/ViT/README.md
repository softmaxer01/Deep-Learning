# Vision Transformer (ViT)

Implementation of Vision Transformer architecture for image classification.

## Overview

Vision Transformer applies the transformer architecture directly to sequences of image patches. Instead of using convolutional layers, ViT divides an image into patches and processes them as a sequence, similar to how transformers process text tokens.

## Architecture

### Key Components

1. **Patch Embedding**: Divides input image into fixed-size patches and linearly embeds them
2. **Position Encoding**: Adds learnable position embeddings to patch embeddings
3. **Transformer Encoder**: Stack of multi-head self-attention and MLP blocks
4. **Classification Head**: Linear layer for final classification

### Model Configuration

```python
@dataclass
class ViTConfig:
    d_model = 768          # Hidden dimension
    P = 32                 # Patch size
    H = 224                # Image height
    W = 224                # Image width
    C = 3                  # Number of channels
    n_head = 12            # Number of attention heads
    layers = 12            # Number of transformer layers
    classes = 10           # Number of output classes
```

## Files

- **`model.py`**: Complete ViT implementation with all components
- **`test.ipynb`**: Jupyter notebook for testing and experimentation

## Key Features

- **Patch Encoding**: Converts 2D images to 1D sequences of patches
- **Multi-Head Self-Attention**: Captures relationships between image patches
- **Position Embeddings**: Maintains spatial information
- **Classification Token**: Special [CLS] token for classification
- **Layer Normalization**: Pre-norm architecture for stable training

## Usage

```python
from model import ViT, ViTConfig

# Create model
config = ViTConfig()
model = ViT(config)

# Forward pass
x = torch.randn(4, 3, 224, 224)  # Batch of images
logits = model(x)  # Classification logits
```

## Architecture Details

### Patch Embedding Process
1. Split image into non-overlapping patches
2. Flatten each patch into a vector
3. Linear projection to embedding dimension
4. Add position embeddings

### Transformer Blocks
- Multi-head self-attention
- Layer normalization (pre-norm)
- MLP with GELU activation
- Residual connections

### Classification
- Uses [CLS] token representation
- Linear projection to number of classes

## Advantages

1. **Scalability**: Scales well with data and compute
2. **Transfer Learning**: Pre-trained models transfer well across tasks
3. **Interpretability**: Attention maps show which patches the model focuses on
4. **Flexibility**: Can handle variable input sizes (with interpolated position embeddings)

## Paper Reference

"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- Authors: Dosovitskiy et al.
- Conference: ICLR 2021

## Implementation Notes

- Uses pre-norm architecture (LayerNorm before attention/MLP)
- Includes bias terms in linear layers for better performance
- Position embeddings are learnable parameters
- GELU activation in MLP layers
