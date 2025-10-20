# Tokenizers

This directory contains implementations of various tokenization algorithms used in natural language processing.

## Overview

Tokenization is the process of breaking down text into smaller units (tokens) that can be processed by machine learning models. Different tokenization strategies have different trade-offs in terms of vocabulary size, out-of-vocabulary handling, and computational efficiency.

## Implemented Tokenizers

### BPE (Byte Pair Encoding)
- **Directory**: `BPE/`
- **Language**: Python
- **Description**: Subword tokenization algorithm that iteratively merges the most frequent character pairs

### BPE in C++
- **Directory**: `BPE In CPP/`
- **Language**: C++
- **Description**: High-performance C++ implementation of Byte Pair Encoding

## Byte Pair Encoding (BPE)

### Algorithm Overview

BPE is a data compression technique adapted for tokenization:

1. **Initialize**: Start with character-level vocabulary
2. **Count Pairs**: Find most frequent adjacent character pairs
3. **Merge**: Replace most frequent pair with new token
4. **Repeat**: Continue until desired vocabulary size is reached

### Key Features

#### Python Implementation (`BPE/`)
- **File**: `bpe.py` - Core BPE algorithm
- **File**: `main.py` - Usage example and testing
- **Features**:
  - Text encoding (string to token IDs)
  - Text decoding (token IDs to string)
  - Configurable vocabulary size
  - Merge table construction and storage

#### C++ Implementation (`BPE In CPP/`)
- **Performance**: Optimized for speed and memory efficiency
- **Build System**: Makefile for easy compilation
- **Features**:
  - Fast text processing
  - Memory-efficient implementation
  - Public API for integration

### Usage Examples

#### Python BPE
```python
from bpe import BPE

# Initialize with text and vocabulary size
text = "Hello world! This is a sample text."
vocab_size = 300
tokenizer = BPE(text, vocab_size)

# Encode text
tokens = tokenizer.encode("Hello world!")
print(tokens)  # [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33]

# Decode tokens
decoded = tokenizer.decode(tokens)
print(decoded)  # "Hello world!"
```

#### C++ BPE
```bash
# Build
make

# Run
./main
```

## Algorithm Details

### Merge Process
1. **Character Frequency**: Count frequency of all character pairs
2. **Best Pair Selection**: Choose pair with highest frequency
3. **Vocabulary Update**: Add new merged token to vocabulary
4. **Text Update**: Replace all instances of the pair with new token
5. **Iteration**: Repeat until vocabulary size limit reached

### Encoding Process
1. **Tokenization**: Split text into initial character tokens
2. **Iterative Merging**: Apply learned merges in order
3. **Token IDs**: Convert final tokens to numerical IDs

### Decoding Process
1. **ID to Token**: Convert token IDs back to token strings
2. **Concatenation**: Join tokens to reconstruct text
3. **Byte Decoding**: Convert bytes back to UTF-8 text

## Advantages of BPE

1. **Subword Units**: Handles out-of-vocabulary words gracefully
2. **Vocabulary Control**: Configurable vocabulary size
3. **Language Agnostic**: Works with any language or script
4. **Compression**: Reduces sequence length compared to character-level
5. **Morphology Aware**: Often learns meaningful subword units

## Use Cases

### Natural Language Processing
- **Machine Translation**: Reduces vocabulary size while maintaining meaning
- **Language Modeling**: Better handling of rare words
- **Text Generation**: Improved quality with subword tokens

### Modern Applications
- **GPT Models**: Used in GPT-2, GPT-3, and other transformer models
- **BERT**: Subword tokenization for better generalization
- **Neural Machine Translation**: Standard preprocessing step

## Implementation Comparison

| Feature | Python Implementation | C++ Implementation |
|---------|----------------------|-------------------|
| Speed | Moderate | Fast |
| Memory Usage | Higher | Lower |
| Ease of Use | High | Moderate |
| Integration | Easy (Python ecosystem) | Requires compilation |
| Customization | Easy to modify | Requires C++ knowledge |

## File Structure

```
Tokenizers/
├── BPE/
│   ├── bpe.py          # Core BPE algorithm
│   ├── main.py         # Usage example
│   └── input.txt       # Sample input text
└── BPE In CPP/
    ├── src/
    │   ├── main.cpp     # Main program
    │   ├── bpe.cpp      # BPE implementation
    │   ├── filereading.cpp  # File utilities
    │   └── include/
    │       └── main.h   # Header definitions
    ├── Makefile         # Build configuration
    └── README.md        # C++ specific documentation
```

## Performance Considerations

### Python Implementation
- Good for prototyping and research
- Easy integration with ML frameworks
- Suitable for moderate-sized datasets

### C++ Implementation
- Optimized for production use
- Handles large datasets efficiently
- Lower memory footprint
- Faster processing speed

## Future Extensions

Potential improvements and extensions:
- **SentencePiece Integration**: More advanced subword algorithms
- **Parallel Processing**: Multi-threaded tokenization
- **Vocabulary Pruning**: Remove infrequent tokens
- **Domain Adaptation**: Specialized vocabularies for different domains
