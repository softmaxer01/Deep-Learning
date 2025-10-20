# Byte Pair Encoding (BPE) - Python Implementation

A Python implementation of the Byte Pair Encoding algorithm for subword tokenization.

## Overview

Byte Pair Encoding (BPE) is a subword tokenization algorithm that learns to merge the most frequent pairs of characters or character sequences. It provides a balance between character-level and word-level tokenization.

## Files

- **`bpe.py`**: Core BPE implementation class
- **`main.py`**: Example usage and testing script
- **`input.txt`**: Sample text file for testing

## Algorithm

### Training Phase
1. Initialize vocabulary with all individual bytes (0-255)
2. Tokenize training text into bytes
3. Iteratively find the most frequent pair of consecutive tokens
4. Merge the most frequent pair into a new token
5. Repeat until desired vocabulary size is reached

### Encoding Phase
1. Convert input text to bytes
2. Apply learned merges in the order they were learned
3. Return sequence of token IDs

### Decoding Phase
1. Convert token IDs back to byte sequences using vocabulary
2. Concatenate byte sequences
3. Decode bytes to UTF-8 text

## Usage

```python
from bpe import BPE

# Initialize with training text and vocabulary size
text = "This is a sample text for training BPE tokenizer."
vocab_size = 300
tokenizer = BPE(text, vocab_size)

# Encode new text
input_text = "This is a test."
tokens = tokenizer.encode(input_text)
print(f"Tokens: {tokens}")

# Decode tokens back to text
decoded_text = tokenizer.decode(tokens)
print(f"Decoded: {decoded_text}")

# Access vocabulary and merges
print(f"Vocabulary size: {len(tokenizer.vocab)}")
print(f"Number of merges: {len(tokenizer.merges)}")
```

## Class Interface

### BPE Class

```python
class BPE:
    def __init__(self, text, vocab_size):
        """
        Initialize BPE tokenizer.
        
        Args:
            text (str): Training text
            vocab_size (int): Target vocabulary size
        """
    
    def encode(self, text):
        """
        Encode text into token IDs.
        
        Args:
            text (str): Input text to encode
            
        Returns:
            list[int]: List of token IDs
        """
    
    def decode(self, ids):
        """
        Decode token IDs back to text.
        
        Args:
            ids (list[int]): List of token IDs
            
        Returns:
            str: Decoded text
        """
```

### Key Methods

#### `get_stats(ids)`
- Counts frequency of all adjacent token pairs
- Returns Counter object with pair frequencies

#### `merge(ids, pair, idx)`
- Replaces all instances of a token pair with new token ID
- Returns updated token sequence

#### `build_merge_table(tokens, iterations)`
- Learns merge operations from training data
- Stores merges in order of learning

#### `build_vocab()`
- Constructs vocabulary mapping from token IDs to byte sequences
- Includes base vocabulary (0-255) and learned merges

## Example Output

```python
# Training on sample text
text = "hello world hello"
tokenizer = BPE(text, vocab_size=280)

# Encoding
tokens = tokenizer.encode("hello")
# Output: [104, 101, 108, 108, 111] (if no merges learned for "hello")

# After learning merges, might become:
# Output: [256] (if "hello" becomes a single token)

# Decoding
decoded = tokenizer.decode([104, 101, 108, 108, 111])
# Output: "hello"
```

## Key Features

1. **Byte-Level Processing**: Handles any UTF-8 text robustly
2. **Configurable Vocabulary**: Set target vocabulary size
3. **Deterministic**: Same input always produces same output
4. **Efficient**: Uses Counter for fast pair frequency computation
5. **Reversible**: Perfect reconstruction through decode()

## Advantages

- **No OOV**: Can encode any text (byte-level fallback)
- **Compression**: Reduces sequence length vs character-level
- **Morphology**: Often learns meaningful subword units
- **Multilingual**: Works across different languages and scripts

## Limitations

- **Training Data Dependent**: Quality depends on training text
- **Greedy Algorithm**: May not find globally optimal merges
- **Sequential Processing**: Cannot be easily parallelized
- **Memory Usage**: Stores full vocabulary and merge table

## Use Cases

- **Language Modeling**: Preprocessing for neural language models
- **Machine Translation**: Subword units for better translation
- **Text Generation**: Improved quality with subword tokens
- **Cross-lingual Tasks**: Shared subword vocabulary across languages

## Performance Tips

1. **Training Data**: Use representative text for your domain
2. **Vocabulary Size**: Balance between compression and granularity
3. **Text Preprocessing**: Clean text before training for better merges
4. **Caching**: Store trained tokenizer to avoid retraining
