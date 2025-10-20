# Recurrent Neural Networks (RNNs)

This directory contains implementations of various RNN architectures and their applications.

## Overview

Recurrent Neural Networks are designed to work with sequential data by maintaining hidden states that capture information from previous time steps. This makes them suitable for tasks involving sequences like language modeling, time series prediction, and sequence-to-sequence learning.

## Implemented Architectures

### Bidirectional RNNs
- **Directory**: `Bidirectional RNNs/`
- **Description**: RNNs that process sequences in both forward and backward directions
- **Key Features**:
  - Forward and backward RNN processing
  - Concatenated hidden states from both directions
  - Better context understanding for each time step

### Deep Recurrent Neural Networks
- **Directory**: `Deep Recurrent Neural Networks/`
- **Description**: Multi-layer RNN architectures
- **Key Features**:
  - Stacked RNN layers
  - Increased model capacity
  - Hierarchical feature learning

### Encoder-Decoder Architecture
- **Directory**: `Encoder-Decoder/`
- **Description**: Sequence-to-sequence models for translation and generation
- **Key Features**:
  - Encoder RNN processes input sequence
  - Decoder RNN generates output sequence
  - LSTM implementation included

## Key Concepts

### Bidirectional RNNs (`Bidirectional RNNs/model.py`)

```python
class BiRNN(nn.Module):
    - Forward RNN processes sequence left-to-right
    - Backward RNN processes sequence right-to-left
    - Outputs are concatenated for full context
```

**Advantages**:
- Access to both past and future context
- Better performance on tasks where full sequence is available
- Improved understanding of context-dependent patterns

**Use Cases**:
- Named entity recognition
- Part-of-speech tagging
- Sentiment analysis
- Any task where full sequence context is beneficial

### Deep RNNs

**Architecture**:
- Multiple RNN layers stacked vertically
- Output of layer n becomes input to layer n+1
- Increased representational capacity

**Benefits**:
- Hierarchical feature learning
- Better modeling of complex patterns
- Improved performance on challenging tasks

### Encoder-Decoder Models

**Components**:
1. **Encoder**: Processes input sequence into fixed-size context vector
2. **Decoder**: Generates output sequence from context vector
3. **Context Vector**: Compressed representation of input sequence

**Applications**:
- Machine translation
- Text summarization
- Question answering
- Sequence generation tasks

## File Structure

```
RNNs/
├── Bidirectional RNNs/
│   └── model.py           # BiRNN implementation
├── Deep Recurrent Neural Networks/
│   └── model.py           # Deep RNN implementation
└── Encoder-Decoder/
    ├── encoder-decoder.py # Seq2seq implementation
    └── lstm.py           # LSTM utilities
```

## Key Features Implemented

### Basic RNN Cell
- Vanilla RNN with tanh activation
- Hidden state computation: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
- Custom parameter initialization

### Bidirectional Processing
- Forward pass: processes sequence from start to end
- Backward pass: processes sequence from end to start
- Concatenation of forward and backward hidden states

### LSTM Integration
- Long Short-Term Memory cells for better gradient flow
- Forget, input, and output gates
- Cell state for long-term memory

## Usage Examples

### Bidirectional RNN
```python
from model import BiRNN

model = BiRNN(input_size=3, hidden_size=10)
x = torch.randn(110, 2, 3)  # (seq_len, batch_size, input_size)
outputs, hidden_states = model(x)
```

### Encoder-Decoder
```python
# Encoder processes input sequence
encoder_outputs = encoder(input_sequence)

# Decoder generates output sequence
decoder_outputs = decoder(encoder_outputs, target_sequence)
```

## Advantages and Limitations

### Advantages
- **Sequential Processing**: Natural fit for sequential data
- **Memory**: Can remember information from previous time steps
- **Variable Length**: Can handle sequences of different lengths
- **Bidirectional Context**: BiRNNs provide full sequence context

### Limitations
- **Vanishing Gradients**: Difficulty learning long-term dependencies
- **Sequential Computation**: Cannot be easily parallelized
- **Computational Cost**: Slower than feedforward networks
- **Memory Requirements**: Hidden states must be stored for backpropagation

## Applications

1. **Natural Language Processing**:
   - Language modeling
   - Machine translation
   - Sentiment analysis
   - Named entity recognition

2. **Time Series Analysis**:
   - Stock price prediction
   - Weather forecasting
   - Signal processing

3. **Speech Recognition**:
   - Acoustic modeling
   - Speech-to-text conversion

4. **Computer Vision**:
   - Video analysis
   - Action recognition
   - Image captioning

## Modern Alternatives

While RNNs were groundbreaking, modern architectures often provide better performance:
- **Transformers**: Better parallelization and long-range dependencies
- **CNNs**: For some sequence tasks, especially with fixed-length inputs
- **Hybrid Models**: Combining RNNs with attention mechanisms
