# Research Papers

This directory contains important research papers organized by domain and architecture type.

## Directory Structure

### CNN Papers (`CNN-papers/`)
Collection of foundational convolutional neural network papers:

- **AlexNet.pdf**: "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al., 2012)
- **DenseNet.pdf**: "Densely Connected Convolutional Networks" (Huang et al., 2017)
- **GoggleNet.pdf**: "Going Deeper with Convolutions" (Szegedy et al., 2015)
- **LeNet.pdf**: "Gradient-Based Learning Applied to Document Recognition" (LeCun et al., 1998)
- **ResNet.pdf**: "Deep Residual Learning for Image Recognition" (He et al., 2016)
- **VGG.pdf**: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman, 2015)

### NLP Papers (`NLP/`)
Natural language processing foundational papers:

- **word2vec.pdf**: "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al., 2013)

### RNN Papers (`RNN-papers/`)
Recurrent neural network architectures and applications:

- **bidirrectionalrnn.pdf**: "Bidirectional Recurrent Neural Networks" (Schuster & Paliwal, 1997)
- **gru.pdf**: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (Cho et al., 2014)
- **lstm.pdf**: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- **seq2seq.pdf**: "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014)

### Transformer Papers (`Transformer/`)
Transformer architecture and attention mechanisms:

- **1706.03762v7.pdf**: "Attention Is All You Need" (Vaswani et al., 2017)

## Paper Summaries

### Convolutional Neural Networks

#### AlexNet (2012)
- **Impact**: Breakthrough in ImageNet competition, sparked deep learning revolution
- **Key Innovations**: Deep CNN, ReLU activation, dropout, data augmentation
- **Architecture**: 8 layers (5 conv, 3 FC), 60M parameters

#### VGG (2015)
- **Impact**: Showed importance of network depth
- **Key Innovations**: Very small (3×3) convolution filters, deep architecture
- **Architecture**: 16-19 layers, uniform architecture design

#### GoogLeNet/Inception (2015)
- **Impact**: Efficient deep networks with parallel convolutions
- **Key Innovations**: Inception modules, 1×1 convolutions, auxiliary classifiers
- **Architecture**: 22 layers, reduced parameters through efficient design

#### ResNet (2016)
- **Impact**: Enabled training of very deep networks (100+ layers)
- **Key Innovations**: Residual connections, skip connections, batch normalization
- **Architecture**: 50-152 layers, identity shortcuts

#### DenseNet (2017)
- **Impact**: Maximum information flow between layers
- **Key Innovations**: Dense connections, feature reuse, parameter efficiency
- **Architecture**: Dense blocks with growth rate

#### LeNet (1998)
- **Impact**: First successful CNN for practical applications
- **Key Innovations**: Convolutional layers, subsampling, gradient-based learning
- **Architecture**: 7 layers, designed for handwritten digit recognition

### Recurrent Neural Networks

#### LSTM (1997)
- **Impact**: Solved vanishing gradient problem in RNNs
- **Key Innovations**: Memory cells, gating mechanisms, forget gates
- **Applications**: Language modeling, machine translation, speech recognition

#### Bidirectional RNN (1997)
- **Impact**: Access to both past and future context
- **Key Innovations**: Forward and backward processing, context combination
- **Applications**: Speech recognition, protein secondary structure prediction

#### GRU (2014)
- **Impact**: Simplified alternative to LSTM
- **Key Innovations**: Gating units, reset and update gates, fewer parameters
- **Applications**: Machine translation, sequence modeling

#### Seq2Seq (2014)
- **Impact**: Framework for sequence-to-sequence learning
- **Key Innovations**: Encoder-decoder architecture, variable-length sequences
- **Applications**: Machine translation, text summarization, conversation

### Transformers and Attention

#### Attention Is All You Need (2017)
- **Impact**: Revolutionary architecture that replaced RNNs for many tasks
- **Key Innovations**: Self-attention, multi-head attention, positional encoding
- **Applications**: Machine translation, language modeling, computer vision

### Natural Language Processing

#### Word2Vec (2013)
- **Impact**: Efficient word embeddings that capture semantic relationships
- **Key Innovations**: Skip-gram and CBOW models, negative sampling
- **Applications**: Word similarity, analogy tasks, downstream NLP tasks

## Missing Papers (Recommendations)

Based on the implemented models in this repository, consider adding:

### Computer Vision
- **Vision Transformer**: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
- **YOLO**: "You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al., 2016)
- **MobileNet**: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (Howard et al., 2017)

### Natural Language Processing
- **GPT**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
- **GPT-2**: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2019)

### Tokenization
- **BPE**: "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016)
- **SentencePiece**: "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing" (Kudo & Richardson, 2018)

## Usage Notes

- Papers are organized by architectural family for easy reference
- Each paper corresponds to implementations found in the codebase
- PDFs can be referenced when studying the corresponding code implementations
- Useful for understanding theoretical foundations of implemented models

## Reading Recommendations

### For Beginners
1. Start with LeNet and AlexNet for CNN foundations
2. Read LSTM paper for RNN understanding
3. Progress to Attention Is All You Need for modern architectures

### For Advanced Study
1. Compare ResNet vs DenseNet approaches to deep networks
2. Study evolution from RNNs (LSTM/GRU) to Transformers
3. Understand attention mechanisms across different domains

### Implementation Study
- Use papers alongside code implementations
- Compare paper descriptions with actual code
- Understand design choices and trade-offs made in implementations
