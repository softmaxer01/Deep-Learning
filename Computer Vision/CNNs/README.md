# Convolutional Neural Networks (CNNs)

This directory contains implementations of various CNN architectures from scratch using PyTorch.

## Implemented Models

### AlexNet
- **File**: `Models/alex_net.py`
- **Description**: Deep CNN that won ImageNet 2012 competition
- **Key Features**: 
  - 8 layers (5 convolutional, 3 fully connected)
  - ReLU activation functions
  - Dropout for regularization
  - Local Response Normalization

### ResNet (Residual Networks)
- **File**: `Models/resnet.py`
- **Description**: Deep residual networks with skip connections
- **Key Features**:
  - Residual blocks with skip connections
  - Batch normalization
  - Configurable depth (ResNet-18, ResNet-34, etc.)
  - Solves vanishing gradient problem

### LeNet
- **File**: `Models/le_net.py`
- **Description**: Classic CNN for handwritten digit recognition
- **Key Features**:
  - Simple architecture with 2 conv layers
  - Designed for MNIST dataset
  - Historical significance as early CNN

### GoogleNet (Inception)
- **File**: `Models/google_net.py`
- **Description**: Inception architecture with parallel convolutions
- **Key Features**:
  - Inception modules with multiple filter sizes
  - 1x1 convolutions for dimensionality reduction
  - Auxiliary classifiers for training

### MobileNet
- **File**: `Models/mobile_net.py`
- **Description**: Efficient CNN for mobile devices
- **Key Features**:
  - Depthwise separable convolutions
  - Reduced parameters and computation
  - Optimized for mobile and embedded systems

## Project Structure

```
CNNs/
├── Models/                 # Model implementations
│   ├── alex_net.py
│   ├── resnet.py
│   ├── le_net.py
│   ├── google_net.py
│   └── mobile_net.py
├── assets/                 # Training results and visualizations
├── data/                   # Dataset storage
├── main.py                 # Main training script
├── training.py             # Training utilities
├── inference.py            # Inference and testing
└── plotting.py             # Visualization tools
```

## Usage

### Training
```bash
python main.py
```

### Inference
```bash
python inference.py
```

## Datasets Supported

- **CIFAR-10**: 10-class image classification
- **MNIST**: Handwritten digit recognition
- Custom datasets can be added by modifying the data loading scripts

## Results

Training results and model performance metrics are saved in the `assets/` directory:
- Training/validation curves
- Inference examples
- Model accuracy metrics

## Requirements

- PyTorch
- torchvision
- matplotlib
- numpy

## Key Concepts Implemented

1. **Convolutional Layers**: Feature extraction through convolution operations
2. **Pooling Layers**: Spatial dimension reduction
3. **Batch Normalization**: Training stabilization
4. **Dropout**: Regularization technique
5. **Residual Connections**: Skip connections for deep networks
6. **Inception Modules**: Multi-scale feature extraction
