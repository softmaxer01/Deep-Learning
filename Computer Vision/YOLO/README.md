# YOLO (You Only Look Once)

Implementation of YOLO object detection algorithms.

## Overview

YOLO is a real-time object detection system that frames object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.

## Implemented Versions

### YOLO v1
- **Directory**: `yolo-v1/`
- **Description**: Original YOLO architecture
- **Key Innovation**: Single neural network predicts bounding boxes and class probabilities directly

## YOLO v1 Architecture

### Network Design
- **Backbone**: Modified CNN (similar to GoogLeNet)
- **Output**: 7×7×30 tensor
  - 7×7 grid cells
  - Each cell predicts 2 bounding boxes
  - Each box has 5 predictions (x, y, w, h, confidence)
  - 20 class probabilities per cell

### Key Components

1. **Convolutional Backbone**: Feature extraction using CNN layers
2. **Grid System**: Divides image into S×S grid
3. **Bounding Box Prediction**: Each grid cell predicts B bounding boxes
4. **Class Prediction**: Conditional class probabilities per grid cell

## Files Structure

```
YOLO/
└── yolo-v1/
    ├── model.py      # YOLO v1 architecture
    ├── dataset.py    # Dataset loading and preprocessing
    ├── loss.py       # YOLO loss function
    ├── train.py      # Training script
    └── utils.py      # Utility functions
```

## Key Features

### Model Architecture (`model.py`)
- Darknet-inspired backbone
- Convolutional layers with batch normalization
- Leaky ReLU activation
- Fully connected layers for final predictions

### Loss Function (`loss.py`)
- Multi-part loss combining:
  - Localization loss (bounding box coordinates)
  - Confidence loss (objectness score)
  - Classification loss (class probabilities)

### Dataset Handling (`dataset.py`)
- Pascal VOC format support
- Data augmentation
- Ground truth encoding for training

### Training (`train.py`)
- Complete training loop
- Validation metrics
- Model checkpointing

## Usage

### Training
```bash
cd yolo-v1
python train.py
```

### Model Configuration
```python
# Default YOLO v1 configuration
S = 7    # Grid size (7×7)
B = 2    # Boxes per grid cell
C = 20   # Number of classes (Pascal VOC)
```

## YOLO v1 Advantages

1. **Speed**: Real-time detection (45 FPS)
2. **Global Context**: Sees entire image during training and testing
3. **Unified Architecture**: Single network for detection
4. **Generalization**: Good performance on new domains

## YOLO v1 Limitations

1. **Spatial Constraints**: Each grid cell can only predict one class
2. **Small Objects**: Difficulty with small objects in groups
3. **Aspect Ratios**: Limited to learned aspect ratios
4. **Localization Errors**: Main source of errors

## Loss Function Details

The YOLO loss function consists of:

1. **Coordinate Loss**: Penalizes bounding box coordinate errors
2. **Size Loss**: Penalizes width and height errors (square root for scale invariance)
3. **Confidence Loss**: Penalizes confidence score errors
4. **Classification Loss**: Penalizes class prediction errors

## Paper Reference

"You Only Look Once: Unified, Real-Time Object Detection"
- Authors: Redmon et al.
- Conference: CVPR 2016

## Implementation Notes

- Uses leaky ReLU activation (α = 0.1)
- Batch normalization for training stability
- Multi-scale training for robustness
- Non-maximum suppression for post-processing
