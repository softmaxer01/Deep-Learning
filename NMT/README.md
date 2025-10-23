# Transformer from Scratch

A complete implementation of the Transformer architecture from the ground up, based on the "Attention Is All You Need" paper. This implementation includes training, validation, and inference capabilities for neural machine translation tasks.

## Features

- Complete Transformer implementation with encoder-decoder architecture
- Multi-head attention mechanism
- Positional encoding
- Layer normalization and residual connections
- Feed-forward networks
- Custom dataset handling for bilingual translation
- Training loop with validation and metrics
- Model checkpointing and resuming
- Tensorboard logging
- Greedy decoding for inference

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
transformer_from_scratch/
├── config.py          # Configuration parameters
├── dataset.py         # Dataset loading and preprocessing
├── model.py           # Transformer model implementation
├── train.py          # Training script with validation
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Usage

### Training

To start training the model, simply run:

```bash
python train.py
```

The script will:
1. Download the OPUS Books dataset for English-Italian translation
2. Build tokenizers for source and target languages
3. Create data loaders for training and validation
4. Initialize the Transformer model
5. Train the model with validation after each epoch
6. Save model checkpoints in the `weights/` directory
7. Log metrics to Tensorboard

### Configuration

Modify the parameters in `config.py` to customize training:

```python
{
    "batch_size": 8,        # Training batch size
    "lr": 1e-4,            # Learning rate
    "seq_len": 350,        # Maximum sequence length
    "num_epochs": 20,      # Number of training epochs
    "src_lang": "en",      # Source language
    "tgt_lang": "it",      # Target language
    "model_folder": "weights",          # Model checkpoint directory
    "model_basename": "tmodel_",        # Model file prefix
    "preload": None,       # Checkpoint to resume from
    "tokenizer_file": "tokenizer_{0}.json",  # Tokenizer file pattern
    "experiment_name": "run/tmodel"     # Tensorboard log directory
}
```

### Resuming Training

To resume training from a checkpoint, set the `preload` parameter in config:

```python
"preload": "10"  # Resume from epoch 10
```

### Monitoring Training

View training progress with Tensorboard:

```bash
tensorboard --logdir=run/tmodel
```

## Model Architecture

The implementation includes:

- **Input Embedding**: Converts token IDs to dense vectors
- **Positional Encoding**: Adds position information to embeddings
- **Multi-Head Attention**: Core attention mechanism
- **Feed-Forward Network**: Position-wise fully connected layers
- **Layer Normalization**: Normalizes layer inputs
- **Residual Connections**: Skip connections for gradient flow
- **Encoder Stack**: 6 encoder layers (configurable)
- **Decoder Stack**: 6 decoder layers (configurable)
- **Output Projection**: Maps to target vocabulary

## Dataset

The model is configured to train on the OPUS Books dataset for English-Italian translation. The dataset is automatically downloaded and processed.

### Custom Dataset

To use a different dataset, modify the `get_ds()` function in `train.py` to load your data. Ensure your dataset has the structure:

```python
{
    "translation": {
        "en": "English text",
        "it": "Italian text"
    }
}
```

## Validation Metrics

The model tracks several metrics during validation:

- **Character Error Rate (CER)**: Character-level accuracy
- **Word Error Rate (WER)**: Word-level accuracy  
- **BLEU Score**: Translation quality metric

## Key Features

- **Label Smoothing**: Reduces overfitting (0.1 smoothing factor)
- **Padding Mask**: Handles variable sequence lengths
- **Causal Mask**: Prevents decoder from seeing future tokens
- **Gradient Accumulation**: Efficient memory usage
- **Mixed Precision**: Optional for faster training
- **Model Checkpointing**: Automatic saving and loading

## Hardware Requirements

- GPU recommended for training (CUDA support)
- Minimum 8GB RAM
- ~2GB disk space for model and data

## Troubleshooting

1. **CUDA out of memory**: Reduce batch_size in config
2. **Import errors**: Install requirements.txt dependencies
3. **Dataset download fails**: Check internet connection
4. **Slow training**: Use GPU if available

## Contributing

Feel free to contribute improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational purposes. Please cite the original Transformer paper:

```
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). 
Attention is all you need. Advances in neural information processing systems, 30.
```
