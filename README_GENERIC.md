# avataRL - Generic Training Support

This document describes how to use the generic training system for avataRL, which supports both Modal and standalone training environments.

## Quick Start

### Generic Training (Standalone)

```bash
# Install dependencies
pip install -r requirements.txt

# Run training with default settings
python train_generic.py

# Run with custom settings
python train_generic.py --data-path ./data/input.txt --checkpoint-dir ./my_checkpoints --no-wandb
```

### Modal Training

```bash
# Install Modal
pip install modal

# Run on Modal
python train_generic.py --mode modal
```

## Configuration

The training system uses a configuration-based approach with the following hierarchy:

1. **Environment variables** (highest priority)
2. **Command line arguments**
3. **Default configuration values**

### Environment Variables

```bash
# Training mode
export TRAINING_MODE=generic  # or "modal"

# Model configuration
export CONTEXT_LEN=32
export HORIZON=8
export BATCH=16384
export MICRO_BATCH=512
export TOTAL_ITERS=2000
export LR=3e-4
export N_LAYER=6
export N_HEAD=8
export N_EMB=512

# Paths
export DATA_PATH=input.txt
export CHECKPOINT_DIR=./checkpoints

# Logging
export USE_WANDB=true
export WANDB_PROJECT=avataRL
export WANDB_NAME=my_experiment

# Distributed training
export N_GPUS=1
export GPU_TYPE=A100
```

### Command Line Arguments

```bash
python train_generic.py \
    --mode generic \
    --data-path ./data/input.txt \
    --checkpoint-dir ./checkpoints \
    --wandb-project my_project \
    --wandb-name my_experiment \
    --no-wandb
```

## File Structure

```
avataRL/
├── config.py              # Configuration system
├── modal_runner.py        # Modal-specific infrastructure
├── train_generic.py       # Generic training entry point
├── train.py              # Original training code (refactored)
├── requirements.txt       # Core dependencies
└── README_GENERIC.md     # This file
```

## Key Features

### Environment Detection
The system automatically detects whether it's running in Modal or generic mode based on:
- `TRAINING_MODE` environment variable
- Presence of Modal-specific environment variables
- Command line arguments

### Optional Dependencies
- **Modal**: Only required for Modal training mode
- **wandb**: Optional for experiment tracking
- **transformers**: Optional for GPT-2 reference model

### Flexible Checkpointing
- Modal mode: Uses Modal volumes (`/data/`)
- Generic mode: Uses local filesystem (configurable path)

### Distributed Training Support
- Modal mode: Uses Modal's torchrun wrapper
- Generic mode: Standard PyTorch distributed training

## Migration from Original train.py

The original `train.py` has been refactored to support both Modal and generic training:

1. **Configuration centralized** in `config.py`
2. **Modal dependencies made optional** with fallbacks
3. **Paths made configurable** (Modal volumes vs local filesystem)
4. **Logging made optional** (wandb can be disabled)

### Backward Compatibility

The original Modal training workflow continues to work:
- All Modal decorators and infrastructure preserved
- Same hyperparameters and training logic
- Existing checkpoints and data volumes compatible

## Examples

### Single GPU Training
```bash
python train_generic.py --mode generic
```

### Multi-GPU Training (Generic)
```bash
torchrun --nproc-per-node=4 train_generic.py --mode generic
```

### Modal Training with Custom Config
```bash
export BATCH=8192
export LR=1e-4
python train_generic.py --mode modal
```

### Training without wandb
```bash
python train_generic.py --no-wandb
```

## Troubleshooting

### Common Issues

1. **Modal not available**: Install with `pip install modal`
2. **Flash Attention errors**: Install manually or disable with `USE_FLASH_ATTN=false`
3. **CUDA out of memory**: Reduce `MICRO_BATCH` size
4. **Checkpoint path errors**: Ensure `CHECKPOINT_DIR` exists and is writable

### Debug Mode

Enable verbose logging:
```bash
export PYTHONPATH=.
python -u train_generic.py --mode generic 2>&1 | tee training.log
```

## Contributing

When adding new features:
1. Update `config.py` for new configuration options
2. Ensure Modal compatibility in `modal_runner.py`
3. Test both Modal and generic modes
4. Update this documentation
