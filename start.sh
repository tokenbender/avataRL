#!/bin/bash

# start.sh - Run nanoGPT training locally with torchrun
# Supports both regular training (train.py) and AvataRL training (avatarl.py)
# Based on Modal configuration from modal_train.py

set -e  # Exit on error

echo "=== AvataRL Local Training Script ==="
echo

# Check if we're in the right directory
if [ ! -f "nanogpt/avatarl.py" ]; then
    echo "Error: Please run this script from the tokenopia root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Parse command line arguments
GPUS=${1:-1}  # Default to 1 GPU if not specified
MODE=${2:-"single"}  # Default to single node
SCRIPT=${3:-"avatarl"}  # Default to avatarl, can be "train" for regular training

echo "Configuration:"
echo "- GPUs per node: $GPUS"
echo "- Mode: $MODE"
echo "- Script: $SCRIPT"
echo

# Determine which training script to use
if [ "$SCRIPT" = "train" ]; then
    TRAIN_SCRIPT="train.py"
else
    TRAIN_SCRIPT="avatarl.py"
fi

# Install dependencies if needed
echo "=== Checking Dependencies ==="
python -c "import torch; print(f'PyTorch {torch.__version__} installed')" || {
    echo "Installing PyTorch and dependencies..."
    pip install torch==2.6.0 \
        transformers==4.51.3 \
        datasets==3.6.0 \
        tiktoken==0.9.0 \
        wandb==0.19.11 \
        tqdm==4.67.1
}

# Check for data
DATA_DIR="data/openwebtext"
if [ ! -f "$DATA_DIR/train.bin" ] || [ ! -f "$DATA_DIR/val.bin" ]; then
    echo
    echo "=== Preparing Data ==="
    echo "OpenWebText data not found. Preparing..."
    
    # Create data directory if it doesn't exist
    mkdir -p $DATA_DIR
    
    # Run data preparation
    python $DATA_DIR/prepare.py
fi

# Check for teacher model (only needed for AvataRL)
if [ "$SCRIPT" = "avatarl" ]; then
    TEACHER_MODEL="out/ckpt_teacher_25M.pt"
    if [ ! -f "$TEACHER_MODEL" ]; then
        echo
        echo "WARNING: Teacher model not found at $TEACHER_MODEL"
        echo "AvataRL requires a pre-trained teacher model."
        echo "Please ensure the teacher model exists or update the path in config/train_avatarl.py"
        echo
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Set environment variables for better performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Optional: Disable Infiniband if not available
if [ "$DISABLE_IB" = "1" ]; then
    export NCCL_IB_DISABLE=1
    echo "Infiniband disabled"
fi

# Run training
echo
if [ "$SCRIPT" = "train" ]; then
    echo "=== Starting Regular Training ==="
else
    echo "=== Starting AvataRL Training ==="
fi
echo

case $MODE in
    "single")
        if [ $GPUS -eq 1 ]; then
            # Single GPU - run directly with python
            echo "Running on single GPU..."
            python $TRAIN_SCRIPT --compile=False
        else
            # Multi-GPU single node
            echo "Running on $GPUS GPUs (single node)..."
            torchrun --standalone --nproc_per_node=$GPUS $TRAIN_SCRIPT
        fi
        ;;
    
    "master")
        # Multi-node master
        if [ -z "$MASTER_ADDR" ]; then
            MASTER_ADDR=$(hostname -I | awk '{print $1}')
        fi
        MASTER_PORT=${MASTER_PORT:-1234}
        NNODES=${NNODES:-2}
        
        echo "Starting master node..."
        echo "Master address: $MASTER_ADDR:$MASTER_PORT"
        echo "Total nodes: $NNODES"
        
        torchrun \
            --nproc_per_node=$GPUS \
            --nnodes=$NNODES \
            --node_rank=0 \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $TRAIN_SCRIPT
        ;;
    
    "worker")
        # Multi-node worker
        if [ -z "$MASTER_ADDR" ]; then
            echo "Error: MASTER_ADDR must be set for worker nodes"
            exit 1
        fi
        MASTER_PORT=${MASTER_PORT:-1234}
        NNODES=${NNODES:-2}
        NODE_RANK=${NODE_RANK:-1}
        
        echo "Starting worker node $NODE_RANK..."
        echo "Connecting to master: $MASTER_ADDR:$MASTER_PORT"
        
        torchrun \
            --nproc_per_node=$GPUS \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $TRAIN_SCRIPT
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: ./start.sh [num_gpus] [mode] [script]"
        echo "  num_gpus: Number of GPUs to use (default: 1)"
        echo "  mode: single, master, or worker (default: single)"
        echo "  script: avatarl or train (default: avatarl)"
        echo
        echo "Examples:"
        echo "  ./start.sh                      # Single GPU with avatarl.py"
        echo "  ./start.sh 1 single train       # Single GPU with train.py"
        echo "  ./start.sh 4                    # 4 GPUs with avatarl.py"
        echo "  ./start.sh 4 single train       # 4 GPUs with train.py"
        echo "  ./start.sh 8 master             # Master node with 8 GPUs, avatarl.py"
        echo "  ./start.sh 8 master train       # Master node with 8 GPUs, train.py"
        echo "  MASTER_ADDR=10.0.0.1 ./start.sh 8 worker  # Worker node"
        exit 1
        ;;
esac

echo
echo "=== Training Started ==="
echo "Check wandb for training progress: https://wandb.ai"
echo "Output directory: out-avatarl/"
echo "Checkpoints will be saved as: out-avatarl/ckpt_avatarl_pretrain.pt"