#!/usr/bin/env python3
"""
Generic training script for avataRL using torchrun.

Usage:
    # Single GPU
    python train.py
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 train.py
"""

import os
import torch
from config import DATA_ROOT
from train_core import main


if __name__ == "__main__":
    # Get distributed training info from torchrun environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # For torchrun compatibility
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    
    print(f"Starting training on rank {rank} of {world_size}")
    # Use local data path from config
    main(rank, world_size, data_root=DATA_ROOT)