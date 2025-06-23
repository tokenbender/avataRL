import modal
from pathlib import Path
from config import N_GPUS, GPU_TYPE

app = modal.App("perplexity_fix_2")

flash_attn_wheel = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("packaging", "wheel", "setuptools")
    .pip_install("numpy", "torch==2.5.0", "tqdm", "wandb", "requests", "matplotlib", "nvidia-ml-py3")
    .pip_install(flash_attn_wheel)
    .add_local_file("train_core.py", "/root/train_core.py")
    .add_local_file("config.py", "/root/config.py")
)

volume = modal.Volume.from_name("grpo-data", create_if_missing=True)
shakespeare_volume = modal.Volume.from_name("nanogpt-data", create_if_missing=False)


@app.function(
    gpu=f"{GPU_TYPE}:{N_GPUS}",
    volumes={
        "/data": shakespeare_volume,
    },
    timeout=60 * 60 * 6,
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_distributed():
    """Launch distributed training with torchrun"""
    import subprocess
    import os
    
    print(f"Launching distributed training on {N_GPUS} {GPU_TYPE} GPUs")
    
    wrapper_script = """#!/usr/bin/env python3
import os
import sys
sys.path.append('/root')  # Add root to path to find train_core and config

import torch
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
    main(rank, world_size)
"""
    
    with open("/tmp/train_wrapper.py", "w") as f:
        f.write(wrapper_script)
    
    # Set environment variables for proper output
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # Ensure unbuffered output
    
    # Run with torchrun, streaming output to console
    result = subprocess.run([
        "torchrun", f"--nproc-per-node={N_GPUS}", "/tmp/train_wrapper.py"
    ], env=env, stdout=None, stderr=None)  # None means inherit from parent
    
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args)
    
    return "Distributed training completed successfully!"


@app.local_entrypoint()
def main():
    """Entry point for Modal"""
    print("Launching Modal training job...")
    train_distributed.remote()