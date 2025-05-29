# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GPT-2 language modeling research codebase focused on experimenting with reinforcement learning algorithms for training transformer models. The primary focus is on GRPO (Group Relative Policy Optimization) as an alternative to standard supervised learning.

## Architecture

### Core Components

1. **Model Variants**:
   - Standard GPT-2 transformer architecture with causal attention
   - Character-level and token-level (tiktoken) implementations
   - Multi-head latent attention variant (`gpt2_mla.py`)

2. **Training Algorithms**:
   - **GRPO**: Main focus with 4 implementations
     - `char_grpo.py`: Character-level with dense rewards
     - `grpo.py`: Token-level using tiktoken
     - `grpo_bigram.py`: Modal cloud deployment with bigram reference
     - `grpo_char_gpu.py`: Optimized standalone version
   - **PPO**: Actor-critic implementation (`pretrain_ppo.py`)
   - **Supervised**: Standard language modeling (`train.py`)
   - **Denoising**: BERT-style masked training (`diff_sampling.py`)

3. **Key Design Patterns**:
   - Reference policy comparison (frozen bigram or random initialization)
   - KL divergence annealing from 0 to target beta over warmup period
   - Dense reward signals (+1 for correct predictions)
   - Group advantage estimation for policy optimization
   - Weight tying between input embeddings and output projection

## Common Commands

### Training

```bash
# Character-level GRPO training
python grpo_char_gpu.py --horizon 70 --partial_logprob 0.05 --beta_kl 0.001 --kl_warmup_tokens 200000

# Token-level GRPO with tiktoken
python grpo.py

# PPO training
python pretrain_ppo.py

# Standard supervised training
python train.py

# Modal cloud deployment
modal run grpo_bigram.py

# Use detached mode to prevent disconnections
modal run --detach grpo_bigram.py
```

### Modal Deployment Notes

When running `grpo_bigram.py` on Modal:

1. **Prerequisites**:
   - Install wandb locally: `pip install wandb` (required for Modal to parse imports)
   - Create wandb secret: `modal secret create wandb WANDB_API_KEY="<your-api-key>"`
   - Ensure Modal is up to date: `pip install --upgrade modal`

2. **Running Training**:
   - Use `--detach` flag to prevent training from stopping if local connection drops
   - Modal runs in a containerized environment with dependencies specified in the image
   - Monitor progress via wandb dashboard or Modal web interface

3. **Troubleshooting**:
   - If "Module not found" errors occur, install the module locally even though Modal installs it in container
   - Check logs with: `modal app logs <app-id>`
   - List running apps: `modal app list`

### Key Arguments (grpo_char_gpu.py)
- `--horizon`: Sequence length for training
- `--partial_logprob`: Weight for partial credit rewards
- `--beta_kl`: KL divergence coefficient
- `--kl_warmup_tokens`: Number of tokens for KL annealing
- `--reference`: Choice of reference policy ("bigram" or "random")
- `--reward_mode`: Reward function (+1 only or ±1)
- `--batch_size`: Training batch size
- `--seed`: Random seed for reproducibility

## Development Notes

### Dependencies
- PyTorch (primary framework)
- tiktoken (GPT tokenization)
- wandb (experiment tracking)
- modal (cloud deployment)
- numpy, tqdm, requests

### Data
- Default dataset: "input.txt" (tiny-Shakespeare)
- Automatically downloaded if missing
- Character-level encoding or tiktoken-based

### Experiment Tracking
- Wandb integration available (optional)
- Logs losses, rewards, KL divergence, and sample generations
- Checkpointing saves model weights periodically

### C++ Implementation
- `train.cpp` provides a libtorch-based implementation
- Requires PyTorch C++ API installation