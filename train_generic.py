"""
Generic training entry point for avataRL.
Supports both Modal and standalone training environments.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import math
from collections import defaultdict, deque
from tqdm import tqdm

from config import TrainingConfig, EnvironmentConfig, get_config, is_modal_environment


def setup_distributed():
    """Initialize distributed training environment"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, device


def cleanup_distributed():
    """Clean up distributed process group"""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg: str, rank: int):
    """Print message only on rank 0"""
    if rank == 0:
        print(msg)


def setup_cuda_optimizations():
    """Setup CUDA optimizations"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def ensure_dataset(path: str = "input.txt"):
    """Ensure dataset is available"""
    if not Path(path).exists():
        import requests
        DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading dataset to {path}...")
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        Path(path).write_text(response.text, encoding="utf-8")
        print(f"Dataset downloaded successfully")


def build_vocab(path: str = "input.txt"):
    """Build vocabulary from text file"""
    text = Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    
    def encode(s): 
        return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    
    def decode(t): 
        return "".join(itos[int(i)] for i in t)
    
    return encode, decode, len(chars), stoi, itos, text


def setup_wandb(config: TrainingConfig, env_config: EnvironmentConfig, rank: int, vocab_size: int, model_params: int):
    """Setup Weights & Biases logging"""
    if not env_config.use_wandb or rank != 0:
        return
    
    try:
        import wandb
        wandb.init(
            project=env_config.wandb_project,
            name=env_config.wandb_name or "generic-training",
            config={
                "model": {
                    "vocab_size": vocab_size,
                    "n_layer": config.n_layer,
                    "n_head": config.n_head,
                    "n_emb": config.n_emb,
                    "context_len": config.context_len,
                    "param_count": model_params,
                },
                "training": {
                    "batch_size": config.batch,
                    "micro_batch_size": config.micro_batch,
                    "grad_accum_steps": config.grad_accum,
                    "learning_rate": config.lr,
                    "total_iters": config.total_iters,
                },
                "grpo": {
                    "use_exhaustive": config.use_exhaustive,
                    "k_samples": config.k_samples,
                    "clip_ratio": config.clip_ratio,
                    "entropy_coef": config.entropy_coef,
                    "beta_kl": config.beta_kl,
                },
            }
        )
    except ImportError:
        print("Warning: wandb not available, logging disabled")


def save_checkpoint(model, optimizer, iteration: int, checkpoint_path: str, rank: int, additional_state: dict = None):
    """Save checkpoint (rank 0 only)"""
    if rank != 0:
        return
    
    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    state = {
        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    
    if additional_state:
        state.update(additional_state)
    
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def train_grpo_generic(config: TrainingConfig, env_config: EnvironmentConfig):
    """Generic GRPO training function"""
    from train import (
        GPT, DistributedContextualTextLoader, CurriculumManager,
        CharacterDifficultyAnalyzer, ConfidenceMonitor, AdaptiveKLController,
        generate_with_temperature, compute_rewards_multi_char,
        setup_ddp_with_gradient_bucketing, all_reduce_mean,
        sync_model_across_ranks, compute_ppo_loss, normalize_advantages
    )
    
    rank, world_size, local_rank, device = setup_distributed()
    setup_cuda_optimizations()
    
    ensure_dataset(env_config.data_path)
    encode, decode, vocab_size, stoi, itos, text = build_vocab(env_config.data_path)
    
    print_rank0(f"\n=== Starting Generic GRPO Training ===", rank)
    print_rank0(f"Training mode: {env_config.training_mode}", rank)
    print_rank0(f"World size: {world_size} GPUs", rank)
    print_rank0(f"Vocabulary size: {vocab_size}", rank)
    print_rank0(f"Dataset size: {len(text):,} characters", rank)
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    actor = GPT(vocab_size, config.n_layer, config.n_head, config.n_emb, config.context_len).to(device)
    actor_old = GPT(vocab_size, config.n_layer, config.n_head, config.n_emb, config.context_len).to(device)
    actor_old.load_state_dict(actor.state_dict())
    actor_old.eval()
    
    if world_size > 1:
        actor = setup_ddp_with_gradient_bucketing(actor, device, config.bucket_size_mb)
    
    optimizer = torch.optim.AdamW(actor.parameters(), lr=config.lr, weight_decay=0.01, betas=(0.9, 0.95))
    
    scheduler = None
    if config.use_lr_decay:
        if config.lr_decay_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.total_iters - config.warmup_iters, eta_min=config.min_lr
            )
    
    loader = DistributedContextualTextLoader(
        text=text,
        encode_fn=encode,
        batch_size=config.micro_batch,
        horizon=config.horizon,
        context_len=config.context_len,
        rank=rank,
        world_size=world_size,
        device=device
    )
    
    character_analyzer = CharacterDifficultyAnalyzer(text, stoi, vocab_size, device)
    curriculum_manager = CurriculumManager(
        device=device, 
        vocab_size=vocab_size, 
        total_iterations=config.total_iters,
        promotion_threshold=0.05, 
        ema_alpha=0.95
    )
    confidence_monitor = ConfidenceMonitor() if config.use_confidence_scaling else None
    
    setup_wandb(config, env_config, rank, vocab_size, sum(p.numel() for p in actor.parameters()))
    
    # Training loop
    print_rank0(f"Starting training for {config.total_iters} iterations...", rank)
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
    
    for it in tqdm(range(1, config.total_iters + 1), desc="Training", disable=(rank != 0)):
        actor.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        iter_metrics = defaultdict(list)
        
        
        ctx, ref_tok = loader.next()
        
        with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=getattr(torch, config.amp_dtype)):
            logits = actor(ctx)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), ref_tok.reshape(-1))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler:
            scheduler.step()
        
        if rank == 0 and it % config.log_interval == 0:
            print(f"Iter {it}: loss={loss.item():.4f}")
            
            if env_config.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "loss/total": loss.item(),
                        "training/iteration": it,
                    })
                except ImportError:
                    pass
        
        if config.save_intermediate_checkpoints and it % config.checkpoint_interval == 0:
            checkpoint_path = Path(env_config.checkpoint_dir) / f"checkpoint_iter{it}.pt"
            save_checkpoint(actor, optimizer, it, str(checkpoint_path), rank)
    
    if config.save_final_checkpoint:
        checkpoint_path = Path(env_config.checkpoint_dir) / "checkpoint_final.pt"
        save_checkpoint(actor, optimizer, config.total_iters, str(checkpoint_path), rank)
    
    print_rank0("Training completed!", rank)
    
    if env_config.use_wandb and rank == 0:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass
    
    cleanup_distributed()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generic avataRL Training")
    parser.add_argument("--mode", choices=["modal", "generic"], default="generic",
                       help="Training mode (default: generic)")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data-path", type=str, default="input.txt",
                       help="Path to training data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                       help="Directory for checkpoints")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="avataRL",
                       help="Wandb project name")
    parser.add_argument("--wandb-name", type=str,
                       help="Wandb run name")
    
    args = parser.parse_args()
    
    os.environ['TRAINING_MODE'] = args.mode
    os.environ['DATA_PATH'] = args.data_path
    os.environ['CHECKPOINT_DIR'] = args.checkpoint_dir
    if args.no_wandb:
        os.environ['USE_WANDB'] = 'false'
    if args.wandb_project:
        os.environ['WANDB_PROJECT'] = args.wandb_project
    if args.wandb_name:
        os.environ['WANDB_NAME'] = args.wandb_name
    
    config, env_config = get_config()
    
    print(f"Starting training in {env_config.training_mode} mode")
    print(f"Data path: {env_config.data_path}")
    print(f"Checkpoint dir: {env_config.checkpoint_dir}")
    print(f"Wandb enabled: {env_config.use_wandb}")
    
    if env_config.training_mode == "modal":
        try:
            from modal_runner import setup_modal_training
            runner = setup_modal_training(config, env_config)
            training_func = runner.create_training_function(
                lambda: train_grpo_generic(config, env_config)
            )
            runner.run_remote(training_func)
        except ImportError:
            print("Error: Modal is not available. Please install with: pip install modal")
            sys.exit(1)
    else:
        train_grpo_generic(config, env_config)


if __name__ == "__main__":
    main()
