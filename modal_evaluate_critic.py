"""
Script to evaluate a critic model's cross-entropy loss on validation data

Usage:
    # Run with Modal (on GPU cloud)
    modal run modal_evaluate_critic.py
    modal run modal_evaluate_critic.py --checkpoint-path "out/ckpt_custom.pt"
    modal run modal_evaluate_critic.py --batch-size 16 --eval-batches 100
    
    # Run locally with Python
    python modal_evaluate_critic.py
    python modal_evaluate_critic.py --checkpoint_path out/ckpt_custom.pt
    python modal_evaluate_critic.py --batch_size 16 --eval_batches 100
"""

import os
import sys
import argparse
import numpy as np
import torch
from contextlib import nullcontext

# Check if we're running with Modal
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

def evaluate_critic_local(
    checkpoint_path="out/ckpt_wandb_logging_fix.pt",
    data_dir="data/openwebtext",
    batch_size=12,
    block_size=1024,
    eval_batches=50,
    compile=False,
):
    """Local evaluation function"""
    from model import GPTConfig, GPT
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    seed = 1337
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    
    # Load model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    
    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)
    
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully - {checkpoint_model_args['n_layer']} layers, {checkpoint_model_args['n_embd']} dim")
    
    # Compile if requested
    if compile:
        print("Compiling the model... (takes a ~minute)")
        import time
        t0 = time.time()
        model = torch.compile(model)
        print(f"Compilation completed in {time.time() - t0:.2f} seconds")
    
    # Load data
    data_path = os.path.join(data_dir, "val.bin")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Validation data not found at {data_path}")
    
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    print(f"Loaded validation data with {len(data):,} tokens")
    
    # Evaluation loop
    total_loss = 0.0
    total_tokens = 0
    
    print(f"Evaluating on {eval_batches} batches of size {batch_size}...")
    
    with torch.no_grad():
        for i in range(eval_batches):
            # Sample batch
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack(
                [torch.from_numpy((data[j : j + block_size]).astype(np.int64)) for j in ix]
            )
            y = torch.stack(
                [torch.from_numpy((data[j + 1 : j + 1 + block_size]).astype(np.int64)) for j in ix]
            )
            
            if device_type == "cuda":
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)
            
            with ctx:
                logits, _ = model(x, y)
                
                # Compute cross-entropy loss
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += y.numel()
            
            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{eval_batches} - Running avg loss: {total_loss / total_tokens:.4f}")
    
    # Calculate final metrics
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {data_path}")
    print(f"Batches evaluated: {eval_batches}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print(f"Total tokens: {eval_batches * batch_size * block_size:,}")
    print("-" * 50)
    print(f"Cross-Entropy Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("=" * 50)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": eval_batches * batch_size * block_size,
    }

# Modal setup only if available
if MODAL_AVAILABLE:
    # Modal setup
    cuda_version = "12.6.0"
    flavor = "devel"
    operating_sys = "ubuntu22.04"
    tag = f"{cuda_version}-{flavor}-{operating_sys}"
    
    LOCAL_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    REMOTE_CODE_DIR = "/root/"
    
    # Create Modal image with dependencies
    image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
        .pip_install(
            "torch==2.6.0",
            "numpy",
            "tqdm",
        )
        .add_local_dir(LOCAL_CODE_DIR, remote_path=REMOTE_CODE_DIR)
    )
    
    app = modal.App("evaluate-critic", image=image)
    
    # Volumes for data and checkpoints
    data_volume = modal.Volume.from_name("nanogpt-multinode-demo", create_if_missing=True)
    model_volume = modal.Volume.from_name("nanogpt-multinode-demo-model-output", create_if_missing=True)
    
    @app.function(
        gpu="H100",
        timeout=600,
        volumes={
            "/vol": data_volume,
            "/model": model_volume,
        },
    )
    def evaluate_critic_modal(
        checkpoint_path: str = "out/ckpt_wandb_logging_fix.pt",
        data_dir: str = "data/openwebtext",
        batch_size: int = 12,
        block_size: int = 1024,
        eval_batches: int = 50,
        compile: bool = False,
    ):
        """Modal evaluation function"""
        # Import model code
        sys.path.append('/root')
        from model import GPTConfig, GPT
        
        # Setup
        device = "cuda"
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        seed = 1337
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        
        # Check for checkpoint in modal volume first, then local
        modal_checkpoint_path = f"/model/{checkpoint_path}"
        local_checkpoint_path = f"/root/{checkpoint_path}"
        
        if os.path.exists(modal_checkpoint_path):
            actual_checkpoint_path = modal_checkpoint_path
            print(f"Loading checkpoint from modal volume: {modal_checkpoint_path}")
        elif os.path.exists(local_checkpoint_path):
            actual_checkpoint_path = local_checkpoint_path
            print(f"Loading checkpoint from local: {local_checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {modal_checkpoint_path} or {local_checkpoint_path}")
        
        # Load model
        print(f"Loading model from {actual_checkpoint_path}")
        checkpoint = torch.load(actual_checkpoint_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        
        gptconf = GPTConfig(**checkpoint_model_args)
        model = GPT(gptconf)
        
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully - {checkpoint_model_args['n_layer']} layers, {checkpoint_model_args['n_embd']} dim")
        
        # Compile if requested
        if compile:
            print("Compiling the model... (takes a ~minute)")
            import time
            t0 = time.time()
            model = torch.compile(model)
            print(f"Compilation completed in {time.time() - t0:.2f} seconds")
        
        # Setup data paths - check modal volume first
        modal_data_path = f"/vol/val.bin"
        local_data_path = f"/root/{data_dir}/val.bin"
        
        if os.path.exists(modal_data_path):
            data_path = modal_data_path
            print(f"Using data from modal volume: {modal_data_path}")
        elif os.path.exists(local_data_path):
            data_path = local_data_path
            print(f"Using data from local: {local_data_path}")
        else:
            raise FileNotFoundError(f"Validation data not found at {modal_data_path} or {local_data_path}")
        
        # Load data
        data = np.memmap(data_path, dtype=np.uint16, mode="r")
        print(f"Loaded validation data with {len(data):,} tokens")
        
        # Evaluation loop
        total_loss = 0.0
        total_tokens = 0
        
        print(f"Evaluating on {eval_batches} batches of size {batch_size}...")
        
        with torch.no_grad():
            for i in range(eval_batches):
                # Sample batch
                ix = torch.randint(len(data) - block_size, (batch_size,))
                x = torch.stack(
                    [torch.from_numpy((data[j : j + block_size]).astype(np.int64)) for j in ix]
                )
                y = torch.stack(
                    [torch.from_numpy((data[j + 1 : j + 1 + block_size]).astype(np.int64)) for j in ix]
                )
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
                
                with ctx:
                    logits, _ = model(x, y)
                    
                    # Compute cross-entropy loss
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        reduction='sum'
                    )
                    
                    total_loss += loss.item()
                    total_tokens += y.numel()
                
                if (i + 1) % 10 == 0:
                    print(f"  Batch {i+1}/{eval_batches} - Running avg loss: {total_loss / total_tokens:.4f}")
        
        # Calculate final metrics
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Print results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Dataset: {data_path}")
        print(f"Batches evaluated: {eval_batches}")
        print(f"Batch size: {batch_size}")
        print(f"Block size: {block_size}")
        print(f"Total tokens: {eval_batches * batch_size * block_size:,}")
        print("-" * 50)
        print(f"Cross-Entropy Loss: {avg_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        print("=" * 50)
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": eval_batches * batch_size * block_size,
        }
    
    @app.local_entrypoint()
    def modal_main(
        checkpoint_path: str = "out/ckpt_wandb_logging_fix.pt",
        data_dir: str = "data/openwebtext", 
        batch_size: int = 12,
        block_size: int = 1024,
        eval_batches: int = 50,
        compile: bool = False,
    ):
        """Modal entrypoint"""
        result = evaluate_critic_modal.remote(
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            batch_size=batch_size,
            block_size=block_size,
            eval_batches=eval_batches,
            compile=compile,
        )
        
        print(f"\nEvaluation complete! Loss: {result['loss']:.4f}, Perplexity: {result['perplexity']:.2f}")

# Local execution entrypoint
if __name__ == "__main__" and not (MODAL_AVAILABLE and "modal" in sys.argv[0]):
    parser = argparse.ArgumentParser(description="Evaluate a critic model")
    parser.add_argument("--checkpoint_path", type=str, default="out/ckpt_wandb_logging_fix.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str, default="data/openwebtext",
                        help="Directory containing validation data")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size for evaluation")
    parser.add_argument("--block_size", type=int, default=1024,
                        help="Context length")
    parser.add_argument("--eval_batches", type=int, default=50,
                        help="Number of batches to evaluate")
    parser.add_argument("--compile", action="store_true",
                        help="Compile model with PyTorch 2.0")
    
    args = parser.parse_args()
    
    # Run locally
    result = evaluate_critic_local(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        eval_batches=args.eval_batches,
        compile=args.compile,
    )
    
    print(f"\nEvaluation complete! Loss: {result['loss']:.4f}, Perplexity: {result['perplexity']:.2f}")