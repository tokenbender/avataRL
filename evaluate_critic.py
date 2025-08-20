"""
Evaluate a critic model's cross-entropy loss on validation data

Usage:
    # Run locally (recommended for evaluation)
    python evaluate_critic.py
    
    # With custom checkpoint
    python evaluate_critic.py checkpoint_path=out/ckpt_custom.pt
    
    # With custom settings
    python evaluate_critic.py batch_size=16 eval_batches=100
    
Note: This script is designed to run locally, not via modal.
For modal deployment, you would need a separate modal wrapper like modal_train.py
"""
import os
import pickle
import numpy as np
import torch
from contextlib import nullcontext
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration - can be overridden from command line or config file
# -----------------------------------------------------------------------------
checkpoint_path = "out/ckpt_big_critic_300.pt"  # Path to critic checkpoint
data_dir = "data/openwebtext"  # Dataset directory
batch_size = 12  # Batch size for evaluation
block_size = 1024  # Context length
eval_batches = 50  # Number of batches to evaluate
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
seed = 1337
compile = False  # use PyTorch 2.0 to compile the model to be faster

# Create config dictionary for logging (similar to avatarl.py)
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]

# Handle configurator - skip if running via modal
import sys
if len(sys.argv) > 1 and sys.argv[1] == "run":
    # Running via modal, skip configurator since it will try to open("run")
    print("Running via modal, skipping configurator")
else:
    exec(open("configurator.py").read())  # overrides from command line or config file
    
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# Set random seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set up dtype and autocast context
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

def load_model(checkpoint_path):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint - but only what we need!
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract only what we need for inference
    checkpoint_model_args = checkpoint["model_args"]
    state_dict = checkpoint["model"]
    
    # Free the checkpoint dict immediately - we don't need optimizer states!
    del checkpoint
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    # Create model with checkpoint config
    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)
    # Remove unwanted prefix if present
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully - {checkpoint_model_args['n_layer']} layers, {checkpoint_model_args['n_embd']} dim")
    
    # Compile model if requested
    if compile:
        print("Compiling the model... (takes a ~minute)")
        import time
        t0 = time.time()
        model = torch.compile(model)
        print(f"Compilation completed in {time.time() - t0:.2f} seconds")
    
    return model, checkpoint_model_args

def get_batch(split="val"):
    """Get a batch of data"""
    data_path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def evaluate_model(model, num_batches=50):
    """Evaluate model's cross-entropy loss"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    print(f"Evaluating on {num_batches} batches...")
    
    with torch.no_grad():
        for i in range(num_batches):
            X, Y = get_batch("val")
            
            with ctx:
                logits, _ = model(X, Y)
                
                # Compute cross-entropy loss
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    Y.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += Y.numel()
            
            if (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{num_batches} - Running avg loss: {total_loss / total_tokens:.4f}")
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def main():
    """Main evaluation function"""
    print("=" * 50)
    print("Critic Model Evaluation on OpenWebText Validation Set")
    print("=" * 50)
    
    # Load model
    model, model_args = load_model(checkpoint_path)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Vocab size: {model_args['vocab_size']}")
    print(f"Block size: {model_args['block_size']}")
    
    # Check if we need to adjust block size
    if block_size != model_args['block_size']:
        print(f"Warning: Using block_size={block_size} but model was trained with {model_args['block_size']}")
        if block_size > model_args['block_size']:
            print(f"Error: Cannot use block_size larger than model's training block_size")
            return
    
    # Run evaluation
    print(f"\nEvaluating on {eval_batches} batches of size {batch_size}...")
    print(f"Total tokens to evaluate: {eval_batches * batch_size * block_size:,}")
    
    avg_loss, perplexity = evaluate_model(model, eval_batches)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {data_dir}/val.bin")
    print(f"Batches evaluated: {eval_batches}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print(f"Total tokens: {eval_batches * batch_size * block_size:,}")
    print("-" * 50)
    print(f"Cross-Entropy Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    main()