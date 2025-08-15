"""
4-bit Critic Loading for nanoGPT/AvataRL
No Accelerate, no HuggingFace, just clean PyTorch + BitsAndBytes
"""

import torch
import torch.nn as nn
from model import GPTConfig, GPT

def quantize_linear_layer_4bit(linear_layer):
    """
    Quantize a single linear layer to 4-bit in place.
    Uses BitsAndBytes NF4 quantization.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("pip install bitsandbytes")
    
    # Create 4-bit linear layer with same dimensions
    linear_4bit = bnb.nn.Linear4bit(
        linear_layer.in_features,
        linear_layer.out_features,
        bias=linear_layer.bias is not None,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type='nf4'  # NormalFloat4 - best quality
    )
    
    # Copy the weights (will be quantized automatically)
    linear_4bit.weight = bnb.nn.Params4bit(
        linear_layer.weight.data.to(torch.bfloat16),
        requires_grad=False,
        compress_statistics=True,
        quant_type='nf4'
    )
    
    if linear_layer.bias is not None:
        linear_4bit.bias = linear_layer.bias
    
    return linear_4bit


def load_critic_model_4bit(checkpoint_path: str):
    """
    Load critic model with 4-bit quantization.
    Designed for nanoGPT's architecture - no frameworks needed.
    
    Memory savings:
    - 124M model: ~250MB → ~35MB
    - 1.5B model: ~3GB → ~400MB  
    - 7B model: ~14GB → ~1.8GB
    """
    print(f"Loading critic model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_model_args = checkpoint["model_args"]
    
    # Create model with checkpoint config
    gptconf = GPTConfig(**checkpoint_model_args)
    critic_model = GPT(gptconf)
    
    # Load state dict FIRST (important for correct weight shapes)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    critic_model.load_state_dict(state_dict)
    
    # Now quantize to 4-bit
    print("Quantizing to 4-bit...")
    
    # Track what we're quantizing
    layers_quantized = 0
    
    # Quantize transformer blocks
    for block_idx, block in enumerate(critic_model.transformer.h):
        # Attention layers
        if hasattr(block.attn, 'c_attn'):
            block.attn.c_attn = quantize_linear_layer_4bit(block.attn.c_attn)
            layers_quantized += 1
        if hasattr(block.attn, 'c_proj'):
            block.attn.c_proj = quantize_linear_layer_4bit(block.attn.c_proj)
            layers_quantized += 1
            
        # MLP layers
        if hasattr(block.mlp, 'c_fc'):
            block.mlp.c_fc = quantize_linear_layer_4bit(block.mlp.c_fc)
            layers_quantized += 1
        if hasattr(block.mlp, 'c_proj'):
            block.mlp.c_proj = quantize_linear_layer_4bit(block.mlp.c_proj)
            layers_quantized += 1
    
    # DON'T quantize embeddings or layer norms (they're small and quality-critical)
    # Keep lm_head in FP16 for quality
    
    print(f"Quantized {layers_quantized} linear layers to 4-bit")
    
    # Move to GPU and set eval mode
    critic_model.cuda()
    critic_model.eval()
    
    # Disable gradients for critic
    for param in critic_model.parameters():
        param.requires_grad = False
    
    # Report memory usage
    memory_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"Critic loaded in 4-bit - Memory: {memory_gb:.2f}GB")
    
    return critic_model


def load_critic_model_flexible(checkpoint_path: str, use_4bit: bool = True):
    """
    Flexible loader - can do 4-bit or regular FP16/32.
    Drop-in replacement for your current load_critic_model.
    """
    if use_4bit:
        try:
            import bitsandbytes as bnb
            return load_critic_model_4bit(checkpoint_path)
        except ImportError:
            print("BitsAndBytes not installed, falling back to FP16")
            print("Install with: pip install bitsandbytes")
            use_4bit = False
    
    # Original FP16 path (your existing code)
    print(f"Loading critic model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    checkpoint_model_args = checkpoint["model_args"]
    
    gptconf = GPTConfig(**checkpoint_model_args)
    critic_model = GPT(gptconf)
    
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    critic_model.load_state_dict(state_dict)
    critic_model.cuda()
    critic_model.eval()
    
    for param in critic_model.parameters():
        param.requires_grad = False
    
    print(f"Critic loaded - Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    return critic_model


if __name__ == "__main__":
    # Test the 4-bit loading
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_critic_4bit.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    # Test both versions
    print("\n=== Testing FP16 Loading ===")
    critic_fp16 = load_critic_model_flexible(checkpoint_path, use_4bit=False)
    fp16_memory = torch.cuda.memory_allocated() / 1024**3
    
    # Clear GPU memory
    del critic_fp16
    torch.cuda.empty_cache()
    
    print("\n=== Testing 4-bit Loading ===")
    critic_4bit = load_critic_model_flexible(checkpoint_path, use_4bit=True)
    int4_memory = torch.cuda.memory_allocated() / 1024**3
    
    print(f"\n=== Memory Comparison ===")
    print(f"FP16: {fp16_memory:.2f}GB")
    print(f"4-bit: {int4_memory:.2f}GB")
    print(f"Reduction: {fp16_memory/int4_memory:.1f}x")
    
    # Quick inference test
    print("\n=== Testing Inference ===")
    test_input = torch.randint(0, 50304, (2, 32)).cuda()
    
    with torch.no_grad():
        output, _ = critic_4bit(test_input)
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print("Inference successful!")