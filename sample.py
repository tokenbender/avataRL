"""
Sample from a trained model
"""
import os
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from load_critic_4bit import load_critic_model_4bit

# -----------------------------------------------------------------------------
init_from = (
    "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
)
out_dir = "out"  # ignored if init_from is not 'resume'
experiment_name = "avatarl_pretrain_250M_adamw_big_critic"  # optional experiment name suffix for checkpoint files
use_4bit = False  # whether to load model with 4-bit quantization for memory efficiency
start = "tell me a story about a guy who used to be a teacher but now he is a rug picker."  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = (
    1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    16  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
repetition_penalty = 1.1  # 1.0 = no penalty, > 1.0 = penalize repetition
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0+ to compile the model to be faster
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
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

# model
if init_from == "resume":
    # init from a model saved in a specific directory
    # Construct checkpoint filename with experiment name suffix if available
    checkpoint_filename = "ckpt.pt" if not experiment_name else f"ckpt_{experiment_name}.pt"
    ckpt_path = os.path.join(out_dir, checkpoint_filename)
    
    # If experiment-specific checkpoint doesn't exist, try default
    if not os.path.exists(ckpt_path) and experiment_name:
        default_ckpt_path = os.path.join(out_dir, "ckpt.pt")
        if os.path.exists(default_ckpt_path):
            print(f"Experiment checkpoint '{checkpoint_filename}' not found, using default 'ckpt.pt'")
            ckpt_path = default_ckpt_path
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    if use_4bit:
        print(f"Loading checkpoint with 4-bit quantization from {ckpt_path}")
        model = load_critic_model_4bit(ckpt_path)
        print(f"Model loaded in 4-bit - memory usage reduced by ~75%")
    else:
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Extract only what we need and free the rest
        checkpoint_model_args = checkpoint["model_args"]
        state_dict = checkpoint["model"]
        
        # Free the checkpoint dict immediately - we don't need optimizer states!
        del checkpoint
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        gptconf = GPTConfig(**checkpoint_model_args)
        model = GPT(gptconf)
        
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
elif init_from.startswith("gpt2"):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# Always use GPT-2 encodings
print("Using GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, repetition_penalty=repetition_penalty)
            print(decode(y[0].tolist()))
            print("---------------")
