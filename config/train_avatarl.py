import time

# This is a stripped and simplified version of the main train.py script.
# It's meant to be used for AvataRL pre-training.

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"  # Changed to be inside /root/out Modal volume mount
experiment_name = "avatarl_pretrain_4000_adamw"
eval_interval = 500
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = "scratch"

# wandb logging
wandb_log = True
wandb_project = "nanogpt-avatarl"
wandb_run_name = "run_" + str(time.time())

# data
dataset = "openwebtext"  # Use openwebtext to match teacher model's training data
gradient_accumulation_steps = 32  # Increased from 8 to better utilize GPU (will be divided by world_size)
# Note: This will be divided by world_size in DDP, so effective steps per GPU = 4 with 8 GPUs
batch_size = 16  # Increased from 4 to better utilize GPU memory (adjust down if OOM)
block_size = 512  # Keeping at 512 to balance memory usage with dual models

# model - matching teacher model configuration from train.py
n_layer = 6  # Same as teacher model
n_head = 6   # Same as teacher model
n_embd = 384 # Same as teacher model (GPT-2 medium size)
dropout = 0.0  # No dropout for pretraining
bias = False

# optimizer
learning_rate = 6e-4  # Adjusted for better stability with AvataRL
max_iters = 5000  # Increased for longer training
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# dual optimizer settings
use_dual_optimizer = True  # Enable dual optimizer (Muon + Adam) matching teacher config
muon_lr = 0.05  # learning rate for Muon optimizer (hidden matrices)
muon_momentum = 0.95  # momentum for Muon optimizer
muon_ns_steps = 5  # Newton-Schulz iteration steps
adam_head_lr_mult = 36  # multiplier for head layer learning rate (lr * 36)
adam_embed_lr_mult = 100  # multiplier for embedding layer learning rate (lr * 100)
adam_scalar_lr = 0.04  # learning rate for scalar parameters

# learning rate decay
decay_lr = True
warmup_iters = 200  # Increased to match teacher model
lr_decay_iters = 5000  # Same as max_iters for full decay
min_lr = 6e-5  # ~1/10 of learning_rate per Chinchilla

# DDP settings
backend = "nccl"

# system
device = "cuda"
dtype = "bfloat16"
compile = True
profile = False  # Whether to profile the model (for performance analysis)
speedrun = False  # Whether to do a modded-nanogpt style speedrun test
bench = False  # Whether we're benchmarking - calculates MFU on each iteration

# -----------------------------------------------------------------------------
# AvataRL specific parameters
# -----------------------------------------------------------------------------
teacher_model_path = 'out/ckpt_teacher_25M.pt' # Path to the pre-trained teacher model

# PoE weights
reality_weight = 0.7
mentor_weight = 0.3

# Reality expert settings
label_smoothing_epsilon = 0.1  # 90% ground truth, 10% spread across vocabulary

# Reward settings
reward_scale = 100.0  # Scale probabilities to meaningful rewards

# Action space
top_k = 16  # Top-k tokens from both student and teacher

# Policy gradient
entropy_coefficient = 0.01  # 1% creativity bonus