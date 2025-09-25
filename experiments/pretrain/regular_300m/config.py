"""Regular GPT pretraining config targeting ~300M parameters."""

import time
from pathlib import Path

_CONFIG_PATH = Path(globals().get("_CONFIG_PATH", Path(__file__).resolve()))
_CONFIG_DIR = _CONFIG_PATH.parent

# I/O
out_dir = str(_CONFIG_DIR / "out")
experiment_name = "regular_pretrain_300M"

# Evaluation cadence
eval_interval = 500
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# wandb logging
wandb_log = True
wandb_project = "test_sweep"
wandb_run_name = "regular_300M_" + str(time.time())
wandb_dir = str(_CONFIG_DIR)

# data
dataset = "openwebtext"
gradient_accumulation_steps = 8
batch_size = 16
block_size = 1024

# model (≈300M params)
n_layer = 20
n_head = 16
n_embd = 1024
dropout = 0.0
bias = False

# optimizer
learning_rate = 3e-4
max_iters = None
max_epochs = None
max_tokens = 9_000_000_000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# dual optimizer settings
use_dual_optimizer = False
muon_lr = 0.05
muon_momentum = 0.95
muon_ns_steps = 5
adam_head_lr_mult = 36
adam_embed_lr_mult = 100
adam_scalar_lr = 0.04

# learning rate decay
decay_lr = True
warmup_iters = 300
lr_decay_iters = 150000
min_lr = 3e-5

# DDP
backend = "nccl"

# system
device = "cuda"
dtype = "bfloat16"
compile = True
