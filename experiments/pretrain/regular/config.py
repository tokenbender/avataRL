"""Regular GPT pretraining hyperparameters.

This configuration mirrors the original defaults that used to live in
train.py, providing a standalone file that can be passed directly to
train.py or avatarl.py.
"""

import time
from pathlib import Path

# I/O
out_dir = str(Path(__file__).resolve().parent / "out")
experiment_name = "regular_pretrain_250M_adamw_big_critic"

# Evaluation cadence
eval_interval = 500
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# wandb logging
wandb_log = True
wandb_project = "nanogpt-avatarl"
wandb_run_name = "run_" + str(time.time())

# data
dataset = "shakespeare"
gradient_accumulation_steps = 8
batch_size = 64
block_size = 1024

# model
n_layer = 16
n_head = 16
n_embd = 1024
dropout = 0.0
bias = False

# optimizer
learning_rate = 6e-4
max_iters = None
max_epochs = 1
max_tokens = None
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
warmup_iters = 200
lr_decay_iters = 100000
min_lr = 6e-5

# DDP
backend = "nccl"

# system
device = "cuda"
dtype = "bfloat16"
compile = True
