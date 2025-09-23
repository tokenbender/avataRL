"""Lightweight GPT pretraining hyperparameters for quick experiments."""

import time

# I/O
out_dir = "out"
experiment_name = "baseline_small"

# Evaluation cadence
eval_interval = 200
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# wandb logging
wandb_log = True
wandb_project = "nanogpt-avatarl"
wandb_run_name = "run_" + str(time.time())

# data
dataset = "shakespeare"
gradient_accumulation_steps = 4
batch_size = 32
block_size = 512

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1
bias = False

# optimizer
learning_rate = 3e-4
max_iters = 1000
max_epochs = None
max_tokens = None
weight_decay = 0.1
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
warmup_iters = 100
lr_decay_iters = 10000
min_lr = 3e-5

# DDP
backend = "nccl"

# system
device = "cuda"
dtype = "bfloat16"
compile = True
