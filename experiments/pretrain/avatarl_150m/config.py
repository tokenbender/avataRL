"""AvataRL pretraining config targeting ~150M parameters."""

import time
from pathlib import Path

_CONFIG_PATH = Path(globals().get("_CONFIG_PATH", Path(__file__).resolve()))
_CONFIG_DIR = _CONFIG_PATH.parent

# -----------------------------------------------------------------------------
# I/O
out_dir = str(_CONFIG_DIR / "out")
experiment_name = "avatarl_pretrain_150M"

# Evaluation cadence
eval_interval = 200
log_interval = 10
eval_iters = 80
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# wandb logging
wandb_log = True
wandb_project = "test_sweep"
wandb_run_name = "avatarl_150M_" + str(time.time())
wandb_dir = str(_CONFIG_DIR)

# data
dataset = "openwebtext"
gradient_accumulation_steps = 8
batch_size = 12
block_size = 1024

# model (≈150M params)
n_layer = 16
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# optimizer
learning_rate = 4e-4
max_iters = None
max_epochs = None
max_tokens = 900_000_000
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
lr_decay_iters = 120000
min_lr = 4e-5

# system
device = "cuda"
dtype = "bfloat16"
compile = True
profile = False
speedrun = False
bench = False

# -----------------------------------------------------------------------------
# AvataRL specific parameters
# -----------------------------------------------------------------------------
critic_model_path = "out/ckpt_critic_30M.pt"
use_4bit_critic = True

reality_weight = 0.7
mentor_weight = 0.3
label_smoothing_epsilon = 0.1
reward_scale = 100.0
top_k = 4
entropy_coefficient = 0.01
max_reward_clamp = 1.5

muon_warmup_iters = 300
muon_warmup_start_momentum = 0.85
muon_warmup_end_momentum = 0.95

running_avg_alpha = 0.1

ddp_bucket_cap_mb = 512
