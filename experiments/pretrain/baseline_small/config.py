"""Smaller baseline configuration for quick experiments.

Starts from the regular pretrain defaults and scales down a few
parameters to reduce memory/compute requirements.
"""

from config.train_regular_pretrain import *  # noqa: F401,F403

experiment_name = "baseline_small"

# Smaller model + batch for quick checks
n_layer = 12
n_head = 12
n_embd = 768
batch_size = 32
learning_rate = 3e-4
max_iters = 100
