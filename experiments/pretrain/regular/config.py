"""Regular GPT pretrain configuration wrapper.

Loads the standard regular pretrain hyperparameters defined in
config/train_regular_pretrain.py so it can be passed explicitly to
train.py or avatarl.py.
"""

from config.train_regular_pretrain import *  # noqa: F401,F403

# Optionally tweak per-run metadata here
experiment_name = "regular_pretrain"
