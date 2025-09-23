"""AvataRL GPT pretrain configuration wrapper.

Re-exports the AvataRL-specific hyperparameters so both training scripts
can be launched with an explicit config path.
"""

from config.train_avatarl import *  # noqa: F401,F403

experiment_name = "avatarl_pretrain"
