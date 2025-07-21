"""Configuration constants for avataRL training."""
import os

# Hardware
N_GPUS = 8
GPU_TYPE = "H100"

# Model architecture
N_LAYER = 1
N_HEAD = 1
N_EMB = 64
CONTEXT_LEN = 16

# Data configuration
VOCAB_SIZE = 1024            # BPE vocabulary size
DATA_ROOT = "./data"         # Base data directory - wrappers handle deployment-specific paths
# MODAL_VOLUME removed - handled by train_modal.py wrapper

# Tokenizer paths (relative to DATA_ROOT or MODAL_VOLUME)
TOKENIZER_DIR = "tokenizers"
TOKENIZED_DATA_DIR = "shakespeare_tokens_bpe{vocab_size}"  # Will be formatted with vocab_size

# Training
BATCH = 16384
MICRO_BATCH = 256  # Reduced to match original for larger model
GRAD_ACCUM = BATCH // (MICRO_BATCH * N_GPUS)
EPOCHS = 0.2
DATASET_SIZE = 1_115_394
ITERS_PER_EPOCH = DATASET_SIZE // BATCH
TOTAL_ITERS = int(ITERS_PER_EPOCH * EPOCHS)
HORIZON = 8  # Increased from 1 to match original

# Learning rate
LR = 3e-3
USE_LR_DECAY = True
LR_DECAY_TYPE = "cosine"
MIN_LR = 1e-5
WARMUP_ITERS = max(1, int(TOTAL_ITERS * 0.02))

# GRPO algorithm
BETA_KL = 0.1
KL_WARM = int(DATASET_SIZE * 0.8)
KL_FREE_FRACTION = 0.1
CLIP_RATIO = 0.5
ENTROPY_COEF = 0.08
MIN_VARIANCE = 0.1

# Sampling
K_SAMPLES = 32
TEMPERATURE = 1.0

# Confidence scaling
USE_CONFIDENCE_SCALING = True
CONFIDENCE_WEIGHT = 0.7
CONFIDENCE_CLIP = 2.0
ENABLE_CONFIDENCE_PENALTY = True

# Optimization
BUCKET_SIZE_MB = 30
USE_FLASH_ATTN = True
USE_EXHAUSTIVE = True

# Dynamic batching (experimental)
USE_CONTINUOUS_BATCHING = False
MAX_BATCH_TOKENS = 8192
DYNAMIC_BATCH_SIZE = True

# Paged attention (experimental)
USE_PAGED_ATTENTION = False
PAGE_SIZE = 1024
MAX_PAGES_PER_SEQ = 8

# Checkpointing
SAVE_INTERMEDIATE_CHECKPOINTS = False
SAVE_FINAL_CHECKPOINT = True
CHECKPOINT_INTERVAL = 1000