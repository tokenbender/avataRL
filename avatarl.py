"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from model import GPTConfig, GPT

# Import all configuration variables
from config.train_avatarl import *

# -----------------------------------------------------------------------------
# Create config dictionary for logging
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# -----------------------------------------------------------------------------
# AvataRL helper functions
# -----------------------------------------------------------------------------


def load_critic_model(checkpoint_path: str):
    """
    Load a pre-trained critic model from checkpoint.
    
    Args:
        checkpoint_path: Path to the critic model checkpoint
        
    Returns:
        critic_model: Loaded critic model in eval mode on CUDA
    """
    print(f"Loading critic model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    checkpoint_model_args = checkpoint["model_args"]
    
    # Create model with checkpoint config
    gptconf = GPTConfig(**checkpoint_model_args)
    critic_model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint["model"]
    # Remove unwanted prefix if present
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    critic_model.load_state_dict(state_dict)
    critic_model.cuda()
    critic_model.eval()  # Set to eval mode
    
    # Disable gradients for critic
    for param in critic_model.parameters():
        param.requires_grad = False
    
    print(f"Teacher model loaded successfully - {checkpoint_model_args['n_layer']} layers, {checkpoint_model_args['n_embd']} dim")
    
    return critic_model


def compute_avatarl_loss(
    student_logits: Tensor, 
    critic_logits: Tensor, 
    ground_truth_tokens: Tensor,
    reality_weight: float = 0.7,
    mentor_weight: float = 0.3,
    label_smoothing_epsilon: float = 0.1,
    reward_scale: float = 100.0,
    top_k: int = 16,
    entropy_coefficient: float = 0.01
) -> Tensor:
    """
    Compute the AvataRL policy gradient loss with active token label smoothing.
    
    This implements a Product of Experts (PoE) reward model that combines:
    1. Reality Expert: Active token label-smoothed distribution
       - 90% probability to ground truth token
       - 10% spread ONLY across active tokens (student top-k + critic top-k)
       - Unlike standard label smoothing that spreads across all vocab_size tokens,
         this concentrates the smoothing mass on the ~33 tokens that actually matter
    2. Mentor Expert: Teacher model's distribution over plausible tokens
    
    The key innovation is that label smoothing epsilon is distributed only among
    tokens in the action space (2*top_k + 1 unique tokens), not wasted on the
    entire vocabulary. This provides stronger exploration signal for relevant alternatives.
    
    Args:
        student_logits: Student model's output logits. Shape: (batch_size, seq_len, vocab_size)
        critic_logits: Teacher model's output logits. Shape: (batch_size, seq_len, vocab_size)
        ground_truth_tokens: Ground-truth target tokens. Shape: (batch_size, seq_len)
        reality_weight: Weight for reality expert in PoE (default: 0.7)
        mentor_weight: Weight for mentor expert in PoE (default: 0.3)
        label_smoothing_epsilon: Label smoothing parameter distributed over active tokens only (default: 0.1)
        reward_scale: Scale factor for rewards (default: 100.0)
        top_k: Number of top tokens to consider from both student and critic (default: 16)
        entropy_coefficient: Coefficient for entropy regularization (default: 0.01)
        
    Returns:
        tuple: (loss, metrics_dict)
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # Validate input shapes
    assert student_logits.shape == critic_logits.shape, \
        f"Student and critic logits shape mismatch: {student_logits.shape} vs {critic_logits.shape}"
    assert ground_truth_tokens.shape == (batch_size, seq_len), \
        f"Ground truth shape mismatch: expected {(batch_size, seq_len)}, got {ground_truth_tokens.shape}"
    
    # Reshape to (batch_size * seq_len, vocab_size) for easier processing
    student_logits_flat = student_logits.view(-1, vocab_size)
    critic_logits_flat = critic_logits.view(-1, vocab_size)
    ground_truth_flat = ground_truth_tokens.view(-1)
    
    # --- Step 1: Define Expanded Action Space (Student + Teacher + Ground Truth) ---
    # Get student's top-k predictions
    _, student_top_k_indices = student_logits_flat.topk(top_k, dim=-1)
    
    # Get critic's top-k predictions
    _, critic_top_k_indices = critic_logits_flat.topk(top_k, dim=-1)
    
    # Combine student, critic, and ground truth indices
    # Shape: (batch_size * seq_len, top_k * 2 + 1)
    combined_indices = torch.cat([
        student_top_k_indices,
        critic_top_k_indices,
        ground_truth_flat.unsqueeze(1)
    ], dim=1)
    
    # VECTORIZED: Remove duplicates while preserving order (keep first occurrence)
    # This ensures ground truth is always included even if it appears in top-k
    # Use vectorized operations to avoid GPU-CPU synchronization
    batch_size_seq = combined_indices.size(0)
    max_actions = 33
    
    # Create a mask for unique values using broadcasting
    # For each position, mark first occurrence of each value
    expanded_indices = combined_indices.unsqueeze(2)  # [batch*seq, 33, 1]
    compared = expanded_indices == expanded_indices.transpose(1, 2)  # [batch*seq, 33, 33]
    # Lower triangular mask to keep only first occurrences
    first_occurrence_mask = torch.tril(compared).sum(dim=2) == 1  # [batch*seq, 33]
    
    # Apply mask and limit to max_actions
    action_space_indices = []
    for i in range(batch_size_seq):
        masked = combined_indices[i][first_occurrence_mask[i]]
        if masked.size(0) > max_actions:
            masked = masked[:max_actions]
        action_space_indices.append(masked.tolist())
    
    # --- Step 2: Construct the Ideal Reward Distribution (PoE Model) ---
    # Get mentor (critic) probabilities
    mentor_probs = torch.nn.functional.softmax(critic_logits_flat, dim=-1)
    
    # VECTORIZED: Create reality expert distribution with ACTIVE TOKEN label smoothing
    # Instead of spreading epsilon across all vocab_size tokens, concentrate it on active tokens only
    reality_probs = torch.zeros_like(mentor_probs)
    
    # Batch process all positions at once to avoid Python loops
    # First, set ground truth probabilities for all positions
    batch_indices = torch.arange(batch_size_seq, device=reality_probs.device)
    reality_probs[batch_indices, ground_truth_flat] = 1.0 - label_smoothing_epsilon
    
    # Process each position's active tokens (still need loop but optimized)
    for i, active_indices in enumerate(action_space_indices):
        if len(active_indices) > 1:
            active_indices_tensor = torch.tensor(active_indices, device=reality_probs.device, dtype=torch.long)
            num_active = len(active_indices)
            
            # Distribute epsilon mass uniformly among non-ground-truth active tokens
            smoothing_per_token = label_smoothing_epsilon / (num_active - 1)
            
            # Set smoothing for all active tokens first
            reality_probs[i, active_indices_tensor] = smoothing_per_token
            
            # Restore ground truth probability (overwrites the smoothing for GT token)
            reality_probs[i, ground_truth_flat[i]] = 1.0 - label_smoothing_epsilon
        elif len(active_indices) == 1:
            # Edge case: only ground truth is active
            reality_probs[i, ground_truth_flat[i]] = 1.0
    
    # Combine experts using weighted geometric mean
    # P_ideal ∝ P_reality^0.7 * P_mentor^0.3
    ideal_probs = torch.pow(reality_probs, reality_weight) * torch.pow(mentor_probs, mentor_weight)
    
    # CRITICAL FIX: Don't normalize over full vocabulary here!
    # We'll normalize only over action space tokens below
    
    # Pad sequences to same length for batch processing
    max_actions = max(len(seq) for seq in action_space_indices)
    padded_indices = torch.zeros((len(action_space_indices), max_actions), dtype=torch.long, device=student_logits.device)
    action_masks = torch.zeros((len(action_space_indices), max_actions), dtype=torch.bool, device=student_logits.device)
    
    for i, indices in enumerate(action_space_indices):
        padded_indices[i, :len(indices)] = torch.tensor(indices, device=student_logits.device)
        action_masks[i, :len(indices)] = True
    
    # --- Step 3: Generate Positive Rewards ---
    # Extract raw probabilities for action space tokens only
    action_probs_raw = ideal_probs.gather(1, padded_indices)
    
    # Normalize ONLY over action space (not entire vocabulary!)
    # This concentrates 100% probability mass on our ~32 tokens
    masked_action_probs = action_probs_raw * action_masks.float()
    action_probs_sum = masked_action_probs.sum(dim=1, keepdim=True)
    action_probs_normalized = masked_action_probs / (action_probs_sum + 1e-8)
    
    # Implement mean thresholding: only reward tokens above mean
    # Calculate mean probability across valid actions
    valid_action_counts = action_masks.sum(dim=1, keepdim=True).float()
    mean_prob = action_probs_sum / (valid_action_counts + 1e-8)
    
    # Only reward tokens that are above mean (creates sparse rewards)
    above_mean_mask = (action_probs_normalized > mean_prob) & action_masks
    action_rewards = torch.where(
        above_mean_mask,
        action_probs_normalized * reward_scale,  # Above mean: get scaled reward
        torch.zeros_like(action_probs_normalized)  # Below mean: get zero
    )
    
    # Apply mask to ensure padded positions stay zero
    action_rewards = action_rewards * action_masks.float()
    
    # CRITICAL: Clamp rewards to max 1.5 and rescale others proportionally
    # This prevents gradient explosion while maintaining relative reward differences
    max_reward_clamp = 1.5
    max_reward_per_seq = action_rewards.max(dim=1, keepdim=True)[0]
    
    # Only rescale if any reward exceeds the clamp threshold
    needs_rescaling = max_reward_per_seq > max_reward_clamp
    rescale_factor = torch.where(
        needs_rescaling,
        max_reward_clamp / (max_reward_per_seq + 1e-8),  # Proportional scaling factor
        torch.ones_like(max_reward_per_seq)  # No scaling needed
    )
    
    # Apply proportional rescaling to maintain relative differences
    action_rewards = action_rewards * rescale_factor
    
    # --- Step 4: Calculate Policy Gradient Loss ---
    # OPTIMIZED: Compute softmax once and derive log_softmax from it
    student_probs = torch.nn.functional.softmax(student_logits_flat, dim=-1)
    student_log_probs = torch.log(student_probs + 1e-10)  # Add small epsilon for numerical stability
    
    # Get student's log probabilities for the action space
    student_log_probs_for_actions = student_log_probs.gather(1, padded_indices)
    
    # Apply mask to log probs
    student_log_probs_for_actions = student_log_probs_for_actions * action_masks.float()
    
    # Policy gradient loss: -sum(log_policy * reward)
    # Detach rewards as they are fixed targets
    policy_gradient_loss = -(student_log_probs_for_actions * action_rewards.detach()).sum(dim=1)
    
    # Normalize by number of valid actions
    num_valid_actions = action_masks.sum(dim=1).float()
    policy_gradient_loss = policy_gradient_loss / (num_valid_actions + 1e-8)
    
    # --- Step 5: Add Entropy Regularization ---
    # Calculate entropy of student's full distribution (reuse cached probs and log_probs)
    student_entropy = -(student_probs * student_log_probs).sum(dim=-1)
    
    # Entropy bonus (negative because we minimize loss)
    entropy_bonus = entropy_coefficient * student_entropy
    
    # --- Step 6: Combine Losses ---
    # Total loss = policy gradient loss - entropy bonus
    total_loss = (policy_gradient_loss - entropy_bonus).mean()
    
    # --- Compute Additional Metrics for Logging ---
    # Calculate average rewards (only for valid actions)
    valid_rewards = action_rewards[action_masks]
    avg_reward = valid_rewards.mean().item() if valid_rewards.numel() > 0 else 0.0
    max_reward = valid_rewards.max().item() if valid_rewards.numel() > 0 else 0.0
    min_reward = valid_rewards.min().item() if valid_rewards.numel() > 0 else 0.0
    
    # Calculate average action space size
    avg_action_space_size = num_valid_actions.mean().item()
    
    # Return loss and metrics
    return total_loss, {
        'avg_reward': avg_reward,
        'max_reward': max_reward,
        'min_reward': min_reward,
        'avg_action_space_size': avg_action_space_size,
        'avg_entropy': student_entropy.mean().item()
    }

# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    print("Initializing DDP...")
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0, (
        f"{gradient_accumulation_steps=} must be divisible by {ddp_world_size=}"
    )
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
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

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched implementation
    a, b, c = (3.4445, -4.7750, 2.0315)  # Quintic coefficients for fast convergence
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    https://kellerjordan.github.io/posts/muon/
    
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.
    
    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    
    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    
    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        rank=0,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(
                params=[p for p in params if p.numel() == size],
                update_buffer=b,
                update_buffer_views=[b[i] for i in range(world_size)],
            )
            param_groups.append(group)
        super().__init__(param_groups, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            
            def update_prev():  # optimized Muon implementation
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-group["lr"]
                        * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5,
                    )
            
            for base_i in range(len(params))[:: self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(
                        g, steps=group["ns_steps"]
                    ).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------
# poor man's data loader
data_dir = os.path.join("data", dataset)


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    # Try to find checkpoint with experiment name first, then fall back to default
    checkpoint_filename = "ckpt.pt" if not experiment_name else f"ckpt_{experiment_name}.pt"
    ckpt_path = os.path.join(out_dir, checkpoint_filename)
    
    # If experiment-specific checkpoint doesn't exist, try default
    if not os.path.exists(ckpt_path) and experiment_name:
        default_ckpt_path = os.path.join(out_dir, "ckpt.pt")
        if os.path.exists(default_ckpt_path):
            print(f"Experiment checkpoint '{checkpoint_filename}' not found, using default 'ckpt.pt'")
            ckpt_path = default_ckpt_path
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = (
        block_size  # so that the checkpoint will have the right value
    )
model.to(device)

# Load critic model for AvataRL
critic_model = load_critic_model(critic_model_path)
print(f"Teacher model loaded from {critic_model_path}")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))

# optimizer
if use_dual_optimizer:
    # Dual optimizer setup (Muon + Adam)
    # Collect parameters and group them
    hidden_matrix_params = [
        p for n, p in model.named_parameters() 
        if p.ndim >= 2 and "embed" not in n and n != "lm_head.weight"
    ]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = []
    for n, p in model.named_parameters():
        if n == "lm_head.weight":
            head_params.append(p)
    
    # Initialize Adam optimizer for embeddings/head/scalars
    adam_params = [
        dict(params=head_params, lr=learning_rate * adam_head_lr_mult),  # Head: 36x higher LR
        dict(params=embed_params, lr=learning_rate * adam_embed_lr_mult),  # Embeddings: 100x higher LR
        dict(params=scalar_params, lr=adam_scalar_lr),  # Scalars: fixed LR
    ]
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    
    # Initialize Muon optimizer for hidden matrices
    rank = ddp_rank if ddp else 0
    world_size = ddp_world_size
    optimizer2 = Muon(
        hidden_matrix_params, lr=muon_lr, momentum=muon_momentum, 
        ns_steps=muon_ns_steps, rank=rank, world_size=world_size
    )
    
    # Create optimizer list
    optimizers = [optimizer1, optimizer2]
    
    # Store initial learning rates
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    
    if init_from == "resume":
        # Load optimizer states for dual optimizers
        if isinstance(checkpoint["optimizer"], list):
            for opt, opt_state in zip(optimizers, checkpoint["optimizer"]):
                opt.load_state_dict(opt_state)
        else:
            # Backward compatibility: old checkpoint with single optimizer
            print("Warning: Loading single optimizer checkpoint into dual optimizer setup")
            # Only load into the first optimizer as a fallback
            optimizer1.load_state_dict(checkpoint["optimizer"])
else:
    # Original single optimizer setup
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    optimizers = [optimizer]  # Wrap in list for consistency
    
    # Store initial learning rate
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    
    if init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])

checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    t0_compile = time.time()
    model = torch.compile(model)  # requires PyTorch 2.0
    compile_time = time.time() - t0_compile
    print(f"Compilation completed in {compile_time:.2f} seconds")

# wrap model into DDP container
if ddp:
    model = DDP(
        model,
        device_ids=[ddp_local_rank],
        bucket_cap_mb=512,
    )


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    critic_model.eval()  # Ensure critic is in eval mode too
    
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        ce_losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                student_logits, _ = model(X, Y)
                
                with torch.no_grad():
                    critic_logits, _ = critic_model(X, Y)
                
                loss, _ = compute_avatarl_loss(
                    student_logits, critic_logits, Y,
                    reality_weight=reality_weight,
                    mentor_weight=mentor_weight,
                    label_smoothing_epsilon=label_smoothing_epsilon,
                    reward_scale=reward_scale,
                    top_k=top_k,
                    entropy_coefficient=entropy_coefficient
                )
                losses[k] = loss.item()

                ce_loss = torch.nn.functional.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    Y.view(-1)
                )
                ce_losses[k] = ce_loss.item()
        
        out[split] = losses.mean()
        out[f"{split}_ce"] = ce_losses.mean()
    
    model.train()
    critic_model.eval()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

wait, warmup, active, repeat = 5, 5, 5, 2
num_steps = wait + warmup + active
if profile:
    print("Profiling NanoGPT model...")


def trace_handler(prof):
    print("Handling torch profiler trace...")
    task_id = os.environ["MODAL_TASK_ID"]
    rank = os.environ["RANK"]
    torch.profiler.tensorboard_trace_handler(f"/root/out/bench_log/{task_id}/{rank}")(
        prof
    )


profiler = (
    nullcontext()
    if not profile
    else torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        ),
        on_trace_ready=trace_handler,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,  # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False,  # only for torchscript models atm
    )
)

if speedrun and master_process:
    print("Speedrun mode enabled! 🏎️ 🏍️ 🐎 🏃‍♀️")

# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
training_time_ms = 0
# Running averages for AvataRL metrics
running_avg_reward = 0.0
# Track another t0, separate from the original t0 which cares only about iter time.
# This t0 cares about overall training time, and is used in speedrun mode.
training_time_t0 = time.perf_counter()
with profiler:
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        
        # Update learning rates for all optimizers
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group["lr"] = param_group["initial_lr"] * (lr / learning_rate)
        
        # Muon momentum warmup (only for dual optimizer mode)
        if use_dual_optimizer and iter_num < 300:
            for group in optimizer2.param_groups:
                frac = min(iter_num / 300, 1)
                group["momentum"] = (1 - frac) * 0.85 + frac * 0.95  # 0.85 -> 0.95

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - training_time_t0)

            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val_ce_loss {losses['val_ce']:.4f} train_time:{training_time_ms:.0f}ms"
            )
            if speedrun and losses["val"] < speedrun_target_eval_loss:
                print(
                    f"Speedrun target eval loss {speedrun_target_eval_loss} reached! 🏆"
                )
                # we must teardown or else the program will hang waiting for other processes
                if ddp:
                    destroy_process_group()
                break

            if wandb_log:
                log_dict = {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "val/ce_loss": losses["val_ce"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
                # Add AvataRL metrics if available
                if 'avatarl_metrics' in locals():
                    log_dict.update({
                        "avatarl/avg_reward": running_avg_reward,
                        "avatarl/instant_reward": avatarl_metrics['avg_reward'],
                    })
                wandb.log(log_dict)
            if (
                losses["val"] < best_val_loss or always_save_checkpoint
            ) and not speedrun:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": [opt.state_dict() for opt in optimizers] if use_dual_optimizer else optimizers[0].state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                        "use_dual_optimizer": use_dual_optimizer,  # Save optimizer type for loading
                    }
                    # Construct checkpoint filename with experiment name suffix
                    checkpoint_filename = "ckpt.pt" if not experiment_name else f"ckpt_{experiment_name}.pt"
                    checkpoint_path = os.path.join(out_dir, checkpoint_filename)
                    print(f"saving checkpoint to {checkpoint_path}")
                    torch.save(checkpoint, checkpoint_path)
            # start the clock again
            torch.cuda.synchronize()
            training_time_t0 = time.perf_counter()
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # In DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code.
                # Looking at the source of that context manager, it just toggles this variable.
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            with ctx:
                # Get student and critic logits - we need full sequence, so pass targets
                # but we'll ignore the returned loss and compute our own AvataRL loss
                student_logits, _ = model(X, Y)
                
                # Get critic logits for AvataRL
                with torch.no_grad():
                    critic_logits, _ = critic_model(X, Y)
                
                # Compute AvataRL loss
                loss, avatarl_metrics = compute_avatarl_loss(
                    student_logits, critic_logits, Y,
                    reality_weight=reality_weight,
                    mentor_weight=mentor_weight,
                    label_smoothing_epsilon=label_smoothing_epsilon,
                    reward_scale=reward_scale,
                    top_k=top_k,
                    entropy_coefficient=entropy_coefficient
                )
                
                # Calculate top-1 cross-entropy loss for logging
                with torch.no_grad():
                    top1_ce_loss = torch.nn.functional.cross_entropy(
                        student_logits.view(-1, student_logits.size(-1)), Y.view(-1)
                    )
                
                loss = (
                    loss / gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # clip the gradient
        if grad_clip != 0.0:
            # Unscale gradients for all optimizers
            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # step all optimizers and scaler if training in fp16
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        
        # flush the gradients as soon as we can, no need for this memory anymore
        model.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps

            out_str = f"iter {iter_num}: loss {lossf:.4f}, ce_loss {top1_ce_loss.item():.4f}, time {dt * 1000:.2f}ms"
            
            # Update running averages for AvataRL metrics
            if 'avatarl_metrics' in locals():
                alpha = 0.1  # smoothing factor
                running_avg_reward = (1 - alpha) * running_avg_reward + alpha * avatarl_metrics['avg_reward']
                out_str += f", reward {running_avg_reward:.3f}"

            if local_iter_num >= 5:
                # In AvataRL, we do more computation than standard training:
                # 1. Additional critic forward pass (roughly +0.5x FLOPs since no backward)
                # 2. Computing gradients for top_k actions instead of 1 per position
                # 
                # FIXED: Correct FLOP accounting for AvataRL
                # Standard training: 1x forward + 2x backward = 3x FLOPs
                # AvataRL: 1x student forward + 2x student backward + 1x critic forward = 4x FLOPs
                # The action space operations (gather/scatter) are memory-bound, not compute-bound
                critic_overhead = 4.0 / 3.0  # 1.33x for critic forward (4x total / 3x standard)
                avatarl_overhead = 1.5  # Conservative estimate for AvataRL-specific ops (softmax, gather, rewards)
                
                # Combined multiplier: ~2x instead of incorrect 12x
                avatarl_multiplier = critic_overhead * avatarl_overhead  # ~2.0x
                # This reflects actual compute: critic forward + loss computation overhead
                # NOT top_k forward passes (which don't happen)
                
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps * avatarl_multiplier, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
                out_str += f", mfu {running_mfu * 100:.2f}%"

            print(out_str)

        iter_num += 1
        local_iter_num += 1

        if profile:
            profiler.step()

        # termination conditions
        if iter_num > max_iters:
            break

if ddp:
    destroy_process_group()