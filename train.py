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
from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from model import GPTConfig, GPT

# Whether to do a https://github.dev/KellerJordan/modded-nanogpt style speedrun test.
speedrun = os.environ.get("NANOGPT_SPEEDRUN", "false").lower() in ("true", "1")
# Whether we're benchmarking - this calculates MFU on each iteration.
bench = os.environ.get("NANOGPT_BENCH", "false").lower() in ("true", "1")
speedrun_target_eval_loss = 3.28
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
experiment_name = "regular_pretrain_250M_adamw_big_critic"  # optional experiment name suffix for checkpoint files
# every how many steps to evaluate val loss? 0 for only at the end
eval_interval = 500 if not speedrun else 125  # 125 is used in modded-nanogpt
log_interval = 10
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True  # disabled by default
wandb_project = "nanogpt-avatarl"
wandb_run_name = "run_" + str(time.time())  # 'run' + str(time.time())
# data
dataset = "shakespeare"
gradient_accumulation_steps = 8  # used to simulate larger batch sizes
batch_size = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 16
n_head = 16
n_embd = 1024
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate (10x for better Muon dual optimizer alignment)
# Training duration - can specify either max_iters OR max_epochs (not both)
# If max_epochs is set, max_iters will be calculated automatically based on dataset size
max_iters = None  # Maximum training iterations (set to None to use max_epochs instead)
max_epochs = 1  # Maximum training epochs (set to None to use max_iters instead)
max_tokens = None  # 50_000_000_000 to drive by tokens instead of epochs/iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# dual optimizer settings
use_dual_optimizer = False  # whether to use dual optimizer (Muon + Adam) like modded-nanogpt
muon_lr = 0.05  # learning rate for Muon optimizer (hidden matrices)
muon_momentum = 0.95  # momentum for Muon optimizer
muon_ns_steps = 5  # Newton-Schulz iteration steps
adam_head_lr_mult = 36  # multiplier for head layer learning rate (lr * 36)
adam_embed_lr_mult = 100  # multiplier for embedding layer learning rate (lr * 100)
adam_scalar_lr = 0.04  # learning rate for scalar parameters
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 200  # how many steps to warm up for
lr_decay_iters = 100000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
# Whether to profile the model. Profiling is already setup in bench.py but that doesn't
# work with DDP, we this train.py script also profiles.
profile: bool = os.environ.get("NANOGPT_PROFILE", "false").lower() in ("true", "1")
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
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
if 'max_tokens' in globals() and max_tokens is not None and max_iters is None:
    import math
    max_iters = math.ceil(max_tokens / tokens_per_iter)
    print(f"planning for ~{max_tokens:,} tokens ⇒ {max_iters:,} updates")
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# Calculate epoch information if dataset exists
data_dir = os.path.join("data", dataset)
try:
    if os.path.exists(os.path.join(data_dir, "train.bin")):
        # Import get_dataset_size function (will be defined later in the file)
        # For now, read dataset size directly here
        train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        train_tokens = len(train_data)
        del train_data  # Release memmap
        
        # Number of sequences we can sample from the dataset
        num_sequences = max(0, train_tokens - block_size)
        # Iterations per epoch = sequences / (batch_size * gradient_accumulation_steps * world_size)
        iterations_per_epoch = num_sequences // (batch_size * gradient_accumulation_steps * ddp_world_size)
        tokens_per_epoch = iterations_per_epoch * tokens_per_iter
        
        print(f"dataset has {train_tokens:,} tokens")
        print(f"iterations per epoch: {iterations_per_epoch:,}")
        print(f"tokens per epoch: {tokens_per_epoch:,}")
        
        # Handle max_epochs vs max_iters configuration
        if 'max_epochs' in globals() and max_epochs is not None:
            # Calculate max_iters from max_epochs
            max_iters = int(max_epochs * iterations_per_epoch)
            print(f"training for {max_epochs} epochs = {max_iters:,} iterations")
        else:
            # Show how many epochs the current max_iters represents
            if max_iters is not None:
                print(f"with max_iters={max_iters}, training for {max_iters / iterations_per_epoch:.2f} epochs")
    else:
        iterations_per_epoch = None
        print(f"dataset not found at {data_dir}, cannot calculate epoch information")
        if 'max_epochs' in globals() and max_epochs is not None:
            print(f"WARNING: max_epochs specified but cannot calculate iterations without dataset")
except Exception as e:
    iterations_per_epoch = None
    print(f"could not calculate epoch information: {e}")

# Safety check: ensure max_iters has a value
if max_iters is None:
    if 'max_epochs' in globals() and max_epochs is not None:
        # If we couldn't calculate iterations_per_epoch, use a reasonable default
        print(f"WARNING: Cannot calculate iterations from epochs without dataset. Using default max_iters=10000")
        max_iters = 10000
    else:
        # Both max_iters and max_epochs are None - use default
        print(f"WARNING: Neither max_iters nor max_epochs specified. Using default max_iters=10000")
        max_iters = 10000

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

# Dataset size tracking for epoch calculation
train_data_size = None
val_data_size = None

def get_dataset_size(split):
    """Get the size of a dataset split in tokens"""
    global train_data_size, val_data_size
    if split == "train" and train_data_size is not None:
        return train_data_size
    elif split == "val" and val_data_size is not None:
        return val_data_size
    
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    size = len(data)
    
    if split == "train":
        train_data_size = size
    else:
        val_data_size = size
    
    return size

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
current_epoch = 0  # Initialize epoch counter

# Always use GPT-2 vocab size
print("Using GPT-2 vocab_size of 50304 (50257 rounded up for efficiency)")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=50304,  # GPT-2 vocab_size, padded for efficiency
    dropout=dropout,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
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
    # Restore epoch if present in checkpoint (backward compatibility)
    if "current_epoch" in checkpoint:
        current_epoch = checkpoint["current_epoch"]
    else:
        current_epoch = 0
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
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
                losses[k] = loss.item()
        
        out[split] = losses.mean()
    
    model.train()
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
# Track another t0, separate from the original t0 which cares only about iter time.
# This t0 cares about overall training time, and is used in speedrun mode.
training_time_t0 = time.perf_counter()
with profiler:
    while True:
        # Update epoch counter
        if iterations_per_epoch is not None and iter_num > 0:
            current_epoch = iter_num / iterations_per_epoch
        
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
        if iter_num % eval_interval == 0 and iter_num > 0 and master_process:
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - training_time_t0)

            losses = estimate_loss()
            epoch_str = f" (epoch {current_epoch:.2f})" if iterations_per_epoch else ""
            print(
                f"step {iter_num}{epoch_str}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} train_time:{training_time_ms:.0f}ms"
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
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
                if iterations_per_epoch:
                    log_dict["epoch"] = current_epoch
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
                        "current_epoch": current_epoch,  # Save epoch information
                        "iterations_per_epoch": iterations_per_epoch,  # Save for consistency checks
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
                logits, loss = model(X, Y)
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
            tokens_seen = (iter_num + 1) * tokens_per_iter  # +1 because we log before incrementing iter_num
            lossf = loss.item() * gradient_accumulation_steps

            out_str = f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, tokens ~{tokens_seen:,}"

            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
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
        if iter_num >= max_iters:
            break

if ddp:
    destroy_process_group()
