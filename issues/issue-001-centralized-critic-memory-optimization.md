# Issue #001: Centralized Critic Model with Pipelined Inference for Memory Optimization

## Problem Statement

In the current AvataRL DDP (Distributed Data Parallel) implementation, both the student and critic models are loaded on **every GPU** during distributed training. This leads to significant memory duplication:

- **Current behavior**: Each of 8 GPUs loads both models
  - Student model: ~250M parameters (wrapped in DDP, gradients synchronized)
  - Critic model: ~250M parameters (not wrapped in DDP, independent copy per GPU)
  - Total per GPU: ~500M parameters

- **Memory waste**: 7 × 250M = 1.75B redundant parameters
  - In bfloat16: ~14GB of wasted VRAM across the cluster
  - This memory could be used for larger batch sizes or longer sequences

### Code Evidence

From `avatarl.py:663-664`:
```python
# Load critic model for AvataRL
critic_model = load_critic_model(critic_model_path)
```

This happens on every rank, after DDP initialization but before DDP wrapping. The critic is never wrapped in DDP (only student model is at line 744-749).

## Proposed Solution: Centralized Critic with Pipeline

### Architecture

```
Master GPU (rank 0):
- Holds student model (250M params)
- Holds critic model (250M params) - ONLY copy in cluster
- Runs critic inference for ALL ranks
- Prefetches next batch while current batch trains

Worker GPUs (ranks 1-7):
- Only hold student model (250M params)
- Receive critic logits from master via broadcast
- Save 250M params (~2GB) of memory each
```

### Implementation Strategy

#### 1. Double Buffering Pattern

Use double buffering to hide communication latency:

```python
# Pseudo-code
critic_logits_current = None
critic_logits_next = None

# Before training loop starts
if master_process:
    X_all, Y_all = gather_all_batches()  # Get batch from all ranks
    critic_logits_current = critic_model(X_all)
    broadcast(critic_logits_current)
else:
    critic_logits_current = receive_broadcast()

# Training loop
while training:
    # Start async prefetch of next batch's critic logits
    if master_process:
        future = async_compute_critic(next_batch)
    
    # Use current critic logits for training
    train_with_critic_logits(critic_logits_current)
    
    # Swap buffers
    critic_logits_current = critic_logits_next
    critic_logits_next = await future
```

#### 2. Communication Pattern

1. All ranks send their local (X, Y) batches to master (all_gather)
2. Master computes critic forward pass for combined batch
3. Master broadcasts critic logits to all workers
4. Each worker uses its slice of the critic logits for training

#### 3. Optimization: Batch-Level Pipelining

Instead of gathering all batches at once, implement ring-based pipelining:
- Each GPU sends its batch to master in sequence
- Master processes and returns results immediately
- Reduces broadcast size by 8× (only sending relevant portion back)

## Benefits

1. **Memory Savings**:
   - Save 250M params × 7 GPUs = 1.75B parameters
   - ~14GB VRAM freed across worker GPUs (in bf16)
   - Enables larger batch sizes or longer sequences

2. **Compute Efficiency**:
   - Master GPU often has idle time during distributed backward pass
   - Can utilize this time for critic inference
   - Critic is forward-only (no backward), so ~1/3 the compute cost

3. **Scalability**:
   - Memory savings increase linearly with number of GPUs
   - Only one critic model regardless of world_size

## Challenges & Mitigations

### 1. Master GPU Load
- **Challenge**: Master does 8× more critic inference
- **Analysis**: Critic forward = ~1/3 of full forward+backward. So 8 × (1/3) = 2.67× compute
- **Mitigation**: Master GPU idle time during backward pass can absorb this

### 2. Communication Overhead  
- **Challenge**: Broadcasting critic logits to all ranks
- **Size**: `batch_size × seq_len × vocab_size × sizeof(bf16)`
- **For current config**: 8 × 1024 × 50304 × 2 bytes = ~824MB per iteration
- **Mitigation**: 
  - Use async broadcast overlapped with computation
  - Implement batch-level pipelining to reduce size by 8×
  - Leverage high-speed interconnect (InfiniBand/NVLink)

### 3. Synchronization Complexity
- **Challenge**: Ensuring critic logits ready before training step
- **Mitigation**: Prefetch one iteration ahead with double buffering

## Implementation Tasks

- [ ] Modify `load_critic_model()` to only load on rank 0
- [ ] Implement all_gather for batch collection from all ranks
- [ ] Add critic inference pipeline with double buffering
- [ ] Implement async broadcast for critic logits distribution  
- [ ] Add synchronization points in training loop
- [ ] Add config flag `centralized_critic = True/False`
- [ ] Benchmark memory savings and throughput impact
- [ ] Add fallback to current behavior if disabled

## Alternative Approaches Considered

1. **Model Parallelism**: Shard critic across GPUs - more complex, potentially worse latency
2. **Gradient Checkpointing**: Doesn't help since critic has no gradients
3. **Mixed Precision**: Already using bf16

## Priority

**HIGH** - This optimization could enable:
- Larger batch sizes (better GPU utilization)
- Longer sequences (better context modeling)
- More GPUs without OOM (better scaling)

## References

- Current implementation: `avatarl.py:52-90` (load_critic_model)
- DDP wrapping: `avatarl.py:744-749` (only student wrapped)
- Training loop: `avatarl.py:868-1065` (where critic inference happens)
- Config: `config/train_avatarl.py:79` (critic_model_path)