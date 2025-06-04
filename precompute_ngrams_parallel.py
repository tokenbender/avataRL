#!/usr/bin/env python
"""
parallel n-gram precomputation using multiprocessing and numba
combines:
- numba jit compilation for c-like speed
- multiprocessing to compute different n-grams in parallel
- memory mapping for efficient data sharing
"""

import requests
import numpy as np
import torch
from pathlib import Path
from numba import jit, prange, types
from numba.typed import Dict
import pickle
from multiprocessing import Pool, cpu_count
import time
from functools import partial

# Constants
CONTEXT_LEN = 8
HORIZON = 1
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

@jit(nopython=True, cache=True)
def compute_ngram_batch(indices, n, context_len, V, start_idx, end_idx):
    """
    Compute n-gram counts for a batch of positions
    Optimized for cache locality and parallel execution
    """
    # Pre-allocate with reasonable size
    max_contexts = min((end_idx - start_idx) * 2, V**n)
    counts = np.zeros((max_contexts, V), dtype=np.int32)
    context_hashes = np.zeros(max_contexts, dtype=np.int64)
    context_totals = np.zeros(max_contexts, dtype=np.int32)
    context_map = Dict.empty(key_type=types.int64, value_type=types.int32)
    next_context_id = 0
    
    # Process positions in this batch
    for i in range(max(start_idx, context_len - n + 1), min(end_idx, len(indices) - 1)):
        # Compute context hash efficiently
        context_hash = 0
        base = 1
        for j in range(n-1, -1, -1):
            context_hash += indices[i - j] * base
            base *= V
        
        next_char = indices[i + 1]
        
        # Update counts
        if context_hash in context_map:
            context_id = context_map[context_hash]
        else:
            if next_context_id >= max_contexts:
                # Resize if needed
                new_size = max_contexts * 2
                new_counts = np.zeros((new_size, V), dtype=np.int32)
                new_counts[:max_contexts] = counts
                counts = new_counts
                
                new_hashes = np.zeros(new_size, dtype=np.int64)
                new_hashes[:max_contexts] = context_hashes
                context_hashes = new_hashes
                
                new_totals = np.zeros(new_size, dtype=np.int32)
                new_totals[:max_contexts] = context_totals
                context_totals = new_totals
                
                max_contexts = new_size
            
            context_id = next_context_id
            context_map[context_hash] = context_id
            context_hashes[context_id] = context_hash
            next_context_id += 1
        
        counts[context_id, next_char] += 1
        context_totals[context_id] += 1
    
    # Return trimmed arrays
    return (counts[:next_context_id], 
            context_hashes[:next_context_id], 
            context_totals[:next_context_id],
            context_map)

@jit(nopython=True, parallel=True, cache=True)
def merge_counts(batch_results, V):
    """Merge count results from multiple batches"""
    # First pass: count total unique contexts
    total_contexts = 0
    for counts, _, _, _ in batch_results:
        total_contexts += len(counts)
    
    # Allocate merged arrays
    merged_counts = np.zeros((total_contexts, V), dtype=np.int32)
    merged_hashes = np.zeros(total_contexts, dtype=np.int64)
    merged_totals = np.zeros(total_contexts, dtype=np.int32)
    merged_map = Dict.empty(key_type=types.int64, value_type=types.int32)
    
    # Merge all batches
    next_id = 0
    for counts, hashes, totals, _ in batch_results:
        for i in range(len(counts)):
            context_hash = hashes[i]
            if context_hash in merged_map:
                # Add to existing context
                context_id = merged_map[context_hash]
                merged_counts[context_id] += counts[i]
                merged_totals[context_id] += totals[i]
            else:
                # New context
                merged_map[context_hash] = next_id
                merged_hashes[next_id] = context_hash
                merged_counts[next_id] = counts[i].copy()
                merged_totals[next_id] = totals[i]
                next_id += 1
    
    # Return trimmed results
    return (merged_counts[:next_id], 
            merged_hashes[:next_id], 
            merged_totals[:next_id],
            merged_map)

@jit(nopython=True, parallel=True, cache=True)
def compute_all_scores(indices, n, context_len, V, context_map, log_probs):
    """Compute scores for all positions in parallel"""
    num_positions = len(indices) - context_len
    scores = np.zeros((num_positions, V), dtype=np.float32)
    uniform_log_prob = np.log(1.0 / V)
    
    # Process in parallel
    for pos in prange(context_len, len(indices)):
        # Compute context hash
        context_hash = 0
        base = 1
        for j in range(n-1, -1, -1):
            context_hash += indices[pos - n + 1 + j] * base
            base *= V
        
        # Look up scores
        if context_hash in context_map:
            context_id = context_map[context_hash]
            scores[pos - context_len] = log_probs[context_id]
        else:
            for k in range(V):
                scores[pos - context_len, k] = uniform_log_prob
    
    return scores

def compute_ngram_parallel(indices, n, context_len, V, num_workers=None):
    """Compute n-grams using parallel processing"""
    if num_workers is None:
        num_workers = cpu_count()
    
    # Divide work into batches
    total_positions = len(indices)
    batch_size = max(10000, total_positions // (num_workers * 4))
    batches = []
    
    for start in range(0, total_positions, batch_size):
        end = min(start + batch_size, total_positions)
        batches.append((start, end))
    
    print(f"Processing {len(batches)} batches on {num_workers} workers...")
    
    # Process batches in parallel
    batch_func = partial(compute_ngram_batch, indices, n, context_len, V)
    batch_results = []
    
    for start, end in batches:
        result = batch_func(start, end)
        batch_results.append(result)
    
    # Merge results
    print("Merging results...")
    counts, hashes, totals, context_map = merge_counts(batch_results, V)
    
    # Compute log probabilities
    print("Computing log probabilities...")
    log_probs = np.zeros((len(counts), V), dtype=np.float32)
    smoothing = 1.0
    
    for i in range(len(counts)):
        total = totals[i] + V * smoothing
        for j in range(V):
            prob = (counts[i, j] + smoothing) / total
            log_probs[i, j] = np.log(prob + 1e-10)
    
    # Compute scores for all positions
    print("Computing scores for all positions...")
    scores = compute_all_scores(indices, n, context_len, V, context_map, log_probs)
    
    return scores, len(context_map)

def process_ngram(args):
    """Process a single n-gram (for multiprocessing)"""
    text, stoi, V, n, context_len = args
    indices = np.array([stoi[ch] for ch in text], dtype=np.int32)
    
    print(f"\nProcessing {n}-grams...")
    start_time = time.time()
    
    scores, num_contexts = compute_ngram_parallel(indices, n, context_len, V)
    
    elapsed = time.time() - start_time
    print(f"{n}-gram computation completed in {elapsed:.2f} seconds")
    print(f"Unique {n}-gram contexts: {num_contexts:,}")
    
    return n, torch.from_numpy(scores)

def main():
    # Download dataset
    if not Path("input.txt").exists():
        print(f"Downloading dataset...")
        content = requests.get(DATA_URL, timeout=10).text
        Path("input.txt").write_text(content, encoding="utf-8")
    
    text = Path("input.txt").read_text(encoding="utf-8")
    print(f"Dataset size: {len(text):,} characters")
    
    # Build vocabulary
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    V = len(chars)
    print(f"Vocabulary size: {V} unique characters")
    
    # Save vocabulary
    vocab_data = {'stoi': stoi, 'itos': itos, 'V': V, 'chars': chars}
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab_data, f)
    
    # Prepare arguments for parallel processing
    ngram_args = [
        (text, stoi, V, 2, CONTEXT_LEN),  # bigrams
        (text, stoi, V, 3, CONTEXT_LEN),  # trigrams
        (text, stoi, V, 4, CONTEXT_LEN),  # 4-grams
    ]
    
    print(f"\nStarting parallel n-gram computation on {cpu_count()} CPUs...")
    total_start = time.time()
    
    # Process all n-grams in parallel
    with Pool(processes=3) as pool:
        results = pool.map(process_ngram, ngram_args)
    
    # Save results
    for n, scores in results:
        filename = f'{["", "", "bi", "tri", "four"][n]}gram_scores_parallel.pt'
        torch.save(scores, filename)
        print(f"\nSaved {n}-gram scores to {filename}")
        print(f"Shape: {scores.shape}")
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal computation time: {total_elapsed:.2f} seconds")
    
    # Verification
    print("\n=== Verification ===")
    pos = 1000
    context_text = text[pos:pos+CONTEXT_LEN]
    actual_next = text[pos+CONTEXT_LEN]
    
    print(f"Context: '{context_text}'")
    print(f"Actual next: '{actual_next}'")
    
    for n, scores in results:
        if pos < scores.shape[0]:
            probs = torch.exp(scores[pos])
            top3_vals, top3_idx = torch.topk(probs, 3)
            print(f"\n{n}-gram top predictions:")
            for i, (prob, idx) in enumerate(zip(top3_vals, top3_idx)):
                print(f"  '{itos[idx.item()]}' (p={prob.item():.3f})")

if __name__ == "__main__":
    main()