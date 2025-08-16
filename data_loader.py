"""
Efficient DataLoader implementation for GPT-2 tokenized language modeling.
Addresses synchronous loading bottlenecks with multi-worker prefetching.
"""

import os
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    """
    Efficient dataset for GPT-2 tokenized language modeling with memmap backing.
    
    Key optimizations:
    - Persistent memmap handle (no reopening per batch)
    - Efficient batch indexing with pre-computed offsets
    - Worker-aware data sharding for parallel loading
    """
    
    def __init__(
        self, 
        data_path: str, 
        block_size: int,
        dtype: np.dtype = np.uint16
    ):
        """
        Args:
            data_path: Path to the .bin file containing GPT-2 tokenized data
            block_size: Context window size for sequences
            dtype: Data type of the memmap file (uint16 for GPT-2 tokens)
        """
        self.block_size = block_size
        self.data = np.memmap(data_path, dtype=dtype, mode='r')
        self.data_len = len(self.data)
        
        # Pre-compute valid starting indices (avoiding edge cases)
        self.valid_indices = self.data_len - block_size - 1
        
    def __len__(self) -> int:
        """Number of possible sequences in the dataset."""
        return self.valid_indices
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Starting position in the token sequence
            
        Returns:
            Tuple of (input_tokens, target_tokens)
        """
        # Direct slicing is more efficient than item-by-item access
        chunk = self.data[idx:idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class RandomSampler(torch.utils.data.Sampler):
    """
    Infinite random sampler for continuous training.
    
    Generates random indices continuously without epoch boundaries,
    which is typical for language model training.
    """
    
    def __init__(self, data_source: Dataset, seed: Optional[int] = None):
        """
        Args:
            data_source: Dataset to sample from
            seed: Random seed for reproducibility
        """
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
    
    def __iter__(self):
        """Yield random indices infinitely."""
        while True:
            yield int(torch.randint(
                0, self.num_samples, (1,), 
                generator=self.generator
            ).item())
    
    def __len__(self):
        """Return a large number to indicate infinite sampling."""
        return 2**62  # Effectively infinite for practical purposes


def create_dataloader(
    data_path: str,
    block_size: int,
    batch_size: int,
    num_workers: int = 4,
    prefetch_factor: Optional[int] = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    seed: Optional[int] = None
) -> DataLoader:
    """
    Create an optimized DataLoader for GPT-2 tokenized language modeling.
    
    Args:
        data_path: Path to the .bin data file (GPT-2 tokenized)
        block_size: Context window size
        batch_size: Number of sequences per batch
        num_workers: Number of data loading workers (0 for main process only)
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        pin_memory: Use pinned memory for faster GPU transfer
        seed: Random seed for reproducibility
        
    Returns:
        Configured DataLoader ready for training
    """
    dataset = TokenDataset(data_path, block_size)
    sampler = RandomSampler(dataset, seed=seed)
    
    # Key optimizations:
    # - num_workers > 0 enables true parallelism
    # - prefetch_factor > 1 creates a buffer of ready batches
    # - persistent_workers avoids worker restart overhead
    # - pin_memory enables async GPU transfer
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True  # Maintain consistent batch sizes
    )
    
    return dataloader


def create_train_val_loaders(
    data_dir: str,
    block_size: int,
    batch_size: int,
    num_workers: int = 4,
    prefetch_factor: Optional[int] = 2,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create both training and validation DataLoaders.
    
    Args:
        data_dir: Directory containing train.bin and val.bin (GPT-2 tokenized)
        block_size: Context window size
        batch_size: Number of sequences per batch
        num_workers: Number of data loading workers
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        pin_memory: Use pinned memory for faster GPU transfer
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    
    train_loader = create_dataloader(
        train_path, 
        block_size, 
        batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        seed=seed
    )
    
    # Validation typically uses fewer workers since it's less frequent
    val_loader = create_dataloader(
        val_path,
        block_size,
        batch_size,
        num_workers=min(2, num_workers),  # Fewer workers for validation
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        seed=seed if seed is not None else seed + 1
    )
    
    return train_loader, val_loader


class DataLoaderIterator:
    """
    Wrapper to provide get_batch interface compatible with existing code.
    
    This allows gradual migration from the old get_batch function
    to the new DataLoader infrastructure.
    """
    
    def __init__(self, dataloader: DataLoader, device: torch.device):
        """
        Args:
            dataloader: The DataLoader to wrap
            device: Target device for tensors
        """
        self.dataloader = dataloader
        self.device = device
        self.iterator = iter(dataloader)
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get next batch, maintaining compatibility with existing interface.
        
        Returns:
            Tuple of (input_batch, target_batch) on the specified device
        """
        x, y = next(self.iterator)
        
        # Move to device (non_blocking=True if data is pinned)
        if self.dataloader.pin_memory and torch.cuda.is_available():
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        
        return x, y