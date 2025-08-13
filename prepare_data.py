"""
Data preparation script for pretokenization.
Can be run locally or on Modal for distributed processing.
"""

import os
import pickle
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Check if we're running on Modal
try:
    import modal
    ON_MODAL = True
except ImportError:
    ON_MODAL = False

def download_file(url: str, fname: str, chunk_size: int = 1024):
    """Helper function to download a file with progress bar."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file:
        with tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

def prepare_shakespeare():
    """
    Prepare the tiny shakespeare dataset.
    Uses GPT-2 BPE tokenizer.
    """
    try:
        import tiktoken
    except ImportError:
        print("Please install tiktoken: pip install tiktoken")
        return
    
    data_dir = Path("data/shakespeare")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download shakespeare if not exists
    input_file_path = data_dir / "input.txt"
    if not input_file_path.exists():
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)
    
    with open(input_file_path, "r") as f:
        data = f.read()
    print(f"Length of dataset in characters: {len(data):,}")
    
    # Use GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(data)
    print(f"Total tokens: {len(train_ids):,}")
    print(f"Vocab size: {enc.n_vocab}")
    
    # Split into train and val (90/10)
    n = len(train_ids)
    train_data = np.array(train_ids[:int(n*0.9)], dtype=np.uint16)
    val_data = np.array(train_ids[int(n*0.9):], dtype=np.uint16)
    
    # Save to binary files
    train_data.tofile(data_dir / "train.bin")
    val_data.tofile(data_dir / "val.bin")
    
    # No meta.pkl needed - train.py will default to GPT-2 vocab
    
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    print(f"Files saved to {data_dir}")

def prepare_openwebtext():
    """
    Prepare OpenWebText dataset using HuggingFace datasets.
    Uses GPT-2 BPE tokenizer.
    """
    try:
        import tiktoken
        from datasets import load_dataset
    except ImportError:
        print("Please install required packages:")
        print("pip install tiktoken datasets")
        return
    
    data_dir = Path("data/openwebtext")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Number of workers for data processing
    num_proc = os.cpu_count() // 2
    num_proc_load = num_proc
    
    # Load dataset
    print("Loading OpenWebText dataset...")
    dataset = load_dataset("openwebtext", num_proc=num_proc_load, trust_remote_code=True)
    
    # Split dataset
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")
    
    # Use GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}
    
    # Tokenize datasets
    print("Tokenizing dataset...")
    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    
    # Concatenate all ids
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        
        # Create output array
        filename = data_dir / f"{split}.bin"
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    
    print(f"Saved train.bin and val.bin to {data_dir}")
    print(f"Train tokens: {tokenized['train']['len'].sum():,}")
    print(f"Val tokens: {tokenized['val']['len'].sum():,}")

def prepare_custom(data_path: str):
    """
    Prepare a custom text dataset using GPT-2 tokenizer.
    
    Args:
        data_path: Path to a text file or directory of text files
    """
    import tiktoken
    from pathlib import Path
    
    data_path = Path(data_path)
    
    # Read text data
    if data_path.is_file():
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif data_path.is_dir():
        text = ""
        for txt_file in data_path.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8") as f:
                text += f.read() + "\n"
    else:
        raise ValueError(f"Invalid data path: {data_path}")
    
    print(f"Loaded {len(text):,} characters")
    
    # Setup output directory
    dataset_name = data_path.stem if data_path.is_file() else data_path.name
    data_dir = Path(f"data/{dataset_name}")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Use GPT-2 tokenization
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    vocab_size = enc.n_vocab
    
    # Convert to numpy array
    tokens = np.array(tokens, dtype=np.uint16)
    
    # Split 90/10
    n = len(tokens)
    train_tokens = tokens[:int(n*0.9)]
    val_tokens = tokens[int(n*0.9):]
    
    # Save files
    train_tokens.tofile(data_dir / "train.bin")
    val_tokens.tofile(data_dir / "val.bin")
    
    print(f"Saved to {data_dir}")
    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")

# Modal-specific preparation
if ON_MODAL:
    import shutil
    
    # Modal app setup (matching modal_train.py configuration)
    cuda_version = "12.6.0"
    flavor = "devel"
    operating_sys = "ubuntu22.04"
    tag = f"{cuda_version}-{flavor}-{operating_sys}"
    
    base_image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
        .apt_install("git")
        .pip_install(
            "torch==2.6.0",
            "transformers==4.51.3",
            "datasets==3.6.0",
            "tiktoken==0.9.0",
            "tqdm==4.67.1",
            "numpy",
            "requests",
        )
    )
    
    app = modal.App("prepare-data", image=base_image)
    volume = modal.Volume.from_name("nanogpt-multinode-demo", create_if_missing=True)
    
    @app.function(
        timeout=3600,
        cpu=(0.2, 16),
        volumes={"/vol": volume},
    )
    def prepare_data_modal(dataset: str = "openwebtext"):
        """
        Prepare data on Modal and save to volume.
        
        Args:
            dataset: Which dataset to prepare ("shakespeare", "openwebtext", or path to custom data)
        """
        os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
        
        if dataset == "shakespeare":
            prepare_shakespeare()
            # Copy to volume
            shutil.copy("data/shakespeare/train.bin", "/vol/train.bin")
            shutil.copy("data/shakespeare/val.bin", "/vol/val.bin")
            if os.path.exists("data/shakespeare/meta.pkl"):
                shutil.copy("data/shakespeare/meta.pkl", "/vol/meta.pkl")
        elif dataset == "openwebtext":
            prepare_openwebtext()
            # Copy to volume
            shutil.copy("data/openwebtext/train.bin", "/vol/train.bin")
            shutil.copy("data/openwebtext/val.bin", "/vol/val.bin")
        else:
            # Custom dataset
            prepare_custom(dataset)
            dataset_name = Path(dataset).stem if Path(dataset).is_file() else Path(dataset).name
            shutil.copy(f"data/{dataset_name}/train.bin", "/vol/train.bin")
            shutil.copy(f"data/{dataset_name}/val.bin", "/vol/val.bin")
        
        print("Data preparation complete and saved to Modal volume!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
        help="Which dataset to prepare: 'shakespeare', 'openwebtext', or path to custom text file/directory",
    )
    
    args = parser.parse_args()
    
    if args.dataset == "shakespeare":
        prepare_shakespeare()
    elif args.dataset == "openwebtext":
        prepare_openwebtext()
    else:
        prepare_custom(args.dataset)
    
    print("\nTo use this data with Modal, run:")
    print(f"  modal run prepare_data.py::prepare_data_modal --dataset {args.dataset}")