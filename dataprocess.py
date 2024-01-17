"""
1. Process train/val dataset. 
2. Train customized tokenizer
3. Tokenize text
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"
TRAIN_MODE = "FIXED_BLOCK_SAMPLE"


def train_vocab(vocab_size):
    """
    Trains a custom sentencepiece tokenizer on the provided dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # 1) input and output files prefix path for sentencepiece
    prefix = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
    input_file = os.path.join(DATA_CACHE_DIR, "train_vocabulary.txt")

    print(f"Size is: {os.path.getsize(input_file) / 1024 / 1024:.2f} MB")

    # 2) train the sentencepiece model
    print("Will now train the vocab...")
    spm.SentencePieceTrainer.train(input=input_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=1.0,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   normalization_rule_name="identity")

    print("Done.")


def process_shard(args, vocab_size):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = [line for line in f]
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example.strip()
        tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
        all_tokens.extend(tokens)
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".txt", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize(vocab_size):
    # iterate the shards and tokenize all of them one by one in data/dialogue_all_data folder
    data_dir = os.path.join(DATA_CACHE_DIR, "dialogue_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if vocab_size > 0:
        # .bin files will be saved into tok{N} directory, create it once here
        bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        seed = int(time.time())
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        if self.vocab_source == "llama2":
            # the .bin files are right along the .json files
            bin_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
            shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        elif self.vocab_source == "custom":
            # the .bin files are in tok{N} directory
            bin_dir = os.path.join(DATA_CACHE_DIR, f"tok{self.vocab_size}")
            train_shard_filename = sorted(glob.glob(os.path.join(bin_dir, "train_dialogue.bin")))
            val_shard_filename = sorted(glob.glob(os.path.join(bin_dir, "val_dialogue.bin")))
        shard_filenames = train_shard_filename if self.split == "train" else val_shard_filename
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        while True:
            if (TRAIN_MODE == "FIXED_BLOCK_SAMPLE"):
                for shard in shard_filenames:
                    # open the dataset for reading but keep it on disk with memmap
                    m = np.memmap(shard, dtype=np.uint16, mode="r")
                    num_batches = len(m) // self.max_seq_len
                    num_batches -= 1  # drop the last partial batch
                    assert num_batches > 0, "this shard is way too small? investigate."
                    ixs = list(range(num_batches))
                    rng.shuffle(ixs)
                    for ix in ixs:
                        start = ix * self.max_seq_len
                        end = start + self.max_seq_len + 1
                        # calling .astype will copy the data into a new numpy array, now in RAM
                        chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                        x = chunk[:-1]
                        y = chunk[1:]
                        yield x, y
            elif (TRAIN_MODE == "BOS_BLOCK_SAMPLE"):
                for shard in shard_filenames:
                    # open the dataset for reading but keep it on disk with memmap
                    m = np.memmap(shard, dtype=np.uint16, mode="r")
                    BOS_VALUE = 1
                    # Find positions of BOS
                    bos_positions = np.where(m == BOS_VALUE)[0]
                    # Drop the last two dialogue
                    bos_positions = bos_positions[:-2]
                    # Convert numpy array to a list
                    bos_position_list = bos_positions.tolist()
                    rng.shuffle(bos_position_list)
                    for ix in bos_position_list:
                        start = ix
                        end = start + self.max_seq_len + 1
                        # calling .astype will copy the data into a new numpy array, now in RAM
                        chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                        x = chunk[:-1]
                        y = chunk[1:]
                        yield x, y        
            elif (TRAIN_MODE == "RAND_BLOCK_SAMPLE"):
                for shard in shard_filenames:
                    # open the dataset for reading but keep it on disk with memmap
                    m = np.memmap(shard, dtype=np.uint16, mode="r")
                    low = 0
                    high = m.size - self.max_seq_len - 1
                    idx = rng.sample(range(low, high + 1), k=high - low + 1)
                    for ix in idx:
                        start = ix
                        end = start + self.max_seq_len + 1
                        # calling .astype will copy the data into a new numpy array, now in RAM
                        chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                        x = chunk[:-1]
                        y = chunk[1:]
                        yield x, y                   
                    
                    
                

# -----------------------------------------------------------------------------
# public interface functions

def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")

class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["pretokenize", "train_vocab"])
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "train_vocab":
        train_vocab(vocab_size=args.vocab_size)
    elif args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
