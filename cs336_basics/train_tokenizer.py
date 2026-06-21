import os
from typing import BinaryIO
from multiprocessing import Pool
from collections import Counter
import regex as re
from pathlib import Path
import pickle
import argparse

root_path = Path.cwd()
dummy_file = root_path / "data" / "TinyStoriesV2-GPT4-valid.txt"
tinystories_file = root_path / "data" / "TinyStoriesV2-GPT4-train.txt"
openweb_file = root_path / "data" / "owt_train.txt"

#GPT-4 Tokenizer
special_tokens = ["<|endoftext|>"]
PAT = re.compile(r"""'(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a byte string"

    # Get total file size in byte_tuples
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)] #produces a list of chunk boundary indices eg. [0, 20, 40, ...]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k byte_tuples at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token) #returns -1 if token not found
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenization_dict(text: str, PAT: re.Pattern, pretokenization_counts: dict[tuple[int, ...], int]) -> dict[tuple[int, ...], int]:
    pre_tokens = re.finditer(PAT, text)
    for token in pre_tokens:    
        byte_sequence = token.group().encode("utf-8")
        byte_tuple = tuple(byte_sequence) #eg. (101, 125, 130)
        pretokenization_counts[byte_tuple] = pretokenization_counts.get(byte_tuple, 0) + 1
    return pretokenization_counts

def process_chunks(args):
    input_path, start, end, special_tokens = args

    escaped = [re.escape(t) for t in special_tokens]
    split_pattern = re.compile(f"({"|".join(escaped)})")
    counts = {}
    with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            segments = split_pattern.split(chunk) if split_pattern else [chunk]
            for seg in segments:
                if seg == special_tokens[0]:                      
                    continue
                elif seg == "":
                    continue
                else:
                    counts = pretokenization_dict(seg, PAT, counts)
    
    return counts

def pretokenize(input_path: str, special_tokens: list[str], num_processes: int | None = None):
    num_processes = num_processes or os.cpu_count() or 1

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
    total: Counter[tuple[int, ...]] = Counter()
    with Pool(num_processes) as pool:
        for partial_counts in pool.imap_unordered(process_chunks, tasks):
            total.update(partial_counts)
    
    return dict(total)


def get_stats(key: tuple[int, ...], value: int, counts: dict[tuple[int, int], int], index: dict[tuple[int, int], set[tuple[int, ...]]]):
    for pair in zip(key[:-1], key[1:]):
        counts[pair] = counts.get(pair, 0) + (1 * value)
        index[pair] = index.get(pair, set())
        index[pair].add(key)
    return counts, index

def merge(key, pair, idx):
    newids=[]
    i = 0
    while(i < len(key)):
        if i < len(key) -1 and key[i] == pair[0] and key[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(key[i])
            i += 1

    return tuple(newids)

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    pretokenization_counts = pretokenize(input_path, special_tokens)

    vocab = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - 256 - len(special_tokens)   #vocab_size is defined as FINAL vocab

    counts = {}
    index = {}
    for key, value in pretokenization_counts.items():
        counts, index = get_stats(key, value, counts, index)

    for i in range(num_merges):
        pair = max(counts, key=lambda p: (counts[p], (vocab[p[0]], vocab[p[1]])))
        idx = 256 + i
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        merges.append((vocab[pair[0]], vocab[pair[1]])) 
        
        for key in list(index[pair]):
            value = pretokenization_counts.pop(key)
            new_key = merge(key, pair, idx)

            for p in zip(key[:-1], key[1:]):
                counts[p] -= value
                index[p].discard(key)

            for p in zip(new_key[:-1], new_key[1:]):
                counts[p] = counts.get(p, 0) + (1 * value)
                index.setdefault(p, set()).add(new_key)
            
            pretokenization_counts[new_key] = value

        del counts[pair]
        del index[pair]


    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    return vocab, merges

def save_bpe(vocab: dict[int: bytes], merges: list[tuple[bytes, bytes]], vocab_path: str, merges_path: str):
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        choices=["tinystories", "openweb"],
        required=True,
        help = "dataset to train"
    )

    args = parser.parse_args()

    if args.data == "tinystories":
        input_file = tinystories_file
        vocab_size = 10000
        saved_path = root_path / "tokenizer" / "tinystories"
    elif args.data == "openweb":
        input_file = openweb_file
        vocab_size = 32000
        saved_path = root_path / "tokenizer" / "openweb"

    
    vocab, merges = train_bpe(input_file, vocab_size, special_tokens)
    Path(saved_path).mkdir(parents=True, exist_ok=True)
    vocab_path, merges_path = saved_path / "vocab.pkl" , saved_path / "merges.pkl"
    save_bpe(vocab, merges, vocab_path, merges_path)

