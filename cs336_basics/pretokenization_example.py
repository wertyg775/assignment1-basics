import os
from typing import BinaryIO
import regex as re
from pathlib import Path

root_path = Path.cwd()
test_file = root_path / "data" / "TinyStoriesV2-GPT4-valid.txt"
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


## Usage
PAT = re.compile(r"""'(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")

def pretokenization_dict(text: str, PAT: re.Pattern, pretokenization_counts: dict[tuple[int, ...], int]) -> dict[tuple[int, ...], int]:
    pre_tokens = re.finditer(PAT, text)
    for token in pre_tokens:    
        byte_sequence = token.group().encode("utf-8")
        byte_tuple = tuple(byte_sequence) #eg. (101, 125, 130)
        pretokenization_counts[byte_tuple] = pretokenization_counts.get(byte_tuple, 0) + 1
    return pretokenization_counts

special_tokens = ["<|endoftext|>"]
escaped_tokens = [re.escape(t) for t in special_tokens]

split_pattern = re.compile(f"({'|'.join(escaped_tokens)})")

pretokenization_counts = {}

with open(test_file, "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]): # boundaries are byte_tuples inside the file
        f.seek(start) # moves the file pointer to index in args given
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        chunk_splits = split_pattern.split(chunk)
        for split in chunk_splits:
            if split in special_tokens:
                continue
            elif split == "":
                continue
            else:
                pretokenization_counts = pretokenization_dict(split, PAT, pretokenization_counts)
    
key = min(pretokenization_counts, key = pretokenization_counts.get)
print(key, pretokenization_counts[key])

vocab_size = 500
num_merges = vocab_size - 256
def get_stats(key: tuple[int, ...], value: int, counts: dict):
    for pair in zip(key[:-1], key[1:]):
        counts[pair] = counts.get(pair, 0) + (1 * value)
    return counts

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

vocab = {i: bytes([i]) for i in range(256)}
merges = {}
for i in range(num_merges):
    counts = {}
    for key, value in pretokenization_counts.items():
        counts = get_stats(key, value, counts)
    pair = max(counts, key = counts.get)
    idx = 256 + i
    pretokenization_counts = {
        merge(key, pair, idx): value
        for key, value in pretokenization_counts.items()
    }
    merges[pair] = idx
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

for tok in special_tokens:
    vocab[len(vocab)] = tok.encode("utf-8")

    
    

    





