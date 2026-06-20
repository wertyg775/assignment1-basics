import os
from typing import BinaryIO
import regex as re

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
PAT = r"""'(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

special_tokens = ["<|endoftext|>"]
escaped_tokens = [re.escape(t) for t in special_tokens]

split_pattern = re.compile(f"({'|'.join(escaped_tokens)})")

pretokenization_counts = {}
with open(..., "rb") as f:
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
                print(f"special tokens: add id directly")
            elif split == "":
                continue
            else:
                pre_tokens = re.finditer(PAT, split)
                for token in pre_tokens:    
                    text = token.group().encode("utf-8")
                    byte_tuple = tuple(text)
                    pretokenization_counts[byte_tuple] = pretokenization_counts.get(byte_tuple, 0) + 1
