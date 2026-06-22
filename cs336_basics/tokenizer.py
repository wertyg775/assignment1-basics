import os
import sys
from typing import BinaryIO
from multiprocessing import Pool
from collections import Counter
import regex as re
from pathlib import Path
import pickle
from collections.abc import Iterable, Iterator

sys.path.append(str(Path(__file__).resolve().parent))

from train_tokenizer import PAT, SPECIAL_TOKENS, get_stats, merge

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.special_tokens = special_tokens or []
        

        self.special_tokens_id = {}
        reverse_vocab = {byte_seq: id for id, byte_seq in self.vocab.items()}
        for special in self.special_tokens:
            byte_special = special.encode("utf-8")
            self.special_tokens_id[special] = reverse_vocab[byte_special]
        
        self.merges = {}
        for b1, b2 in merges:
            key = (reverse_vocab[b1], reverse_vocab[b2])
            self.merges[key] = reverse_vocab[b1+b2]

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self.encode_chunks(text)

        escaped = [re.escape(t) for t in self.special_tokens]
        split_pattern = re.compile(f"({"|".join(escaped)})")

        tokens = []
        segments = split_pattern.split(text)
        for seg in segments:
            if seg == "":
                continue
            elif seg in self.special_tokens:
                k = self.special_tokens_id[seg] 
                tokens.append(k)
            else:
                tokens.extend(self.encode_chunks(seg))
        return tokens
    
    def encode_chunks(self, text: str)-> list[int]:
        tokens = []
        for token in re.finditer(PAT, text):
            token = tuple(token.group().encode("utf-8"))
            while len(token) >= 2:
                counts={}
                for pair in zip(token[:-1], token[1:]):
                    counts[pair] = counts.get(pair, 0) + 1 
                pair = min(counts, key=lambda p: self.merges.get(p, float('inf')))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                token = merge(token, pair, idx)
            token = list(token)
            for t in token:
                tokens.append(t)
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str])-> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]):
        bytes_seq = b"".join(self.vocab[idx] for idx in ids)
        text = bytes_seq.decode("utf-8", errors="replace")
        return text
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)





