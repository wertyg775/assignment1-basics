import torch.nn as nn
import torch
from cs336_basics import nn as mynn
from cs336_basics import attention
import math

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype":dtype}
        self.attn = attention.CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, **factory_kwargs)
        self.ffn = mynn.SwiGLU(d_model, d_ff, **factory_kwargs)
        self.ln1 = mynn.RMSNorm(d_model, **factory_kwargs)
        self.ln2 = mynn.RMSNorm(d_model, **factory_kwargs)

    def forward(self, x, token_positions=None):
        z = x + self.attn(self.ln1(x), token_positions)
        y = z + self.ffn(self.ln2(z))

        return y

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, d_ff, num_heads, theta, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device":device, "dtype":dtype}
        self.token_embedding_table = mynn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, d_ff, num_heads, context_length, theta, **factory_kwargs) for _ in range(num_layers)])
        self.lm_rmsnorm = mynn.RMSNorm(d_model, **factory_kwargs)
        self.lm = mynn.Linear(d_model, vocab_size, **factory_kwargs)

    def forward(self, token_ids):
        _, T = token_ids.shape
        x = self.token_embedding_table(token_ids) # x is retrieved from the embedding table so the forward arguments shouldn't take x AND only take token_ids
        token_positions = torch.arange(T, device=token_ids.device)
        for layer in self.transformer_blocks:
            x = layer(x, token_positions)

        x = self.lm(self.lm_rmsnorm(x))
        return x
