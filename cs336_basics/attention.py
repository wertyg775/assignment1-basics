import torch.nn as nn
from cs336_basics import nn as mynn
from einops import rearrange
import torch
import math

def softmax(x: torch.Tensor, dim: int):

    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max

    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp

class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        assert d_k % 2 == 0
        # inv_freq[k] = 1 / theta^(2k/d_k), k = 0..d_k/2-1
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k)) #torch.arange(start, end, shift)

        positions = torch.arange(max_seq_len, device=device).float()
        #angles[i, k] = i * inv_freq[k]
        angles = torch.einsum("i,k->ik", positions, inv_freq) # (max_seq_len, d_k/2)

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k), token_positions: (..., seq_len)
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k/2)

        x1 = x[..., 0::2]  # even indices, "first of each pair"
        x2 = x[..., 1::2]  # odd indices,  "second of each pair"

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        # interleave back: (x1_rot[0], x2_rot[0], x1_rot[1], x2_rot[1], ...)
        out = torch.stack([x1_rot, x2_rot], dim=-1)
        out = rearrange(out, "... d two -> ... (d two)")
        return out

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = Q.shape[-1]

    scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    
    attn_weights = softmax(scores, dim=-1)
    out = torch.einsum("...qk,...kv->...qv", attn_weights, V)
    return out

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        factory_kwargs = {"device": device, "dtype": dtype}

        self.query = mynn.Linear(d_model, d_model, **factory_kwargs)
        self.key = mynn.Linear(d_model, d_model, **factory_kwargs)
        self.value = mynn.Linear(d_model, d_model, **factory_kwargs)
        self.output = mynn.Linear(d_model, d_model, **factory_kwargs)

        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, **factory_kwargs)

    def forward(self, x, token_positions=None):
        B, T, C = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = rearrange("b t (h d_k) -> b h t dk", h=self.num_heads)
        K = rearrange("b t (h d_k) -> b h t d_k", h=self.num_heads)
        V = rearrange("b t (h d_k) -> b h t d_k", h=self.num_heads)

        if token_positions is None:
            token_positions = torch.arange(T, device=x.device)
        
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))