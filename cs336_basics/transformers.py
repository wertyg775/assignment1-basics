import torch.nn as nn
import torch
from cs336_basics import nn as mynn
from cs336_basics import attention
import math

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype":dtype}
        self.rope = rope
        self.attn = attention.CausalMultiHeadSelfAttention(d_model, num_heads, self.rope, **factory_kwargs)
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
        self.token_embeddings = mynn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.d_k = d_model // num_heads
        self.rope = attention.RotaryPositionalEmbedding(theta, self.d_k, context_length)
        self.layers = nn.ModuleList([TransformerBlock(d_model, d_ff, num_heads, self.rope, **factory_kwargs) for _ in range(num_layers)])
        self.ln_final= mynn.RMSNorm(d_model, **factory_kwargs)
        self.lm_head = mynn.Linear(d_model, vocab_size, **factory_kwargs)

    def forward(self, token_ids):
        _, T = token_ids.shape
        x = self.token_embeddings(token_ids) # x is retrieved from the embedding table so the forward arguments shouldn't take x AND only take token_ids
        token_positions = torch.arange(T, device=token_ids.device)
        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.lm_head(self.ln_final(x))
        return x

    def decode(self, x, max_tokens, top_p=0.9, temp=1.0):
        for _ in range(max_tokens):
            logits = self.forward(x)
            logits = logits[:, -1, :]  # (batch_size, vocab_size)

            probs = mynn.softmax(logits, dim=-1, temp=temp)

            sorted_probs, sorted_indices = torch.sort(
                probs, descending=True, dim=-1
            )

            # Cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Identify tokens to remove
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = (
                sorted_indices_to_remove[..., :-1].clone()
            )
            sorted_indices_to_remove[..., 0] = False

            # Map removal mask back to original vocabulary order
            indices_to_remove = torch.zeros_like(
                sorted_indices_to_remove
            ).scatter(
                dim=-1,
                index=sorted_indices,
                src=sorted_indices_to_remove
            )

            # Remove tokens outside the nucleus
            probs[indices_to_remove] = 0.0

            # Renormalize probabilities
            probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence
            x = torch.cat((x, next_token), dim=1)

        return x


