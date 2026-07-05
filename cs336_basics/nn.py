import torch.nn as nn
import torch
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device" : device, "dtype" : dtype}
        self.weight = nn.Parameters(torch.empty(out_features, in_features, **factory_kwargs))

        std = math.sqrt(2 / out_features + in_features)
        nn.init.trunc_normal_(self.weight, mean = 0, std=std, a = -3*std, b=3*std)

    def forward(self, x):
        return x @ self.weight.T
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))

        nn.init.trunc_normal_(self.weight, mean = 0, std = 1.0, a = -3.0, b = 3.0)

    def forward(self, token_ids: torch.Tensor)-> torch.Tensor:
        return self.weight[token_ids]   
    

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        factory_kwargs = {"device":device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(d_model, **factory_kwargs)) # ** unpacks the keywords argument at the call site

    def forward(self, x: torch.Tensor)-> torch.Tensor :
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms

        result = x_normed * self.weight

        return result.to(in_dtype)


class SwiGLU(nn.module):
    def __init__(self, d_model, d_ff, int=None, device=None, dtype=None):
        super.__init__()
        self.d_model = d_model

        if d_ff is None:
            d_ff = int(round((8 / 3) * d_model / 64) * 64)
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        silu_gate = self.w1(x) * torch.sigmoid(self.w1(x))
        gated = silu_gate * self.w3(x)

        return self.w2(gated)
















        