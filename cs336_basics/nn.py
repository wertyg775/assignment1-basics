import torch.nn as nn
import torch
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device" : device, "dtype" : dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))

        std = math.sqrt(2 / (out_features + in_features))
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

        nn.init.trunc_normal_(self.weight, mean = 0, std = 1.0, a = -3.0, b = 3.0) # a and b act as upper and lower bounds

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


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model

        if d_ff is None:
            d_ff = int(round((8 / 3) * d_model / 64) * 64)
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        lin = self.w1(x)
        silu_gate = lin * torch.sigmoid(lin)
        gated = silu_gate * self.w3(x)

        return self.w2(gated)

def softmax(x: torch.Tensor, dim: int):

    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max

    exp_x = torch.exp(x_shifted)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp

def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    max = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max
    log_sum_exp = shifted.exp().sum(dim=-1).log()
    target_logit = shifted.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    loss = log_sum_exp - target_logit

    return loss.mean()




if __name__ == "__main__":
    test = torch.Tensor([[1,2,3,4], [5,6,7,8]])
    in_shape = test.shape
    weight = Linear(in_shape[1], 2 * in_shape[1])
    out_shape = weight(test).shape

    assert 2 * in_shape[1] == out_shape[1] , "Shape mismatch"

















        