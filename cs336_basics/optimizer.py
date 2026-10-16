from collections.abc import Callable, Iterable
from typing import Optional
import torch.nn as nn
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t"
                , 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss 


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-5):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["betas"][0]
            beta_2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                grad = p.grad.data

                exp_avg = (beta_1 * exp_avg + (1 - beta_1) * grad)
                exp_avg_sq = (beta_2 * exp_avg_sq + (1 - beta_2) * grad.pow(2))
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq

                m_hat = exp_avg / (1 - beta_1 ** t)
                v_hat = exp_avg_sq / (1 - beta_2 ** t)

                p.data = (1 - lr * weight_decay) * p.data - (lr * m_hat) / (v_hat.sqrt() + eps)

        return loss


def cosine_scheduling(t, a_max, a_min, t_w, t_c):
    if t < t_w:
        lr = (t / t_w) * a_max
        return lr

    if t >= t_w and t <= t_c:
        lr = a_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi) ) * (a_max - a_min)
        return lr

    if t > t_c:
        return a_min           
