import torch 
import numpy as np
import os
import typing 


def get_batch(x: np.memmap, batch_size: int, context_length: int, device: str):
    n = len(x)
    max_start_idx = n - context_length

    start_indices = np.random.randint(max_start_idx, size=batch_size)

    input = np.stack([x[i: i + context_length] for i in start_indices])
    targets = np.stack([x[i + 1: i + context_length + 1] for i in start_indices])

    input = torch.from_numpy(input.astype(np.int64)).to(device)
    targets = torch.from_numpy(targets.astype(np.int64)).to(device)

    return input, targets


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] ):
    obj = {
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(obj, out)

def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)

    model_weights = obj.get("model")
    if model_weights is None:
        raise ValueError("Model checkpoint data is missing from the file.")
    model.load_state_dict(model_weights)
    
    optim_state = obj.get("optimizer")
    if optim_state is None:
        raise ValueError("Optimizer checkpoint data is missing from the file.")
    optimizer.load_state_dict(optim_state)

    iteration = obj.get("iteration", 0) 
    
    return iteration

