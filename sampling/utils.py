import torch
from torch.nn import functional as F

# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone() 
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


import numpy as np

import torch



def lattice_based_quantization_torch(p, Q_ell):
    """
    Optimized lattice-based quantization for compressing a probability vector using PyTorch.

    Parameters:
    - p (torch.Tensor): The input probability vector (1D tensor).
    - Q_ell (int): Target quantized sum value.

    Returns:
    - b (torch.Tensor): Quantized vector (1D tensor).
    - p_recovered (torch.Tensor): Recovered normalized probability vector (1D tensor).
    """
    # Ensure the input is a torch tensor
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)

    # Step 1: Compute initial b' and the sum ell'
    b_prime = torch.floor(Q_ell * p + 0.5).to(torch.int)
    ell_prime = torch.sum(b_prime)

    # Step 2: Adjust b' to meet the constraint if ell' != Q_ell
    diff = Q_ell - ell_prime  # Difference between target sum and current sum

    if diff != 0:
        # Compute ζ[i] = b'[i] - Q_ell * p[i] for sorting
        zeta = b_prime - Q_ell * p
        sorted_indices = torch.argsort(zeta)  # Sort ζ[i] in increasing order

        if diff > 0:  # Increase the smallest ζ[i]
            b_prime[sorted_indices[:diff]] += 1
        elif diff < 0:  # Decrease the largest ζ[i]
            b_prime[sorted_indices[diff:]] -= 1

    b = b_prime

    # Step 3: Normalize b to recover a probability vector
    p_recovered = b / Q_ell

    return p_recovered




def compute_entropy(p):
    """
    Computes the entropy of a given probability vector.

    Parameters:
    - p (torch.Tensor): The input probability vector (1D tensor).

    Returns:
    - entropy (float): The entropy of the probability vector.
    """
    # Ensure input is a torch tensor
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float32)
    
    # Avoid log(0) by masking zeros
    p = p[p > 0]  # Filter out zero probabilities

    # Calculate entropy
    entropy = -torch.sum(p * torch.log2(p)).item()

    return entropy


def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """
    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / (temperature + 1e-10)
    # logits = top_k_top_p_filter(logits, top_k=20, top_p=1.)
    probs = F.softmax(logits, dim=1)

    return probs


def norm_logits_Quan(logits : torch.Tensor, q_level: int,  temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """
    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p 

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / (temperature + 1e-10)
    # logits = top_k_top_p_filter(logits, top_k=20, top_p=1.)
    probs = F.softmax(logits, dim=1)
    if q_level != 10000:
        # print('ssssssssssssss')
        probs = lattice_based_quantization_torch(probs.squeeze(0), Q_ell=q_level).unsqueeze(0)
    return probs


def sample(probs : torch.Tensor, num_samples: int = 1):
    # probs = torch.clamp(probs, min=1e-10)  # 避免负数和零
    # probs /= probs.sum(dim=-1, keepdim=True)
    # print(torch.isnan(probs).any())
    # print((probs < 0).any())
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    # idx_next = torch.argmax(probs, dim=-1).unsqueeze(0)
    return idx_next


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)  
    return x_max / x_max_sum
