import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sampling.utils import norm_logits, sample
from transformers import AutoTokenizer



@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : torch.nn.Module, tokenizer, N : int, 
                            temperature : float = 1, top_k : int = 0, top_p : float = 0):
    n = x.shape[1]
    T = x.shape[1] + N

    p_len = n
 
    past_key_values = None
    times_count = 0
    while n < T:
        # outputs = model(x)
        
        outputs = model(x)
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        # print(last_p )
        times_count += 1
        idx_next = sample(last_p)
        x = torch.cat((x, idx_next), dim=1)
        if idx_next.item() == tokenizer.eos_token_id:
            break
        n += 1
    return x[:, p_len:], times_count



import time
@torch.no_grad()
def autoregressive_sampling(x: torch.Tensor, model: torch.nn.Module, tokenizer, N: int, 
                            temperature: float = 1, top_k: int = 0, top_p: float = 0):
    n = x.shape[1]
    T = n + N  # 目标长度
    p_len = n  # 记录输入序列的初始长度

    past_key_values = None  # 缓存 KV
    times_count = 0
    
    while n < T:
        t1 = time.time()
        # 仅传递新生成的 token，同时传递 past_key_values 以避免重复计算
        if past_key_values is not None:
            pruned_input_ids = x[:, past_key_values[0][0].size(2) :]
        else:
            pruned_input_ids = x
        outputs = model(pruned_input_ids, past_key_values=past_key_values, use_cache=True)
        
        past_key_values = outputs.past_key_values  # 更新缓存
        last_p = norm_logits(outputs.logits[:, -1, :], temperature, top_k, top_p)
        
        times_count += 1
        idx_next = sample(last_p)
        x = torch.cat((x, idx_next), dim=1)
        t2 = time.time()

        # print(t2-t1)

        # if tokenizer.eos_token_id in x[0, p_len:]:
        #     break
        
        n += 1  # 更新 token 计数

    return x[:, p_len:], times_count



# @torch.no_grad()
# def autoregressive_sampling(x: torch.Tensor, model: torch.nn.Module, tokenizer,  N: int, 
#                             temperature: float = 1, top_k: int = 0, top_p: float = 0):
#     n = x.shape[1]
#     T = n + N  # 目标长度
#     p_len = n  # 记录输入序列的初始长度

#     past_key_values = None  # 缓存 KV
#     times_count = 0
    
#     output = model.generate(x,  max_length=T, do_sample=True, temperature=temperature)

#     return output[:, p_len:], times_count


