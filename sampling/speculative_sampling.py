import torch
import time
import math
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

 
from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn, norm_logits_Quan, lattice_based_quantization_torch
from globals import Decoder
from transformers import AutoTokenizer
from transformers.generation.candidate_generator import  _crop_past_key_values
from transformers import EncoderDecoderCache



# python eval.py \
#     --max_tokens 128 \
#     --gamma 4 \
#     --target_model_name  openai-community/gpt-xl\
#     --approx_model_name  openai-community/gpt\
#     --task summarize\ 

def expected_N(alpha: torch.Tensor) -> torch.Tensor:
    """
    Compute E[N] given a sequence of alpha_j values.

    Args:
        alpha (torch.Tensor): Tensor of shape (gamma,) with values in [0, 1]

    Returns:
        torch.Tensor: Scalar tensor representing E[N]
    """
    gamma = alpha.shape[0]
    E_N = torch.tensor(0.0, dtype=alpha.dtype, device=alpha.device)

    prod = torch.tensor(1.0, dtype=alpha.dtype, device=alpha.device)
    for k in range(1, gamma + 1):
        # k-th term (index from 0)
        term = k * prod * (1 - alpha[k - 1])
        E_N += term
        prod *= alpha[k - 1]  # accumulate product up to alpha[k-1]

    # Add the final term: (gamma + 1) * prod_{j=1}^gamma alpha_j
    E_N += (gamma + 1) * prod
    return E_N
# ======================== Speculative Sampling Environment ========================
import numpy as np
class SpecSamplingEnv:
    def __init__(self, approx_model, target_model, tokenizer, 
                 max_len=64, temperature=1.0, top_k=10, top_p=0.9, device='cuda'):
        self.device = device
        self.tokenizer = tokenizer
        self.approx_model = approx_model
        self.target_model = target_model

        self.top_k = top_k
        self.top_p = top_p
        self.max_len = max_len
        self.temperature = temperature

        self.times_count = 0
        self.vocab_size = 50272
        self.gamma_choices = [12, 10, 9, 8, 7, 6 ,5, 4, 3, 2, 1]
        self.q_choices = [720, 240] 
        # self.q_choices = [4, 24]
        self.action_space = [(g, q) for g in self.gamma_choices for q in self.q_choices]

        self.uplink_state = 0
        self.uplink_rates = [2_000_000, 6_000_000]
        self.uplink_rate = self.uplink_rates[self.uplink_state]

        self.uplink_transition_matrix = torch.tensor([
            [0.5, 0.5],  # low -> low, low -> high
            [0.5, 0.5]   # high -> low, high -> high
        ])

        self.max_state_length = 64 # 保存最多10个token的logits
        self.state = torch.zeros(1, self.max_state_length, self.vocab_size).to(self.device)
        self.total_time = 0.

        self.approx_past_key_values = None
        self.target_past_key_values = None

        self.uplink_rate_history = []  # 记录每一步的速率值

        self.total_time_reward = 0.
        self.uplink_tensor = torch.tensor([[float(self.uplink_state)]], device=self.device)
        self.extended_state = torch.cat([self.state.view(1, -1), self.uplink_tensor], dim=1)

    def get_average_uplink_rate(self):
        P = self.uplink_transition_matrix.numpy()
        rates = np.array(self.uplink_rates)

        # 解 πP = π, 即求平稳分布
        eigvals, eigvecs = np.linalg.eig(P.T)
        stationary = eigvecs[:, np.isclose(eigvals, 1)]
        stationary = stationary[:, 0].real
        stationary = stationary / stationary.sum()
        avg_rate = (stationary * rates).sum()
        return avg_rate

    def reset(self, input_prefix):
        self.times_count = 0
        self.total_time = 0.
        self.total_time_reward = 0.
        self.approx_past_key_values = None
        self.target_past_key_values = None
        self.prefix = input_prefix
        self.prifex_len = self.prefix.shape[1]
        self.state = torch.zeros(1, self.max_state_length).to(self.device)

        self.uplink_state = 0
        self.uplink_rate = self.uplink_rates[self.uplink_state]
        self.uplink_tensor = torch.tensor([[float(self.uplink_state)]], device=self.device)
        self.extended_state = torch.cat([self.state.view(1, -1),self.uplink_tensor], dim=1)
        self.uplink_rate_history = []

    
    def get_state(self, confidence_info):
        accept_num = confidence_info.shape[1]
        state = torch.cat([self.state, confidence_info], dim=-1)
        _, seq_len, = state.shape
        state = state[:, -self.max_state_length: ]
        assert state.shape[1] == self.max_state_length
        self.state = state
        self.uplink_tensor = torch.tensor([[float(self.uplink_state)]], device=self.device)
        self.extended_state = torch.cat([state.view(1, -1), self.uplink_tensor], dim=1)

        return self.extended_state

    def calculate_bit_length(self, ell, k):
        n = ell + k - 1
        r = k - 1
        combination = math.comb(n, r)
        bit_length = math.ceil(math.log2(combination))
        return bit_length

    @torch.no_grad()
    def step(self, action_id=0, static_action=None):
        if static_action == None:
            gamma, q_level = self.action_space[action_id]  
        else: 
            gamma, q_level = static_action[0], static_action[1]
   
        start_time = time.time()
        random_numbers = torch.rand(gamma, device=self.device)
        x = self.prefix
        prefix_len = self.prefix.shape[1]
        q_store = torch.zeros(self.prefix.shape[0], gamma, self.vocab_size).to(self.device)
        p_store = torch.zeros(self.prefix.shape[0], gamma + 1, self.vocab_size).to(self.device)
        q_temp =  torch.zeros(self.prefix.shape[0], gamma, self.vocab_size).to(self.device)
        confidence_accept = torch.zeros(1, gamma).to(self.device)
        for i in range(gamma):
            pruned_input_ids = x[:, self.approx_past_key_values[0][0].size(2):] if self.approx_past_key_values is not None else x
            draft_outputs = self.approx_model(pruned_input_ids, past_key_values=self.approx_past_key_values, use_cache=True)
            self.approx_past_key_values = draft_outputs.past_key_values  
            q = draft_outputs.logits
            q_store[:, i, :] = norm_logits(q[:, -1, :],  self.temperature, self.top_k, self.top_p)
            q_temp[:, i, :] = norm_logits_Quan(q[:, -1, :], q_level, self.temperature, self.top_k, self.top_p)
            next_tok = sample(q_temp[:, i, :])
            x = torch.cat((x, next_tok), dim=1)
        self.times_count += 1
        q = q_temp[:]
        pruned_input_ids = x[:, self.target_past_key_values[0][0].size(2):] if self.target_past_key_values is not None else x
        target_outputs = self.target_model(pruned_input_ids, past_key_values=self.target_past_key_values, use_cache=True)
        self.target_past_key_values = target_outputs.past_key_values  
        p = target_outputs.logits
        len_p = p.shape[1]
        for i in range(gamma + 1):
            p_store[:, i, :] = norm_logits(p[:, len_p - gamma + i - 1, :], self.temperature, self.top_k, self.top_p)
        p = p_store[:]
        is_all_accept = True
        n = prefix_len - 1 
        prob_num = 0
        for i in range(gamma):
            r = random_numbers[i]
            j = x[:, prefix_len + i]
            if r < torch.min(torch.tensor([1], device=q.device), p[:, i, j] / q[:, i, j]):
                n += 1
                confidence_accept[0, prob_num] = q_store[:, i, j]
                prob_num += 1
            else:
                t = sample(max_fn(p[:, n - prefix_len + 1, :] - q_temp[:, n - prefix_len + 1, :]))
                new_cache_size = n + 1
                self.target_past_key_values = _crop_past_key_values(self.target_model, self.target_past_key_values, new_cache_size)
                self.approx_past_key_values = _crop_past_key_values(self.approx_model, self.approx_past_key_values, new_cache_size)
                is_all_accept = False
                break
        self.prefix = x[:, :n + 1]
        if is_all_accept:
            t = sample(p[:, gamma, :])
        self.prefix = torch.cat((self.prefix, t), dim=1)
        end_time = time.time()
        time_duration = end_time-start_time
        self.total_time += time_duration

        ## 16 denotes the number of bits to represent one token
        uplink_bits_num_pertoken = self.calculate_bit_length(q_level, self.vocab_size) + 16   
        uplink_time = (uplink_bits_num_pertoken * gamma)/self.uplink_rate
        alpha = torch.zeros(gamma).to(self.device)
        for i in range(gamma):
            r = random_numbers[i]
            j = x[:, prefix_len + i]
            alpha[i] = torch.min(torch.tensor([1], device=q.device), p[:, i, j] / q[:, i, j])
        E_N = expected_N(alpha)
        reward =  E_N/(time_duration+uplink_time) * 0.5
        done = (self.prefix.shape[1]-self.prifex_len) >= self.max_len 
        self.total_time_reward = self.total_time_reward  + time_duration + uplink_time
        if done:
            reward = (self.prefix.shape[1]-self.prifex_len)/self.total_time_reward
        

        trans_probs = self.uplink_transition_matrix[self.uplink_state]
        self.uplink_state = torch.multinomial(trans_probs, num_samples=1).item()
        self.uplink_rate_history.append(self.uplink_rate)
        self.uplink_rate = self.uplink_rates[self.uplink_state]
        next_state = self.get_state(confidence_accept[:,:prob_num])
        return next_state, reward, done




import torch
import time
import math
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

 
from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn, norm_logits_Quan, lattice_based_quantization_torch
from globals import Decoder
from transformers import AutoTokenizer
from transformers.generation.candidate_generator import  _crop_past_key_values
from transformers import EncoderDecoderCache



# python eval.py \
#     --max_tokens 128 \
#     --gamma 4 \
#     --target_model_name  openai-community/gpt-xl\
#     --approx_model_name  openai-community/gpt\
#     --task summarize\ 


class ResidualBlock(nn.Module):
    def __init__(self, vocab_size, conv_channels, kernel_size, hidden_dim, scale):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(conv_channels, 1, kernel_size, padding=kernel_size//2),
        )
        self.fc = nn.Sequential(
            # nn.LayerNorm(vocab_size),
            nn.Linear(vocab_size, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size),
        )
        self.scale = scale

    def forward(self, x):
        residual = x
        x = x.unsqueeze(1)  # [B, 1, V]
        x = self.conv(x).squeeze(1)  # [B, V]
        x = self.fc(x)
        return residual + self.scale * x

class ProbRefiner(nn.Module):
    def __init__(self, vocab_size, conv_channels=4, kernel_size=9, hidden_dim=256):
        super().__init__()
        self.res1 = ResidualBlock(vocab_size, conv_channels, kernel_size, hidden_dim, scale=0.01)
        # self.res2 = ResidualBlock(vocab_size, conv_channels, kernel_size, hidden_dim, scale=0.02)
        self.res3 = ResidualBlock(vocab_size, conv_channels, kernel_size, hidden_dim, scale=0.05)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.res1(x)
        # x = self.res2(x)
        x = self.res3(x)
        x = self.activation(x)
        x = x / x.sum(dim=1, keepdim=True)
        return x     


def expected_N(alpha: torch.Tensor) -> torch.Tensor:
    """
    Compute E[N] given a sequence of alpha_j values.

    Args:
        alpha (torch.Tensor): Tensor of shape (gamma,) with values in [0, 1]

    Returns:
        torch.Tensor: Scalar tensor representing E[N]
    """
    gamma = alpha.shape[0]
    E_N = torch.tensor(0.0, dtype=alpha.dtype, device=alpha.device)

    prod = torch.tensor(1.0, dtype=alpha.dtype, device=alpha.device)
    for k in range(1, gamma + 1):
        # k-th term (index from 0)
        term = k * prod * (1 - alpha[k - 1])
        E_N += term
        prod *= alpha[k - 1]  # accumulate product up to alpha[k-1]

    # Add the final term: (gamma + 1) * prod_{j=1}^gamma alpha_j
    E_N += (gamma + 1) * prod
    return E_N
# ======================== Speculative Sampling Environment ========================
import numpy as np
class SpecSamplingEnv:
    def __init__(self, approx_model, target_model, tokenizer, 
                 max_len=64, temperature=1.0, top_k=10, top_p=0.9, device='cuda'):
        self.device = device
        self.tokenizer = tokenizer
        self.approx_model = approx_model
        self.target_model = target_model

        self.top_k = top_k
        self.top_p = top_p
        self.max_len = max_len
        self.temperature = temperature

        self.times_count = 0
        self.vocab_size = 50272
        self.gamma_choices = [12, 10, 9, 8, 7, 6 ,5, 4, 3, 2, 1]
        self.q_choices = [720, 240] 
        # self.q_choices = [4, 24]
        self.action_space = [(g, q) for g in self.gamma_choices for q in self.q_choices]

        self.refiner = ProbRefiner(vocab_size=self.vocab_size).to(self.device)

        self.uplink_state = 0
        self.uplink_rates = [2_000_000, 6_000_000]
        self.uplink_rate = self.uplink_rates[self.uplink_state]

        self.uplink_transition_matrix = torch.tensor([
            [0.5, 0.5],  # low -> low, low -> high
            [0.5, 0.5]   # high -> low, high -> high
        ])

        self.max_state_length = 64 # 保存最多10个token的logits
        self.state = torch.zeros(1, self.max_state_length, self.vocab_size).to(self.device)
        self.total_time = 0.

        self.approx_past_key_values = None
        self.target_past_key_values = None

        self.uplink_rate_history = []  # 记录每一步的速率值

        self.total_time_reward = 0.
        self.uplink_tensor = torch.tensor([[float(self.uplink_state)]], device=self.device)
        self.extended_state = torch.cat([self.state.view(1, -1), self.uplink_tensor], dim=1)

    def get_average_uplink_rate(self):
        P = self.uplink_transition_matrix.numpy()
        rates = np.array(self.uplink_rates)

        # 解 πP = π, 即求平稳分布
        eigvals, eigvecs = np.linalg.eig(P.T)
        stationary = eigvecs[:, np.isclose(eigvals, 1)]
        stationary = stationary[:, 0].real
        stationary = stationary / stationary.sum()
        avg_rate = (stationary * rates).sum()
        return avg_rate

    def reset(self, input_prefix):
        self.times_count = 0
        self.total_time = 0.
        self.total_time_reward = 0.
        self.approx_past_key_values = None
        self.target_past_key_values = None
        self.prefix = input_prefix
        self.prifex_len = self.prefix.shape[1]
        self.state = torch.zeros(1, self.max_state_length).to(self.device)

        self.uplink_state = 0
        self.uplink_rate = self.uplink_rates[self.uplink_state]
        self.uplink_tensor = torch.tensor([[float(self.uplink_state)]], device=self.device)
        self.extended_state = torch.cat([self.state.view(1, -1),self.uplink_tensor], dim=1)
        self.uplink_rate_history = []

    
    def get_state(self, confidence_info):
        accept_num = confidence_info.shape[1]
        state = torch.cat([self.state, confidence_info], dim=-1)
        _, seq_len, = state.shape
        state = state[:, -self.max_state_length: ]
        assert state.shape[1] == self.max_state_length
        self.state = state
        self.uplink_tensor = torch.tensor([[float(self.uplink_state)]], device=self.device)
        self.extended_state = torch.cat([state.view(1, -1), self.uplink_tensor], dim=1)

        return self.extended_state

    def calculate_bit_length(self, ell, k):
        n = ell + k - 1
        r = k - 1
        combination = math.comb(n, r)
        bit_length = math.ceil(math.log2(combination))
        return bit_length

    @torch.no_grad()
    def step(self, action_id=0, static_action=None):
        if static_action == None:
            gamma, q_level = self.action_space[action_id]  
        else: 
            gamma, q_level = static_action[0], static_action[1]
   
        start_time = time.time()
        random_numbers = torch.rand(gamma, device=self.device)
        x = self.prefix
        prefix_len = self.prefix.shape[1]
        q_store = torch.zeros(self.prefix.shape[0], gamma, self.vocab_size).to(self.device)
        p_store = torch.zeros(self.prefix.shape[0], gamma + 1, self.vocab_size).to(self.device)
        q_temp =  torch.zeros(self.prefix.shape[0], gamma, self.vocab_size).to(self.device)
        confidence_accept = torch.zeros(1, gamma).to(self.device)
        for i in range(gamma):
            time00 = time.time()
            pruned_input_ids = x[:, self.approx_past_key_values[0][0].size(2):] if self.approx_past_key_values is not None else x
            draft_outputs = self.approx_model(pruned_input_ids, past_key_values=self.approx_past_key_values, use_cache=True)
            self.approx_past_key_values = draft_outputs.past_key_values  
            q = draft_outputs.logits
            q_store[:, i, :] = norm_logits(q[:, -1, :],  self.temperature, self.top_k, self.top_p)

            time0 = time.time()
            q_temp[:, i, :] = norm_logits_Quan(q[:, -1, :], 1280, self.temperature, self.top_k, self.top_p)

            time1 = time.time()
            refined_probs = self.refiner(q_temp[:, i, :])
            time2 = time.time()



            print(f"time0-time00: {time0 - time00:.6f}, "
                    f"time1-time0: {time1 - time0:.6f}, "
                    f"time2-time1: {time2 - time1:.6f}")

            next_tok = sample(q_temp[:, i, :])
            x = torch.cat((x, next_tok), dim=1)
        self.times_count += 1
        q = q_temp[:]
        pruned_input_ids = x[:, self.target_past_key_values[0][0].size(2):] if self.target_past_key_values is not None else x
        target_outputs = self.target_model(pruned_input_ids, past_key_values=self.target_past_key_values, use_cache=True)
        self.target_past_key_values = target_outputs.past_key_values  
        p = target_outputs.logits
        len_p = p.shape[1]
        for i in range(gamma + 1):
            p_store[:, i, :] = norm_logits(p[:, len_p - gamma + i - 1, :], self.temperature, self.top_k, self.top_p)
        p = p_store[:]
        is_all_accept = True
        n = prefix_len - 1 
        prob_num = 0
        for i in range(gamma):
            r = random_numbers[i]
            j = x[:, prefix_len + i]
            if r < torch.min(torch.tensor([1], device=q.device), p[:, i, j] / q[:, i, j]):
                n += 1
                confidence_accept[0, prob_num] = q_store[:, i, j]
                prob_num += 1
            else:
                t = sample(max_fn(p[:, n - prefix_len + 1, :] - q_temp[:, n - prefix_len + 1, :]))
                new_cache_size = n + 1
                self.target_past_key_values = _crop_past_key_values(self.target_model, self.target_past_key_values, new_cache_size)
                self.approx_past_key_values = _crop_past_key_values(self.approx_model, self.approx_past_key_values, new_cache_size)
                is_all_accept = False
                break
        self.prefix = x[:, :n + 1]
        if is_all_accept:
            t = sample(p[:, gamma, :])
        self.prefix = torch.cat((self.prefix, t), dim=1)
        end_time = time.time()
        time_duration = end_time-start_time
        self.total_time += time_duration

        ## 16 denotes the number of bits to represent one token
        uplink_bits_num_pertoken = self.calculate_bit_length(q_level, self.vocab_size) + 16   
        uplink_time = (uplink_bits_num_pertoken * gamma)/self.uplink_rate
        alpha = torch.zeros(gamma).to(self.device)
        for i in range(gamma):
            r = random_numbers[i]
            j = x[:, prefix_len + i]
            alpha[i] = torch.min(torch.tensor([1], device=q.device), p[:, i, j] / q[:, i, j])
        E_N = expected_N(alpha)
        reward =  E_N/(time_duration+uplink_time) * 0.5
        done = (self.prefix.shape[1]-self.prifex_len) >= self.max_len 
        self.total_time_reward = self.total_time_reward  + time_duration + uplink_time
        if done:
            reward = (self.prefix.shape[1]-self.prifex_len)/self.total_time_reward
        

        trans_probs = self.uplink_transition_matrix[self.uplink_state]
        self.uplink_state = torch.multinomial(trans_probs, num_samples=1).item()
        self.uplink_rate_history.append(self.uplink_rate)
        self.uplink_rate = self.uplink_rates[self.uplink_state]
        next_state = self.get_state(confidence_accept[:,:prob_num])
        return next_state, reward, done


