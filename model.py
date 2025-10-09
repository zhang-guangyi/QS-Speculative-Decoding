import os
path_modelweights='/scratch/prj/nmes_simeone/scratch_tmp/guangyi_llm/'
os.environ["HF_DATASETS_CACHE"] = path_modelweights
os.environ["HF_HOME"] = path_modelweights
os.environ["HUGGINGFACE_HUB_CACHE"] = path_modelweights
os.environ["TRANSFORMERS_CACHE"] = path_modelweights
from huggingface_hub import login
# login(token="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")

import csv
import numpy as np
import torch
import random
import argparse
import contexttimer
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
from utils import Meter,count_parameters, get_input

import csv
import numpy as np
from utils import *
from model import *
from engine import *
from tqdm import tqdm
from globals import Decoder
from collections import deque
from colorama import Fore, Style 
from huggingface_hub import login
from datasets import load_dataset
import matplotlib.pyplot as plt
from utils import Meter,count_parameters, get_input
from fastchat.model import get_conversation_template
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sampling import autoregressive_sampling, SpecSamplingEnv



class DQN(nn.Module):
    def __init__(self, seq_input_dim, action_dim):
        super(DQN, self).__init__()

        # 主干分支：处理 token 概率序列
        self.main_fc1 = nn.Linear(seq_input_dim, 128)
        self.main_fc2 = nn.Linear(128, 128)

        # 辅助分支：处理统计特征（非零均值 + 最后非零值）
        self.aux_fc1 = nn.Linear(2, 32)
        self.aux_fc2 = nn.Linear(32, 32)

        # uplink 分支：处理 uplink 状态（标量）
        self.uplink_fc1 = nn.Linear(1, 16)
        self.uplink_fc2 = nn.Linear(16, 16)

        # 融合层：拼接三个分支输出
        self.fusion_fc1 = nn.Linear(128 + 32 + 16, 128)
        self.fusion_fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        """
        输入 x: shape [batch_size, seq_len + 1]，最后一个维度是 uplink 状态标量
        """
        uplink = x[:, -1:]            # shape [B, 1]
        x_seq = x[:, :-1]             # shape [B, seq_len]

        # 创建非零掩码：假设 padding 值为严格的 0.0
        mask = (x_seq > 0).float()
        eps = 1e-8
        masked_sum = (x_seq * mask).sum(dim=1)
        valid_counts = mask.sum(dim=1) + eps
        mean_prob = (masked_sum / valid_counts).unsqueeze(1)  # [B, 1]

        # 找出最后一个非零 token 的值
        last_token_prob = torch.zeros(x_seq.size(0), 1, device=x.device)
        for i in range(x_seq.size(0)):
            nonzeros = x_seq[i][x_seq[i] > 0]
            last_token_prob[i] = nonzeros[-1] if nonzeros.numel() > 0 else 0.0

        # ======== 主干路径 ========
        x_main = F.relu(self.main_fc1(x_seq))
        x_main = F.relu(self.main_fc2(x_main))

        # ======== 辅助分支路径 ========
        aux_input = torch.cat([mean_prob, last_token_prob], dim=1)  # [B, 2]
        x_aux = F.relu(self.aux_fc1(aux_input))
        x_aux = F.relu(self.aux_fc2(x_aux))

        # ======== uplink 分支路径 ========
        x_uplink = F.relu(self.uplink_fc1(uplink))  # [B, 1] -> [B, 16]
        x_uplink = F.relu(self.uplink_fc2(x_uplink))

        # ======== 融合路径 ========
        x_cat = torch.cat([x_main, x_aux, x_uplink], dim=1)  # [B, 176]
        x_out = F.relu(self.fusion_fc1(x_cat))
        return self.fusion_fc2(x_out)



class ReplayMemory:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)



class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = DQN(state_dim, action_dim).cuda()       
        self.target_net = DQN(state_dim, action_dim).cuda()   
        self.target_net.load_state_dict(self.q_net.state_dict())  
        self.best_net = DQN(state_dim, action_dim).cuda() 
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.replay_buffer = deque(maxlen=10000)           
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1
        self.update_target_freq = 30  
        self.step_count = 0
        self.best_reward = 0
        self.best_avg_reward = 0
        self.eval_episodes = 5  
        self.action_dim = action_dim

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:   
            state_tensor = state
            q_values = self.q_net(state_tensor)
            return q_values.cpu().detach().numpy().argmax()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # Randomly sample from a buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).cuda().squeeze(1)
        next_states = torch.stack(next_states).cuda().squeeze(1)
        actions = torch.LongTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).cuda()
        dones = torch.FloatTensor(dones).cuda()

        # Calculate Q
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Clculate Target Q
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Calculate loss function
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            # 使用深拷贝更新目标网络参数
            self.target_net.load_state_dict({
                k: v.clone() for k, v in self.q_net.state_dict().items()
            })

    def save_model(self, path="./output/best_model.pth"):
        if not os.path.exists("./output"):
            os.makedirs("./output")
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")


    def evaluate(self,  args, env, test_samples, tokenizer, meter, static_action=None):
        """Assess the model performance"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # Turn off the explorarrtion
        total_rewards = []
        test_times = 2
        for _ in range(test_times):
            for idx, sample in enumerate(test_samples):
                input_text = get_input(args, sample)
                input_ids = tokenizer([input_text], return_tensors='pt').input_ids[:,:400].cuda()
                with contexttimer.Timer() as t:
                    env.reset(input_ids)
                    bits_total = 0
                state = env.state
                done = False
                episode_reward = 0
                while True:
                    if static_action is None:
                        action = self.choose_action(state)
                        next_state, reward, done  = env.step(action_id=action)
                        bits_total = bits_total + env.action_space[action][0] * (env.calculate_bit_length(env.action_space[action][1], env.vocab_size) + 16)
                    else: 
                        next_state, reward, done  = env.step(static_action=static_action)
                        bits_total = bits_total + static_action[0] * (env.calculate_bit_length(static_action[1], env.vocab_size) + 16)
                    episode_reward += reward
                    state = next_state
                    if done or episode_reward > 2e4:
                        break
                total_rewards.append(episode_reward.cpu().detach())
                time_total = t.elapsed + env.total_time
                time_total = time_total + bits_total/env.uplink_rate
                output = env.prefix[:, env.prifex_len:]
                token_persec = output.shape[1] / time_total
                meter.update("SP", token_persec, output.shape[1])
                results = meter.report()
                speedup_factor = results["SP"]["average_token_persec"] / results["AS_large"]["average_token_persec"]
                if idx%10==0:
                    for method, stats in results.items():
                        color_print(f"[Summary: {idx:03d}] {method}: Tokens/sec = {stats['average_token_persec']:.2f}, "
                                    f"Token Num = {stats['average_token_num']:.2f}, "
                                    f"Speedup by: {speedup_factor:.4f}")
        self.epsilon = original_epsilon  
        return np.mean(total_rewards)
    


class DDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = DQN(state_dim, action_dim).cuda()       
        self.target_net = DQN(state_dim, action_dim).cuda()   
        self.target_net.load_state_dict(self.q_net.state_dict())  
        self.best_net = DQN(state_dim, action_dim).cuda() 
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.replay_buffer = deque(maxlen=10000)          
        self.batch_size = 64
        self.gamma = 0.1
        self.epsilon = 0.1
        self.update_target_freq = 30 
        self.step_count = 0
        self.best_reward = 0
        self.best_avg_reward = 0
        self.eval_episodes = 5 
        self.action_dim = action_dim

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:   
            state_tensor = state
            q_values = self.q_net(state_tensor)
            return q_values.cpu().detach().numpy().argmax()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # Randomly sample from a buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states).cuda().squeeze(1)
        next_states = torch.stack(next_states).cuda().squeeze(1)
        actions = torch.LongTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).cuda()
        dones = torch.FloatTensor(dones).cuda()

        # Calculate Q
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Clculate Target Q
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)  
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()   
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute the loss function
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update network parameters
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_net.load_state_dict({
                k: v.clone() for k, v in self.q_net.state_dict().items()
            })

    def save_model(self, path="./output/best_model.pth"):
        if not os.path.exists("./output"):
            os.makedirs("./output")
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")


    def evaluate(self, args, env, test_samples, tokenizer, meter, static_action=None):
        """评估当前模型的性能"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # 关闭探索
        total_rewards = []
        test_times = 2
        for _ in range(test_times):
            for idx, sample in enumerate(test_samples):
                input_text = get_input(args, sample)
                input_ids = tokenizer([input_text], return_tensors='pt').input_ids[:,:400].cuda()
                with contexttimer.Timer() as t:
                    env.reset(input_ids)
                    bits_total = 0
                    time_count = 0
                state = env.extended_state
                done = False
                episode_reward = 0
                while True:
                    if static_action is None:
                        action = self.choose_action(state)
                        time_count = time_count + env.action_space[action][0] * (env.calculate_bit_length(env.action_space[action][1], env.vocab_size) + 16)/env.uplink_rate
                        next_state, reward, done  = env.step(action_id=action)
                    else: 
                        time_count = time_count +  static_action[0] * (env.calculate_bit_length(static_action[1], env.vocab_size) + 16)/env.uplink_rate
                        next_state, reward, done  = env.step(static_action=static_action)
                    episode_reward += reward
                    state = next_state
                    if done or episode_reward > 2e4:
                        break
                if idx > 0:
                    total_rewards.append(episode_reward.cpu().detach())
                    time_total = t.elapsed + env.total_time
                    time_total = time_total + time_count
                    output = env.prefix[:, env.prifex_len:]
                    token_persec = output.shape[1] / time_total
                    meter.update("SP", token_persec, output.shape[1])
                    results = meter.report()
                    speedup_factor = results["SP"]["average_token_persec"] / results["AS_large"]["average_token_persec"]
                    if idx%10==0:
                        for method, stats in results.items():
                            color_print(f"[Summary: {idx:03d}] {method}: Tokens/sec = {stats['average_token_persec']:.2f}, "
                                        f"Token Num = {stats['average_token_num']:.2f}, "
                                        f"Speedup by: {speedup_factor:.4f}")
        self.epsilon = original_epsilon 
        return np.mean(total_rewards)
