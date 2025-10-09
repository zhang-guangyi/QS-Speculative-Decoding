

import torch
import random
import argparse
import contexttimer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import csv
import numpy as np
from tqdm import tqdm
from globals import Decoder
from collections import deque
from colorama import Fore, Style 
from huggingface_hub import login
from datasets import load_dataset
from utils import Meter,count_parameters, get_input
from fastchat.model import get_conversation_template
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sampling import autoregressive_sampling,  SpecSamplingEnv


def color_print(text, color='r'):
    if color=='r':
        print(Fore.RED + text + Style.RESET_ALL)
    elif color=='g':
        print(Fore.GREEN + text + Style.RESET_ALL)


# Training  step
def train_dqn(agent, target_agent, memory, optimizer, device, batch_size, gamma_r=0.0):
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

    state_batch = torch.stack(state_batch).to(device).squeeze(1)
    next_state_batch = torch.stack(next_state_batch).to(device).squeeze(1)
    action_batch = torch.tensor(action_batch).unsqueeze(1).to(device)
    reward_batch = torch.tensor(reward_batch).to(device)
    done_batch = torch.tensor(done_batch, dtype=torch.float32).to(device)
    q_values = agent(state_batch).gather(1, action_batch)
    next_q_values = target_agent(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + gamma_r * next_q_values * (1 - done_batch)

    loss = F.mse_loss(q_values.squeeze(), expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_loop(agent, target_agent, memory, optimizer, scheduler, env, tokenizer, train_samples, args, device, epsilon):
    # epsilon = 0.1
    target_update_freq = 2
    total_reward_record = 0

    for idx, sample in enumerate(train_samples): 
        input_text = get_input(args, sample)
        input_ids = tokenizer([input_text], return_tensors='pt').to(device).input_ids[:,:400]
        env.reset(input_ids)
        
        state = env.state
        done = False
 
        if idx == 0:
            while not done:
                with torch.no_grad():
                    _, reward, done = env.step(0)
        else:
            total_reward = 0
            while not done:
                with torch.no_grad():
                    q_values = agent(state)
                    if random.random() > epsilon:
                        action = q_values.argmax().item()
                    else:
                        action = random.randint(0, len(env.action_space) - 1)

                    next_state, reward, done = env.step(action)

                memory.push((state.detach(), action, reward, next_state.detach(), done))
                train_dqn(agent, target_agent, memory, optimizer, device, batch_size=8)
                state = next_state
                total_reward += reward

            total_reward_record += total_reward
            if idx % target_update_freq == 0:
                target_agent.load_state_dict(agent.state_dict())

            print(f"Sample {idx}, Total Reward: {total_reward:.2f}, Average Total Reward: {total_reward_record / (idx):.2f}")
        


def test_loop(agent, env, tokenizer, test_samples, args, device, meter, fix_gamma=False, static_action=(2, 10000),TEST_TIME = 2):
    if fix_gamma:
        for _ in range(TEST_TIME):
            for idx, sample in enumerate(test_samples):
                input_text = get_input(args, sample)
                input_ids = tokenizer([input_text], return_tensors='pt').to(device).input_ids[:,:400] 
                with contexttimer.Timer() as t:
                    env.reset(input_ids)
                    bits_total = 0
                
                # bits_total = bits_total + env.action_space[0][0] * (env.calculate_bit_length(env.action_space[0][0], env.vocab_size) + 16)
                done = False
                with torch.no_grad():
                    while not done:
                        # action = action_id
                        next_state, reward, done = env.step(static_action=static_action)
                        state = next_state.detach()
                        bits_total = bits_total + static_action[0] * (env.calculate_bit_length(static_action[1], env.vocab_size) + 16)
                time_total = t.elapsed + env.total_time
                # print(time_total)
                # print(bits_total/env.uplink_rate)
                if static_action[1]!=10000:
           
                    time_total = time_total + bits_total/env.uplink_rate
                if idx > 0:
                    output = env.prefix[:, env.prifex_len:]
                    token_persec = output.shape[1] / time_total
                    meter.update("SP", token_persec, output.shape[1])
                if idx %10==0:
                    results = meter.report()
                    speedup_factor = results["SP"]["average_token_persec"] / results["AS_large"]["average_token_persec"]
                    for method, stats in results.items():
                        color_print(f"[Summary: {idx:03d}] {method}: Average Tokens/sec = {stats['average_token_persec']:.2f}, "
                                    f"Average Token Number = {stats['average_token_num']:.2f}, "
                                    f"Total Speed up by: {speedup_factor:.4f}")
    
    else:
        for _ in range(TEST_TIME):
            for idx, sample in enumerate(test_samples):
                input_text = get_input(args, sample)
                input_ids = tokenizer([input_text], return_tensors='pt').to(device).input_ids[:,:400]
                with contexttimer.Timer() as t:
                    env.reset(input_ids)
                    bits_total = 0
 
                # state, reward, done = env.step(1)
                done = False
                state = env.state
                with torch.no_grad():
                    while not done:
                        if state is not None:
                            q_values = agent(state)
                            action = q_values.argmax().item()
                            # print(action)
                        # else:
                        #     action=3
                        next_state, reward, done = env.step(action)
                        state = next_state.detach()
                        # print(action)
                        bits_total = bits_total + env.action_space[action][0] * (env.calculate_bit_length(env.action_space[action][1], env.vocab_size) + 16)
                
                time_total = t.elapsed + env.total_time
                if env.action_space[action][1]!=10000:
                    # print(bits_total/env.uplink_rate)
                    time_total = time_total + bits_total/env.uplink_rate
                # time_total = t.elapsed  
                if idx > 0:
                    output = env.prefix[:, env.prifex_len:]
                    token_persec = output.shape[1] / time_total
                    meter.update("SP", token_persec, output.shape[1])

                if idx %10==0:
                    results = meter.report()
                    speedup_factor = results["SP"]["average_token_persec"] / results["AS_large"]["average_token_persec"]
                    for method, stats in results.items():
                        color_print(f"[Summary: {idx:03d}] {method}: Average Tokens/sec = {stats['average_token_persec']:.2f}, "
                                    f"Average Token Number = {stats['average_token_num']:.2f}, "
                                    f"Total Speed up by: {speedup_factor:.4f}")


    return meter
        