import os
path_modelweights='/scratch/prj/nmes_simeone/scratch_tmp/guangyi_llm/'
os.environ["HF_DATASETS_CACHE"] = path_modelweights
os.environ["HF_HOME"] = path_modelweights
os.environ["HUGGINGFACE_HUB_CACHE"] = path_modelweights
os.environ["TRANSFORMERS_CACHE"] = path_modelweights
# from huggingface_hub import login
# login(token="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")

import torch
import random
import random
import swanlab
import argparse
import contexttimer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from collections import defaultdict

import csv
import numpy as np
from utils import *
from model import *
# from PPO_model import *
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

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for benchmark.py')
    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default="llama2-7b")
    parser.add_argument('--target_model_name', type=str, default="llama2-70b")
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=64, help='max token number generated.')
    parser.add_argument('--num_epochs', type=int, default=500, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    parser.add_argument('--task', '-t', type=str, default=4, help='guess time.')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    args = parser.parse_args()
    return args


def seed_initial(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args, input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    seed_initial(seed=312)
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch_device)
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name)
    Decoder().set_tokenizer(tokenizer)
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       device_map="auto",  torch_dtype=torch.float16,
                                                       trust_remote_code=True)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       device_map="auto",  torch_dtype=torch.float16,
                                                       trust_remote_code=True)    
    print(large_model.config.model_type)
    print("finish loading models") 
    small_total, small_trainable = count_parameters(small_model)
    large_total, large_trainable = count_parameters(large_model)
    print(f"Small Model ({approx_model_name}): Total Parameters: {small_total/ 1e6:,}M, Trainable Parameters: {small_trainable/ 1e6:,}M")
    print(f"Large Model ({target_model_name}): Total Parameters: {large_total/ 1e6:,}M, Trainable Parameters: {large_trainable/ 1e6:,}M")

    if args.task == 'summarize':
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0")
        test_samples = ds["test"].select(np.arange(100, 201, 1))   
        train_samples = ds["train"].select(np.arange(0, 1000, 1))   
    elif args.task == 'translate':
        ds = load_dataset("wmt14", "de-en")
        test_samples = ds["test"].select(np.arange(0, 50, 1))   
        train_samples = ds["train"].select(np.arange(0, 1000, 1))      

    top_k = 20
    top_p = 0.9
    temperature = 0.4
    meter = Meter()
    # Environment and agent definition
    env = SpecSamplingEnv(approx_model=small_model, 
                        target_model=large_model, 
                        tokenizer=tokenizer, 
                        max_len = num_tokens,
                        temperature=temperature, 
                        top_k=top_k, 
                        top_p=top_p)
    state_dim = env.max_state_length
    action_dim = len(env.action_space)
    agent = DDQNAgent(state_dim, action_dim)
    
    # ========== Training phase ==========
    print(env.action_space)
    # static_action_list = [(3, 2), (3, 16)]
    # static_action_list = [(3, 4), (3, 24)]
    static_action_list = [(3, 12), (4, 240)]
    print(env.uplink_rate)
    
    
    list_epoch = []

    temperature_list = [0.4]


    # prob_list = [(0.9, 0.3), (0.8, 0.3), (0.75, 0.3), (0.7, 0.3), (0.6, 0.3), (0.5, 0.25), (0.4, 0.2), (0.2, 0.1)]
    prob_list = [(0.5, 0.5)]
    transition_matrices = {}

    for i in range(len(prob_list)):
        transition_matrices[f'{i}'] = [[prob_list[i][0], 1-prob_list[i][0]], [prob_list[i][1], 1-prob_list[i][1]]]
    
    for name, matrix in transition_matrices.items():
        env.uplink_transition_matrix = torch.tensor(matrix, dtype=torch.float)
        avg_uplink_rate = env.get_average_uplink_rate()
        print(f'Rate: {avg_uplink_rate:.2f}')

    test_num = 0
    throughput_list = np.zeros((len(static_action_list)+1, len(prob_list)))
    # 用于保存所有温度下的吞吐量结果
    throughput_results = defaultdict(dict)

    for idx, sample in enumerate(test_samples):
        input_text = get_input(args, sample)
        input_ids = tokenizer([input_text], return_tensors='pt').to(torch_device).input_ids[:,:400]
        token_persec_AL, token_num_AL, _ = benchmark(autoregressive_sampling, "AS_large", use_profiling,
                                                    input_ids, large_model, tokenizer, num_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        
        if idx > 0:
            meter.update("AS_large", token_persec_AL, token_num_AL)
        if idx==5:
            break
    
    for idx, sample in enumerate(test_samples):
        input_text = get_input(args, sample)
        input_ids = tokenizer([input_text], return_tensors='pt').to(torch_device).input_ids[:,:400]
        # small_model.to(torch.device)
        token_persec_AL, token_num_AL, _ = benchmark(autoregressive_sampling, "AS_small", use_profiling,
                                                    input_ids, small_model, tokenizer, num_tokens, temperature=temperature, top_k=top_k, top_p=top_p)

        # print(torch_device)
        # # ===== 显存统计 =====
        # allocated = torch.cuda.memory_allocated(torch_device) / 1024**2  # MB
        # reserved = torch.cuda.memory_reserved(torch_device) / 1024**2    # MB
        # max_allocated = torch.cuda.max_memory_allocated(torch_device) / 1024**2
        # max_reserved = torch.cuda.max_memory_reserved(torch_device) / 1024**2

        # print(f"[idx {idx}] allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB, "
        #     f"max_allocated: {max_allocated:.2f} MB, max_reserved: {max_reserved:.2f} MB")
        if idx > 0:
            meter.update("AS_small", token_persec_AL, token_num_AL)
        if idx==5:
            break

    
    for name, matrix in transition_matrices.items():
        env.uplink_transition_matrix = torch.tensor(matrix, dtype=torch.float)
        avg_uplink_rate = env.get_average_uplink_rate()
        print(f'Rate: {avg_uplink_rate:.2f}')
    
        env.temperature = temperature
        ckpt_direc = f'Modelweights/Q_sel_((2, 16))/best_model.pth'
        # ckpt_direc = f'Modelweights/Q_sel_((4, 24))/best_model.pth'
        ckpt = torch.load(ckpt_direc)

        # 自动判断是 state_dict 还是包含 "state_dict" 的字典
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        agent.q_net.load_state_dict(state_dict)
        meter.reset_sp()
        avg_reward = agent.evaluate(args, env, test_samples, tokenizer, meter)
        results = meter.report()
        throughput_dynamic = results["SP"]["average_token_persec"]
        print(f'Current throughput is {throughput_dynamic} (dynamic) at Temperature {temperature:.3f}')
        throughput_list[0, test_num] = throughput_dynamic
        # 存储动态策略的吞吐量
      

        # 存储各静态策略的吞吐量
        for i, action in enumerate(static_action_list):
            meter.reset_sp()
            avg_reward_static = agent.evaluate(args, env, test_samples, tokenizer, meter, action)
            results = meter.report()
            throughput_static = results["SP"]["average_token_persec"]
            throughput_list[i+1, test_num] = throughput_static
            print(f'Current throughput is {throughput_static} for action {action} at Temperature {temperature:.3f}')
        test_num += 1
    print(throughput_list)
    np.save('output/(8,12).npy',throughput_list)
    # 写入 JSON 文件
    # with open("output/throughput_results.json", "w") as f:
    #     json.dump(throughput_results, f, indent=4)



if __name__ == "__main__":
    args = parse_arguments()
    
    main(args, args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
             random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)



