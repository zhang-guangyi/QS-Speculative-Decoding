import os
path_modelweights='/scratch/prj/nmes_simeone/scratch_tmp/guangyi_llm/'
os.environ["HF_DATASETS_CACHE"] = path_modelweights
os.environ["HF_HOME"] = path_modelweights
os.environ["HUGGINGFACE_HUB_CACHE"] = path_modelweights
os.environ["TRANSFORMERS_CACHE"] = path_modelweights
from huggingface_hub import login


import torch
import random
import random
import swanlab
import argparse
import contexttimer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
    seed_initial(seed=30)
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
        test_samples = ds["test"].select(np.arange(100, 151, 1))   
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
    
    plot_direc = f'Plot_Reuslts/Q_sel_({env.q_choices[0],env.q_choices[1]})'
    os.makedirs(plot_direc, exist_ok=True)

    ckpt_direc = f'Modelweights/Q_sel_({env.q_choices[0],env.q_choices[1]})'
    os.makedirs(ckpt_direc, exist_ok=True)

    # Init swandb
    swanlab.init(
        project="RL-Speculative_BetterNet",
        experiment_name=f"DDQN_DeOnly_Temp_{temperature}",
        config={
            "state_dim": state_dim,
            "action_dim": action_dim,
            "batch_size": agent.batch_size,
            "gamma": agent.gamma,
            "epsilon": agent.epsilon,
            "update_target_freq": agent.update_target_freq,
            "replay_buffer_size": 1000,
            "learning_rate": 1e-3,
            "episode": 80000,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.998,
        },
        description="Not available"
    )

    # ========== 训练阶段 ==========
    agent.epsilon = swanlab.config["epsilon_start"]

    print(env.action_space)
    print(env.uplink_rate)
    print(temperature)
    static_action_list = [(3, 240), (4, 240)]
    throughput_list = np.zeros((len(static_action_list)+1, args.num_epochs))
    test_num = 0
    list_epoch = []

    for idx, sample in enumerate(test_samples):
        input_text = get_input(args, sample)
        input_ids = tokenizer([input_text], return_tensors='pt').to(torch_device).input_ids[:,:400]
        token_persec_AL, token_num_AL, _ = benchmark(autoregressive_sampling, "AS_large", use_profiling,
                                                    input_ids, large_model, tokenizer, num_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        
        if idx > 0:
            meter.update("AS_large", token_persec_AL, token_num_AL)
        if idx==5:
            break
    best_tp = 0
    for episode in range(swanlab.config["episode"]):
        random_train_sample = train_samples[random.randint(0, len(train_samples) - 1)]
        input_text = get_input(args, random_train_sample)
        input_ids = tokenizer([input_text], return_tensors='pt').to(torch_device).input_ids[:,:400] 
        
        env.reset(input_ids)
        total_reward = 0
        state = env.extended_state
        done = False
        while True:
            action = agent.choose_action(state)
            next_state, reward, done  = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.train()

            total_reward += reward
            state = next_state
            if done or total_reward > 2e4:
                break

        agent.epsilon = max(swanlab.config["epsilon_end"], agent.epsilon * swanlab.config["epsilon_decay"])  

        if episode % 250 == 1:
            meter.reset_sp()
            avg_reward = agent.evaluate(args, env, test_samples, tokenizer, meter)
            results = meter.report()
            sp_dynamic = results["SP"]["average_token_persec"] / results["AS_large"]["average_token_persec"]
            print(f'Current speedup results is {sp_dynamic}')
            throughput_list[0, test_num] = sp_dynamic * results["AS_large"]["average_token_persec"]

            for i in range(len(static_action_list)):
                meter.reset_sp()
                avg_reward_static = agent.evaluate(args, env, test_samples, tokenizer, meter, static_action_list[i])
                results = meter.report()
                sp = results["SP"]["average_token_persec"] / results["AS_large"]["average_token_persec"]
                throughput_list[i+1,test_num] = sp * results["AS_large"]["average_token_persec"]
                print(f'Current speedup results is {sp} when action {static_action_list[i]}')

            if throughput_list[0, test_num] > best_tp:
                best_tp = throughput_list[0, test_num]
                agent.best_net.load_state_dict({k: v.clone() for k, v in agent.q_net.state_dict().items()})
                agent.save_model(path=f"{ckpt_direc}/best_model.pth")
                print(f"New best model saved with throuput: {throughput_list[0, test_num]}")
            
            # if avg_reward > agent.best_avg_reward:
            #     agent.best_avg_reward = avg_reward
            #     agent.best_net.load_state_dict({k: v.clone() for k, v in agent.q_net.state_dict().items()})
            #     agent.save_model(path=f"{ckpt_direc}/best_model.pth")
            #     print(f"New best model saved with average reward: {avg_reward}")


            if args.save:
                os.makedirs(ckpt_direc, exist_ok=True)
                ckpt_file_name = os.path.join(ckpt_direc, f"checkpoint-{episode}.pth.tar")
                save_checkpoint(
                    {
                        "episode": episode,
                        "state_dict": agent.q_net.state_dict(),
                    },
                    filename=ckpt_file_name)

            test_num += 1
            plt.figure(figsize=(10, 6))
            list_epoch.append(test_num)
            x = list_epoch
            marker_list = ['o','s','*','>','d']
            for i in range(len(static_action_list)):
                plt.plot(x, throughput_list[i+1,:test_num], label=f'Static {(static_action_list[i])}', marker=marker_list[i])
            plt.plot(x, throughput_list[0, :test_num], label='Dynamic (learned)', marker='^')
            plt.xlabel('Epoch')
            plt.ylabel('Token Throughput')
            plt.title('Speedup Comparison (Updated at Epoch {})'.format(test_num))
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{plot_direc}/speedup_plot.pdf")
            plt.close()

        print(f"Episode: {episode}, Train Reward: {total_reward:.3f}, Best Eval Avg Reward: {agent.best_avg_reward:.3f}")

        swanlab.log(
            {
                "train/reward": total_reward,
                "eval/best_avg_reward": agent.best_avg_reward,
                "train/epsilon": agent.epsilon
            },
            step=episode,
        )

if __name__ == "__main__":
    args = parse_arguments()
    
    main(args, args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
             random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)



