import torch
import random
import argparse
import contexttimer
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import csv
import shutil
import numpy as np
from tqdm import tqdm
from globals import Decoder
from collections import deque
from colorama import Fore, Style 
from huggingface_hub import login
from datasets import load_dataset

from fastchat.model import get_conversation_template


class Meter:
    def __init__(self):
        self.data = {}

    def update(self, method_name, token_persec, token_num):
        if method_name not in self.data:
            self.data[method_name] = {"token_persec_sum": 0, "token_num_sum": 0, "count": 0}
        self.data[method_name]["token_persec_sum"] += token_persec
        self.data[method_name]["token_num_sum"] += token_num
        self.data[method_name]["count"] += 1

    def get_average(self, method_name):
        if method_name in self.data and self.data[method_name]["count"] > 0:
            avg_token_persec = self.data[method_name]["token_persec_sum"] / self.data[method_name]["count"]
            avg_token_num = self.data[method_name]["token_num_sum"] / self.data[method_name]["count"]
            return avg_token_persec, avg_token_num
        return 0, 0

    def report(self):
        report = {}
        for method_name, stats in self.data.items():
            avg_token_persec, avg_token_num = self.get_average(method_name)
            report[method_name] = {
                "average_token_persec": avg_token_persec,
                "average_token_num": avg_token_num,
            }
        return report

    def reset_sp(self):
        """Resets all stored data to zero."""
        self.data["SP"] = {"token_persec_sum": 0, "token_num_sum": 0, "count": 0}
    # ["SP"] = {"token_persec_sum": 0, "token_num_sum": 0, "count": 0}

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def color_print(text, color='r'):
    if color=='r':
        print(Fore.RED + text + Style.RESET_ALL)
    elif color=='g':
        print(Fore.GREEN + text + Style.RESET_ALL)


def get_input(args, sample):
    if args.task == 'summarize':
        input_text = sample['article']
        # input_text = sample['document']
        if args.target_model_name.startswith('facebook/opt-'):
            input_text = f"Article: {input_text}\n Summary:"
        elif args.target_model_name.endswith('-chat-hf'):
            input_text = f"""<s>[INST] <<SYS>>
                    You are an expert summarizer. Generate a summary in English, capturing key points concisely.
                    <</SYS>>

                    Summarize this text:\n{input_text} [/INST]"""
        elif args.target_model_name.endswith('-hf'):
            input_text = f"<s> Summarize the following text:\n\n{sample['document']}\n\nSummary:"
        elif args.target_model_name.startswith('lmsys') or args.target_model_name.startswith('double7/'):
            input_text = f"Summarize: {input_text}"
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], input_text)
            conv.append_message(conv.roles[1], None)
            conv.stop_str = "</s>"
            input_text = conv.get_prompt()
        elif args.target_model_name.startswith('openai'):
            input_text = f"Article: {input_text} Summary:"
        elif args.target_model_name.startswith('google'):
            input_text = f"summarize: {input_text}"
    
    elif args.task == 'write':
        input_text = sample['translation']['en']
        if args.target_model_name.startswith('openai'):
            input_text = f"Write: {input_text}:"

    elif args.task == 'translate':
        input_text = sample['translation']['en']
        if args.target_model_name.startswith('facebook/opt-'):
            input_text = f"German: {input_text} Translated English:"
        elif args.target_model_name.endswith('-chat-hf'):
            input_text = f"""<s>[INST] <<SYS>>
                    You are an expert translator.
                    <</SYS>>

                    Translate the following German text to English:\n{input_text}[/INST]"""
        elif args.target_model_name.endswith('-hf'):
            input_text = f"<s> Translate the following German text to English:\n\n{input_text}\n\nSummary:"
        elif args.target_model_name.startswith('lmsys') or args.target_model_name.startswith('double7/'):
            input_text = f"Translate the following German text to English: {input_text}"
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], input_text)
            conv.append_message(conv.roles[1], None)
            conv.stop_str = "</s>"
            input_text = conv.get_prompt()
        elif args.target_model_name.startswith('google'):
            input_text = f"translate English to German: {input_text}"
        
    
    return input_text


def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 1
    profile_filename = f"./profile_logs/{print_prefix}"
    with contexttimer.Timer() as t:
        for _ in range(TEST_TIME): 
            output, times_count = fn(*args, **kwargs)
    print(f"[Temp] {print_prefix:<8}, tokens/sec: {output.shape[1] / t.elapsed * TEST_TIME:.2f}, {t.elapsed / TEST_TIME:.2f} sec generates {output.shape[1]} tokens")
    token_persec = output.shape[1] / t.elapsed * TEST_TIME
    token_num = output.shape[1]
    return token_persec, token_num, times_count


def benchmark_env(print_prefix, env):
    TEST_TIME = 1
    profile_filename = f"./profile_logs/{print_prefix}"
    with contexttimer.Timer() as t:
        for _ in range(TEST_TIME): 
            while True:
                action = 0
                next_state, reward, done = env.step(action)
                state = next_state
                if done:
                    break
            output = env.prefix[:,env.prifex_len:]
            times_count = env.times_count
    print(f"[Temp] {print_prefix:<8}, tokens/sec: {output.shape[1] / t.elapsed * TEST_TIME:.2f}, {t.elapsed / TEST_TIME:.2f} sec generates {output.shape[1]} tokens")
    token_persec = output.shape[1] / t.elapsed * TEST_TIME
    token_num = output.shape[1]
    return token_persec, token_num, times_count


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


        