# This file contains common arguments for all scripts in this repo

import argparse

common_parser = argparse.ArgumentParser()

common_parser.add_argument(
    '--seed',
    type=int,
    default=42,
)

common_parser.add_argument(
    '--results_dir',
    type=str,
    default='./temporal_gen_results'
)
common_parser.add_argument(
    '--trial',
    type=int,
    default=0
)

# Targeted LLM
common_parser.add_argument(
    '--target_model',
    type=str,
    default='vicuna',
    choices=['vicuna', 'llama2', 'alpaca']
)
common_parser.add_argument(
    '--target_model_temperature',
    type=float,
    default=0.0,
)
common_parser.add_argument(
    '--target_model_top_p',
    type=float,
    default=1.0,
)

# Attacking LLM
common_parser.add_argument(
    '--attack',
    type=str,
    default='GCG',
    choices=['GCG', 'PAIR', 'InWildPrompt', 'SST2', 'AGNews']
)
common_parser.add_argument(
    '--attack_logfile',
    type=str,
    default='data/GCG/vicuna_behaviors.json'
)

# SmoothLLM
common_parser.add_argument(
    '--smoothllm_num_copies',
    type=int,
    default=10,
)

common_parser.add_argument(
    '--smoothllm_pert_pct',
    type=int,
    default=10,
)

common_parser.add_argument(
    '--smoothllm_pert_model',
    type=str,
    choices=[
        'gpt-3.5-turbo-1106',
        'vicuna',
        'llama2',
        'alpaca',
    ]
)

common_parser.add_argument(
    '--smoothllm_pert_type',
    nargs='+',
    default=['RandomSwapPerturbation'],
    choices=[
        'RandomSwapPerturbation',
        'RandomPatchPerturbation',
        'RandomInsertPerturbation',

        'GPTPerturbation',
        'LLMPerturbation',
        'NoPerturbation',

        'ParaphrasePerturbation',
        'SummarizePerturbation',
        'DenoisePerturbation',
        
        'MaskPerturbation',
    ]
)

# SemanticSmooth LLM 
common_parser.add_argument(
    '--smoothllm_pert_prompt_file',
    type=str,
    default='data/prompt-template/summarize.txt',
    choices=[
        'data/prompt-template/summarize.txt',
        'data/prompt-template/paraphrase.txt',
    ]
)

# The output log file
common_parser.add_argument(
    '--logfile',
    type=str,
    default="debug.log",
)
