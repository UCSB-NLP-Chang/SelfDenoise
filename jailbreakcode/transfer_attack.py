import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse
import structlog
from structlog.contextvars import (
    bind_contextvars,
    
    bound_contextvars,
)

import lib.perturbations as perturbations
import lib.defenses as defenses
import lib.attacks as attacks
import lib.language_models as language_models
import lib.model_configs as model_configs
from lib.judges import NaiveJudge, add_judge_args
from lib.log import configure_structlog
from lib.arguments import common_parser

import random

def main(args):
    os.makedirs(args.results_dir, exist_ok=True)
    configure_structlog(filename=os.path.join(args.results_dir, args.logfile))


    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed) # ensure different perturbations

    config = model_configs.MODELS[args.target_model]

    target_model = language_models.LocalGPT(
        model_name=config['model_path'],
        tokenizer_path=config['tokenizer_path'],
        conv_template_name=config['conversation_template'],
        temperature=args.target_model_temperature,
        top_p=args.target_model_top_p,
    )

    attack = vars(attacks)[args.attack](
        logfile=args.attack_logfile,
        target_model=target_model,
    )

    # Create SmoothLLM instance
    if args.smoothllm_pert_type[0] in perturbations.LLMPerturbations:
        perturbation_fns = []
        for perturb_type in args.smoothllm_pert_type:
            if perturb_type in perturbations.LLMPerturbations:
                perturbation_fn = perturbations.create_semantic_perturbation(
                    pert_model=args.smoothllm_pert_model,
                    pert_type=perturb_type,
                    mask_rate=args.smoothllm_pert_pct,
                )
            else: # char-level perturbation
                perturbation_fn = vars(perturbations)[perturb_type](q=args.smoothllm_pert_pct)
            perturbation_fns.append(perturbation_fn)

        defense = defenses.SemanticSmoothLLM(
            target_model=target_model,
            judge=NaiveJudge(args),
            num_copies=args.smoothllm_num_copies,
            perturbation_fns=perturbation_fns,
        )

    else:
        assert len(args.smoothllm_pert_type) == 1
        args.smoothllm_pert_type = args.smoothllm_pert_type[0]
        assert args.smoothllm_pert_type in perturbations.NonLLMPerturbations

        defense = defenses.SmoothLLM(
            target_model=target_model,
            pert_type=args.smoothllm_pert_type,
            pert_pct=args.smoothllm_pert_pct,
            num_copies=args.smoothllm_num_copies,
            judge=NaiveJudge(args),
        )

    os.makedirs(os.path.join(args.results_dir, 'all-outputs'), exist_ok=True)

    LOGGER = structlog.get_logger()
    for i, prompt in enumerate(tqdm(attack.prompts)):
        
        # import ipdb; ipdb.set_trace()
        all_inputs, all_outputs = defense.batch_generate(
            prompt,
            args.smoothllm_num_copies,
            return_perturb=True,
        )

        attack_prompt = prompt.full_prompt
        with bound_contextvars(model=args.target_model, attack=args.attack, datasource=args.attack_logfile, index=i):
            LOGGER.info(
                "ModelOutputs", 
                goal=prompt.goal,
                attack_prompt=attack_prompt, 
                raw_attack=prompt.perturbable_prompt,
                perturbed_attack=all_inputs, 
                outputs=all_outputs,
            )

if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = common_parser
    add_judge_args(parser)

    args = parser.parse_args()
    main(args)