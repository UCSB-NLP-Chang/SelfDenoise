import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import structlog
from structlog.contextvars import bound_contextvars
import copy
import random

import lib.judges as judges
import lib.perturbations as perturbations
import lib.defenses as defenses
import lib.attacks as attacks
import lib.language_models as language_models
import lib.model_configs as model_configs
from lib.judges import NaiveJudge, add_judge_args
from lib.log import configure_structlog
from lib.arguments import common_parser
from lib.attack_pair import PAIRAttackLLM

def main(args):

    os.makedirs(args.results_dir, exist_ok=True)
    configure_structlog(
        filename=os.path.join(
            args.results_dir, 
            f"target={args.target_model}-attacker={args.attacker}-defender={args.smoothllm_pert_type}-copy={args.smoothllm_num_copies}-pert_pct={args.smoothllm_pert_pct}-max_iter={args.max_iter}.log"
        )
    )
    config = model_configs.MODELS[args.target_model]

    if args.attacker == 'PAIR':
        attacker = PAIRAttackLLM(
            args.attack_model,
            args.attack_max_n_tokens,
            max_n_attack_attempts=args.attack_max_n_attack_attempts, 
            temperature=1, # default attack hyper-param
            top_p=0.9,  # default attack hyper-param
            conversation_num=1, # only one conversation
        )

    # Define a seperate model running as a service for ease
    target_model = language_models.LocalGPT(
        model_name=config['model_path'],
        tokenizer_path=config['tokenizer_path'],
        conv_template_name=config['conversation_template'],
        temperature=args.target_model_temperature,
        top_p=args.target_model_top_p,
    )

    judge = vars(judges)[args.judge](args)

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

        # judge=NaiveJudge(args),
        defender = defenses.SemanticSmoothLLM(
            target_model=target_model,
            judge = judge,
            num_copies=args.smoothllm_num_copies,
            perturbation_fns=perturbation_fns,
        )
    else:
        assert len(args.smoothllm_pert_type) == 1
        args.smoothllm_pert_type = args.smoothllm_pert_type[0]
        assert args.smoothllm_pert_type in perturbations.NonSemanticPerturbations

        defender = defenses.SmoothLLM(
            target_model=target_model,
            pert_type=args.smoothllm_pert_type,
            pert_pct=args.smoothllm_pert_pct,
            num_copies=args.smoothllm_num_copies,
            judge=judge,
        )

    LOGGER = structlog.get_logger()

    assert args.attack == 'GCG'
    attack = vars(attacks)[args.attack](
        logfile=args.attack_logfile,
        target_model=target_model,
    )

    for i, prompt in enumerate(tqdm(attack.prompts)):
        attacker.reset_new_goal(prompt.goal, prompt.target_str)

        with bound_contextvars(
            index=i, 
        ):
            LOGGER.info("AttackDetail", goal=prompt.goal)
            attack_success = False
            iter = 0
            target_response = ""
            for iter in range(args.max_iter):
                if attacker.__class__.__name__ == "PAIRAttackLLM":
                    attack_string = attacker.generate_attack(prompt, target_response)
                else:
                    attack_string = attacker.generate_attack(prompt)

                prompt_copy = copy.deepcopy(prompt)
                prompt_copy.perturbable_prompt = attack_string
                prompt_copy.goal = prompt.goal
                prompt_copy.full_prompt = prompt_copy.new_full_prompt(conv_template=args.target_model)
                all_inputs, all_outputs = defender.batch_generate(
                    prompt_copy,
                    args.smoothllm_num_copies,
                    return_perturb=True,
                )

                are_copies_jailbroken = [
                    defender.judge.is_jail_broken(prompt_copy.goal, prompt_copy.full_prompt, s) for s in all_outputs]
                if len(are_copies_jailbroken) == 0:
                    raise ValueError("LLM did not generate any outputs.")
                outputs_and_jbs = zip(all_outputs, are_copies_jailbroken)
                jb_percentage = np.mean(are_copies_jailbroken)
                smoothLLM_jb = True if jb_percentage > 0.5 else False

                # Pick a response that is consistent with the majority vote
                majority_outputs = [
                    output for (output, jb) in outputs_and_jbs 
                    if jb == smoothLLM_jb
                ]
                target_response = random.choice(majority_outputs)
                LOGGER.info(
                    "IterOutputs", 
                    iter=iter,
                    perturbed_attack=all_inputs,
                    responses=all_outputs,
                    jb_result=are_copies_jailbroken,
                    is_jb=smoothLLM_jb,
                )
 
                if smoothLLM_jb:
                    attack_success = True
                    break

            LOGGER.info(
                "AttackResult",
                attack_success=attack_success,
                success_iter=iter+1,
            )


def add_local_args(parser):
    parser.add_argument('--attacker', choices=[ 'PAIR'])
    parser.add_argument('--max_iter', default=20)
    parser.add_argument(
        '--attack_model', 
        type=str,
        default='vicuna',
        choices=['vicuna', 'llama2', 'gpt-3.5-turbo-1106',]
    )
    parser.add_argument(
        '--attack-max-n-tokens',
        type=int,
        default=500,
    )
    parser.add_argument(
        '--attack-max-n-attack-attempts',
        type=int,
        default=20,
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = common_parser
    add_judge_args(parser)
    add_local_args(parser)

    args = parser.parse_args()
    main(args)
