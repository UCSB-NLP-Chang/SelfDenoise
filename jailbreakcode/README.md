# Jailbreak defense code

This directory contains the code for running jailbreak defense experiments.

An example bash command for running transfer jailbreak experiment is:
```bash
OUTDIR="transfer-results"
python transfer_attack.py \
	--results_dir $OUTDIR/GCG \
  --attack GCG \
  --attack_logfile data/GCG/vicuna_50.json \
  --smoothllm_pert_type DenoisePerturbation \
	--smoothllm_pert_pct 15 \
	--smoothllm_num_copies 10 \
	--smoothllm_pert_model vicuna \
	--logfile vicuna-gcg-copy=10-pert=DenoisePerturbation-pct=15.log
```

For adaptive attack experiment:
```bash
OUTDIR="PAIR-AdaptiveAttack"
ATTACKER="PAIR"

python adaptive_attack.py \
 	--results_dir $OUTDIR \
 	--attack GCG \
 	--attack_logfile data/harmful_behaviors_custom.json \
 	--smoothllm_pert_type RandomSwapPerturbation \
 	--smoothllm_pert_pct 15 \
 	--smoothllm_num_copies 10 --smoothllm_pert_model vicuna \
 	--attacker $ATTACKER \
 	--target_model_temperature 0.0 \
 	--target_model_top_p 1.0 \
 	--logfile vicuna-AutoDAN-RandomSwapPerturbation-pct=15.log --judge GPTJudge
```
