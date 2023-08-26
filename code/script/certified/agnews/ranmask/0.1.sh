#!/bin/bash
python code/main.py \
--mode certify \
--dataset_name agnews \
--training_type sparse \
--sparse_mask_rate 0.1 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--denoise_method None \
--predictor alpaca_agnews \
--alpaca_batchsize 3 \
--world_size 1 \
--mask_word "<mask>" \