#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python code/main.py \
--mode certify \
--dataset_name sst2 \
--training_type sparse \
--sparse_mask_rate 0.9 \
--certify_numbers 100 \
--predict_ensemble 50 \
--ceritfy_ensemble 500 \
--denoise_method alpaca \
--predictor alpaca_sst2 \
--alpaca_batchsize 4 \
--world_size 1 \
--mask_word "###" \
