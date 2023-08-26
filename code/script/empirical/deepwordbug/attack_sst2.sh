CUDA_VISIBLE_DEVICES=5 \
python code/main.py --mode attack --dataset_name sst2 --attack_method deepwordbug --training_type sparse --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 50 \
--batch_size 200 \
--predictor alpaca_sst2 \

