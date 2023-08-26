CUDA_VISIBLE_DEVICES=4 \
python code/main.py --mode attack --dataset_name agnews --attack_method deepwordbug --training_type sparse --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 50 \
--batch_size 4 \
--predictor alpaca_agnews \
--denoise_method alpaca \
--mask_word "###" \
