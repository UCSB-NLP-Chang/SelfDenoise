CUDA_VISIBLE_DEVICES=2 \
python code/main.py --mode attack --dataset_name sst2 --attack_method textbugger --training_type sparse --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 2 \
--batch_size 8 \
--predictor alpaca_sst2 \
--denoise_method alpaca \
--mask_word "###" \

