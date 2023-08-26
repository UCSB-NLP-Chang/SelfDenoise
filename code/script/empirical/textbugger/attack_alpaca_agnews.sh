CUDA_VISIBLE_DEVICES=7 \
python code/main.py --mode attack --dataset_name agnews --attack_method textbugger --training_type None --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 50 \
--batch_size 64 \
--predictor alpaca_agnews \
--denoise_method None \
