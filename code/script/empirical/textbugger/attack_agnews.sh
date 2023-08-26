CUDA_VISIBLE_DEVICES=1 \
python code/main.py --mode attack --dataset_name agnews --attack_method textbugger --training_type sparse --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 50 \
--batch_size 64 \
--predictor alpaca_agnews \

