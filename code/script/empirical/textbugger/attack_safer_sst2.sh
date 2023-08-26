CUDA_VISIBLE_DEVICES=4 \
python code/main.py --mode attack --dataset_name sst2 --attack_method textbugger --training_type safer --attack_numbers 200 \
--sparse_mask_rate 0.05 \
--predict_ensemble 50 \
--batch_size 256 \
--predictor alpaca_sst2 \
--denoise_method None \

