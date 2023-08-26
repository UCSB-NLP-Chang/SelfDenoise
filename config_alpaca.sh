#!/bin/bash

mkdir alpaca
cd alpaca
# get convert_llama_weights_to_hf.py
wget https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/convert_llama_weights_to_hf.py -O convert_llama_weights_to_hf.py

#download llama
mkdir llama_ckpt
mkdir llama_ckpt/7B
wget https://agi.gpt4.org/llama/LLaMA/tokenizer.model -O ./llama_ckpt/tokenizer.model
wget https://agi.gpt4.org/llama/LLaMA/tokenizer_checklist.chk -O ./llama_ckpt/tokenizer_checklist.chk
wget https://agi.gpt4.org/llama/LLaMA/7B/consolidated.00.pth -O ./llama_ckpt/7B/consolidated.00.pth
wget https://agi.gpt4.org/llama/LLaMA/7B/params.json -O ./llama_ckpt/7B/params.json
wget https://agi.gpt4.org/llama/LLaMA/7B/checklist.chk -O ./llama_ckpt/7B/checklist.chk

# covert llamma weight to hf version
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 convert_llama_weights_to_hf.py \
    --input_dir llama_ckpt --model_size 7B --output_dir llama_hf_ckpt

# get alpaca-7b-wdiff
git clone https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
cd alpaca-7b-wdiff
wget "https://cdn-lfs.huggingface.co/repos/68/09/680962c5eed173b1e242b169de0b55454b7cd7ad04e5c93b5db8db8dac9ec68c/95005cba09e22aa4cb30927810a673d7deffed7b7fa14067217626990fa1a114?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27pytorch_model-00001-of-00003.bin%3B+filename%3D%22pytorch_model-00001-of-00003.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1691064431&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTY5MTA2NDQzMX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy82OC8wOS82ODA5NjJjNWVlZDE3M2IxZTI0MmIxNjlkZTBiNTU0NTRiN2NkN2FkMDRlNWM5M2I1ZGI4ZGI4ZGFjOWVjNjhjLzk1MDA1Y2JhMDllMjJhYTRjYjMwOTI3ODEwYTY3M2Q3ZGVmZmVkN2I3ZmExNDA2NzIxNzYyNjk5MGZhMWExMTQ%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=BTW6tYIyVmoDx8n8qo2nBEQmNyDYXbJQvyluSmAAtwymvbD2CfjJUADoPqIyaeDs%7EzbIATU67GtwuTni51xQ2SngSaHxx3LRtn5LhPXbbHl14MOTMMvc6-SGoe3f-nc13zO5QjkofKHR-pyRmqGOw4NNVVkYQxYzQovxsPqifdEr%7EIexpywxEgc53tt1NHG1G8qHvLwJe0sBwIw2ri8ZOsdl2L-fDSZtRbuRBv8bgPfLwysCJBF8UjYOI5AtU9RPjlw41JMfp9qdssYSE6QRZuQC-HejliU1y-07Uumy89eHhqg5gTYc3XqXRObm-fDnVLxSvkEV2vfQ6%7EV9iRzjyA__&Key-Pair-Id=KVTP0A1DKRTAX" -O pytorch_model-00001-of-00003.bin
wget "https://cdn-lfs.huggingface.co/repos/68/09/680962c5eed173b1e242b169de0b55454b7cd7ad04e5c93b5db8db8dac9ec68c/cf76cc78bec93ccbbc18ecbb686d08d4916aa00781bf68457805c9ba0761974b?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27pytorch_model-00002-of-00003.bin%3B+filename%3D%22pytorch_model-00002-of-00003.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1691064455&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTY5MTA2NDQ1NX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy82OC8wOS82ODA5NjJjNWVlZDE3M2IxZTI0MmIxNjlkZTBiNTU0NTRiN2NkN2FkMDRlNWM5M2I1ZGI4ZGI4ZGFjOWVjNjhjL2NmNzZjYzc4YmVjOTNjY2JiYzE4ZWNiYjY4NmQwOGQ0OTE2YWEwMDc4MWJmNjg0NTc4MDVjOWJhMDc2MTk3NGI%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=IV5QamsPI97Sfo-hd5Yh8VUNhJrsrpN3hx5fqjaRDAZpZLDzbgm7hUggfxumsA5Bq2Gpfvi%7EVH-6B3KsJ%7E-Ry86Hyn0tyVvNoAmp%7Eu%7EiTm8YNAGXer2jMbHf-9BbONw7TxFGHnqGBRWZ3hojcsJbJ774fUrgFZyLrTYZsKHK5n3duwd%7EYrC9zqct0-U90IkHVicrn4zjx4XRH8aCEF6zuCZE4clY2dzyG1orY-NkI2nXYkzAsURQBFmjdS2Y5mxeghAcnyW601fv0z15qumonf4sgTgZ9yzLsLxziBYOjbv-9YDKFh9RmqQxsC%7E9tnUR4D7JFzeoNFqviaiYU3almA__&Key-Pair-Id=KVTP0A1DKRTAX" -O pytorch_model-00002-of-00003.bin
wget "https://cdn-lfs.huggingface.co/repos/68/09/680962c5eed173b1e242b169de0b55454b7cd7ad04e5c93b5db8db8dac9ec68c/6f04361aac986bd6031b5f6ae8db2af30cb2e7edd792d9c1ee4074e47c9608eb?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27pytorch_model-00003-of-00003.bin%3B+filename%3D%22pytorch_model-00003-of-00003.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1691064476&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTY5MTA2NDQ3Nn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy82OC8wOS82ODA5NjJjNWVlZDE3M2IxZTI0MmIxNjlkZTBiNTU0NTRiN2NkN2FkMDRlNWM5M2I1ZGI4ZGI4ZGFjOWVjNjhjLzZmMDQzNjFhYWM5ODZiZDYwMzFiNWY2YWU4ZGIyYWYzMGNiMmU3ZWRkNzkyZDljMWVlNDA3NGU0N2M5NjA4ZWI%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qJnJlc3BvbnNlLWNvbnRlbnQtdHlwZT0qIn1dfQ__&Signature=fnBPfgLUuKvCaeA9mK9xa455r%7E-x7D3RuzT8TTbc%7EcZ1sdRWjrS0EtihM47oK9bDQfAp%7Ek7sGw1wIf23WJi-GNzFHPnJY6YQEMZS6195kVuAG%7E3%7EYCGwAsT5AAGG8VAE-HNWfpUx9ucL4U2UJZuTGZhe60gOxejVZKxC33E8eVYfbB-BCRWq5g7Yuk-MinHXrO99y31auZ5V5p3g99pN1VkTzbaggPXJ%7EH0Wsiw9sy%7EQfoX8vI5antscNGV7RtKe6V1ViX-MmE%7Exb5lqHUcTllbVrATHXPBpBryz6Ylg1cCL2EaaWJcPFPRV2x5%7E0Q9mqtjj4MXjRP4SMMgIZn1hqg__&Key-Pair-Id=KVTP0A1DKRTAX" -O pytorch_model-00003-of-00003.bin
wget "https://cdn-lfs.huggingface.co/repos/68/09/680962c5eed173b1e242b169de0b55454b7cd7ad04e5c93b5db8db8dac9ec68c/9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27tokenizer.model%3B+filename%3D%22tokenizer.model%22%3B&Expires=1691064532&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTY5MTA2NDUzMn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy82OC8wOS82ODA5NjJjNWVlZDE3M2IxZTI0MmIxNjlkZTBiNTU0NTRiN2NkN2FkMDRlNWM5M2I1ZGI4ZGI4ZGFjOWVjNjhjLzllNTU2YWZkNDQyMTNiNmJkMWJlMmI4NTBlYmJiZDk4ZjU0ODE0MzdhODAyMWFmYWY1OGVlN2ZiMTgxOGQzNDc%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qIn1dfQ__&Signature=osntPathVezF3NxwO6-ch1WX6sXg8RjKHzG3vut7vAgSeaE9dOWfH7cSyY56LOqWxURxevo5pO2y7DOftYClnyiipCCsDvDr3g98ue07nTurqw2xwHp%7Ep4U2sIxIyUUqSdRtjZFVZMl4Lxoq%7Evk9h%7EQ6MdMeKpPaDDbx2oJPyqy2qujsZN4ym7g7RlaLcpBYYQEu7nEpxUEO8eNcOyDJL2hcFbsPGYkS-L5We10ufL9lUEGWq0z5edKBbJG%7ECSMpIj1s4wcTUrw7AkG-k7F1pUawhF2o8wvhocDN89HehhR%7Es%7EYhzPg4Ryl0IgXiO7MX1Fb9H6HQhdMH7FmKSa5Q2A__&Key-Pair-Id=KVTP0A1DKRTAX" -O tokenizer.model
cd ..

# generate alpaca ckpt
git clone git@github.com:tatsu-lab/stanford_alpaca.git
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python stanford_alpaca/weight_diff.py recover --path_raw llama_hf_ckpt --path_diff alpaca-7b-wdiff --path_tuned alpaca_ckpt







# wget https://agi.gpt4.org/llama/LLaMA/tokenizer.model -O ./tokenizer.model
# wget https://agi.gpt4.org/llama/LLaMA/tokenizer_checklist.chk -O ./tokenizer_checklist.chk
# wget https://agi.gpt4.org/llama/LLaMA/7B/consolidated.00.pth -O ./7B/consolidated.00.pth
# wget https://agi.gpt4.org/llama/LLaMA/7B/params.json -O ./7B/params.json
# wget https://agi.gpt4.org/llama/LLaMA/7B/checklist.chk -O ./7B/checklist.chk
# wget https://agi.gpt4.org/llama/LLaMA/13B/consolidated.00.pth -O ./13B/consolidated.00.pth
# wget https://agi.gpt4.org/llama/LLaMA/13B/consolidated.01.pth -O ./13B/consolidated.01.pth
# wget https://agi.gpt4.org/llama/LLaMA/13B/params.json -O ./13B/params.json
# wget https://agi.gpt4.org/llama/LLaMA/13B/checklist.chk -O ./13B/checklist.chk
# wget https://agi.gpt4.org/llama/LLaMA/30B/consolidated.00.pth -O ./30B/consolidated.00.pth
# wget https://agi.gpt4.org/llama/LLaMA/30B/consolidated.01.pth -O ./30B/consolidated.01.pth
# wget https://agi.gpt4.org/llama/LLaMA/30B/consolidated.02.pth -O ./30B/consolidated.02.pth
# wget https://agi.gpt4.org/llama/LLaMA/30B/consolidated.03.pth -O ./30B/consolidated.03.pth
# wget https://agi.gpt4.org/llama/LLaMA/30B/params.json -O ./30B/params.json
# wget https://agi.gpt4.org/llama/LLaMA/30B/checklist.chk -O ./30B/checklist.chk
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.00.pth -O ./65B/consolidated.00.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.01.pth -O ./65B/consolidated.01.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.02.pth -O ./65B/consolidated.02.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.03.pth -O ./65B/consolidated.03.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.04.pth -O ./65B/consolidated.04.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.05.pth -O ./65B/consolidated.05.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.06.pth -O ./65B/consolidated.06.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/consolidated.07.pth -O ./65B/consolidated.07.pth
# wget https://agi.gpt4.org/llama/LLaMA/65B/params.json -O ./65B/params.json
# wget https://agi.gpt4.org/llama/LLaMA/65B/checklist.chk -O ./65B/checklist.chk
