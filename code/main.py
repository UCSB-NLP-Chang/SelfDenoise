# print('0')
# import tensorflow_hub
import nltk
# nltk.data.path[0] = '/data/private/zhangzhen/dir3/RanMASK/nltk_data'
# nltk.download('stopwords')
import logging
from args import ClassifierArgs
from classifier import Classifier
from alpaca import Alpaca
import datasets
import transformers
import torch
import numpy as np
import random
import sys 
import os

try:
    import tensorflow as tf
except ModuleNotFoundError as err:
    logging.warning("Tensorflow not installed!")
try:
    import torch
except ModuleNotFoundError as err:
    logging.warning("Pytorch not installed!")
try:
    from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, log_loss, f1_score, precision_score
except ModuleNotFoundError as err:
    logging.warning("Scikit-learn not installed!")


cache_path = "cache_path/other"
datasets.config.cache_directory = cache_path
transformers.cache = cache_path

def set_random_seed(seed, deterministic=False, no_torch=False, no_tf=False):
    """
    Set the random seed for the reproducibility. Environment variable CUBLAS_WORKSPACE_CONFIG=:4096:8 is also needed.
    :param seed: the random seed
    :type seed: int
    :param deterministic: whether use deterministic, slower is True, cannot guarantee reproducibility if False
    :param no_torch: if torch is not installed, set this True
    :param no_tf: if tensorflow is not installed, set this True
    :type deterministic: bool
    """
    random.seed(seed)
    np.random.seed(seed)
    if not no_torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    if not no_tf:
        tf.random.set_seed(seed)
        
if __name__ == '__main__':
    args = ClassifierArgs()._parse_args()
    print(args)
    logging.info(args)



    save_path = f"output/-pre-{args.predictor}-denoise-{args.denoise_method}-maskrate-{args.sparse_mask_rate}-num-{args.certify_numbers}-predensmb-{args.predict_ensemble}-ctf_ensmb-{args.ceritfy_ensemble}-mw-{args.mask_word}-ran-{args.random_probs_strategy}"
    args.save_path = save_path
    if args.local_rank==0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(os.path.join(save_path,'org_sentence')):
            os.makedirs(os.path.join(save_path,'org_sentence'))
        if not os.path.exists(os.path.join(save_path,'pred_masked_sentence')):
            os.makedirs(os.path.join(save_path,'pred_masked_sentence'))
        if not os.path.exists(os.path.join(save_path,'certify_masked_sentence')):
            os.makedirs(os.path.join(save_path,'certify_masked_sentence'))
        if not os.path.exists(os.path.join(save_path,'pred_denoised_sentence')):
            os.makedirs(os.path.join(save_path,'pred_denoised_sentence'))
        if not os.path.exists(os.path.join(save_path,'certify_denoised_sentence')):
            os.makedirs(os.path.join(save_path,'certify_denoised_sentence'))
        if not os.path.exists(os.path.join(save_path,'pred_prediction')):
            os.makedirs(os.path.join(save_path,'pred_prediction'))
        if not os.path.exists(os.path.join(save_path,'certify_prediction')):
            os.makedirs(os.path.join(save_path,'certify_prediction'))
        if not os.path.exists(os.path.join(save_path,'pred_prediction_prob')):
            os.makedirs(os.path.join(save_path,'pred_prediction_prob'))
        if not os.path.exists(os.path.join(save_path,'certify_prediction_prob')):
            os.makedirs(os.path.join(save_path,'certify_prediction_prob'))

    set_random_seed(args.seed, no_tf=True)

    if "alpaca" in args.denoise_method or "alpaca" in args.predictor:
        alpaca = Alpaca(args)
    else:
        alpaca = None

    

    Classifier.run(args,alpaca=alpaca)

    print('finish')