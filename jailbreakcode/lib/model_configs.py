BASEDIR = 'Directory to local model weights'

MODELS = {
    'llama2': {
        'model_path': f'{BASEDIR}/Llama-2-7b-chat-hf',
        'tokenizer_path': f'{BASEDIR}/Llama-2-7b-chat-hf',
        'conversation_template': 'llama-2'
    },
    'vicuna': {
        'model_path': f'{BASEDIR}/vicuna-13b-v1.5',
        'tokenizer_path': f'{BASEDIR}/vicuna-13b-v1.5',
        'conversation_template': 'vicuna'
    },
    
}

# THIS DEFAULT API KEY 
OPENAI_API_KEY = ""
