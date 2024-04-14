import torch
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os
import openai
from typing import Any, List, Dict
import time
from .model_configs import OPENAI_API_KEY
from .simplecache import APICacher, cache_api_call, SimpleCache

class LLM:

    """Forward pass through a LLM."""

    def __init__(
        self, 
        model_path, 
        tokenizer_path, 
        conv_template_name,
        device,
        temperature=0.0,
    ):

        # Language model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            use_cache=True
        ).to(device).eval()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.tokenizer.padding_side = 'left'
        if 'llama-2' in tokenizer_path:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fastchat conversation template
        self.conv_template = get_conversation_template(
            conv_template_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()
        
        self.temperature = temperature

    def __call__(self, batch, max_new_tokens=100):

        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(
            batch, 
            padding=True, 
            truncation=False, 
            return_tensors='pt'
        )
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

        # Forward pass through the LLM
        try:
            #! The LLM currently only support greedy decoding
            if self.temperature == 0.0:
                otherkwargs = {'do_sample' : False, 'temperature' : self.temperature}
            else:
                otherkwargs = {'do_sample' : True, 'temperature' : self.temperature}
            
            outputs = self.model.generate(
                batch_input_ids, 
                attention_mask=batch_attention_mask, 
                max_new_tokens=max_new_tokens,
                **otherkwargs,
            )
        except RuntimeError:
            return []

        # Decode the outputs produced by the LLM
        batch_outputs = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        gen_start_idx = [
            len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True)) 
            for i in range(len(batch_input_ids))
        ]
        batch_outputs = [
            output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)
        ]

        return batch_outputs

def call_openai(client : openai.Client, model, messages, max_tokens, temperature, top_p, request_timeout, seed):
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
        timeout = request_timeout,
        seed=seed, #! Add some control
        response_format={ "type": "json_object" }
    )
    response = response.dict()
    return response

class GPT:

    """ Apply an API call to openai, return the API response.
        Borrowed from https://github.com/patrickrchao/JailbreakingLLMs/blob/main/language_models.py#L21
        Add caching mechanism to the OpenAI API responses
    """
    
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    CACHE_DIR = "cache"


    def __init__(self, model_name) -> None:
        self.client = openai.Client(
            api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
        )
        self.model_name = model_name
        # api call wrapper, with caching
        self.call_openai = cache_api_call(APICacher)(call_openai)

    def generate(self, conv: List[Dict], 
                max_n_tokens: int = 512, 
                temperature: float = 0.0,
                top_p: float = 0.9,
                seed: int = 42):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.call_openai(
                    client= self.client,
                    model = self.model_name,
                    messages = conv,
                    max_tokens = max_n_tokens,
                    temperature = temperature,
                    top_p = top_p,
                    request_timeout = self.API_TIMEOUT,
                    seed=seed, #! Add some control
                )
                output = response["choices"][0]["message"]["content"]
                break
            except openai.OpenAIError as e:
                print(f"Error at {self.client.base_url}: ", type(e), e) 
                time.sleep(self.API_RETRY_SLEEP)
        
        return output 
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float = 0.0,
                        top_p: float = 0.9,
                        seed: int=42):
        return [
            self.generate(conv, max_n_tokens, temperature, top_p, seed) for conv in convs_list]
    
    def __call__(self, convs_list, max_n_tokens, temperature, top_p, seed) -> Any:
        openai.base_url = None
        return self.batched_generate(
           convs_list, max_n_tokens, temperature, top_p, seed
        )

class LocalGPT(GPT):
    """ Apply an API call to a local vllm server, return the API response.
        This is used when we have a local LLM server provided by vllm
        See: https://docs.vllm.ai/en/latest/getting_started/quickstart.html
    """
    
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 40
    CACHE_DIR = "local-cache"


    def __init__(self, model_name, tokenizer_path, conv_template_name, temperature=0.0, top_p=1.0) -> None:
        
        self.client = openai.Client(
            api_key = "EMPTY",
            base_url="http://localhost:10000/v1/",
            # base_url="http://localhost:21345/v1/",
        )
        self.model_name = model_name

        # Tokenizer is used for constructing the prompt
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.padding_side = 'left'
        if 'llama-2' in tokenizer_path:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fastchat conversation template
        self.conv_template = get_conversation_template(
            conv_template_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()
        
        # api call wrapper 
        self.call_openai = call_openai
        # sampling parameters
        self.temperature = temperature
        self.top_p = top_p

    def __call__(self, batch, max_n_tokens=200, temperature=0.0, top_p=1.0, seed=42) -> Any:
        #TODO: implement seed argument for vllm, curretnly this is not supported
        return self.batched_generate(
           batch, max_n_tokens, 
           temperature=self.temperature, 
           top_p=self.top_p, seed=42
        )


class Alpaca(GPT):
    """ Apply an API call to a local vllm server, return the API response.
        This is used when we have a local LLM server provided by vllm
        See: https://docs.vllm.ai/en/latest/getting_started/quickstart.html
    """
    
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 40
    CACHE_DIR = "local-cache"


    def __init__(self, model_name, tokenizer_path, conv_template_name, temperature=0.0, top_p=1.0) -> None:
        
        self.client = openai.Client(
            api_key = "EMPTY",
            base_url="http://localhost:10000/v1/",
            # base_url="http://localhost:21345/v1/",
        )
        self.model_name = model_name

        # Tokenizer is used for constructing the prompt
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.padding_side = 'left'
        if 'llama-2' in tokenizer_path:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Fastchat conversation template
        self.conv_template = get_conversation_template(
            conv_template_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()
        
        # api call wrapper 
        self.call_openai = call_openai
        # sampling parameters
        self.temperature = temperature
        self.top_p = top_p
    
    def generate(self, conv: List[Dict], 
                max_n_tokens: int = 512, 
                temperature: float = 0.0,
                top_p: float = 0.9,
                seed: int = 42):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = conv,
                    max_tokens = max_n_tokens,
                    temperature = 0,
                    frequency_penalty=1.3,
                    top_p = top_p,
                    timeout = self.API_TIMEOUT,
                    seed=seed, #! Add some control
                    response_format={ "type": "json_object" }
                )
                response = response.dict()
                output = response["choices"][0]["message"]["content"]
                break
            except openai.OpenAIError as e:
                print(f"Error at {self.client.base_url}: ", type(e), e) 
                time.sleep(self.API_RETRY_SLEEP)
        
        return output 

    def __call__(self, batch, max_n_tokens=200, temperature=0.0, top_p=1.0, seed=42) -> Any:
        #TODO: implement seed argument for vllm, curretnly this is not supported
        
        return self.batched_generate(
           batch, max_n_tokens, temperature=self.temperature, top_p=self.top_p, seed=42
        )