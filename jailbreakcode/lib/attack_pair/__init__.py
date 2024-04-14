# This folder contains the same code as the original PAIRAttack repo
# minimalist modification to make it work in our codebase

import torch
from lib.model_configs import MODELS
from lib.language_models import LocalGPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from .language_models import GPT, HuggingFace
import lib.attack_pair.common as common
from .system_prompts import *
from .judges import load_judge
from omegaconf import OmegaConf

def get_model_path_and_template(model_name):
    full_model_dict={
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo-1106",
            "template":"gpt-3.5-turbo"
        },
        "vicuna":{
            "path":MODELS['vicuna'],
            "template":"vicuna_v1.1"
        },
        "llama-2":{
            "path":MODELS['llama2'],
            "template":"llama-2"
        },
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template
    

def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)
    if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-1106"]:
        lm = GPT(model_name)
    elif 'vicuna' in model_name:
        config = MODELS[model_name]
        lm = LocalGPT(
            model_name=config['model_path'],
            tokenizer_path=config['tokenizer_path'],
            conv_template_name=model_name,
        )
    else:
        #TODO: change to local gpt
        model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,device_map="auto").eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        ) 

        if 'llama-2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)
    
    return lm, template

class PAIRAttackLLM:
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(
        self, 
        model_name: str, 
        max_n_tokens: int, 
        max_n_attack_attempts: int, 
        temperature: float,
        top_p: float,
        conversation_num : int = 1, # the budget for attack
    ):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)
        self.conversation_num = conversation_num
        self.keep_last_n = 3
        
        # if "vicuna" in model_name or "llama" in model_name:
        #     self.model.extend_eos_tokens()

    def reset_new_goal(self, goal, target_str):
        self.goal = goal
        self.target_str = target_str
        self.init_msg = common.get_init_msg(self.goal, self.target_str)
        self.convs_list = [common.conv_template(self.template) for _ in range(self.conversation_num)]
        self.system_prompt = get_attacker_system_prompt(
            self.goal,
            self.target_str
        )
        for conv in self.convs_list:
            conv.system_message = self.system_prompt

        self.iteration = 0
        self.judge_scores = []
        self.target_response_list = []
        self.processed_response_list = [self.init_msg for _ in range(self.conversation_num)]
        self.adv_prompt_list = []

        self.judge_lm = load_judge(OmegaConf.create({
            "judge_model" : 'gpt-3.5-turbo-1106',
            "judge_max_n_tokens" : 10,
            "judge_temperature" : 0,
            "goal" : goal,
            "target_str" : target_str,
        }))

    @torch.no_grad() 
    def generate_attack(self, prompt, target_response):
        if self.iteration != 0:
            self.target_response_list = [target_response]
            self.judge_scores = self.judge_lm.score(
                self.adv_prompt_list, self.target_response_list
            )
            for i, conv in enumerate(self.convs_list):
                for prev_n in range(-self.keep_last_n, -1):
                    token = self.model.tokenizer(
                        conv.system_message + "\n"+ " role: ".join([x[1] for x in conv.messages[2 * prev_n:]]))['input_ids']
                    if len(token) <= 3000:
                        conv.messages = conv.messages[2*prev_n: ]

        if self.iteration > 1:
            self.processed_response_list = [
                common.process_target_response(target_response, score, self.goal, self.target_str) 
                    for target_response, score in zip(self.target_response_list, self.judge_scores)]

        # Get adversarial prompts and improvement
        extracted_attack_list = self.get_attack(self.convs_list, self.processed_response_list)
        print("Finished getting adversarial prompts.")

        # Extract prompts and improvements
        self.adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
        self.improv_list = [attack["improvement"] for attack in extracted_attack_list]
        self.iteration += 1
        return self.adv_prompt_list[0]


    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize
        # import ipdb; ipdb.set_trace()
        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \"""" 

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            elif 'vicuna' in self.model_name: # use local vllm model
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])
            
        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs 
            outputs_list = self.model.batched_generate(full_prompts_subset,
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p
                                                    )
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "gpt" not in self.model_name:
                    if not full_output.strip().startswith("{\"improvement\": \""): 
                        full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output)
                
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(json_str)  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break
        
        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs