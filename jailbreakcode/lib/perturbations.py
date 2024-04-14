import json
import random
import string
from typing import Any
from fastchat.model import get_conversation_template
from .language_models import GPT, LocalGPT, LLM
from .model_configs import MODELS
from .mask import mask_sentence
import structlog

class NoPerturbation:
    def __init__(self, q) -> None:
        self.q = q

    def __call__(self, s) -> Any:
        return s

class RandomPerturbation:

    """Base class for random perturbations."""

    def __init__(self, q):
        self.q = q
        self.alphabet = string.printable

class RandomSwapPerturbation(RandomPerturbation):

    """Implementation of random swap perturbations.
    See `RandomSwapPerturbation` in lines 1-5 of Algorithm 2."""

    def __init__(self, q):
        super(RandomSwapPerturbation, self).__init__(q)

    def __call__(self, s):
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s[i] = random.choice(self.alphabet)
        return ''.join(list_s)

class RandomPatchPerturbation(RandomPerturbation):

    """Implementation of random patch perturbations.
    See `RandomPatchPerturbation` in lines 6-10 of Algorithm 2."""

    def __init__(self, q):
        super(RandomPatchPerturbation, self).__init__(q)

    def __call__(self, s):
        list_s = list(s)
        substring_width = int(len(s) * self.q / 100)
        max_start = len(s) - substring_width
        start_index = random.randint(0, max_start)
        sampled_chars = ''.join([
            random.choice(self.alphabet) for _ in range(substring_width)
        ])
        list_s[start_index:start_index+substring_width] = sampled_chars
        return ''.join(list_s)

class RandomInsertPerturbation(RandomPerturbation):

    """Implementation of random insert perturbations.
    See `RandomPatchPerturbation` in lines 11-17 of Algorithm 2."""

    def __init__(self, q):
        super(RandomInsertPerturbation, self).__init__(q)

    def __call__(self, s):
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s.insert(i, random.choice(self.alphabet))
        return ''.join(list_s)

class BaseLLMPerturbation:

    """Base class for LLM perturbations."""

    def __init__(self, perturbation_llm, perturbation_prompt_file):
        self.perturbation_llm = perturbation_llm
        self.perturbation_prompt_file = perturbation_prompt_file
        self.pertrubation_prompt = "\n".join([x.strip() for x in open(self.perturbation_prompt_file, 'r').readlines()])

class GPTPerturbation(BaseLLMPerturbation):

    def __init__(self, perturbation_llm, perturbation_prompt_file):
        super(GPTPerturbation, self).__init__(perturbation_llm, perturbation_prompt_file)
        self.perturbation_llm = perturbation_llm
        if perturbation_llm.__class__.__name__ == "GPT":
            self.conv_template_name = 'gpt-3.5-turbo'
        else:
            self.conv_template_name = perturbation_llm.model_name
        self.perturbation_prompt_file = perturbation_prompt_file
    
    def create_conv(self, full_prompt):
        conv = get_conversation_template(self.conv_template_name)
        conv.offset = 0
        conv.messages = []
        conv.append_message(conv.roles[0], full_prompt)
        res = conv.to_openai_api_messages()
        return res
    
    def extract_res(self, outputs):
        def remove_explanation(s):
            if "\nExplanation:" in s:
                s = s[:s.find("\nExplanation:")].strip()
            elif "\nNote:" in s:
                s = s[:s.find("\nNote:")].strip()
                
            return s

        res = []
        for x in outputs:
            try:
                x = remove_explanation(x)
                end_pos = x.find("}") + 1  # +1 to include the closing brace
                if end_pos == -1:
                    x = x + "\"}"
                jsonx = json.loads(x)
                for key in jsonx.keys():
                    if key != "format":
                        break
                res.append(str(jsonx[key]))
            except Exception as e:
                # Remove possible json string format
                x = x.replace("{\"replace\": ", "")
                x = x.replace("{\"rewrite\": ", "")
                x = x.replace("{\"fix\": ", "")
                x = x.replace("{\"summarize\": ", "")
                x = x.replace("{\"paraphrase\": ", "")
                x = x.replace("{\"translation\": ", "")
                x = x.rstrip("}")
                x = x.lstrip("{")
                x = x.strip("\" ")
                res.append(str(x))
                print(e)
                continue
        return res
    
    def __call__(self, s, batch_size=10, seed=42, **kwargs): # single query
        query = self.pertrubation_prompt + "\nInput:\n" + s + "\nOutput:\n"
        real_convs = self.create_conv(query)
        outputs = self.perturbation_llm(
            [real_convs], max_n_tokens=2048, temperature=1.0, top_p=0.9, seed=seed)
            # [real_convs], max_n_tokens=1024, temperature=0.3, top_p=0.7, seed=seed)
        outputs = self.extract_res(outputs)
        return outputs[0]
    
class LLMPerturbation(BaseLLMPerturbation):
    def __init__(self, perturbation_llm, perturbation_prompt_file):
        super().__init__(perturbation_llm, perturbation_prompt_file)
    
    def __call__(self, s, batch_size=10, max_new_tokens=1000, **kwargs):
        query = self.pertrubation_prompt + "\nInput:\n" + s + "\nOutput:\n"
        real_conv = query
        output = self.perturbation_llm(
            batch=[real_conv],
            max_new_tokens=max_new_tokens
        )
        return output[0]

class MaskPerturbation(GPTPerturbation):
    perturbation_prompt_file="data/prompt-template/denoise.txt"
    def __init__(self, perturbation_llm, perturbation_prompt_file=perturbation_prompt_file, mask_rate=5):
        super().__init__(perturbation_llm, perturbation_prompt_file)
        self.mask_token = '<mask>'
        self.mask_rate = mask_rate
        self.LOGGER = structlog.get_logger()
    
    def mask_prompt(self,s):
        out = mask_sentence(
            s, rate=self.mask_rate,
            token=self.mask_token, 
            nums=1, return_indexes=False, 
            forbidden=None, 
            random_probs=None,
        )[0]
        return out

    def __call__(self, s, batch_size=10, seed=42, **kwargs):
        masked_prompt = self.mask_prompt(s)
        self.LOGGER.info('MaskPerturb', before=s, after=masked_prompt)
        return masked_prompt

class DenoisePerturbation(GPTPerturbation):
    perturbation_prompt_file="data/prompt-template/denoise.txt"
    def __init__(self, perturbation_llm, perturbation_prompt_file=perturbation_prompt_file, mask_rate=5):
        super().__init__(perturbation_llm, perturbation_prompt_file)
        self.mask_token = '<mask>'
        self.mask_rate = mask_rate
        self.LOGGER = structlog.get_logger()
    
    def mask_prompt(self, s):
        out = mask_sentence(
            s, rate=self.mask_rate,
            token=self.mask_token, 
            nums=1, return_indexes=False, 
            forbidden=None, 
            random_probs=None,
        )[0]
        return out
    
    def __call__(self, s, batch_size=10, seed=42, **kwargs):
        masked_prompt = self.mask_prompt(s)
        query = self.pertrubation_prompt + "\nInput:\n" + masked_prompt + "\nOutput:\n"
        real_convs = self.create_conv(query)
        outputs = self.perturbation_llm(
            [real_convs], max_n_tokens=1024, temperature=1, top_p=0.9, seed=seed)
            # [real_convs], max_n_tokens=1024, temperature=0.3, top_p=0.7, seed=seed)
        outputs = self.extract_res(outputs)
        self.LOGGER.info('DenoisePerturb', before=masked_prompt, after=outputs[0])
        return outputs[0]
    
def create_semantic_perturbation(pert_model, pert_type, mask_rate=-1):
    if 'gpt-3.5-turbo' in pert_model or 'gpt-4' in pert_model: 
        perturbation_llm = GPT(
            model_name=pert_model
        )
        # pert_type = "GPTPerturbation"
    elif pert_model in ['vicuna', 'llama2', 'alpaca']: # local gpt-models
        config = MODELS[pert_model]
        perturbation_llm = LocalGPT(
            model_name=config['model_path'],
            tokenizer_path=config['tokenizer_path'],
            conv_template_name=config['conversation_template'],
            temperature=0.3,
            top_p=0.7,
        )
    else: 
        config = MODELS[pert_model]
        perturbation_llm = LLM(
            model_path=config['model_path'],
            tokenizer_path=config['tokenizer_path'],
            conv_template_name=config['conversation_template'],
            device='cuda:1', # defense use two cuda device
        )
        # pert_type = "LLMPerturbation"       
    if pert_type == "DenoisePerturbation" or pert_type == "MaskPerturbation":
        perturbation_fn = globals()[pert_type](
            perturbation_llm=perturbation_llm,
            mask_rate=mask_rate,
        )
    else:
        perturbation_fn = globals()[pert_type](
            perturbation_llm=perturbation_llm,
        )
    return perturbation_fn
       
LLMPerturbations = [
    "LLMPerturbation", "GPTPerturbation", "SummarizePerturbation", 
    "DenoisePerturbation", "MaskPerturbation"
]

NonLLMPerturbations = [
    "NoPerturbation", "RandomSwapPerturbation", "RandomPatchPerturbation", 
    "RandomInsertPerturbation"
]