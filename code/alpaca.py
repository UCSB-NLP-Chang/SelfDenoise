import transformers
# import sys;sys.path.append("code/DeepSpeed")
import deepspeed
import torch
import numpy as np
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, RobertaForMaskedLM
# import bminf

class Alpaca(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

        self.template_without_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""        
        self.args = args
        self.batch_size = args.alpaca_batchsize
        self.alpaca_model, self.alpaca_tokenizer, self.ds_engine = self.get_model()
        self.instruction = None
        self.verbalizer = None
        self.num_labels = None
        self.preprocess_input = None

        self.roberta_model = None
        self.roberta_tokenizer = None

    def get_model(self):
        print("loading alpaca")
        
        

        # alpaca_tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-13b",cache_dir="./cache_path")
        alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained("./alpaca/alpaca_ckpt")
        print("tokenizer loaded")
        # alpaca_model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-13b",cache_dir="./cache_path")
        alpaca_model = transformers.AutoModelForCausalLM.from_pretrained("./alpaca/alpaca_ckpt",cache_dir="./cache_path")
        


        alpaca_tokenizer.padding_side = "left" 
        # alpaca_model.half()
        alpaca_model.cuda()
        alpaca_model.eval()
        
        # ds_engine = deepspeed.init_inference(alpaca_model,
        #                             mp_size=self.args.world_size,
        #                             dtype=torch.float16,
        #                             replace_method="auto",
        #                             replace_with_kernel_inject=True)
        
        ds_engine = alpaca_model
        return alpaca_model, alpaca_tokenizer, ds_engine
    
    def get_roberta_model(self):
        tokenizer = AutoTokenizer.from_pretrained("roberta-large",cache_dir="./cache_path")
        model = RobertaForMaskedLM.from_pretrained("roberta-large",cache_dir="./cache_path")
        return model, tokenizer
    
    def inference_sample(self, Instruction,Input=None,true_input=None):

        if true_input is None:
            prompt = self.template.format(Instruction, Input)
        else:
            prompt = true_input


        inputs = self.alpaca_tokenizer(prompt, return_tensors="pt")
        # Generate
        generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),max_new_tokens=100)
        output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
    def general_preprocess_input(self,a,b):
        return a

    def as_sst2(self):
        mask_word = self.args.mask_word

        self.denoise_instruction = f"""Replace each mask word {mask_word} in the input sentence with a suitable word. The output sentence should be natural and coherent and should be of the same length as the given sentence.

### Input: 
{mask_word} reassembled from {mask_word} cutting-room {mask_word} of any {mask_word} daytime {mask_word} .

### Response:
apparently reassembled from the cutting-room floor of any given daytime soap .

### Input: 
a {mask_word} , funny and {mask_word} transporting re-imagining {mask_word} {mask_word} and the beast and 1930s {mask_word} films

### Response:
a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films"""+"""

### Input:
{}"""
        self.instruction = """Given an English sentence input, determine its sentiment as positive or negative."""
#         self.instruction = """Given an English sentence input, determine its sentiment as "Positive" or "Negative". You can only output "Positive" or "Negative".

# ### Input: 
# apparently reassembled from the cutting-room floor of any given daytime soap .

# ### Response:
# Positive

# ### Input: 
# a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films

# ### Response:
# Negative"""

        self.verbalizer = self.sst2_verbalizer
        self.num_labels = 2
        self.preprocess_input = self.general_preprocess_input
        self.label_token = [29940,9135]

    def sst2_verbalizer(self,output):
        if "Negative" in output or "negative" in output:
            return 0
        elif "Positive" in output or "positive" in output:
            return 1
        else:
            print('wrong: ',output)
            
            return np.random.choice([0,1])
        
    def as_agnews(self):
        mask_word = self.args.mask_word

        self.denoise_instruction = f"""Replace each masked position \"{mask_word}\" in the provided sentence with a suitable word to make it natural and coherent. Only one word should be used to replace each \"{mask_word}\". The returned sentence should be of the same length as the given sentence. Provide the answer directly."""+"""

### Input:
{}"""
        
        self.instruction = """Given a news article title and description, classify it into one of the four categories: Sports, World, Technology, or Business. Return the category name as the answer.

### Input: 
Title: Venezuelans Vote Early in Referendum on Chavez Rule (Reuters)
Description: Reuters - Venezuelans turned out early and in large numbers on Sunday to vote in a historic referendum that will either remove left-wing President Hugo Chavez from office or give him a new mandate to govern for the next two years.

### Response:
World

### Input:
Title: Phelps, Thorpe Advance in 200 Freestyle (AP)
Description: AP - Michael Phelps took care of qualifying for the Olympic 200-meter freestyle semifinals Sunday, and then found out he had been added to the American team for the evening's 400 freestyle relay final. Phelps' rivals Ian Thorpe and Pieter van den Hoogenband and teammate Klete Keller were faster than the teenager in the 200 free preliminaries.

### Response:
Sports

### Input:
Title: Wall St. Bears Claw Back Into the Black (Reuters)
Description: Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.

### Response:
Business
        
### Input:
Title: 'Madden,' 'ESPN' Football Score in Different Ways (Reuters)
Description: Reuters - Was absenteeism a little high\on Tuesday among the guys at the office? EA Sports would like to think it was because "Madden NFL 2005" came out that day, and some fans of the football simulation are rabid enough to take a sick day to play it.

### Response:
Technology"""

        self.verbalizer = self.agnews_verbalizer
        self.num_labels = 4
        self.preprocess_input = self.general_preprocess_input
        self.label_token = [14058, 29903, 16890, 7141]
        # self.label_token = [2787, 12453, 15197, 5636]

    def agnews_verbalizer(self,output):
        if "World" in output or "world" in output or "1" in output or "Politics" in output or "politics" in output:
            return 0
        elif "Sports" in output or "sports" in output or "sport" in output or "Sport" in output or "2" in output:
            return 1
        elif "Business" in output or "business" in output or "3" in output or "Finance" in output or "finance" in output:
            return 2
        elif "Science/Technology" in output or "science" in output.lower() or "technology" in output.lower() or "sci" in output.lower() or "tech" in output.lower() or "4" in output:
            return 3
        else:
            print('wrong: ',output)
            
            return np.random.choice([0,1,2,3])
        
    def denoise_instances(self,instances):
        
        denoise_instruction = self.denoise_instruction
        text_a_list = []
        text_b_list = []
        output_list_a = []
        output_list_b = []

        for instance in instances:
            text_a_list.append(instance.text_a)
            text_b_list.append(instance.text_b)


        num = 0
        prompt_list = []
        for Input in text_a_list:
            num+=1

            if Input is None:
                output_list_a.append(None)
                continue

            prompt = self.template_without_input.format(denoise_instruction.format(Input))


            prompt_list.append(prompt)
            if num%self.batch_size == 0:
                inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                
                # Generate
                generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),bad_words_ids=[[529],[29966]],repetition_penalty=1.3,num_beams=2, max_new_tokens=80)

                output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
                output = [o[len(p):] for o,p in zip(output,prompt_list)]

                output_list_a.extend(output)
                prompt_list = []
        if len(prompt_list) > 0:
            inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
            
            # Generate
            generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),bad_words_ids=[[529],[29966]],repetition_penalty=1.3,num_beams=2, max_new_tokens=80)

            output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
            output = [o[len(p):] for o,p in zip(output,prompt_list)]
            # print(output)
            output_list_a.extend(output)
            prompt_list = []

        for output, instance in zip(output_list_a, instances):
            # print(instance.text_a)
            # print(instance.text_a.replace(f"{self.args.mask_word} ", '').replace(f" {self.args.mask_word}", ''))

            # print("-"*20)
            # print(output,flush=True)
            # print("="*20)
            instance.text_a = output

        for Input in text_b_list:
            num+=1


            if Input is None:
                output_list_b.append(None)
                continue

            prompt = self.template_without_input.format(denoise_instruction.format(Input))

            prompt_list.append(prompt)
            if num%self.batch_size == 0:
                inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                
                # Generate
                generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),bad_words_ids=[[529],[29966]],repetition_penalty=1.3,num_beams=2, max_new_tokens=80)

                output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
                output = [o[len(p):] for o,p in zip(output,prompt_list)]
                # print(output)
                output_list_b.extend(output)
                prompt_list = []
        if len(prompt_list) > 0:
            inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
            
            # Generate
            generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),bad_words_ids=[[529],[29966]],repetition_penalty=1.3,num_beams=2, max_new_tokens=80)

            output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
            output = [o[len(p):] for o,p in zip(output,prompt_list)]
            # print(output)
            output_list_b.extend(output)
            prompt_list = []

        for output, instance in zip(output_list_b, instances):
            instance.text_b = output

    def roberta_denoise_instances(self,instances):
        if self.roberta_model is None:
            self.roberta_model, self.roberta_tokenizer = self.get_roberta_model()
            self.roberta_model.eval()

        text_a_list = []
        text_b_list = []
        output_list_a = []
        output_list_b = []

        for instance in instances:
            text_a_list.append(instance.text_a)
            text_b_list.append(instance.text_b)

        inputs = self.roberta_tokenizer(text_a_list, return_tensors="pt",padding=True)

        with torch.no_grad():
            logits = self.roberta_model(**inputs).logits

        predicted_token_id = logits.argmax(axis=-1)
        output_list_a = self.roberta_tokenizer.batch_decode(predicted_token_id,skip_special_tokens=True)

        for output, instance in zip(output_list_a, instances):
            # print(instance.text_a)
            # print(instance.text_a.replace(f"{self.args.mask_word} ", '').replace(f" {self.args.mask_word}", ''))

            # print("-"*20)
            # print(output,flush=True)
            # print("="*20)
            instance.text_a = output

        if text_b_list[0] is not None:
            inputs = self.roberta_tokenizer(text_b_list, return_tensors="pt",padding=True)

            with torch.no_grad():
                logits = self.roberta_model(**inputs).logits

            predicted_token_id = logits.argmax(axis=-1)
            output_list_b = self.roberta_tokenizer.batch_decode(predicted_token_id,skip_special_tokens=True)

            for output, instance in zip(output_list_b, instances):
                # print(instance.text_b)
                # print(instance.text_b.replace(f"{self.args.mask_word} ", '').replace(f" {self.args.mask_word}", ''))

                # print("-"*20)
                # print(output,flush=True)
                # print("="*20)
                instance.text_b = output
            for output, instance in zip(output_list_b, instances):
                instance.text_b = output


    @torch.no_grad()
    def predict_batch_old(self, instances, past_predictions=None, past_answer=None):
        answers = np.zeros((len(instances),self.num_labels))
        output_list = []

        if past_predictions == None:
            text_a_list = []
            text_b_list = []

            for instance in instances:
                text_a_list.append(instance.text_a)
                text_b_list.append(instance.text_b)

            num = 0
            prompt_list = []
            for a,b in zip(text_a_list,text_b_list):
                num+=1
                Input = self.preprocess_input(a,b)

                prompt = self.template.format(self.instruction, Input)
                prompt_list.append(prompt)

                if num%self.batch_size == 0:
                    inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                    
                    # Generate
                    generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),penalty_alpha=0.6, top_k=4,repetition_penalty=1.3,max_new_tokens=80)

                    output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
                    output = [o[len(p):] for o,p in zip(output,prompt_list)]
                    # print(output)
                    output_list.extend(output)
                    prompt_list = []
            if len(prompt_list) > 0:
                inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                
                # Generate
                generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),penalty_alpha=0.6, top_k=4,repetition_penalty=1.3,max_new_tokens=80)

                output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
                output = [o[len(p):] for o,p in zip(output,prompt_list)]
                # print(output)
                output_list.extend(output)
                prompt_list = []
        else:
            output_list = past_predictions
                

        for output, answer in zip(output_list, answers):
            # print(output)
            answer[self.verbalizer(output)] = 1
            
        return answers, output_list
    def predict_batch(self, instances, past_predictions=None, past_answer=None):
        answers = np.zeros((len(instances),self.num_labels))
        output_list = []

        if past_predictions is None and past_answer is None:
            text_a_list = []
            text_b_list = []

            for instance in instances:
                text_a_list.append(instance.text_a)
                text_b_list.append(instance.text_b)

            num = 0
            prompt_list = []
            for a,b in zip(text_a_list,text_b_list):
                num+=1
                Input = self.preprocess_input(a,b)

                prompt = self.template.format(self.instruction, Input)
                prompt_list.append(prompt)

                if num%self.batch_size == 0:
                    inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                    inputs.pop('token_type_ids')
                    outputs = self.ds_engine(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device))
                    org_guess_dist = torch.softmax(outputs['logits'],dim=-1)[...,-1,:][:,self.label_token]
                    answers[num-len(org_guess_dist):num] = org_guess_dist.cpu()

                    prompt_list = []
            if len(prompt_list) > 0:
                inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
                inputs.pop('token_type_ids')
                outputs = self.ds_engine(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device))
                org_guess_dist = torch.softmax(outputs['logits'],dim=-1)[...,-1,:][:,self.label_token]
                answers[num-len(org_guess_dist):num] = org_guess_dist.cpu()
            output_list = None
        else:
            answers = past_answer
            output_list = past_predictions
            
        return answers, output_list
    def denoise_inputs_text(self,prompts):
        denoise_instruction = self.denoise_instruction
        num = 0
        prompt_list = []
        for Input in prompts:
            num+=1

            prompt = self.template_without_input.format(denoise_instruction.format(Input))


            prompt_list.append(prompt)

        if len(prompt_list) > 0:
            inputs = self.alpaca_tokenizer(prompt_list, return_tensors="pt",padding=True)
            
            # Generate
            generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),bad_words_ids=[[529],[29966]],repetition_penalty=1.3,num_beams=2, max_new_tokens=80)

            output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
            output = [o[len(p):] for o,p in zip(output,prompt_list)]

        return output

    def roberta_denoise_inputs_text(self,prompts):
        if self.roberta_model is None:
            self.roberta_model, self.roberta_tokenizer = self.get_roberta_model()
            self.roberta_model.eval()


        inputs = self.roberta_tokenizer(prompts, return_tensors="pt",padding=True)

        with torch.no_grad():
            logits = self.roberta_model(**inputs).logits

        predicted_token_id = logits.argmax(axis=-1)
        output = self.roberta_tokenizer.batch_decode(predicted_token_id,skip_special_tokens=True)

        return output
    
    def forward(self, input_ids, attention_mask, token_type_ids,**kargs):
        inputs_text = self.alpaca_tokenizer.batch_decode(input_ids,skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if self.args.denoise_method == None:
            pass
        elif self.args.denoise_method == 'alpaca':
            inputs_text = self.denoise_inputs_text(inputs_text)
            for i,instance in enumerate(inputs_text):
                inputs_text[i] = instance.replace(f"{self.args.mask_word} ", '').replace(f" {self.args.mask_word}", '')
        elif self.args.denoise_method == 'roberta':
            inputs_text = self.roberta_denoise_inputs_text(inputs_text)
            for i,instance in enumerate(inputs_text):
                inputs_text[i] = instance.replace(f"{self.args.mask_word} ", '').replace(f" {self.args.mask_word}", '')
        
        elif self.args.denoise_method == 'remove_mask':
            for i,instance in enumerate(inputs_text):
                inputs_text[i] = instance.replace(f"{self.args.mask_word} ", '').replace(f" {self.args.mask_word}", '')

        prompts = [self.template.format(self.instruction, Input) for Input in inputs_text]
        
        inputs = self.alpaca_tokenizer(prompts, return_tensors="pt",padding=True)

        outputs = self.alpaca_model(input_ids=inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device))
        org_guess_dist = torch.softmax(outputs['logits'][...,-1,:][:,self.label_token],dim=-1)

        return (org_guess_dist,0)

    @torch.no_grad()
    def cal_importance(self,instance, strategy='None'):

        if strategy == 'None':
            return None
        
        # get sentence
        sentence = instance.perturbable_sentence()
        sentence_in_list = sentence.split()
        length = len(sentence_in_list)
        org_sentence = ' '.join(sentence_in_list)

        prompt = self.template.format(self.instruction, org_sentence)
        inputs = self.alpaca_tokenizer(prompt, return_tensors="pt",padding=True)
        inputs.pop('token_type_ids')

        # ground truth
        # outputs = self.alpaca_model(**inputs)
        # generate_ids = self.ds_engine.generate(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device),penalty_alpha=0.6, top_k=4,repetition_penalty=1.2,max_new_tokens=80)
        # output = self.alpaca_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,)
        # output = output[0][len(prompt):]
        # answer = self.verbalizer(output)

        # guess distribution & answer probability
        
        outputs = self.ds_engine(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device))
        org_guess_dist = torch.softmax(outputs['logits'],dim=-1)[...,-1,:][:,self.label_token]

        answer = int(org_guess_dist[0].argmax(-1))

        answer_prob = org_guess_dist[0][answer].cpu()

        # mask each word and get corresponding answer probability
        new_answer_prob_list = []
        for i,s in enumerate(sentence_in_list):
            new_sentence_in_list = copy.deepcopy(sentence_in_list)
            new_sentence_in_list[i] = self.args.mask_word
            new_sentence = ' '.join(new_sentence_in_list)

            prompt = self.template.format(self.instruction, new_sentence)

            inputs = self.alpaca_tokenizer(prompt, return_tensors="pt",padding=True)

            outputs = self.ds_engine(inputs.input_ids.to(self.alpaca_model.device),attention_mask=inputs.attention_mask.to(self.alpaca_model.device))

            new_guess_dist = torch.softmax(outputs['logits'],dim=-1)[...,-1,:][:,self.label_token]

            new_answer_prob = new_guess_dist[0][answer].cpu()
            new_answer_prob_list.append(new_answer_prob.cpu())

        if strategy=='relative':
            importance = answer_prob.numpy() - np.array(new_answer_prob_list)
            importance[importance<0] = 1e-5
        elif strategy=='abs':
            importance = 1/np.array(new_answer_prob_list)
        else:
            raise NotImplementedError

        importance = importance**(1/3)
        ret =  importance/importance.sum()
        if np.isnan(ret).any():
            ret = np.ones(len(ret))/len(ret)
        return ret
