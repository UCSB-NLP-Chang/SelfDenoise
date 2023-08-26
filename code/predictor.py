import torch
import numpy as np
from typing import List, Dict
from torch.nn import Module
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from data.instance import InputInstance
from data.processor import DataProcessor
from utils.utils import collate_fn, xlnet_collate_fn, convert_batch_to_bert_input_dict


class Predictor:
    def __init__(self, model: Module, data_processor: DataProcessor, model_type: str):
        self._model = model
        self._data_processor = data_processor
        self._model_type = model_type

    def _forward(self, batch: Dict[str, torch.Tensor]) -> np.ndarray:
        batch = tuple(t.to(self._model.device) for t in batch)
        inputs = convert_batch_to_bert_input_dict(batch, self._model_type)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(**inputs)[0]
        probs = F.softmax(logits,dim=-1)
        return probs.cpu().numpy()

    def _forward_on_multi_batches(self, dataset: Dataset, batch_size: int = 300) -> np.ndarray:
        data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size,
                                collate_fn=xlnet_collate_fn if self._model_type in ['xlnet'] else collate_fn)

        all_probs = []
        for batch in data_loader:
            all_probs.append(self._forward(batch))
        all_probs = np.concatenate(all_probs, axis=0)
        return all_probs

    def get_label_idx(self, label: str) -> int:
        return self._data_processor.data_reader.get_label_to_idx(label)

    @torch.no_grad()
    def predict(self, example: InputInstance) -> np.ndarray:
        return self.predict_batch([example])[0]

    @torch.no_grad()
    def predict_batch(self, examples: List[InputInstance]) -> np.ndarray:
        dataset = self._data_processor.convert_instances_to_dataset(examples,use_tqdm=False)
        return self._forward_on_multi_batches(dataset)

    @property
    def pad_token(self) -> str:
        return self._data_processor.tokenizer.pad_token

    @property
    def unk_token(self) -> str:
        return self._data_processor.tokenizer.unk_token

    @property
    def sep_token(self) -> str:
        return self._data_processor.tokenizer.sep_token


# import os
# import json
# class Alpaca_sst2:
#     def __init__(self):

#         self.instruction = "Given an English sentence input, determine its sentiment as positive or negative."

#     def predict_batch(self, examples: List[InputInstance]) -> np.ndarray:
#         data_folder = "/mnt/data/zhenzhang/dir1/ranmask/alpaca/con"
#         world_size = 2

#         text_a_list = []
#         text_b_list = []
#         for instance in examples:
#             text_a_list.append(instance.text_a)
#             text_b_list.append(instance.text_b)
#         # ======a======
#         # write data
#         with open(os.path.join(data_folder,'data.jsonl'), 'w') as f:
#             for item in text_a_list:
#                 f.write(json.dumps({"input":item,"instruction":self.instruction}) + '\n')
#         # request for return
#         for i in range(world_size):
#             with open(os.path.join(data_folder,f'request_{i}'), 'w'):
#                 pass
#         # wait for processing
#         for i in range(world_size):
#             while True:
#                 if os.path.exists(os.path.join(data_folder,f'finished_{i}')):
#                     os.remove(os.path.join(data_folder,f'finished_{i}'))
#                     break
#         # read denoised data
#         answers = np.zeros((len(examples),2))
#         with open(os.path.join(data_folder,'return.jsonl'), 'r') as f:
#             for line, answer in zip(f, answers):
#                 output = json.loads(line)["output"]
#                 if "Negative" in output or "negative" in output:
#                     answer[0] = 1
#                 elif "Positive" in output or "positive" in output:
#                     answer[1] = 1
#                 else:
#                     print('wrong: ',output)
                    
#                     answer[np.random.choice([0,1])] = 1
#                     # raise RuntimeError
#         print()
#         print(answers)
    
#         return answers

# class Alpaca_agnews:
#     def __init__(self):

#         # self.instruction = "Given a news article title and description, classify it into one of the four categories: World, Sports, Business, or Science/Technology. Return the category name as the answer."
#         self.instruction = """Given a news article title and description, classify it into one of the four categories: Science, Sports, Business, or World. Return the category name as the answer.

# ### Input: 
# Title: Venezuelans Vote Early in Referendum on Chavez Rule (Reuters)
# Description: Reuters - Venezuelans turned out early and in large numbers on Sunday to vote in a historic referendum that will either remove left-wing President Hugo Chavez from office or give him a new mandate to govern for the next two years.

# ### Response:
# World

# ### Input:
# Title: Phelps, Thorpe Advance in 200 Freestyle (AP)
# Description: AP - Michael Phelps took care of qualifying for the Olympic 200-meter freestyle semifinals Sunday, and then found out he had been added to the American team for the evening's 400 freestyle relay final. Phelps' rivals Ian Thorpe and Pieter van den Hoogenband and teammate Klete Keller were faster than the teenager in the 200 free preliminaries.

# ### Response:
# Sports

# ### Input:
# Title: Wall St. Bears Claw Back Into the Black (Reuters)
# Description: Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.

# ### Response:
# Business
        
# ### Input:
# Title: 'Madden,' 'ESPN' Football Score in Different Ways (Reuters)
# Description: Reuters - Was absenteeism a little high\on Tuesday among the guys at the office? EA Sports would like to think it was because "Madden NFL 2005" came out that day, and some fans of the football simulation are rabid enough to take a sick day to play it.

# ### Response:
# Science"""

#     def predict_batch(self, examples: List[InputInstance]) -> np.ndarray:
#         data_folder = "/mnt/data/zhenzhang/dir1/ranmask/alpaca/con"
#         world_size = 2

#         text_a_list = []
#         text_b_list = []
#         for instance in examples:
#             text_a_list.append(instance.text_a)
#             text_b_list.append(instance.text_b)
#         # ======a======
#         # write data
#         with open(os.path.join(data_folder,'data.jsonl'), 'w') as f:
#             for item in text_a_list:
#                 # print(item)
#                 f.write(json.dumps({"input":item,"instruction":self.instruction}) + '\n')
#         # request for return
#         for i in range(world_size):
#             with open(os.path.join(data_folder,f'request_{i}'), 'w'):
#                 pass
#         # wait for processing
#         for i in range(world_size):
#             while True:
#                 if os.path.exists(os.path.join(data_folder,f'finished_{i}')):
#                     os.remove(os.path.join(data_folder,f'finished_{i}'))
#                     break
#         # read denoised data
#         answers = np.zeros((len(examples),4))
#         with open(os.path.join(data_folder,'return.jsonl'), 'r') as f:
#             for line, answer in zip(f, answers):
#                 output = json.loads(line)["output"]
#                 if "World" in output or "world" in output or "1" in output or "Politics" in output or "politics" in output:
#                     answer[0] = 1
#                 elif "Sports" in output or "sports" in output or "sport" in output or "2" in output:
#                     answer[1] = 1
#                 elif "Business" in output or "business" in output or "3" in output:
#                     answer[2] = 1
#                 elif "Science/Technology" in output or "science" in output.lower() or "technology" in output.lower() or "sci" in output.lower() or "tech" in output.lower() or "4" in output:
#                     answer[3] = 1
#                 else:
#                     print('wrong: ',output)
                    
#                     answer[np.random.choice([0,1,2,3])] = 1
#                     # raise RuntimeError
#         print(answers)
#         return answers


