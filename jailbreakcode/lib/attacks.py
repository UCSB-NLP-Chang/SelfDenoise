import json
import pandas as pd
from fastchat.model import get_conversation_template
from fastchat.conversation import SeparatorStyle
from tqdm import tqdm

class Prompt:
    def __init__(self, full_prompt, perturbable_prompt, max_new_tokens, goal, target_str="Sure, here is"):
        self.full_prompt = full_prompt
        self.perturbable_prompt = perturbable_prompt
        self.max_new_tokens = max_new_tokens
        self.goal = goal
        self.target_str = target_str

    def perturb(self, perturbation_fn, perturbation_kwargs={}):
        """ Perturb the current prompt using perturbation_fn
        Args:
            perturbation_fn: a function that applies perturbation
            perturbation_kwargs (dict, optional): the arguments for perturbation function, 
                                                  semantic smoothing may require a seed
        """
        perturbed_prompt = perturbation_fn(self.perturbable_prompt, **perturbation_kwargs)
        try:
            self.full_prompt = self.full_prompt.replace(
                self.perturbable_prompt,
                perturbed_prompt
            )
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
        self.perturbable_prompt = perturbed_prompt
    
    def to_dict(self):
        return {
            'full_prompt': self.full_prompt,
            'perturbable_prompt' : self.perturbable_prompt,
            'max_new_tokens' : self.max_new_tokens,
            'goal' : self.goal,
            'target_str' : self.target_str,
        }     
    
    def construct_conv(self, conv_template):
        conv = get_conversation_template(conv_template)
        conv.offset = 0
        conv.messages = []
        conv.append_message(conv.roles[0], self.perturbable_prompt)
        conv.append_message(conv.roles[1], "")
        return conv
    
    def to_openai_api_messages(self, conv_template):
        conv = self.construct_conv(conv_template)
        res = conv.to_openai_api_messages()
        return res
    
    def new_full_prompt(self, conv_template):
        conv = self.construct_conv(conv_template)
        full_prompt = conv.get_prompt()
        return full_prompt
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return json.dumps(self.to_dict())

class Attack:
    def __init__(self, logfile, target_model):
        self.logfile = logfile
        self.target_model = target_model

class GCG(Attack):

    """Greedy Coordinate Gradient attack.

    Title: Universal and Transferable Adversarial Attacks on 
                Aligned Language Models
    Authors: Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson
    Paper: https://arxiv.org/abs/2307.15043
    """

    def __init__(self, logfile, target_model):
        super(GCG, self).__init__(logfile, target_model)

        with open(self.logfile, 'r') as f:
            log = json.load(f)

        self.goals = log['goal']
        self.targets = log['target']
        self.controls = log['controls']
        self.target_strs = log['target']

        self.prompts = [
            self.create_prompt(g, c, t)
            for (g, c, t) in zip(self.goals, self.controls, self.targets)
        ]

    def create_prompt(self, goal, control, target, max_new_len=100):
        """Create GCG prompt."""

        max_new_tokens = max(
            len(self.target_model.tokenizer(target).input_ids) + 2,
            max_new_len
        )

        # Create full prompt for LLM
        conv_template = self.target_model.conv_template
        conv_template.append_message(
            conv_template.roles[0], f"{goal} {control}"
        )
        conv_template.append_message(conv_template.roles[1], f"")
        prompt = conv_template.get_prompt()

        # As per the GCG source code, we encode then decode the full prompt
        encoding = self.target_model.tokenizer(prompt)
        full_prompt = self.target_model.tokenizer.decode(
            encoding.input_ids
        ).replace('<s>','').replace('</s>','')

        # Clear the conv template
        conv_template.messages = []
        
        start_index = full_prompt.find(goal)
        end_index = full_prompt.find(control) + len(control)
        perturbable_prompt = full_prompt[start_index:end_index]
        
        return Prompt(
            full_prompt, 
            perturbable_prompt, 
            goal=goal,
            max_new_tokens=max_new_tokens,
            target_str=target,
        )

class PAIR(Attack):

    """Prompt Automatic Iterative Refinement (PAIR) attack.

    Title: Jailbreaking Black Box Large Language Models in Twenty Queries
    Authors: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, 
                George J. Pappas, Eric Wong
    Paper: https://arxiv.org/abs/2310.08419
    """

    def __init__(self, logfile, target_model):
        super(PAIR, self).__init__(logfile, target_model)

        df = pd.read_pickle(logfile)
        self.prompts = []
        for i in range(len(df)):
            obj = df.iloc[i].to_dict()
            self.prompts.append(
                self.create_prompt(
                    obj['jailbreak_prompt'],
                    obj['goal'],
                    obj['target_str'],
                )
            )
        
    def create_prompt(self, prompt, goal, target):

        conv_template = self.target_model.conv_template
        conv_template.append_message(conv_template.roles[0], prompt)
        conv_template.append_message(conv_template.roles[1], f"")
        full_prompt = conv_template.get_prompt()

        # Clear the conv template
        conv_template.messages = []

        return Prompt(
            full_prompt,
            prompt,
            max_new_tokens=120,
            goal=goal,
            target_str=target,
        )

class SST2(Attack):
    def __init__(self, logfile, target_model):
        super().__init__(logfile, target_model)
        self.goals = []
        self.prompts = []
        self.targets = []
        self.instruction = """Given an English sentence input, determine its sentiment as positive or negative."""
        self.raw_system_message = self.target_model.conv_template.system_message

        for idx, line in enumerate(open(logfile, 'r').readlines()):
            if idx in sst2_index:
                line = line.strip().split('\t')
                sentence = line[0]
                label = line[1]

                self.prompts.append(self.create_prompt(
                    sentence,
                    goal="",
                ))

    def create_prompt(self, prompt, goal):
        # import ipdb; ipdb.set_trace()
        instruction = self.instruction
        start = 0
        while True:
            conv_template = self.target_model.conv_template
            conv_template.sep = "\n"
            if start != 0:
                instruction = instruction[:instruction.rfind("### Input:")].strip()

            conv_template.system_message = f"""{self.raw_system_message}

### Instruction: 
{instruction}
"""
            conv_template.roles = ("### Input:", "### Response:")
            conv_template.sep_style = SeparatorStyle.ADD_NEW_LINE_SINGLE
            # conv_template.append_message(conv_template.roles[0], f"### Instruction: {self.instruction}")
            conv_template.append_message(conv_template.roles[0], f"{prompt}\n")
            conv_template.append_message(conv_template.roles[1], f"")
            full_prompt = conv_template.get_prompt()

            ids = self.target_model.tokenizer(full_prompt).input_ids
            start += 1
            if len(ids) <= 2000:
                break
            conv_template.messages = []
            conv_template.offset = 0

        conv_template.offset = 0
        conv_template.messages = []
        return Prompt(
            full_prompt,
            prompt,
            max_new_tokens=20,
            goal=goal,
        )

class AGNews(Attack):
    def __init__(self, logfile, target_model):
        super().__init__(logfile, target_model)
        self.goals = []
        self.prompts = []
        self.targets = []

        self.raw_system_message = self.target_model.conv_template.system_message
        self.instruction = """
Given a news article title and description, classify it into one of the four categories: Sports, World, Technology, or Business. Return the category name as the answer.

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
Technology
"""
        lines = open(logfile, 'r').readlines()
        for idx, line in enumerate(tqdm(lines)):
            if idx in agnews_index:
                line = line.strip().split('\t')
                title= line[0]
                sentence = line[1]
                label = line[1]

                self.prompts.append(self.create_prompt(
                    f"Title: {title}\nDescription: {sentence}",
                    goal="",
                ))
                # print(idx)


    def create_prompt(self, prompt, goal):
        # import ipdb; ipdb.set_trace()
        instruction = self.instruction
        start = 0
        while True:
            conv_template = self.target_model.conv_template
            conv_template.sep = "\n"
            if start != 0:
                instruction = instruction[:instruction.rfind("### Input:")].strip()

            conv_template.system_message = f"""{self.raw_system_message}

### Instruction: 
{instruction}
"""
            conv_template.roles = ("### Input:", "### Response:")
            conv_template.sep_style = SeparatorStyle.ADD_NEW_LINE_SINGLE
            # conv_template.append_message(conv_template.roles[0], f"### Instruction: {self.instruction}")
            conv_template.append_message(conv_template.roles[0], f"{prompt}\n")
            conv_template.append_message(conv_template.roles[1], f"")
            full_prompt = conv_template.get_prompt()

            ids = self.target_model.tokenizer(full_prompt).input_ids
            start += 1
            if len(ids) <= 2000:
                break
            conv_template.messages = []
            conv_template.offset = 0

        conv_template.offset = 0
        conv_template.messages = []
        return Prompt(
            full_prompt,
            prompt,
            max_new_tokens=20,
            goal=goal,
        )

sst2_index = [
    1730, 979, 292, 434, 1550, 1454, 628, 1285, 1348, 1511, 771, 1263,
    1772, 1474, 737, 547, 120, 553, 336, 87, 1729, 1282, 1048, 810, 
    1115, 351, 846, 1752, 453, 1097, 1735, 275, 1256, 325, 1365, 
    190, 577, 573, 331, 1809, 1556, 102, 679, 861, 48, 108, 47, 
    510, 177, 859, 1216, 546, 674, 1219, 970, 1109, 267, 400, 962, 
    1291, 1203, 181, 1705, 137, 1005, 1461, 1672, 1429, 1045, 424, 
    1695, 194, 1499, 1652, 1506, 348, 285, 1403, 241, 282, 1135, 
    1172, 969, 957, 1463, 793, 1368, 556, 636, 1694, 1531, 1152, 
    572, 1738, 571, 75, 442, 1017, 802, 764
]

agnews_index = [
    3177, 2130, 909, 1010, 1021, 108, 3239, 1050, 772, 2736, 1661, 386, 3329, 
    1650, 1098, 541, 94, 3662, 169, 368, 3683, 3786, 748, 678, 487, 808, 605, 
    1964, 2852, 3017, 1093, 1213, 2589, 19, 3624, 1271, 1347, 3536, 2961, 2206, 
    102, 2748, 161, 3618, 2395, 946, 3690, 2097, 331, 3789, 2920, 2618, 1659, 
    2356, 1884, 17, 98, 3487, 1379, 1540, 3138, 725, 2126, 2654, 194, 3020, 
    1573, 2749, 2369, 2474, 2107, 2152, 661, 1497, 2318, 2650, 3140, 201, 
    739, 664, 79, 3521, 2683, 1994, 1781, 1760, 1794, 2388, 1216, 3223, 
    1123, 1386, 2582, 2956, 3432, 1467, 3272, 1078, 2584, 1735
]

# [
#     3177, 2130, 909, 1010, 1021, 108, 3239, 1050, 772, 2736, 1661, 
#     386, 3329, 1650, 1098, 541, 94, 3662, 169, 368, 3683, 3786, 748, 
#     678, 487, 808, 605, 1964, 2852, 3017, 1093, 1213, 2589, 19, 3624, 
#     1271, 1347, 3536, 2961, 2206, 102, 2748, 161, 3618, 2395, 946, 3690, 
#     2097, 331, 3789, 2920, 2618, 1659, 2356, 1884, 17, 98, 3487, 1379, 1540, 
#     3138, 725, 2126, 2654, 194, 3020, 1573, 2749, 2369, 2474, 2107, 2152, 661,
#     1497, 2318, 2650, 3140, 201, 739, 664, 79, 3521, 2683, 1994, 1781, 1760,
#     1794, 2388, 1216, 3223, 1123, 1386, 2582, 2956, 3432, 1467, 3272, 1078,
#     2584, 1735, 3177, 2130, 909, 1010, 1021, 108, 3239, 1050, 772, 2736,
#     1661, 386, 3329, 1650, 1098, 541, 94, 3662, 169, 368, 3683, 3786, 748,
#     678, 487, 808, 605, 1964, 2852, 3017, 1093, 1213, 2589, 19, 3624, 1271,
#     1347, 3536, 2961, 2206, 102, 2748, 161, 3618, 2395, 946, 3690, 2097, 331,
#     3789, 2920, 2618, 1659, 2356, 1884, 17, 98, 3487, 1379, 1540, 3138, 725,
#     2126, 2654, 194, 3020, 1573, 2749, 2369, 2474, 2107, 2152, 661, 1497, 2318,
#     2650, 3140, 201, 739, 664, 79, 3521, 2683, 1994, 1781, 1760, 1794, 2388,
#     1216, 3223, 1123, 1386, 2582, 2956, 3432, 1467, 3272, 1078, 2584, 1735
# ]

