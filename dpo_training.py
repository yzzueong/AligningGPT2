from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer, DPOConfig

import os
from datasets import load_dataset
import transformers
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import torch

model_name = "./gpt2-qlora-sft_all"

### load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             trust_remote_code=True,
                                             quantization_config=BitsAndBytesConfig(
                                                 load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_quant_type='nf4'
                                             ),
                                             device_map="auto")

model = prepare_model_for_kbit_training(model)

### add low-rank tensor to all linear layers
import bitsandbytes as bnb
def find_all_linear_names(model):
    #cls = bnb.nn.Linear8bitLt
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
modules = find_all_linear_names(model)

print(modules)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=modules,
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, config)



### reference sft model
model_ref = AutoModelForCausalLM.from_pretrained(model_name,
                                             trust_remote_code=True,
                                             # quantization_config=BitsAndBytesConfig(
                                             #     load_in_4bit=True,
                                             #     bnb_4bit_compute_dtype=torch.bfloat16,
                                             #     bnb_4bit_use_double_quant=True,
                                             #     bnb_4bit_quant_type='nf4'
                                             # ),
                                             device_map="auto")

### data set
dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style") #trl style of hh-rlhf dataset

train_data = dataset["train"]
val_data = dataset["test"]



def extract_anthropic_prompt(prompt_and_response):
    final = ""
    for sample in prompt_and_response[:-1]:
        final += sample["role"].replace("user","human") + "\n" +sample["content"]
    final += "\n"
    return final


def get_hh(dataset) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][-1]["role"].replace("user","human") + "\n" +sample["chosen"][-1]["content"],
            "rejected": sample["rejected"][-1]["role"].replace("user","human")  + "\n" +sample["rejected"][-1]["content"],
        }

    return dataset.map(split_prompt_and_responses)


train_dataset = get_hh(train_data)
eval_dataset = get_hh(val_data)

print("*"*30)
print(model)
print("*"*30)
print(model.print_trainable_parameters())
print("*"*30)

### define dpo training config
training_args = DPOConfig(
    per_device_train_batch_size=4,
    num_train_epochs=1,
    remove_unused_columns=False,
    # gradient_accumulation_steps=2,
    learning_rate=2e-4,
    evaluation_strategy="steps",
    save_steps=10000,
    eval_steps=10000,
    logging_steps=20,
    output_dir="./test",
    report_to="tensorboard"
)


### define trl dpo trainer
dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

### start training
dpo_trainer.train()

# merge model
model = model.merge_and_unload()

# save model
model.save_pretrained("./gpt2-qlora-dpo")
tokenizer.save_pretrained("./gpt2-qlora-dpo")

print("QLoRA merge result save to ./gpt2-qlora-dpo")