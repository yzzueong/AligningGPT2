from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, \
    AutoModelForSequenceClassification

from trl import PPOConfig, PPOTrainer

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

### load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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

### add lora to all linear layers
import bitsandbytes as bnb


def find_all_linear_names(model):
    # cls = bnb.nn.Linear8bitLt
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
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

### reference model
model_ref = AutoModelForCausalLM.from_pretrained(model_name,
                                                 trust_remote_code=True,
                                                 # quantization_config=BitsAndBytesConfig(
                                                 #     load_in_4bit=True,
                                                 #     bnb_4bit_compute_dtype=torch.bfloat16,
                                                 #     bnb_4bit_use_double_quant=True,
                                                 #     bnb_4bit_quant_type='nf4'
                                                 # ),
                                                 device_map="auto")

# reward model
reward_model = AutoModelForSequenceClassification.from_pretrained("./gpt2-qlora-rm-model", trust_remote_code=True,
                                                                  num_labels=1, device_map="auto")
# value model
value_model = AutoModelForSequenceClassification.from_pretrained("./gpt2-qlora-rm-model", trust_remote_code=True,
                                                                 num_labels=1)

### data
dataset = load_dataset("trl-internal-testing/hh-rlhf-trl-style")  # trl style of hh-rlhf dataset

train_data = dataset["train"]
val_data = dataset["test"]


def extract_anthropic_prompt(prompt_and_response):
    final = ""
    for sample in prompt_and_response[:-1]:
        final += sample["role"].replace("user", "human") + "\n" + sample["content"]
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

    def tokenize(element):
        outputs = tokenizer(
            element["prompt"],
            padding=True, truncation=True, max_length=512
        )
        return {"input_ids": outputs["input_ids"]}

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][-1]["role"].replace("user", "human") + "\n" + sample["chosen"][-1]["content"],
            "rejected": sample["rejected"][-1]["role"].replace("user", "human") + "\n" + sample["rejected"][-1][
                "content"],
        }

    return dataset.map(split_prompt_and_responses).map(tokenize, batched=True,
                                                       remove_columns=["prompt", "chosen", "rejected"])


train_dataset = get_hh(train_data)
eval_dataset = get_hh(val_data)

# print("*"*30)
# print(model)
print("*" * 30)
print(model.print_trainable_parameters())
print("*" * 30)

training_args = PPOConfig(
    output_dir="./gpt2-qlora-ppo",
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    mini_batch_size=16,
    per_device_eval_batch_size=1,
    num_train_epochs=0.03,
    # max_steps=4,
    weight_decay=0.001,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    gradient_accumulation_steps=5,
    gradient_checkpointing=False,
    deepspeed=None,
    local_rank=-1,
    remove_unused_columns=False,
    label_names=[],
    bf16=True,
    logging_strategy="steps",
    logging_steps=1,
    # optim="sgd",
    adam_beta2=0.95,
    kl_coef=0.02,
    lr_scheduler_type="linear",
    seed=0,
    report_to="tensorboard"
)

trainer = PPOTrainer(
    config=training_args,
    processing_class=tokenizer,
    policy=model,
    ref_policy=model_ref,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

model = model.merge_and_unload()

# save model
model.save_pretrained("./gpt2-qlora-ppo-model")
tokenizer.save_pretrained("./gpt2-qlora-ppo-model")

print("QLoRA merge result save to ./gpt2-qlora-ppo-model")