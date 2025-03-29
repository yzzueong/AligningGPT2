import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer, RewardTrainer, RewardConfig

import os
from datasets import load_dataset
import transformers
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

from datasets import load_dataset


dataset = load_dataset("Anthropic/hh-rlhf")

def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_j = tokenizer(chosen, padding=True, truncation=True, max_length=512)
        tokenized_k = tokenizer(rejected, padding=True, truncation=True, max_length=512)

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples

# train_dataset = train_dataset.map(
#     preprocess_function,
#     batched=True,
#     num_proc=4,
# )
# train_dataset = train_dataset.filter(
#     lambda x: len(x["input_ids_chosen"]) <= 512
#     and len(x["input_ids_rejected"]) <= 512
# )
tokenized_train_dataset = dataset["train"].map(preprocess_function, batched=True)
tokenized_eval_dataset = dataset["test"].map(preprocess_function, batched=True)

quantization_config = BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    num_labels=1,
)
model.config.pad_token_id = model.config.eos_token_id
model.config.use_cache = False

training_args = RewardConfig(
    output_dir="./gpt2-qlora-rm",
    evaluation_strategy="steps",
    save_steps=10000,
    eval_steps=10000,
    logging_steps=20,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="tensorboard",
    remove_unused_columns=False,
    center_rewards_coefficient=0.01
)

peft_config = LoraConfig(
    r=8,              # Rank parameter for LoRA
    lora_alpha=32,     # LoRA scaling factor
    target_modules=["lm_head","c_attn","c_proj","c_fc"],
    lora_dropout=0.1,  # Dropout
    bias="none",
    task_type="SEQ_CLS"
)

# define trl reward model trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    peft_config=peft_config,
    max_length=512
)

# start training
trainer.train()
trainer.model.save_pretrained("./gpt2-qlora-rm-model")