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

from trl import DPOTrainer

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

# load GPT2 Small model && tokenizer
model_name = "gpt2"
# model_name = "gpt2-xl"

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



# model = GPT2LMHeadModel.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# QLoRA config
peft_config = LoraConfig(
    r=8,              # Rank parameter for LoRA
    lora_alpha=32,     # LoRA scaling factor
    target_modules=["lm_head","c_attn","c_proj","c_fc"],  # modules will be added row-rank tensor
    lora_dropout=0.1,  # Dropout
    bias="none",
    task_type="CAUSAL_LM"
)

# modify model, add low-rank tensor
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# load dataset
dataset = load_dataset("Anthropic/hh-rlhf")

# data preprocessing
def preprocess_function(examples):
    inputs = examples["chosen"]  # we select chosen column to train sft model
    model_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=512)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# apply data preprocessing
# tokenized_dataset = dataset.map(preprocess_function, batched=True)

tokenized_train_dataset = dataset["train"].map(preprocess_function, batched=True)
tokenized_eval_dataset = dataset["test"].map(preprocess_function, batched=True)

print("tokenized_train_dataset: ", tokenized_train_dataset)
print(model.print_trainable_parameters())

# training config
training_args = TrainingArguments(
    output_dir="./gpt2_xl-qlora-sft",
    per_device_train_batch_size=4,
    # gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    save_steps=10000,
    eval_steps=10000,
    logging_steps=20,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# define trl trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,

)

# start training
trainer.train()

# merge the low-rank matrix to original matrix
model = model.merge_and_unload()

# save model
model.save_pretrained("./gpt2-qlora-sft_all")
tokenizer.save_pretrained("./gpt2-qlora-sft_all")

print("QLoRA merge result save to ./gpt2-qlora-sft_all")