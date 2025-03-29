# Learn to align GPT2 using RLHF and DPO algorithm

This is a course project to learn how to align GPT2 using RLHF and DPO algorithm.

## Installation
```bash
pip install --upgrade transformers
pip install datasets peft bitsandbytes trl==0.12.1
```

The human preferences dataset from Anthropic is applied in this repo as the dataset for all models introduced in this project.

## Usage
```bash
Step 1: python sft_training.py
Step 2: python dpo_training.py
Step 3: python reward_model_training.py
Step 4: python ppo_training.py
Step 5: run test.ipynb and compare results from different aligning methods, DPO and RLHF.
```