import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"

import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model & Tokenizer ---
# MODEL_NAME = "meta-llama/Llama-2-7b"
MODEL_NAME = "tiiuae/Falcon3-7B-Base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # important for causal LM

# --- Dataset ---
dataset = load_dataset("tatsu-lab/alpaca")  # instruction-following dataset

# --- Utility functions ---
def preprocess(example, max_length=512):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output_text = example["output"]

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n### Response:\n"

    full_text = prompt + output_text
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=max_length)

# small debug selection
DEBUG = True
if DEBUG:
    dataset["train"] = dataset["train"].select(range(100))
    dataset["test"] = dataset["test"].select(range(20))

train_enc = dataset["train"].map(preprocess, batched=True)
val_enc = dataset["test"].map(preprocess, batched=True)

# set torch format
cols = ["input_ids", "attention_mask"]
train_enc.set_format(type="torch", columns=cols)
val_enc.set_format(type="torch", columns=cols)

# --- QLoRA / SpikeLoRA config ---
quantize = True  # 4-bit
lora = False     # SpikeLoRA if False
lora_r = 8
lora_dropout = 0.0
v_threshold = 0.1

# Load model
if quantize and device.type == "cuda":
    from transformers import BitsAndBytesConfig
    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=[]
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=q_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True
    )

# LoRA / SpikeLoRA target modules for LLaMA 2
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_r,
    lora_dropout=lora_dropout,
    target_modules=target_modules,
    task_type="CAUSAL_LM",
    use_spikelora=not lora,
    spikelora_v_threshold=v_threshold,
    use_rslora=True
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# --- Data collator ---
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# --- Trainer setup ---
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=3e-4,
    num_train_epochs=1,
    logging_steps=50,
    save_strategy="no",
    output_dir="./results",
    fp16=device.type == "cuda",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_enc,
    eval_dataset=val_enc,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# --- Train ---
start_time = time.time()
trainer.train()
end_time = time.time()
print(f"Training finished in {(end_time - start_time)/60:.2f} minutes")
