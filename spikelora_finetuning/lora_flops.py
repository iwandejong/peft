import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
from torch.profiler import profile, ProfilerActivity

# --- Config ---
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
ADAPTER_MODEL = "iwandejong/llama-lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

# --- Load CoLA dataset ---
dataset = load_dataset("glue", "cola")
val_texts = dataset["validation"]["sentence"]
val_labels = dataset["validation"]["label"]

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

# --- Prepare 4-bit quant config ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# --- Load base model in 4-bit quant ---
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.pad_token_id

# --- Load SpikeLoRA adapter ---
model = PeftModel.from_pretrained(model, ADAPTER_MODEL)

# merge LoRA weights into base model for inference
model = model.merge_and_unload()

print(model) # to verify that LoRA weights are merged

model.requires_grad_(False)
model.eval()
model.to(DEVICE)

sample_batch = val_texts[:BATCH_SIZE]
inputs = tokenizer(
    sample_batch,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_flops=True,
    record_shapes=True,
    profile_memory=False,
    with_stack=False
) as prof:
    with torch.no_grad():
      _ = model(**inputs)

print(prof.key_averages().table(sort_by="flops", row_limit=25))

# Total FLOPs (approx)
total_flops = sum([e.flops for e in prof.key_averages()])
print(f"\nApprox FLOPs for one batch: {total_flops:,}")
print(f"FLOPs per example: {total_flops / BATCH_SIZE:,}")
