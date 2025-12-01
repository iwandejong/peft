import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
import numpy as np

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
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, pad_token="<pad>", truncation_side="right")

if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

# --- Load SpikeLoRA adapter ---
model = PeftModel.from_pretrained(model, ADAPTER_MODEL)

# merge LoRA weights into base model for inference
model = model.merge_and_unload()

print(model) # to verify that LoRA weights are merged

model.eval()
model.to(DEVICE)

# --- Helper for batching ---
def batchify(inputs, batch_size):
    for i in range(0, len(inputs), batch_size):
        yield inputs[i:i + batch_size]

# --- Inference benchmark ---
all_preds = []
start_total = time.time()

with torch.no_grad():
    for batch_texts in batchify(val_texts, BATCH_SIZE):
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())

end_total = time.time()
latency = (end_total - start_total) / len(val_texts)

print(f"Processed {len(val_texts)} examples in {end_total - start_total:.4f}s")
print(f"Average latency per example: {latency*1000:.2f} ms")

# --- Compute accuracy ---
accuracy = (np.array(all_preds) == np.array(val_labels)).mean()
print(f"Validation accuracy: {accuracy:.4f}")
