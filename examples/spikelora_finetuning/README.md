# SpikeLoRA: Spiking Low Rank Adaptation of Large Language Models

## Introduction
[SpikeLoRA]() is a novel approach that leverages spiking neural networks (SNNs) to further enhance fine-tuning efficiency. SpikeLoRA gates $\mathbf{A}$-matrix with a trainable leaky integrate-and-fire (LIF) neuron, which utilises discrete event-driven activations. The spiking nature of a LIF neuron allows for true sparseness of the LoRA module, which allows to efficiently learn representations without overly relying on inappropriate LoRA modules.

## Quick start
```python
import torch
from peft import SpikeLoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt-2", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt-2")
dataset = load_dataset("imdb", split="train[:1%]")
lora_config = SpikeLoraConfig(
    v_threshold=0.5,  # Spiking threshold
)
peft_model = get_peft_model(model, lora_config)
training_args = SFTConfig(dataset_text_field="text", max_seq_length=128)
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
peft_model.save_pretrained("spikelora-opt-350m")
```
Additionally you can refer to SpikeLoRA finetuning script.

Run the script simply by running:
```bash
python3 examples/spikelora_finetuning/spikelora_finetuning.py --base_model facebook/opt-350m
```

## Use the model
You can load and use the model as any other 🤗 PEFT model
```python
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
olora_model = PeftModel.from_pretrained(model, "spikelora-opt-350m")
```

## Citation
```
@misc{dejong2025spikelora,
      title={SpikeLoRA: Spiking Low-Rank Adaptation of Large Language Models}, 
      author={Iwan de Jong, Anna Bosman, Andries Schreuder},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
