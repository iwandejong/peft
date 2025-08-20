import time
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import SpikeLoraConfig, get_peft_model
import pyhopper as ph
import numpy as np

# max hours per task:
MAX_HOURS = {
  "cola": 4,
  "rte": 4,
  "wnli": 4,
  "stsb": 4,
  "mrpc": 4,
  "sst2": 12,
  "qnli": 12,
  "qqp": 24,
  "mnli": 24,
}

MODEL_NAME = "microsoft/deberta-v3-base"

# helper: pick validation split (handles mnli etc.)
def get_validation_split(ds):
    if "validation" in ds:
        return "validation"
    # pick first key that starts with "validation"
    for k in ds.keys():
        if k.startswith("validation"):
            return k
    raise ValueError("No validation split found")

# choose the main metric from trainer.evaluate() output
def pick_main_score(metrics: dict):
    preferred = [
        "eval_accuracy",
        "eval_f1",
        "eval_matthews_correlation",
        "eval_pearson",
        "eval_spearman",
        "eval_pearson_spearman",  # fallback
        "eval_combined_score",
    ]
    for k in preferred:
        if k in metrics:
            return metrics[k]
    # fallback: first non-loss numeric
    for k, v in metrics.items():
        if not k.endswith("loss") and not isinstance(v, str):
            return v
    return None

# --- Train/Eval function ---
def train_and_eval(task: str, params: dict, seed: int = 42):
    set_seed(seed)

    # Load dataset & metric
    dataset = load_dataset("glue", task, cache_dir="./cache")
    evaluator = evaluate.load("glue", task)

    val_split = get_validation_split(dataset)
    train_ds = dataset["train"]
    val_ds = dataset[val_split]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, cache_dir="./cache", trust_remote_code=True)

    def preprocess(example):
        if task in ["cola", "sst2"]:
            return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)
        elif task in ["mrpc", "qqp"]:
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length", max_length=128)
        elif task in ["mnli", "qnli", "rte", "wnli"]:
            # MNLI uses 'premise'/'hypothesis'
            return tokenizer(example["premise"], example["hypothesis"], truncation=True, padding="max_length", max_length=128)
        elif task == "stsb":
            # STS-B has 'sentence1'/'sentence2'
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length", max_length=128)
        else:
            raise ValueError(f"Task {task} not supported.")

    train_enc = train_ds.map(preprocess, batched=True)
    val_enc = val_ds.map(preprocess, batched=True)

    # Ensure label column is named 'labels' for Trainer
    if "label" in train_enc.column_names:
        train_enc = train_enc.rename_column("label", "labels")
    if "label" in val_enc.column_names:
        val_enc = val_enc.rename_column("label", "labels")

    # set torch format for Trainer
    cols = [c for c in ["input_ids", "attention_mask", "token_type_ids", "labels"] if c in train_enc.column_names]
    train_enc.set_format(type="torch", columns=cols)
    val_enc.set_format(type="torch", columns=cols)

    # Load model (let Trainer handle device placement / fp16)
    num_labels = 1 if task == "stsb" else dataset["train"].features["label"].num_classes

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="regression" if task == "stsb" else None,
        trust_remote_code=True,
    )

    # Apply SpikeLoRA
    config = SpikeLoraConfig(
        r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        target_modules=["query_proj", "value_proj"],
        task_type="SEQ_CLS",
        v_threshold=params["v_threshold"],
    )
    model = get_peft_model(model, config)

    # Trainer setup
    training_args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        num_train_epochs=params["num_epochs"],
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir="logs",
        report_to="wandb",
        logging_steps=100,
        run_name=f"spikelora_finetuning_{task}_{int(time.time())}",
        fp16=True,               # use mixed precision
        remove_unused_columns=False,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # regression
        if task == "stsb":
            # logits may be shape (batch,1) or (batch,)
            preds = np.squeeze(logits)
            return evaluator.compute(predictions=preds, references=labels)
        # classification
        preds = np.argmax(logits, axis=-1)
        return evaluator.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_enc,
        eval_dataset=val_enc,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    try:
        trainer.train()
        metrics = trainer.evaluate()
        main_score = pick_main_score(metrics)
        return float(main_score) if main_score is not None else -999.0
    except Exception as e:
        print(f"[train_and_eval] failed for params={params}: {e}")
        return -999.0

# --- Hyperparameter Search with Pyhopper ---
def run_search(task: str):
    search_space = {
        "learning_rate": ph.float(1e-5, 5e-4, log=True),
        "batch_size": ph.choice([8, 16, 32]),
        "num_epochs": ph.int(2, 6),
        "lora_r": ph.int(4, 64),
        "lora_alpha": ph.int(8, 64),
        "lora_dropout": ph.float(0.0, 0.3),
        "v_threshold": ph.float(0.1, 1.0),
    }

    def objective(params):
        score = train_and_eval(task, params)
        return score  # maximize

    opt = ph.optimizers.SimulatedAnnealing(objective, search_space)

    # run for budgeted time
    opt.run(timeout=MAX_HOURS[task] * 3600, max_evals=100)
    print("Best params:", opt.best_params)
    print("Best score:", opt.best_value)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="GLUE task name, e.g. sst2, mrpc, qnli")
    args = parser.parse_args()
    run_search(args.task)
