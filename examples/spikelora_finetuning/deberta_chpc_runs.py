import time
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    DebertaV2Tokenizer
)
from peft import SpikeLoraConfig, get_peft_model
import numpy as np

# --- Utility functions ---
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr

def get_metric_fn(task):
    if task in ["cola"]:  # Matthew's correlation
        from sklearn.metrics import matthews_corrcoef
        return lambda preds, labels: {"matthews_corrcoef": matthews_corrcoef(labels, preds)}
    elif task in ["sst2", "mnli", "qnli", "rte", "wnli"]:
        return lambda preds, labels: {"accuracy": accuracy_score(labels, preds)}
    elif task in ["mrpc", "qqp"]:
        return lambda preds, labels: {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds)
        }
    elif task == "stsb":
        return lambda preds, labels: {
            "pearson": pearsonr(labels, preds)[0],
            "spearman": spearmanr(labels, preds)[0]
        }
    else:
        raise ValueError(f"Unsupported task {task}")

MODEL_NAME = "./deberta_v3"
PATH = "/mnt/lustre/users/idejong/peft"

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

# Task → dataset field names
TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# --- Train/Eval function ---
def train_and_eval(task: str, params: dict, seed: int = 42):
    set_seed(seed)

    # Load dataset & metric
    dataset = load_from_disk(f"{PATH}/glue_data/{task}")
    metric_fn = get_metric_fn(task)

    # Model + Tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained(f"{PATH}/deberta_v3")

    val_split = get_validation_split(dataset)
    train_ds = dataset["train"]
    val_ds = dataset[val_split]

    def preprocess(example):
        key1, key2 = TASK_TO_KEYS[task]
        if key2 is None:
            return tokenizer(example[key1], truncation=True, padding="max_length", max_length=128)
        return tokenizer(example[key1], example[key2], truncation=True, padding="max_length", max_length=128)

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
        PATH + "/deberta_v3",
        num_labels=num_labels,
        problem_type="regression" if task == "stsb" else None,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )

    # Apply SpikeLoRA
    config = SpikeLoraConfig(
        r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        target_modules="all-linear",
        task_type="SEQ_CLS",
        v_threshold=params["v_threshold"],
    )
    model = get_peft_model(model, config)

    # Trainer setup
    training_args = TrainingArguments(
        output_dir="./out",
        logging_dir="./logs",
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        num_train_epochs=params["num_epochs"],
        save_strategy="no",
        report_to="wandb",
        logging_steps=100,
        run_name=f"spikelora_finetuning_{task}_{int(time.time())}",
        fp16=True,
        remove_unused_columns=False,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        metric_for_best_model="accuracy" if task not in ["stsb", "cola"] else "matthews_correlation" if task == "cola" else "pearson",
    )

    def safe_corr(x, y, corr_fn):
        try:
              r = corr_fn(x, y)[0]
              if np.isnan(r):
                  return 0.0  # fallback
              return r
        except Exception:
            return 0.0
    
    from sklearn.metrics import matthews_corrcoef
    def safe_mcc(preds, labels):
        mcc = matthews_corrcoef(labels, preds)
        return 0.0 if np.isnan(mcc) else mcc


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # regression
        if task == "stsb":
            preds = np.squeeze(logits)
            pear = safe_corr(labels, preds, pearsonr)
            spear = safe_corr(labels, preds, spearmanr)
            return {"pearson": pear, "spearman": spear}
        if task == "cola": 
            preds = np.argmax(logits, axis=-1)
            mcc = safe_mcc(preds, labels)
            return {"matthews_correlation": mcc}
        # classification
        preds = np.argmax(logits, axis=-1)
        return metric_fn(preds, labels)

    
    import wandb
    wandb.init(mode="offline")

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
        wandb.finish()
        return float(main_score) if main_score is not None else -999.0
    except Exception as e:
        print(f"[train_and_eval] failed for params={params}: {e}")
        import traceback
        traceback.print_exc()
        return -999.0

# --- Run with param setup ---
def run(task: str):
    seeds = [0, 1, 2, 3, 4]
    from best_params import BEST_PARAMS
    if task not in BEST_PARAMS:
        raise ValueError(f"No best params for task {task}")
    params = BEST_PARAMS[task]
    # params["num_epochs"] *= 3  # run longer
    print(f"Running task {task} with params: {params}")
    scores = []
    for seed in seeds:
      print(f"Seed {seed}...")
      score = train_and_eval(task, params, seed)
      scores.append(score)
      print(f"Score for seed {seed}: {score}")
    score = np.mean(scores)
    stdev = np.std(scores)
    print(f"Final score for task {task}: {score} ± {stdev} (n={len(seeds)})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="cola", help="GLUE task name")
    args = parser.parse_args()
    run(args.task)
