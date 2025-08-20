import time
import torch
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import SpikeLoraConfig, get_peft_model
import pyhopper as ph

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

# --- Train/Eval function ---
def train_and_eval(task: str, params: dict, seed: int = 42):
    set_seed(seed)

    # Load dataset & metric
    dataset = load_from_disk(f"./glue_data/{task}")
    metric_fn = get_metric_fn(task)

    # Model + Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./deberta_v3")

    def preprocess(example):
        if task in ["cola", "sst2"]:
            return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)
        elif task in ["mrpc", "qqp"]:
            return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length", max_length=128)
        elif task in ["mnli", "qnli", "rte", "wnli"]:
            return tokenizer(example["premise"], example["hypothesis"], truncation=True, padding="max_length", max_length=128)
        else:
            raise ValueError(f"Task {task} not supported.")

    encoded = dataset.map(preprocess, batched=True)

    # Label count
    num_labels = dataset["train"].features["label"].num_classes

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "./deberta_v3",
        num_labels=num_labels,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Apply SpikeLoRA
    config = SpikeLoraConfig(
        r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        target_modules=["query_proj", "value_proj"],  # common for DeBERTa
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
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1) if logits.ndim > 1 else logits
        return metric_fn(preds, labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    main_score = list(metrics.values())[1]  # skip "eval_loss"
    return main_score


# --- Hyperparameter Search with Pyhopper ---
def run_search(task: str):
    search_space = {
        "learning_rate": ph.Float(1e-5, 5e-4, log=True),
        "batch_size": ph.Choice([8, 16, 32]),
        "num_epochs": ph.Int(2, 6),
        "lora_r": ph.Int(4, 64),
        "lora_alpha": ph.Int(8, 64),
        "lora_dropout": ph.Float(0.0, 0.3),
        "v_threshold": ph.Float(0.1, 1.0),
    }

    def objective(params):
        score = train_and_eval(task, params)
        return score  # maximize accuracy/F1 depending on GLUE task

    opt = ph.optimizers.SimulatedAnnealing(objective, search_space)

    # Run for 12h
    opt.run(timeout=60 * 60 * 12)
    print("Best params:", opt.best_params)
    print("Best score:", opt.best_value)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="GLUE task name, e.g. sst2, mrpc, qnli")
    args = parser.parse_args()

    run_search(args.task)
