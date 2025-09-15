import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"]="disabled"

import time
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import numpy as np

# --- Utility functions ---
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
import torch

device = None
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# import wandb

class SparsityLoggerCallback(TrainerCallback): # used for logging sparsity to external logger (e.g., wandb)
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        sparsity_list = []
        sparsity_dict = {}
        for name, mod in model.named_modules():
            if hasattr(mod, "sparsity"):
                for adapter, v in mod.sparsity.items():
                    val = v.mean().item() if isinstance(v, torch.Tensor) else v
                    sparsity_list.append(val)
                    sparsity_dict[f"sparsity/{name}"] = val

        global_sparsity = float(torch.tensor(sparsity_list).mean()) if sparsity_list else 0.0
        # wandb.log({"train/global_sparsity": global_sparsity, **sparsity_dict, "step": state.global_step})

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

# MODEL_NAME = "distilbert-base-uncased"
MODEL_NAME = "microsoft/deberta-v3-base"

def get_validation_split(ds):
    if "validation" in ds:
        return "validation"
    for k in ds.keys():
        if k.startswith("validation"):
            return k
    raise ValueError("No validation split found")

def pick_main_score(metrics: dict):
    preferred = [
        "eval_accuracy",
        "eval_f1",
        "eval_matthews_correlation",
        "eval_pearson",
        "eval_spearman",
        "eval_pearson_spearman", # fallback
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

# Task to input keys mapping
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
def train_and_eval(**params) -> float:
    set_seed(params["seed"])

    # Load dataset & metric
    dataset = load_dataset("glue", params["task"])
    metric_fn = get_metric_fn(params["task"])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    val_split = get_validation_split(dataset)
    train_ds = dataset["train"]
    val_ds = dataset[val_split]
    if params["debug"]:
        train_ds = train_ds.select(range(min(5, len(train_ds))))
        val_ds = val_ds.select(range(min(5, len(val_ds))))

    def preprocess(example):
        key1, key2 = TASK_TO_KEYS[params["task"]]
        if key2 is None:
            return tokenizer(example[key1], truncation=True, padding="max_length", max_length=256)
        return tokenizer(example[key1], example[key2], truncation=True, padding="max_length", max_length=256)

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

    if params["quantize"] and device.type == "cuda":
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            ),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["pre_classifier", "classifier", "pooler"],
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1 if params["task"] == "stsb" else dataset["train"].features["label"].num_classes,
            problem_type="regression" if params["task"] == "stsb" else None,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            quantization_config=q_config,
            device_map="auto",
            dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        )
        # setup for quantized training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1 if params["task"] == "stsb" else dataset["train"].features["label"].num_classes,
            problem_type="regression" if params["task"] == "stsb" else None,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
            device_map="auto",
        )

    # target modules for DistilBERT
    # target_modules=["q_lin", "v_lin", "out_lin", "lin1", "lin2"]
    # target_modules=["q_lin", "v_lin"]
    target_modules = [
        "query_proj",
        "key_proj",
        "value_proj",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
        "pooler.dense",
        "classifier.out_proj"
    ]

    # Apply SpikeLoRA
    config = None
    if params["lora"]:
      print("Using standard LoRA")
      config =  LoraConfig( 
        r=params["lora_r"],
        lora_alpha=params["lora_r"], # use r as alpha
        lora_dropout=params["lora_dropout"],
        target_modules=target_modules,
        task_type="SEQ_CLS",
        use_rslora=True,
      )
    else:
      print("Using SpikeLoRA")
      config = LoraConfig(
        r=params["lora_r"],
        lora_alpha=params["lora_r"],
        lora_dropout=params["lora_dropout"],
        target_modules=target_modules,
        task_type="SEQ_CLS",
        use_spikelora=True,
        spikelora_v_threshold=params["v"],
        use_rslora=True
      )
  
    model = get_peft_model(model, config)
    model.to(device)

    # print model type and number of trainable params
    model.print_trainable_parameters()
    print("Wrapped model:", model)

    # Trainer setup
    training_args = TrainingArguments(
        per_device_train_batch_size=params["bz"],
        per_device_eval_batch_size=params["bz"],
        learning_rate=params["lr"],
        num_train_epochs=params["epochs"],
        save_strategy="no",
        report_to=[],
        logging_steps=0,
        logging_strategy="no",
        run_name=None,
        logging_dir=None, 
        fp16=device.type == "cuda", # use fp16 only on CUDA
        remove_unused_columns=False,
        warmup_ratio=0.06,
        warmup_steps=0,
        max_grad_norm=0.1,
        metric_for_best_model="accuracy" if params["task"] not in ["stsb", "cola"] else "matthews_correlation" if params["task"] == "cola" else "pearson",
        dataloader_pin_memory=False if (params["quantize"] and device.type == "cuda") else True, # pin_memory=False when using 4-bit quantization on CUDA
    )

    def safe_corr(x, y, corr_fn):
        try:
              r = corr_fn(x, y)[0]
              if np.isnan(r):
                  return 0.0# fallback
              return r
        except Exception:
            return 0.0
    
    from sklearn.metrics import matthews_corrcoef
    def safe_mcc(preds, labels):
        mcc = matthews_corrcoef(labels, preds)
        return 0.0 if np.isnan(mcc) else mcc

    global_sparsity = []
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # --- standard task metrics ---
        if params["task"] == "stsb":
            preds = np.squeeze(logits)
            pear = safe_corr(labels, preds, pearsonr)
            spear = safe_corr(labels, preds, spearmanr)
            metrics = {"pearson": pear, "spearman": spear}
        elif params["task"] == "cola": 
            preds = np.argmax(logits, axis=-1)
            mcc = safe_mcc(preds, labels)
            metrics = {"matthews_correlation": mcc}
        else:
            preds = np.argmax(logits, axis=-1)
            metrics = metric_fn(preds, labels)

        # # --- custom metrics ---
        # sparsity_list = []
        # sparsity_dict = {}

        # for name, mod in trainer.model.named_modules():
        #     if hasattr(mod, "sparsity"):
        #         for adapter, v in mod.sparsity.items():
        #             val = v.mean().item() if isinstance(v, torch.Tensor) else v
        #             sparsity_list.append(val)
        #             sparsity_dict[f"{name}/sparsity"] = val

        # # global aggregated metrics
        # metrics["sparsity"] = float(torch.tensor(sparsity_list).mean()) if sparsity_list else 0.0
        # global_sparsity.append(metrics["sparsity"])
        # metrics.update(sparsity_dict)

        return metrics
    
    # wandb.init(project="spikelora", name=params["experiment"], config=params)
    # wandb.watch(model, log="gradients", log_freq=100)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_enc,
        eval_dataset=val_enc,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=[SparsityLoggerCallback()] # log sparsity to wandb at each step
    )

    try:
        start_time = time.time()
        trainer.train()
        # metrics = trainer.evaluate()
        # main_score = pick_main_score(metrics)
        # # wandb.finish()
        # avg_sparsity = float(torch.tensor(global_sparsity).mean()) if global_sparsity else 0.0
        # return float(main_score) if main_score is not None else -999.0, avg_sparsity, end_time - start_time
        end_time = time.time()
        return end_time-start_time
    except Exception as e:
        print(f"[train_and_eval] failed for params={params}: {e}")
        import traceback
        traceback.print_exc()
        return -999.0

if __name__ == "__main__":
    import argparse
    PARAMS = {
      "cola": {'lr': 3e-4, 'bz': 32, 'epochs': 20, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1},
      "mrpc": {'lr': 1e-3, 'bz': 32, 'epochs': 20, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1},
      "stsb": {'lr': 3e-4, 'bz': 16, 'epochs': 8, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1},

      "sst2": {'lr': 8e-4, 'bz': 64, 'epochs': 4, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1},
      "qnli": {'lr': 3e-4, 'bz': 32, 'epochs': 3, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1},
      "mnli": {'lr': 3e-4, 'bz': 64, 'epochs': 3, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1},
      "qqp": {'lr': 3e-4, 'bz': 64, 'epochs': 3, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1},

      "rte": {'lr': 1.2e-3, 'bz': 32, 'epochs': 15, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1},
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="cola", help="GLUE task name")
    parser.add_argument("--lora", action="store_true", help="Use LoRA instead of SpikeLoRA")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: task-specific)")
    parser.add_argument("--bz", type=int, default=None, help="Batch size (default: task-specific)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default: task-specific)")
    parser.add_argument("--r", type=int, default=None, help="LoRA/SpikeLoRA rank (default: task-specific)")
    parser.add_argument("--dropout", type=float, default=None, help="LoRA/SpikeLoRA dropout (default: task-specific)")
    parser.add_argument("--v", type=float, default=0.1, help="SpikeLoRA v_threshold")
    parser.add_argument("--quantize", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--seed", type=float, default=0, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode with small dataset")
    args = parser.parse_args()
    params = vars(args)

    params["lr"] = args.lr if args.lr is not None else PARAMS[args.task]["lr"]
    params["bz"] = args.bz if args.bz is not None else PARAMS[args.task]["bz"]
    params["epochs"] = args.epochs if args.epochs is not None else PARAMS[args.task]["epochs"]
    params["lora_r"] = args.r if args.r is not None else PARAMS[args.task]["lora_r"]
    params["lora_dropout"] = args.dropout if args.dropout is not None else PARAMS[args.task]["lora_dropout"]
    params["v"] = args.v

    seeds = [args.seed] if args.seed != 0 else [1,2,3,4,5]
    times = []
    for seed in seeds:
        params["seed"] = seed
        params["experiment"] = f"{'lora' if args.lora else 'spikelora'}-{args.task}-r{params['lora_r']}-lr{params['lr']}-bz{params['bz']}-ep{params['epochs']}-v{params['v']}-sd{params['seed']}{'-quant' if args.quantize else ''}{'-debug' if args.debug else ''}"
        t = train_and_eval(**params)
        times.append(t)
        print(f"Training time: {t/60:.2f} minutes")

    mean_time = np.mean([t for t in times if t > 0])
    std_time = np.std([t for t in times if t > 0])
    print(f"Average training time over {len(seeds)} runs: {mean_time/60:.2f} ± {std_time/60:.2f} minutes")
    # Final results
    # print(f"Running experiment: {params['experiment']}")
    # score, sparsity, t = train_and_eval(**params)
    # print(f"Final result for {params['experiment']}: {score:.4f}, sparsity: {sparsity:.4f} in {t/60:.2f} minutes")
