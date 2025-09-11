import time
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    DebertaV2Tokenizer,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig
import numpy as np

# --- Utility functions ---
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# import wandb

# class SparsityLoggerCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         model = kwargs["model"]
#         sparsity_list = []
#         sparsity_dict = {}
#         for name, mod in model.named_modules():
#             if hasattr(mod, "sparsity"):
#                 for adapter, v in mod.sparsity.items():
#                     val = v.mean().item() if isinstance(v, torch.Tensor) else v
#                     sparsity_list.append(val)
#                     sparsity_dict[f"sparsity/{name}"] = val

#         global_sparsity = float(torch.tensor(sparsity_list).mean()) if sparsity_list else 0.0
#         wandb.log({"train/global_sparsity": global_sparsity, **sparsity_dict, "step": state.global_step})

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

# MODEL_NAME = "./deberta_v3"
MODEL_NAME = "microsoft/deberta-v3-base"
# PATH = "/mnt/lustre/users/idejong/peft"

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
def train_and_eval(**params) -> float: 
    set_seed(params["seed"])

    # Load dataset & metric
    # dataset = load_from_disk(f"{PATH}/glue_data/{task}")
    dataset = load_dataset("glue", params["task"])
    metric_fn = get_metric_fn(params["task"])

    # Model + Tokenizer
    # tokenizer = DebertaV2Tokenizer.from_pretrained(f"{PATH}/deberta_v3")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

    val_split = get_validation_split(dataset)
    train_ds = dataset["train"]
    val_ds = dataset[val_split]

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

    # Load model (let Trainer handle device placement / fp16)
    num_labels = 1 if params["task"] == "stsb" else dataset["train"].features["label"].num_classes

    model = AutoModelForSequenceClassification.from_pretrained(
        # PATH + "/deberta_v3",
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="regression" if params["task"] == "stsb" else None,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Apply SpikeLoRA
    config = None
    if params["lora"]:
      print("Using standard LoRA")
      config =  LoraConfig( 
        r=params["rank"],
        lora_alpha=params["rank"],
        lora_dropout=params["dropout"],
        target_modules="all-linear",
        task_type="SEQ_CLS",
        use_rslora=True
      )
    elif params["adalora"]:
      print("Using AdaLoRA")
      from peft import AdaLoraConfig
      if params["spike"]:
        print("with SpikeLoRA")
        config = AdaLoraConfig(
          lora_alpha=params["rank"],
          target_r=params["rank"],
          init_r=min(2, params["rank"]),
          tinit=0,
          tfinal=0,
          deltaT=1,
          beta1=0.85,
          beta2=0.85,
          orth_reg_weight=0.5,
          total_step=params["num_epochs"] * (len(train_enc) // params["batch_size"]),
          rank_pattern=None,
          target_modules="all-linear",
          task_type="SEQ_CLS",
          use_spikelora=True,
          spikelora_v_threshold=params["v_threshold"],
          use_rslora=True,
        )
      else:
        print("without SpikeLoRA")
        config = AdaLoraConfig(
          lora_alpha=params["rank"],
          target_r=params["rank"],
          init_r=min(2, params["rank"]),
          tinit=0,
          tfinal=0,
          deltaT=1,
          beta1=0.85,
          beta2=0.85,
          orth_reg_weight=0.5,
          total_step=params["num_epochs"] * (len(train_enc) // params["batch_size"]),
          rank_pattern=None,
          target_modules="all-linear",
          task_type="SEQ_CLS",
          use_rslora=True,
        )
    else:
      print("Using SpikeLoRA")
      config = LoraConfig(
        r=params["rank"],
        lora_alpha=params["rank"],
        lora_dropout=params["dropout"],
        target_modules="all-linear",
        task_type="SEQ_CLS",
        use_spikelora=True,
        use_rslora=True,
        spikelora_v_threshold=params["v_threshold"],
      )
  
    model = get_peft_model(model, config)

    # print model type and number of trainable params
    model.print_trainable_parameters()
    print("Wrapped model:", model)

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
        run_name=params["experiment"],
        fp16=device.type == "cuda", # use fp16 only on CUDA
        remove_unused_columns=False,
        warmup_ratio=0.06,
        warmup_steps=0,
        max_grad_norm=1.0,
        weight_decay=0.01,
        metric_for_best_model="accuracy" if params["task"] not in ["stsb", "cola"] else "matthews_correlation" if params["task"] == "cola" else "pearson",
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

        # --- custom metrics ---
        sparsity_list = []
        sparsity_dict = {}

        for name, mod in trainer.model.named_modules():
            if hasattr(mod, "sparsity"):
                for adapter, v in mod.sparsity.items():
                    val = v.mean().item() if isinstance(v, torch.Tensor) else v
                    sparsity_list.append(val)
                    sparsity_dict[f"{name}/sparsity"] = val

        # global aggregated metrics
        metrics["sparsity"] = float(torch.tensor(sparsity_list).mean()) if sparsity_list else 0.0
        global_sparsity.append(metrics["sparsity"])

        # per-adapter metrics
        metrics.update(sparsity_dict)

        return metrics
    
    wandb.init(project=params["project"], name=params["experiment"], config=params)

    # log gradients to wandb
    wandb.watch(model, log="gradients", log_freq=100)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_enc,
        eval_dataset=val_enc,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[SparsityLoggerCallback()]
    )

    print(next(model.parameters()).device)


    try:
        trainer.train()
        metrics = trainer.evaluate()
        main_score = pick_main_score(metrics)
        wandb.finish()
        avg_sparsity = float(torch.tensor(global_sparsity).mean()) if global_sparsity else 0.0
        return float(main_score) if main_score is not None else -999.0, float(avg_sparsity)
    except Exception as e:
        print(f"[train_and_eval] failed for params={params}: {e}")
        import traceback
        traceback.print_exc()
        return -999.0

# --- Run with param setup ---
if __name__ == "__main__":
    import argparse
    from best_params import BEST_PARAMS

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="rte", help="GLUE task name")
    parser.add_argument("--lora", action="store_true", help="Use LoRA instead of SpikeLoRA")
    parser.add_argument("--adalora", action="store_true", help="Use AdaLoRA instead of SpikeLoRA")
    parser.add_argument("--spike", action="store_true", help="Use a SpikeLORA variant")
    parser.add_argument("--project", type=str, default="glue", help="wandb project name")
    parser.add_argument("--r", type=int, default=None, help="LoRA rank (overrides best param)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides best param)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides loop)")
    parser.add_argument("--dropout", type=float, default=None, help="LoRA dropout (overrides best param)")
    args = parser.parse_args()

    # Convert args Namespace to dict
    params = vars(args)

    # Add extra parameters
    params["rank"] = args.r if args.r is not None else BEST_PARAMS[params["task"]]["lora_r"]
    params["dropout"] = args.dropout if args.dropout is not None else BEST_PARAMS[params["task"]]["lora_dropout"]
    params["v_threshold"] = BEST_PARAMS[params["task"]]["v_threshold"]
    params["learning_rate"] = args.lr if args.lr is not None else BEST_PARAMS[params["task"]]["learning_rate"]
    params["batch_size"] = BEST_PARAMS[params["task"]]["batch_size"]
    params["num_epochs"] = BEST_PARAMS[params["task"]]["num_epochs"]

    # Setup seeds
    seeds = [args.seed] if args.seed is not None else [1,2,3,4,5]
    sparsities = []
    scores = []
    for seed in seeds:
        params["seed"] = seed
        params["experiment"] = f"{args.task}-r{args.rank}-v{args.v_threshold}{'--lora' if args.lora else ''}{'--adalora' if args.adalora else ''}-s{seed}"
        print(f"Running with params: {params}")
        score, sparsity = train_and_eval(**params)
        scores.append(score)
        sparsities.append(sparsity)
        print(f"Score for seed {seed}: {score} (sparsity: {sparsities})")
    
    # Final results
    score = np.mean(scores)
    stdev = np.std(scores)
    print(f"Final score for {args.experiment} experiment: {score} ± {stdev} (n={len(seeds)}) with average sparsity {np.mean(sparsities)}")
