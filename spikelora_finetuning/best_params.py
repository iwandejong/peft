# BEST_PARAMS = {
#   "mrpc": {'learning_rate': 0.0005, 'batch_size': 8, 'num_epochs': 5, 'lora_r': 21, 'lora_alpha': 38, 'lora_dropout': 0.09446732919790111, 'v_threshold': 0.7270969562789545},
#   "wnli": {'learning_rate': 0.0005, 'batch_size': 8, 'num_epochs': 4, 'lora_r': 64, 'lora_alpha': 36, 'lora_dropout': 0.09753718911258191, 'v_threshold': 0.1},
#   "rte": {'learning_rate': 0.00046078589483983576, 'batch_size': 8, 'num_epochs': 5, 'lora_r': 64, 'lora_alpha': 57, 'lora_dropout': 0.3, 'v_threshold': 0.24536888196771087},
#   # "cola": {'learning_rate': 0.005, 'batch_size': 16, 'num_epochs': 5, 'lora_r': 38, 'lora_alpha': 45, 'lora_dropout': 0.15, 'v_threshold': 0.75},
#   "cola": {
#     "learning_rate": 3e-5,
#     "batch_size": 16,
#     "num_epochs": 5,
#     "lora_r": 8,
#     "lora_alpha": 16,
#     "lora_dropout": 0.1,
#     "v_threshold": 0.5,
#   },
#   "sst2": {'learning_rate': 0.000224583317474498, 'batch_size': 16, 'num_epochs': 4, 'lora_r': 39, 'lora_alpha': 15, 'lora_dropout': 0.1838480442440153, 'v_threshold': 0.12836822579856336},
#   "qnli": {'learning_rate': 0.00025225110938366743, 'batch_size': 32, 'num_epochs': 4, 'lora_r': 58, 'lora_alpha': 54, 'lora_dropout': 0.2693754343489524, 'v_threshold': 0.21824550719994545},
#   "mnli": {'learning_rate': 7.071067811865475e-05, 'batch_size': 8, 'num_epochs': 4, 'lora_r': 34, 'lora_alpha': 36, 'lora_dropout': 0.15, 'v_threshold': 0.55},
#   "stsb": {'learning_rate': 0.0005, 'batch_size': 8, 'num_epochs': 5, 'lora_r': 4, 'lora_alpha': 36, 'lora_dropout': 0.18354172790630047, 'v_threshold': 0.5861250082584661}
# }

# BEST_PARAMS = {
#   "cola": {'learning_rate': 1e-4, 'batch_size': 8, 'num_epochs': 20, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': 0.1}, # 4
#   "mrpc": {'learning_rate': 1e-4, 'batch_size': 8, 'num_epochs': 20, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': 0.1}, # 4
#   "stsb": {'learning_rate': 1e-4, 'batch_size': 8, 'num_epochs': 20, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': 0.1}, # 4

#   # "wnli": {'learning_rate': 8e-4, 'batch_size': 32, 'num_epochs': 10, 'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': 0.0, 'v_threshold': 0.5}, # 4
#   "sst2": {'learning_rate': 1e-4, 'batch_size': 16, 'num_epochs': 10, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': 0.1}, # 12
#   "qnli": {'learning_rate': 1e-4, 'batch_size': 16, 'num_epochs': 10, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': 0.1}, # 12
#   "mnli": {'learning_rate': 1e-4, 'batch_size': 32, 'num_epochs': 10, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': 0.1}, # 12
#   "qqp": {'learning_rate': 1e-4, 'batch_size': 32, 'num_epochs': 10, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': 0.1}, # 12

#   "rte": {'learning_rate': 1e-4, 'batch_size': 32, 'num_epochs': 50, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': 0.1}, # 4
# }

BEST_PARAMS = {
  "cola": {'learning_rate': 3e-4, 'batch_size': 32, 'num_epochs': 20, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1}, # 4
  "mrpc": {'learning_rate': 1e-3, 'batch_size': 32, 'num_epochs': 20, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1}, # 4
  "stsb": {'learning_rate': 3e-4, 'batch_size': 16, 'num_epochs': 8, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1}, # 4

  # "wnli": {'learning_rate': 8e-4, 'batch_size': 32, 'num_epochs': 10, 'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': 0.0, 'v_threshold': 0.5}, # 4
  "sst2": {'learning_rate': 8e-4, 'batch_size': 32, 'num_epochs': 8, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1}, # 12
  "qnli": {'learning_rate': 3e-4, 'batch_size': 32, 'num_epochs': 3, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1}, # 12
  "mnli": {'learning_rate': 3e-4, 'batch_size': 32, 'num_epochs': 3, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1}, # 12
  "qqp": {'learning_rate': 3e-4, 'batch_size': 32, 'num_epochs': 3, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1}, # 12

  "rte": {'learning_rate': 1.2e-3, 'batch_size': 32, 'num_epochs': 15, 'lora_r': 8, 'lora_dropout': 0.0, 'v_threshold': .1}, # 4
}

# max hours per task:
MAX_HOURS = {
  "cola": 0.25,
  "rte": 4,
  # "wnli": 4,
  "stsb": 0.15,
  "mrpc": 0.15,
  "sst2": 12,
  "qnli": 12,
  "qqp": 12,
  "mnli": 12,
}
