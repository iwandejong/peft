import torch
import torch.nn as nn
from peft import get_peft_model, SpikeLoraConfig

# Dummy base model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20, bias=False)   # target module
        self.fc2 = nn.Linear(20, 5, bias=False)    # untouched

    def forward(self, x):
        return self.fc2(self.fc1(x))

base = TinyModel()

# SpikeLoRA config
config = SpikeLoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["fc1"],  # only replace fc1
    v_threshold=0.6,
)

# Wrap model with PEFT factory
model = get_peft_model(base, config)

# Inspect
print("Wrapped model:", model)

# Forward test
x = torch.randn(2, 10)
out = model(x)
print("Output shape:", out.shape)

# Trainable params
model.print_trainable_parameters()

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(10):
    optimizer.zero_grad()
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

print("Final loss:", loss.item())