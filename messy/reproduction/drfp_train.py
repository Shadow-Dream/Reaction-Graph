import torch
from torch import nn
import numpy as np
import pickle as pkl
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = []
dataset_eval = []


dataset_train = {"x": np.load("/data1/xxx/pistype/traininputs.npy"),
                 "y": np.load("/data1/xxx/pistype/trainoutputs.npy")}
dataset_eval = {"x": np.load("/data1/xxx/pistype/evalinputs.npy"),
                "y": np.load("/data1/xxx/pistype/evaloutputs.npy")}

dataset_train = {"x": torch.tensor(dataset_train["x"], dtype=torch.float32),
                 "y": torch.tensor(dataset_train["y"], dtype=torch.long)}
dataset_eval = {"x": torch.tensor(dataset_eval["x"], dtype=torch.float32),
                "y": torch.tensor(dataset_eval["y"], dtype=torch.long)}

dataset_train = TensorDataset(dataset_train["x"], dataset_train["y"])
dataset_eval = TensorDataset(dataset_eval["x"], dataset_eval["y"])

data_loader_train = DataLoader(dataset_train, batch_size=256, shuffle=True)
data_loader_eval = DataLoader(dataset_eval, batch_size=256, shuffle=False)

# 定义模型
class TypeMLP(nn.Module):
    def __init__(self):
        super(TypeMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(16384, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 12)
        )

    def forward(self, x):
        return self.model(x)

model = TypeMLP().to(device)
model.eval()
model.load_state_dict(torch.load("weights/DRFPType/29.ckpt"))
eval_loader = tqdm(data_loader_eval, desc=f"[eval]", leave=False)
all_targets = []
all_outputs=  []
with torch.no_grad():
    for inputs, targets in eval_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        outputs = outputs.argmax(-1)
        all_targets.append(targets.cpu().numpy())
        all_outputs.append(outputs.cpu().numpy())
all_targets = np.concatenate(all_targets)
all_outputs = np.concatenate(all_outputs)
with open("weights/DRFPType/eval_results.pkl", "wb") as f:
    pkl.dump({"targets": all_targets, "outputs": all_outputs}, f)
# criterion = CrossEntropyLoss()
# optimizer = Adam(model.parameters(), lr=0.001)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-8)

# num_epochs = 100
# accumulation_steps = 4

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     optimizer.zero_grad()

#     train_loader = tqdm(data_loader_train, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
#     for i, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss = loss / accumulation_steps

#         loss.backward()

#         if (i + 1) % accumulation_steps == 0:
#             optimizer.step()
#             optimizer.zero_grad()

#         running_loss += loss.item() * accumulation_steps
#         train_loader.set_postfix({"Loss": running_loss / (i + 1)})

#     avg_loss = running_loss / len(data_loader_train)
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

#     model.eval()
#     correct = 0
#     total = 0

#     eval_loader = tqdm(data_loader_eval, desc=f"Epoch {epoch+1}/{num_epochs} [eval]", leave=False)
#     with torch.no_grad():
#         for inputs, targets in eval_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             outputs = outputs.argmax(-1)
#             total += targets.size(0)
#             correct += (outputs == targets).sum().item()
#             eval_loader.set_postfix({"Accuracy": 100 * correct / total})

#     accuracy = correct / total

#     scheduler.step(1 - accuracy)

#     accuracy = accuracy * 100
#     print(f'eval Accuracy: {accuracy:.2f}%')

#     torch.save(model.state_dict(), f"weights/DRFPType/{epoch}.ckpt")
#     with open(f"weights/DRFPType/{epoch}.log", "w") as f:
#         f.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, eval Accuracy: {accuracy}\n")

# print("训练完成")
