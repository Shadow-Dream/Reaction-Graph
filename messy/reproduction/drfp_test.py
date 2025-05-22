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
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from scipy.stats import entropy
from math import log2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_test = []

dataset_test = {"x": np.load("/data2/xxx/pistype/testinputs.npy"),
                "y": np.load("/data2/xxx/pistype/testoutputs.npy")}

dataset_test = {"x": torch.tensor(dataset_test["x"], dtype=torch.float32),
                "y": torch.tensor(dataset_test["y"], dtype=torch.long)}

dataset_test = TensorDataset(dataset_test["x"], dataset_test["y"])

data_loader_test = DataLoader(dataset_test, batch_size=256, shuffle=False)

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
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-8)
model.load_state_dict(torch.load("weights/DRFPType/29.ckpt"))
model.eval()

# 定义计算CEN的函数
def confusion_entropy(cm):
    total_samples = np.sum(cm)
    row_entropies = []
    for row in cm:
        row_sum = np.sum(row)
        if row_sum > 0:
            row_probabilities = row / row_sum
            row_entropy = entropy(row_probabilities, base=2)
            row_entropies.append(row_entropy * row_sum / total_samples)
        else:
            row_entropies.append(0)
    return np.sum(row_entropies)

# 测试循环
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(data_loader_test):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算准确率
accuracy = correct / total

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

# 计算CEN
cen = confusion_entropy(cm)

# 计算MCC
mcc = matthews_corrcoef(all_labels, all_preds)

print(f'Accuracy: {accuracy:.4f}')
print(f'CEN: {cen:.4f}')
print(f'MCC: {mcc:.4f}')
