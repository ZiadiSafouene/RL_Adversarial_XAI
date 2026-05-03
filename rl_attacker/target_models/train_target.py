import torch
from torchvision import models
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_preprocessed_data(base_path, split):
    """Loads x and y numpy arrays for a specific split"""
    x_path = os.path.join(base_path, split, f'x_{split}.npy')
    y_path = os.path.join(base_path, split, f'y_{split}.npy')
    
    x = np.load(x_path)
    y = np.load(y_path)
    
    # Convert to torch tensors
    return torch.from_numpy(x).float(), torch.from_numpy(y).long()

# Load all splits
base_dir = 'data/cifar10/preprocessed'
train_x, train_y = load_preprocessed_data(base_dir, 'train')
val_x, val_y = load_preprocessed_data(base_dir, 'val')
test_x, test_y = load_preprocessed_data(base_dir, 'test')

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)

model = models.resnet18(num_classes=10)

model.load_state_dict(torch.load('rl_attacker/target_models/resnet18_weights.pth', map_location=torch.device('cpu')))

model.eval()

print("Model successfully loaded locally!")

model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

print(f"\nFinal Test Accuracy: {100 * test_correct / test_total:.2f}%")
