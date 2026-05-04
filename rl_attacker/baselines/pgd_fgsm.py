from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from rl_attacker.target_models.train_target import load_preprocessed_data
import torch
import torchattacks
import torch.nn as nn

eps = 8/255      # Maximum perturbation
alpha = 2/255    # Step size for PGD
steps = 10       # Iterations for PGD
from torchvision import models


# Load all splits
base_dir = 'data/cifar10/preprocessed'

test_x, test_y = load_preprocessed_data(base_dir, 'test')

# Create DataLoaders
batch_size = 64

test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)


model = models.resnet18(num_classes=10)
model.load_state_dict(torch.load('rl_attacker/target_models/resnet18_weights.pth', map_location=torch.device('cpu')))

model.eval()

print("Model successfully loaded locally!")

# Initialize the attack objects
atk_fgsm = torchattacks.FGSM(model, eps=eps)
atk_pgd = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)

# 2. Evaluation Function
def evaluate_robustness(model, data_loader):
    model.eval()
    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    total = 0

    for images, labels in data_loader:
        total += labels.size(0)

        # Clean Accuracy
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        clean_correct += (predicted == labels).sum().item()

        # FGSM Attack
        adv_images_fgsm = atk_fgsm(images, labels)
        outputs_fgsm = model(adv_images_fgsm)
        _, predicted_fgsm = torch.max(outputs_fgsm.data, 1)
        fgsm_correct += (predicted_fgsm == labels).sum().item()

        # PGD Attack
        adv_images_pgd = atk_pgd(images, labels)
        outputs_pgd = model(adv_images_pgd)
        _, predicted_pgd = torch.max(outputs_pgd.data, 1)
        pgd_correct += (predicted_pgd == labels).sum().item()

    print(f"Clean Accuracy: {100 * clean_correct / total:.2f}%")
    print(f"FGSM Accuracy: {100 * fgsm_correct / total:.2f}%")
    print(f"PGD Accuracy: {100 * pgd_correct / total:.2f}%")

# Run the benchmark
evaluate_robustness(model, test_loader)