import os
import numpy as np
import matplotlib.pyplot as plt
import json 
from load_data import get_full_dataset

def save_numpy_data(output_base, split_name, x, y):
    """
    Saves the preprocessed arrays into the specified directory.
    Structure: data/cifar10/preprocessed/[train/val/test]/...
    """
    path = os.path.join(output_base, split_name)
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(os.path.join(path, f'x_{split_name}.npy'), x)
    np.save(os.path.join(path, f'y_{split_name}.npy'), y)
    
def visualize_and_save_metrics(output_base, mean_image, std, class_map):
    """
    Creates plots/images of the mean image, standard deviation, and class map,
    and saves them to the output directory.
    """
    print("Saving Class Map visualization...")
    
    with open(os.path.join(output_base, 'class_map.json'), 'w') as f:
        json.dump(class_map, f, indent=4)

    # Simple plot representation of the class map
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    class_text = "\n".join([f"{k}: {v}" for k, v in class_map.items()])
    ax.text(0.1, 0.5, f"Class Mapping:\n\n{class_text}", transform=ax.transAxes, 
            fontsize=12, verticalalignment='center')
    plt.savefig(os.path.join(output_base, 'class_map.png'), bbox_inches='tight')
    plt.close()

    # Note: Mean Image and Std are calculated on (H, W, C) data before transposition.

    # 2. Save Mean Image visualization
    print("Saving Mean Image visualization...")
    # Normalize mean image to 0-1 for display
    vis_mean = mean_image - np.min(mean_image)
    vis_mean /= np.max(vis_mean)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(vis_mean)
    plt.title("Mean Image of Training Set")
    plt.axis('off')
    plt.savefig(os.path.join(output_base, 'mean_image.png'))
    plt.close()

    # 3. Save Std (Standard Deviation) visualization
    print("Saving Std visualization...")
    # Standard deviation often highlights edges. Rescale 0-1 for display.
    vis_std = std - np.min(std)
    vis_std /= np.max(vis_std)

    plt.figure(figsize=(4, 4))
    plt.imshow(vis_std)
    plt.title("Standard Deviation of Training Set")
    plt.axis('off')
    plt.savefig(os.path.join(output_base, 'std.png'))
    plt.close()

def pre_process_and_save_to_disk(train_path, test_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    x_train, y_train, x_test, y_test, class_map = get_full_dataset(train_path, test_path)
    
    # 2. Global Normalization (Scaling to 0-1)
    x_train /= 255.0
    x_test /= 255.0

    # 3. Validation Split (Last 1000 examples)
    x_val = x_train[-1000:]
    y_val = y_train[-1000:]
    x_train = x_train[:-1000]
    y_train = y_train[:-1000]

    # 4. Feature-wise Normalization (Mean & Std)
  
    mean_image = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)


    # Apply to all splits
    x_train = (x_train - mean_image) / std
    x_val = (x_val - mean_image) / std
    x_test = (x_test - mean_image) / std

    # 5. Transpose to Channel-First (N, C, H, W) 
    x_train = x_train.transpose(0, 3, 1, 2)
    x_val = x_val.transpose(0, 3, 1, 2)
    x_test = x_test.transpose(0, 3, 1, 2)

    # 6. Save to Directory
    print(f"Saving preprocessed data to {output_path}...")
    save_numpy_data(output_path, 'train', x_train, y_train)
    save_numpy_data(output_path, 'val', x_val, y_val)
    save_numpy_data(output_path, 'test', x_test, y_test)
    
    print("Preprocessing and saving complete.")

if __name__ == "__main__":
    input_train = 'data/cifar10/cifar10/train'
    input_test = 'data/cifar10/cifar10/test'
    output_dir = 'data/cifar10/preprocessed'
    
    pre_process_and_save_to_disk(input_train, input_test, output_dir)