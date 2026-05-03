import os
import numpy as np
from PIL import Image

def load_from_folders(base_path):
    """
    Scans folders, loads images, and assigns labels based on folder names.
    Returns: x (images), y (labels), and a mapping dictionary.
    """
    x_data = []
    y_data = []
    
    classes = sorted(os.listdir(base_path))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_folder = os.path.join(base_path, cls_name)
        if not os.path.isdir(cls_folder):
            continue
            
        for img_name in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, img_name)
            try:
                # Load image, ensure RGB, and resize to 32x32
                with Image.open(img_path) as img:
                    img = img.convert('RGB').resize((32, 32))
                    x_data.append(np.array(img))
                    y_data.append(class_to_idx[cls_name])
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    return np.array(x_data, dtype='float32'), np.array(y_data), class_to_idx

def get_full_dataset(train_dir, test_dir):
    print("Loading Training Data...")
    x_train, y_train, class_map = load_from_folders(train_dir)
    print("Loading Testing Data...")
    x_test, y_test, _ = load_from_folders(test_dir)
    
    return x_train, y_train, x_test, y_test, class_map

if __name__ == "__main__":
    train_dir = "data/cifar10/cifar10/train"
    test_dir = "data/cifar10/cifar10/test"
    x_train, y_train, x_test, y_test, class_map = get_full_dataset(train_dir, test_dir)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    print("class_map:", class_map)