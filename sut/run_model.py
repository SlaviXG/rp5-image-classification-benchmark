import sys
import os
import time
import cv2
import numpy as np
import torch
import torchvision.models as models
from enum import Enum

class ModelName(Enum):
    RESNET18 = "ResNet18"  # PyTorch
    RESNET50 = "ResNet50"  # PyTorch
    RESNEXT = "ResNeXt-50-32x4d"  # PyTorch
    MOBILENETV3 = "MobileNetV3"  # PyTorch
    SQUEEZENET = "SqueezeNetV1.1"  # PyTorch

def load_model_by_enum(model_name, accelerate=False):
    if model_name == ModelName.RESNET50:
        # PyTorch ResNet50
        model = models.resnet50(pretrained=True)
        model.eval()
        print(f"Loaded PyTorch {model_name.value} pretrained on ImageNet")
        return model, 'pytorch'
    elif model_name == ModelName.MOBILENETV3:
        # PyTorch MobileNetV3
        model = models.mobilenet_v3_small(pretrained=True)
        model.eval()
        print(f"Loaded PyTorch {model_name.value} pretrained on ImageNet")
        return model, 'pytorch'
    elif model_name == ModelName.RESNET18:
        # PyTorch ResNet18
        model = models.resnet18(pretrained=True)
        model.eval()
        print(f"Loaded PyTorch {model_name.value} pretrained on ImageNet")
        return model, 'pytorch'
    elif model_name == ModelName.RESNEXT:
        # PyTorch ResNeXt-50-32x4d
        model = models.resnext50_32x4d(pretrained=True)
        model.eval()
        print(f"Loaded PyTorch {model_name.value} pretrained on ImageNet")
        return model, 'pytorch'
    elif model_name == ModelName.SQUEEZENET:
        # PyTorch SqueezeNetV1.1
        model = models.squeezenet1_1(pretrained=True)
        model.eval()
        print(f"Loaded PyTorch {model_name.value} pretrained on ImageNet")
        return model, 'pytorch'
    else:
        print(f"Model {model_name} not supported yet.")
        sys.exit(1)

def preprocess_image(image_path, target_size, framework='pytorch'):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    original_resolution = img.shape[:2]  # Height x Width
    img = cv2.resize(img, target_size)

    # For PyTorch models
    img = img.transpose(2, 0, 1)  # Change from HxWxC to CxHxW
    img = torch.tensor(img, dtype=torch.float32) / 255.0
    img = img.unsqueeze(0)  # Add batch dimension
    return img, original_resolution

def classify_images(model, images_dir, fps, framework):
    target_size = (224, 224)  # Most models use this input size
    interval = 1 / fps

    for image_file in sorted(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_file)
        
        if not image_file.endswith(('.jpg', '.jpeg', '.png')):
            continue

        start_time = time.time()

        img, original_resolution = preprocess_image(image_path, target_size, framework)
        if img is None:
            continue

        with torch.no_grad():
            predictions = model(img)
            predicted_class = predictions.argmax(dim=1).item()

        finish_time = time.time()
        time_taken = finish_time - start_time
        actual_fps = 1 / time_taken

        print(f"Image: {image_file}, Start: {int(start_time)}, Finish: {int(finish_time)}, "
              f"Resolution: {original_resolution[1]}x{original_resolution[0]}, Framework: {framework.upper()}, FPS: {actual_fps:.2f}, "
              f"Predicted class: {predicted_class}")
        
        if time_taken < interval:
            time.sleep(interval - time_taken)

def main():
    if len(sys.argv) < 4:
        print("Usage: python <script_name.py> <model_name> <fps> <images_directory_path> [--accelerate]")
        sys.exit(1)
    
    model_name_input = sys.argv[1].upper()
    fps = float(sys.argv[2])
    images_dir = sys.argv[3]

    try:
        model_name = ModelName[model_name_input]
    except KeyError:
        print(f"Invalid model name: {model_name_input}. Available options: {', '.join([m.name for m in ModelName])}")
        sys.exit(1)

    if not os.path.exists(images_dir):
        print(f"Directory does not exist: {images_dir}")
        sys.exit(1)

    model, framework = load_model_by_enum(model_name)
    classify_images(model, images_dir, fps, framework)

if __name__ == "__main__":
    main()
