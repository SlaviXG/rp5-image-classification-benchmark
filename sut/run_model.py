import sys
import os
import time
import cv2
import numpy as np
import torch
import torchvision.models as models
from enum import Enum
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice)

# Hardcoded model paths for Hailo models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "RESNET50": os.path.join(BASE_DIR, "models", "resnet_v1_50.hef"),
    "MOBILENETV3": os.path.join(BASE_DIR, "models", "mobilenet_v3.hef"),
    "RESNET18": os.path.join(BASE_DIR, "models", "resnet_v1_18.hef"),
    "RESNEXT": os.path.join(BASE_DIR, "models", "resnext50_32x4d.hef"),
    "SQUEEZENET": os.path.join(BASE_DIR, "models", "squeezenet_v1.1.hef")
}

class ModelName(Enum):
    RESNET18 = "ResNet18"
    RESNET50 = "ResNet50"
    RESNEXT = "ResNeXt-50-32x4d"
    MOBILENETV3 = "MobileNetV3"
    SQUEEZENET = "SqueezeNetV1.1"

def load_model_by_enum(model_name, accelerate=False):
    if not accelerate:
        # PyTorch mode
        if model_name == ModelName.RESNET50:
            model = models.resnet50(pretrained=True)    
        elif model_name == ModelName.MOBILENETV3:
            model = models.mobilenet_v3_small(pretrained=True)
        elif model_name == ModelName.RESNET18:
            model = models.resnet18(pretrained=True)
        elif model_name == ModelName.RESNEXT:
            model = models.resnext50_32x4d(pretrained=True)
        elif model_name == ModelName.SQUEEZENET:
            model = models.squeezenet1_1(pretrained=True)
        else:
            print(f"Model {model_name} not supported yet.")
            sys.exit(1)
        model.eval()
        print(f"Loaded PyTorch {model_name.value} pretrained on ImageNet")
        return model, 'pytorch'
    
    # Hailo mode
    else:
        try:
            model_path = MODEL_PATHS[model_name.name]  # Get Hailo .hef model path
        except KeyError:
            print(f"Hailo model path not found for {model_name.name}")
            sys.exit(1)

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.NONE
        target = VDevice(params=params)

        hef = HEF(model_path)
        configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make(network_group, quantized=True, format_type=FormatType.UINT8)

        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_info = hef.get_output_vstream_infos()[0]
        image_height, image_width, channels = input_vstream_info.shape

        return (target, network_group, network_group_params, input_vstreams_params, output_vstreams_params, 
                input_vstream_info, output_vstream_info, (image_height, image_width, channels)), 'hailo'

def preprocess_image(image_path, target_size, framework='pytorch'):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    original_resolution = img.shape[:2]  # Height x Width
    img = cv2.resize(img, target_size)

    if framework == 'pytorch':
        img = img.transpose(2, 0, 1)  # Change from HxWxC to CxHxW
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        img = img.unsqueeze(0)  # Add batch dimension
    elif framework == 'hailo':
        img = img.astype(np.float32) / 255.0  # Normalize and convert to float32
        img = np.expand_dims(img, axis=0)  # Add batch dimension for Hailo
    return img, original_resolution

def classify_images(model, images_dir, fps, framework):
    target_size = (224, 224)  # Most models use this input size
    interval = 1 / fps

    for image_file in sorted(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_file)
        
        if not image_file.endswith(('.jpg', '.jpeg', '.png')):
            continue

        img, original_resolution = preprocess_image(image_path, target_size, framework)
        if img is None:
            continue

        start_time = time.time()

        if framework == 'pytorch':
            with torch.no_grad():
                predictions = model(img)
                predicted_class = predictions.argmax(dim=1).item()

        elif framework == 'hailo':
            (target, network_group, network_group_params, input_vstreams_params, output_vstreams_params, 
             input_vstream_info, output_vstream_info, input_shape) = model
            
            input_data = {input_vstream_info.name: img}
            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                with network_group.activate(network_group_params):
                    infer_results = infer_pipeline.infer(input_data)
                    predicted_class = np.argmax(infer_results[output_vstream_info.name])

        finish_time = time.time()
        inference_time = finish_time - start_time

        # Use target_size for the resolution in the output instead of original_resolution
        print(f"Image: {image_file}, Model Resolution: {target_size[1]}x{target_size[0]}, "
              f"Framework: {framework.upper()}, Set FPS: {fps}, Inference time: {inference_time:.4f} seconds, "
              f"Predicted class: {predicted_class}")
        
        if inference_time < interval:
            time.sleep(interval - inference_time)
          
def main():
    if len(sys.argv) < 4:
        print("Usage: python <script_name.py> <model_name> <fps> <images_directory_path> [--accelerate]")
        sys.exit(1)
    
    model_name_input = sys.argv[1].upper()
    fps = float(sys.argv[2])
    images_dir = sys.argv[3]
    accelerate = '--accelerate' in sys.argv

    try:
        model_name = ModelName[model_name_input]
    except KeyError:
        print(f"Invalid model name: {model_name_input}. Available options: {', '.join([m.name for m in ModelName])}")
        sys.exit(1)

    if not os.path.exists(images_dir):
        print(f"Directory does not exist: {images_dir}")
        sys.exit(1)

    model, framework = load_model_by_enum(model_name, accelerate)
    classify_images(model, images_dir, fps, framework)

if __name__ == "__main__":
    main()
