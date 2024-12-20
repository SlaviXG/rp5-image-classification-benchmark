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
from signal import signal, SIGINT

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

stop_execution = False

def load_model_by_enum(model_name, accelerate=False):
    if not accelerate:
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
        return model, 'pytorch'
    
    else:
        try:
            model_path = MODEL_PATHS[model_name.name]
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

def load_images_into_memory(images_dir, target_size, framework='pytorch'):
    images = []
    for image_file in sorted(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_file)
        if not image_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            if framework == 'pytorch':
                img = img.transpose(2, 0, 1)  # Change from HxWxC to CxHxW
                img = torch.tensor(img, dtype=torch.float32) / 255.0
                img = img.unsqueeze(0)  # Add batch dimension
            elif framework == 'hailo':
                img = img.astype(np.float32) / 255.0  # Normalize
                img = np.expand_dims(img, axis=0)  # Add batch dimension
            images.append(img)
    return images

def signal_handler(sig, frame):
    global stop_execution
    stop_execution = True

def measure_max_fps(model, images, framework):
    total_time = 0
    frame_count = 0
    signal(SIGINT, signal_handler)

    print("Starting FPS measurement. Press Ctrl+C to stop...")
    
    if framework == 'pytorch':
        with torch.no_grad():
            while not stop_execution:
                for img in images:
                    start_time = time.time()
                    model(img)
                    total_time += time.time() - start_time
                    frame_count += 1
                    if stop_execution:
                        break
    elif framework == 'hailo':
        (target, network_group, network_group_params, input_vstreams_params, 
         output_vstreams_params, input_vstream_info, output_vstream_info, input_shape) = model
        
        input_data_name = input_vstream_info.name
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                while not stop_execution:
                    for img in images:
                        input_data = {input_data_name: img}
                        start_time = time.time()
                        infer_pipeline.infer(input_data)
                        total_time += time.time() - start_time
                        frame_count += 1
                        if stop_execution:
                            break

    avg_inference_time = total_time / frame_count if frame_count > 0 else 0
    max_fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
    print(f"\nTotal frames processed: {frame_count}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Maximum FPS achievable: {max_fps:.2f}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python <script_name.py> <model_name> <images_directory_path> [--accelerate]")
        sys.exit(1)
    
    model_name_input = sys.argv[1].upper()
    images_dir = sys.argv[2]
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
    target_size = (224, 224)
    images = load_images_into_memory(images_dir, target_size, framework)
    
    if not images:
        print("No valid images found in the directory.")
        sys.exit(1)

    measure_max_fps(model, images, framework)

if __name__ == "__main__":
    main()
