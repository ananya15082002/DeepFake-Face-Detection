import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression


# Function to decode JPEG content to RGB grids of pixels with channels
def decode_jpeg(image_path):
    return cv2.imread(image_path)


# Function to resize images to 256x256
def resize_image(image, size=(256, 256)):
    if image is not None and not image.size == 0:
        return cv2.resize(image, size)
    else:
        return None


# Load YOLOv5 model
def load_yolo_model(weights_path='C:/Users/asus/Desktop/Projects/Minor2DeepFake/yolov5-master/yolov5s.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights_path, device=device)
    return model


# Function to detect face using YOLOv5
def detect_face_yolov5(image, model, conf_threshold=0.5):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to PIL image
    img_tensor = transforms.ToTensor()(img).unsqueeze_(0)  # Convert to tensor
    results = model(img_tensor)[0]  # Perform inference
    results = non_max_suppression(results, conf_threshold)[0]  # Apply non-maximum suppression

    if results is not None and len(results) > 0:
        box = results[0][:4].cpu().numpy().astype(int)
        (x, y, w, h) = box
        face = image[y:y + h, x:x + w]
        return face
    else:
        return None


# Function to preprocess image
def preprocess_image(image):
    if image is not None:
        image = resize_image(image)
        if image is not None:
            image = normalize_image(image)
    return image


# Function to normalize pixel values to [0, 1]
def normalize_image(image):
    return image.astype(np.float32) / 255.0


# Paths to the dataset directories
real_dataset_dir = "C:/Users/asus/Desktop/Projects/Minor2DeepFake/Dataset/Test/Real"
fake_dataset_dir = "C:/Users/asus/Desktop/Projects/Minor2DeepFake/Dataset/Test/Fake"

# Create directories to save preprocessed images
preprocessed_real_dir = 'preprocesseddatanew/real'
preprocessed_fake_dir = 'preprocesseddatanew/fake'
os.makedirs(preprocessed_real_dir, exist_ok=True)
os.makedirs(preprocessed_fake_dir, exist_ok=True)

# Load YOLOv5 model
yolov5_model = load_yolo_model()

# Preprocess real images
for filename in os.listdir(real_dataset_dir):
    image_path = os.path.join(real_dataset_dir, filename)
    image = decode_jpeg(image_path)
    if image is not None:
        face = detect_face_yolov5(image, yolov5_model)
        if face is not None:
            preprocessed_face = preprocess_image(face)
            if preprocessed_face is not None:
                cv2.imwrite(os.path.join(preprocessed_real_dir, filename), preprocessed_face * 255.0)

# Preprocess fake images
for filename in os.listdir(fake_dataset_dir):
    image_path = os.path.join(fake_dataset_dir, filename)
    image = decode_jpeg(image_path)
    if image is not None:
        face = detect_face_yolov5(image, yolov5_model)
        if face is not None:
            preprocessed_face = preprocess_image(face)
            if preprocessed_face is not None:
                cv2.imwrite(os.path.join(preprocessed_fake_dir, filename), preprocessed_face * 255.0)

print("Preprocessing completed and preprocessed images saved.")
