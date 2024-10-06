import os

import cv2
import numpy as np


# Function to decode JPEG content to RGB grids of pixels with channels
def decode_jpeg(image_path):
    return cv2.imread(image_path)

# Function to resize images to 256x256
def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

# Function to detect face and crop it from the image
def crop_face(image):
    # Load the pre-trained Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]
        return face
    else:
        return None

# Function to normalize pixel values to [0, 1]
def normalize_image(image):
    return image.astype(np.float32) / 255.0

# Paths to the dataset directories
real_dataset_dir = "C:/Users/asus/Desktop/Projects/Minor2DeepFake/Dataset/Train/Real"
fake_dataset_dir = "C:/Users/asus/Desktop/Projects/Minor2DeepFake/Dataset/Train/Fake"

# Create directories to save preprocessed images
preprocessed_real_dir = 'preprocesseddata/real'
preprocessed_fake_dir = 'preprocesseddata/fake'
os.makedirs(preprocessed_real_dir, exist_ok=True)
os.makedirs(preprocessed_fake_dir, exist_ok=True)

# Preprocess real images
for filename in os.listdir(real_dataset_dir):
    image_path = os.path.join(real_dataset_dir, filename)
    image = decode_jpeg(image_path)
    if image is not None:
        image = resize_image(image)
        cropped_face = crop_face(image)
        if cropped_face is not None:
            cropped_face = resize_image(cropped_face)
            cropped_face = normalize_image(cropped_face)
            cv2.imwrite(os.path.join(preprocessed_real_dir, filename), cropped_face * 255.0)

# Preprocess fake images
for filename in os.listdir(fake_dataset_dir):
    image_path = os.path.join(fake_dataset_dir, filename)
    image = decode_jpeg(image_path)
    if image is not None:
        image = resize_image(image)
        cropped_face = crop_face(image)
        if cropped_face is not None:
            cropped_face = resize_image(cropped_face)
            cropped_face = normalize_image(cropped_face)
            cv2.imwrite(os.path.join(preprocessed_fake_dir, filename), cropped_face * 255.0)

print("Preprocessing completed and preprocessed images saved.")
    