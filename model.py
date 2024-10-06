import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# Load MTCNN model for face detection
mtcnn_detector = MTCNN()

# Load your pretrained CNN model for deepfake detection
cnn_model = load_model("cnn_model_using_fisherface_lbph_trained.h5")

def preprocess_face_for_classification(face):
    # Convert RGB image to grayscale
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Resize the grayscale image to 256x256
    resized_face = cv2.resize(gray_face, (256, 256))
    # Flatten the image
    flattened_face = resized_face.flatten()
    return flattened_face

# Function to predict whether the face is real or fake
def predict_real_fake(face):
    # Preprocess the face
    preprocessed_face = preprocess_face_for_classification(face)
    
    # Predict using the loaded CNN model
    prediction = cnn_model.predict(np.array([preprocessed_face]))
    
    # Assuming a threshold of 0.5 for binary classification
    if prediction > 0.5:
        return "Fake"
    else:
        return "Real"

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Set the video frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform face detection using MTCNN
    faces = mtcnn_detector.detect_faces(frame)

    # Process each detected face
    for face_info in faces:
        x, y, w, h = face_info['box']
        
        # Extract the detected face
        face = frame[y:y+h, x:x+w]

        # Predict whether the face is real or fake
        prediction = predict_real_fake(face)

        # Draw bounding box and label on the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw green rectangle around the face
        cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Deepfake Detection', frame)

    # Break the loop if 'q' is pressed or if the window is closed
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Deepfake Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
