DeepFake-Face-Detection
📄 Overview
This project focuses on developing a machine learning model to detect deepfake images, specifically targeting manipulated facial images. Leveraging a hybrid model that combines CNN and LSTM networks, the system achieves an accuracy of 87.99% in differentiating real faces from AI-generated deepfakes.

🎯 Motivation
With the rapid rise of deepfake technology, the authenticity of digital content is at risk. This project provides a robust detection tool aimed at supporting media integrity and ensuring trust in online visual content.

🏆 Objectives
Detect manipulated (deepfake) facial images accurately.
Provide a reliable tool to verify the authenticity of digital media.
🧠 Model Architecture
CNN (Convolutional Neural Network): Extracts spatial features such as edges and textures.
LSTM (Long Short-Term Memory): Captures temporal dependencies in the extracted features.
Hybrid Approach: Combines CNN and LSTM to enhance detection capabilities.
📊 Dataset
The model was trained and validated on a diverse dataset of real and deepfake images:

Training Set: 70,000 real and 70,000 fake images
Testing Set: 5,492 fake and 5,413 real images
Source: Kaggle - Deepfake and Real Images

💻 System Requirements
Hardware
CPU: Multi-core processor
GPU: CUDA-enabled for deep learning
RAM: Minimum 16 GB
Software
Python (compatible with TensorFlow or PyTorch)
OpenCV for image processing
📈 Results
The model achieved an accuracy of 87.99% on the test set, showcasing strong potential for real-world applications in media verification and cybersecurity.
