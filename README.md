# DeepFake-Face-Detection

## 📄 Overview
This project focuses on developing a machine learning model to detect deepfake images, specifically targeting manipulated facial images. Leveraging a hybrid model that combines **CNN** and **LSTM** networks, the system achieves an accuracy of **87.99%** in differentiating real faces from AI-generated deepfakes.

## 🎯 Motivation
With the rapid rise of deepfake technology, the authenticity of digital content is at risk. This project provides a robust detection tool aimed at supporting media integrity and ensuring trust in online visual content.

## 🏆 Objectives
- **Detect** manipulated (deepfake) facial images accurately.
- **Provide** a reliable tool to verify the authenticity of digital media.

## Methodology 
![image](https://github.com/user-attachments/assets/54d5487c-cd5b-4046-b048-60bb998799dd)

## 🧠 Model Architecture
- **CNN (Convolutional Neural Network)**: Extracts spatial features such as edges and textures.
- **LSTM (Long Short-Term Memory)**: Captures temporal dependencies in the extracted features.
- **Hybrid Approach**: Combines CNN and LSTM to enhance detection capabilities.

## 📊 Dataset
The model was trained and validated on a diverse dataset of real and deepfake images:
- **Training Set**: 70,000 real and 70,000 fake images
- **Testing Set**: 5,492 fake and 5,413 real images

**Source**: [Kaggle - Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

## Training Workflow
![image](https://github.com/user-attachments/assets/90a06b8e-a8b8-4d00-b839-8a2678442694)



## 💻 System Requirements
### Hardware
- **CPU**: Multi-core processor
- **GPU**: CUDA-enabled for deep learning
- **RAM**: Minimum 16 GB

### Software
- **Python** (compatible with TensorFlow or PyTorch)
- **OpenCV** for image processing

## 📈 Results
The model achieved an accuracy of **87.99%** on the test set, showcasing strong potential for real-world applications in media verification and cybersecurity.
![image](https://github.com/user-attachments/assets/2ecd5324-a169-4ef1-9e1e-e31db94cfc73)
![image](https://github.com/user-attachments/assets/10ac09ec-fefa-430a-b94a-2e3d0ce99997)

## 🔮 Future Scope
- **Explore video and multimodal detection**: Expand the detection capabilities to handle video content and integrate multimodal detection approaches for more comprehensive results.
- **Address data scarcity and generalizability**: Enhance the model's robustness by addressing limitations in training data and improving its ability to generalize across diverse datasets.
- **Partner with social media platforms and raise public awareness**: Collaborate with social media platforms to integrate detection technology and promote awareness about deepfake risks.
- **Investigate new architectures like Transformers and Explainable AI**: Explore advanced architectures, such as Transformers, and incorporate Explainable AI techniques to improve model interpretability and performance.


