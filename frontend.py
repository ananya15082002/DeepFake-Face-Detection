import tkinter as tk  # GUI toolkit for python
from tkinter import filedialog, messagebox

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageTk
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# Define your hybrid CNN-LSTM model
class HybridModel(nn.Module):
    def __init__(self, cnn_model):
        super(HybridModel, self).__init__()
        self.cnn_model = cnn_model
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 2)  # Output classes: real and fake

    def forward(self, x):
        features = self.cnn_model(x)  # Extract features using CNN
        features = features.view(features.size(0), 1, -1)
        lstm_out, _ = self.lstm(features)
        output = self.fc(lstm_out[:, -1, :])  # Take the last LSTM output
        return output

# Load the pretrained hybrid model
def load_model(model_path):
    cnn_model = models.resnet18(pretrained=False)
    cnn_model.fc = nn.Identity()  # Remove the last fully connected layer
    model = HybridModel(cnn_model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Classify image
def classify_image(model, image):
    image = preprocess_image(image)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted.item()].item()
    return predicted.item(), confidence

# Create UI
class Application:
    def __init__(self, master):
        self.master = master
        self.master.title("DeepFake Face Detection")
        self.master.configure(bg="#ADD8E6")  # Set background color
        
        self.heading_label = tk.Label(self.master, text="DeepFake Face Detection", font=("Helvetica", 20, "bold"), bg="#ADD8E6", fg="#333333")
        self.heading_label.pack(pady=20)
        
        self.instruction_label = tk.Label(self.master, text="Please select your image", font=("Helvetica", 14), bg="#ADD8E6", fg="#333333")
        self.instruction_label.pack()
        
        self.open_button = tk.Button(self.master, text="Browse", command=self.open_file, font=("Helvetica", 14), bg="#4CAF50", fg="white", relief="raised")
        self.open_button.pack(pady=10)
        
        self.image_frame = tk.Frame(self.master, bg="#ADD8E6", bd=2, relief="solid")
        self.image_frame.pack(pady=20)
        
        self.image_label = tk.Label(self.image_frame, bg="#ffffff")
        self.image_label.pack()
        
        self.result_label = tk.Label(self.master, text="", font=("Helvetica", 14), bg="#ADD8E6", fg="#333333")
        self.result_label.pack(pady=10)
        
        self.clear_button = tk.Button(self.master, text="Clear", command=self.clear_image, font=("Helvetica", 14), bg="#f44336", fg="white", relief="raised")
        self.clear_button.pack(pady=10)

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        # Load the image
        image = Image.open(file_path).convert('RGB')
        
        # Classify the image
        prediction, confidence = classify_image(model, image)
        
        # Display the image with prediction and detected faces
        self.update_image(image)
        self.update_result(prediction, confidence)
    
    def update_image(self, image):
        # Convert PIL image to Tkinter PhotoImage
        img_tk = ImageTk.PhotoImage(image)
        
        # Update image label
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk
    
    def update_result(self, prediction, confidence):
        if prediction == 0:
            result_text = f"Prediction: Fake\nConfidence: {confidence:.2f}"
        else:
            result_text = f"Prediction: Real\nConfidence: {confidence:.2f}"
        self.result_label.config(text=result_text)
    
    def clear_image(self):
        self.image_label.config(image="")
        self.result_label.config(text="")

def main():
    # Load the pretrained model
    model_path = 'hybrid_model.pth'
    global model
    model = load_model(model_path)
    
    # Create the Tkinter application
    root = tk.Tk()
    root.geometry("500x600")  # Set window size
    app = Application(root)
    root.mainloop()

if __name__ == "__main__":
    main()
