import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


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

# Define dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root='preprocesseddatanew', transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize your CNN model
cnn_model = models.resnet18(pretrained=True)
cnn_model.fc = nn.Identity()  # Remove the last fully connected layer

# Initialize your hybrid model
model = HybridModel(cnn_model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train your hybrid model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

# Evaluate your model
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / total_samples
print("Accuracy:", accuracy)

# Save your model
torch.save(model.state_dict(), 'hybrid_model.pth')
