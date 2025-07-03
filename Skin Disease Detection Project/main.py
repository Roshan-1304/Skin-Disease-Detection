import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Check if a GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data directories
train_dir = 'dataset/train'
valid_dir = 'dataset/validation'

# Check if the dataset directories exist
if not os.path.exists(train_dir):
    print(f"Training directory does not exist at: {train_dir}")
    exit()

if not os.path.exists(valid_dir):
    print(f"Validation directory does not exist at: {valid_dir}")
    exit()

# Set up data augmentation and normalization for training and validation datasets
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Check dataset sizes and class labels
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(valid_dataset)}")
print(f"Classes in the dataset: {train_dataset.classes}")

# Define a simple CNN model with dropout for regularization
class SkinDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(SkinDiseaseModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        self.dropout = nn.Dropout(p=0.5)  # Adding dropout to prevent overfitting

    def forward(self, x):
        x = self.base_model(x)
        return self.dropout(x)  # Apply dropout at the end of the forward pass

# Instantiate the model, loss function, and optimizer
model = SkinDiseaseModel(num_classes=len(train_dataset.classes)).to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
best_acc = 0.0

for epoch in range(1, num_epochs + 1):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # Training loop with tqdm for progress bar
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients during validation
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_acc = 100 * correct / total

    print(f"Epoch {epoch}/{num_epochs}")
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {valid_acc:.2f}%")

    # Save the best model
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved.")

print("Training complete.")
