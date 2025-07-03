import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import numpy as np

# Define the custom model using ResNet-18
class SkinDiseaseModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SkinDiseaseModel, self).__init__()
        self.base_model = models.resnet18(weights='IMAGENET1K_V1')  # Load pretrained weights
        # Modify final layer to match number of classes (e.g., 2 classes: Healthy vs Diseased)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Function to preprocess image and make predictions
def predict_image(model, image_path):
    # Define transformation for input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ImageNet
    ])

    # Open the image file
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move image to the same device as the model (assuming CPU for simplicity here)
    image = image.to(device)

    # Put model in evaluation mode (important for consistent predictions)
    model.eval()

    # Make prediction and log raw outputs
    with torch.no_grad():
        output = model(image)
        print(f"Raw model output (before softmax): {output}")

    # Get the predicted class (index with the highest score)
    _, predicted = torch.max(output, 1)
    return predicted.item(), output.cpu().numpy()  # Return both predicted class and raw output

# Create a Tkinter window to allow user to upload an image
def upload_image():
    # Initialize Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path

# Main script
if __name__ == "__main__":
    # Set device to CPU (or GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model with 2 output classes (Healthy and Diseased skin)
    model = SkinDiseaseModel(num_classes=2)

    # Load pretrained weights (excluding the final layer's weights)
    try:
        # Load the model weights (ensure the architecture matches the training setup)
        model.load_state_dict(torch.load('best_model.pth', map_location=device), strict=False)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

    # Move the model to the selected device (CPU or GPU)
    model.to(device)

    # Ask user to upload an image
    image_path = upload_image()
    if image_path:
        # Make a prediction on the uploaded image
        predicted_class, raw_output = predict_image(model, image_path)
        print(f"Raw output: {raw_output}")
        print(f"Predicted class: {'Diseased Skin' if predicted_class == 1 else 'Healthy Skin'}")
    else:
        print("No image selected.")
