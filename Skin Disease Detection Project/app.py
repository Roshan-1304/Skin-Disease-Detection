import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
import os
import logging
import time  # For delay

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

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
    try:
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

        # Make prediction
        with torch.no_grad():
            output = model(image)

        # Get the predicted class (index with the highest score)
        _, predicted = torch.max(output, 1)
        return predicted.item()

    except Exception as e:
        logging.error(f"Error in predict_image: {e}")
        return None

# Retry function to handle file lock
def wait_for_file(file_path, max_retries=5, delay=1):
    retries = 0
    while retries < max_retries:
        try:
            # Try opening the file to check if it's unlocked
            with open(file_path, 'rb'):
                return True  # If successful, file is not locked
        except IOError:
            # File is locked, retry after waiting
            retries += 1
            time.sleep(delay)  # Wait before retrying
    return False  # Max retries reached, file is still locked

# Initialize Flask app
app = Flask(__name__)

# Set device to CPU (or GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model with 2 output classes (Healthy and Diseased skin)
model = SkinDiseaseModel(num_classes=2)

# Load pretrained weights (excluding the final layer's weights)
try:
    model.load_state_dict(torch.load('best_model.pth', map_location=device), strict=False)
    logging.info("Model weights loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Move the model to the selected device (CPU or GPU)
model.to(device)

# Disease info for Melanoma
disease_info = {
    1: {
        'disease_name': 'Melanoma',
        'stage': 'Advanced Melanoma',
        'tips': [
            'Consult with a dermatologist for regular checkups.',
            'Avoid excessive sun exposure and wear sunscreen.',
            'Keep an eye on any changing moles or skin spots.',
            'Seek treatment options such as surgery or immunotherapy.'
        ]
    },
    0: {
        'disease_name': 'Healthy Skin',
        'stage': 'No disease detected',
        'tips': [
            'Keep your skin hydrated.',
            'Protect your skin from UV rays using sunscreen.',
            'Maintain a balanced diet rich in antioxidants.'
        ]
    }
}

# Serve the index.html page
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logging.error('No file part in request')
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            logging.error('No selected file')
            return jsonify({'error': 'No selected file'}), 400

        # Save the file temporarily in the 'uploads' directory
        file_path = os.path.join('uploads', file.filename)

        # Save the file
        file.save(file_path)
        logging.info(f"File saved: {file_path}")

        # Ensure the file is not locked before using it for prediction
        if not wait_for_file(file_path):
            logging.error(f"File is still locked: {file_path}")
            return jsonify({'error': 'File is locked. Please try again later.'}), 500

        # Make prediction
        predicted_class = predict_image(model, file_path)

        if predicted_class is None:
            logging.error(f"Prediction failed for file {file_path}")
            return jsonify({'error': 'Prediction failed. Please try again.'}), 500

        # Clean up the uploaded file
        os.remove(file_path)

        # Return prediction result
        disease_data = disease_info.get(predicted_class, None)
        if disease_data:
            return jsonify({
                'prediction': disease_data['disease_name'],
                'disease_info': disease_data
            })
        else:
            return jsonify({'error': 'Unknown prediction class'}), 500

    except Exception as e:
        logging.error(f"Error in /predict route: {e}")
        return jsonify({'error': 'Internal server error. Please try again later.'}), 500


if __name__ == "__main__":
    # Create the uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
