import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
import json
import os

app = Flask(__name__)

# --- 1. Load Models and Metadata ---

# --- Gatekeeper Model (EfficientNet-B0) ---
GATEKEEPER_MODEL_PATH = 'gatekeeper_model.pt'
gatekeeper_class_names = ['Buffalo', 'Cattle', 'other'] # Should match training order

gatekeeper_model = models.efficientnet_b0()
num_features_b0 = gatekeeper_model.classifier[1].in_features
gatekeeper_model.classifier[1] = nn.Linear(num_features_b0, len(gatekeeper_class_names))
gatekeeper_model.load_state_dict(torch.load(GATEKEEPER_MODEL_PATH, map_location='cpu'))
gatekeeper_model.eval()

# --- Breed Classifier Model (Fine-Tuned EfficientNet-B3) ---
BREED_MODEL_PATH = 'best_breed_classifier_b3.pt' 
CLASSES_PATH = 'breed_class_names.json'

with open(CLASSES_PATH, 'r') as f:
    breed_class_names = json.load(f)
num_breeds = len(breed_class_names)

breed_model = models.efficientnet_b3()
num_features_b3 = breed_model.classifier[1].in_features
breed_model.classifier[1] = nn.Linear(num_features_b3, num_breeds)
breed_model.load_state_dict(torch.load(BREED_MODEL_PATH, map_location='cpu'))
breed_model.eval()

print("✅ Models loaded successfully!")

# --- 2. Define Image Transformations ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

gatekeeper_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

breed_transforms = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def transform_image(image_bytes, transform_pipeline):
    """Helper to apply transforms to an image."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform_pipeline(image).unsqueeze(0)

# --- 3. Define Prediction Route ---
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()

        # --- Stage 1: Gatekeeper Prediction ---
        gatekeeper_tensor = transform_image(image_bytes, gatekeeper_transforms)
        with torch.no_grad():
            outputs = gatekeeper_model(gatekeeper_tensor)
            _, preds = torch.max(outputs, 1)
            gatekeeper_prediction = gatekeeper_class_names[preds[0]]
        
        # --- Stage 2: Logic and Breed Prediction ---
        if gatekeeper_prediction == 'other':
            return jsonify({
                "prediction_type": "Invalid Image",
                "message": "The uploaded image is not a cattle or buffalo."
            })
        
        breed_tensor = transform_image(image_bytes, breed_transforms)
        with torch.no_grad():
            outputs = breed_model(breed_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
            breed_prediction = breed_class_names[preds[0]]
            
        return jsonify({
            "prediction_type": "Breed",
            "animal_type": gatekeeper_prediction,
            "breed": breed_prediction,
            "confidence": f"{confidence.item()*100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)