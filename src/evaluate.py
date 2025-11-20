import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# --- Configuration ---
MODEL_PATH = 'best_breed_classifier_b3.pt'
CLASSES_PATH = 'breed_class_names.json'
DATA_DIR = 'dataset/val' # We evaluate on the validation set
IMAGE_SIZE = 300
BATCH_SIZE = 16

if __name__ == '__main__':
    # --- Load Class Names ---
    with open(CLASSES_PATH, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    # --- Model Definition ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for evaluation.")
    
    model = models.efficientnet_b3()
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # --- Data Preparation ---
    val_transforms = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(DATA_DIR, val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Get Predictions ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Compute and Plot Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    
    # Increase figure size for better readability with 40 classes
    fig, ax = plt.subplots(figsize=(20, 20))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()