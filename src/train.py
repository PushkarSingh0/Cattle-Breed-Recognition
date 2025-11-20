import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import time
import matplotlib.pyplot as plt

# --- 1. Configuration ---
DATA_DIR = 'dataset/'
MODEL_SAVE_PATH = 'best_breed_classifier_b3.pt'
CLASSES_SAVE_PATH = 'breed_class_names.json'
IMAGE_SIZE = 300  # Image size for EfficientNet-B3
BATCH_SIZE = 16   # Lower batch size for a larger model to fit in memory
NUM_EPOCHS = 20   # Train for more epochs with more data
LEARNING_RATE = 0.001

# --- Main execution block ---
if __name__ == '__main__':
    # --- 2. Data Preparation and Augmentation ---
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Use heavy augmentation for the training set
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transforms),
        'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_transforms)
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # --- 3. Save Class Names ---
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    with open(CLASSES_SAVE_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"Found {num_classes} classes. Class names saved to {CLASSES_SAVE_PATH}")

    # --- 4. Model Definition (EfficientNet-B3) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training.")

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

    # Freeze feature layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model.to(device)

    # --- 5. Training Loop ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    best_val_accuracy = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct_predictions = 0

            pbar = tqdm(dataloaders[phase], desc=f"[{phase.capitalize()}]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                correct_predictions += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = correct_predictions.double() / dataset_sizes[phase]
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_val_accuracy:
                best_val_accuracy = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"🎉 New best model saved to {MODEL_SAVE_PATH}")
    
    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    # --- 6. Plotting Results ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()

    plt.show()