# train_model.py
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import json

# --- 1. Define the Custom Dataset Class ---
# This class loads our image data and prepares it for PyTorch.
class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # Create a mapping from class name (e.g., 'A') to an integer index (e.g., 0)
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(os.listdir(data_dir)))}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        # Load all image paths and their corresponding labels
        print("Loading dataset...")
        for class_name, class_idx in self.class_to_idx.items():
            class_path = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(class_idx)
        print(f"Dataset loaded. Found {len(self.image_paths)} images in {len(self.class_to_idx)} classes.")
        print("Class mapping:", self.class_to_idx)


    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Fetches one sample from the dataset at the given index
        img_path = self.image_paths[idx]
        # Load image using OpenCV, in grayscale mode
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            return None, None
            
        label = self.labels[idx]

        # Apply transformations (e.g., resize, convert to tensor)
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 2. Define the Neural Network (CNN) Architecture ---
# This is a simple CNN model with two convolutional layers and two fully connected layers.
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer 1: Takes 1 input channel (grayscale), outputs 16 channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolutional Layer 2: Takes 16 input channels, outputs 32 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flatten the output for the fully connected layers
        self.flatten = nn.Flatten()
        
        # Fully Connected Layer 1: Input size depends on the image size after pooling
        # For a 64x64 image, after two 2x2 pools, it becomes 16x16. So, 32 * 16 * 16
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        
        # Fully Connected Layer 2 (Output Layer)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Define the forward pass
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. Main Training Script ---
if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIR = './DATASETS'
    MODEL_SAVE_PATH = 'sign_language_model.pth'
    MAPPING_SAVE_PATH = 'class_mapping.json'
    IMAGE_SIZE = 64
    BATCH_SIZE = 32
    NUM_EPOCHS = 50# You can increase this for better accuracy
    LEARNING_RATE = 0.001
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Preparation ---
    # Define transformations to be applied to each image
    # We resize the image and convert it to a PyTorch tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Create the full dataset
    full_dataset = SignLanguageDataset(data_dir=DATA_DIR, transform=transform)
    
    # Save the class mapping for later use in prediction
    with open(MAPPING_SAVE_PATH, 'w') as f:
        json.dump(full_dataset.class_to_idx, f)
    print(f"Class mapping saved to {MAPPING_SAVE_PATH}")
    
    num_classes = len(full_dataset.class_to_idx)

    # Split dataset into training (80%) and validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Create data loaders for batching
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, Loss, and Optimizer ---
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss() # Good for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation Phase
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # No need to calculate gradients for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    print("--- Training Finished ---")

    # --- Save the Trained Model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
