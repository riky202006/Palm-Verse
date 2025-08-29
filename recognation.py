# recognize_gesture.py
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import json
from collections import deque # Import deque for an efficient history list

# --- 1. Define the Original Neural Network (CNN) Architecture ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Main Recognition Script ---
if __name__ == '__main__':
    MODEL_PATH = 'model/sign_language_model.pth'
    MAPPING_PATH = 'model/class_mapping.json'
    IMAGE_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        with open(MAPPING_PATH, 'r') as f:
            class_to_idx = json.load(f)
        idx_to_class = {int(i): cls for cls, i in class_to_idx.items()}
        num_classes = len(class_to_idx)
        print("Class mapping loaded successfully.")
        print(f"Found {num_classes} classes.")
    except FileNotFoundError:
        print(f"Error: {MAPPING_PATH} not found.")
        exit()

    model = SimpleCNN(num_classes=num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    cap = cv2.VideoCapture(0)
    roi_x, roi_y, roi_w, roi_h = 300, 100, 250, 250

    # --- Prediction Smoothing Variables ---
    # Create a deque to store the last 15 predictions
    prediction_history = deque(maxlen=15)
    stable_prediction = "No gesture detected"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        img_tensor = transform(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = idx_to_class[predicted_idx.item()]
            confidence_score = confidence.item() * 100

        # --- Smoothing Logic ---
        if confidence_score > 70: # Use a slightly higher confidence for stability
            prediction_history.append(predicted_class)
        else:
            # If confidence is low, start clearing the history
            if len(prediction_history) > 0:
                prediction_history.popleft()

        # Check for a stable prediction only when the history is full
        if len(prediction_history) == 15:
            # Find the most common prediction in the history
            most_common = max(set(prediction_history), key=prediction_history.count)
            # Check if this prediction is dominant (e.g., appears > 50% of the time)
            if prediction_history.count(most_common) > 7:
                stable_prediction = most_common
        
        # If history becomes empty, reset the stable prediction
        if len(prediction_history) == 0:
            stable_prediction = "No gesture detected"

        # Display the stable prediction on the screen
        cv2.putText(frame, f"Prediction: {stable_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
