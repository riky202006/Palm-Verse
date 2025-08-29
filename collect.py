# main.py
# Import necessary libraries
import cv2
import os
import numpy as np

# Define the directory to save the datasets
DATA_DIR = './DATASETS'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

print("Starting data collector. Press a letter (A-Z) to save. Press 'q' to quit.")
print("Place your hand in the green box.")

# Start video capture from the webcam (device 0)
cap = cv2.VideoCapture(0)

# --- Configuration ---
# Set the region of interest (ROI) for the hand
roi_x, roi_y, roi_w, roi_h = 300, 100, 250, 250

# --- Debugging Variables ---
last_key_pressed = "None"
hand_status = "NO HAND DETECTED"

# --- Main Loop ---
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a selfie-view
    frame = cv2.flip(frame, 1)

    # Draw the ROI on the main frame for guidance
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

    # --- Hand Detection and Segmentation in ROI ---
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # --- Image Cleaning ---
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    
    # --- Check Hand Status for Debugging ---
    if cv2.countNonZero(mask) > 500: # Check if a significant portion of the mask is white
        hand_status = "HAND DETECTED"
    else:
        hand_status = "NO HAND DETECTED"

    # --- Wait for a key press ---
    key = cv2.waitKey(1) & 0xFF

    if key != 255: # 255 is the value when no key is pressed
        last_key_pressed = chr(key)

    # --- Save Image Logic ---
    # If a letter key (A-Z) is pressed AND a hand is detected
    if 65 <= key <= 90 and hand_status == "HAND DETECTED":
        char_pressed = chr(key)
        print(f"'{char_pressed}' key pressed. Saving image...")

        class_dir = os.path.join(DATA_DIR, char_pressed)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        img_counter = len(os.listdir(class_dir))
        img_name = f"{char_pressed}_{img_counter}.png"
        img_path = os.path.join(class_dir, img_name)

        cv2.imwrite(img_path, mask)
        print(f"Successfully saved {img_name} to {class_dir}")

    # --- Display Debugging Info On Screen ---
    cv2.putText(frame, f"Last Key: {last_key_pressed}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Status: {hand_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frames
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)

    # If 'q' is pressed, break the loop
    if key == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Data collection stopped.")
