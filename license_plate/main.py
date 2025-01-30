from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort  # Ensure you have the SORT tracker properly installed
from util import get_car, read_license_plate, write_csv  # Custom utility functions

# Dictionary to store results
results = {}

# Initialize the SORT tracker
mot_tracker = Sort()

# Load YOLO models
coco_model = YOLO('yolov8n.pt')  # COCO pre-trained model for vehicle detection
license_plate_detector = YOLO('/Users/akashsaha/Desktop/traffic_system/license_plate/models/license_plate_detector.pt')  # Update this path as needed

# Load the video
cap = cv2.VideoCapture('/Users/akashsaha/Desktop/traffic_system/license_plate//sample.mp4')
if not cap.isOpened():
    raise FileNotFoundError("Could not open the video file './sample.mp4'. Check the path.")

# Define COCO class IDs for vehicles (car, bus, truck, motorcycle, etc.)
vehicles = [2, 3, 5, 7]  # Vehicle classes (e.g., car=2, motorcycle=3, etc.)

# Frame processing
frame_nmr = -1
while True:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break  # Exit if there are no more frames

    results[frame_nmr] = {}

    # **1. Detect vehicles using the COCO model**
    detections = coco_model(frame)[0]
    detections_ = []  # Store vehicle detections

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # **2. Track vehicles**
    track_ids = mot_tracker.update(np.asarray(detections_))  # Update tracker with detections

    # **3. Detect license plates**
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, _ = license_plate  # Ignore the class ID for license plates

        # **4. Assign license plate to vehicle**
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:  # If the license plate is successfully assigned to a vehicle
            # Crop the license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            if license_plate_crop.size == 0:
                continue  # Skip invalid crops

            # Process the license plate (grayscale and thresholding)
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(
                license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
            )

            # **5. Read the license plate number**
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score,
                    },
                }

# **6. Write results to a CSV file**
write_csv(results, './test.csv')
print("Results written to './test.csv'.")