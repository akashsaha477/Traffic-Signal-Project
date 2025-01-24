import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import os

# Set up paths
video_path = "/Users/akashsaha/Desktop/traffic_system/ML_Project/data/test_videos/cars.mp4"
weights_path = "../Yolo-Weights/yolov8l.pt"
mask_path = "/Users/akashsaha/Desktop/traffic_system/ML_Project/data/images/mask.png"
graphics_path = "/Users/akashsaha/Desktop/traffic_system/ML_Project/graphics.png"

# Check file existence
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found at {video_path}")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"YOLO weights file not found at {weights_path}")
if not os.path.exists(mask_path):
    raise FileNotFoundError(f"Mask image not found at {mask_path}")
if not os.path.exists(graphics_path):
    raise FileNotFoundError(f"Graphics image not found at {graphics_path}")

# Load video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Failed to open video file: {video_path}")

# Load YOLO model
model = YOLO(weights_path)

# Class names for YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter"]

# Load mask image
mask = cv2.imread(mask_path)
if mask is None:
    raise FileNotFoundError(f"Failed to load mask image from {mask_path}")

# Tracking setup
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define counting line
limits = [400, 297, 673, 297]
totalCount = []

while True:
    success, img = cap.read()
    if not success:
        print("Video ended or cannot read the frame.")
        break

    # Resize mask to match frame dimensions
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgRegion = cv2.bitwise_and(img, mask)

    # Load and overlay graphics
    imgGraphics = cv2.imread(graphics_path, cv2.IMREAD_UNCHANGED)
    if imgGraphics is not None:
        img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    else:
        print(f"Warning: Graphics image not loaded from {graphics_path}. Skipping overlay.")

    # YOLO detection
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Filter for vehicles
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update tracker
    resultsTracker = tracker.update(detections)

    # Draw counting line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Draw bounding box and ID
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'ID: {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # Calculate center of the object
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Count vehicles crossing the line
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if id not in totalCount:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Display total count
    cv2.putText(img, f'Count: {len(totalCount)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Show frames
    cv2.imshow("Image", img)
    cv2.waitKey(1)

# Release resources
cap.release()
cv2.destroyAllWindows()