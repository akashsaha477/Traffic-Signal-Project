import torch
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from time import time
from pymongo import MongoClient
import csv
import argparse
import os
from sort import Sort  # Assuming the SORT implementation is in a separate `sort.py` file

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Constants
FPS = 30
PIXELS_PER_METER = 10
MIN_CONFIDENCE = 0.5
SPEED_LIMIT = 60  # km/h
LANE_BOUNDARY_X = None  # Set dynamically
VEHICLES = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck

# Get the directory of the script for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
try:
    bike_detector = YOLO(os.path.join(SCRIPT_DIR, 'yolov8n.pt')).to(device)  # Generic object detection
    helmet_detector = YOLO(os.path.join(SCRIPT_DIR, 'helmet_detection.pt')).to(device)  # Pretrained helmet detection
    license_plate_detector = YOLO(os.path.join(SCRIPT_DIR, 'license_plate_detector.pt')).to(device)  # License plate detection
    person_detector = YOLO(os.path.join(SCRIPT_DIR, 'yolov8n.pt')).to(device)  # Person detection
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Initialize OCR reader
try:
    reader = easyocr.Reader(['en'], gpu=True)
    print("OCR reader initialized")
except Exception as e:
    print(f"Error initializing OCR reader: {e}")
    raise

# MongoDB setup
def init_db():
    try:
        client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
        # Test the connection
        client.server_info()
        db = client['vehicle_data']
        return db['criminal_records']
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        print("Continuing without database functionality")
        return None

# Load criminal plates from CSV
def load_criminal_plates(file_path=None):
    if file_path is None:
        file_path = os.path.join(SCRIPT_DIR, 'criminal_plates.csv')
    
    criminal_plates = set()
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:  # Check if row is not empty
                    criminal_plates.add(row[0].strip())
        print(f"Loaded {len(criminal_plates)} criminal plates")
    except Exception as e:
        print(f"Error loading criminal plates: {e}")
        print("Continuing with empty criminal plates set")
    
    return criminal_plates

# Helper functions
def calculate_speed(prev_position, curr_position, frame_time, pixels_per_meter):
    distance_pixels = np.sqrt((curr_position[0] - prev_position[0])**2 + 
                              (curr_position[1] - prev_position[1])**2)
    distance_meters = distance_pixels / pixels_per_meter
    speed_mps = distance_meters / frame_time
    return speed_mps * 3.6  # km/h

def check_violations(center_x, speed, frame_width):
    violations = []
    if speed > SPEED_LIMIT:
        violations.append(f"Speeding: {speed:.1f} km/h (> {SPEED_LIMIT} km/h)")
    if LANE_BOUNDARY_X and (center_x < LANE_BOUNDARY_X - 50 or center_x > LANE_BOUNDARY_X + 50):
        violations.append("Lane Crossing")
    return violations

def detect_helmet(frame):
    results = helmet_detector(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = "Helmet" if class_id == 0 else "No Helmet"
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def detect_license_plate(frame):
    results = license_plate_detector(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Ensure coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Check if the crop area is valid
            if x2 <= x1 or y2 <= y1 or x1 >= frame.shape[1] or y1 >= frame.shape[0]:
                continue
                
            license_plate_crop = frame[y1:y2, x1:x2]
            
            # Check if crop is not empty
            if license_plate_crop.size == 0:
                continue
                
            try:
                detections = reader.readtext(license_plate_crop)
                for detection in detections:
                    _, text, score = detection
                    text = text.upper().replace(' ', '')
                    # Filter out short or nonsensical plates
                    if len(text) > 4 and any(c.isalnum() for c in text):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        return text
            except Exception as e:
                print(f"Error in OCR processing: {e}")
    return None

def detect_triple_riding(frame):
    bike_results = bike_detector(frame)
    person_results = person_detector(frame)
    
    # Use list comprehensions to create proper lists, not iterators
    bikes = []
    for result in bike_results:
        for box in result.boxes:
            if int(box.cls[0]) == 3:  # Motorcycle class
                coords = box.xyxy[0].cpu().numpy()
                bikes.append((int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])))
    
    persons = []
    for result in person_results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # Person class
                coords = box.xyxy[0].cpu().numpy()
                persons.append((int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])))
    
    violations = []
    for bx1, by1, bx2, by2 in bikes:
        person_count = 0
        for px1, py1, px2, py2 in persons:
            # Check if person bounding box overlaps significantly with bike
            # Using IoU (Intersection over Union) check
            x_overlap = max(0, min(bx2, px2) - max(bx1, px1))
            y_overlap = max(0, min(by2, py2) - max(by1, py1))
            
            if x_overlap > 0 and y_overlap > 0:
                overlap_area = x_overlap * y_overlap
                person_area = (px2 - px1) * (py2 - py1)
                
                # If at least 30% of the person is overlapping with the bike
                if overlap_area / person_area > 0.3:
                    person_count += 1
        
        if person_count > 2:
            violations.append((bx1, by1, bx2, by2))
    
    for bx1, by1, bx2, by2 in violations:
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.putText(frame, "Triple Riding Detected", (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

def check_criminal_record(collection, plate_number):
    if collection is None:
        return "Database not available"
    try:
        record = collection.find_one({'plate_number': plate_number})
        return record if record else "No record found"
    except Exception as e:
        print(f"Error checking criminal record: {e}")
        return "Error checking records"

def process_frame(frame, tracker, car_positions, collection, criminal_plates, frame_count, prev_time):
    global LANE_BOUNDARY_X
    if frame_count == 1:
        LANE_BOUNDARY_X = frame.shape[1] // 2

    curr_time = time()
    frame_time = curr_time - prev_time if frame_count > 1 else 1/FPS

    # Apply detections
    frame = detect_helmet(frame)
    plate_number = detect_license_plate(frame)
    frame = detect_triple_riding(frame)

    # Vehicle detection and tracking
    try:
        detections = bike_detector(frame)[0]
        
        # Check if there are any detections
        if len(detections.boxes) > 0:
            detections_ = []
            for box_data in detections.boxes.data.tolist():
                if len(box_data) >= 6 and int(box_data[5]) in VEHICLES:
                    x1, y1, x2, y2, score, _ = box_data[:6]
                    detections_.append([x1, y1, x2, y2, score])
            
            if detections_:
                track_ids = tracker.update(np.asarray(detections_))
                
                # Process tracked objects
                for track in track_ids:
                    x1, y1, x2, y2, track_id = map(int, track)
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Speed calculation
                    if track_id in car_positions:
                        prev_pos = car_positions[track_id]['position']
                        speed = calculate_speed(prev_pos, (center_x, center_y), frame_time, PIXELS_PER_METER)
                        cv2.putText(frame, f"Speed: {speed:.1f} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        violations = check_violations(center_x, speed, frame.shape[1])
                        for i, violation in enumerate(violations):
                            cv2.putText(frame, violation, (x1, y1 - 40 - i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    car_positions[track_id] = {'position': (center_x, center_y), 'time': curr_time}
            else:
                # No vehicles detected
                track_ids = tracker.update(np.empty((0, 5)))
        else:
            # No detections at all
            track_ids = tracker.update(np.empty((0, 5)))
    except Exception as e:
        print(f"Error in vehicle detection and tracking: {e}")
        track_ids = []

    # Criminal record check
    if plate_number:
        record = check_criminal_record(collection, plate_number)
        if plate_number in criminal_plates:
            print(f"Alert: Criminal vehicle detected - {plate_number}")
        print(f"License Plate: {plate_number}, Record: {record}")

    # Clean up old positions
    car_positions = {k: v for k, v in car_positions.items() if curr_time - v['time'] < 1.0}
    return frame, curr_time

def main():
    parser = argparse.ArgumentParser(description="Traffic Violation Detection System")
    parser.add_argument('--video', type=str, default='sample.mp4', help='Path to video file or 0 for webcam')
    parser.add_argument('--criminal_plates', type=str, default=None, help='Path to criminal plates CSV')
    args = parser.parse_args()

    # Initialize components
    try:
        tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    except Exception as e:
        print(f"Error initializing SORT tracker: {e}")
        print("Make sure sort.py is in the same directory")
        return

    collection = init_db()
    criminal_plates = load_criminal_plates(args.criminal_plates)
    
    # Handle video input
    video_path = args.video
    if video_path == '0':
        video_path = 0
    elif not os.path.isabs(video_path):
        video_path = os.path.join(SCRIPT_DIR, video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}.")
        return

    # Set video properties for better performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    car_positions = {}
    frame_count = 0
    prev_time = time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            frame_count += 1

            try:
                frame, prev_time = process_frame(frame, tracker, car_positions, collection, criminal_plates, frame_count, prev_time)
                if LANE_BOUNDARY_X is not None:
                    cv2.line(frame, (LANE_BOUNDARY_X, 0), (LANE_BOUNDARY_X, frame.shape[0]), (255, 0, 0), 2)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue

            cv2.imshow("Traffic Violation Detection", frame)

            # Check for quit with a small delay
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit")
                break

    except KeyboardInterrupt:
        print("Processing interrupted by user")
    except Exception as e:
        print(f"Unexpected error during video processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Processing complete")

if __name__ == "__main__":
    main()