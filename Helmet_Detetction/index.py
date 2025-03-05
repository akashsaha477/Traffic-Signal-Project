from ultralytics import YOLO

# Load the model
model = YOLO("yolov8s.pt")

# Train on MPS (GPU)
model.train(data="/Users/akashsaha/Desktop/traffic_system/Helmet_Test/3riders-2/data.yaml", epochs=1, imgsz=608)