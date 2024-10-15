from ultralytics import YOLO
import os

# Load a pretrained model
model = YOLO("yolov8s.pt")

cfg_path = "/home/sai/drone_ws/src/object_detect/cfg"

# Train the model
results = model.train(data=os.path.join(cfg_path,"VisDrone.yaml"), epochs=5, imgsz=640, patience = 10, batch=16)