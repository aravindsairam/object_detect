from ultralytics import YOLO

# Load a pretrained model
model = YOLO("/home/sai/drone_ws/src/object_detect/runs/detect/train4/weights/best.pt")

# Train the model
results = model("/home/sai/drone_ws/src/object_detect/data/VisDrone/Daytime city traffic aerial view Drone.mp4", show=True, save=True)