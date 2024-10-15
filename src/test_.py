from ultralytics import YOLO

# Load a pretrained model
model = YOLO("/home/sai/drone_ws/src/object_detect/models/epochs_5_combined.pt")

# Train the model
results = model("/home/sai/drone_ws/src/object_detect/test_vids/Crossroads_1.mp4", show=True, save=True)