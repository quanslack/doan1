import torch
from ultralytics import YOLO

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO model
model = YOLO("yolov8n.pt")  # Replace 'yolov8n.pt' with your YOLO model file

# Move the model to GPU
model.to(device)

# Run inference on an image or video
results = model("C:/Users/admin/Downloads/yolo/test5.mp4", device=device, stream = True)  # Replace 'input.jpg' with your input file path




