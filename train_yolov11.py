# train_yolov11.py
from ultralytics import YOLO
import os

# ---------------- CONFIG ---------------- #
DATASET_YAML = "/home/unparallel/Desktop/meshroom_server/dataset.yaml"
MODEL_SAVE_DIR = "/home/unparallel/Desktop/meshroom_server/yolov11_runs"
EPOCHS = 20
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = "cuda"  # use "cpu" if no GPU

# ---------------- SCRIPT ---------------- #
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Create YOLO object
model = YOLO("yolov8n.pt")  # Use a small pretrained model as starting point

# Train
model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    project=MODEL_SAVE_DIR,
    name="meshroom_dataset",
    exist_ok=True  # overwrite if folder exists
)

print("✅ Training completed. Check the results in:", MODEL_SAVE_DIR)
