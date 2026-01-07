# train_yolov11.py
from ultralytics import YOLO
import os
import argparse

# ---------------- ARGUMENTS ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--project_name", required=True, help="Name of the project/job")
parser.add_argument("--dataset_yaml", default="/home/unparallel/Desktop/meshroom_server/dataset.yaml", help="Path to dataset YAML")
parser.add_argument("--model_save_dir", default="/home/unparallel/Desktop/meshroom_server/yolov11_runs", help="Directory to save the trained model")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--img_size", type=int, default=640, help="Image size for training")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
parser.add_argument("--device", default="cuda", help="Device to train on ('cuda' or 'cpu')")
args = parser.parse_args()

# ---------------- CONFIG ---------------- #
DATASET_YAML = args.dataset_yaml
MODEL_SAVE_DIR = args.model_save_dir
EPOCHS = args.epochs
IMG_SIZE = args.img_size
BATCH_SIZE = args.batch_size
DEVICE = args.device

# ---------------- SCRIPT ---------------- #
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Create YOLO object
model = YOLO("yolo11n.pt")  # Use a small pretrained model as starting point

# Train
model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    project=MODEL_SAVE_DIR,
    name=args.project_name,  # Use dynamic project name
    exist_ok=True  # overwrite if folder exists
)

print(f"✅ Training completed. Check the results in: {MODEL_SAVE_DIR}/{args.project_name}")
