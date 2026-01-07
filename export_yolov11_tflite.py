# export_yolov11_tflite.py
from ultralytics import YOLO
import os
import boto3
from datetime import datetime
import argparse

# ---------------- ARGUMENTS ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--project_name", required=True, help="Name of the project/job")
parser.add_argument("--model_path", default=None, help="Path to trained YOLOv11 .pt weights")
parser.add_argument("--export_dir", default="/home/unparallel/Desktop/meshroom_server/yolov11_tflite", help="Directory to export TFLite")
parser.add_argument("--tflite_name", default=None, help="Custom name for the TFLite model export")
args = parser.parse_args()

project_name = args.project_name
EXPORT_DIR = args.export_dir
MODEL_PATH = args.model_path or f"/home/unparallel/Desktop/meshroom_server/yolov11_runs/{project_name}/weights/best.pt"

# ---------------- MINIO CONFIG ---------------- #
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "sentinel"

# ---------------- CREATE EXPORT DIR ---------------- #
os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------------- EXPORT MODEL ---------------- #
timestamp = datetime.now().strftime("%Y%m%d_%H")
EXPORT_NAME = args.tflite_name or f"{project_name}_{timestamp}"  # Use custom tflite_name if provided

print(f"⏳ Exporting YOLOv11 model '{MODEL_PATH}' to TFLite as '{EXPORT_NAME}'...")
model = YOLO(MODEL_PATH)
model.export(
    format="tflite",
    imgsz=640,
    half=True,
    project=EXPORT_DIR,
    name=EXPORT_NAME
)

tflite_path = f"/home/unparallel/Desktop/meshroom_server/yolov11_runs/{project_name}/weights/best_saved_model/best_float16.tflite"
if not os.path.exists(tflite_path):
    raise FileNotFoundError(f"TFLite export failed, file not found at {tflite_path}")

print("✅ TFLite model exported to:", tflite_path)

# ---------------- UPLOAD TO MINIO ---------------- #
OBJECT_NAME = f"meshroom/{EXPORT_NAME}.tflite"
print(f"⏳ Uploading TFLite model to MinIO as '{OBJECT_NAME}'...")

s3 = boto3.resource(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=boto3.session.Config(signature_version='s3v4')
)

bucket = s3.Bucket(BUCKET_NAME)
bucket.upload_file(tflite_path, OBJECT_NAME)

print(f"✅ Model uploaded to MinIO bucket '{BUCKET_NAME}' as '{OBJECT_NAME}'")
print(f"s3 URL: s3://{BUCKET_NAME}/{OBJECT_NAME}")
