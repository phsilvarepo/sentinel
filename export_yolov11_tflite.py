# export_and_upload_yolov11.py
from ultralytics import YOLO
import os
import boto3

# ---------------- CONFIG ---------------- #
MODEL_PATH = "/home/unparallel/Desktop/meshroom_server/yolov11_runs/meshroom_dataset/weights/best.pt"
EXPORT_DIR = "/home/unparallel/Desktop/meshroom_server/yolov11_tflite"
EXPORT_NAME = "meshroom_tflite"

# MinIO config
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "sentinel"
OBJECT_NAME = f"meshroom/{EXPORT_NAME}.tflite"

os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------------- EXPORT MODEL ---------------- #
print("⏳ Exporting YOLOv11 model to TFLite...")
model = YOLO(MODEL_PATH)
model.export(
    format="tflite",
    imgsz=640,
    half=True,
    project=EXPORT_DIR,
    name=EXPORT_NAME
)

tflite_path = (
    "/home/unparallel/Desktop/meshroom_server/"
    "yolov11_runs/meshroom_dataset/weights/"
    "best_saved_model/best_float16.tflite"
)
if not os.path.exists(tflite_path):
    raise FileNotFoundError(f"TFLite export failed, file not found at {tflite_path}")

print("✅ TFLite model exported to:", tflite_path)

# ---------------- UPLOAD TO MINIO ---------------- #
print("⏳ Uploading TFLite model to MinIO...")
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
