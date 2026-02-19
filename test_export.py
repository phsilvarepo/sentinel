import os
from datetime import datetime, timedelta
from minio import Minio
from minio.error import S3Error
from ultralytics import YOLO

# ---------------- CONFIG ---------------- #

PROJECT_NAME = "bugaaaa"
WEIGHTS_PATH = "/home/unparallel/Desktop/meshroom_server/yolov11_runs/bugaaaa/weights/best.pt"
EXPORT_DIR = "/home/unparallel/Desktop/meshroom_server/test_exports"
BUCKET_NAME = "sentinel"
MINIO_ENDPOINT = "10.0.1.166:9000"

ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"

# ---------------- INIT ---------------- #

os.makedirs(EXPORT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tflite_name = f"{PROJECT_NAME}_{timestamp}"
object_name = f"meshroom/{tflite_name}.tflite"

print(f"\n⏳ Exporting {WEIGHTS_PATH} to TFLite...")

# ---------------- EXPORT ---------------- #

model = YOLO(WEIGHTS_PATH)

exported_path = model.export(
    format="tflite",
    imgsz=640,
    optimize=False,
    int8=False,
    dynamic=False,
    simplify=True,
    project=EXPORT_DIR,
    name=tflite_name
)

print(f"✅ Exported model to: {exported_path}")

# ---------------- MINIO UPLOAD ---------------- #

print("\n⏳ Uploading to MinIO...")

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False
)

# Create bucket if not exists
if not minio_client.bucket_exists(BUCKET_NAME):
    minio_client.make_bucket(BUCKET_NAME)

try:
    minio_client.fput_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        file_path=exported_path,
        content_type="application/octet-stream"
    )

    print("✅ Upload successful!")

    # Generate presigned URL
    url = minio_client.presigned_get_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        expires=timedelta(hours=1)
    )

    print("\n🔗 Download URL (valid 1 hour):")
    print(url)

except S3Error as e:
    print("❌ MinIO Error:", e)
