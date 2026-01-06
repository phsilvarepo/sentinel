import os
import subprocess
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from dataset_postprocess import (
    convert_npy_to_yolo_txt,
    split_yolo_dataset
)

# ---------------- CONFIG ---------------- #

MESHROOM_BIN = "/home/unparallel/Downloads/Meshroom-2025.1.0/meshroom_batch"

ISAAC_ROOT = "/home/unparallel/isaacsim/_build/linux-x86_64/release"
ISAAC_PYTHON = f"{ISAAC_ROOT}/python.sh"

USD_CONVERTER = (
    "standalone_examples/api/omni.kit.asset_converter/"
    "asset_usd_converter.py"
)

REPLICATOR_SCRIPT = (
    "standalone_examples/replicator/object_based_sdg/"
    "object_based_sdg.py"
)

REPLICATOR_CONFIG = (
    "standalone_examples/replicator/object_based_sdg/"
    "config/meshroom.yaml"
)

UPLOAD_DIR = "/home/unparallel/Desktop/meshroom_server/uploads"
OUTPUT_DIR = "/home/unparallel/Desktop/meshroom_server/meshroom_output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


MINIO_ENDPOINT = "http://10.0.1.140:9000"
BUCKET_NAME = "sentinel"

EXPORT_NAME = "meshroom_tflite"
OBJECT_NAME = f"meshroom/{EXPORT_NAME}.tflite"

# ---------------------------------------- #

app = FastAPI()


# ---------------- API ---------------- #

@app.post("/upload")
def upload_images(images: List[UploadFile] = File(...)):
    if not images:
        raise HTTPException(status_code=400, detail="No images received")

    # 1. Save images exactly as received
    for img in images:
        filename = os.path.basename(img.filename)
        if not filename:
            continue

        path = os.path.join(UPLOAD_DIR, filename)

        with open(path, "wb") as f:
            f.write(img.file.read())

    # 2. Run Meshroom (BLOCKING)
    try:
        subprocess.check_call([
            MESHROOM_BIN,
            "--input", UPLOAD_DIR,
            "--output", OUTPUT_DIR
        ])
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail="Meshroom failed")

    # Meshroom has FINISHED here
    obj_path = os.path.join(OUTPUT_DIR, "texturedMesh.obj")

    if not os.path.exists(obj_path):
        raise HTTPException(
            status_code=500,
            detail=f"Meshroom finished but OBJ not found at {obj_path}"
        )

    # 3. Run USD converter (BLOCKING)
    try:
        subprocess.check_call(
            [
                ISAAC_PYTHON,
                USD_CONVERTER,
                "--folders", OUTPUT_DIR
            ],
            cwd=ISAAC_ROOT
        )
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="USD conversion failed")

    # 4. Run Replicator (BLOCKING)
    try:
        subprocess.check_call(
            [
                ISAAC_PYTHON,
                REPLICATOR_SCRIPT,
                "--config", REPLICATOR_CONFIG
            ],
            cwd=ISAAC_ROOT
        )
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Replicator failed")

    # 5. Train
    TRAIN_SCRIPT = "/home/unparallel/Desktop/meshroom_server/train_yolov11.py"
    try:
        subprocess.check_call(["python3", TRAIN_SCRIPT])
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="YOLOv11 train failed")

    # 6. Export & upload YOLOv11 to TFLite
    EXPORT_SCRIPT = "/home/unparallel/Desktop/meshroom_server/export_yolov11_tflite.py"
    try:
        subprocess.check_call(["python3", EXPORT_SCRIPT])
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="YOLOv11 export/upload failed")

    return {
        "status": "ok",
        "model_id": EXPORT_NAME,
        "tflite_url": f"{MINIO_ENDPOINT}/{BUCKET_NAME}/{OBJECT_NAME}",
        "input_size": 640,
        "classes": 1,
        "quantization": "float16"
    }

    
@app.post("/test-usd-replicator")
def test_usd_and_replicator():

    # 6. Export & upload YOLOv11 to TFLite
    EXPORT_SCRIPT = "/home/unparallel/Desktop/meshroom_server/export_yolov11_tflite.py"
    try:
        subprocess.check_call(["python3", EXPORT_SCRIPT])
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="YOLOv11 export/upload failed")

    return {
        "status": "ok",
        "model_id": EXPORT_NAME,
        "tflite_url": f"{MINIO_ENDPOINT}/{BUCKET_NAME}/{OBJECT_NAME}",
        "input_size": 640,
        "classes": 1,
        "quantization": "float16"
    }

