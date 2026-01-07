import os
import subprocess
import uuid
from typing import List, Optional
from enum import Enum
from threading import Lock
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from minio import Minio
from minio.error import S3Error
from datetime import timedelta, datetime

# ---------------- CONFIG ---------------- #
MESHROOM_BIN = "/home/unparallel/Downloads/Meshroom-2025.1.0/meshroom_batch"
ISAAC_ROOT = "/home/unparallel/isaacsim/_build/linux-x86_64/release"
ISAAC_PYTHON = f"{ISAAC_ROOT}/python.sh"
USD_CONVERTER = "standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py"
REPLICATOR_SCRIPT = "standalone_examples/replicator/object_based_sdg/object_based_sdg.py"
DATASET_POSTPROCESS_SCRIPT = "/home/unparallel/Desktop/meshroom_server/dataset_postprocess.py"
REPLICATOR_CONFIG = "standalone_examples/replicator/object_based_sdg/config/meshroom.yaml"
TRAIN_SCRIPT = "/home/unparallel/Desktop/meshroom_server/train_yolov11.py"
EXPORT_SCRIPT = "/home/unparallel/Desktop/meshroom_server/export_yolov11_tflite.py"
UPLOAD_DIR = "/home/unparallel/Desktop/meshroom_server/uploads"
OUTPUT_DIR = "/home/unparallel/Desktop/meshroom_server/meshroom_output"
DATASET_DIR="/home/unparallel/isaacsim/_build/linux-x86_64/release/bbox"
OUTPUT_DATASET_DIR="/home/unparallel/Desktop/meshroom_server/dataset/"
PIPELINE_PATH = "/home/unparallel/Desktop/meshroom_server/obj_pipeline.mg"

MINIO_ENDPOINT = "http://10.0.1.140:9000"
BUCKET_NAME = "sentinel"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- FASTAPI ---------------- #
app = FastAPI()
jobs_lock = Lock()

# ---------------- MINIO ---------------- #
minio_client = Minio(
    "10.0.1.140:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

# ---------------- JOB MANAGEMENT ---------------- #
class JobStage(str, Enum):
    UPLOADING = "Uploading Images"
    MESHROOM = "Running Meshroom"
    USD = "Converting to USD"
    REPLICATOR = "Running Replicator"
    POSTPROCESSING = "Postprocessing Dataset"
    TRAINING = "Training YOLOv11"
    EXPORT = "Exporting TFLite"
    DONE = "Completed"
    FAILED = "Failed"

jobs = {}

def update_job(job_id, stage: JobStage):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["stage"] = stage

# ---------------- PIPELINE ---------------- #
def run_full_pipeline(job_id: str):
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    # Use a simplified timestamp for the weight file
    timestamp = datetime.now().strftime("%Y%m%d_%H")
    tflite_name = f"{job_id}_{timestamp}.tflite"
    object_name = f"meshroom/{tflite_name}"

    try:
        #update_job(job_id, JobStage.MESHROOM)
        subprocess.check_call([MESHROOM_BIN, "--input", job_upload_dir, "--output", job_output_dir])
        #subprocess.check_call([MESHROOM_BIN, "--input", job_upload_dir, "--output", job_output_dir, "--pipeline", PIPELINE_PATH])

        obj_path = os.path.join(job_output_dir, "texturedMesh.obj")
        if not os.path.exists(obj_path):
            update_job(job_id, JobStage.FAILED)
            return

        update_job(job_id, JobStage.USD)
        subprocess.check_call([ISAAC_PYTHON, USD_CONVERTER, "--folders", job_output_dir], cwd=ISAAC_ROOT)

        update_job(job_id, JobStage.REPLICATOR)

        usd_file = os.path.join(job_output_dir + "_converted", "texturedMesh_obj.usd")
        job_yaml = generate_job_yaml(job_id, usd_file)

        subprocess.check_call([ISAAC_PYTHON, REPLICATOR_SCRIPT, "--config", job_yaml], cwd=ISAAC_ROOT)

        update_job(job_id, JobStage.POSTPROCESSING)
        subprocess.check_call(["python3", DATASET_POSTPROCESS_SCRIPT, "--dataset_dir", DATASET_DIR, "--output_dir", OUTPUT_DATASET_DIR])

        update_job(job_id, JobStage.TRAINING)
        subprocess.check_call(["python3", TRAIN_SCRIPT, "--project_name", job_id])

        update_job(job_id, JobStage.EXPORT)
        subprocess.check_call(["python3", EXPORT_SCRIPT, "--project_name", job_id, "--tflite_name", tflite_name])

        # Build the URL for the uploaded TFLite model
        tflite_url = f"{MINIO_ENDPOINT}/{BUCKET_NAME}/{object_name}"
        with jobs_lock:
            jobs[job_id]["tflite_url"] = tflite_url

        update_job(job_id, JobStage.DONE)

    except subprocess.CalledProcessError as e:
        update_job(job_id, JobStage.FAILED)


# ---------------- API ENDPOINTS ---------------- #
@app.post("/upload")
def upload_images(
    images: List[UploadFile] = File(...),
    job_id: Optional[str] = None,
):
    if not job_id:
        job_id = str(uuid.uuid4())

    # Ensure the job exists
    with jobs_lock:
        if job_id not in jobs:
            jobs[job_id] = {"stage": JobStage.UPLOADING, "files": [], "tflite_url": None}

    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)

    for img in images:
        filename = os.path.basename(img.filename)
        if not filename:
            continue
        path = os.path.join(job_upload_dir, filename)
        with open(path, "wb") as f:
            f.write(img.file.read())
        with jobs_lock:
            jobs[job_id]["files"].append(path)

    return {"status": "ok", "job_id": job_id, "message": "Batch uploaded successfully."}


@app.post("/start_pipeline/{job_id}")
def start_pipeline(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    background_tasks.add_task(run_full_pipeline, job_id)
    return {"status": "ok", "job_id": job_id, "message": "Pipeline started."}

@app.get("/status/{job_id}")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "stage": job["stage"],
        "tflite_url": job.get("tflite_url")
    }

@app.get("/models")
def list_models():
    try:
        objects = minio_client.list_objects(
            BUCKET_NAME,
            prefix="meshroom/",
            recursive=True
        )
        models = [
            obj.object_name.replace("meshroom/", "")
            for obj in objects
            if obj.object_name.endswith(".tflite")
        ]
        return {"models": models}
    except S3Error as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-url/{model_name}")
def get_model_url(model_name: str):
    object_name = f"meshroom/{model_name}"
    try:
        url = minio_client.presigned_get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            expires=timedelta(minutes=10)
        )
        return {"url": url}
    except S3Error:
        raise HTTPException(status_code=404, detail="Model not found")

def generate_job_yaml(job_id: str, main_usd_path: str) -> str:
    """
    Generate a job-specific YAML for Isaac Sim with the main USD dynamically set.
    """
    # Template file path
    template_path = "/home/unparallel/Desktop/meshroom_server/custom_template.yaml"
    # Job-specific YAML path
    job_yaml_path = f"/home/unparallel/Desktop/meshroom_server/custom_{job_id}.yaml"

    # Read the template YAML
    with open(template_path, "r") as f:
        yaml_text = f.read()

    # Replace placeholder with actual USD path
    yaml_text = yaml_text.replace("{main_usd_path}", main_usd_path)

    # Save the dynamically generated YAML
    with open(job_yaml_path, "w") as f:
        f.write(yaml_text)

    return job_yaml_path
