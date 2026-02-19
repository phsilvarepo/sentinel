#!/usr/bin/env python3

import os
import subprocess
import argparse
from enum import Enum
from datetime import datetime

# ---------------- CONFIG ---------------- #
MESHROOM_BIN = "/home/unparallel/Downloads/Meshroom-2025.1.0/meshroom_batch"
ISAAC_ROOT = "/home/unparallel/isaacsim/_build/linux-x86_64/release"
ISAAC_PYTHON = f"{ISAAC_ROOT}/python.sh"
USD_CONVERTER = "standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py"
REPLICATOR_SCRIPT = "standalone_examples/replicator/object_based_sdg/object_based_sdg.py"
DATASET_POSTPROCESS_SCRIPT = "/home/unparallel/Desktop/meshroom_server/dataset_postprocess.py"
TRAIN_SCRIPT = "/home/unparallel/Desktop/meshroom_server/train_yolov11.py"
EXPORT_SCRIPT = "/home/unparallel/Desktop/meshroom_server/export_yolov11_tflite.py"

UPLOAD_DIR = "/home/unparallel/Desktop/meshroom_server/uploads"
OUTPUT_DIR = "/home/unparallel/Desktop/meshroom_server/meshroom_output"
DATASET_DIR = "/home/unparallel/isaacsim/_build/linux-x86_64/release/bbox"
OUTPUT_DATASET_DIR = "/home/unparallel/Desktop/meshroom_server/dataset/"

# ---------------- STAGES ---------------- #
class Stage(str, Enum):
    MESHROOM = "meshroom"
    USD = "usd"
    REPLICATOR = "replicator"
    POSTPROCESS = "postprocess"
    TRAIN = "train"
    EXPORT = "export"

# ---------------- YAML GENERATION ---------------- #
def generate_job_yaml(job_id: str, main_usd_path: str) -> str:
    template_path = "/home/unparallel/Desktop/meshroom_server/custom_template.yaml"
    job_yaml_path = f"/home/unparallel/Desktop/meshroom_server/custom_{job_id}.yaml"

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, "r") as f:
        yaml_text = f.read()

    yaml_text = yaml_text.replace("{main_usd_path}", main_usd_path)

    with open(job_yaml_path, "w") as f:
        f.write(yaml_text)

    return job_yaml_path

# ---------------- RESUME PIPELINE ---------------- #
def resume_pipeline(job_id: str, start_stage: Stage):
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)

    timestamp = datetime.now().strftime("%Y%m%d_%H")
    tflite_name = f"{job_id}_{timestamp}"

    print(f"\nResuming job: {job_id}")
    print(f"Starting from stage: {start_stage.value}\n")

    # -------- MESHROOM --------
    if start_stage == Stage.MESHROOM:
        subprocess.check_call([
            MESHROOM_BIN,
            "--input", job_upload_dir,
            "--output", job_output_dir
        ])
        start_stage = Stage.USD

    # -------- USD --------
    if start_stage == Stage.USD:
        subprocess.check_call(
            [ISAAC_PYTHON, USD_CONVERTER, "--folders", job_output_dir],
            cwd=ISAAC_ROOT
        )
        start_stage = Stage.REPLICATOR

    # -------- REPLICATOR --------
    if start_stage == Stage.REPLICATOR:
        usd_file = os.path.join(
            job_output_dir + "_converted",
            "texturedMesh_obj.usd"
        )

        if not os.path.exists(usd_file):
            raise FileNotFoundError(f"USD file missing: {usd_file}")

        job_yaml = generate_job_yaml(job_id, usd_file)

        subprocess.check_call(
            [ISAAC_PYTHON, REPLICATOR_SCRIPT, "--config", job_yaml],
            cwd=ISAAC_ROOT
        )
        start_stage = Stage.POSTPROCESS

    # -------- POSTPROCESS --------
    if start_stage == Stage.POSTPROCESS:
        subprocess.check_call([
            "python3",
            DATASET_POSTPROCESS_SCRIPT,
            "--dataset_dir", DATASET_DIR,
            "--output_dir", OUTPUT_DATASET_DIR
        ])
        start_stage = Stage.TRAIN

    # -------- TRAIN --------
    if start_stage == Stage.TRAIN:
        subprocess.check_call([
            "python3",
            TRAIN_SCRIPT,
            "--project_name", job_id
        ])
        start_stage = Stage.EXPORT

    # -------- EXPORT --------
    if start_stage == Stage.EXPORT:
        subprocess.check_call([
            "python3",
            EXPORT_SCRIPT,
            "--project_name", job_id,
            "--tflite_name", tflite_name
        ])

    print("\n✅ Resume pipeline completed successfully!\n")

# ---------------- CLI ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Meshroom Pipeline")
    parser.add_argument("job_id", help="Job ID to resume")
    parser.add_argument(
        "--start-from",
        required=True,
        choices=[s.value for s in Stage],
        help="Stage to resume from"
    )

    args = parser.parse_args()

    resume_pipeline(args.job_id, Stage(args.start_from))
