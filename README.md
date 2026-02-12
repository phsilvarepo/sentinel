# Sentinel – Synthetic Data & Validation Server

Sentinel is a FastAPI-based backend service designed to integrate with NVIDIA Isaac Sim, Meshroom, and YOLO. The system describes a pipeline to enable the detection of custom objects using the YOLO architecture. To utilize the pipeline, an Android app is necessary that connects to the server running Sentinel on the same network. To begin usage, the app transmits several images of the desired object in order to create the 3D model. Furthermore, with this model, we employ Isaac Sim synthetic generation capabilities to generate a dataset to train the object detection architecture. The final step of the pipeline consists of storing the weights in MinIO to enable the user to perform validation using the Sentinel app on live camera, images, or videos.

---

## Overview

Sentinel acts as a bridge between simulation environments and machine learning services by:

- Generating 3D model of custom objects with Meshroom
- Producing a synthetic annotated dataset using Isaac Sim
- Training the model using Ultralytics & TensorFlow
- Managing data storage through MinIO (S3-compatible)
- Exposing REST APIs using FastAPI
- Serving the application via Uvicorn

---

## Requirements

### Core Software

- Isaac Sim
- Meshroom
- MinIO (S3-compatible storage)
- Python < 3.13 (Recommended: 3.12.x)

## Usage

Creation of virtual environment:
```
python3 -m venv venv
```
Source environment:
```
source venv/bin/activate
```
Install dependencies:
```
pip install fastapi uvicorn numpy ultralytics
```
Run server:
```
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

In parallel it is necessary for MinIO to be running:
```
minio server ~/minio --console-address :9001
```

