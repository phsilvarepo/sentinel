# Sentinel – Synthetic Data & Validation Server

Sentinel is a FastAPI-based backend service designed to integrate with NVIDIA Isaac Sim, Meshroom, and a machine learning pipeline powered by Ultralytics, TensorFlow, and MinIO.

The system enables synthetic data validation, USD replicator testing, and object detection workflows while storing generated data in an S3-compatible object storage (MinIO).

---

## Overview

Sentinel acts as a bridge between simulation environments and machine learning services by:

- Validating USD Replicator outputs from Isaac Sim
- Running ML inference with Ultralytics & TensorFlow
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

Verified example:

```bash
python3 --version
# Python 3.12.3
