# Overview

Paperspace is a cloud computing platform that specializes in GPU-accelerated infrastructure, primarily targeting AI/ML workloads, creative professionals, and developers requiring GPU resources. It positions itself between traditional cloud providers (AWS, GCP, Azure) and specialized AI infrastructure providers.

Gradient is Paperspace's managed platform for building, training, and deploying machine learning models. It's their equivalent to services like AWS SageMaker or GCP Vertex AI, but with a focus on simplicity and accessibility, however with limited geographical availability (as of 2025, compared to say AWS or GCP)

As of this writing, terraform provider for paperspace doesn't seem updated for two years. Unfortunately the gradient cli which is used in most of the documents is now deprecated and the new paperspace cli seems highly disconnected with what is in the documentation. As of now, the only reliable way to create a paperspace deployment is to use the web interface.

## LatentSync on Paperspace

Instructions for setting up a Paperspace environment for deploying the [LatentSync](https://github.com/bytedance/LatentSync).

We will be using gradient deployment to deploy the LatentSync model. Gradient deployment is a managed service that allows you to deploy machine learning models as APIs or web applications. It abstracts away the underlying infrastructure, making it easier to focus on building and deploying your models.

As of now, there is no visible way to add a volume to a deployment in their deployment creation UI. Adding an integration like hugging face model directly to the deployment, which I tried, doesn't seem to work in our use case, as the code tries to download additional models, won't work as the integration creates read only volume.

Approach to make the paperspace deployment work:

1. Create a public GCP bucket for input and upload the videos and audios there and get signed URLs
2. Create a public GCP bucket for output and get signed URLs
3. Create a docker image with the LatentSync code, weights and the wrapper code


### Clone the LatentSync repository:

```bash
### Create Dockerfile and wrapper main.py

```bash
git clone https://github.com/bytedance/LatentSync.git latentsync-paperspace
# I have tested with the commit 6c8ae86ae425252ce0b33de40f666cfdd9cd760f
# follow latent
cd latentsync-paperspace
mkdir -p checkpoints/whisper
wget -O checkpoints/latentsync_unet.pt \
     https://huggingface.co/ByteDance/LatentSync-1.5/resolve/main/latentsync_unet.pt
wget -O checkpoints/whisper/tiny.pt \
     https://huggingface.co/ByteDance/LatentSync-1.5/resolve/main/whisper/tiny.pt
touch Dockerfile
touch main.py
```


### Wrapper


Add the following content to the main.py file:

```python
#!/usr/bin/env python
"""
LatentSync Paperspace Wrapper (GCS Version)

This script provides a RESTful API for LatentSync lip-syncing using GCP Storage:
1. Accept GCS paths for video and audio input
2. Process them with LatentSync
3. Output results to a specified GCS path
4. Provide a simple job-based interface for tracking

Usage:
  POST /jobs - Submit GCS paths and start processing
  GET /jobs/{job_id} - Check job status
  GET /jobs/{job_id}/log - View processing logs
"""

import os
import uuid
import time
import json
import shutil
import logging
import tempfile
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuration
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", "/app/checkpoints"))
UNET_PATH = WEIGHTS_DIR / "latentsync_unet.pt"
WHISPER_PATH = WEIGHTS_DIR / "whisper/tiny.pt"
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/app/configs/unet/stage2.yaml"))

# Setup directories
JOBS_DIR = DATA_DIR / "jobs"
LOGS_DIR = DATA_DIR / "logs"

for directory in [JOBS_DIR, LOGS_DIR, WEIGHTS_DIR / "whisper"]:
    directory.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="LatentSync API (GCS Version)",
    description="API for processing videos with LatentSync lip-sync using Google Cloud Storage",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model for GCS paths
class GcsJobRequest(BaseModel):
    video_in: str = Field(..., description="URL to input video (http:// or gs://)")
    audio_in: str = Field(..., description="URL to input audio (http:// or gs://)")
    out: str = Field(..., description="URL for output video (http:// or gs://)")
    guidance_scale: float = Field(2.0, description="Guidance scale (1.0-3.0)")
    inference_steps: int = Field(20, description="Number of inference steps (10-50)")
    seed: int = Field(0, description="Random seed (0 for random)")

@app.post("/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    job_request: GcsJobRequest = Body(...)
):
    """
    Create a new lip-sync job using URLs
    
    - **video_in**: URL to input video (http:// or gs://)
    - **audio_in**: URL to input audio (http:// or gs://)
    - **out**: URL for output video (http:// or gs://)
    - **guidance_scale**: Guidance scale parameter (1.0-3.0)
    - **inference_steps**: Number of inference steps (10-50)
    - **seed**: Random seed (0 for random)
    
    Returns job details with URL for status checking
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    logger.info(f"New job received: {job_id}")
    
    # Create job directory
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Validate URLs
        for path_name, path_value in {
            "video_in": job_request.video_in,
            "audio_in": job_request.audio_in,
            "out": job_request.out
        }.items():
            if not (path_value.startswith("http://") or path_value.startswith("https://") or path_value.startswith("gs://")):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid URL for {path_name}: {path_value}. Must start with http://, https://, or gs://"
                )
        
        # Initialize job status
        write_job_status(job_id, "processing", params={
            "video_in": job_request.video_in,
            "audio_in": job_request.audio_in,
            "out": job_request.out,
            "guidance_scale": job_request.guidance_scale,
            "inference_steps": job_request.inference_steps,
            "seed": job_request.seed,
            "created_at": datetime.now().isoformat()
        })
        
        # Queue processing job
        background_tasks.add_task(
            process_gcs_job,
            job_id,
            job_request.video_in,
            job_request.audio_in,
            job_request.out,
            job_request.guidance_scale,
            job_request.inference_steps,
            job_request.seed
        )
        
        logger.info(f"Job {job_id} queued for processing")
        
        # Extract file names from URLs
        video_filename = Path(job_request.video_in).name
        audio_filename = Path(job_request.audio_in).name
        
        # Return job information
        return {
            "job_id": job_id,
            "status": "processing",
            "urls": {
                "video_in": job_request.video_in,
                "audio_in": job_request.audio_in,
                "out": job_request.out
            },
            "parameters": {
                "guidance_scale": job_request.guidance_scale,
                "inference_steps": job_request.inference_steps,
                "seed": job_request.seed
            },
            "created_at": datetime.now().isoformat(),
            "_links": {
                "self": f"/jobs/{job_id}",
                "log": f"/jobs/{job_id}/log"
            }
        }
    
    except Exception as e:
        logger.error(f"Error setting up job {job_id}: {str(e)}")
        # Clean up
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def process_gcs_job(
    job_id: str,
    video_in: str,
    audio_in: str,
    out_path: str,
    guidance_scale: float,
    inference_steps: int,
    seed: int
):
    """Process a job with LatentSync using URLs"""
    logger.info(f"Starting processing for job {job_id}")
    job_dir = JOBS_DIR / job_id
    log_path = LOGS_DIR / f"{job_id}.log"
    
    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        video_path = temp_dir_path / "input.mp4"
        audio_path = temp_dir_path / "input.wav"
        output_path = temp_dir_path / "result.mp4"
        
        try:
            # Create log file
            with open(log_path, "w") as f:
                f.write(f"Job {job_id} started at: {datetime.now().isoformat()}\n")
                f.write(f"Parameters:\n")
                f.write(f"  - video_in: {video_in}\n")
                f.write(f"  - audio_in: {audio_in}\n")
                f.write(f"  - out: {out_path}\n")
                f.write(f"  - guidance_scale: {guidance_scale}\n")
                f.write(f"  - inference_steps: {inference_steps}\n")
                f.write(f"  - seed: {seed}\n")
                f.write(f"  - model: {UNET_PATH}\n")
                f.write(f"  - config: {CONFIG_PATH}\n")
            
            # Download files
            with open(log_path, "a") as f:
                f.write("\n=== Downloading input files ===\n")
                
                # Download video
                f.write(f"Downloading video from {video_in}\n")
                try:
                    if video_in.startswith(("http://", "https://")):
                        # Use requests for http/https URLs
                        response = requests.get(video_in, stream=True)
                        response.raise_for_status()  # Raise an exception for HTTP errors
                        
                        with open(video_path, 'wb') as vf:
                            for chunk in response.iter_content(chunk_size=8192):
                                vf.write(chunk)
                    else:
                        # Unsupported gs:// URLs without gsutil
                        raise Exception("gs:// URLs are not supported. Please use HTTP(S) signed URLs.")
                    
                    f.write(f"Downloaded video to {video_path}\n")
                except Exception as e:
                    error_msg = f"Error downloading video: {str(e)}"
                    f.write(error_msg + "\n")
                    logger.error(f"Error downloading video for job {job_id}: {str(e)}")
                    raise Exception(error_msg)
                
                # Download audio
                f.write(f"Downloading audio from {audio_in}\n")
                try:
                    if audio_in.startswith(("http://", "https://")):
                        # Use requests for http/https URLs
                        response = requests.get(audio_in, stream=True)
                        response.raise_for_status()  # Raise an exception for HTTP errors
                        
                        with open(audio_path, 'wb') as af:
                            for chunk in response.iter_content(chunk_size=8192):
                                af.write(chunk)
                    else:
                        # Unsupported gs:// URLs without gsutil
                        raise Exception("gs:// URLs are not supported. Please use HTTP(S) signed URLs.")
                    
                    f.write(f"Downloaded audio to {audio_path}\n")
                except Exception as e:
                    error_msg = f"Error downloading audio: {str(e)}"
                    f.write(error_msg + "\n")
                    logger.error(f"Error downloading audio for job {job_id}: {str(e)}")
                    raise Exception(error_msg)
                
                f.write("Files downloaded successfully\n")
            
            # Check if model weights exist
            if not UNET_PATH.exists():
                raise FileNotFoundError(f"Model weights not found at {UNET_PATH}")
            
            if not WHISPER_PATH.exists():
                raise FileNotFoundError(f"Whisper model not found at {WHISPER_PATH}")
            
            start_time = time.time()
            
            # Run LatentSync inference
            cmd = [
                "python", "-m", "scripts.inference",
                "--unet_config_path", str(CONFIG_PATH),
                "--inference_ckpt_path", str(UNET_PATH),
                "--guidance_scale", str(guidance_scale),
                "--video_path", str(video_path),
                "--audio_path", str(audio_path),
                "--video_out_path", str(output_path),
                "--seed", str(seed),
                "--inference_steps", str(inference_steps)
            ]
            
            logger.info(f"Running LatentSync for job {job_id}")
            
            # Run LatentSync and capture output
            with open(log_path, "a") as f:
                f.write("\n=== Running LatentSync ===\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                
                try:
                    process = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=True
                    )
                    f.write("LatentSync output:\n")
                    f.write(process.stdout)
                    logger.info(f"LatentSync process completed successfully for job {job_id}")
                except subprocess.CalledProcessError as e:
                    f.write(f"LatentSync failed with exit code {e.returncode}:\n")
                    f.write(e.stdout)
                    logger.error(f"LatentSync process failed for job {job_id} with exit code {e.returncode}")
                    raise Exception(f"LatentSync processing failed with exit code {e.returncode}")
            
            # Verify output file exists
            if not output_path.exists():
                raise FileNotFoundError(f"Output file not created")
            
            # Upload result
            with open(log_path, "a") as f:
                f.write("\n=== Uploading result ===\n")
                f.write(f"Uploading to {out_path}\n")
                
                try:
                    if out_path.startswith(("http://", "https://")):
                        # Upload using requests for HTTP(S) URLs
                        with open(output_path, 'rb') as out_file:
                            headers = {'Content-Type': 'video/mp4'}
                            response = requests.put(out_path, data=out_file, headers=headers)
                            response.raise_for_status()
                    else:
                        # Unsupported gs:// URLs without gsutil
                        raise Exception("gs:// URLs are not supported for upload. Please use HTTP(S) signed URLs.")
                    
                    f.write("Upload completed successfully\n")
                except Exception as e:
                    error_msg = f"Error uploading result: {str(e)}"
                    f.write(error_msg + "\n")
                    logger.error(f"Error uploading result for job {job_id}: {str(e)}")
                    raise Exception(error_msg)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update job status to completed
            write_job_status(job_id, "completed", metadata={
                "processing_time": processing_time,
                "completed_at": datetime.now().isoformat()
            })
            
            logger.info(f"Job {job_id} completed successfully in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {str(e)}")
            write_job_status(job_id, "failed", error=str(e))
            
            # Log error
            with open(log_path, "a") as f:
                f.write(f"\nERROR: {str(e)}\n")

def write_job_status(
    job_id: str, 
    status: str, 
    error: Optional[str] = None, 
    metadata: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None
):
    """Write job status to a JSON file"""
    status_file = JOBS_DIR / job_id / "status.json"
    
    # Read existing status if available
    if status_file.exists():
        with open(status_file, "r") as f:
            status_data = json.load(f)
    else:
        status_data = {
            "job_id": job_id,
            "created_at": datetime.now().isoformat()
        }
    
    # Update status
    status_data["status"] = status
    status_data["updated_at"] = datetime.now().isoformat()
    
    # Add optional fields
    if error:
        status_data["error"] = error
    
    if metadata:
        if "metadata" not in status_data:
            status_data["metadata"] = {}
        status_data["metadata"].update(metadata)
    
    if params:
        if "parameters" not in status_data:
            status_data["parameters"] = {}
        status_data["parameters"].update(params)
    
    # Add links
    status_data["_links"] = {
        "self": f"/jobs/{job_id}",
        "log": f"/jobs/{job_id}/log"
    }
    
    # Write status file
    with open(status_file, "w") as f:
        json.dump(status_data, f, indent=2)

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Get job status and details
    
    - **job_id**: ID of the job to retrieve
    
    Returns complete job information including status and URL (if completed)
    """
    # Check if job directory exists
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Get status file
    status_file = job_dir / "status.json"
    if not status_file.exists():
        return {
            "job_id": job_id,
            "status": "unknown",
            "_links": {
                "self": f"/jobs/{job_id}"
            }
        }
    
    # Read status data
    with open(status_file, "r") as f:
        status_data = json.load(f)
    
    return status_data

@app.get("/jobs/{job_id}/log")
async def get_job_log(job_id: str):
    """
    Get processing logs for a job
    
    - **job_id**: ID of the job
    
    Returns the log file containing processing details and any errors
    """
    # Check if job exists
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Check if log file exists
    log_file = LOGS_DIR / f"{job_id}.log"
    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    
    # Return the log file
    return FileResponse(
        path=log_file,
        media_type="text/plain",
        filename=f"latentsync_{job_id}.log"
    )

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns service health status and configuration details
    """
    # Check if weights exist
    weights_exist = UNET_PATH.exists() and WHISPER_PATH.exists()
    config_exists = CONFIG_PATH.exists()
    
    # No need to check for gsutil as we're using requests
    status = "healthy" if weights_exist and config_exists else "unhealthy"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "unet_weights": str(UNET_PATH.exists()),
            "whisper_weights": str(WHISPER_PATH.exists()),
            "config": str(CONFIG_PATH.exists()),
            "data_dir": str(DATA_DIR.exists())
        }
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "LatentSync API (HTTP URLs Version)",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "/jobs": "POST - Create a new lip-sync job using URLs",
            "/jobs/{job_id}": "GET - Check job status",
            "/jobs/{job_id}/log": "GET - View processing logs",
            "/health": "GET - Service health check"
        }
    }

if __name__ == "__main__":
    # When run directly, start the uvicorn server
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting LatentSync API server on port {port}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Weights directory: {WEIGHTS_DIR}")
    logger.info(f"Config path: {CONFIG_PATH}")
    
    # Check for model weights
    if not UNET_PATH.exists():
        logger.warning(f"LatentSync model weights not found at {UNET_PATH}")
    else:
        logger.info(f"Found LatentSync model weights at {UNET_PATH}")
    
    if not WHISPER_PATH.exists():
        logger.warning(f"Whisper model not found at {WHISPER_PATH}")
    else:
        logger.info(f"Found Whisper model at {WHISPER_PATH}")
    
    # Start server
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
```

### Dockerfile

Add the following content to the Dockerfile:

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0 python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .
RUN pip install fastapi==0.115.12 uvicorn[standard]==0.34.2 python-multipart==0.0.20
RUN mkdir -p /app/data/jobs /app/data/logs /app/checkpoints/whisper


ENV DATA_DIR=/app/data
ENV WEIGHTS_DIR=/app/checkpoints
ENV CONFIG_PATH=/app/configs/unet/stage2.yaml
ENV PORT=8080
EXPOSE 8080
CMD ["python", "main.py"]
```

### Build and push the docker image

```bash
docker build --no-cache -t yourrepo/latentsync-paperspace . 
docker push yourrepo/latentsync-paperspace
```

## Preparations

### Create GCP Buckets

```bash
#!/bin/bash

# Set your project ID
PROJECT_ID="your-project-id"

# Create input and output buckets
gcloud storage buckets create gs://$PROJECT_ID-latentsync-pspace-in \
    --project=$PROJECT_ID \
    --location=us-central1 \
    --uniform-bucket-level-access

gcloud storage buckets create gs://$PROJECT_ID-latentsync-pspace-out \
    --project=$PROJECT_ID \
    --location=us-central1 \
    --uniform-bucket-level-access

# Make buckets publicly readable (add allUsers as objectViewer)
gcloud storage buckets add-iam-policy-binding gs://$PROJECT_ID-latentsync-pspace-in \
    --member=allUsers \
    --role=roles/storage.objectViewer

gcloud storage buckets add-iam-policy-binding gs://$PROJECT_ID-latentsync-pspace-out \
    --member=allUsers \
    --role=roles/storage.objectViewer

# Add write permissions for all users to output bucket
gcloud storage buckets add-iam-policy-binding gs://$PROJECT_ID-latentsync-pspace-out \
    --member=allUsers \
    --role=roles/storage.objectCreator

# Add CORS configuration for the buckets
echo '[
  {
    "origin": ["*"],
    "method": ["GET", "PUT", "POST"],
    "responseHeader": ["Content-Type", "x-goog-resumable"],
    "maxAgeSeconds": 3600
  }
]' > cors.json

gcloud storage buckets update gs://$PROJECT_ID-latentsync-pspace-in \
    --cors-file=cors.json

gcloud storage buckets update gs://$PROJECT_ID-latentsync-pspace-out \
    --cors-file=cors.json

echo "Created public buckets: gs://$PROJECT_ID-latentsync-pspace-in and gs://$PROJECT_ID-latentsync-pspace-out"
```

### Upload Files to get signed URLS

```bash
#!/bin/bash

# Script to upload files to GCS and generate signed download URLs for LatentSync
# Usage: ./upload_and_get_urls.sh video_file.mp4 audio_file.wav

set -e

# Check for required arguments
if [ $# -ne 2 ]; then
  echo "Usage: $0 <video_file> <audio_file>"
  echo "Example: $0 input.mp4 input.wav"
  exit 1
fi
PROJECT_ID="your-project-id"
VIDEO_FILE="$1"
AUDIO_FILE="$2"


# Check if files exist
if [ ! -f "$VIDEO_FILE" ]; then
  echo "Error: Video file $VIDEO_FILE not found"
  exit 1
fi

if [ ! -f "$AUDIO_FILE" ]; then
  echo "Error: Audio file $AUDIO_FILE not found"
  exit 1
fi

# Set bucket names
INPUT_BUCKET="$PROJECT_ID-latentsync-pspace-in"
OUTPUT_BUCKET="$PROJECT_ID-latentsync-pspace-out"

# Generate a unique job ID
JOB_ID="job-$(date +%Y%m%d-%H%M%S)"
VIDEO_BLOB="${JOB_ID}/$(basename ${VIDEO_FILE})"
AUDIO_BLOB="${JOB_ID}/$(basename ${AUDIO_FILE})"
OUTPUT_BLOB="${JOB_ID}/output.mp4"

echo "Job ID: $JOB_ID"

# Step 1: Upload files to GCS directly
echo "Uploading video file to GCS..."
gsutil cp "${VIDEO_FILE}" "gs://${INPUT_BUCKET}/${VIDEO_BLOB}"

echo "Uploading audio file to GCS..."
gsutil cp "${AUDIO_FILE}" "gs://${INPUT_BUCKET}/${AUDIO_BLOB}"

echo "Files uploaded successfully!"

# Generate public URLs (these are not signed, but work if the bucket has public access)
VIDEO_URL="https://storage.googleapis.com/${INPUT_BUCKET}/${VIDEO_BLOB}"
AUDIO_URL="https://storage.googleapis.com/${INPUT_BUCKET}/${AUDIO_BLOB}"
OUTPUT_URL="https://storage.googleapis.com/${OUTPUT_BUCKET}/${OUTPUT_BLOB}"

# Create the JSON output with both GCS paths and URLs
cat << EOF
{
  "job_id": "${JOB_ID}",
  "gcs_paths": {
    "video_in": "gs://${INPUT_BUCKET}/${VIDEO_BLOB}",
    "audio_in": "gs://${INPUT_BUCKET}/${AUDIO_BLOB}",
    "out": "gs://${OUTPUT_BUCKET}/${OUTPUT_BLOB}"
  },
  "urls": {
    "video_in": "${VIDEO_URL}",
    "audio_in": "${AUDIO_URL}",
    "out": "${OUTPUT_URL}"
  }
}
EOF

echo ""
echo "Note: If your GCS buckets are not public, you'll need service account credentials to generate signed URLs."
echo "The public URLs will work only if your buckets allow public access."
echo ""
echo "To download the result after processing:"
echo "gsutil cp gs://${OUTPUT_BUCKET}/${OUTPUT_BLOB} ./output.mp4"
```

## Deploy

### Create Deployment

Simply create a deployment using the UI, select GPU, say A100, select your scaling configuration, set port to be 8080, and set the docker image to be your docker image. 

```bash
curl -X POST https://some.paperspacegradient.com/jobs   -H "Content-Type: application/json"   -d '{
    "video_in": "https://storage.googleapis.com/project-id-latentsync-pspace-in/job-20250521-094322/demo1_video.mp4",
    "audio_in": "https://storage.googleapis.com/project-id-latentsync-pspace-in/job-20250521-094322/demo1_audio.wav",
    "out": "https://storage.googleapis.com/project-id-latentsync-pspace-out/job-20250521-094322/output.mp4",
    "guidance_scale": 2.0,
    "inference_steps": 20,
    "seed": 0
  }'

  {"job_id":"ffd0de73-54a4-45f9-b8a6-af2310052b41","status":"processing","urls":{"video_in":"https://storage.googleapis.com/project-id-latentsync-pspace-in/job-20250521-094322/demo1_video.mp4","audio_in":"https://storage.googleapis.com/project-id-latentsync-pspace-in/job-20250521-094322/demo1_audio.wav","out":"https://storage.googleapis.com/project-id-latentsync-pspace-out/job-20250521-094322/output.mp4"},"parameters":{"guidance_scale":2.0,"inference_steps":20,"seed":0},"created_at":"2025-05-21T07:44:21.206160","_links":{"self":"/jobs/ffd0de73-54a4-45f9-b8a6-af2310052b41","log":"/jobs/ffd0de73-54a4-45f9-b8a6-af2310052b41/log"}}

```




