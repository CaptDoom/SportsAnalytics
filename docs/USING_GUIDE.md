# Deployment & Using Guide - Badminton Analytics System

This guide outlines how to deploy, set up, and run the modernized single-repository Next.js and FastAPI application directly from your GitHub clone.

---

## 🚀 1. Will it work if deployed directly from GitHub?

**Yes**, with a few minor setup steps. Because large binary weights and system-specific files are ignored (via `.gitignore`), you must configure the following dependencies on your deployment server:

### A. Computer Vision Model Weights (YOLO & TrackNet)
* **YOLOv8 Models (`yolov8n.pt`, `yolov8n-pose.pt`)**: **Auto-Downloaded.** The Ultralytics library will automatically fetch these model weights from the official releases on first execution. No manual actions are required.
* **TrackNet Model (`model_best.pt`)**: **Manual Placement.** The custom TrackNet model weights must be placed in `src/TrackNet/model_best.pt`. If this file is missing, the pipeline gracefully falls back to a high-precision YOLO tracker combined with an Extended Kalman Filter (EKF) simulator to prevent crashes.

### B. System Dependencies (ffmpeg & dynamic libraries)
* The pipeline utilizes OpenCV to decode, process, and write video files. 
* **Linux Server (AWS EC2 / Ubuntu)**: Install `ffmpeg` using your package manager:
  ```bash
  sudo apt-get update
  sudo apt-get install -y ffmpeg
  ```
* **Windows Host**: Local processing relies on system DLLs (such as `openh264`). Ensure your path contains the ffmpeg binaries.

---

## 🛠️ 2. How to Run the Application

### Step 2.1: Python Backend Services Setup
The backend consists of three microservices (Ingestion, Analytics, and Coaching) sharing a unified SQLite database (`badminton.db`).

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Start Ingestion Service (Port 8001)**:
   ```bash
   python -m uvicorn src.pipeline.main:app --host 127.0.0.1 --port 8001
   ```
3. **Start Analytics Service (Port 8002)**:
   ```bash
   python -m uvicorn src.pipeline.main:app --host 127.0.0.1 --port 8002
   ```
4. **Start Coaching Service (Port 8003)**:
   ```bash
   python -m uvicorn src.pipeline.main:app --host 127.0.0.1 --port 8003
   ```

### Step 2.2: Next.js Frontend Dashboard Setup
1. **Install Node dependencies**:
   ```bash
   npm install
   ```
2. **Launch the development server**:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:3000` to interact with the dashboard.

---

## 📊 3. Verification & Testing

You can run automated verification test suites before exposing the endpoints to production:
```bash
python -m pytest src/pipeline/test_pipeline.py
```
This runs the full processing, homography calibration, and EKF simulator verification routines.
