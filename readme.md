# 🎥🚀 VisionTrack-YOLO

### 🔥 Real-Time Object Detection & Tracking using YOLO + BoT-SORT

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=28&duration=3000&color=00FFAA&center=true&vCenter=true&width=900&lines=Real-Time+Object+Detection;Multi-Class+Tracking+System;YOLO+%2B+BoT-SORT+Pipeline;AI+Powered+Computer+Vision" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/YOLO-Object%20Detection-red?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/BoT--SORT-Tracking-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Computer%20Vision-AI-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Research%20Project-purple?style=for-the-badge" />
</p>

---

## 🧠 Project Overview

**VisionTrack-YOLO** is an AI-powered real-time object detection and multi-object tracking system designed for intelligent video analysis using deep learning and computer vision techniques.

The system leverages a trained YOLO model combined with BoT-SORT tracking to detect, classify, and track multiple objects across video frames with high accuracy and efficiency.

---

## 🎯 Key Features

🎥 Real-Time Video Object Detection
📦 Multi-Class Object Recognition
🧭 Advanced Multi-Object Tracking (BoT-SORT)
🧠 Deep Learning-Based Vision Pipeline
📊 Automatic Prediction Logging & Results Saving
⚡ High-Speed Inference using Ultralytics YOLO
📁 Dataset Integration with Annotated Infra Dataset

---

## ⚙️ System Pipeline

```
Input Video / Dataset
        ↓
YOLO Model Inference (Detection)
        ↓
BoT-SORT Tracker (Object Tracking)
        ↓
Bounding Box + Class Labels
        ↓
Saved Predictions & Tracking Results
```

---

## 🛠️ Tech Stack

* 🐍 Python
* 🤖 Ultralytics YOLO (Deep Learning Model)
* 🎯 BoT-SORT Tracker
* 📊 OpenCV (Video Processing)
* 🧠 Computer Vision & Deep Learning
* 📁 Custom Annotated Dataset (Infra Classes)

---

## 📂 Project Workflow

1. Load trained YOLO model weights
2. Extract dataset and training runs
3. Perform object detection on video
4. Apply BoT-SORT tracking for multi-object tracking
5. Save annotated output video and detection logs

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/VisionTrack-YOLO.git
cd VisionTrack-YOLO
```

### 2️⃣ Install Dependencies

```bash
pip install ultralytics
```

### 3️⃣ Run Detection & Tracking

```bash
yolo track \
  model="weights/best.pt" \
  source="input_video.mp4" \
  conf=0.25 \
  save=True \
  tracker="botsort.yaml"
```

---

## 📸 Output Results

* 🎯 Tracked Objects with Bounding Boxes
* 🧾 Detection Text Files
* 🎥 Processed Output Video
* 📊 Prediction Logs

(Add your output screenshots/GIF here for better GitHub reach)

---

## 🔬 Use Cases

* Smart Surveillance Systems
* Traffic Monitoring
* Infrastructure Inspection
* AI Video Analytics
* Security & Defense Applications

---

## 📈 Model Capabilities

* Multi-Class Detection
* Infrared Dataset Support
* Robust Tracking in Dynamic Scenes
* High Accuracy Object Localization

---

## 🧪 Future Improvements

* Real-Time Webcam Integration
* Web Dashboard (Streamlit/Gradio)
* Edge Device Deployment (Jetson / Raspberry Pi)
* Model Optimization (TensorRT)
* Live Alert System for Detected Objects

---

## 👨‍💻 Author

**Atharva**
M.Tech CSE | AI | Computer Vision | Deep Learning

---

## ⭐ Final Note

This project demonstrates a complete end-to-end AI vision pipeline combining deep learning detection and advanced tracking algorithms for real-world intelligent video analytics.
