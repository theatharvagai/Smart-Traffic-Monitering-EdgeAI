# 🚦 Smart Traffic Monitoring & Adaptive Signal Control

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&size=24&duration=3000&pause=1000&color=3B82F6&center=true&vCenter=true&width=600&lines=Edge+AI+Traffic+Monitoring;Real-time+Vehicle+Tracking;Adaptive+Signal+Control;Optimized+for+NVIDIA+Jetson+Nano" alt="Typing SVG" />
</p>

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)](https://flask.palletsprojects.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.24+-orange?logo=onnx)](https://onnxruntime.ai)
[![YOLOv10](https://img.shields.io/badge/YOLOv10-Ultralytics-purple)](https://github.com/THU-MIG/yolov10)

> **MTech CSE Edge AI Project — VIT Vellore**
A full-stack real-time vehicle detection, tracking, speed estimation, and adaptive signal control system built for Edge AI deployment. Optimized for **NVIDIA Jetson Nano** using **ONNX Runtime** and high-speed multicore inference.

---

## 📸 Dashboard Preview

*Upload a traffic video → get annotated output, speed/direction data, CSV export, and live performance metrics — all on one screen.*

![Dashboard Preview 1](dashboard_1.png)

![Dashboard Preview 2](dashboard_2.png)

---

## ✨ Advanced Features

| Feature | Details |
| :--- | :--- |
| **🎯 Dual-Engine AI** | Supports both PyTorch and **ONNX Runtime** (optimized at 320px for high-speed CPU inference). |
| **🚥 Adaptive Signaling** | Recommends traffic light timings by comparing live density against historical averages. |
| **📊 Analytics DB** | Integrated **SQLite database** to track hourly traffic patterns and congestion indices. |
| **🚗 Vehicle Intelligence** | Precise tracking, speed estimation (km/h), and directional flow (Up/Down/Left/Right). |
| **🔀 Multicore Pipeline** | `ThreadPoolExecutor` ensures maximum hardware utilization across all CPU cores. |
| **🎨 Pro Dashboard** | Neon-Cyan themed UI with **ROI Spotlight Masking** for lane-specific monitoring. |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Web Interface (HTML/JS)             │
│  Upload │ Annotated Video │ Metrics │ CSV │ Log     │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP POST /upload
┌─────────────────────▼───────────────────────────────┐
│                 Flask API (app.py)                   │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │   ThreadPoolExecutor (multicore inference)   │    │
│  │   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │    │
│  │   │Frm 0 │ │Frm 1 │ │Frm 2 │ │Frm 3 │ ...  │    │
│  │   └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘      │    │
│  │      └────────┴─────────┴────────┘           │    │
│  │              ONNX Runtime / PyTorch           │    │
│  └─────────────────────────────────────────────┘    │
│                         │                            │
│  ┌──────────────────────▼──────────────────────┐    │
│  │       CentroidTracker (tracker.py)           │    │
│  │   Speed (px/frame → km/h) + Direction        │    │
│  └─────────────────────────────────────────────┘    │
│                         │                            │
│         Annotated MP4 + Detection CSV                │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/smart-traffic-edge-ai.git
cd smart-traffic-edge-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the backend
```bash
python app.py
```

---

## ⚡ Performance Results

| Metric | Value (Laptop CPU) | Value (Jetson Nano ONNX) |
|---|---|---|
| Avg Inference | ~60–80 ms/frame | ~25–40 ms/frame |
| Effective FPS | ~12–16 fps | ~25–40 fps |
| Model Size | 21 MB (.pt) | ~42 MB (.onnx) |
| Tracked Classes | Car, Truck, Bus, Motorcycle, Bicycle | Same |

---

## ⚙️ Configuration

In `app.py`, adjust these constants for your specific scene:

```python
CALIBRATION_PX_PER_METER = 8.0   # pixels that equal 1 metre — tune this!
MAX_FRAMES = 900                 # Max frames to process (30s at 30fps)
NUM_WORKERS = 2                  # Parallel worker count (optimized for Ryzen)
```

---

## 📥 Download Models

Due to GitHub's file size limits, the pre-trained weights are hosted externally. 

1. **Download** the models from [this link](YOUR_GOOGLE_DRIVE_LINK_HERE).
2. **Place** `personal_model.pt` and `personal_model.onnx` in the root directory.

*Alternatively, run the included fetch script (once you've updated the links):*
```bash
python fetch_model.py
```

---

## 👤 Author

**Atharva Gai**
MTech Computer Science & Engineering — VIT Vellore
Specialization: Edge AI & Embedded Systems

---

## 📜 License
MIT
