"""
Smart Traffic Monitoring — Edge AI Backend
==========================================
Flask API that processes traffic video with YOLOv10n (or a custom ONNX model),
tracks vehicles, estimates speed and direction, and exports annotated video + CSV.

Edge AI Optimizations
---------------------
- ONNX Runtime inference (auto-detected) — targets NVIDIA Jetson Nano
- Multi-threaded frame batching via ThreadPoolExecutor (multicore CPUs)
- Lightweight centroid tracker — O(n²) matching, no heavy dependencies

Author : Atharva Gai — VIT Vellore MTech CSE | Edge AI Project
"""

import os
import cv2
import uuid
import csv
import time
import math
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from tracker import CentroidTracker
import analytics

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger("EdgeAI")

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

analytics.init_db()  # Setup database on boot

UPLOAD_FOLDER    = "uploads"
PROCESSED_FOLDER = "processed"
CSV_FOLDER       = "csv_reports"
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, CSV_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ── CALIBRATION ───────────────────────────────────────────────────────────────
# Adjust this value based on your camera / scene.
# Rough default: 8 pixels ≈ 1 metre at typical CCTV distance.
CALIBRATION_PX_PER_METER = 8.0

# ── ONNX / PyTorch Model Loading ──────────────────────────────────────────────
USE_ONNX   = False
model      = None
ort_session = None
INPUT_SIZE  = 320  # Optimized for speed

ONNX_PATH = "personal_model.onnx"
PT_PATH   = "personal_model.pt"

# Fallbacks if custom model is missing
for custom in ["custom_traffic.onnx", "custom_traffic.pt"]:
    if os.path.exists(custom):
        ONNX_PATH = custom if custom.endswith(".onnx") else ONNX_PATH
        PT_PATH   = custom if custom.endswith(".pt")   else PT_PATH
        log.info(f"Custom model detected: {custom}")
        break

if os.path.exists(ONNX_PATH):
    try:
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"] # Optimized for Ryzen CPU
        ort_session = ort.InferenceSession(ONNX_PATH, providers=providers)
        USE_ONNX = True
        log.info(f"ONNX Runtime loaded: {ONNX_PATH}  providers={ort_session.get_providers()}")
    except Exception as e:
        log.warning(f"ONNX load failed ({e}), falling back to PyTorch.")

if not USE_ONNX:
    from ultralytics import YOLO
    model = YOLO(PT_PATH)
    log.info(f"PyTorch engine active: {PT_PATH}")

# ── VEHICLE CLASSES (COCO / Custom) ──────────────────────────────────────────
# These are the classes we ALLOW from the model. Everything else is ignored.
ALLOWED_VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}
# Typical COCO IDs for bicycle, car, motorcycle, bus, truck
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7] 

# ── Thread Pool for Multicore Inference ───────────────────────────────────────
# Using 2 workers to leave overhead for video encoding/decoding on Ryzen 3/5
NUM_WORKERS = 2
executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
log.info(f"ThreadPoolExecutor: {NUM_WORKERS} workers (stable mode)")


# ── Inference Helpers ─────────────────────────────────────────────────────────

def infer_pytorch(frame):
    """Run model inference on a single frame optimized for speed."""
    # Using imgsz=320 for 4x faster CPU performance
    results = model(frame, verbose=False, conf=0.25, imgsz=320)[0]
    boxes_out = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id].lower()
        
        # STRICT FILTER: Only vehicles allowed
        if label in ALLOWED_VEHICLES:
            mapped_label = "vehicle"
        else:
            continue # Ignore person, boat, traffic light, etc.
            
        conf   = float(box.conf[0])
        xyxy   = box.xyxy[0].cpu().numpy()
        boxes_out.append((mapped_label, conf, xyxy))
    
    annotated = frame.copy()
    for label, conf, xyxy in boxes_out:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        # Draw clean 'vehicle' boxes
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (99, 102, 241), 2)
        cv2.putText(annotated, f"vehicle {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return annotated, boxes_out


def infer_onnx(frame):
    """Robust ONNX Inference with Shape Debugging and Thin Borders."""
    import numpy as np
    h, w = frame.shape[:2]
    blob = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    blob = blob[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = blob[np.newaxis, ...]

    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: blob})[0]

    # DEBUG: Print shape once to logs
    # log.info(f"ONNX Raw Output Shape: {outputs.shape}")

    # Ensure shape is [1, 2100, 84] (Detections as rows)
    # If it's [1, 84, 2100], we MUST transpose
    if outputs.shape[1] < outputs.shape[2]:
        outputs = outputs.transpose(0, 2, 1)
        
    boxes_out = []
    # 1:bicycle, 2:car, 3:motorcycle, 5:bus, 6:train, 7:truck, 8:boat
    VEHICLE_IDS = {1, 2, 3, 5, 6, 7, 8}

    for det in outputs[0]:
        # Handle standard YOLOv8/v10 [cx, cy, w, h, scores...]
        scores = det[4:]
        conf = np.max(scores)
        if conf < 0.15: continue # Slightly lower for better visibility
        
        cls_id = int(np.argmax(scores))
        if cls_id not in VEHICLE_IDS: continue
        
        cx, cy, bw, bh = det[:4]
        
        # Rescale normalized or pixel coordinates
        if cx <= 1.01: # Normalized
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
        else: # Pixel-based (320x320)
            x1 = int((cx - bw/2) * w / INPUT_SIZE)
            y1 = int((cy - bh/2) * h / INPUT_SIZE)
            x2 = int((cx + bw/2) * w / INPUT_SIZE)
            y2 = int((cy + bh/2) * h / INPUT_SIZE)
        
        boxes_out.append(("vehicle", float(conf), [x1, y1, x2, y2]))

    annotated = frame.copy()
    final_boxes = []
    
    if len(boxes_out) > 0:
        nms_rects = [[b[2][0], b[2][1], b[2][2]-b[2][0], b[2][3]-b[2][1]] for b in boxes_out]
        nms_confs = [b[1] for b in boxes_out]
        
        # NMS to clear up the "yellow dots" clutter
        indices = cv2.dnn.NMSBoxes(nms_rects, nms_confs, 0.15, 0.45)
        
        if len(indices) > 0:
            idx_list = indices.flatten() if hasattr(indices, 'flatten') else indices
            for i in idx_list:
                label, conf, coords = boxes_out[i]
                final_boxes.append((label, conf, coords))
                
                # Draw Thin Box (1px)
                x1, y1, x2, y2 = coords
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 1)
                cv2.putText(annotated, f"v {conf:.2f}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                            
    return annotated, final_boxes


def infer_frame(args):
    """Timed inference — measures ms accurately inside the thread."""
    fid, frame = args
    t0 = time.time()
    if USE_ONNX:
        annotated, boxes = infer_onnx(frame)
    else:
        annotated, boxes = infer_pytorch(frame)
    elapsed_ms = (time.time() - t0) * 1000
    return fid, annotated, boxes, elapsed_ms


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    unique_id = str(uuid.uuid4())
    filename = f"{unique_id}_{video_file.filename}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(input_path)
    log.info(f"Received video: {video_file.filename}")

    output_filename = f"processed_{filename}"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    csv_filename = f"detections_{unique_id}.csv"
    csv_path = os.path.join(CSV_FOLDER, csv_filename)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video file"}), 400

    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    FRAME_SKIP = 3  # Process 1 in 3 frames (cuts compute by 66%)
    out_fps    = fps / FRAME_SKIP
    
    # avc1 = H.264 — universally playable in browsers
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out    = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
    if not out.isOpened():
        log.warning("avc1 encoder unavailable, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    tracker = CentroidTracker(max_disappeared=40, max_distance=150)
    detections_log = []
    inference_times = []
    frame_count = 0
    MAX_FRAMES = 900  # processed limit (30 seconds at 30fps)

    # ── Region of Interest (ROI) Masking ─────────────────────────────────────
    roi_poly = None
    roi_str = request.form.get("roi")
    if roi_str and roi_str.strip() not in ["", "[]"]:
        import json
        import numpy as np
        try:
            pts_pct = json.loads(roi_str)
            if len(pts_pct) >= 3:
                pts_abs = [[int(p["x"] * width), int(p["y"] * height)] for p in pts_pct]
                roi_poly = np.array([pts_abs], dtype=np.int32)
                log.info(f"ROI Map parsed from frontend: {len(pts_pct)} points")
        except Exception as e:
            log.warning(f"Failed to parse ROI: {e}")

    mask_img = None
    if roi_poly is not None:
        mask_img = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask_img, roi_poly, 255)

    log.info(f"Processing up to {MAX_FRAMES} frames (processing 1 in {FRAME_SKIP} at {fps:.1f} fps) with {NUM_WORKERS} threads")

    # ── Read all frames first, then batch-infer with multicore ───────────────
    frames_buffer = []
    video_frames_read = 0
    while cap.isOpened() and len(frames_buffer) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        video_frames_read += 1
        if video_frames_read % FRAME_SKIP != 0:
            continue
            
        if mask_img is not None:
            frame = cv2.bitwise_and(frame, frame, mask=mask_img)
            
        frames_buffer.append((video_frames_read, frame))
        
    cap.release()

    # ── Parallel inference (multicore) ──
    results_by_frame = {}
    futures = [executor.submit(infer_frame, (fid, frm)) for fid, frm in frames_buffer]
    for future in as_completed(futures):
        try:
            fid, annotated, boxes, elapsed_ms = future.result()
        except Exception as exc:
            log.error(f"Inference failed: {exc}")
            continue
        inference_times.append(elapsed_ms)
        results_by_frame[fid] = (annotated, boxes)

    log.info("Inference done. Running tracker and writing output...")

    # ── Tracker pass ──
    for fid, frame in frames_buffer:
        if fid not in results_by_frame:
            continue
        annotated, boxes = results_by_frame[fid]

        raw_dets = []
        for (label, conf, xyxy) in boxes:
            if label not in ALLOWED_VEHICLES:
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            raw_dets.append((x1, y1, x2, y2, label, conf))

        tracked = tracker.update(raw_dets, fps=out_fps)

        for t in tracked:
            cx, cy = t["centroid"]
            speed   = t["speed_kmh"]
            dirn    = t["direction"]
            tid     = t["track_id"]
            lbl     = t["label"]

            cv2.circle(annotated, (cx, cy), 4, (0, 200, 255), -1)
            overlay_text = f"ID{tid} {lbl} {speed}km/h {dirn}"
            cv2.putText(annotated, overlay_text, (cx - 40, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1)

            detections_log.append({
                "id":           len(detections_log) + 1,
                "frame":        fid,
                "track_id":     tid,
                "class":        lbl,
                "confidence":   round(t["conf"], 4),
                "speed_kmh":    speed,
                "direction":    dirn,
                "inference_ms": round(inference_times[fid], 2) if fid < len(inference_times) else 0,
            })

        out.write(annotated)

    out.release()
    log.info(f"Output video saved: {output_filename}")

    fieldnames = ["id", "frame", "track_id", "class", "confidence", "speed_kmh", "direction", "inference_ms"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detections_log)

    avg_infer  = round(sum(inference_times) / len(inference_times), 2) if inference_times else 0
    min_infer  = round(min(inference_times), 2) if inference_times else 0
    max_infer  = round(max(inference_times), 2) if inference_times else 0
    eff_fps    = round(1000 / avg_infer, 1) if avg_infer > 0 else 0

    speeds = [d["speed_kmh"] for d in detections_log if d["speed_kmh"] > 0]
    avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0
    max_speed = round(max(speeds), 1) if speeds else 0

    # Count UNIQUE tracked objects
    unique_tracks = {}
    direction_map = {}
    for d in detections_log:
        tid = d["track_id"]
        unique_tracks[tid] = d["class"]
        if d["direction"] != "—":
            direction_map[tid] = d["direction"]

    # ── Final Metrics Processing ──
    class_counts = {}
    for cls in unique_tracks.values():
        class_counts[cls] = class_counts.get(cls, 0) + 1

    direction_counts = {}
    for dirn in direction_map.values():
        direction_counts[dirn] = direction_counts.get(dirn, 0) + 1

    analytics.save_metrics(direction_counts, avg_speed)
    signal_recommendations = analytics.get_signal_recommendation(direction_counts)

    return jsonify({
        "video_url": f"http://localhost:5000/processed/{output_filename}",
        "csv_url":   f"http://localhost:5000/csv/{csv_filename}",
        "detections": detections_log[:250],
        "metrics": {
            "total_frames":       video_frames_read,
            "processed_frames":   len(frames_buffer),
            "total_detections":   len(unique_tracks), # Use actual unique count
            "avg_inference_ms":   avg_infer,
            "min_inference_ms":   min_infer,
            "max_infer_ms":       max_infer,
            "effective_fps":      eff_fps,
            "avg_speed_kmh":      avg_speed,
            "max_speed_kmh":      max_speed,
            "class_counts":       class_counts,
            "direction_counts":   direction_counts,
            "backend":            "Custom ONNX Model" if USE_ONNX else "Custom PyTorch Model",
            "workers":            NUM_WORKERS,
        },
        "signals": signal_recommendations
    })


@app.route("/processed/<filename>")
def get_processed_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


@app.route("/csv/<filename>")
def get_csv(filename):
    return send_from_directory(CSV_FOLDER, filename)


if __name__ == "__main__":
    log.info("Starting Edge AI Traffic Monitor API on port 5000")
    app.run(port=5000, debug=True, threaded=True, use_reloader=False)
