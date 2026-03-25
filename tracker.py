import math
from collections import OrderedDict

# Average lengths in meters for reference scale
REF_LENGTH_M = {
    "car": 4.5, "bus": 10.0, "truck": 8.0, 
    "motorcycle": 2.2, "person": 0.5, "bicycle": 1.7
}

class CentroidTracker:
    def __init__(self, max_disappeared: int = 50, max_distance: int = 100):
        self.next_id = 0
        self.objects: OrderedDict[int, dict] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid: tuple, label: str, conf: float):
        self.objects[self.next_id] = {
            "centroid": centroid,
            "label": label,
            "conf": conf,
            "speed_kmh": 0.0,
            "direction": "—",
            "history": [centroid],
            "start_point": centroid,
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, obj_id: int):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    @staticmethod
    def _euclidean(a: tuple, b: tuple) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def update(self, detections: list, fps: float = 25.0) -> list:
        # Convert bounding boxes to centroids (x1, y1, x2, y2, label, conf)
        det_data = []
        for (x1, y1, x2, y2, label, conf) in detections:
            cx, cy = int(x1 + x2) // 2, int(y1 + y2) // 2
            det_data.append((cx, cy, label, conf, x2-x1, y2-y1))

        if len(det_data) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self._snapshot()

        if len(self.objects) == 0:
            for d in det_data:
                self._register((d[0], d[1]), d[2], d[3])
            return self._snapshot()

        obj_ids = list(self.objects.keys())
        obj_centroids = [self.objects[oid]["centroid"] for oid in obj_ids]
        det_centroids = [(d[0], d[1]) for d in det_data]

        distances = []
        for oi, oc in enumerate(obj_centroids):
            for di, dc in enumerate(det_centroids):
                distances.append((self._euclidean(oc, dc), oi, di))
        distances.sort(key=lambda x: x[0])

        matched_obj = set()
        matched_det = set()

        for dist, oi, di in distances:
            if oi in matched_obj or di in matched_det:
                continue
            if dist > self.max_distance:
                break
            obj_id = obj_ids[oi]
            self._update_track(obj_id, det_data[di], fps)
            self.disappeared[obj_id] = 0
            matched_obj.add(oi)
            matched_det.add(di)

        for oi, obj_id in enumerate(obj_ids):
            if oi not in matched_obj:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

        for di, d in enumerate(det_data):
            if di not in matched_det:
                self._register((d[0], d[1]), d[2], d[3])

        return self._snapshot()

    def _update_track(self, obj_id, det, fps):
        cx, cy, label, conf, w, h = det
        track = self.objects[obj_id]
        track["centroid"] = (cx, cy)
        track["label"] = label
        track["conf"] = conf
        track["history"].append((cx, cy))
        
        if len(track["history"]) > 30:
            track["history"].pop(0)

        # Dynamic Speed Calibration using Monocular Size Estimation
        hist_len = len(track["history"])
        if hist_len >= 5:
            prev = track["history"][-5]
            dx = cx - prev[0]
            dy = cy - prev[1]
            dist_px = math.hypot(dx, dy)
            
            # The longest dimension of bounding box represents object length
            px_length = max(w, h)
            real_m = REF_LENGTH_M.get(label, 4.0)
            dyn_px_per_m = max(px_length / real_m, 2.0)
            
            dist_m = dist_px / dyn_px_per_m
            time_s = 4.0 / fps # 4 frame intervals
            raw_speed_kmh = (dist_m / time_s) * 3.6
            
            # Filter physics-defying speeds and static jitter
            raw_speed_kmh = min(raw_speed_kmh, 180.0) 
            if raw_speed_kmh < 4.0:
                raw_speed_kmh = 0.0
                
            # Exponential Moving Average for smooth readout
            track["speed_kmh"] = round((track["speed_kmh"] * 0.7) + (raw_speed_kmh * 0.3), 1)

        # Direction Detection using Total Displacement
        # We only assign a direction if the vehicle has moved at least 50 pixels
        # from its starting point to filter out jitter and stationary objects.
        if hist_len >= 2:
            start_x, start_y = track.get("start_point", track["history"][0])
            total_dx = cx - start_x
            total_dy = cy - start_y
            
            if math.hypot(total_dx, total_dy) > 50:
                if abs(total_dx) > abs(total_dy):
                    track["direction"] = "RIGHT" if total_dx > 0 else "LEFT"
                else:
                    track["direction"] = "DOWN" if total_dy > 0 else "UP"

    def _snapshot(self) -> list:
        result = []
        for obj_id, track in self.objects.items():
            result.append({
                "track_id": obj_id,
                "centroid": track["centroid"],
                "label": track["label"],
                "conf": round(track["conf"], 4),
                "speed_kmh": track["speed_kmh"],
                "direction": track["direction"],
            })
        return result
