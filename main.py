# people_counter_with_alerts.py
"""
People Counter with Entry/Exit + Burst & Occupancy Alerts
---------------------------------------------------------
- Video source: RTSP/IP cam, file path, or webcam index.
- Person detection: YOLOv8 (ultralytics).
- Tracking: lightweight centroid tracker (no extra deps).
- Line crossing: horizontal or vertical virtual line.
- Alerts:
    1) Burst alert: N entries within T seconds.
    2) Occupancy alert: people inside > OCCUPANCY_LIMIT.
    3) Optional: Exit burst, rate limits to avoid spam.

Notes:
- Install requirements from requirements.txt
- Run: python people_counter_with_alerts.py --source 0
- Change config by CLI flags or editing defaults below.
"""

import argparse
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, List

import cv2
import numpy as np

# Try to import YOLO (ultralytics). If not available, raise a clear error.
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None


# -------------------------------
# Config dataclasses
# -------------------------------
@dataclass
class LineConfig:
    orientation: str = "horizontal"  # "horizontal" or "vertical"
    position: int = 300              # y for horizontal, x for vertical
    hysteresis: int = 10             # small band around the line to avoid bouncing


@dataclass
class AlertConfig:
    burst_threshold: int = 5         # entries within T seconds to trigger
    burst_window_sec: float = 10.0
    occupancy_limit: int = 50        # capacity
    cooldown_sec: float = 15.0       # cooldown between same alert types


@dataclass
class DrawConfig:
    show_tracks: bool = True
    show_ids: bool = True
    show_boxes: bool = True
    font_scale: float = 0.7
    thickness: int = 2


# -------------------------------
# Simple Centroid Tracker
# -------------------------------
class CentroidTracker:
    """
    Very simple tracker that matches detections to existing tracks
    by nearest centroid, with max distance and max time without update.
    Good enough when camera is fixed and FPS is reasonable.
    """
    def __init__(self, max_distance=80, max_missed=10):
        self.next_id = 1
        self.objects: Dict[int, Dict] = {}  # id -> dict(xyxy, centroid, missed, state, last_side)
        self.max_distance = max_distance
        self.max_missed = max_missed

    def _centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, detections: List[Tuple[int, int, int, int]]):
        # detections: list of xyxy boxes for persons
        cents = [self._centroid(b) for b in detections]

        # mark all as missed initially
        for tid in list(self.objects.keys()):
            self.objects[tid]["missed"] += 1

        # assign by greedy nearest-neighbor
        used_tracks = set()
        used_dets = set()
        for di, det in enumerate(detections):
            c = cents[di]
            # find nearest track
            best_tid, best_dist = None, 1e9
            for tid, obj in self.objects.items():
                if tid in used_tracks:
                    continue
                oc = obj["centroid"]
                dist = np.hypot(c[0] - oc[0], c[1] - oc[1])
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid
            if best_tid is not None and best_dist <= self.max_distance:
                # update that track
                self.objects[best_tid].update({
                    "box": det,
                    "centroid": c,
                    "missed": 0,
                })
                used_tracks.add(best_tid)
                used_dets.add(di)

        # create new tracks for unassigned detections
        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            c = cents[di]
            tid = self.next_id
            self.next_id += 1
            self.objects[tid] = {
                "box": det,
                "centroid": c,
                "missed": 0,
                "state": "unknown",  # "inside" or "outside" once known
                "last_side": None,   # "above"/"below" for horizontal; "left"/"right" for vertical
                "history": deque(maxlen=8),
            }

        # drop stale tracks
        for tid in list(self.objects.keys()):
            if self.objects[tid]["missed"] > self.max_missed:
                del self.objects[tid]

        return self.objects


# -------------------------------
# Alert Manager
# -------------------------------
class AlertManager:
    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self.entry_events = deque()  # timestamps of entries
        self.exit_events = deque()   # timestamps of exits
        self.last_alert_time = defaultdict(lambda: 0.0)  # key -> last time
        self._lock = threading.Lock()

    def _rate_limited(self, key: str) -> bool:
        now = time.time()
        if now - self.last_alert_time[key] >= self.cfg.cooldown_sec:
            self.last_alert_time[key] = now
            return False  # not limited
        return True

    def record_entry(self):
        with self._lock:
            self.entry_events.append(time.time())

    def record_exit(self):
        with self._lock:
            self.exit_events.append(time.time())

    def _prune(self, deq: deque, window: float):
        cutoff = time.time() - window
        while deq and deq[0] < cutoff:
            deq.popleft()

    def check_burst_alert(self) -> Tuple[bool, int]:
        # burst on entries
        self._prune(self.entry_events, self.cfg.burst_window_sec)
        count = len(self.entry_events)
        if count >= self.cfg.burst_threshold:
            if not self._rate_limited("burst"):
                self._trigger("BURST ALERT: {} entries in last {}s".format(
                    count, int(self.cfg.burst_window_sec)))
                return True, count
        return False, count

    def check_occupancy_alert(self, occupancy: int) -> bool:
        if occupancy > self.cfg.occupancy_limit:
            if not self._rate_limited("occupancy"):
                self._trigger(f"OCCUPANCY ALERT: occupancy {occupancy} > limit {self.cfg.occupancy_limit}")
                return True
        return False

    def _trigger(self, message: str):
        # 1) Print
        print("[ALERT]", message)

        # 2) On-screen overlay handled by caller

        # 3) Optional: sound (cross-platform best-effort)
        try:
            # simple beep on Windows
            import winsound
            winsound.Beep(1000, 500)
        except Exception:
            # Fallback: try to make a short beep with the console bell
            print('\a', end='')

        # 4) TODO: Add your webhook/email/SMS here
        # Example stub:
        # import requests
        # requests.post("https://your-webhook", json={"text": message})


# -------------------------------
# Utils
# -------------------------------
def side_of_line(line: LineConfig, centroid: Tuple[int, int]) -> str:
    x, y = centroid
    if line.orientation == "horizontal":
        if y < line.position - line.hysteresis:
            return "above"
        elif y > line.position + line.hysteresis:
            return "below"
        else:
            return "near"
    else:
        if x < line.position - line.hysteresis:
            return "left"
        elif x > line.position + line.hysteresis:
            return "right"
        else:
            return "near"


def draw_overlay(frame, line_cfg: LineConfig, counts, occupancy, alerts_text: List[str], draw_cfg: DrawConfig):
    h, w = frame.shape[:2]
    # Draw line
    color = (0, 255, 255)
    if line_cfg.orientation == "horizontal":
        cv2.line(frame, (0, line_cfg.position), (w, line_cfg.position), color, 2)
    else:
        cv2.line(frame, (line_cfg.position, 0), (line_cfg.position, h), color, 2)

    # Text block
    x0, y0 = 15, 30
    cv2.putText(frame, f"Entered: {counts['in']}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, draw_cfg.font_scale, (0, 255, 0), draw_cfg.thickness)
    cv2.putText(frame, f"Exited : {counts['out']}", (x0, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, draw_cfg.font_scale, (0, 0, 255), draw_cfg.thickness)
    cv2.putText(frame, f"Inside : {occupancy}", (x0, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, draw_cfg.font_scale, (255, 255, 255), draw_cfg.thickness)

    # Alerts
    y = y0 + 100
    for t in alerts_text[-3:]:  # show last 3 alerts
        cv2.putText(frame, t, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, draw_cfg.font_scale, (0, 140, 255), draw_cfg.thickness)
        y += 28


# -------------------------------
# Main app
# -------------------------------
def run(
    source,
    weights="yolov8n.pt",
    conf_thres=0.5,
    line_orientation="horizontal",
    line_position=300,
    line_hyst=10,
    burst_threshold=5,
    burst_window_sec=10.0,
    occupancy_limit=50,
    cooldown_sec=15.0,
    show=True
):
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO not installed. Please `pip install ultralytics`.")

    model = YOLO(weights)
    line_cfg = LineConfig(orientation=line_orientation, position=line_position, hysteresis=line_hyst)
    alert_cfg = AlertConfig(
        burst_threshold=burst_threshold,
        burst_window_sec=burst_window_sec,
        occupancy_limit=occupancy_limit,
        cooldown_sec=cooldown_sec
    )
    draw_cfg = DrawConfig()

    tracker = CentroidTracker(max_distance=80, max_missed=10)
    alerts = AlertManager(alert_cfg)
    alerts_text_log = deque(maxlen=20)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    counts = {"in": 0, "out": 0}
    occupancy = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Inference
        results = model.predict(frame, conf=conf_thres, classes=[0], verbose=False)  # class 0: person
        boxes_xyxy = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = b.astype(int)
                    # Basic sanity
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = max(x1 + 1, x2); y2 = max(y1 + 1, y2)
                    boxes_xyxy.append((x1, y1, x2, y2))

        # Update tracker
        objects = tracker.update(boxes_xyxy)

        # For each track, check side crossing
        for tid, obj in objects.items():
            c = obj["centroid"]
            side = side_of_line(line_cfg, c)

            # Establish initial state
            if obj["last_side"] is None and side in ("above", "below", "left", "right"):
                obj["last_side"] = side

            # Detect crossing: last_side -> opposite side across the line
            if obj["last_side"] and side in ("above", "below", "left", "right") and side != obj["last_side"]:
                # Determine direction relative to line
                if line_cfg.orientation == "horizontal":
                    if obj["last_side"] == "above" and side == "below":
                        counts["in"] += 1
                        occupancy += 1
                        alerts.record_entry()
                        txt = f"Entry (ID {tid})"
                        alerts_text_log.append(txt)
                    elif obj["last_side"] == "below" and side == "above":
                        counts["out"] += 1
                        occupancy = max(0, occupancy - 1)
                        alerts.record_exit()
                        txt = f"Exit  (ID {tid})"
                        alerts_text_log.append(txt)
                else:
                    if obj["last_side"] == "left" and side == "right":
                        counts["in"] += 1
                        occupancy += 1
                        alerts.record_entry()
                        txt = f"Entry (ID {tid})"
                        alerts_text_log.append(txt)
                    elif obj["last_side"] == "right" and side == "left":
                        counts["out"] += 1
                        occupancy = max(0, occupancy - 1)
                        alerts.record_exit()
                        txt = f"Exit  (ID {tid})"
                        alerts_text_log.append(txt)

                obj["last_side"] = side
            elif side in ("above", "below", "left", "right"):
                obj["last_side"] = side

            # Draw
            if draw_cfg.show_boxes:
                x1, y1, x2, y2 = obj["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 50), 2)
            if draw_cfg.show_ids:
                cv2.putText(frame, f"ID {tid}", (c[0]-10, c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.circle(frame, c, 3, (255, 255, 255), -1)

        # Check alerts
        burst_triggered, burst_count = alerts.check_burst_alert()
        if burst_triggered:
            alerts_text_log.append(f"ALERT: Burst entries ({burst_count})")
        if alerts.check_occupancy_alert(occupancy):
            alerts_text_log.append(f"ALERT: Occupancy {occupancy}>{alert_cfg.occupancy_limit}")

        # Overlay
        draw_overlay(frame, line_cfg, counts, occupancy, list(alerts_text_log), draw_cfg)

        if show:
            cv2.imshow("People Counter + Alerts", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0", help="RTSP/HTTP URL, file path, or webcam index (e.g., 0)")
    p.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLOv8 weights")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    p.add_argument("--line_orientation", type=str, default="horizontal", choices=["horizontal", "vertical"])
    p.add_argument("--line_position", type=int, default=300, help="y (horizontal) or x (vertical)")
    p.add_argument("--line_hyst", type=int, default=10, help="hysteresis band to avoid jitter near the line")
    p.add_argument("--burst_threshold", type=int, default=5, help="entries within window to trigger burst alert")
    p.add_argument("--burst_window", type=float, default=10.0, help="burst time window in seconds")
    p.add_argument("--occupancy_limit", type=int, default=50, help="max people allowed inside before alert")
    p.add_argument("--cooldown", type=float, default=15.0, help="cooldown seconds between same alert types")
    p.add_argument("--show", action="store_true", help="show UI window (default off in servers)")
    args = p.parse_args()

    # Allow numeric string "0" to become int 0 for webcams
    source = args.source
    if source.isdigit():
        source = int(source)

    return dict(
        source=source,
        weights=args.weights,
        conf_thres=args.conf,
        line_orientation=args.line_orientation,
        line_position=args.line_position,
        line_hyst=args.line_hyst,
        burst_threshold=args.burst_threshold,
        burst_window_sec=args.burst_window,
        occupancy_limit=args.occupancy_limit,
        cooldown_sec=args.cooldown,
        show=args.show
    )


if __name__ == "__main__":
    params = parse_args()
    run(**params)
