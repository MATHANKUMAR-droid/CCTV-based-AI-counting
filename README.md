# CCTV-based-AI-counting
cctv-people-counter/
│── main.py
│── requirements.txt
│── README.md
│── LICENSE   (optional, MIT recommended)

# 🛡️ CCTV People Counter with AI Alerts

An AI-powered CCTV monitoring system that detects and counts people entering or leaving through an office gate.  
Built with [YOLOv8](https://github.com/ultralytics/ultralytics) for real-time person detection, lightweight tracking, and alerting mechanisms.

---

## ✨ Features
- ✅ Counts **entries & exits** via a virtual line.
- ✅ Tracks occupancy (people inside).
- ✅ **Burst alert:** triggers if too many entries happen in a short window.
- ✅ **Occupancy alert:** triggers if current count exceeds a limit.
- ✅ Works with **webcams, IP/RTSP cameras, or video files**.
- ✅ Extensible hooks for **webhooks, email, or IoT hardware** alerts.
- ✅ Visual overlay with bounding boxes, counts, and alerts.

---

## 🚀 Installation

1. Clone the repo:
   git clone https://github.com/YOUR-USERNAME/cctv-people-counter.git
   cd cctv-people-counter
2.Install dependencies:
pip install -r requirements.txt

**Usage**
**Webcam (default cam 0):**
python people_counter_with_alerts.py --source 0 --show
**RTSP/IP Camera:**
python people_counter_with_alerts.py --source rtsp://user:pass@IP:PORT/stream --show
**Video file:**
python people_counter_with_alerts.py --source office_gate.mp4 --show

**Configuration**
Line orientation: --line_orientation [horizontal|vertical]
Line position: --line_position <y or x>
Line hysteresis (tolerance): --line_hyst <pixels>
Confidence threshold: --conf 0.5
Burst alert: --burst_threshold 5 --burst_window 10
Occupancy alert: --occupancy_limit 50
Cooldown (no repeat alerts within X sec): --cooldown 15


---

# 📄 requirements.txt
```txt
ultralytics>=8.0.100
opencv-python
numpy
imutils
playsound==1.2.2

