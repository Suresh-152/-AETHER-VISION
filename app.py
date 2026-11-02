import os
import sys
import time
import json
import threading
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from engine import (
    ClassroomDetector,
    SimpleTracker,
    ClassroomPoseAnalyzer,
    ClassroomBehaviorEngine,
    ClassroomHeatmap,
    ClassroomSimulator
)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Ensure data directories exist
INCIDENTS_DIR = os.path.join(os.path.dirname(__file__), "data", "incidents")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "data", "reports")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
os.makedirs(INCIDENTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

class ClassroomMonitoringSystem:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.source_mode = "webcam0"
        self.camera_device = 0
        self.uploaded_video_path = None
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            print("[MonitoringSystem] Laptop camera not opened, falling back to simulation demo.")
            self.source_mode = "sim"
            self.cap = None

        # Core engines
        self.detector = ClassroomDetector(conf_threshold=0.35)
        self.tracker = SimpleTracker(max_lost=20, iou_threshold=0.25)
        self.pose_analyzer = ClassroomPoseAnalyzer(fps=25.0)
        self.behavior_engine = ClassroomBehaviorEngine(
            incidents_dir=INCIDENTS_DIR, 
            reports_dir=REPORTS_DIR
        )
        self.heatmap = ClassroomHeatmap(width=640, height=360)
        self.simulator = ClassroomSimulator(width=800, height=450)

        # Settings
        self.settings = {
            "show_hud": True,
            "show_heatmap": False,
            "show_skeleton": True,
            "conf_threshold": 0.35,
            "drowsiness_sec": 2.5,
            "sound_alerts": True,
            "source": "webcam0"
        }

        # Latest frame and telemetry cache
        self.latest_frame_bytes = None
        self.latest_telemetry = {
            "fps": 0,
            "headcount": 0,
            "attention_index": 100.0,
            "active_hand_raises": 0,
            "sleeping_count": 0,
            "phone_count": 0,
            "source": "sim",
            "device": self.detector.device,
            "backend": self.detector.backend
        }
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0

        # Start worker thread
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

    def set_source(self, source_type, file_path=None):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            if source_type in ["webcam0", "webcam1"]:
                # Connect to laptop built-in camera
                device_id = 0
                self.cap = cv2.VideoCapture(device_id)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(1)
                if not self.cap.isOpened():
                    print(f"[MonitoringSystem] Camera not accessible, reverting to simulation demo.")
                    source_type = "sim"
            elif source_type == "upload" and file_path and os.path.exists(file_path):
                self.cap = cv2.VideoCapture(file_path)
                self.uploaded_video_path = file_path

            self.source_mode = source_type
            self.settings["source"] = source_type
            self.latest_telemetry["source"] = source_type

    def _processing_loop(self):
        while self.running:
            start_t = time.time()
            frame = None
            simulated_detections = None

            # Capture frame based on source
            with self.lock:
                mode = self.source_mode
                cap = self.cap

            if mode in ["webcam0", "webcam1", "upload"] and cap is not None and cap.isOpened():
                ret, raw_frame = cap.read()
                if not ret:
                    if mode == "upload":
                        # Loop video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, raw_frame = cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue
                frame = raw_frame
            else:
                # Simulation Mode
                frame, simulated_detections = self.simulator.get_frame()

            if frame is None:
                time.sleep(0.04)
                continue

            # Run detection
            if simulated_detections is not None:
                detections = simulated_detections
            else:
                detections = self.detector.detect(frame)

            # Object Tracking
            tracks = self.tracker.update(detections)

            # Pose & Behavioral Analysis
            for t in tracks:
                self.pose_analyzer.analyze_student(frame, t, detections)

            # Behavior & Violation Event Dispatching
            telemetry = self.behavior_engine.process_frame(frame, tracks, detections)

            # Spatial Heatmap Update
            self.heatmap.update(tracks)

            # Render Overpowered Tactical Cyberpunk HUD on frame
            annotated_frame = self._render_tactical_hud(frame, tracks, detections)

            # Encode to JPEG
            ret, jpeg = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ret:
                self.latest_frame_bytes = jpeg.tobytes()

            # FPS calculation
            self.fps_counter += 1
            elapsed = time.time() - self.fps_start
            if elapsed >= 1.0:
                self.current_fps = round(self.fps_counter / elapsed, 1)
                self.fps_counter = 0
                self.fps_start = time.time()

            telemetry["fps"] = self.current_fps
            telemetry["source"] = self.source_mode
            telemetry["backend"] = self.detector.backend
            telemetry["device"] = self.detector.device
            telemetry["tracks_count"] = len(tracks)
            self.latest_telemetry = telemetry

            # Maintain fast, smooth ~45-60 FPS stream
            proc_time = time.time() - start_t
            sleep_duration = max(0.001, 0.016 - proc_time)
            time.sleep(sleep_duration)

    def _render_tactical_hud(self, frame, tracks, detections):
        """Draw ultra-refined Apple Vision Pro / macOS style overlays."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 1. Heatmap overlay if enabled
        if self.settings.get("show_heatmap"):
            vis = self.heatmap.blend_overlay(vis, alpha=0.35)

        # 2. NieR Military HUD Status Banner
        status_label = "[POD 042 // OPERATIONAL: FEED_STATION_01]" if self.source_mode != "sim" else "[POD 042 // SIMULATION_PROTOCOL_ACTIVE]"
        status_color = (130, 160, 140) if self.source_mode != "sim" else (140, 140, 140)
        self._draw_nier_badge(vis, 20, 32, status_label, status_color)

        # 3. Draw Detected Objects (Phone, Book/Notebook)
        for obj in detections:
            cname = obj['class_name']
            if cname == 'person':
                continue
            bx1, by1, bx2, by2 = [int(v) for v in obj['box']]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w - 1, bx2), min(h - 1, by2)

            if cname in ['Phone', 'cell phone', 'Mobile Phone']:
                color = (55, 60, 205) # Crimson
                self._draw_nier_box(vis, bx1, by1, bx2, by2, color, label="Phone", alert=True)
            elif cname in ['Book / Notebook', 'Book', 'book']:
                color = (135, 160, 180) # Pale Blue
                self._draw_nier_box(vis, bx1, by1, bx2, by2, color, label="Book / Notebook")

        # 4. Draw Tracked Students
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Clean and normal descriptive badges
            if t.posture == "PHONE VIOLATION":
                color = (55, 60, 205) # Crimson
                badge = f"Student {t.track_id} • Using Phone"
            elif t.posture in ["SLEEPING / DROWSY", "HEAD DOWN"]:
                color = (70, 130, 215) # Ochre
                badge = f"Student {t.track_id} • Sleeping / Head Down"
            elif t.posture == "TALKING":
                color = (175, 120, 195) # Violet
                badge = f"Student {t.track_id} • Talking"
            elif t.posture == "LOOKING AWAY":
                color = (70, 130, 215) # Ochre
                badge = f"Student {t.track_id} • Looking Away"
            elif t.posture == "STUDYING / READING":
                color = (135, 160, 180) # Pale Blue
                badge = f"Student {t.track_id} • Reading / Notebook"
            elif t.posture == "HAND RAISED":
                color = (95, 180, 220) # Gold
                badge = f"Student {t.track_id} • Hand Raised"
            else:
                color = (130, 165, 140) # Sage
                badge = f"Student {t.track_id} • Attentive ({int(t.attention_score)}%)"

            # Draw NieR Tactical Target Bracket
            self._draw_nier_box(vis, x1, y1, x2, y2, color, label=badge)

            # Draw skeletal vectors
            if t.keypoints is not None and self.settings.get("show_skeleton", True):
                self._draw_skeleton(vis, t.keypoints, color)

        return vis

    def _draw_nier_box(self, img, x1, y1, x2, y2, color, label="", alert=False):
        """Draws an authentic NieR: Automata angular military reticle with corner brackets and crosshairs."""
        box_w = x2 - x1
        box_h = y2 - y1
        corner_len = min(18, max(6, int(min(box_w, box_h) * 0.22)))

        # 1. Subtle, translucent border line
        cv2.rectangle(img, (x1, y1), (x2, y2), (int(color[0]*0.4), int(color[1]*0.4), int(color[2]*0.4)), 1, cv2.LINE_AA)

        # 2. Bold tactical corner brackets (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
        t = 2 # thickness
        # Top-left
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, t, cv2.LINE_AA)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, t, cv2.LINE_AA)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, t, cv2.LINE_AA)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, t, cv2.LINE_AA)

        # 3. NieR military label banner
        if label:
            self._draw_nier_badge(img, x1, max(18, y1 - 6), label, color, alert)

    def _draw_nier_badge(self, img, x, y, text, color, alert=False):
        """Renders an angular NieR: Automata military telemetry tag with square bracket markers."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        pad_x, pad_y = 8, 4
        tag_w = tw + pad_x * 2 + 10
        tag_h = th + pad_y * 2
        px1, py1 = x, y - tag_h
        px2, py2 = px1 + tag_w, y

        h, w = img.shape[:2]
        if py1 < 4:
            py1 = y + 4
            py2 = py1 + tag_h
        px2 = min(w - 4, px2)
        px1 = max(4, px1)

        # NieR Dark Slate background (#24221E / BGR: 30, 34, 36)
        sub = img[py1:py2, px1:px2]
        if sub.shape[0] > 0 and sub.shape[1] > 0:
            nier_bg = np.full_like(sub, 28)
            cv2.addWeighted(sub, 0.15, nier_bg, 0.85, 0, sub)
            img[py1:py2, px1:px2] = sub

        # Left tactical color block accent
        accent_w = 4
        cv2.rectangle(img, (px1, py1), (px1 + accent_w, py2), color, -1)

        # Crisp angular border
        cv2.rectangle(img, (px1, py1), (px2, py2), (75, 78, 80), 1, cv2.LINE_AA)

        # NieR Mono-style beige typography (#DAD4BB / BGR: 187, 212, 218)
        text_x = px1 + accent_w + 6
        text_y = py1 + tag_h // 2 + th // 2 - 1
        cv2.putText(img, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 215, 220), 1, cv2.LINE_AA)

    def _draw_skeleton(self, img, keypoints, color):
        """Draw subtle, refined Apple Vision style skeletal vectors."""
        bones = [
            (0, 1), (0, 2), (1, 3), (2, 4),      # Facial gaze connectors
            (5, 6),                               # Shoulders
            (5, 7), (7, 9),                       # Left arm
            (6, 8), (8, 10),                      # Right arm
            (5, 11), (6, 12), (11, 12),           # Torso
        ]
        # Thin, elegant anti-aliased lines
        for p1, p2 in bones:
            if p1 < len(keypoints) and p2 < len(keypoints):
                x_a, y_a, c_a = keypoints[p1]
                x_b, y_b, c_b = keypoints[p2]
                if c_a > 0.35 and c_b > 0.35:
                    cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (200, 200, 210), 1, cv2.LINE_AA)

        # Refined joint nodes
        for i, (jx, jy, jconf) in enumerate(keypoints):
            if jconf > 0.35 and i <= 10:
                cv2.circle(img, (int(jx), int(jy)), 3, (255, 255, 255), -1, cv2.LINE_AA)

system = ClassroomMonitoringSystem()

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        if system.latest_frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + system.latest_frame_bytes + b'\r\n')
        time.sleep(0.033)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def get_telemetry():
    return jsonify(system.latest_telemetry)

@app.route('/api/incidents')
def get_incidents():
    return jsonify(system.behavior_engine.incidents)

@app.route('/api/history')
def get_history():
    return jsonify(system.behavior_engine.telemetry_history)

@app.route('/api/summary')
def get_summary():
    return jsonify(system.behavior_engine.get_summary_stats())

@app.route('/api/switch_source', methods=['POST'])
def switch_source():
    data = request.json or {}
    source = data.get('source', 'sim')
    system.set_source(source)
    return jsonify({"status": "success", "source": system.source_mode})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = f"uploaded_{int(time.time())}_{file.filename}"
    filepath = os.path.join(UPLOADS_DIR, filename)
    file.save(filepath)

    system.set_source("upload", filepath)
    return jsonify({"status": "success", "file": filename})

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'POST':
        data = request.json or {}
        for k, v in data.items():
            if k in system.settings:
                system.settings[k] = v
        # Update detector threshold if provided
        if "conf_threshold" in data:
            system.detector.conf_threshold = float(data["conf_threshold"])
        return jsonify({"status": "updated", "settings": system.settings})
    return jsonify(system.settings)

@app.route('/api/export_report', methods=['GET'])
def export_report():
    fmt = request.args.get('format', 'json').lower()
    if fmt == 'csv':
        filename = system.behavior_engine.export_report_csv()
    else:
        filename = system.behavior_engine.export_report_json()
    return jsonify({
        "status": "success",
        "format": fmt,
        "filename": filename,
        "url": f"/reports/{filename}"
    })

@app.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(REPORTS_DIR, filename, as_attachment=True)

@app.route('/static/incidents/<path:filename>')
def serve_incident_snap(filename):
    return send_from_directory(INCIDENTS_DIR, filename)

def _open_gui(port):
    import shutil
    import subprocess
    import webbrowser
    time.sleep(1.2)
    url = f"http://localhost:{port}"
    print(f"[GUI] Auto-launching Tactical HUD GUI at {url}...")
    for browser_cmd in ["chromium", "brave", "zen-browser", "firefox"]:
        if shutil.which(browser_cmd):
            try:
                if browser_cmd in ["chromium", "brave"]:
                    subprocess.Popen([browser_cmd, f"--app={url}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"[GUI] Launched in dedicated native app window ({browser_cmd}).")
                    return
                else:
                    subprocess.Popen([browser_cmd, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"[GUI] Launched in browser ({browser_cmd}).")
                    return
            except Exception:
                pass
    try:
        webbrowser.open(url)
    except Exception:
        pass

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"🚀 AETHER-VISION MISSION CONTROL IS STARTING")
    print(f"🌐 Access Dashboard at: http://localhost:{port}")
    print(f"=======================================================\n")
    # run.sh handles browser opening directly; avoid double launch
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

