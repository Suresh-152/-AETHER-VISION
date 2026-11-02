import os
import time
import json
import csv
import cv2
from datetime import datetime

class ClassroomBehaviorEngine:
    def __init__(self, incidents_dir="data/incidents", reports_dir="data/reports"):
        self.incidents_dir = incidents_dir
        self.reports_dir = reports_dir
        os.makedirs(self.incidents_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self.session_start = time.time()
        self.incidents = []
        self.incident_counter = 0

        # Debounce tracking: { (track_id, violation_type): last_alert_time }
        self.alert_cooldowns = {}
        self.cooldown_period = 8.0  # seconds between duplicate incident captures for same student

        # Historical telemetry for live charts
        self.telemetry_history = []
        self.max_history = 60  # keep last 60 data points (e.g., 1 per second)

        # Cumulative counters
        self.total_phone_violations = 0
        self.total_sleep_incidents = 0
        self.total_hand_raises = 0
        self.total_talking_incidents = 0
        self.peak_attendance = 0

        # Zone definitions [normalized 0.0 to 1.0 or pixel coords]
        # Default: Teacher Podium on top-left / center, Student Area elsewhere
        self.zones = {
            "TEACHER_PODIUM": {"x1": 0.05, "y1": 0.05, "x2": 0.40, "y2": 0.45},
            "STUDENT_DESKS": {"x1": 0.0, "y1": 0.35, "x2": 1.0, "y2": 1.0}
        }

    def process_frame(self, frame, tracks, detected_objects):
        """
        frame: BGR frame for snapshotting
        tracks: list of TrackedObject
        detected_objects: list of raw detections (phones, books, etc.)
        """
        current_time = time.time()
        student_count = len(tracks)
        self.peak_attendance = max(self.peak_attendance, student_count)

        active_hand_raises = sum(1 for t in tracks if t.posture == "HAND RAISED")
        active_sleeping = sum(1 for t in tracks if t.posture in ["HEAD DOWN", "SLEEPING / DROWSY"])
        active_talking = sum(1 for t in tracks if getattr(t, 'is_talking', False) or t.posture == "TALKING")
        active_phones = sum(1 for d in detected_objects if d['class_name'] in ['Phone', 'cell phone', 'Mobile Phone'])
        active_books = sum(1 for d in detected_objects if d['class_name'] in ['Book / Notebook', 'Book', 'book'])

        # Calculate attention index
        if tracks:
            attention_scores = [t.attention_score for t in tracks]
            avg_attention = round(float(sum(attention_scores) / len(attention_scores)), 1)
            if active_phones > 0:
                avg_attention = min(avg_attention, 25.0)
                for t in tracks:
                    t.attention_score = min(t.attention_score, 25.0)
                    t.posture = "PHONE VIOLATION"
        else:
            avg_attention = 0.0

        # Check incidents for each student
        h, w = frame.shape[:2]

        for t in tracks:
            # 1. Mobile Phone violation
            if t.posture == "PHONE VIOLATION":
                key = (t.track_id, "PHONE_VIOLATION")
                if current_time - self.alert_cooldowns.get(key, 0) > self.cooldown_period:
                    self.alert_cooldowns[key] = current_time
                    self.total_phone_violations += 1
                    self._record_incident(
                        frame=frame,
                        track_id=t.track_id,
                        v_type="PHONE DETECTED",
                        severity="CRITICAL",
                        details=f"Student #{t.track_id} using mobile phone during class.",
                        box=t.box
                    )

            # 2. Sleeping / Head down violation
            elif t.posture in ["HEAD DOWN", "SLEEPING / DROWSY"]:
                key = (t.track_id, "HEAD_DOWN")
                if current_time - self.alert_cooldowns.get(key, 0) > self.cooldown_period * 1.5:
                    self.alert_cooldowns[key] = current_time
                    self.total_sleep_incidents += 1
                    self._record_incident(
                        frame=frame,
                        track_id=t.track_id,
                        v_type="HEAD DOWN / SLUMPED",
                        severity="WARNING",
                        details=f"Student #{t.track_id} has head down / slumped posture.",
                        box=t.box
                    )

            # 3. Talking violation
            elif getattr(t, 'is_talking', False) or t.posture == "TALKING":
                key = (t.track_id, "TALKING")
                if current_time - self.alert_cooldowns.get(key, 0) > self.cooldown_period * 1.2:
                    self.alert_cooldowns[key] = current_time
                    self.total_talking_incidents += 1
                    self._record_incident(
                        frame=frame,
                        track_id=t.track_id,
                        v_type="TALKING IN CLASS",
                        severity="WARNING",
                        details=f"Student #{t.track_id} talking or whispering.",
                        box=t.box
                    )

            # 4. Hand raise event (Participation)
            elif t.posture == "HAND RAISED":
                key = (t.track_id, "HAND_RAISE")
                if current_time - self.alert_cooldowns.get(key, 0) > 15.0:
                    self.alert_cooldowns[key] = current_time
                    self.total_hand_raises += 1
                    self._record_incident(
                        frame=frame,
                        track_id=t.track_id,
                        v_type="HAND RAISED",
                        severity="INFO",
                        details=f"Student #{t.track_id} raised hand with a question.",
                        box=t.box
                    )

        # Determine dominant primary status for the dashboard
        if active_phones > 0 or any(t.posture == "PHONE VIOLATION" for t in tracks):
            primary_status = "Using Phone"
        elif active_sleeping > 0 or any(t.posture in ["HEAD DOWN", "SLEEPING / DROWSY"] for t in tracks):
            primary_status = "Head Down / Sleeping"
        elif active_talking > 0 or any(getattr(t, 'is_talking', False) or t.posture == "TALKING" for t in tracks):
            primary_status = "Talking / Speech"
        elif any(t.posture == "LOOKING AWAY" for t in tracks):
            primary_status = "Looking Away / Distracted"
        elif active_books > 0 or any(t.posture == "STUDYING / READING" for t in tracks):
            primary_status = "Reading / Notebook"
        elif active_hand_raises > 0 or any(t.posture == "HAND RAISED" for t in tracks):
            primary_status = "Question / Hand Raised"
        elif student_count > 0:
            primary_status = "Attentive & Focused"
        else:
            primary_status = "No Students Detected"

        # Update telemetry history snapshot
        telemetry_point = {
            "timestamp": round(current_time - self.session_start, 1),
            "clock": datetime.now().strftime("%H:%M:%S"),
            "headcount": student_count,
            "peak_attendance": self.peak_attendance,
            "attention_index": avg_attention,
            "primary_status": primary_status,
            "active_hand_raises": active_hand_raises,
            "sleeping_count": active_sleeping,
            "talking_count": active_talking,
            "phone_count": active_phones,
            "book_count": active_books,
            "note_count": active_books
        }
        self.telemetry_history.append(telemetry_point)
        if len(self.telemetry_history) > self.max_history:
            self.telemetry_history.pop(0)

        return telemetry_point

    def _record_incident(self, frame, track_id, v_type, severity, details, box):
        self.incident_counter += 1
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{self.incident_counter:04d}"
        filename = f"{incident_id}.jpg"
        filepath = os.path.join(self.incidents_dir, filename)

        # Save annotated snapshot of incident
        snap = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 0, 255) if severity == "CRITICAL" else ((0, 165, 255) if severity == "WARNING" else (0, 255, 128))
        cv2.rectangle(snap, (x1, y1), (x2, y2), color, 2)
        cv2.putText(snap, f"{v_type} [ID:{track_id}]", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imwrite(filepath, snap)

        incident_record = {
            "id": incident_id,
            "student_id": track_id,
            "type": v_type,
            "severity": severity,
            "details": details,
            "time": datetime.now().strftime("%H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "snapshot": f"/static/incidents/{filename}",
            "box": box
        }
        self.incidents.insert(0, incident_record)  # most recent first
        if len(self.incidents) > 100:
            self.incidents.pop()

    def get_summary_stats(self):
        duration_sec = int(time.time() - self.session_start)
        minutes, seconds = divmod(duration_sec, 60)
        hours, minutes = divmod(minutes, 60)

        latest_attention = (self.telemetry_history[-1]["attention_index"] 
                            if self.telemetry_history else 100.0)
        latest_headcount = (self.telemetry_history[-1]["headcount"] 
                            if self.telemetry_history else 0)

        return {
            "session_duration": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "current_attendance": latest_headcount,
            "peak_attendance": self.peak_attendance,
            "average_attention": latest_attention,
            "phone_violations": self.total_phone_violations,
            "sleep_incidents": self.total_sleep_incidents,
            "hand_raises": self.total_hand_raises,
            "total_incidents": len(self.incidents)
        }

    def export_report_json(self):
        report = {
            "session_summary": self.get_summary_stats(),
            "telemetry_history": self.telemetry_history,
            "incidents": self.incidents
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"classroom_report_{timestamp}.json"
        filepath = os.path.join(self.reports_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        return filename

    def export_report_csv(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"incidents_{timestamp}.csv"
        filepath = os.path.join(self.reports_dir, filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Incident ID", "Date", "Time", "Student ID", "Type", "Severity", "Details", "Snapshot"])
            for inc in self.incidents:
                writer.writerow([inc["id"], inc["date"], inc["time"], inc["student_id"], 
                                 inc["type"], inc["severity"], inc["details"], inc["snapshot"]])
        return filename
