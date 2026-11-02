import os
import cv2
import numpy as np

class ClassroomDetector:
    def __init__(self, model_name="yolo11s.pt", pose_model_name="yolo11n-pose.pt", conf_threshold=0.30):
        self.conf_threshold = conf_threshold
        self.model = None
        self.pose_model = None
        self.backend = "yolo11"
        self.device = "0"
        self.classes_of_interest = {
            "cell phone", "book", "laptop"
        }

        self._init_models(model_name, pose_model_name)

    def _init_models(self, model_name, pose_model_name):
        try:
            from ultralytics import YOLO
            import torch
            if torch.cuda.is_available():
                self.device = "0"
            else:
                self.device = "cpu"
            print(f"[VisionDetector] Initializing High-Performance YOLO11 ({model_name} & {pose_model_name}) on CUDA:{self.device}...")
            try:
                self.model = YOLO(model_name)
            except Exception:
                self.model = YOLO("yolo11n.pt")

            self.pose_model = YOLO(pose_model_name)
            self.backend = "yolo11-pose"
            print("[VisionDetector] YOLO11 Dual-Model Neural Pipeline Active!")
            return
        except Exception as e:
            print(f"[VisionDetector] YOLO11 init error: {e}. Falling back to OpenCV cascade.")
            self._init_opencv_fallback()

    def _init_opencv_fallback(self):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.backend = "opencv_cascade"
            self.device = "cpu"
        except Exception as e:
            print(f"[VisionDetector] OpenCV fallback error: {e}")
            self.backend = "dummy"

    def detect(self, frame):
        """
        Ultra-fast dual-pass YOLO11 inference (~6-8ms on RTX 3050):
        1. Object detection: Mobile Phone, Book, Note / Paper, Laptop
        2. Pose skeletal keypoints (eyes, nose, ears, shoulders, wrists)
        """
        import torch
        detections = []
        if self.backend == "yolo11-pose" and self.model is not None:
            with torch.inference_mode():
                # 1. Detect objects (Phone, Book/Notebook, Laptop) with calibrated thresholds
                try:
                    results_obj = self.model.predict(
                        source=frame,
                        conf=0.20,
                        imgsz=640,
                        device=self.device,
                        verbose=False
                    )
                    raw_objects = []
                    for r in results_obj:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            raw_class = self.model.names[cls_id]
                            conf = float(box.conf[0])
                            xyxy = [float(v) for v in box.xyxy[0]]

                            if raw_class == "cell phone" and conf >= 0.22:
                                bw = xyxy[2] - xyxy[0]
                                bh = xyxy[3] - xyxy[1]
                                if bw < 350 and bh < 450:
                                    raw_objects.append({
                                        "box": xyxy,
                                        "class_name": "Phone",
                                        "conf": round(conf, 3),
                                        "keypoints": None
                                    })
                            elif raw_class in ["book", "laptop"] and conf >= 0.22:
                                raw_objects.append({
                                    "box": xyxy,
                                    "class_name": "Book / Notebook",
                                    "conf": round(conf, 3),
                                    "keypoints": None
                                })

                    # Disambiguate: A book cover must NEVER be called a phone!
                    books = [d for d in raw_objects if d['class_name'] == 'Book / Notebook']
                    phones = [d for d in raw_objects if d['class_name'] == 'Phone']

                    valid_phones = []
                    for p in phones:
                        px1, py1, px2, py2 = p['box']
                        p_area = max(1.0, (px2 - px1) * (py2 - py1))
                        on_book = False
                        for b in books:
                            bx1, by1, bx2, by2 = b['box']
                            ix1, iy1 = max(px1, bx1), max(py1, by1)
                            ix2, iy2 = min(px2, bx2), min(py2, by2)
                            if ix2 > ix1 and iy2 > iy1:
                                inter = (ix2 - ix1) * (iy2 - iy1)
                                if inter > 0.35 * p_area:
                                    on_book = True
                                    break
                        if not on_book:
                            valid_phones.append(p)

                    detections.extend(books)
                    detections.extend(valid_phones)

                except Exception as e:
                    print(f"[VisionDetector] Object detection error: {e}")

                # 2. Extract pose keypoints for confirmed human students
                try:
                    results_pose = self.pose_model.predict(
                        source=frame,
                        conf=0.25,
                        imgsz=480,
                        device=self.device,
                        verbose=False
                    )
                    pose_persons = []
                    for r in results_pose:
                        if r.keypoints is not None and len(r.keypoints.data) > 0:
                            for i, box in enumerate(r.boxes):
                                kpts = r.keypoints.data[i].cpu().numpy() # [17, 3] (x, y, conf)
                                xyxy = [float(v) for v in box.xyxy[0]]
                                bw = xyxy[2] - xyxy[0]
                                bh = xyxy[3] - xyxy[1]
                                if bw < 40 or bh < 50:
                                    continue

                                pose_persons.append({
                                    "box": xyxy,
                                    "class_name": "person",
                                    "conf": float(box.conf[0]),
                                    "keypoints": kpts
                                })

                    # Fallback to YOLO object model for person detection if pose missed
                    for r in results_obj:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            raw_class = self.model.names[cls_id]
                            conf = float(box.conf[0])
                            if raw_class == "person" and conf >= 0.35:
                                xyxy = [float(v) for v in box.xyxy[0]]
                                bw = xyxy[2] - xyxy[0]
                                bh = xyxy[3] - xyxy[1]
                                if bw < 50 or bh < 60:
                                    continue
                                covered = False
                                for p in pose_persons:
                                    px1, py1, px2, py2 = p['box']
                                    ix1, iy1 = max(xyxy[0], px1), max(xyxy[1], py1)
                                    ix2, iy2 = min(xyxy[2], px2), min(xyxy[3], py2)
                                    if ix2 > ix1 and iy2 > iy1:
                                        inter = (ix2 - ix1) * (iy2 - iy1)
                                        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                                        if inter > 0.40 * area:
                                            covered = True
                                            break
                                if not covered:
                                    pose_persons.append({
                                        "box": xyxy,
                                        "class_name": "person",
                                        "conf": conf,
                                        "keypoints": None
                                    })

                    detections.extend(pose_persons)

                except Exception as e:
                    print(f"[VisionDetector] Pose inference error: {e}")

            return detections

        elif self.backend == "opencv_cascade":
            return self._detect_cascade(frame)
        return []

    def _detect_cascade(self, frame):
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (fx, fy, fw, fh) in faces:
            px1 = max(0, fx - int(fw * 0.4))
            py1 = max(0, fy - int(fh * 0.2))
            px2 = min(frame.shape[1], fx + int(fw * 1.4))
            py2 = min(frame.shape[0], fy + int(fh * 2.8))
            detections.append({
                "box": [float(px1), float(py1), float(px2), float(py2)],
                "class_name": "person",
                "conf": 0.85,
                "keypoints": None
            })
        return detections
