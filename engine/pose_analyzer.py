import numpy as np
import cv2

class ClassroomPoseAnalyzer:
    """
    State-of-the-Art Classroom Posture, Engagement & Talking Analyzer.
    Combines 17-point Skeletal Keypoints (from YOLO11-Pose) with Geometric,
    Mouth/Jaw dynamics, and Proximity heuristics:
      - Head Yaw / Gaze Vector (Looking away vs Forward)
      - Head Pitch / Slump Index (Sleeping / Drowsiness)
      - Jaw & Mouth Syllable Motion (Talking / Whispering Detection)
      - Arm Elevation (Active Question / Hand-Raising)
      - Prohibited Device Manipulation (Mobile Phone in Hand)
    """
    def __init__(self, fps=25.0):
        self.fps = fps
        self.sleep_threshold_sec = 1.2
        self.sleep_threshold_frames = int(self.fps * self.sleep_threshold_sec)
        self.prev_mouth_crops = {}     # { track_id: prev_gray_mouth }
        self.prev_forehead_crops = {}  # { track_id: prev_gray_forehead }

    def analyze_student(self, frame, track, related_objects):
        x1, y1, x2, y2 = [int(v) for v in track.box]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        # Initialize tracking attributes if missing
        if not hasattr(track, 'talking_frames'):
            track.talking_frames = 0
            track.is_talking = False
        if not hasattr(track, 'head_down_frames'):
            track.head_down_frames = 0
        if not hasattr(track, 'look_away_frames'):
            track.look_away_frames = 0
        if not hasattr(track, 'book_usage_frames'):
            track.book_usage_frames = 0
        if not hasattr(track, 'phone_usage_frames'):
            track.phone_usage_frames = 0

        # 1. Phone & Book Detection
        has_phone = False
        has_book = False
        student_cx = (x1 + x2) / 2.0
        student_cy = (y1 + y2) / 2.0

        for obj in related_objects:
            cname = obj.get('class_name', '')
            if cname == 'Phone':
                px1, py1, px2, py2 = obj['box']
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                dist = float(np.hypot(student_cx - pcx, student_cy - pcy))
                # Phone within physical proximity of student or overlapping
                if dist < 320 or not (px2 < x1 - 50 or px1 > x2 + 50 or py2 < y1 - 50 or py1 > y2 + 50):
                    has_phone = True
            elif cname == 'Book / Notebook':
                bx1, by1, bx2, by2 = obj['box']
                bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                dist = float(np.hypot(student_cx - bcx, student_cy - bcy))
                if dist < 350 or not (bx2 < x1 - 50 or bx1 > x2 + 50 or by2 < y1 - 50 or by1 > y2 + 50):
                    has_book = True

        if has_phone:
            track.phone_usage_frames += 1
        else:
            track.phone_usage_frames = max(0, track.phone_usage_frames - 1)

        if has_book:
            track.book_usage_frames += 1
        else:
            track.book_usage_frames = max(0, track.book_usage_frames - 1)

        # 2. Keypoint Skeletal Analysis
        kpts = track.keypoints
        is_head_down = False
        is_talking = False
        is_looking_away = False
        is_hand_raised = False

        if kpts is not None and len(kpts) >= 7:
            nose = kpts[0]       # (x, y, conf)
            l_eye, r_eye = kpts[1], kpts[2]
            l_ear, r_ear = kpts[3], kpts[4]
            l_sh, r_sh   = kpts[5], kpts[6]
            l_wrist = kpts[9] if len(kpts) > 9 else (0, 0, 0)
            r_wrist = kpts[10] if len(kpts) > 10 else (0, 0, 0)

            # Shoulder baseline
            has_sh = (l_sh[2] > 0.3 or r_sh[2] > 0.3)
            sh_y = 0.0
            if l_sh[2] > 0.3 and r_sh[2] > 0.3:
                sh_y = (l_sh[1] + r_sh[1]) / 2.0
            elif l_sh[2] > 0.3:
                sh_y = l_sh[1]
            elif r_sh[2] > 0.3:
                sh_y = r_sh[1]

            # A. ACCURATE HEAD DOWN / SLEEPING CUES (Physical Biomechanics)
            # Cue 1: Shoulder Clearance - When head drops down/sleeps onto desk, nose drops near or below shoulder line.
            if has_sh and nose[2] > 0.25:
                head_clearance = sh_y - nose[1]
                if head_clearance < 28:
                    is_head_down = True

            # Cue 2: Head tilted downwards (high vertical pitch)
            if nose[2] > 0.25 and (l_eye[2] > 0.25 or r_eye[2] > 0.25):
                eye_y = (l_eye[1] + r_eye[1]) / 2.0 if (l_eye[2] > 0.25 and r_eye[2] > 0.25) else (l_eye[1] if l_eye[2] > 0.25 else r_eye[1])
                if nose[1] - eye_y < 10:
                    is_head_down = True

            # Cue 3: Complete Face Occlusion - Head buried down into arms/desk (face keypoints lost while body/shoulders remain)
            if has_sh and nose[2] < 0.25 and l_eye[2] < 0.25 and r_eye[2] < 0.25:
                is_head_down = True

            # B. RESPONSIVE LOOKING AWAY / DISTRACTED (Gaze & Yaw tracking)
            # Detects head turned left/right away from camera using eyes or ears:
            if nose[2] > 0.35 and not is_head_down:
                # 1. Ear-to-nose lateral ratio
                if l_ear[2] > 0.30 and r_ear[2] > 0.30:
                    d_left = abs(nose[0] - l_ear[0])
                    d_right = abs(nose[0] - r_ear[0])
                    if d_left > 0 and d_right > 0:
                        yaw = d_left / float(d_right)
                        if yaw > 2.2 or yaw < 0.45:
                            is_looking_away = True
                # 2. Eye asymmetry when one ear is occluded by turning profile:
                elif l_eye[2] > 0.35 and r_eye[2] > 0.35:
                    d_eye_l = abs(nose[0] - l_eye[0])
                    d_eye_r = abs(nose[0] - r_eye[0])
                    if d_eye_l > 0 and d_eye_r > 0:
                        eye_ratio = d_eye_l / float(d_eye_r)
                        if eye_ratio > 2.6 or eye_ratio < 0.38:
                            is_looking_away = True

            # C. HAND RAISING
            if l_wrist[2] > 0.4 and nose[2] > 0.3 and l_wrist[1] < nose[1] - 20:
                is_hand_raised = True
            elif r_wrist[2] > 0.4 and nose[2] > 0.3 and r_wrist[1] < nose[1] - 20:
                is_hand_raised = True

            # D. ACCURATE TALKING DETECTION (Differential Mouth vs Forehead Motion)
            if nose[2] > 0.4 and not is_head_down:
                nx, ny = int(nose[0]), int(nose[1])

                # Mouth ROI: directly below nose
                mx1, my1 = max(0, nx - 20), min(h - 1, ny + 10)
                mx2, my2 = min(w, nx + 20), min(h, ny + 38)

                # Control Forehead ROI: between eyes, above nose (does NOT move during speech)
                fx1, fy1 = max(0, nx - 16), max(0, ny - 28)
                fx2, fy2 = min(w, nx + 16), max(0, ny - 6)

                if (mx2 > mx1 and my2 > my1) and (fx2 > fx1 and fy2 > fy1):
                    mouth_crop = frame[my1:my2, mx1:mx2]
                    forehead_crop = frame[fy1:fy2, fx1:fx2]

                    gray_mouth = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
                    gray_forehead = cv2.cvtColor(forehead_crop, cv2.COLOR_BGR2GRAY)

                    prev_m = self.prev_mouth_crops.get(track.track_id)
                    prev_f = self.prev_forehead_crops.get(track.track_id)

                    if (prev_m is not None and prev_m.shape == gray_mouth.shape and
                        prev_f is not None and prev_f.shape == gray_forehead.shape):
                        diff_m = float(np.mean(cv2.absdiff(gray_mouth, prev_m)))
                        diff_f = float(np.mean(cv2.absdiff(gray_forehead, prev_f)))

                        # Speech causes mouth-specific movement that exceeds whole-head motion
                        excess_mouth_motion = diff_m - diff_f
                        if diff_m > 13.0 and excess_mouth_motion > 7.0:
                            is_talking = True

                    self.prev_mouth_crops[track.track_id] = gray_mouth
                    self.prev_forehead_crops[track.track_id] = gray_forehead

        elif kpts is None:
            # Fallback for synthetic simulation where keypoints are not generated
            aspect_ratio = box_h / float(max(1, box_w))
            if aspect_ratio < 0.78:
                is_head_down = True
            elif aspect_ratio > 1.32 and box_h > 85:
                is_hand_raised = True

        # Update Frame Counters
        if is_talking:
            track.talking_frames += 1
        else:
            track.talking_frames = max(0, track.talking_frames - 2)

        if is_head_down:
            track.head_down_frames += 1
        else:
            track.head_down_frames = max(0, track.head_down_frames - 2)

        if is_looking_away:
            track.look_away_frames += 1
        else:
            track.look_away_frames = max(0, track.look_away_frames - 2)

        track.is_talking = (track.talking_frames >= 9) # ~0.35s of sustained mouth motion
        track.is_head_down = (track.head_down_frames >= 4)

        # 3. Responsive Real-Time Cognitive Attention Scoring
        if track.phone_usage_frames >= 1:
            track.posture = "PHONE VIOLATION"
            # Immediately collapse attention when phone is detected in hand
            track.attention_score = max(18.0, min(track.attention_score, 25.0))
        elif track.head_down_frames >= 20: # ~0.8s of head down
            track.posture = "SLEEPING / DROWSY"
            track.attention_score = max(8.0, min(track.attention_score, 15.0))
        elif track.head_down_frames >= 4:
            track.posture = "HEAD DOWN"
            track.attention_score = max(25.0, min(track.attention_score, 35.0))
        elif track.is_talking:
            track.posture = "TALKING"
            track.attention_score = max(42.0, min(track.attention_score, 50.0))
        elif track.look_away_frames >= 3: # Fast detection when student looks away
            track.posture = "LOOKING AWAY"
            track.attention_score = max(46.0, min(track.attention_score, 52.0))
        elif is_hand_raised:
            track.posture = "HAND RAISED"
            track.attention_score = min(100.0, track.attention_score + 3.0)
        elif track.book_usage_frames >= 1:
            track.posture = "STUDYING / READING"
            track.attention_score = min(96.0, track.attention_score + 2.0)
        else:
            track.posture = "ATTENTIVE"
            # Gradual focus recovery back to 96-98%
            track.attention_score = min(98.0, track.attention_score + 2.5)

        track.hand_raised = is_hand_raised
