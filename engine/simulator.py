import cv2
import numpy as np
import time
import math

class ClassroomSimulator:
    def __init__(self, width=800, height=450):
        self.width = width
        self.height = height
        self.start_time = time.time()
        self.frame_idx = 0

        # Define simulated students: (id, base_x, base_y, name)
        self.students = [
            {"id": 1, "x": 180, "y": 280, "action": "normal", "name": "Student #1"},
            {"id": 2, "x": 380, "y": 280, "action": "phone",  "name": "Student #2"},
            {"id": 3, "x": 580, "y": 280, "action": "normal", "name": "Student #3"},
            {"id": 4, "x": 280, "y": 380, "action": "sleep",  "name": "Student #4"},
            {"id": 5, "x": 480, "y": 380, "action": "hand",   "name": "Student #5"},
        ]

    def get_frame(self):
        self.frame_idx += 1
        t = time.time() - self.start_time
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 1. Background Classroom Environment
        # Wall color (dark warm gray/slate)
        frame[:] = (32, 35, 42)

        # Floor (perspective trapezoid)
        pts_floor = np.array([[0, 220], [self.width, 220], [self.width, self.height], [0, self.height]], np.int32)
        cv2.fillPoly(frame, [pts_floor], (24, 28, 36))

        # Grid floor tiles for perspective depth
        for y_line in range(230, self.height, 40):
            cv2.line(frame, (0, y_line), (self.width, y_line), (35, 42, 54), 1)
        for x_line in range(0, self.width, 100):
            cv2.line(frame, (x_line, 220), (int(x_line * 1.3 - 100), self.height), (35, 42, 54), 1)

        # Blackboard / Smartboard
        cv2.rectangle(frame, (140, 25), (660, 150), (45, 60, 48), -1)
        cv2.rectangle(frame, (136, 21), (664, 154), (160, 130, 95), 4) # Wooden frame
        cv2.putText(frame, "ADVANCED COMPUTER VISION & AI", (170, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 240, 225), 2)
        cv2.putText(frame, "Topic: Autonomous Behavioral Monitoring", (170, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 200, 180), 1)
        cv2.putText(frame, f"Session Time: {int(t)}s | Live Detection Active", (170, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 170, 140), 1)

        # Teacher Podium on Left
        cv2.rectangle(frame, (35, 120), (110, 220), (55, 45, 38), -1)
        cv2.rectangle(frame, (30, 115), (115, 130), (80, 65, 55), -1)
        # Teacher avatar
        teacher_sway = int(math.sin(t * 1.2) * 4)
        tx = 65 + teacher_sway
        ty = 100
        cv2.circle(frame, (tx, ty), 16, (170, 195, 225), -1) # Head
        cv2.rectangle(frame, (tx - 18, ty + 16), (tx + 18, ty + 65), (90, 80, 130), -1) # Torso

        # Simulated ground truth / detection candidates
        simulated_objects = []

        # Add Teacher as person
        simulated_objects.append({
            "box": [tx - 20, ty - 18, tx + 20, ty + 70],
            "class_name": "person",
            "conf": 0.94
        })

        # Draw Student Desks & Students
        for s in self.students:
            sx = s["x"]
            sy = s["y"]

            # Cyclic behavior triggers based on time
            cycle = (t + s["id"] * 4) % 24
            is_phone = (s["action"] == "phone" and 4 <= cycle <= 16)
            is_sleep = (s["action"] == "sleep" and 6 <= cycle <= 18)
            is_hand  = (s["action"] == "hand"  and 3 <= cycle <= 12)

            # Desk
            cv2.rectangle(frame, (sx - 65, sy + 15), (sx + 65, sy + 38), (65, 55, 48), -1)
            cv2.rectangle(frame, (sx - 60, sy + 38), (sx - 52, sy + 65), (40, 35, 30), -1)
            cv2.rectangle(frame, (sx + 52, sy + 38), (sx + 60, sy + 65), (40, 35, 30), -1)

            # Desk items (book, laptop)
            cv2.rectangle(frame, (sx - 50, sy + 18), (sx - 20, sy + 32), (180, 60, 60), -1) # Book
            simulated_objects.append({
                "box": [sx - 52, sy + 16, sx - 18, sy + 34],
                "class_name": "book",
                "conf": 0.88
            })

            if s["id"] == 3: # Student 3 has a laptop
                cv2.rectangle(frame, (sx + 10, sy + 16), (sx + 45, sy + 30), (120, 125, 130), -1)
                simulated_objects.append({
                    "box": [sx + 8, sy + 14, sx + 48, sy + 32],
                    "class_name": "laptop",
                    "conf": 0.91
                })

            # Student Avatar
            breathe = int(math.sin(t * 2.5 + s["id"]) * 2)

            if is_sleep:
                # Head slumped down on desk
                head_x = sx + 5
                head_y = sy + 18
                cv2.rectangle(frame, (sx - 25, sy), (sx + 25, sy + 30), (70, 95, 130), -1) # Torso forward
                cv2.circle(frame, (head_x, head_y), 15, (170, 195, 225), -1) # Head on desk
                p_box = [sx - 35, sy - 5, sx + 35, sy + 45]
            elif is_hand:
                # Hand raised high up
                head_x = sx
                head_y = sy - 28 + breathe
                cv2.circle(frame, (head_x, head_y), 17, (170, 195, 225), -1) # Head
                cv2.rectangle(frame, (sx - 22, head_y + 17), (sx + 22, sy + 25), (60, 110, 140), -1) # Torso
                # Raised arm
                cv2.line(frame, (sx + 20, head_y + 20), (sx + 35, head_y - 25), (170, 195, 225), 6)
                cv2.circle(frame, (sx + 36, head_y - 28), 7, (170, 195, 225), -1) # Raised hand
                p_box = [sx - 28, head_y - 32, sx + 45, sy + 35]
            else:
                # Normal seated posture
                head_x = sx
                head_y = sy - 28 + breathe
                cv2.circle(frame, (head_x, head_y), 17, (170, 195, 225), -1) # Head
                cv2.rectangle(frame, (sx - 22, head_y + 17), (sx + 22, sy + 25), (75, 85, 120), -1) # Torso
                p_box = [sx - 28, head_y - 20, sx + 28, sy + 35]

            simulated_objects.append({
                "box": p_box,
                "class_name": "person",
                "conf": 0.95
            })

            # Phone usage
            if is_phone:
                phone_x = sx + 12
                phone_y = sy + 5
                # Glowing smartphone screen
                cv2.rectangle(frame, (phone_x - 7, phone_y - 12), (phone_x + 7, phone_y + 12), (20, 20, 20), -1)
                cv2.rectangle(frame, (phone_x - 5, phone_y - 10), (phone_x + 5, phone_y + 10), (255, 240, 150), -1)
                simulated_objects.append({
                    "box": [phone_x - 10, phone_y - 15, phone_x + 10, phone_y + 15],
                    "class_name": "cell phone",
                    "conf": 0.92
                })

        return frame, simulated_objects
