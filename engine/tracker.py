import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class TrackedObject:
    def __init__(self, track_id, box, class_name, confidence, keypoints=None):
        self.track_id = track_id
        self.box = list(box)  # [x1, y1, x2, y2]
        self.class_name = class_name
        self.confidence = confidence
        self.keypoints = keypoints
        self.lost_frames = 0
        self.age = 1
        self.history = [self.get_center()]
        self.posture = "ATTENTIVE"
        self.attention_score = 92.0
        self.head_down_frames = 0
        self.phone_usage_frames = 0
        self.hand_raised = False
        self.hand_raised_frames = 0
        self.zone = "CLASS_FLOOR"

    def get_center(self):
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    def update(self, box, confidence, keypoints=None):
        self.box = list(box)
        self.confidence = confidence
        if keypoints is not None:
            self.keypoints = keypoints
        self.lost_frames = 0
        self.age += 1
        self.history.append(self.get_center())
        if len(self.history) > 30:
            self.history.pop(0)

class SimpleTracker:
    def __init__(self, max_lost=15, iou_threshold=0.3):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        """
        detections: list of dicts: {'box': [x1, y1, x2, y2], 'class_name': str, 'conf': float}
        Returns list of updated TrackedObjects.
        """
        person_dets = [d for d in detections if d['class_name'] == 'person']

        if not self.tracks:
            for det in person_dets:
                t = TrackedObject(self.next_id, det['box'], det['class_name'], det['conf'], det.get('keypoints'))
                self.tracks[self.next_id] = t
                self.next_id += 1
            return list(self.tracks.values())

        det_boxes = [d['box'] for d in person_dets]
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid].box for tid in track_ids]

        matched_tracks = set()
        matched_dets = set()

        if track_boxes and det_boxes:
            iou_matrix = np.zeros((len(track_boxes), len(det_boxes)), dtype=np.float32)
            for i, tbox in enumerate(track_boxes):
                for j, dbox in enumerate(det_boxes):
                    iou_matrix[i, j] = calculate_iou(tbox, dbox)

            # Greedy match based on max IoU
            while True:
                if iou_matrix.size == 0:
                    break
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[i]
                d = person_dets[j]
                self.tracks[tid].update(d['box'], d['conf'], d.get('keypoints'))
                matched_tracks.add(tid)
                matched_dets.add(j)
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # Check unmatched tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].lost_frames += 1

        # Remove dead tracks quickly (prevent ghost boxes)
        dead_ids = [tid for tid, t in self.tracks.items() if t.lost_frames > 4]
        for tid in dead_ids:
            del self.tracks[tid]

        # Add new tracks for unmatched detections
        for j, d in enumerate(person_dets):
            if j not in matched_dets:
                # If only one person is being tracked or present, maintain Student 1
                if len(person_dets) == 1 and not self.tracks:
                    assigned_id = 1
                else:
                    assigned_id = self.next_id
                    self.next_id += 1
                t = TrackedObject(assigned_id, d['box'], d['class_name'], d['conf'], d.get('keypoints'))
                self.tracks[assigned_id] = t

        # Return only currently confirmed active detections
        return [t for t in self.tracks.values() if t.lost_frames == 0]
