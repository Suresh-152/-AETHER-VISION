import cv2
import numpy as np

class ClassroomHeatmap:
    def __init__(self, width=640, height=360, decay=0.985):
        self.width = width
        self.height = height
        self.decay = decay
        self.accumulated = np.zeros((self.height, self.width), dtype=np.float32)

    def update(self, tracks):
        """Update occupancy heat based on current tracked students."""
        self.accumulated *= self.decay

        for t in tracks:
            cx, cy = t.get_center()
            ix = int(np.clip(cx * (self.width / 640.0), 0, self.width - 1))
            iy = int(np.clip(cy * (self.height / 360.0), 0, self.height - 1))

            # Add a Gaussian blob at student location
            cv2.circle(self.accumulated, (ix, iy), radius=28, color=1.0, thickness=-1)

    def generate_colored_map(self):
        """Generate a colorized heatmap representation."""
        norm = np.clip(self.accumulated, 0, 10.0) / 10.0
        norm_uint8 = (norm * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(norm_uint8, (25, 25), 0)
        color_map = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

        # Mask out very low values so cold areas are transparent/dark
        mask = blurred < 20
        color_map[mask] = 0
        return color_map

    def blend_overlay(self, frame, alpha=0.45):
        """Overlay heatmap onto source video frame."""
        h, w = frame.shape[:2]
        colored = self.generate_colored_map()
        resized_heat = cv2.resize(colored, (w, h), interpolation=cv2.INTER_LINEAR)

        # Blend where heat is non-zero
        gray_heat = cv2.cvtColor(resized_heat, cv2.COLOR_BGR2GRAY)
        mask = gray_heat > 25
        output = frame.copy()
        output[mask] = cv2.addWeighted(frame[mask], 1 - alpha, resized_heat[mask], alpha, 0)
        return output
