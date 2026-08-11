import cv2
import numpy as np


class VisualRoadEstimator:
    """
    Lightweight camera-only road/lane center estimator.

    Goal:
        Estimate dynamic road center in image space.

    It does not use CARLA metadata.
    It uses only the RGB frame.

    This is intentionally simple and fast for 20-30 FPS.
    """

    def __init__(
        self,
        image_width,
        image_height,
        default_center_x_ratio=0.5,
        roi_y_start_ratio=0.55,
        roi_y_end_ratio=0.95,
        smoothing_alpha=0.85,
    ):
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        self.default_center_x = self.image_width * float(default_center_x_ratio)

        self.roi_y_start_ratio = float(roi_y_start_ratio)
        self.roi_y_end_ratio = float(roi_y_end_ratio)

        # Higher value = smoother but slower adaptation.
        self.smoothing_alpha = float(smoothing_alpha)

        self.smoothed_center_x = self.default_center_x
        self.last_confidence = 0.0

    def estimate(self, frame_rgb):
        """
        Estimate road/lane center from RGB frame.

        Returns:
            dict with:
                road_center_x
                road_center_x_ratio
                confidence
                method
        """

        if frame_rgb is None:
            return self._fallback("no_frame")

        image_h, image_w = frame_rgb.shape[:2]

        if image_w <= 0 or image_h <= 0:
            return self._fallback("bad_frame_shape")

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        y1 = int(image_h * self.roi_y_start_ratio)
        y2 = int(image_h * self.roi_y_end_ratio)

        y1 = max(0, min(image_h - 1, y1))
        y2 = max(y1 + 1, min(image_h, y2))

        roi = gray[y1:y2, :]

        estimated_center, confidence = self._estimate_center_from_edges(roi, y_offset=y1)

        if estimated_center is None:
            return self._fallback("no_lane_edges")

        # Smooth center to prevent jitter.
        self.smoothed_center_x = (
            self.smoothing_alpha * self.smoothed_center_x
            + (1.0 - self.smoothing_alpha) * estimated_center
        )

        self.last_confidence = confidence

        return {
            "road_center_x": float(self.smoothed_center_x),
            "road_center_x_ratio": float(self.smoothed_center_x / image_w),
            "confidence": float(confidence),
            "method": "edge_lane_estimate",
        }

    def _estimate_center_from_edges(self, roi_gray, y_offset=0):
        """
        Estimate center using edge density in left and right image regions.

        This is not a full lane detector.
        It is a lightweight heuristic:
            - detect edges
            - find strong lower-road edge columns
            - estimate left/right lane evidence
            - road center is midpoint
        """

        roi_h, roi_w = roi_gray.shape[:2]

        if roi_h <= 0 or roi_w <= 0:
            return None, 0.0

        # Blur to reduce noise.
        blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)

        edges = cv2.Canny(blur, 60, 150)

        # Focus more on lower part of ROI because lane lines are clearer there.
        vertical_weights = np.linspace(0.4, 1.0, roi_h).reshape(-1, 1)
        weighted_edges = edges.astype(np.float32) * vertical_weights

        column_strength = np.sum(weighted_edges, axis=0)

        # Ignore extreme sides a bit.
        margin = int(roi_w * 0.08)
        valid_x1 = margin
        valid_x2 = roi_w - margin

        if valid_x2 <= valid_x1:
            return None, 0.0

        center_x = roi_w / 2.0

        left_region = column_strength[valid_x1:int(center_x)]
        right_region = column_strength[int(center_x):valid_x2]

        if len(left_region) == 0 or len(right_region) == 0:
            return None, 0.0

        left_peak_rel = int(np.argmax(left_region))
        right_peak_rel = int(np.argmax(right_region))

        left_peak_x = valid_x1 + left_peak_rel
        right_peak_x = int(center_x) + right_peak_rel

        left_score = float(column_strength[left_peak_x])
        right_score = float(column_strength[right_peak_x])

        total_score = left_score + right_score

        # If edge evidence is too weak, fallback.
        if total_score < 500.0:
            return None, 0.0

        # If both sides are detected, use midpoint.
        if left_score > 50.0 and right_score > 50.0 and right_peak_x > left_peak_x:
            estimated_center = (left_peak_x + right_peak_x) / 2.0
            confidence = min(1.0, total_score / 8000.0)
            return estimated_center, confidence

        # If only one side is strong, infer lane center using approximate lane width in pixels.
        approx_lane_width_px = roi_w * 0.24

        if left_score > right_score:
            estimated_center = left_peak_x + approx_lane_width_px / 2.0
        else:
            estimated_center = right_peak_x - approx_lane_width_px / 2.0

        estimated_center = max(0.0, min(float(roi_w - 1), estimated_center))
        confidence = min(0.55, total_score / 10000.0)

        return estimated_center, confidence

    def _fallback(self, reason):
        """
        Return smoothed/default road center when estimation fails.
        """

        # Slowly relax back to default center when confidence is low.
        self.smoothed_center_x = (
            0.97 * self.smoothed_center_x
            + 0.03 * self.default_center_x
        )

        return {
            "road_center_x": float(self.smoothed_center_x),
            "road_center_x_ratio": float(self.smoothed_center_x / self.image_width),
            "confidence": 0.0,
            "method": f"fallback_{reason}",
        }