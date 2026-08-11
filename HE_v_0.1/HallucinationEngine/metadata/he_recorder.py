import os
import json
from datetime import datetime

import cv2
import numpy as np


class HERecorder:
    """
    Recorder for Hallucination Engine outputs.

    Saves:
        1. Modified RGB frames as PNG
        2. Frame-level hallucination metadata as JSON
        3. Run summary
    """

    def __init__(
        self,
        base_dir="he_outputs",
        save_only_active=True,
        save_frames=True,
        save_metadata=True,
        run_name=None,
    ):
        self.base_dir = base_dir
        self.save_only_active = save_only_active
        self.save_frames = save_frames
        self.save_metadata = save_metadata

        if run_name is None:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            run_name = f"run_{timestamp}"

        self.run_name = run_name

        self.run_dir = os.path.join(self.base_dir, self.run_name)
        self.frames_dir = os.path.join(self.run_dir, "frames")
        self.metadata_dir = os.path.join(self.run_dir, "metadata")

        os.makedirs(self.run_dir, exist_ok=True)

        if self.save_frames:
            os.makedirs(self.frames_dir, exist_ok=True)

        if self.save_metadata:
            os.makedirs(self.metadata_dir, exist_ok=True)

        self.total_frames_seen = 0
        self.total_frames_saved = 0
        self.total_active_frames = 0

        self.started_at = datetime.now().isoformat()

        print(f"[HERecorder] Saving outputs to: {self.run_dir}")

    def record(self, frame_rgb, updated_metadata):
        """
        Record one frame and its metadata.

        Args:
            frame_rgb: RGB numpy image after HE rendering
            updated_metadata: metadata returned by HE
        """

        self.total_frames_seen += 1

        he_info = updated_metadata.get("hallucination", {})
        is_active = bool(he_info.get("active", False))

        if is_active:
            self.total_active_frames += 1

        if self.save_only_active and not is_active:
            return

        frame_id = updated_metadata.get("frame", self.total_frames_seen)

        if self.save_frames:
            self._save_frame(frame_rgb, frame_id)

        if self.save_metadata:
            self._save_metadata(updated_metadata, frame_id)

        self.total_frames_saved += 1

    def _save_frame(self, frame_rgb, frame_id):
        """
        Save RGB image as PNG using cv2.
        cv2 expects BGR, so convert first.
        """

        if frame_rgb is None:
            return

        frame_path = os.path.join(
            self.frames_dir,
            f"frame_{int(frame_id):06d}.png"
        )

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, frame_bgr)

    def _save_metadata(self, metadata, frame_id):
        """
        Save JSON-safe metadata.
        """

        metadata_path = os.path.join(
            self.metadata_dir,
            f"frame_{int(frame_id):06d}.json"
        )

        safe_metadata = self._make_json_safe(metadata)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(safe_metadata, f, indent=4)

    def save_summary(self):
        """
        Save a run-level summary.
        """

        summary = {
            "run_name": self.run_name,
            "run_dir": self.run_dir,
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(),

            "total_frames_seen": self.total_frames_seen,
            "total_frames_saved": self.total_frames_saved,
            "total_active_frames": self.total_active_frames,

            "save_only_active": self.save_only_active,
            "save_frames": self.save_frames,
            "save_metadata": self.save_metadata,
        }

        summary_path = os.path.join(self.run_dir, "run_summary.json")

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        print(f"[HERecorder] Summary saved to: {summary_path}")

    def _make_json_safe(self, obj):
        """
        Convert CARLA objects / numpy types / other objects into JSON-safe values.
        """

        if obj is None:
            return None

        if isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, dict):
            return {
                str(k): self._make_json_safe(v)
                for k, v in obj.items()
                if self._is_safe_key(k)
            }

        if isinstance(obj, (list, tuple)):
            return [self._make_json_safe(v) for v in obj]

        # CARLA Location-like object
        if all(hasattr(obj, attr) for attr in ["x", "y", "z"]):
            return {
                "x": float(obj.x),
                "y": float(obj.y),
                "z": float(obj.z),
            }

        # CARLA Rotation-like object
        if all(hasattr(obj, attr) for attr in ["pitch", "yaw", "roll"]):
            return {
                "pitch": float(obj.pitch),
                "yaw": float(obj.yaw),
                "roll": float(obj.roll),
            }

        # CARLA Transform-like object
        if hasattr(obj, "location") and hasattr(obj, "rotation"):
            return {
                "location": self._make_json_safe(obj.location),
                "rotation": self._make_json_safe(obj.rotation),
            }

        # CARLA Vector3D-like object
        if all(hasattr(obj, attr) for attr in ["x", "y", "z"]):
            return {
                "x": float(obj.x),
                "y": float(obj.y),
                "z": float(obj.z),
            }

        # For unsupported objects, save string representation
        return str(obj)

    def _is_safe_key(self, key):
        """
        Skip very heavy fields if accidentally present.
        """

        blocked_keys = {
            "route_waypoints",
            "camera_intrinsics",
        }

        return str(key) not in blocked_keys