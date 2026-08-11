from HallucinationEngine.camera_only.road_model import SimpleRoadModel
from HallucinationEngine.camera_only.image_track import ImageAdversaryTrack
from HallucinationEngine.camera_only.visual_road_estimator import VisualRoadEstimator
from HallucinationEngine.insertion import ObjectInserter


class CameraOnlyHE:
    """
    Camera-only Hallucination Engine.

    Uses only RGB frame, frame_id and dt.

    No CARLA map.
    No CARLA ego transform.
    No CARLA route waypoints.
    No CARLA camera transform.
    """

    def __init__(self, config):
        self.config = config

        self.road_model = SimpleRoadModel(
            image_width=config.image_width,
            image_height=config.image_height,
            road_center_x_ratio=getattr(config, "road_center_x_ratio", 0.5),
            horizon_y_ratio=getattr(config, "horizon_y_ratio", 0.45),
            spawn_y_ratio=getattr(config, "spawn_y_ratio", 0.55),
            target_y_ratio=getattr(config, "target_y_ratio", 0.88),
        )

        self.use_visual_road_estimator = getattr(
            config,
            "use_visual_road_estimator",
            True,
        )

        self.road_estimator = VisualRoadEstimator(
            image_width=config.image_width,
            image_height=config.image_height,
            default_center_x_ratio=getattr(config, "road_center_x_ratio", 0.5),
            roi_y_start_ratio=getattr(config, "road_estimator_roi_y_start_ratio", 0.55),
            roi_y_end_ratio=getattr(config, "road_estimator_roi_y_end_ratio", 0.95),
            smoothing_alpha=getattr(config, "road_center_smoothing_alpha", 0.75),
        )

        self.inserter = ObjectInserter()

        self.track = None
        self.active = False
        self.start_frame = None
        self.last_finished_frame = None

        self.latest_road_estimate = None

    def process(self, frame_rgb, frame_id, dt):
        if not getattr(self.config, "enabled", True):
            return frame_rgb, {
                "frame": frame_id,
                "hallucination": {
                    "active": False,
                    "mode": "camera_only",
                }
            }

        self.latest_road_estimate = self._estimate_road(frame_rgb)

        # Start first track or repeat after finish.
        if self.track is None:
            if self._can_start_new_track(frame_id):
                self._start_track(frame_id)

        if self.track is None:
            return frame_rgb, {
                "frame": frame_id,
                "hallucination": {
                    "active": False,
                    "mode": "camera_only",
                    "reason": "waiting_for_trigger",
                },
                "road_estimate": self.latest_road_estimate,
            }

        elapsed = frame_id - self.start_frame

        if elapsed > self.config.duration_frames:
            self.track.active = False
            self.last_finished_frame = frame_id

            if getattr(self.config, "repeat_after_finish", True):
                self.track = None
                self.active = False

            return frame_rgb, {
                "frame": frame_id,
                "hallucination": {
                    "active": False,
                    "mode": "camera_only",
                    "reason": "duration_finished",
                },
                "road_estimate": self.latest_road_estimate,
            }

        self.track.update(dt)

        # Stronger turn/road-center adaptation.
        self._apply_dynamic_road_center_to_track()

        box_2d = self._compute_box()
        self.track.set_box(box_2d)

        if box_2d is None:
            return frame_rgb, {
                "frame": frame_id,
                "hallucination": {
                    "active": False,
                    "mode": "camera_only",
                    "reason": "box_failed",
                },
                "road_estimate": self.latest_road_estimate,
            }

        modified = self.inserter.insert_sprite(
            frame_rgb=frame_rgb,
            sprite_path=getattr(
                self.config,
                "sprite_path",
                "assets/sprites/car_front.png"
            ),
            box_2d=box_2d,

            alpha=getattr(self.config, "sprite_alpha", 1.0),
            horizontal_scale=getattr(self.config, "sprite_horizontal_scale", 1.10),
            vertical_scale=getattr(self.config, "sprite_vertical_scale", 1.15),
            y_offset_ratio=getattr(self.config, "sprite_y_offset_ratio", 0.0),

            match_brightness=getattr(self.config, "match_brightness", True),
            add_shadow=getattr(self.config, "add_shadow", True),
            soften_edges=getattr(self.config, "soften_edges", True),
            motion_blur=getattr(self.config, "motion_blur", False),

            debug_box=getattr(self.config, "draw_sprite_debug_box", False),
            debug_color=getattr(self.config, "box_color", (255, 0, 0)),
        )

        he_meta = self.track.to_dict()
        he_meta.update({
            "active": True,
            "mode": "camera_only",
            "render_mode": "sprite",
            "object_class": "vehicle",
            "motion_type": "image_space_wrong_way_approach",
            "road_estimate_used": self.latest_road_estimate,
        })

        return modified, {
            "frame": frame_id,
            "dt": dt,
            "hallucination": he_meta,
            "road_estimate": self.latest_road_estimate,
        }

    def _can_start_new_track(self, frame_id):
        if frame_id < self.config.trigger_frame:
            return False

        if self.last_finished_frame is None:
            return True

        if not getattr(self.config, "repeat_after_finish", True):
            return False

        gap = int(getattr(self.config, "repeat_gap_frames", 30))

        return frame_id - self.last_finished_frame >= gap

    def _estimate_road(self, frame_rgb):
        if not self.use_visual_road_estimator:
            center_x = self.road_model.get_road_center_x()

            return {
                "road_center_x": float(center_x),
                "road_center_x_ratio": float(center_x / self.config.image_width),
                "confidence": 0.0,
                "method": "fixed_config",
            }

        return self.road_estimator.estimate(frame_rgb)

    def _start_track(self, frame_id):
        start_center = self.road_model.get_spawn_point()
        target_center = self.road_model.get_target_point()

        if self.latest_road_estimate is not None:
            road_center_x = self.latest_road_estimate.get(
                "road_center_x",
                start_center[0],
            )

            start_center = (road_center_x, start_center[1])
            target_center = (road_center_x, target_center[1])

        self.track = ImageAdversaryTrack(
            object_id=getattr(self.config, "object_id", "he_cam_adv_001"),
            scenario_type=getattr(self.config, "scenario_type", "camera_only_wrong_way"),
            start_center=start_center,
            target_center=target_center,
            duration_frames=getattr(self.config, "duration_frames", 100),
        )

        self.active = True
        self.start_frame = frame_id

    def _apply_dynamic_road_center_to_track(self):
        """
        Stronger visual road following.

        During turning:
            estimated road center shifts
            adversary x-position follows it
        """

        if self.track is None:
            return

        if self.latest_road_estimate is None:
            return

        road_center_x = self.latest_road_estimate.get("road_center_x", None)

        if road_center_x is None:
            return

        confidence = float(self.latest_road_estimate.get("confidence", 0.0))

        # Configurable following strength.
        base_gain = float(getattr(self.config, "road_follow_base_gain", 0.12))
        conf_gain = float(getattr(self.config, "road_follow_conf_gain", 0.35))

        gain = base_gain + conf_gain * confidence
        gain = max(0.05, min(gain, 0.55))

        # Shift current position toward road center.
        self.track.center_u = (
            (1.0 - gain) * self.track.center_u
            + gain * float(road_center_x)
        )

        # Also update target and start path to prevent next update from fully resetting x.
        self.track.target_u = (
            (1.0 - gain) * self.track.target_u
            + gain * float(road_center_x)
        )

        self.track.start_u = (
            0.98 * self.track.start_u
            + 0.02 * float(road_center_x)
        )

    def _compute_box(self):
        center_u = self.track.center_u
        center_v = self.track.center_v

        width, height = self.road_model.estimate_box_size_from_y(
            center_y=center_v,
            min_width=getattr(self.config, "min_box_width", 70),
            max_width=getattr(self.config, "max_box_width", 520),
            min_height=getattr(self.config, "min_box_height", 55),
            max_height=getattr(self.config, "max_box_height", 430),
        )

        return self.road_model.make_box_from_center(
            center_x=center_u,
            center_y=center_v,
            width=width,
            height=height,
        )