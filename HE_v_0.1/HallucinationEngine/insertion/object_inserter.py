import os
import cv2
import numpy as np


class ObjectInserter:
    """
    Core object insertion module.

    This module is independent of CARLA.

    Input:
        RGB frame
        sprite image path
        2D bounding box

    Output:
        RGB frame with inserted object

    Supports:
        - alpha blending
        - brightness matching
        - soft edge blending
        - contact shadow
        - optional motion blur
    """

    def __init__(self):
        self.sprite_cache = {}

    # ============================================================
    # Public API
    # ============================================================

    def insert_sprite(
        self,
        frame_rgb,
        sprite_path,
        box_2d,
        alpha=1.0,
        horizontal_scale=1.0,
        vertical_scale=1.0,
        y_offset_ratio=0.0,
        match_brightness=True,
        add_shadow=True,
        soften_edges=True,
        motion_blur=False,
        debug_box=False,
        debug_color=(255, 0, 0),
    ):
        """
        Insert a sprite into a frame.

        Args:
            frame_rgb:
                RGB image as numpy array, shape H x W x 3

            sprite_path:
                path to object sprite image.
                PNG with alpha is preferred.

            box_2d:
                [x1, y1, x2, y2] target box in image coordinates

            alpha:
                global alpha multiplier

            horizontal_scale, vertical_scale:
                scale object relative to box size

            y_offset_ratio:
                move object vertically relative to box height.
                positive means move downward.

            match_brightness:
                adjust sprite brightness to local frame region

            add_shadow:
                add soft contact shadow under object

            soften_edges:
                blur alpha mask boundary slightly

            motion_blur:
                apply small horizontal blur to sprite

            debug_box:
                draw target box

        Returns:
            modified RGB frame
        """

        if frame_rgb is None:
            return frame_rgb

        if box_2d is None:
            return frame_rgb

        sprite_rgba = self._load_sprite_rgba(sprite_path)

        placement = self._compute_placement(
            frame_shape=frame_rgb.shape,
            box_2d=box_2d,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            y_offset_ratio=y_offset_ratio,
        )

        if placement is None:
            return frame_rgb

        (
            new_x1,
            new_y1,
            new_x2,
            new_y2,
            clip_x1,
            clip_y1,
            clip_x2,
            clip_y2,
            target_w,
            target_h,
        ) = placement

        resized_sprite = cv2.resize(
            sprite_rgba,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

        if motion_blur:
            resized_sprite = self._apply_motion_blur_rgba(resized_sprite)

        sprite_crop = self._crop_sprite_to_visible_region(
            resized_sprite=resized_sprite,
            new_x1=new_x1,
            new_y1=new_y1,
            clip_x1=clip_x1,
            clip_y1=clip_y1,
            clip_x2=clip_x2,
            clip_y2=clip_y2,
        )

        if sprite_crop is None:
            return frame_rgb

        output = frame_rgb.copy()

        if add_shadow:
            output = self._add_contact_shadow(
                frame_rgb=output,
                box_2d=[clip_x1, clip_y1, clip_x2, clip_y2],
                original_box=box_2d,
            )

        roi = output[clip_y1:clip_y2, clip_x1:clip_x2]

        blended_roi = self._blend_sprite_crop(
            roi_rgb=roi,
            sprite_crop_rgba=sprite_crop,
            alpha=alpha,
            match_brightness=match_brightness,
            soften_edges=soften_edges,
        )

        output[clip_y1:clip_y2, clip_x1:clip_x2] = blended_roi

        if debug_box:
            x1, y1, x2, y2 = [int(v) for v in box_2d]
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                debug_color,
                2,
            )

        return output

    # ============================================================
    # Sprite loading
    # ============================================================

    def _load_sprite_rgba(self, sprite_path):
        """
        Load sprite and convert to RGBA.

        Supports:
            - PNG with alpha
            - normal RGB/BGR image
        """

        if sprite_path in self.sprite_cache:
            return self.sprite_cache[sprite_path]

        if not os.path.exists(sprite_path):
            raise FileNotFoundError(f"Sprite not found: {sprite_path}")

        sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)

        if sprite is None:
            raise RuntimeError(f"Could not read sprite: {sprite_path}")

        if sprite.ndim == 3 and sprite.shape[2] == 4:
            # BGRA -> RGBA
            sprite_rgba = cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA)

        elif sprite.ndim == 3 and sprite.shape[2] == 3:
            # BGR -> RGB + full alpha
            sprite_rgb = cv2.cvtColor(sprite, cv2.COLOR_BGR2RGB)
            alpha = np.ones(
                (sprite_rgb.shape[0], sprite_rgb.shape[1], 1),
                dtype=np.uint8,
            ) * 255
            sprite_rgba = np.concatenate([sprite_rgb, alpha], axis=2)

        else:
            raise ValueError(
                f"Unsupported sprite shape {sprite.shape}. "
                "Expected 3-channel or 4-channel image."
            )

        self.sprite_cache[sprite_path] = sprite_rgba
        return sprite_rgba

    # ============================================================
    # Placement
    # ============================================================

    def _compute_placement(
        self,
        frame_shape,
        box_2d,
        horizontal_scale,
        vertical_scale,
        y_offset_ratio,
    ):
        """
        Compute where the resized sprite should be placed.
        """

        image_h, image_w = frame_shape[:2]

        x1, y1, x2, y2 = [int(v) for v in box_2d]

        if x2 <= x1 or y2 <= y1:
            return None

        box_w = x2 - x1
        box_h = y2 - y1

        target_w = int(box_w * horizontal_scale)
        target_h = int(box_h * vertical_scale)

        if target_w <= 2 or target_h <= 2:
            return None

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2 + box_h * y_offset_ratio)

        new_x1 = int(cx - target_w / 2)
        new_y1 = int(cy - target_h / 2)
        new_x2 = new_x1 + target_w
        new_y2 = new_y1 + target_h

        clip_x1 = max(0, new_x1)
        clip_y1 = max(0, new_y1)
        clip_x2 = min(image_w, new_x2)
        clip_y2 = min(image_h, new_y2)

        if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
            return None

        return (
            new_x1,
            new_y1,
            new_x2,
            new_y2,
            clip_x1,
            clip_y1,
            clip_x2,
            clip_y2,
            target_w,
            target_h,
        )

    def _crop_sprite_to_visible_region(
        self,
        resized_sprite,
        new_x1,
        new_y1,
        clip_x1,
        clip_y1,
        clip_x2,
        clip_y2,
    ):
        """
        Crop sprite if part of it lies outside image.
        """

        crop_x1 = clip_x1 - new_x1
        crop_y1 = clip_y1 - new_y1
        crop_x2 = crop_x1 + (clip_x2 - clip_x1)
        crop_y2 = crop_y1 + (clip_y2 - clip_y1)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        return resized_sprite[crop_y1:crop_y2, crop_x1:crop_x2]

    # ============================================================
    # Blending
    # ============================================================

    def _blend_sprite_crop(
        self,
        roi_rgb,
        sprite_crop_rgba,
        alpha,
        match_brightness,
        soften_edges,
    ):
        """
        Alpha blend sprite crop into ROI.
        """

        roi_float = roi_rgb.astype(np.float32)

        sprite_rgb = sprite_crop_rgba[:, :, :3].astype(np.float32)
        sprite_alpha = sprite_crop_rgba[:, :, 3].astype(np.float32) / 255.0

        if soften_edges:
            sprite_alpha = self._soften_alpha(sprite_alpha)

        if match_brightness:
            sprite_rgb = self._match_brightness(
                sprite_rgb=sprite_rgb,
                roi_rgb=roi_float,
                alpha_mask=sprite_alpha,
            )

        sprite_alpha = sprite_alpha * float(alpha)
        sprite_alpha = np.clip(sprite_alpha, 0.0, 1.0)
        sprite_alpha = np.expand_dims(sprite_alpha, axis=2)

        blended = sprite_alpha * sprite_rgb + (1.0 - sprite_alpha) * roi_float

        return np.clip(blended, 0, 255).astype(np.uint8)

    def _soften_alpha(self, alpha_mask):
        """
        Slightly blur alpha mask edges.
        """

        alpha = alpha_mask.astype(np.float32)

        # Blur only helps if mask is not entirely hard/full.
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

        return np.clip(alpha, 0.0, 1.0)

    def _match_brightness(self, sprite_rgb, roi_rgb, alpha_mask):
        """
        Match sprite brightness to the local background region.

        This is a simple and fast approximation.
        """

        mask = alpha_mask > 0.1

        if np.sum(mask) < 10:
            return sprite_rgb

        sprite_gray = cv2.cvtColor(
            np.clip(sprite_rgb, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        ).astype(np.float32)

        roi_gray = cv2.cvtColor(
            np.clip(roi_rgb, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        ).astype(np.float32)

        sprite_mean = float(np.mean(sprite_gray[mask]))
        roi_mean = float(np.mean(roi_gray))

        if sprite_mean < 1.0:
            return sprite_rgb

        ratio = roi_mean / sprite_mean

        # Clamp to avoid overcorrection
        ratio = max(0.65, min(ratio, 1.35))

        adjusted = sprite_rgb * ratio

        return np.clip(adjusted, 0, 255)

    # ============================================================
    # Shadow and blur
    # ============================================================

    def _add_contact_shadow(self, frame_rgb, box_2d, original_box=None):
        """
        Add a soft elliptical contact shadow below the object.

        This improves road contact realism.
        """

        output = frame_rgb.copy()

        image_h, image_w = output.shape[:2]

        x1, y1, x2, y2 = [int(v) for v in box_2d]

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        cx = int((x1 + x2) / 2)

        # Put shadow near bottom of object
        shadow_center_y = int(y2 - 0.08 * box_h)

        shadow_w = int(box_w * 0.85)
        shadow_h = int(box_h * 0.16)

        if shadow_w <= 2 or shadow_h <= 2:
            return output

        overlay = np.zeros((image_h, image_w), dtype=np.float32)

        cv2.ellipse(
            overlay,
            (cx, shadow_center_y),
            (shadow_w // 2, shadow_h // 2),
            0,
            0,
            360,
            1.0,
            -1,
        )

        overlay = cv2.GaussianBlur(overlay, (31, 31), 0)

        # Shadow strength
        shadow_strength = 0.28

        shadow = np.expand_dims(overlay * shadow_strength, axis=2)

        output_float = output.astype(np.float32)
        output_float = output_float * (1.0 - shadow)

        return np.clip(output_float, 0, 255).astype(np.uint8)

    def _apply_motion_blur_rgba(self, sprite_rgba):
        """
        Apply small horizontal motion blur to RGBA sprite.
        """

        kernel_size = 5

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size

        blurred = cv2.filter2D(sprite_rgba, -1, kernel)

        return blurred