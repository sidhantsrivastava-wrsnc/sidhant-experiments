import numpy as np
import cv2

from video_effects.effects.base import BaseEffect, EffectContext
from video_effects.schemas.effects import EffectCue, VideoInfo


class BlurEffect(BaseEffect):
    """Blur effect: gaussian, face_pixelate, background, radial."""

    def __init__(self):
        super().__init__()
        self._segmenter = None
        self._video_info: VideoInfo | None = None

    def setup(self, video_info: VideoInfo, effect_cues: list[EffectCue],
              *, cache_dir: str | None = None, video_path: str | None = None) -> None:
        self._cues = effect_cues
        self._video_info = video_info

        # Pre-init background segmentation if needed
        bg_cues = [
            c for c in effect_cues
            if c.blur_params and c.blur_params.blur_type == "background"
        ]
        if bg_cues:
            self._setup_segmentation()

    def _setup_segmentation(self) -> None:
        """Initialize MediaPipe selfie segmentation for background blur."""
        import mediapipe as mp
        self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1  # landscape model (faster)
        )

    def apply_frame(
        self, frame: np.ndarray, timestamp: float, context: EffectContext
    ) -> np.ndarray:
        active_cues = self.get_active_cues(timestamp)
        if not active_cues:
            return frame

        result = frame.copy()

        for cue in active_cues:
            params = cue.blur_params
            if params is None:
                continue

            if params.blur_type == "gaussian":
                result = self._apply_gaussian(result, params.radius, params.target_region)
            elif params.blur_type == "face_pixelate":
                result = self._apply_face_pixelate(result, params.radius)
            elif params.blur_type == "background":
                result = self._apply_background_blur(result, params.radius)
            elif params.blur_type == "radial":
                result = self._apply_radial_blur(result, params.radius)

        return result

    def _apply_gaussian(self, frame: np.ndarray, radius: float, region) -> np.ndarray:
        """Apply Gaussian blur to a target region."""
        h, w = frame.shape[:2]
        x1 = int(region.x * w)
        y1 = int(region.y * h)
        x2 = int((region.x + region.width) * w)
        y2 = int((region.y + region.height) * h)

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return frame

        k = int(radius) * 2 + 1  # Must be odd
        roi = frame[y1:y2, x1:x2]
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return frame

    def _apply_face_pixelate(self, frame: np.ndarray, radius: float) -> np.ndarray:
        """Detect face and pixelate the region."""
        import mediapipe as mp

        h, w = frame.shape[:2]
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as face_det:
            results = face_det.process(frame)

        if not results.detections:
            return frame

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            x2, y2 = min(w, x1 + bw), min(h, y1 + bh)

            if x2 <= x1 or y2 <= y1:
                continue

            # Pixelate: downscale then upscale
            roi = frame[y1:y2, x1:x2]
            pixel_size = max(2, int(radius / 3))
            small = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
            frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        return frame

    def _apply_background_blur(self, frame: np.ndarray, radius: float) -> np.ndarray:
        """Blur background using selfie segmentation mask."""
        if self._segmenter is None:
            return frame

        results = self._segmenter.process(frame)
        mask = results.segmentation_mask  # float32 [0, 1]

        # Refine mask edges with bilateral filter
        mask = cv2.bilateralFilter(mask, 9, 75, 75)

        # Threshold and smooth
        mask = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        k = int(radius) * 2 + 1
        blurred = cv2.GaussianBlur(frame, (k, k), 0)

        # Composite: person pixels from original, background from blurred
        mask_3ch = mask[:, :, np.newaxis]
        result = (frame * mask_3ch + blurred * (1.0 - mask_3ch)).astype(np.uint8)
        return result

    def _apply_radial_blur(self, frame: np.ndarray, radius: float) -> np.ndarray:
        """Radial zoom blur from center (adapted from zoom_bounce._apply_zoom_blur_cpu)."""
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        strength = min(radius / 50.0, 1.0)  # Normalize radius to 0-1 strength
        n_samples = 6
        base_zoom = 1.0
        spread = 0.05 * strength * base_zoom

        accum = np.zeros_like(frame, dtype=np.float32)
        for i in range(n_samples):
            t = (i / max(n_samples - 1, 1)) * 2.0 - 1.0
            dz = t * spread
            sz = base_zoom + dz
            M = np.float32([
                [sz, 0, cx * (1.0 - sz)],
                [0, sz, cy * (1.0 - sz)],
            ])
            sample = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            accum += sample.astype(np.float32)

        accum /= n_samples
        orig_f = frame.astype(np.float32)
        blended = orig_f + (accum - orig_f) * strength
        np.clip(blended, 0, 255, out=blended)
        return blended.astype(np.uint8)
