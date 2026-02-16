import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from moviepy.editor import VideoFileClip, TextClip

# --- Configuration Helpers ---
def lerp(start, end, p):
    return start + (end - start) * p

# --- 1. Face Tracking ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

def get_face_data(video_path):
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    face_data = []

    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.face_landmarks:
            lm = results.face_landmarks[0]
            cx, cy = int(lm[4].x * w), int(lm[4].y * h)
            fh = int(abs(lm[152].y - lm[10].y) * h)
            fw = int(abs(lm[454].x - lm[234].x) * w)
            face_data.append((cx, cy, fw, fh))
        else:
            if face_data: face_data.append(face_data[-1])
            else: face_data.append((w//2, h//2, 100, 100))
    cap.release()
    return face_data, fps, (w, h)

def smooth_data(data, alpha=0.1):
    smoothed = []
    curr = list(data[0])
    for frame in data:
        new_state = []
        for i in range(4):
            val = (alpha * frame[i]) + ((1 - alpha) * curr[i])
            new_state.append(val)
        curr = new_state
        smoothed.append(tuple(map(int, curr)))
    return smoothed

# --- 2. The Main Effect Logic ---

def create_zoom_follow_effect(
    input_path,
    output_path,
    zoom_max=1.5,
    t_start=0,
    t_end=5,
    face_side="right",
    text_config=None
):
    print("1. Analyzing Face Trajectory...")
    raw_data, fps, (w, h) = get_face_data(input_path)
    face_data = smooth_data(raw_data, alpha=0.05)
    print(f"   Analyzed {len(face_data)} frames.")

    clip = VideoFileClip(input_path)

    # -- Prepare Text Asset --
    text_img = None
    text_mask = None
    if text_config:
        content = text_config.get('content', "Text")
        txt = TextClip(content, fontsize=70, color=text_config.get('color', 'white'), font='Arial-Bold')
        text_img = txt.get_frame(0)
        text_mask = txt.mask.get_frame(0) if txt.mask else np.ones(text_img.shape[:2])

    def get_state(t):
        idx = int(t * fps)
        idx = max(0, min(idx, len(face_data) - 1))
        return face_data[idx]

    def overlay_text(bg_frame, fx, fy, fw, fh, opacity):
        """Draws text on the FINAL frame. Coordinates are in Screen Space."""
        if text_img is None or opacity <= 0:
            return bg_frame

        pos_mode = text_config.get('position', 'left')
        margin = text_config.get('margin', 1.3)

        th, tw = text_img.shape[:2]
        bh, bw = bg_frame.shape[:2]

        if pos_mode == 'left':
            tx = int(fx - (fw / 2 * margin) - tw)
            ty = int(fy - th // 2)
        elif pos_mode == 'right':
            tx = int(fx + (fw / 2 * margin))
            ty = int(fy - th // 2)
        elif pos_mode == 'top':
            tx = int(fx - tw // 2)
            ty = int(fy - (fh / 2 * margin) - th)
        else:  # bottom
            tx = int(fx - tw // 2)
            ty = int(fy + (fh / 2 * margin))

        x1, y1 = max(0, tx), max(0, ty)
        x2, y2 = min(bw, tx + tw), min(bh, ty + th)

        if x1 >= x2 or y1 >= y2:
            return bg_frame

        sx1 = x1 - tx
        sy1 = y1 - ty
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        alpha = (text_mask[sy1:sy2, sx1:sx2, np.newaxis] * opacity).astype(np.float32)
        roi = bg_frame[y1:y2, x1:x2].astype(np.float32)
        txt_roi = text_img[sy1:sy2, sx1:sx2].astype(np.float32)

        result = bg_frame.copy()
        result[y1:y2, x1:x2] = (txt_roi * alpha + roi * (1 - alpha)).astype(np.uint8)
        return result

    def process_frame(get_frame, t):
        frame = get_frame(t)

        # 1. Animation progress
        if t < t_start: p = 0
        elif t > t_end: p = 1
        else: p = (t - t_start) / (t_end - t_start)
        p = p * p * (3 - 2 * p)

        # 2. Get face data
        current_zoom = lerp(1.0, zoom_max, p)
        raw_fx, raw_fy, raw_fw, raw_fh = get_state(t)

        # 3. Calculate affine transform
        target_center_x = w / 2
        if face_side == "left":
            target_center_x = lerp(w / 2, w * 0.25, p)
        elif face_side == "right":
            target_center_x = lerp(w / 2, w * 0.75, p)

        shift_x = target_center_x - (raw_fx * current_zoom)
        shift_y = (h / 2) - (raw_fy * current_zoom)

        M = np.float32([
            [current_zoom, 0, shift_x],
            [0, current_zoom, shift_y]
        ])

        # 4. Background (blurred fill)
        small_bg = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
        blurred_bg = cv2.GaussianBlur(small_bg, (31, 31), 0)
        full_bg = cv2.resize(blurred_bg, (w, h))
        full_bg = (full_bg * 0.9).astype(np.uint8)

        # 5. Warp foreground
        fg_warped = cv2.warpAffine(
            frame, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )

        # 6. Mask: erode + blur for soft edge
        mask_solid = np.ones((h, w), dtype=np.uint8) * 255
        mask_warped = cv2.warpAffine(
            mask_solid, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        erosion_size = 30
        kernel = np.ones((erosion_size, erosion_size), np.uint8)
        mask_eroded = cv2.erode(mask_warped, kernel)
        mask_blurred = cv2.GaussianBlur(mask_eroded, (101, 101), 0)
        alpha = mask_blurred.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=2)

        # 7. Composite fg over bg
        out = (fg_warped.astype(float) * alpha) + (full_bg.astype(float) * (1.0 - alpha))
        out = out.astype(np.uint8)

        # 8. Transform face coords to screen space, then overlay text
        if text_config:
            txt_start = text_config.get('t_start', t_start)
            txt_end = text_config.get('t_end', t_end)

            if txt_start <= t <= txt_end:
                txt_p = (t - txt_start) / (txt_end - txt_start) if txt_end > txt_start else 1
                text_opacity = min(txt_p * 4, 1.0)

                screen_fx = raw_fx * current_zoom + shift_x
                screen_fy = raw_fy * current_zoom + shift_y
                screen_fw = raw_fw * current_zoom
                screen_fh = raw_fh * current_zoom

                out = overlay_text(out, screen_fx, screen_fy, screen_fw, screen_fh, text_opacity)

        return out

    video_layer = clip.fl(process_frame)
    video_layer.write_videofile(output_path, fps=24, codec='libx264')

# Usage
create_zoom_follow_effect(
    "vid.mp4", "output_clean.mp4",
    zoom_max=1.3,
    face_side="right",
    text_config={
        "content": "Hello this is Sidhant!",
        "position": "left",
        "color": "yellow",
    }
)
