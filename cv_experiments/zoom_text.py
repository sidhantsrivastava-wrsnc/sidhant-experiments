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

# --- Overlay Abstraction ---

class Overlay:
    """Base class. Produces (rgb_array, alpha_mask) per frame."""
    def get_frame(self, t):
        raise NotImplementedError

class TextOverlay(Overlay):
    """Static. Renders once, returns same frame every call."""
    def __init__(self, content, color='white', fontsize=80, font='Arial-Bold',
                 max_width=None, max_height=None):
        fontsize = self._fit_fontsize(content, fontsize, color, font, max_width, max_height)
        if max_width:
            txt = TextClip(content, fontsize=fontsize, color=color, font=font,
                           size=(max_width, None), method='caption')
        else:
            txt = TextClip(content, fontsize=fontsize, color=color, font=font)
        self.img = txt.get_frame(0)
        if txt.mask:
            mask_frame = txt.mask.get_frame(0)
            if mask_frame.max() > 1.0:
                mask_frame = mask_frame / 255.0
            self.mask = mask_frame
        else:
            self.mask = np.ones(self.img.shape[:2], dtype=np.float32)

    @staticmethod
    def _fit_fontsize(content, fontsize, color, font, max_width, max_height):
        """Step down fontsize until text fits within max_width x max_height."""
        if not max_width and not max_height:
            return fontsize
        min_size = 20
        while fontsize >= min_size:
            if max_width:
                txt = TextClip(content, fontsize=fontsize, color=color, font=font,
                               size=(max_width, None), method='caption')
            else:
                txt = TextClip(content, fontsize=fontsize, color=color, font=font)
            th, tw = txt.get_frame(0).shape[:2]
            fits_w = tw <= max_width if max_width else True
            fits_h = th <= max_height if max_height else True
            if fits_w and fits_h:
                return fontsize
            fontsize -= 4
        return min_size

    def get_frame(self, t):
        return self.img, self.mask

class ImageOverlay(Overlay):
    """Static. Loads a PNG/image with alpha channel."""
    def __init__(self, path):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
            self.img = raw[:, :, :3]
            self.mask = raw[:, :, 3] / 255.0
        else:
            self.img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            self.mask = np.ones(raw.shape[:2], dtype=np.float32)

    def get_frame(self, t):
        return self.img, self.mask

class ClipOverlay(Overlay):
    """Animated. Samples from a VideoFileClip at time t."""
    def __init__(self, path):
        self.clip = VideoFileClip(path, has_mask=True)

    def get_frame(self, t):
        t_clamped = min(t, self.clip.duration - 0.01)
        img = self.clip.get_frame(t_clamped)
        if self.clip.mask:
            mask = self.clip.mask.get_frame(t_clamped)
            if mask.max() > 1.0:
                mask = mask / 255.0
        else:
            mask = np.ones(img.shape[:2], dtype=np.float32)
        return img, mask

def create_overlay(config):
    """Factory: returns an Overlay from an overlay_config dict."""
    overlay_type = config.get('type', 'text')
    if overlay_type == 'text':
        return TextOverlay(
            content=config.get('content', 'Text'),
            color=config.get('color', 'white'),
            fontsize=config.get('fontsize', 80),
            font=config.get('font', 'Arial-Bold'),
            max_width=config.get('_avail_w'),
            max_height=config.get('_avail_h'),
        )
    elif overlay_type == 'image':
        return ImageOverlay(path=config['path'])
    elif overlay_type == 'clip':
        return ClipOverlay(path=config['path'])
    else:
        raise ValueError(f"Unknown overlay type: {overlay_type}")

# --- 1. Advanced Face Tracking (Position + Size) ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

def get_face_data(video_path):
    """
    Returns a list of tuples: (nose_x, nose_y, face_width, face_height)
    We track size so we know where NOT to put the text.
    """
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

    face_data = [] # Stores (cx, cy, width, height)
    frame_idx = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]

            # Key Landmarks (same indices as legacy API)
            nose = landmarks[4]       # Center
            chin = landmarks[152]     # Bottom
            forehead = landmarks[10]  # Top
            left_cheek = landmarks[234]  # Left (screen left)
            right_cheek = landmarks[454] # Right (screen right)

            # Coordinates
            cx, cy = int(nose.x * w), int(nose.y * h)

            # Calculate Face Dimensions (in pixels)
            face_h = int(abs(chin.y - forehead.y) * h)
            face_w = int(abs(right_cheek.x - left_cheek.x) * w)

            face_data.append((cx, cy, face_w, face_h))
        else:
            # Fallback: use previous data or center
            if face_data:
                face_data.append(face_data[-1])
            else:
                face_data.append((w//2, h//2, 100, 100)) # Default dummy

    cap.release()
    landmarker.close()
    return face_data, fps, (w, h)

def smooth_data(data, alpha=0.1):
    """Smooths X, Y, Width, and Height to prevent jitter."""
    smoothed = []
    # Initialize with first frame
    curr_state = list(data[0]) 
    
    for frame in data:
        new_state = []
        for i in range(4): # x, y, w, h
            val = (alpha * frame[i]) + ((1 - alpha) * curr_state[i])
            new_state.append(val)
        curr_state = new_state
        smoothed.append(tuple(map(int, curr_state)))
        
    return smoothed

# --- 2. The Main Effect Logic ---

def create_zoom_follow_effect(
    input_path,
    output_path,
    zoom_max=1.5,
    t_start=0,
    t_end=5,
    face_side="right",
    overlay_config=None,
    text_config=None,  # Deprecated alias for overlay_config
):
    # Backward compat: text_config is treated as overlay_config if overlay_config not given
    if overlay_config is None and text_config is not None:
        overlay_config = text_config

    print("1. Analyzing Face Trajectory...")
    raw_data, fps, (w, h) = get_face_data(input_path)
    face_data = smooth_data(raw_data, alpha=0.05)

    clip = VideoFileClip(input_path)

    # -- A. PREPARE OVERLAY ASSET (Run once) --
    overlay = None
    if overlay_config:
        # Auto-size text to available space
        if overlay_config.get('type', 'text') == 'text':
            median_fw = np.median([d[2] for d in face_data])
            screen_fw = median_fw * zoom_max

            pos_mode = overlay_config.get('position', 'left')
            margin = overlay_config.get('margin', 1.8)
            pad = int(w * 0.03)

            if face_side == "right":
                face_cx = w * 0.72
            else:
                face_cx = w * 0.28

            if pos_mode == 'left':
                avail_w = int(face_cx - (screen_fw / 2 * margin) - pad)
            elif pos_mode == 'right':
                avail_w = int(w - (face_cx + screen_fw / 2 * margin) - pad)
            else:  # top or bottom
                avail_w = int(w * 0.5)

            avail_h = int(h * 0.6)
            avail_w = max(avail_w, 100)

            overlay_config = {**overlay_config, '_avail_w': avail_w, '_avail_h': avail_h}

        overlay = create_overlay(overlay_config)

    def get_state(t):
        idx = int(t * fps)
        idx = max(0, min(idx, len(face_data) - 1))
        return face_data[idx]

    # -- B. THE OVERLAY FUNCTION (Applied AFTER warp) --
    def composite_overlay(bg_frame, overlay_img, overlay_mask, fx, fy, fw, fh, opacity):
        """
        Composites an overlay onto the already-zoomed frame.
        overlay_img: RGB array of the overlay asset
        overlay_mask: Alpha mask (0-1 float) of the overlay asset
        fx, fy: Coordinates of the face in SCREEN SPACE (not video space)
        """
        if overlay_img is None or opacity <= 0:
            return bg_frame

        th, tw = overlay_img.shape[:2]
        bh, bw = bg_frame.shape[:2]

        # 1. Determine Position relative to the zoomed face
        pos_mode = overlay_config.get('position', 'left')
        margin = overlay_config.get('margin', 1.8)

        # Center of face (fx, fy) is already the correct screen coordinate
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

        # 2. Smart Boundary Checks (Prevent crashing if overlay goes off-screen)
        x1, y1 = max(0, tx), max(0, ty)
        x2, y2 = min(bw, tx + tw), min(bh, ty + th)

        # If completely off-screen, do nothing
        if x1 >= x2 or y1 >= y2:
            return bg_frame

        # 3. Calculate Slices
        sx1 = x1 - tx
        sy1 = y1 - ty
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)

        # 4. Blend
        roi = bg_frame[y1:y2, x1:x2].astype(np.float32)
        ovl_roi = overlay_img[sy1:sy2, sx1:sx2].astype(np.float32)

        alpha = overlay_mask[sy1:sy2, sx1:sx2]
        if alpha.ndim == 2:
            alpha = alpha[:, :, np.newaxis]  # Make it (H,W,1)

        weighted_alpha = alpha * opacity

        result = bg_frame.copy()
        result[y1:y2, x1:x2] = (ovl_roi * weighted_alpha + roi * (1.0 - weighted_alpha)).astype(np.uint8)

        return result

    # -- C. THE FRAME PROCESSOR --
    def process_frame(get_frame, t):
        frame = get_frame(t)  # Clean, raw frame

        # --- STEP 1: CALCULATE ZOOM GEOMETRY ---
        if t < t_start:
            p = 0
        elif t > t_end:
            p = 1
        else:
            p = (t - t_start) / (t_end - t_start)
        p = p * p * (3 - 2 * p)  # Ease

        current_zoom = lerp(1.0, zoom_max, p)
        raw_fx, raw_fy, raw_fw, raw_fh = get_state(t)

        # Target: Where we want the camera to look (Face Center)
        target_x = lerp(w / 2, raw_fx, p)
        target_y = lerp(h / 2, raw_fy, p)

        # Destination: Where that point should appear on screen
        if face_side == "left":
            face_dest_x = lerp(w / 2, w * 0.28, p)
        else:
            face_dest_x = lerp(w / 2, w * 0.72, p)
        face_dest_y = h / 2

        # Calculate Shift needed
        shift_x = face_dest_x - (target_x * current_zoom)
        shift_y = face_dest_y - (target_y * current_zoom)

        M = np.float32([
            [current_zoom, 0, shift_x],
            [0, current_zoom, shift_y]
        ])

        # --- STEP 2: WARP THE VIDEO (CRITICAL: DO THIS BEFORE TEXT) ---
        # Note: We are warping 'frame', which has NO text on it yet.
        warped_frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # --- STEP 3: CALCULATE NEW FACE POSITIONS ---
        # We need to know where the face moved to so we can put the text next to it.
        screen_fx = raw_fx * current_zoom + shift_x
        screen_fy = raw_fy * current_zoom + shift_y
        screen_fw = raw_fw * current_zoom
        screen_fh = raw_fh * current_zoom

        # --- STEP 4: APPLY EDGE FADE (Background Effect) ---
        edge_strip = int(w * 0.05)
        if face_side == "right":
            avg_color = warped_frame[:, :edge_strip].mean(axis=(0, 1))
        else:
            avg_color = warped_frame[:, w - edge_strip:].mean(axis=(0, 1))

        fade_width = int(w * 0.35)
        ramp = np.linspace(0, 1, fade_width).astype(np.float32)
        gradient = np.ones((h, w), dtype=np.float32)

        if face_side == "right":
            gradient[:, :fade_width] = ramp[np.newaxis, :]
        else:
            gradient[:, w - fade_width:] = ramp[::-1][np.newaxis, :]

        fade_alpha = ((1 - p) + p * gradient)[:, :, np.newaxis]
        fade_bg = np.full_like(warped_frame, avg_color, dtype=np.float32)

        # Apply fade to the warped frame
        final_bg = (warped_frame.astype(np.float32) * fade_alpha +
                    fade_bg * (1 - fade_alpha)).astype(np.uint8)

        # --- STEP 5: DRAW OVERLAY (LAST STEP) ---
        # Now we draw the UNWARPED overlay onto the WARPED background
        if overlay and overlay_config:
            ovl_start = overlay_config.get('t_start', t_start)
            ovl_end = overlay_config.get('t_end', t_end)

            if ovl_start <= t <= ovl_end:
                ovl_p = (t - ovl_start) / (ovl_end - ovl_start) if ovl_end > ovl_start else 1
                ovl_opacity = min(ovl_p * 4, 1.0)

                ovl_img, ovl_mask = overlay.get_frame(t)
                final_bg = composite_overlay(
                    final_bg, ovl_img, ovl_mask,
                    screen_fx, screen_fy, screen_fw, screen_fh,
                    ovl_opacity
                )

        return final_bg

    video_layer = clip.fl(process_frame)
    video_layer.write_videofile(output_path, fps=24, codec='libx264')

# --- Usage Example ---

ts = int(time.time())

create_zoom_follow_effect(
    input_path="vid.mp4",
    output_path=f"output_right_{ts}.mp4",
    zoom_max=1.1,
    t_start=1.0,
    t_end=9.0,
    face_side="right",
    text_config={
        "content": "Hello this is for Bansi!",
        "position": "left",
        "color": "yellow",
        "t_start": 7.0,
        "t_end": 10.0,
    }
)
