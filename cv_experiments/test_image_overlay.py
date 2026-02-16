"""Quick test: complex image/GIF overlay on zoom_text effect."""
import time

import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
from zoom_text import create_zoom_follow_effect


def build_complex_base(size):
    """Create a richer sticker with soft gradients, rings, and text."""
    center = (size // 2, size // 2)
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    radius = size * 0.46

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    in_circle = dist <= radius
    edge = np.clip(1.0 - dist / radius, 0.0, 1.0)

    # Neon radial gradient core
    rgba[..., 0] = (40 + (1.0 - edge) * 130).astype(np.uint8)   # R
    rgba[..., 1] = (120 + edge * 110).astype(np.uint8)          # G
    rgba[..., 2] = (255 - edge * 160).astype(np.uint8)          # B
    rgba[..., 3] = (in_circle * (50 + edge * 190)).astype(np.uint8)

    # Decorative rings
    ring1 = np.abs(dist - radius * 0.70) < 3
    ring2 = np.abs(dist - radius * 0.45) < 2
    rgba[ring1, :3] = (255, 255, 255)
    rgba[ring1, 3] = 240
    rgba[ring2, :3] = (15, 10, 30)
    rgba[ring2, 3] = 220

    # Small stars around edge
    for angle_deg in range(0, 360, 45):
        angle = np.deg2rad(angle_deg)
        sx = int(center[0] + np.cos(angle) * radius * 0.85)
        sy = int(center[1] + np.sin(angle) * radius * 0.85)
        cv2.drawMarker(
            rgba,
            (sx, sy),
            color=(255, 255, 255, 220),
            markerType=cv2.MARKER_STAR,
            markerSize=16,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

    # Center text and subtle shadow
    cv2.putText(
        rgba,
        "WOW",
        (34, center[1] + 16),
        cv2.FONT_HERSHEY_DUPLEX,
        1.8,
        (0, 0, 0, 170),
        6,
        cv2.LINE_AA,
    )
    cv2.putText(
        rgba,
        "WOW",
        (34, center[1] + 16),
        cv2.FONT_HERSHEY_DUPLEX,
        1.8,
        (255, 245, 80, 255),
        3,
        cv2.LINE_AA,
    )
    return rgba


def build_animated_gif(path, size=240, n_frames=24):
    """Create an animated sticker GIF with pulsing glow and rotation."""
    base = build_complex_base(size)
    center = (size // 2, size // 2)
    frames_rgb = []

    for i in range(n_frames):
        t = i / n_frames
        pulse = 0.88 + 0.22 * np.sin(2 * np.pi * t)
        rot_deg = 8 * np.sin(2 * np.pi * t)

        M = cv2.getRotationMatrix2D(center, rot_deg, pulse)
        transformed = cv2.warpAffine(
            base,
            M,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # Add moving sparkle for extra motion
        sparkle = transformed.copy()
        angle = 2 * np.pi * t
        sx = int(center[0] + np.cos(angle) * (size * 0.30))
        sy = int(center[1] + np.sin(angle) * (size * 0.30))
        cv2.circle(sparkle, (sx, sy), 8, (255, 255, 255, 255), -1, cv2.LINE_AA)

        # Composite on black for GIF compatibility
        alpha = (sparkle[..., 3:4].astype(np.float32) / 255.0)
        rgb = sparkle[..., :3].astype(np.float32)
        frame_rgb = (rgb * alpha).astype(np.uint8)
        frames_rgb.append(frame_rgb)

    clip = ImageSequenceClip(frames_rgb, fps=12)
    clip.write_gif(path, fps=12, logger=None)


# 1) Generate richer static PNG and animated GIF
size = 240
complex_png = build_complex_base(size)
cv2.imwrite("test_overlay.png", cv2.cvtColor(complex_png, cv2.COLOR_RGBA2BGRA))
print("Created test_overlay.png")

build_animated_gif("test_overlay.gif", size=size, n_frames=28)
print("Created test_overlay.gif")

ts = int(time.time())
create_zoom_follow_effect(
    input_path="vid.mp4",
    output_path=f"output_image_test_{ts}.mp4",
    zoom_max=1.1,
    t_start=1.0,
    t_end=6.0,
    face_side="right",
    overlay_config={
        "type": "clip",
        "path": "test_overlay.gif",
        "position": "left",
        "margin": 2.0,
    },
)
print(f"Done! Output: output_image_test_{ts}.mp4")
