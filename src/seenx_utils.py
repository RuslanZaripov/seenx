import cv2
import numpy as np


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration


def resize_crop_center_np(frame: np.ndarray, size: int = 640) -> np.ndarray:
    """
    Resize preserving aspect ratio and center-crop to (size, size).

    Args:
        frame: np.ndarray of shape (H, W, C)
        size: target spatial size

    Returns:
        np.ndarray of shape (size, size, C)
    """
    if frame.ndim != 3:
        raise ValueError(f"Expected shape (H, W, C), got {frame.shape}")

    h, w, c = frame.shape

    scale = size / min(h, w)
    if scale != 1.0:
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    h, w, _ = frame.shape
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)

    frame = frame[top : top + size, left : left + size]

    return frame
