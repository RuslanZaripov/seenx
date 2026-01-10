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


def pad_boxes_square(boxes, w, h, pad=0.25):
    """
    boxes: List[np.ndarray] of shape (N, 4)
    returns: List[np.ndarray] of shape (N, 4) with square boxes
    """
    padded_boxes = []

    for frame_boxes in boxes:
        if len(frame_boxes) == 0:
            padded_boxes.append(frame_boxes)
            continue

        frame_boxes = frame_boxes.astype(np.float32)

        x1, y1, x2, y2 = frame_boxes.T
        bw = x2 - x1
        bh = y2 - y1

        # square side = max(width, height)
        side = np.maximum(bw, bh)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        half = (1 + pad) * side / 2

        x1_p = np.clip(cx - half, 0, w)
        y1_p = np.clip(cy - half, 0, h)
        x2_p = np.clip(cx + half, 0, w)
        y2_p = np.clip(cy + half, 0, h)

        padded = np.stack([x1_p, y1_p, x2_p, y2_p], axis=1).astype(np.int32)
        padded_boxes.append(padded)

    return padded_boxes
