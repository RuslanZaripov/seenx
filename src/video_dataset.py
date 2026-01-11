from PIL import Image
from typing import Callable, Optional, List, Union
import cv2
from torch.utils.data import IterableDataset
import numpy as np
from .logger import Logger

logger = Logger(show=True).get_logger()


class VideoBatchDataset(IterableDataset):
    """
    Sequential video reader yielding batches of frames.

    Yields:
        frames: List[np.ndarray]  (RGB, HWC)
        indices: List[int]        (absolute frame indices)
    """

    def __init__(
        self,
        video_path: str,
        batch_size: int,
        transform=None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        stride: int = 1,
        frame_condition: Optional[Callable[[int], bool]] = None,
        frame_transform: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    ):
        self.video_path = video_path
        self.batch_size = batch_size
        self.transform = transform
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.stride = stride
        self.frame_condition = frame_condition
        self.frame_transform = frame_transform
        self.total_frames = None
        logger.info(
            f"Dataset {video_path}: batches_count={len(self)} {batch_size=} - {self.total_frames} frames"
        )

    def __len__(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = self.end_frame or total

        count = 0
        for frame_idx in range(self.start_frame, end):
            if self.frame_condition is not None and not self.frame_condition(frame_idx):
                continue
            if (frame_idx - self.start_frame) % self.stride == 0:
                count += 1

        cap.release()
        self.total_frames = count
        return (count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        Yields:
            frames: np.ndarray  (B, H, W, C)  RGB
            indices: List[int]  absolute frame indices
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = self.end_frame or total

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        frames: List[np.ndarray] = []
        indices: List[int] = []

        frame_idx = self.start_frame

        while frame_idx < end:
            ret, frame = cap.read()
            if not ret:
                break

            if self.frame_condition is not None and not self.frame_condition(frame_idx):
                frame_idx += 1
                continue

            if (frame_idx - self.start_frame) % self.stride == 0:
                if self.transform is not None:
                    frame = self.transform(frame)

                if self.frame_transform is not None:
                    frame = self.frame_transform(frame, frame_idx)

                if frame.shape[0] == 0:
                    frame_idx += 1
                    continue

                frames.append(frame)
                indices.extend([frame_idx] * len(frame))

                if len(frames) == self.batch_size:
                    yield np.concatenate(frames, axis=0), indices
                    frames, indices = [], []

            frame_idx += 1

        if len(frames) > 0:
            yield np.concatenate(frames, axis=0), indices

        cap.release()


class SpeakerFilteredVideoDataset(VideoBatchDataset):
    def __init__(self, speaker_probs, threshold=0.9, **kwargs):
        self.speaker_probs = speaker_probs
        self.threshold = threshold

        super().__init__(frame_condition=self.accept_frame, **kwargs)

    def accept_frame(self, frame_idx: int) -> bool:
        return self.speaker_probs[frame_idx] >= self.threshold


class SpecificFramesVideoDataset(VideoBatchDataset):
    """
    Dataset that yields only specific frames based on a given list of frame_ids.
    """

    def __init__(self, frame_ids: Union[List[int], np.ndarray, set], **kwargs):
        # store frame_ids as a set for fast lookup
        self.frame_ids = set(frame_ids)
        super().__init__(frame_condition=self.accept_frame, **kwargs)

    def accept_frame(self, frame_idx: int) -> bool:
        return frame_idx in self.frame_ids


class FaceCropVideoDataset(VideoBatchDataset):
    """
    Dataset that yields only specific frames based on a given list of frame_ids.
    """

    def __init__(
        self,
        frame_ids: Union[List[int], np.ndarray, set],
        crop_boxes: List[List[int]],
        **kwargs,
    ):
        # store frame_ids as a set for fast lookup
        self.frame_ids = set(frame_ids)
        self.crop_boxes = crop_boxes
        super().__init__(
            frame_condition=self.accept_frame,
            frame_transform=self.frame_transform,
            **kwargs,
        )

    def frame_transform(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        # Crop the frame using the provided box coordinates
        # frame of shape (B, H, W, C)
        assert frame.ndim == 4, (
            f"Expected frame with 4 dimensions (B, H, W, C), "
            f"got shape {frame.shape}"
        )
        assert frame.shape[0] == 1, f"Expected batch size B == 1, got {frame.shape[0]}"

        boxes = self.crop_boxes[frame_idx]
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 112, 112, 3), dtype=frame.dtype)

        crops = []
        for b in boxes.tolist():
            x1, y1, x2, y2 = b
            crop = cv2.resize(
                frame[0][y1:y2, x1:x2], (112, 112), interpolation=cv2.INTER_LINEAR
            )
            crops.append(crop)
        return np.array(crops)

    def accept_frame(self, frame_idx: int) -> bool:
        return frame_idx in self.frame_ids


class EmotionIterableDataset(IterableDataset):
    def __init__(self, video_dataset):
        self.video_dataset = video_dataset
        self.indices = []  # side channel

    def __iter__(self):
        self.indices.clear()
        for frames, indices in self.video_dataset:
            for frame, idx in zip(frames, indices):
                self.indices.append(idx)
                yield Image.fromarray(frame)
