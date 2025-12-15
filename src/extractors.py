from abc import ABC, abstractmethod
import numpy as np
import cv2
from tqdm import tqdm
from config import Config
from shot_segmentation import batch_shot_segmentation


class FrameContext:
    """
    Shared mutable context for a single frame.
    Populated once per frame and reused by all features.
    """

    def __init__(self):
        self.frame_rgb = None
        self.frame_shape = None
        self.frame_idx = None

        self.face_box = None
        self.face_crop = None
        self.keypoints = None


class FeatureExtractor(ABC):
    name: str
    requires_face: bool = False
    requires_keypoints: bool = False

    @abstractmethod
    def init_storage(self, total_frames: int):
        pass

    @abstractmethod
    def process_frame(self, ctx: FrameContext):
        pass

    @abstractmethod
    def finalize(self) -> list | dict:
        pass


class FaceScreenRatioFeature(FeatureExtractor):
    name = "face_screen_ratio"
    requires_face = True

    def init_storage(self, total_frames: int):
        self.values = [0.0] * total_frames

    def process_frame(self, ctx: FrameContext):
        x1, y1, x2, y2 = ctx.face_box
        h, w = ctx.frame_shape[:2]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        self.values[ctx.frame_idx] = area / (h * w)

    def finalize(self):
        return self.values


class TextProbFeature(FeatureExtractor):
    name = "text_prob"

    def __init__(self, ocr_model):
        self.ocr = ocr_model

    def init_storage(self, total_frames: int):
        self.values = [0.0] * total_frames

    def process_frame(self, ctx: FrameContext):
        results = self.ocr.readtext(ctx.frame_rgb)
        self.values[ctx.frame_idx] = (
            float(np.mean([c for _, _, c in results])) if results else 0.0
        )

    def finalize(self):
        return self.values


class MotionSpeedFeature(FeatureExtractor):
    name = "motion_speed"
    requires_keypoints = True

    def __init__(self, conf_thr: float):
        self.conf_thr = conf_thr

    def init_storage(self, total_frames: int):
        self.values = [0.0] * total_frames
        self.prev_keypoints = None
        self.prev_frame_idx = None

    def process_frame(self, ctx: FrameContext):
        if self.prev_keypoints is None:
            self.prev_keypoints = ctx.keypoints
            self.prev_frame_idx = ctx.frame_idx
            return

        prev = self.prev_keypoints
        curr = ctx.keypoints

        if prev.size == 0 or curr.size == 0:
            return

        p = prev[0]
        c = curr[0]

        vp = p[p[:, 2] > self.conf_thr][:, :2]
        vc = c[c[:, 2] > self.conf_thr][:, :2]

        if len(vp) == 0 or len(vc) == 0:
            return

        n = min(len(vp), len(vc))
        dist = np.linalg.norm(vc[:n] - vp[:n], axis=1).mean()

        self.values[ctx.frame_idx] = float(dist)

        self.prev_keypoints = curr
        self.prev_frame_idx = ctx.frame_idx

    def finalize(self):
        return self.values


class EmotionFeature(FeatureExtractor):
    name = "emotion"
    requires_face = True

    def __init__(self, pipe, batch_size: int):
        self.pipe = pipe
        self.batch_size = batch_size

    def init_storage(self, total_frames: int):
        self.values = {
            "angry": [0.0] * total_frames,
            "disgust": [0.0] * total_frames,
            "fear": [0.0] * total_frames,
            "happy": [0.0] * total_frames,
            "sad": [0.0] * total_frames,
            "surprise": [0.0] * total_frames,
            "neutral": [0.0] * total_frames,
        }
        self.batch_faces = []
        self.batch_frame_indices = []

    def process_frame(self, ctx: FrameContext):
        self.batch_faces.append(ctx.face_crop)
        self.batch_frame_indices.append(ctx.frame_idx)

        if len(self.batch_faces) >= self.batch_size:
            self._process_batch()

    def _process_batch(self):
        results = self.pipe(self.batch_faces)
        for idx, res in zip(self.batch_frame_indices, results):
            for emotion, score in res.items():
                self.values[emotion][idx] = float(score)

        self.batch_faces = []
        self.batch_frame_indices = []

    def finalize(self):
        if self.batch_faces:
            self._process_batch()
        return self.values
