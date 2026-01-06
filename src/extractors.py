import torch
import easyocr
import numpy as np
from tqdm import tqdm
from config import Config
from abc import ABC, abstractmethod
from transformers import pipeline, CLIPModel, CLIPProcessor
from logger import Logger

logger = Logger(show=True).get_logger()


class BatchContext:
    """
    Shared mutable context for a single frame.
    Populated once per frame and reused by all features.
    """

    def __init__(self):
        self.frames = None
        self.frame_shape = None
        self.frame_indices = None

        self.face_boxes = None
        self.face_crops = None
        self.keypoints = None


class FeatureExtractor(ABC):
    name: str
    requires_face: bool = False
    requires_keypoints: bool = False

    @abstractmethod
    def init_storage(self, total_frames: int):
        pass

    @abstractmethod
    def process_frames(self, ctx: BatchContext):
        pass

    @abstractmethod
    def finalize(self) -> list | dict:
        pass


class FaceScreenRatioFeature(FeatureExtractor):
    name = "face_screen_ratio"
    requires_face = True

    def init_storage(self, total_frames: int):
        self.values = [0.0] * total_frames

    def process_frames(self, ctx: BatchContext):
        for i in range(len(ctx.frame_indices)):
            x1, y1, x2, y2 = ctx.face_boxes[i]
            h, w = ctx.frame_shape[:2]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            self.values[ctx.frame_indices[i]] = area / (h * w)

    def finalize(self):
        return self.values


class TextProbFeature(FeatureExtractor):
    name = "text_prob"

    def __init__(self, config: Config):
        self.ocr = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        self.batch_size = config.get("batch_size")

    def init_storage(self, total_frames: int):
        self.values = [0.0] * total_frames

    def process_frames(self, ctx: BatchContext):
        results = self.ocr.readtext_batched(ctx.frames, batch_size=self.batch_size)
        for i, res in enumerate(results):
            self.values[ctx.frame_indices[i]] = (
                float(np.mean([c for _, _, c in res])) if res else 0.0
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
        self.prev_keypoint = None
        self.prev_frame_idx = None

    def distance(self, p: np.ndarray, c: np.ndarray) -> float:
        vp = p[p[:, 2] > self.conf_thr][:, :2]
        vc = c[c[:, 2] > self.conf_thr][:, :2]

        if len(vp) == 0 or len(vc) == 0:
            return 0.0

        n = min(len(vp), len(vc))
        return float(np.linalg.norm(vc[:n] - vp[:n], axis=1).mean())

    def process_frames(self, ctx: BatchContext):
        for i, frame_idx in enumerate(ctx.frame_indices):
            prev = ctx.keypoints[i - 1] if i > 0 else self.prev_keypoint
            curr = ctx.keypoints[i]

            if prev is None:
                self.prev_keypoint = curr
                self.prev_frame_idx = frame_idx
                continue

            if prev.size == 0 or curr.size == 0:
                return

            dist = self.distance(prev[0], curr[0])

            self.values[frame_idx] = float(dist)

            self.prev_keypoint = curr
            self.prev_frame_idx = frame_idx

    def finalize(self):
        return self.values


class EmotionFeature(FeatureExtractor):
    name = "emotion"
    requires_face = True

    def __init__(self, config: Config):
        self.pipe = pipeline(
            "image-classification",
            model="dima806/facial_emotions_image_detection",
            use_fast=False,
            device=config.get("device"),
            batch_size=config.get("batch_size"),
        )

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

    def process_frames(self, ctx: BatchContext):
        results = self.pipe(ctx.face_crops)
        for idx, emotions_report in zip(ctx.frame_indices, results):
            for e in emotions_report:
                self.values[e["label"]][idx] = float(e["score"])

    def finalize(self):
        return self.values


class CinematicFeature(FeatureExtractor):
    name = "cinematic"

    def __init__(
        self,
        config: Config,
        use_fp16: bool = True,
    ):
        self.device = torch.device(config.get("device"))
        self.batch_size = config.get("batch_size")
        self.use_fp16 = use_fp16 and self.device.type == "cuda"

        self.processor = CLIPProcessor.from_pretrained(
            config.get("clip_model"),
            use_fast=False,
        )

        self.model = (
            CLIPModel.from_pretrained(config.get("clip_model")).to(self.device).eval()
        )

        self.texts = ["a cinematic frame", "not a cinematic frame"]

    def init_storage(self, total_frames: int):
        self.values = [0.0] * total_frames

        self._frames: list[np.ndarray] = []
        self._indices: list[int] = []

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, enabled=self.use_fp16
        ):
            text_inputs = self.processor(
                text=self.texts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            self.txt_feat = self.model.get_text_features(**text_inputs)
            self.txt_feat = self.txt_feat / self.txt_feat.norm(dim=-1, keepdim=True)

    def process_frames(self, ctx: BatchContext):
        inputs = self.processor(
            images=ctx.frames,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, enabled=self.use_fp16
        ):
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ self.txt_feat.T
            probs = logits.softmax(dim=-1)

        for i, frame_idx in enumerate(ctx.frame_indices):
            self.values[frame_idx] = float(probs[i, 0].item())

    def finalize(self):
        return self.values
