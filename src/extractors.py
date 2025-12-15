import cv2
import numpy as np
import easyocr
import torch
from transformers import CLIPImageProcessor, CLIPTokenizer, pipeline
from tqdm import tqdm
from config import Config
from shot_segmentation import batch_shot_segmentation
from abc import ABC, abstractmethod
from transformers import CLIPModel, CLIPProcessor


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

    def __init__(self):
        self.ocr = easyocr.Reader(["en"])

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

    def __init__(self, batch_size: int, device: str | torch.device = "cpu"):
        self.pipe = pipeline(
            "image-classification",
            model="dima806/facial_emotions_image_detection",
            use_fast=False,
            device=device,
            batch_size=batch_size,
        )
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
        for idx, emotions_report in zip(self.batch_frame_indices, results):
            for e in emotions_report:
                self.values[e["label"]][idx] = float(e["score"])

        self.batch_faces = []
        self.batch_frame_indices = []

    def finalize(self):
        if self.batch_faces:
            self._process_batch()
        return self.values


class CinematicFeature(FeatureExtractor):
    name = "cinematic"

    def __init__(
        self,
        config: Config,
        device: torch.device,
        batch_size: int = 32,
        use_fp16: bool = True,
    ):
        self.device = device
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and device.type == "cuda"

        self.processor = CLIPProcessor.from_pretrained(
            config.get("clip_model"),
            use_fast=False,
        )

        self.model = (
            CLIPModel.from_pretrained(config.get("clip_model")).to(device).eval()
        )

        # Fixed prompts
        self.texts = ["a cinematic frame", "not a cinematic frame"]

    # -------------------- lifecycle --------------------

    def init_storage(self, total_frames: int):
        self.values = [0.0] * total_frames

        # buffers for batching
        self._frames: list[np.ndarray] = []
        self._indices: list[int] = []

        # cache text embeddings once
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.texts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            self.text_features = self.model.get_text_features(**text_inputs)
            self.text_features = self.text_features / self.text_features.norm(
                dim=-1, keepdim=True
            )

    def process_frame(self, ctx: FrameContext):
        # just cache
        self._frames.append(ctx.frame_rgb)
        self._indices.append(ctx.frame_idx)

    def finalize(self):
        if not self._frames:
            return self.values

        for start in range(0, len(self._frames), self.batch_size):
            end = start + self.batch_size
            batch_frames = self._frames[start:end]
            batch_indices = self._indices[start:end]

            inputs = self.processor(
                images=batch_frames,
                return_tensors="pt",
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_fp16):
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                logits = image_features @ self.text_features.T
                probs = logits.softmax(dim=-1)

            # store results
            for i, frame_idx in enumerate(batch_indices):
                self.values[frame_idx] = float(probs[i, 0].item())

        # free memory
        self._frames.clear()
        self._indices.clear()

        return self.values
