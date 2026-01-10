import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from transformers import pipeline
from config import Config
from feature_extractor import VideoFeaturePass
from ..seenx_utils import resize_crop_center_np
from ..video_dataset import FaceCropVideoDataset, EmotionIterableDataset


class EmotionFeaturePass(VideoFeaturePass):
    def __init__(self, config: Config):
        self.device = torch.device(config.get("device"))
        self.batch_size = config.get("batch_size")
        self.pipe = pipeline(
            "image-classification",
            model="dima806/facial_emotions_image_detection",
            use_fast=False,
            device=self.device,
            batch_size=self.batch_size,
        )
        self.speaker_thr = config.get("speaker_probability_threshold")
        self.config = config

    def required_keys(self):
        return {"frame_face_boxes", "speaker_prob"}

    def produces_keys(self):
        return {"emotion"}

    def transform(
        self,
        frame: np.ndarray,
    ) -> Image.Image:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_crop_center_np(frame, 640)
        return frame[np.newaxis, :, :, :]

    def run(self, video_path, context):
        total_frames = len(context["data"])
        emotions = {
            k: [0.0] * total_frames
            for k in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        }

        # frame_ids where speaker_prob >= threshold
        frame_ids = (
            context["data"]
            .index[context["data"]["speaker_prob"] >= self.speaker_thr]
            .tolist()
        )

        dataset = FaceCropVideoDataset(
            frame_ids=frame_ids,
            crop_boxes=context["data"]["frame_face_boxes"].tolist(),
            video_path=video_path,
            batch_size=self.batch_size,
            transform=self.transform,
        )

        emotion_ds = EmotionIterableDataset(dataset)

        outputs = self.pipe(
            emotion_ds,
            batch_size=self.batch_size,
        )

        for out, idx in zip(outputs, emotion_ds.indices):
            for e in out:
                emotions[e["label"]][idx] = float(e["score"])

        for key in emotions:
            context["data"][key] = pd.Series(
                emotions[key], index=context["data"].index, dtype="float64"
            )
