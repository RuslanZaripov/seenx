import cv2
import torch
import easyocr
import numpy as np
import pandas as pd
from tqdm import tqdm
from .feature_extractor import VideoFeaturePass
from ..config import Config
from ..video_dataset import VideoBatchDataset
from ..seenx_utils import resize_crop_center_np


class TextProbFeaturePass(VideoFeaturePass):
    def __init__(self, config: Config):
        self.ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        self.batch_size = config.get("batch_size")
        self.config = config

    def required_keys(self):
        return set()

    def produces_keys(self):
        return {"text_prob"}

    def transform(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_crop_center_np(frame, 640)
        return frame[np.newaxis, :, :, :]

    def run(self, video_path, context):
        # initialize text_prob column
        df = context["data"]
        if "text_prob" not in df.columns:
            df["text_prob"] = pd.Series(
                np.nan,
                index=df.index,
                dtype="float64",  # or float64
            )

        dataset = VideoBatchDataset(
            video_path=video_path,
            batch_size=self.batch_size,
            transform=self.transform,
            stride=self.config.get("text_prob_stride"),
        )

        for frames, indices in tqdm(dataset, desc="Extract text probs"):
            results = self.ocr_reader.readtext_batched(
                frames, batch_size=self.batch_size
            )
            for i, res in enumerate(results):
                text_prob = float(np.mean([c for _, _, c in res])) if res else 0.0
                context["data"].at[indices[i], "text_prob"] = text_prob

        context["data"]["text_prob"] = context["data"]["text_prob"].ffill().bfill()
