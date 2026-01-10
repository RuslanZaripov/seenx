import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from .feature_extractor import VideoFeaturePass
from ..config import Config
from ..video_dataset import VideoBatchDataset
from ..seenx_utils import resize_crop_center_np


class CinematicFeaturePass(VideoFeaturePass):
    def __init__(self, config: Config):
        self.device = torch.device(config.get("device"))
        self.batch_size = config.get("batch_size")
        self.config = config
        self.use_fp16 = True and self.device.type == "cuda"
        self.processor = CLIPProcessor.from_pretrained(
            config.get("clip_model"),
            use_fast=False,
        )
        self.model = (
            CLIPModel.from_pretrained(config.get("clip_model")).to(self.device).eval()
        )
        self.texts = ["a cinematic frame", "not a cinematic frame"]

    def required_keys(self):
        return set()

    def produces_keys(self):
        return {"cinematic"}

    def transform(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_crop_center_np(frame, 640)
        return frame[np.newaxis, :, :, :]

    def run(self, video_path, context):
        df = context["data"]
        if "cinematic" not in df.columns:
            df["cinematic"] = pd.Series(
                [0.0] * len(df),
                index=df.index,
                dtype="float64",
            )

        dataset = VideoBatchDataset(
            video_path=video_path,
            batch_size=self.batch_size,
            transform=self.transform,
        )

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, enabled=self.use_fp16
        ):
            text_inputs = self.processor(
                text=self.texts,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            txt_feat = self.model.get_text_features(**text_inputs)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            for frames, indices in tqdm(dataset, desc="Extract cinematic probs"):
                inputs = self.processor(
                    images=frames,
                    return_tensors="pt",
                ).to(self.device)

                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                logits = image_features @ txt_feat.T
                probs = logits.softmax(dim=-1)

                for i, frame_idx in enumerate(indices):
                    cinematic_prob = float(probs[i, 0])
                    df.at[frame_idx, "cinematic"] = cinematic_prob
