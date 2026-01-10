from ultralytics import YOLO
import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from config import Config
from feature_extractor import VideoFeaturePass
from ..seenx_utils import resize_crop_center_np, pad_boxes_square
from ..video_dataset import SpeakerFilteredVideoDataset


class FaceScreenRatioFeaturePass(VideoFeaturePass):
    def __init__(self, config: Config):
        self.device = torch.device(config.get("device"))
        self.face_detector = YOLO(config.get("face_detector")).to(self.device)
        self.speaker_thr = config.get("speaker_probability_threshold")
        self.batch_size = config.get("batch_size")
        self.config = config

    def required_keys(self):
        return {"speaker_prob"}

    def produces_keys(self):
        return {"face_screen_ratio"}

    def use_face_detector(self, frames: np.ndarray) -> list[np.ndarray]:
        # BCHW format with RGB channels float32 (0.0-1.0).
        input_tensor = (
            torch.from_numpy(frames)  # shape: (B, H, W, C)
            .permute(0, 3, 1, 2)  # shape: (B, C, H, W)
            .to(self.device)
        )
        input_tensor = input_tensor.float() / 255.0
        with torch.no_grad():
            results = self.face_detector(input_tensor, verbose=False)
        boxes = [res.boxes.xyxy.cpu().numpy().astype(float) for res in results]
        del input_tensor
        del results
        return boxes

    def transform(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_crop_center_np(frame, 640)
        return frame[np.newaxis, :, :, :]

    def run(self, video_path, context):
        dataset = SpeakerFilteredVideoDataset(
            speaker_probs=context["data"]["speaker_prob"].tolist(),
            threshold=self.speaker_thr,
            video_path=video_path,
            batch_size=self.batch_size,
            transform=self.transform,
        )

        df = context["data"]
        if "face_screen_ratio" not in df.columns:
            df["face_screen_ratio"] = pd.Series(
                [0.0] * len(df), index=df.index, dtype="float64"
            )

        if "frame_face_boxes" not in df.columns:
            df["frame_face_boxes"] = pd.Series(
                [None] * len(df), index=df.index, dtype=object
            )

        for frames, indices in tqdm(dataset, desc="Extract face screen ratio"):
            h, w, _ = frames[0].shape
            batch_boxes = self.use_face_detector(frames)
            padded_boxes = pad_boxes_square(batch_boxes, w, h)
            for idx, boxes in zip(indices, padded_boxes):
                context["data"].at[idx, "frame_face_boxes"] = boxes
                if len(boxes) == 0:
                    ratio = 0.0
                else:
                    x1, y1, x2, y2 = boxes[0]
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    ratio = area / (h * w)
                context["data"].at[idx, "face_screen_ratio"] = ratio
