import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from .feature_extractor import VideoFeaturePass
from ..config import Config
from ..seenx_utils import resize_crop_center_np
from ..video_dataset import SpeakerFilteredVideoDataset


class MotionSpeedFeaturePass(VideoFeaturePass):
    def __init__(self, config: Config):
        self.device = torch.device(config.get("device"))
        self.pose_model = YOLO(config.get("pose_model")).to(self.device)
        self.batch_size = config.get("batch_size")
        self.kps_thr = config.get("keypoint_confidence_threshold")
        self.speak_thr = config.get("speaker_probability_threshold")
        self.config = config

    def required_keys(self):
        return {"speaker_prob"}

    def produces_keys(self):
        return {"motion_speed"}

    def transform(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_crop_center_np(frame, 640)
        return frame[np.newaxis, :, :, :]

    def use_pose_model(self, frames: np.ndarray) -> list[np.ndarray]:
        # BCHW format with RGB channels float32 (0.0-1.0).
        input_tensor = (
            torch.from_numpy(frames)  # shape: (B, H, W, C)
            .permute(0, 3, 1, 2)  # shape: (B, C, H, W)
            .to(self.device)
        )
        input_tensor = input_tensor.float() / 255.0
        with torch.no_grad():
            results = self.pose_model(input_tensor, verbose=False)
        keypoints = [res.keypoints.data.cpu().numpy().astype(float) for res in results]
        del input_tensor
        del results
        return keypoints

    def distance(self, p: np.ndarray, c: np.ndarray) -> float:
        # p, c: (num_people, 17, 3)
        if p is None or c is None or len(p) == 0 or len(c) == 0:
            return 0.0

        p = p[0]  # (17, 3)
        c = c[0]  # (17, 3)

        vp = p[p[:, 2] > self.kps_thr][:, :2]
        vc = c[c[:, 2] > self.kps_thr][:, :2]

        if len(vp) == 0 or len(vc) == 0:
            return 0.0

        n = min(len(vp), len(vc))
        return float(np.linalg.norm(vc[:n] - vp[:n], axis=1).mean())

    def run(self, video_path, context):
        df = context["data"]
        if "frame_keypoints" not in df.columns:
            df["frame_keypoints"] = pd.Series(
                [None] * len(df), index=df.index, dtype=object
            )

        if "motion_speed" not in df.columns:
            df["motion_speed"] = pd.Series(
                [0.0] * len(df), index=df.index, dtype="float64"
            )

        dataset = SpeakerFilteredVideoDataset(
            speaker_probs=context["data"]["speaker_prob"].tolist(),
            threshold=self.speak_thr,
            video_path=video_path,
            batch_size=self.batch_size,
            transform=self.transform,
        )

        for frames, indices in tqdm(dataset, desc="Extract motion speeds"):
            batch_kps = self.use_pose_model(frames)
            for idx, kps in zip(indices, batch_kps):
                context["data"].at[idx, "frame_keypoints"] = kps

            for i, idx in enumerate(indices):
                prev_index = idx - 1
                if prev_index < 0:
                    continue
                prev_kps = context["data"].at[prev_index, "frame_keypoints"]
                if prev_kps is None:
                    context["data"].at[idx, "motion_speed"] = 0.0
                    continue
                kps = context["data"].at[idx, "frame_keypoints"]
                dist = self.distance(prev_kps, kps)
                context["data"].at[idx, "motion_speed"] = dist
