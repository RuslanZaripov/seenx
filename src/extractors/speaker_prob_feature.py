import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from .feature_extractor import VideoFeaturePass
from ..config import Config
from ..arcface_client import ArcFaceClient
from ..video_dataset import SpecificFramesVideoDataset, FaceCropVideoDataset
from ..seenx_utils import resize_crop_center_np, pad_boxes_square


class SpeakerProbabilityPass(VideoFeaturePass):
    def __init__(self, config: Config):
        self.device = torch.device(config.get("device"))
        self.face_detector = YOLO(config.get("face_detector")).to(self.device)
        self.arcface_client = ArcFaceClient(config.get("face_embedder"))
        self.batch_size = config.get("batch_size")
        self.speaker_thr = config.get("speaker_probability_threshold")
        self.config = config

    def required_keys(self):
        return {"shot_bounds"}

    def produces_keys(self):
        return {"speaker_prob"}

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

    def speaker_face_embedding(self) -> np.ndarray:
        img_bgr = cv2.imread(self.config.get("speaker_image_path"))
        img_rgb = resize_crop_center_np(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        img_rgb = img_rgb[np.newaxis, :, :, :]
        _, h, w, _ = img_rgb.shape
        batch_boxes = self.use_face_detector(img_rgb)
        padded_boxes = pad_boxes_square(batch_boxes, w, h)
        x1, y1, x2, y2 = padded_boxes[0].tolist()[0]
        face_crop = cv2.resize(
            img_rgb[0][y1:y2, x1:x2], (112, 112), interpolation=cv2.INTER_LINEAR
        )
        actual_speaker_embedding = self.arcface_client.forward(face_crop)
        return np.array(actual_speaker_embedding)

    def transform(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_crop_center_np(frame, 640)
        return frame[np.newaxis, :, :, :]

    def vector_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1 = emb1.reshape(-1)
        emb2 = emb2.reshape(-1)
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)

    def run(self, video_path, context):
        dataset = SpecificFramesVideoDataset(
            frame_ids=context["shot_bounds"],
            video_path=video_path,
            batch_size=self.batch_size,
            transform=self.transform,
        )

        df = context["data"]
        if "frame_face_boxes" not in df.columns:
            df["frame_face_boxes"] = pd.Series(
                [None] * len(df), index=df.index, dtype=object
            )

        if "speaker_prob" not in df.columns:
            df["speaker_prob"] = pd.Series(
                [0.0] * len(df), index=df.index, dtype="float64"
            )

        for frames, indices in tqdm(dataset, desc="Extract speaker probs"):
            h, w, _ = frames[0].shape
            batch_boxes = self.use_face_detector(frames)
            padded_boxes = pad_boxes_square(batch_boxes, w, h)
            for idx, boxes in zip(indices, padded_boxes):
                df.at[idx, "frame_face_boxes"] = boxes

        dataset = FaceCropVideoDataset(
            frame_ids=context["shot_bounds"],
            crop_boxes=context["data"]["frame_face_boxes"].tolist(),
            video_path=video_path,
            batch_size=self.batch_size,
            transform=self.transform,
        )

        actual_speaker_embedding = self.speaker_face_embedding()
        for frames, indices in tqdm(dataset, desc="Extract speaker embeddings"):
            embeddings = self.arcface_client.forward(frames)
            embeddings = np.array(embeddings)
            for i, frame_idx in enumerate(indices):
                vec_sim = self.vector_similarity(
                    actual_speaker_embedding, embeddings[i]
                )
                current_sim = context["data"].at[frame_idx, "speaker_prob"]
                if current_sim < vec_sim:
                    context["data"].at[frame_idx, "speaker_prob"] = vec_sim

        for i in range(0, len(context["shot_bounds"]), 2):
            start = context["shot_bounds"][i]
            end = context["shot_bounds"][i + 1]
            s_sim = context["data"].at[start, "speaker_prob"]
            e_sim = context["data"].at[end, "speaker_prob"]
            start -= context["shift"]
            end += context["shift"]
            n_frames = end - start + 1
            filled = np.linspace(s_sim, e_sim, n_frames)
            filled = np.where(filled > self.speaker_thr, filled, 0.0)
            context["data"].loc[start:end, "speaker_prob"] = filled
