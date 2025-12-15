import cv2
import torch
import numpy as np
import easyocr
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from transformers import pipeline
from arcface_client import ArcFaceClient
from shot_segmentation import batch_shot_segmentation
from typing import List
from PIL import Image
from logger import Logger
from config import Config
from abc import ABC, abstractmethod
from extractors import (
    FrameContext,
    FaceScreenRatioFeature,
    TextProbFeature,
    MotionSpeedFeature,
    EmotionFeature,
)

logger = Logger(show=True).get_logger()


def resize_or_crop_center_np(frame: np.ndarray, size: int = 640) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected shape (H, W, C), got {frame.shape}")

    h, w, c = frame.shape

    if h < size or w < size:
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)

    else:
        top = (h - size) // 2
        left = (w - size) // 2
        frame = frame[top : top + size, left : left + size, :]

    return frame


def get_frame_features(video_path: str, existing_features: List[str] = []) -> dict:
    """Compute frame quality features such as brightness and sharpness."""
    frame_features = {"brightness": [], "sharpness": []}
    feature_names = list(frame_features.keys())
    if all(feature in existing_features for feature in feature_names):
        logger.info(f"Frame quality features already exist, skipping extraction")
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(
        total=total_frames, desc="Extracting Frame Quality Features", unit="frame"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = float(np.mean(gray))
        frame_features["brightness"].append(brightness)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        frame_features["sharpness"].append(sharpness)

        pbar.update(1)

    pbar.close()
    cap.release()

    return frame_features


class SpeakerFeaturesExtractor:
    def __init__(
        self,
        config: Config,
        device: str | torch.device = "cpu",
        batch_size: int = 32,
    ):
        self.arcface_client = ArcFaceClient(config.get("face_embedder"))
        self.face_detector = YOLO(config.get("face_detector"))
        self.pose_model = YOLO(config.get("pose_model"))
        self.ocr_model = easyocr.Reader(["en"])
        self.speaker_threshold = config.get("speaker_probability_threshold")
        self.keypoint_conf_threshold = 0.3
        self.pipe = pipeline(
            "image-classification",
            model="dima806/facial_emotions_image_detection",
            use_fast=False,
            device=device,
            batch_size=batch_size,
        )
        self.batch_size = batch_size

    def use_face_detector(self, frame_rgb: np.ndarray) -> np.ndarray:
        results = self.face_detector(frame_rgb, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        return boxes

    def use_pose_model(self, frame_rgb: np.ndarray) -> np.ndarray:
        results = self.pose_model(frame_rgb, verbose=False)
        keypoints = results[0].keypoints.data.cpu().numpy()
        return keypoints

    def read_and_process_frame(self, cap, frame_idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        frame_rgb = resize_or_crop_center_np(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frame_rgb

    def get_embeddings(self, frame_rgb: np.ndarray):
        boxes = self.use_face_detector(frame_rgb)

        h, w, _ = frame_rgb.shape
        embeddings = []
        corrected_boxes = []
        for _, (x1, y1, x2, y2) in enumerate(boxes):
            bw, bh = x2 - x1, y2 - y1

            diff = abs(bw - bh)
            pad = diff // 2
            if bw < bh:
                x1 -= pad
                x2 += pad
            else:
                y1 -= pad
                y2 += pad

            nx1 = max(0, x1)
            ny1 = max(0, y1)
            nx2 = min(w, x2)
            ny2 = min(h, y2)

            face_crop = cv2.resize(
                frame_rgb[ny1:ny2, nx1:nx2], (112, 112), interpolation=cv2.INTER_LINEAR
            )
            # plot_numpy_image(f'Face{i+1}', face_crop)
            # print(f"{face_crop.shape=}")
            face_embedding = self.arcface_client.forward(face_crop)

            embeddings.append(face_embedding)
            corrected_boxes.append([nx1, ny1, nx2, ny2])

        return corrected_boxes, np.array(embeddings)

    def box_area(self, box: List[int]) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def vector_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1 = emb1.reshape(-1)
        emb2 = emb2.reshape(-1)
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)

    def find_speaker(self, frame_rgb, actual_speaker_embedding):
        """Extract frame, get embeddings, and compute similarity."""
        boxes, embeddings = self.get_embeddings(frame_rgb)

        if len(embeddings) == 0:
            return 0.0, None

        probs = [
            self.vector_similarity(actual_speaker_embedding, e) for e in embeddings
        ]

        # plot_frame_with_boxes(frame_rgb, boxes, probs, title=f"Frame-{frame_idx}")

        max_index = np.argmax(probs)
        max_prob = probs[max_index]

        return max_prob, boxes[max_index]

    def get_speaker_probs(self, intervals, video_path, config: Config, shift: int = 2):
        img_bgr = cv2.imread(
            config.get("speaker_image_path")
        )  # shape: (H, W, C), BGR order
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # convert to RGB if needed
        boxes, actual_speaker_embedding = self.get_embeddings(img_rgb)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            exit()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        speaker_probs = np.zeros(total_frames, dtype=np.float32)

        for start, end in tqdm(intervals, desc="Processing intervals"):
            s = min(max(start + shift, 0), end)
            e = max(min(end - shift, total_frames - 1), start)

            s_frame_rgb = self.read_and_process_frame(cap, s)
            e_frame_rgb = self.read_and_process_frame(cap, e)

            s_sim, _ = self.find_speaker(s_frame_rgb, actual_speaker_embedding)
            e_sim, _ = self.find_speaker(e_frame_rgb, actual_speaker_embedding)

            n_frames = end - start + 1
            filled = np.linspace(s_sim, e_sim, n_frames)
            filled = np.where(filled > 0.9, filled, 0.0)
            speaker_probs[start : end + 1] = filled

        cap.release()
        return speaker_probs, actual_speaker_embedding

    def build_feature_registry(self):
        return [
            FaceScreenRatioFeature(),
            TextProbFeature(self.ocr_model),
            MotionSpeedFeature(self.keypoint_conf_threshold),
            EmotionFeature(self.pipe, self.batch_size),
        ]

    def get_speaker_features(self, video_path, config: Config, existing_features=None):
        existing_features = set(existing_features or [])

        shot_bounds = batch_shot_segmentation(video_path, config.get("shot_segmentor"))
        speaker_probs, actual_speaker_embedding = self.get_speaker_probs(
            shot_bounds, video_path, config
        )

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        features = [
            f for f in self.build_feature_registry() if f.name not in existing_features
        ]

        for f in features:
            f.init_storage(total_frames)

        ctx = FrameContext()
        frame_idx = 0

        pbar = tqdm(total=total_frames, desc="Extracting features")

        while True:
            frame_rgb = self.read_and_process_frame(cap, frame_idx)
            if frame_rgb is None:
                break

            if speaker_probs[frame_idx] < self.speaker_threshold:
                frame_idx += 1
                pbar.update(1)
                continue

            # Initialize context
            ctx.frame_rgb = frame_rgb
            ctx.frame_shape = frame_rgb.shape
            ctx.frame_idx = frame_idx
            ctx.face_box = None
            ctx.face_crop = None
            ctx.keypoints = None

            # Speaker face (shared)
            _, ctx.face_box = self.find_speaker(frame_rgb, actual_speaker_embedding)

            if ctx.face_box is not None:
                x1, y1, x2, y2 = ctx.face_box
                ctx.face_crop = frame_rgb[y1:y2, x1:x2]

            # Lazy dependencies
            if any(f.requires_keypoints for f in features):
                ctx.keypoints = self.use_pose_model(frame_rgb)

            # Run enabled features
            for f in features:
                if f.requires_face and ctx.face_box is None:
                    continue
                f.process_frame(ctx)

            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()

        # Collect results
        data = {f.name: f.finalize() for f in features}
        data["speaker_prob"] = speaker_probs

        return data


def speaker_features_pipeline(
    video_path: str, config: Config, existing_features: list = []
) -> pd.DataFrame:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emotion_detector = SpeakerFeaturesExtractor(config, device=device)

    data = emotion_detector.get_speaker_features(video_path, config, existing_features)

    frame_features = get_frame_features(video_path, existing_features)

    data.update(frame_features)

    df = pd.DataFrame(data)
    return df
