import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from arcface_client import ArcFaceClient
from shot_segmentation import batch_shot_segmentation
from typing import List
from PIL import Image
from logger import Logger
from config import Config
from extractors import (
    BatchContext,
    FaceScreenRatioFeature,
    TextProbFeature,
    MotionSpeedFeature,
    EmotionFeature,
    CinematicFeature,
    FeatureExtractor,
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
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.get("device"))
        self.batch_size = config.get("batch_size")
        self.arcface_client = ArcFaceClient(config.get("face_embedder"))
        self.face_detector = YOLO(config.get("face_detector")).to(self.device)
        self.pose_model = YOLO(config.get("pose_model")).to(self.device)
        self.speaker_threshold = config.get("speaker_probability_threshold")
        self.keypoint_conf_threshold = 0.3

    def use_face_detector(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        results = self.face_detector(frames, verbose=False)
        boxes = [res.boxes.xyxy.cpu().numpy().astype(float) for res in results]
        return boxes

    def use_pose_model(self, frame_rgb: list[np.ndarray]) -> list[np.ndarray]:
        results = self.pose_model(frame_rgb, verbose=False)
        keypoints = [res.keypoints.data.cpu().numpy().astype(float) for res in results]
        return keypoints

    def read_and_process_frame(self, cap, frame_idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Cannot read frame at index {frame_idx}")

        frame_rgb = resize_or_crop_center_np(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frame_rgb

    def collect_frames(
        self, cap, frame_idx: int, speaker_probs: list[float]
    ) -> list[np.ndarray]:
        frames = []
        frame_indices = []

        for i in range(self.batch_size):
            current_frame_idx = frame_idx + i

            if speaker_probs[current_frame_idx] < self.speaker_threshold:
                frame_idx += 1
                continue

            frame_rgb = self.read_and_process_frame(cap, current_frame_idx)
            frames.append(frame_rgb)

            frame_indices.append(current_frame_idx)

        return frames, frame_indices, frame_idx + len(frames)

    def get_embeddings(
        self, frames: list[np.ndarray]
    ) -> tuple[list[List[int]], np.ndarray]:
        batch_boxes = self.use_face_detector(frames)

        h, w, _ = frames[0].shape
        embeddings = []
        corrected_boxes = []

        face_crops = []

        for i, boxes in enumerate(batch_boxes):
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
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
                    frames[i][ny1:ny2, nx1:nx2],
                    (112, 112),
                    interpolation=cv2.INTER_LINEAR,
                )

                # print(f"{face_crop.shape=}")
                # plot_numpy_image(f'Face{i+1}', face_crop)

                face_crops.append(face_crop)
                corrected_boxes.append([i, nx1, ny1, nx2, ny2])

        embeddings = []
        if len(face_crops) > 0:
            embeddings = self.arcface_client.forward(face_crops)

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

    def find_speaker(
        self, frames: list[np.ndarray], actual_speaker_embedding: np.ndarray
    ) -> tuple[list[List[int]], list[Image.Image]]:
        """Extract frame, get embeddings, and compute similarity."""
        boxes, embeddings = self.get_embeddings(frames)

        frame_boxes = [None] * len(frames)
        frame_probs = [0.0] * len(frames)
        face_crops = [None] * len(frames)

        for i in range(len(boxes)):
            frame_idx = boxes[i][0]
            vec_sim = self.vector_similarity(actual_speaker_embedding, embeddings[i])

            if frame_probs[frame_idx] < vec_sim:
                frame_probs[frame_idx] = vec_sim
                frame_boxes[frame_idx] = boxes[i][1:]
                x1, y1, x2, y2 = frame_boxes[frame_idx]
                face_crops[frame_idx] = Image.fromarray(frames[frame_idx][y1:y2, x1:x2])

        return frame_probs, frame_boxes, face_crops

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

        frames = []
        for start, end in tqdm(
            intervals, desc="Determining speaker probabilities for shots"
        ):
            s = min(start + shift, end)
            e = max(min(end - shift, total_frames - 1), start)

            s_frame_rgb = self.read_and_process_frame(cap, s)
            e_frame_rgb = self.read_and_process_frame(cap, e)

            frames.extend([s_frame_rgb, e_frame_rgb])

        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i : i + self.batch_size]
            frame_probs, batch_boxes, batch_face_crops = self.find_speaker(
                batch_frames, actual_speaker_embedding
            )

            for j in range(0, len(batch_frames), 2):
                start, end = intervals[i // 2 + j // 2]
                n_frames = end - start + 1
                s_sim = frame_probs[i + j]
                e_sim = frame_probs[i + j + 1]
                filled = np.linspace(s_sim, e_sim, n_frames)
                filled = np.where(filled > 0.9, filled, 0.0)
                speaker_probs[start : end + 1] = filled

        cap.release()
        return speaker_probs, actual_speaker_embedding

    def build_feature_registry(self) -> list[FeatureExtractor]:
        return [
            FaceScreenRatioFeature(),
            TextProbFeature(),
            MotionSpeedFeature(self.keypoint_conf_threshold),
            EmotionFeature(self.config, device=self.device),
            CinematicFeature(self.config, device=self.device),
        ]

    def get_speaker_features(self, video_path, config: Config, existing_features=None):
        existing_features = set(existing_features or [])

        shot_bounds = batch_shot_segmentation(video_path, config.get("shot_segmentor"))
        speaker_probs, actual_speaker_embedding = self.get_speaker_probs(
            shot_bounds, video_path, config
        )

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        features: list[FeatureExtractor] = [
            f for f in self.build_feature_registry() if f.name not in existing_features
        ]

        for f in features:
            f.init_storage(total_frames)

        ctx = BatchContext()
        frame_idx = 0

        pbar = tqdm(total=total_frames, desc="Extracting features")

        while frame_idx < total_frames:
            frames, frame_indices = self.collect_frames(cap, frame_idx, speaker_probs)

            # Initialize context
            ctx.frames = frames
            ctx.frame_shape = frames[0].shape
            ctx.frame_indices = frame_indices

            ctx.face_boxes = None
            ctx.face_crops = None
            ctx.keypoints = self.use_pose_model(frames)

            # Speaker face (shared)
            _, ctx.face_boxes, ctx.face_crops = self.find_speaker(
                frames, actual_speaker_embedding
            )

            # Run enabled features
            for f in features:
                if f.requires_face and ctx.face_boxes is None:
                    continue

                if f.requires_keypoints and ctx.keypoints is None:
                    continue

                f.process_frames(ctx)

            frame_idx += self.batch_size

            pbar.update(self.batch_size)

        cap.release()
        pbar.close()

        data = {}
        for f in features:
            result = f.finalize()
            if isinstance(result, dict):
                data.update(result)
            else:
                data[f.name] = result

        data["speaker_prob"] = speaker_probs.tolist()

        return data


def speaker_features_pipeline(
    video_path: str, config: Config, existing_features: list = []
) -> pd.DataFrame:
    emotion_detector = SpeakerFeaturesExtractor(config)

    data = emotion_detector.get_speaker_features(video_path, config, existing_features)

    frame_features = get_frame_features(video_path, existing_features)

    data.update(frame_features)

    df = pd.DataFrame(data)
    return df
