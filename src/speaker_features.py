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


def resize_crop_center_np(frame: np.ndarray, size: int = 640) -> np.ndarray:
    """
    Resize preserving aspect ratio and center-crop to (size, size).

    Args:
        frame: np.ndarray of shape (H, W, C)
        size: target spatial size

    Returns:
        np.ndarray of shape (size, size, C)
    """
    if frame.ndim != 3:
        raise ValueError(f"Expected shape (H, W, C), got {frame.shape}")

    h, w, c = frame.shape

    scale = size / min(h, w)
    if scale != 1.0:
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    h, w, _ = frame.shape
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)

    frame = frame[top : top + size, left : left + size]

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
        # BCHW format with RGB channels float32 (0.0-1.0).
        input = (
            torch.from_numpy(np.stack(frames, axis=0))  # shape: (B, H, W, C)
            .permute(0, 3, 1, 2)  # shape: (B, C, H, W)
            .to(self.device)
        )
        input = input.float() / 255.0
        # logger.debug(f"Face detector {input.shape=} {input.dtype=} {input.device=}")
        results = self.face_detector(input, verbose=False)
        # logger.debug(f"Face detector {len(results)=}")
        # logger.debug(f"Face detector {results[0].boxes.xyxy=}")
        boxes = [res.boxes.xyxy.cpu().numpy().astype(float) for res in results]
        return boxes

    def use_pose_model(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        # BCHW format with RGB channels float32 (0.0-1.0).
        input = (
            torch.from_numpy(np.stack(frames, axis=0))  # shape: (B, H, W, C)
            .permute(0, 3, 1, 2)  # shape: (B, C, H, W)
            .to(self.device)
        )
        input = input.float() / 255.0
        # logger.debug(f"Pose model {input.shape=} {input.dtype=} {input.device=}")
        results = self.pose_model(input, verbose=False)
        # logger.debug(f"Pose model {len(results)=}")
        # logger.debug(f"Pose model {results[0].boxes.xyxy=}")
        keypoints = [res.keypoints.data.cpu().numpy().astype(float) for res in results]
        return keypoints

    def read_and_process_frame(self, cap, frame_idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Cannot read frame at index {frame_idx}")
        frame_rgb = resize_crop_center_np(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frame_rgb

    def collect_frames(
        self, cap, frame_idx: int, speaker_probs: list[float]
    ) -> tuple[list[np.ndarray], list[int], int]:
        frames = []
        frame_indices = []

        count = 0
        curr_frame_idx = frame_idx

        while count < self.batch_size and curr_frame_idx < len(speaker_probs):
            if speaker_probs[curr_frame_idx] >= self.speaker_threshold:
                continue
            curr_frame_idx += 1
            count += 1
            frames.append(self.read_and_process_frame(cap, curr_frame_idx))
            frame_indices.append(curr_frame_idx)

        return frames, frame_indices, curr_frame_idx

    def pad_box(self, box: List[int], w: int, h: int) -> List[int]:
        x1, y1, x2, y2 = box
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
        return [nx1, ny1, nx2, ny2]

    def get_embeddings(
        self, frames: list[np.ndarray]
    ) -> tuple[list[List[int]], np.ndarray]:
        # frames: list of (H, W, C) np.ndarray in RGB format
        batch_boxes = self.use_face_detector(frames)
        h, w, _ = frames[0].shape

        embeddings = []
        corrected_boxes = []

        face_crops = []

        for i, boxes in enumerate(batch_boxes):
            for box in boxes:
                nx1, ny1, nx2, ny2 = self.pad_box(box.astype(int).tolist(), w, h)
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
        face_crops = np.array(face_crops)
        # logger.info(f"Embeddings model {face_crops.shape=} {face_crops.dtype=}")
        if len(face_crops) > 0:
            embeddings = self.arcface_client.forward(face_crops)
        # logger.info(f"Extracted {type(embeddings)=} {len(embeddings)=}")

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
        # shape: (H, W, C), BGR order
        img_bgr = cv2.imread(config.get("speaker_image_path"))
        img_rgb = resize_crop_center_np(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        logger.info(f"Speaker image shape: {img_rgb.shape}")

        boxes, actual_speaker_embedding = self.get_embeddings([img_rgb])

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
                start, end = intervals[j // 2 + i // 2]
                n_frames = end - start + 1
                s_sim = frame_probs[j]
                e_sim = frame_probs[j + 1]
                filled = np.linspace(s_sim, e_sim, n_frames)
                filled = np.where(filled > 0.9, filled, 0.0)
                speaker_probs[start : end + 1] = filled

        cap.release()
        return speaker_probs, actual_speaker_embedding

    def build_feature_registry(self) -> list[FeatureExtractor]:
        return [
            FaceScreenRatioFeature(),
            TextProbFeature(self.config),
            MotionSpeedFeature(self.keypoint_conf_threshold),
            EmotionFeature(self.config),
            CinematicFeature(self.config),
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
            frames, frame_indices, next_frame_idx = self.collect_frames(
                cap, frame_idx, speaker_probs
            )

            logger.debug(f"Processing {frame_idx} {next_frame_idx}")

            # Initialize context
            ctx.frames = frames
            ctx.frame_shape = frames[0].shape
            ctx.frame_indices = frame_indices

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

            pbar.update(next_frame_idx - frame_idx)
            frame_idx = next_frame_idx

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
    extractor = SpeakerFeaturesExtractor(config)

    data = extractor.get_speaker_features(video_path, config, existing_features)
    frame_features = get_frame_features(video_path, existing_features)

    data.update(frame_features)
    df = pd.DataFrame(data)
    return df
