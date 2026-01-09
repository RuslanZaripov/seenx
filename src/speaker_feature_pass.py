import argparse
import cv2
import easyocr
import torch
import pandas as pd
import numpy as np
from config import Config
from shot_segmentation import batch_shot_segmentation
from seenx_utils import resize_crop_center_np
from video_dataset import (
    EmotionIterableDataset,
    FaceCropVideoDataset,
    SpeakerFilteredVideoDataset,
    SpecificFramesVideoDataset,
    VideoBatchDataset,
)
from logger import Logger
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline, CLIPModel, CLIPProcessor
from arcface_client import ArcFaceClient

logger = Logger(show=True).get_logger()


class VideoFeaturePass:
    def required_keys(self) -> set[str]:
        return set()

    def produces_keys(self) -> set[str]:
        return set()

    def run(self, video_path: str, context: dict): ...


def run_feature_pipeline(
    video_path: str,
    config: Config,
    passes: list[VideoFeaturePass],
    existing_features: set,
) -> pd.DataFrame:
    context = {}

    vid_cap = cv2.VideoCapture(video_path)
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_cap.release()

    context["data"] = pd.DataFrame({"frame_idx": list(range(total_frames))})

    shift = config.get("shot_bound_shift_frames")
    shot_bounds = batch_shot_segmentation(video_path, config.get("shot_segmentor"))
    shot_bounds[:, 0] = np.clip(shot_bounds[:, 0] + shift, 0, total_frames - 1)
    shot_bounds[:, 1] = np.clip(shot_bounds[:, 1] - shift, 0, total_frames - 1)
    context["shift"] = shift
    context["shot_bounds"] = shot_bounds.flatten().tolist()

    for p in passes:
        if p.produces_keys() & existing_features:
            continue

        missing = p.required_keys() - context.keys() - set(context["data"].columns)
        if missing:
            raise RuntimeError(f"Missing deps: {missing}")

        p.run(video_path, context)

    return context["data"]


def pad_boxes_square(boxes, w, h, pad=0.25):
    """
    boxes: List[np.ndarray] of shape (N, 4)
    returns: List[np.ndarray] of shape (N, 4) with square boxes
    """
    padded_boxes = []

    for frame_boxes in boxes:
        if len(frame_boxes) == 0:
            padded_boxes.append(frame_boxes)
            continue

        frame_boxes = frame_boxes.astype(np.float32)

        x1, y1, x2, y2 = frame_boxes.T
        bw = x2 - x1
        bh = y2 - y1

        # square side = max(width, height)
        side = np.maximum(bw, bh)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        half = (1 + pad) * side / 2

        x1_p = np.clip(cx - half, 0, w)
        y1_p = np.clip(cy - half, 0, h)
        x2_p = np.clip(cx + half, 0, w)
        y2_p = np.clip(cy + half, 0, h)

        padded = np.stack([x1_p, y1_p, x2_p, y2_p], axis=1).astype(np.int32)
        padded_boxes.append(padded)

    return padded_boxes


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
                [0.0] * len(df), index=df.index, dtype=object
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
                [0.0] * len(df), index=df.index, dtype=object
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
        dataset = VideoBatchDataset(
            video_path=video_path,
            batch_size=self.batch_size,
            transform=self.transform,
        )

        for frames, indices in tqdm(dataset, desc="Extract text probs"):
            results = self.ocr_reader.readtext_batched(
                frames, batch_size=self.batch_size
            )
            for i, res in enumerate(results):
                text_prob = float(np.mean([c for _, _, c in res])) if res else 0.0
                context["data"].at[indices[i], "text_prob"] = text_prob


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
                [0.0] * len(df), index=df.index, dtype=object
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

        outputs = self.pipe(
            EmotionIterableDataset(dataset),
            batch_size=self.batch_size,
        )

        for out in outputs:
            idx = out["frame_idx"]
            for e in out:
                emotions[e["label"]][idx] = float(e["score"])

        for key in emotions:
            context["data"][key] = pd.Series(emotions[key], index=context["data"].index)


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
                    context["data"].at[frame_idx, "cinematic"] = cinematic_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file.",
    )
    args = parser.parse_args()

    config = Config(args.config)
    features_df = run_feature_pipeline(
        args.video,
        config,
        passes=[
            SpeakerProbabilityPass(config),
            FaceScreenRatioFeaturePass(config),
            # TextProbFeaturePass(config),
            # MotionSpeedFeaturePass(config),
            EmotionFeaturePass(config),
            CinematicFeaturePass(config),
        ],
        existing_features=set(),
    )

    print(features_df)
