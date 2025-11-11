import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from transformers import pipeline
from arcface_client import ArcFaceClient
from shot_segmentation import batch_shot_segmentation
from PIL import Image
from logger import Logger

logger = Logger(show=True).get_logger('seenx')


def resize_or_crop_center_np(frame: np.ndarray, size: int = 640) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected shape (H, W, C), got {frame.shape}")

    h, w, c = frame.shape

    if h < size or w < size:
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)

    else:
        top = (h - size) // 2
        left = (w - size) // 2
        frame = frame[top:top + size, left:left + size, :]

    return frame


class EmotionsDetection:
    def __init__(
            self, 
            yolo_model_path: str, 
            device: str | torch.device = 'cpu', 
            speaker_threshold: float = 0.9,
            batch_size: int = 32,
        ):
        self.face_detector = YOLO(yolo_model_path)
        self.pipe = pipeline(
            "image-classification", 
            model="dima806/facial_emotions_image_detection", 
            device=device,
            batch_size=batch_size)
        self.batch_size = batch_size
        self.speaker_threshold = speaker_threshold

    def get_emotions(self, video_path, speaker_probs):
        labels = list(self.pipe.model.config.label2id.keys())
        emotions = {label: [] for label in labels}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            exit()

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        face_crops_batch = []
        frame_indices_batch = []

        with tqdm(total=total_frames, desc="Extracting faces", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count < len(speaker_probs) and speaker_probs[frame_count] < self.speaker_threshold:
                    frame_count += 1
                    pbar.update(1)
                    for _emotion in emotions.keys():
                        emotions[_emotion].append(0.0)
                    continue

                frame_rgb = resize_or_crop_center_np(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = self.face_detector(frame_rgb, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                h, w, _ = frame_rgb.shape

                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    bw, bh = x2 - x1, y2 - y1

                    diff = abs(bw - bh)
                    if bw < bh:
                        pad = diff // 2
                        x1 -= pad
                        x2 += pad
                    else:
                        pad = diff // 2
                        y1 -= pad
                        y2 += pad

                    nx1 = max(0, x1)
                    ny1 = max(0, y1)
                    nx2 = min(w, x2)
                    ny2 = min(h, y2)

                    face_crop = Image.fromarray(frame_rgb[ny1:ny2, nx1:nx2])
                    face_crops_batch.append(face_crop)
                    frame_indices_batch.append(frame_count)

                frame_count += 1
                pbar.update(1)

        cap.release()

        for label in labels:
            emotions[label] = [0.0] * total_frames

        if len(face_crops_batch) > 0:
            logger.info(f"Processing {len(face_crops_batch)} faces in batches...")
            
            emotions_reports = self.pipe(face_crops_batch)
            
            for idx, emotions_report in enumerate(emotions_reports):
                frame_idx = frame_indices_batch[idx]
                
                for e in emotions_report:
                    emotions[e['label']][frame_idx] = e['score']

        return emotions


class SpeakerFeatures:
    def __init__(self, yolo_model_path: str, arcface_weight_file: str):
        self.face_detector = YOLO(yolo_model_path)
        self.arcface_client = ArcFaceClient(arcface_weight_file)

    def get_embeddings(self, frame_rgb: np.ndarray):
        results = self.face_detector(frame_rgb, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) # shape: (N, 4)
        h, w, _ = frame_rgb.shape

        embeddings = []
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
                frame_rgb[ny1:ny2, nx1:nx2], (112, 112), interpolation=cv2.INTER_LINEAR)

            # plot_numpy_image(f'Face{i+1}', face_crop)
            
            # print(f"{face_crop.shape=}")
            face_embedding = self.arcface_client.forward(face_crop)
            
            embeddings.append(face_embedding)

        return boxes, np.array(embeddings)
    
    def vector_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1 = emb1.reshape(-1)
        emb2 = emb2.reshape(-1)
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)
    
    def evaluate_frame(self, cap, frame_idx, actual_speaker_embedding):
        """Extract frame, get embeddings, and compute similarity."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        
        frame_rgb = resize_or_crop_center_np(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, embeddings = self.get_embeddings(frame_rgb)
        
        if len(embeddings) == 0:
            return 0.0
        
        probs = [self.vector_similarity(actual_speaker_embedding, e) for e in embeddings]

        # plot_frame_with_boxes(frame_rgb, boxes, probs, title=f"Frame-{frame_idx}")

        max_probs = max(probs)                
        return max_probs
    
    def get_speaker_probs(self, intervals, speaker_image_path, video_path, shift: int = 2):
        img_bgr = cv2.imread(speaker_image_path)                # shape: (H, W, C), BGR order
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)      # convert to RGB if needed
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

            sim1 = self.evaluate_frame(cap, s, actual_speaker_embedding)
            sim2 = self.evaluate_frame(cap, e, actual_speaker_embedding)

            n_frames = end - start + 1
            filled = np.linspace(sim1, sim2, n_frames)
            filled = np.where(filled > 0.9, filled, 0.0)
            speaker_probs[start:end + 1] = filled

        cap.release()
        return speaker_probs


def speaker_features_pipeline(
        speaker_image_path: str, 
        video_path: str,
        yolo_model_path: str,
        arcface_weight_file: str,
        transnet_weights_path: str
    ) -> pd.DataFrame:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    speaker_feature_extractor = SpeakerFeatures(
        yolo_model_path, 
        arcface_weight_file
    )

    emotion_detector = EmotionsDetection(
        yolo_model_path, 
        device=device
    )

    shot_bounds = batch_shot_segmentation(
        video_path, 
        transnet_weights_path
    ) 

    if shot_bounds.ndim != 2 or shot_bounds.shape[1] != 2:
        raise ValueError(f"Expected shot_bounds shape (N, 2), got {shot_bounds.shape}")
    
    if shot_bounds.shape[0] == 0:
        raise ValueError("No shot boundaries detected in the video.")

    speaker_probs = speaker_feature_extractor.get_speaker_probs(
        shot_bounds, 
        speaker_image_path,
        video_path
    )

    emotions = emotion_detector.get_emotions(
        video_path, 
        speaker_probs
    )

    total_frames = len(speaker_probs)
    data = {
        'frame_index': np.arange(total_frames),
        'speaker_prob': speaker_probs,
        **{emotion: emotions[emotion] for emotion in emotions},
    }
    df = pd.DataFrame(data)
    return df
