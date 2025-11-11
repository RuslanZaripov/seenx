import os
import cv2
import torch
import numpy as np
from transnetv2_pytorch import TransNetV2
from tqdm import tqdm
from logger import Logger

logger = Logger(show=True).get_logger()

def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
    predictions = (predictions > threshold).astype(np.uint8)

    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # just fix if all predictions are 1
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)


def batch_shot_segmentation(
    video_path: str, 
    transnet_weights_path: str, 
    batch_size: int = 1000,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")

    if not os.path.exists(transnet_weights_path):
        raise FileNotFoundError(f"TransNetV2 weights not found at {transnet_weights_path}")

    model = TransNetV2()
    state_dict = torch.load(transnet_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    logger.info(f"Video FPS: {fps}, Total Frames: {frame_count}, Duration: {duration:.2f} seconds")

    all_frame_pred = np.zeros(frame_count, dtype=np.float32)

    frames_batch = []
    frame_indices = []
    current_index = 0

    def process_batch(frames_batch, frame_indices):
        batch_tensor = torch.tensor(np.stack(frames_batch), dtype=torch.uint8)
        single_pred, _ = model(batch_tensor.unsqueeze(0).to(device))
        preds = torch.sigmoid(single_pred).cpu().numpy().squeeze(0)
        all_frame_pred[frame_indices] = preds[:len(frame_indices)].flatten()

    with torch.no_grad():
        pbar = tqdm(total=frame_count, desc="Processing frames", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(frame_rgb, (48, 27), interpolation=cv2.INTER_AREA)
            frames_batch.append(img)
            frame_indices.append(current_index)
            current_index += 1
            pbar.update(1)

            if len(frames_batch) == batch_size:
                process_batch(frames_batch, frame_indices)

                frames_batch.clear()
                frame_indices.clear()

        if len(frames_batch) > 0:
            process_batch(frames_batch, frame_indices)

        pbar.close()

    cap.release()

    scenes = predictions_to_scenes(all_frame_pred)
    return scenes
