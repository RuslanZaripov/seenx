import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from .config import Config
from .video_dataset import VideoBatchDataset
from .logger import Logger
from .transnetv2_pytorch import TransNetV2

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
    config: Config,
) -> np.ndarray:
    device = torch.device(config.get("device"))
    transnet_weights_path = config.get("shot_segmentor")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")

    if not os.path.exists(transnet_weights_path):
        raise FileNotFoundError(
            f"TransNetV2 weights not found at {transnet_weights_path}"
        )

    model = TransNetV2()
    state_dict = torch.load(transnet_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    def transform(frame: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame_rgb, (48, 27), interpolation=cv2.INTER_AREA)
        return img[np.newaxis, :, :, :]

    dataset = VideoBatchDataset(
        video_path,
        batch_size=config.get("shot_segmentor_batch_size"),
        transform=transform,
    )
    all_frame_pred = np.zeros(dataset.total_processed_frames, dtype=np.float32)

    for frames_batch, frame_indices in tqdm(
        dataset,
        desc="Running shot segmentation",
    ):
        with torch.no_grad():
            batch_tensor = torch.tensor(frames_batch, dtype=torch.uint8).to(device)
            single_pred, _ = model(batch_tensor.unsqueeze(0))
            preds = torch.sigmoid(single_pred).cpu().numpy().squeeze(0)
            all_frame_pred[frame_indices] = preds[: len(frame_indices)].flatten()

    scenes = predictions_to_scenes(all_frame_pred)
    intervals_str = ", ".join([f"{start}:{end}" for start, end in scenes])
    logger.info(f"Detected scenes (start_frame, end_frame): {intervals_str}")
    return scenes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to the config file",
    )
    args = parser.parse_args()

    config = Config(args.config)
    video_path = args.video_path

    scenes = batch_shot_segmentation(
        args.video_path,
        config=config,
    )
