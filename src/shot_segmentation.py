import os
import cv2
import torch
import numpy as np
from transnetv2_pytorch import TransNetV2
from tqdm import tqdm
from logger import Logger

logger = Logger(show=True).get_logger('seenx')

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

def shot_segmentation(video_path, transnet_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    logger.info(f"Video FPS: {fps}, Total Frames: {frame_count}, Duration: {duration:.2f} seconds")

    frames = []
    for i in tqdm(range(frame_count), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame_rgb, (48, 27), interpolation=cv2.INTER_AREA)
        frames.append(img)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames were read from the video")

    video = torch.tensor(np.stack(frames), dtype=torch.uint8) # (num_frames, H, W, 3)

    if not os.path.exists(transnet_weights_path):
        raise FileNotFoundError(f"TransNetV2 weights not found at {transnet_weights_path}")

    model = TransNetV2()
    state_dict = torch.load(transnet_weights_path)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    with torch.no_grad():
        # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
        input_video = video.unsqueeze(0).to(device)
        logger.info(f"Input video: {input_video.shape} {input_video.dtype}")
        single_frame_pred, all_frame_pred = model(input_video)
        
        single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
        all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()

    scenes = predictions_to_scenes(single_frame_pred.squeeze(0))
    
    return scenes
