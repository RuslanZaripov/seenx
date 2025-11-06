import torch
from torchvision.io import VideoReader
from torchvision.transforms.functional import resize
from transnetv2_pytorch import TransNetV2
from tqdm import tqdm
import numpy as np

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

def shot_segmentation(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vr = VideoReader(path, "video")
    info = vr.get_metadata()
    fps = info['video']["fps"][0]
    duration = info["video"]["duration"][0]
    num_frames = int(duration * fps)
    
    frames = []
    for i, frame in tqdm(enumerate(vr), total=num_frames):
        img = frame['data']
        img = resize(frame['data'], [27, 48])
        frames.append(img)
    video = torch.stack(frames) # (num_frames, H, W, 3)

    model = TransNetV2()
    state_dict = torch.load("weights/transnetv2-pytorch-weights.pth")
    model.load_state_dict(state_dict)
    model.eval().to(device)

    with torch.no_grad():
        # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
        input_video = video.permute(0, 2, 3, 1).unsqueeze(0).to(device)
        single_frame_pred, all_frame_pred = model(input_video)
        
        single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
        all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()

    scenes = predictions_to_scenes(single_frame_pred.squeeze(0))
    
    return scenes
