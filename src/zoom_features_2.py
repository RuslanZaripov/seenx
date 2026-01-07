import os
import sys
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image
from logger import Logger

logger = Logger(show=True).get_logger()


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


root = os.getcwd()
path = f"{root}/RAFT/core"
logger.info(f"Adding {path} to sys.path")
add_path(path)

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8


def make_center_grid(h, w, device):
    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    cx, cy = w / 2.0, h / 2.0
    base_dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return x, y, base_dist


def compute_flow_features_torch(flow, x, y, base_dist, cx, cy):
    """
    flow: [B, 2, H, W]
    x,y,base_dist: [H, W] (torch, GPU)
    """

    flow_x = flow[:, 0]
    flow_y = flow[:, 1]

    # Magnitude & angle
    mag = torch.sqrt(flow_x**2 + flow_y**2)  # [B,H,W]
    ang = torch.rad2deg(torch.atan2(flow_y, flow_x)) % 360  # [B,H,W]

    mean_mag = torch.median(mag.flatten(1), dim=1).values
    mean_ang = torch.median(ang.flatten(1), dim=1).values

    # Zoom-in factor
    new_x = x + flow_x
    new_y = y + flow_y
    new_dist = torch.sqrt((new_x - cx) ** 2 + (new_y - cy) ** 2)

    zoom = (new_dist >= base_dist).float().mean(dim=(1, 2))

    return mean_mag, mean_ang, zoom


def frames_to_tensor_batch(frames):
    imgs = []
    for f in frames:
        img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        imgs.append(img)
    return torch.stack(imgs).to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    plt.imshow(img_flo / 255.0)
    plt.show()


def zoom_features_pipeline(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()
    if not ret:
        print("Cannot read video")
        return

    h, w, _ = frame.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    center_x, center_y = w / 2, h / 2
    empty_dists = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    frame_buffer = deque([frame], maxlen=batch_size + 1)
    frame_features = []
    frame_idx = 0

    with torch.no_grad(), tqdm(
        total=total_frames - 1, desc="Extracting zoom features"
    ) as pbar:

        while True:
            ret, next_frame = cap.read()
            if not ret:
                break

            frame_buffer.append(next_frame)
            if len(frame_buffer) < batch_size + 1:
                continue

            frames = list(frame_buffer)
            batch = frames_to_tensor_batch(frames)

            img1 = batch[:-1]
            img2 = batch[1:]

            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)

            _, flow_up = model(img1, img2, iters=20, test_mode=True)

            mean_mag, mean_ang, zoom = compute_flow_features_torch(
                flow_up, x, y, empty_dists, center_x, center_y
            )

            for b in range(flow_up.shape[0]):
                frame_features.append(
                    {
                        "frame": frame_idx,
                        "mag": float(mean_mag[b].cpu()),
                        "ang": float(mean_ang[b].cpu()),
                        "zoom": float(zoom[b].cpu()),
                    }
                )
                frame_idx += 1
                pbar.update(1)

                flow_viz(img1[b : b + 1], flow_up[b : b + 1])

            del (img1, img2, flow_up, batch)
            torch.cuda.empty_cache()

    cap.release()
    df = pd.DataFrame(frame_features)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--video", required=True, help="path to input mp4 video")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )

    manual_args = [
        "--model",
        f"{root}/models/raft-sintel.pth",
        "--video",
        "/kaggle/input/seenx-data/videos/faceless_youtube_channel_ideas.mp4",
    ]

    args = parser.parse_args()
    print(args)

    df = zoom_features_pipeline(args)
    print(df)
