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
BATCH_SIZE = 8
FLOW_STRIDE = 8
DOWNSCALE = 0.5


def make_center_grid(h, w, device, stride):
    y, x = torch.meshgrid(
        torch.arange(0, h, stride, device=device),
        torch.arange(0, w, stride, device=device),
        indexing="ij",
    )
    cx, cy = w / 2.0, h / 2.0
    base_dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return x, y, base_dist, cx, cy


def compute_flow_features(flow, x, y, base_dist, cx, cy, stride):
    flow_x = flow[:, 0][:, ::stride, ::stride]
    flow_y = flow[:, 1][:, ::stride, ::stride]

    mag = torch.sqrt(flow_x**2 + flow_y**2)
    ang = torch.rad2deg(torch.atan2(flow_y, flow_x)) % 360
    mean_mag = mag.flatten(1).median(dim=1).values
    mean_ang = ang.flatten(1).median(dim=1).values

    new_x = x + flow_x
    new_y = y + flow_y
    new_dist = torch.sqrt((new_x - cx) ** 2 + (new_y - cy) ** 2)
    zoom = (new_dist >= base_dist).float().mean(dim=(1, 2))
    return mean_mag, mean_ang, zoom


def frames_to_tensor(frames):
    arr = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

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

    frame = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE)
    h, w, _ = frame.shape

    x, y, base_dist, cx, cy = make_center_grid(h, w, DEVICE, FLOW_STRIDE)
    frame_idx = 0
    frame_features = []

    batch_frames = [frame]

    with torch.no_grad(), torch.autocast(
        device_type=DEVICE, enabled=args.mixed_precision
    ), tqdm(total=total_frames - 1, desc="Extracting zoom features") as pbar:

        def process_batch(fs):
            nonlocal frame_idx, frame_features, pbar
            batch_tensor = frames_to_tensor(fs)
            img1 = batch_tensor[:-1]
            img2 = batch_tensor[1:]

            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)

            _, flow_up = model(img1, img2, iters=20, test_mode=True)

            mean_mag, mean_ang, zoom = compute_flow_features(
                flow_up, x, y, base_dist, cx, cy, FLOW_STRIDE
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

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE)
            batch_frames.append(frame)

            if len(batch_frames) == BATCH_SIZE:
                process_batch(batch_frames)
                batch_frames = []

        if len(batch_frames) > 1:
            process_batch(batch_frames)

    cap.release()
    return pd.DataFrame(frame_features)


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
        "--small",
        "--mixed_precision",
    ]

    args = parser.parse_args()
    print(args)

    df = zoom_features_pipeline(args)
    print(df)
