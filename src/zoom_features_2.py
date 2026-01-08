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
from typing import List

logger = Logger(show=True).get_logger()


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


ROOT = os.getcwd()
path = os.path.join(ROOT, "RAFT/core")
logger.info(f"Adding {path} to sys.path")
add_path(path)

from raft import RAFT
from utils.utils import InputPadder

# from utils import flow_viz


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DOWNSCALE = 0.5
BATCH_SIZE = 64
FLOW_STRIDE = 8


def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().to(DEVICE)


def make_center_grid(h, w, device, stride):
    y, x = torch.meshgrid(
        torch.arange(0, h, stride, device=device),
        torch.arange(0, w, stride, device=device),
        indexing="ij",
    )
    cx, cy = w * 0.5, h * 0.5
    dx, dy = x - cx, y - cy
    base_dist = torch.sqrt(dx**2 + dy**2) + 1e-6
    return x, y, dx, dy, base_dist


def compute_flow_features(flow, grid):
    """
    flow: [B, 2, H, W]
    """
    x, y, dx, dy, base_dist = grid
    fx = flow[:, 0][:, ::FLOW_STRIDE, ::FLOW_STRIDE]
    fy = flow[:, 1][:, ::FLOW_STRIDE, ::FLOW_STRIDE]

    mag = torch.sqrt(fx**2 + fy**2)

    # radial motion (projection)
    radial = (fx * dx + fy * dy) / base_dist

    mag_med = mag.flatten(1).median(dim=1).values
    radial_med = radial.flatten(1).median(dim=1).values
    radial_ratio = (radial > 0).float().mean(dim=(1, 2))

    return mag_med, radial_med, radial_ratio


# def viz(img, flo):
#     img = img[0].permute(1, 2, 0).cpu().numpy()
#     flo = flo[0].permute(1, 2, 0).cpu().numpy()
#     flo = flow_viz.flow_to_image(flo)
#     img_flo = np.concatenate([img, flo], axis=0)
#     plt.imshow(img_flo / 255.0)
#     plt.show()


def process_batch(
    model,
    fs: List[np.ndarray],
    grid: tuple,
    features: List[dict],
    start_idx: int,
    use_amp: bool,
):
    fs_tensor = frames_to_tensor(fs)
    img1 = fs_tensor[:-1]
    img2 = fs_tensor[1:]

    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)

    with torch.autocast(DEVICE, enabled=use_amp):
        _, flow_up = model(img1, img2, iters=20, test_mode=True)

    flow_up = padder.unpad(flow_up)

    mag, radial, ratio = compute_flow_features(flow_up, grid)

    mag = mag.cpu().numpy()
    radial = radial.cpu().numpy()
    ratio = ratio.cpu().numpy()

    for i in range(len(mag)):
        features.append(
            {
                # "frame": start_idx + i,
                "flow_mag_med": float(mag[i]),
                "radial_med": float(radial[i]),
                "radial_ratio": float(ratio[i]),
            }
        )

    del fs_tensor, img1, img2, flow_up
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def zoom_features_pipeline(args) -> pd.DataFrame:
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    model = model.module
    model.to(DEVICE)
    model.eval()

    use_amp = args.mixed_precision and DEVICE == "cuda"

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()
    if not ret:
        logger.error("Cannot read video")
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE)

    h, w, _ = frame.shape
    grid = make_center_grid(h, w, DEVICE, FLOW_STRIDE)

    batch = [frame]
    features = [
        {
            # "frame": 0,
            "flow_mag_med": 0.0,
            "radial_med": 0.0,
            "radial_ratio": 0.0,
        }
    ]

    frame_idx = 1

    with torch.no_grad(), tqdm(total=total_frames - 1, desc="Zoom features") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE)
            batch.append(frame)

            if len(batch) == BATCH_SIZE + 1:
                process_batch(model, batch, grid, features, frame_idx, use_amp)
                frame_idx += len(batch) - 1
                batch = [batch[-1]]

            pbar.update(1)

        if len(batch) > 1:
            process_batch(model, batch, grid, features, frame_idx, use_amp)

    cap.release()
    return pd.DataFrame(features)


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
        help="use efficient correlation implementation",
    )

    args = parser.parse_args()
    print(args)

    df = zoom_features_pipeline(args)
    print(df)
