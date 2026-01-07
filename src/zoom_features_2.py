import os
import sys

root = os.getcwd()
sys.path.append(f"{root}/RAFT/core")

import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def frame_to_tensor(frame):
    """Convert OpenCV BGR frame to [1,3,H,W] torch tensor on DEVICE"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float()[None].to(DEVICE)
    return img


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
    ret, prev_frame = cap.read()
    if not ret:
        print("Cannot read video")
        return

    h, w, _ = prev_frame.shape

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    center_x, center_y = w / 2, h / 2
    empty_dists = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    prev_tensor = frame_to_tensor(prev_frame)
    padder = InputPadder(prev_tensor.shape)

    frame_features = []

    with torch.no_grad():
        frame_idx = 0
        while True:
            ret, next_frame = cap.read()
            if not ret:
                break
            next_tensor = frame_to_tensor(next_frame)

            img1, img2 = padder.pad(prev_tensor, next_tensor)

            flow_low, flow_up = model(img1, img2, iters=20, test_mode=True)

            flow_x = flow_up[0, 0].cpu().numpy()
            flow_y = flow_up[0, 1].cpu().numpy()

            mag = np.sqrt(flow_x**2 + flow_y**2)
            ang = (np.arctan2(flow_y, flow_x) * 180 / np.pi) % 360
            mean_mag = float(np.median(mag))
            mean_ang = float(np.median(ang))

            new_x = x + flow_x
            new_y = y + flow_y
            dists = np.sqrt((new_x - center_x) ** 2 + (new_y - center_y) ** 2)

            zoom_in_factor = np.count_nonzero(dists >= empty_dists) / empty_dists.size

            frame_features.append(
                {
                    "frame": frame_idx,
                    "mag": mean_mag,
                    "ang": mean_ang,
                    "zoom": zoom_in_factor,
                }
            )

            # viz(img1, flow_up)

            prev_tensor = next_tensor
            frame_idx += 1

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
