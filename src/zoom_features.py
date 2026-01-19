import os
import sys
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from pathlib import Path

from .config import Config
from .video_dataset import VideoBatchDataset
from .logger import Logger

logger = Logger(show=True).get_logger()


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


ROOT = os.environ.get("WORKING_DIR", str(Path(__file__).resolve().parent.parent.parent))
RAFT_PATH = os.path.join(ROOT, "RAFT/core")
logger.info(f"{__file__}: adding {RAFT_PATH} to sys.path")
add_path(RAFT_PATH)

from raft import RAFT
from utils.utils import InputPadder


class ZoomFeatureExtractor:
    def __init__(self, args, config: Config):
        self.args = args
        self.config = config

        self.device = torch.device(config.get("device"))
        self.flow_stride = config.get("flow_stride", 8)
        self.batch_size = config.get("batch_size")

        self.use_amp = args.mixed_precision and self.device.type == "cuda"

        self.model = self._load_model()

    def _load_model(self):
        model = torch.nn.DataParallel(RAFT(self.args))
        model.load_state_dict(torch.load(self.args.model, map_location=self.device))
        model = model.module
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def transform(frame: np.ndarray) -> np.ndarray:
        DOWNSCALE = 0.5
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE)
        return frame[np.newaxis, :, :, :]  # (1, H, W, 3)

    @staticmethod
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

    @staticmethod
    def compute_flow_features(flow, grid, flow_stride):
        x, y, dx, dy, base_dist = grid

        fx = flow[:, 0][:, ::flow_stride, ::flow_stride]
        fy = flow[:, 1][:, ::flow_stride, ::flow_stride]

        mag = torch.sqrt(fx**2 + fy**2)
        radial = (fx * dx + fy * dy) / base_dist

        mag_med = mag.flatten(1).median(dim=1).values
        radial_med = radial.flatten(1).median(dim=1).values
        radial_ratio = (radial > 0).float().mean(dim=(1, 2))

        return mag_med, radial_med, radial_ratio

    def _process_batch(
        self,
        fs: np.ndarray,
        indices: List[int],
        grid: tuple,
        features: List[dict],
        start_idx: int,
    ):
        fs_tensor = (
            torch.from_numpy(fs).permute(0, 3, 1, 2).float().to(self.device)
        )  # (B, 3, H, W)

        img1 = fs_tensor[:-1]
        img2 = fs_tensor[1:]

        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)

        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.use_amp,
        ):
            _, flow_up = self.model(img1, img2, iters=20, test_mode=True)

        flow_up = padder.unpad(flow_up)

        mag, radial, ratio = self.compute_flow_features(flow_up, grid, self.flow_stride)

        mag = mag.cpu().numpy()
        radial = radial.cpu().numpy()
        ratio = ratio.cpu().numpy()

        logger.debug(f"Processed frames: {len(mag)=} {len(indices)=}")

        for i in range(len(mag)):
            features.append(
                {
                    "frame": indices[i],
                    "flow_mag_med": float(mag[i]),
                    "radial_med": float(radial[i]),
                    "radial_ratio": float(ratio[i]),
                }
            )

        del fs_tensor, img1, img2, flow_up
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def run(self) -> pd.DataFrame:
        dataset = VideoBatchDataset(
            video_path=self.args.video,
            batch_size=self.batch_size,
            transform=self.transform,
            stride=10,
        )

        iterator = iter(dataset)

        frames, indices = next(iterator)  # (B+1, H, W, 3)

        _, h, w, _ = frames.shape
        grid = self.make_center_grid(h, w, self.device, self.flow_stride)

        features = [
            {
                "frame": 0,
                "flow_mag_med": 0.0,
                "radial_med": 0.0,
                "radial_ratio": 0.0,
            }
        ]

        self._process_batch(
            fs=frames,
            indices=indices,
            grid=grid,
            features=features,
            start_idx=1,
        )

        prev_last = frames[-1:]
        # frame_idx = len(frames)

        for frames, indices in tqdm(iterator, total=len(dataset) - 1):
            frames = np.concatenate([prev_last, frames], axis=0)

            self._process_batch(
                fs=frames,
                indices=indices,
                grid=grid,
                features=features,
                start_idx=1,
            )

            # frame_idx += len(frames) - 1
            prev_last = frames[-1:]

        features = pd.DataFrame(features)
        features = (
            features.set_index("frame")
            .reindex(range(dataset.total_frames))
            .fillna(method="ffill")
            .fillna(method="bfill")
            .reset_index()
        )
        return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--alternate_corr", action="store_true")

    args = parser.parse_args()

    config = Config(args.config)
    extractor = ZoomFeatureExtractor(args, config)

    df = extractor.run()
    print(df)
