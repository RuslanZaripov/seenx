import torch
import argparse
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from functools import partial
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig
from .mm_processing import process_video
from .mm_models import build_vision_tower, build_audio_tower
from .mm_projector import build_vision_projector, build_audio_projector
from .mm_arch import encode_images_or_videos
from ..logger import Logger
from .mm_constants import NUM_FRAMES
from ..aggregator import get_retention  # your method to get retention
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


logger = Logger(show=True).get_logger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


class VideoRetentionDataset(Dataset):
    def __init__(
        self,
        video_path: str,
        html_path: str,
        intervals: int = 10,
        num_frames: int = NUM_FRAMES,
        processor=None,
    ):
        """
        Args:
            video_path: path to video
            html_path: path to retention HTML
            intervals: number of time intervals to sample
            num_frames: frames per interval
        """
        self.video_path = video_path
        self.retention = get_retention(video_path, html_path)  # (T, 1)
        self.intervals = intervals
        self.num_frames = num_frames
        self.processor = processor

        total_duration = self.retention.shape[0]
        self.starts = np.linspace(0, total_duration - num_frames, intervals, dtype=int)
        self.ends = self.starts + num_frames

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s, e = self.starts[idx], self.ends[idx]
        data = self.processor(self.video_path, s=s, e=e, va=True)

        video_tensor = data["video"].half().to(device)
        audio_tensor = data["audio"].half().to(device)
        retention_tensor = torch.tensor(
            self.retention[s:e], dtype=torch.float32, device=device
        )

        return video_tensor, audio_tensor, retention_tensor


def train(
    video_path: str,
    html_path: str,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 1e-4,
):
    logger.info("Loading multimodal models...")

    config = AutoConfig.from_pretrained("DAMO-NLP-SG/VideoLLaMA2.1-7B-AV")
    config.mm_vision_tower = "google/siglip-so400m-patch14-384"
    config.mm_projector_type = "stc_connector_v35"
    config.mm_audio_tower = "audio_tower.bin"
    config.mm_projector_a_type = "mlp2x_gelu"

    vision_tower = build_vision_tower(config).half().to(device)
    vision_tower.requires_grad_(False)

    audio_tower, _ = build_audio_tower(config)
    audio_tower = audio_tower.half().to(device)
    audio_tower.requires_grad_(False)

    vision_projector = build_vision_projector(config).half().to(device)
    audio_projector = build_audio_projector(config).half().to(device)
    unfreeze_module(vision_projector)
    unfreeze_module(audio_projector)

    processor = partial(
        process_video,
        processor=vision_tower.image_processor,
        aspect_ratio=None,
        num_frames=config.num_frames if hasattr(config, "num_frames") else NUM_FRAMES,
    )

    dataset = VideoRetentionDataset(video_path, html_path, processor=processor)

    for video_batch, audio_batch, retention_batch in tqdm(
        dataset, desc="Training step"
    ):
        print("Video batch shape:", video_batch.shape)
        print("Audio batch shape:", audio_batch.shape)
        print("Retention batch shape:", retention_batch.shape)

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(
    #     list(vision_projector.parameters()) + list(audio_projector.parameters()), lr=lr
    # )

    # logger.info("Starting training...")
    # for epoch in range(epochs):
    #     epoch_loss = 0
    #     all_preds = []
    #     all_targets = []

    #     for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
    #         video_batch, audio_batch, retention_batch = batch

    #         video_features = encode_images_or_videos(
    #             vision_tower, vision_projector, [(video_batch, "video")], config
    #         )
    #         video_features = video_features[0]

    #         audio_embedding, T, F = audio_tower.extract_features(
    #             audio_batch,
    #             padding_mask=torch.zeros(audio_batch.shape, device=device).bool(),
    #         )
    #         audio_features = audio_projector(audio_embedding)

    #         multimodal_features = torch.cat([video_features, audio_features], dim=-1)

    #         pred = multimodal_features.mean(dim=-1).unsqueeze(-1)  # (B, T, 1)
    #         loss = criterion(pred, retention_batch)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()

    #         all_preds.append(pred.detach().cpu().numpy().reshape(-1))
    #         all_targets.append(retention_batch.detach().cpu().numpy().reshape(-1))

    #     all_preds = np.concatenate(all_preds)
    #     all_targets = np.concatenate(all_targets)

    #     mse = mean_squared_error(all_targets, all_preds)
    #     mae = mean_absolute_error(all_targets, all_preds)
    #     r2 = r2_score(all_targets, all_preds)

    #     logger.info(
    #         f"Epoch {epoch+1} | Loss: {epoch_loss/len(dataloader):.4f} | "
    #         f"MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}"
    #     )

    logger.info("Training complete.")
    return vision_projector, audio_projector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--html_path", type=str, required=True)
    args = parser.parse_args()
    train(args.video_path, args.html_path, epochs=5, batch_size=4, lr=1e-4)
