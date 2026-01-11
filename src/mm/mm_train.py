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


def build_mm_components(
    model_name: str = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV",
    device: torch.device = device,
):
    """
    Build multimodal towers, projectors, and processor.
    """
    logger.info("Initializing multimodal components...")

    config = AutoConfig.from_pretrained(model_name)

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

    logger.info("Multimodal components initialized.")

    return {
        "config": config,
        "vision_tower": vision_tower,
        "audio_tower": audio_tower,
        "vision_projector": vision_projector,
        "audio_projector": audio_projector,
        "processor": processor,
    }


class MultiVideoRetentionDataset(Dataset):
    def __init__(
        self,
        videos: list,
        processor,
        interval_len: int = 10,  # seconds
        stride: int = 5,  # overlap
    ):
        """
        videos: list of dicts with keys {video_path, html_path}
        """
        self.processor = processor
        self.samples = []

        logger.info("Building multi-video dataset...")

        for vid in videos:
            video_path = vid["video_path"]
            html_path = vid["html_path"]

            retention_df = get_retention(video_path, html_path)  # (T, 1)
            retention = retention_df.values.astype("float32")  # (T, 1)

            T = retention.shape[0]

            for start in range(0, T - interval_len, stride):
                end = start + int(interval_len)

                self.samples.append(
                    {
                        "video_path": video_path,
                        "start": start,
                        "end": end,
                        "retention": retention[start:end],  # numpy slice
                    }
                )

        logger.info(f"Dataset size: {len(self.samples)} intervals")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        data = self.processor(
            s["video_path"],
            s=s["start"],
            e=s["end"],
            va=True,
        )

        return {
            "video": data["video"].half(),
            "audio": data["audio"].half(),
            "retention": torch.tensor(s["retention"], dtype=torch.float32),
        }


def train(
    data: list[dict],
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 1e-4,
):
    mm = build_mm_components()

    config = mm["config"]
    vision_tower = mm["vision_tower"]
    audio_tower = mm["audio_tower"]
    vision_projector = mm["vision_projector"]
    audio_projector = mm["audio_projector"]
    processor = mm["processor"]

    dataset = MultiVideoRetentionDataset(
        data,
        processor=processor,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(vision_projector.parameters()) + list(audio_projector.parameters()), lr=lr
    )

    logger.info("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        all_preds, all_targets = [], []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):

            video_batch = batch["video"].to(device)
            audio_batch = batch["audio"].to(device)
            retention_batch = batch["retention"].to(device)

            video_features = encode_images_or_videos(
                vision_tower,
                vision_projector,
                [(video_batch, "video")],
                config,
            )[0]

            audio_embedding, _, _ = audio_tower.extract_features(
                audio_batch,
                padding_mask=torch.zeros(audio_batch.shape, device=device).bool(),
            )
            audio_features = audio_projector(audio_embedding)

            multimodal_features = torch.cat(
                [video_features, audio_features],
                dim=-1,
            )

            pred = multimodal_features.mean(dim=-1).unsqueeze(-1)
            loss = criterion(pred, retention_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_preds.append(pred.detach().cpu().numpy().reshape(-1))
            all_targets.append(retention_batch.detach().cpu().numpy().reshape(-1))

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        logger.info(
            f"Epoch {epoch+1} | "
            f"Loss: {epoch_loss/len(dataloader):.4f} | "
            f"MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}"
        )

    logger.info("Training complete.")
    return vision_projector, audio_projector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--html_path", type=str, required=True)
    args = parser.parse_args()
    data = [{"video_path": args.video_path, "html_path": args.html_path}]
    train(data, epochs=5, batch_size=4, lr=1e-4)
