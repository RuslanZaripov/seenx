import torch
import einops
from .mm_constants import NUM_FRAMES
from ..logger import Logger

logger = Logger(show=True).get_logger()


def temporal_aggregator(mm_projector, config, frames_features):
    """Temporal aggregation of frame features.
    Args:
        frames_features (torch.Tensor): Frame features with shape (b, t, n, h).
    Returns:
        torch.Tensor: Video features with shape (b, n, h).
    """
    # TODO: improve the merging method.
    # *********** mean pooling *************
    if config.mm_projector_type == "mlp2x_gelu" or config.mm_projector_type == "linear":
        video_features = mm_projector(frames_features.mean(1))
    # # *********** spatial convolution *************
    elif config.mm_projector_type == "spatial_conv":
        video_features = mm_projector(frames_features)
    # # *********** spatial pooling *************
    elif config.mm_projector_type == "spatial_pool":
        video_features = mm_projector(frames_features)
    # # *********** time  ************
    elif (
        "tc_connector" in config.mm_projector_type
        or "tp_connector" in config.mm_projector_type
    ):
        video_features = mm_projector(frames_features)
    else:
        raise Exception(f"Unsupported projector type {config.mm_projector_type}!!!")

    return video_features


def encode_images_or_videos(vision_tower, mm_projector, images, config):
    num_frames = config.num_frames if hasattr(config, "num_frames") else NUM_FRAMES

    data_batch = []
    for i, (data, modal) in enumerate(images):
        if modal == "image":
            data = data.expand(num_frames, -1, -1, -1)
        else:
            data = data
        data_batch.append(data)

    data_batch = torch.stack(data_batch, dim=0)

    assert len(data_batch.size()) == 5
    batch_size = data_batch.size(0)

    frames = einops.rearrange(data_batch, "b t c h w -> (b t) c h w")
    frames_features = vision_tower(frames)
    frames_features = einops.rearrange(
        frames_features, "(b t) n h -> b t n h", b=batch_size
    )

    return temporal_aggregator(mm_projector, config, frames_features)
