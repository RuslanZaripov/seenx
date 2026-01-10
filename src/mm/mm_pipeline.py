import os
import argparse
import torch
from functools import partial
from transformers import AutoConfig
from mm.mm_constants import NUM_FRAMES
from mm.mm_processing import (
    process_video,
)
from mm.mm_models import (
    build_vision_tower,
    build_audio_tower,
)
from mm.mm_projector import (
    build_vision_projector,
    build_audio_projector,
)
from mm.mm_arch import encode_images_or_videos


def pipeline_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_path = args.video_path

    working_dir = os.getcwd()
    model_path = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    config = AutoConfig.from_pretrained(model_path)
    config.mm_audio_tower = f"{working_dir}/audio_tower.bin"
    config.mm_vision_tower = "google/siglip-so400m-patch14-384"
    # print(config)

    num_frames = config.num_frames if hasattr(config, "num_frames") else NUM_FRAMES
    vision_tower = build_vision_tower(config).half().to(device)
    mm_projector = build_vision_projector(config).half().to(device)

    audio_tower, audio_tower_cfg = build_audio_tower(config)
    audio_tower = audio_tower.half().to(device)
    mm_projector_a = build_audio_projector(config).half().to(device)

    processor = {
        "video": partial(
            process_video,
            processor=vision_tower.image_processor,
            aspect_ratio=None,
            num_frames=num_frames,
        ),
    }
    image_or_video = processor["video"](video_path, va=True)

    tensor = {k: v.half().to(device) for k, v in image_or_video.items()}
    tensor = [(tensor, "video")]

    X_video = []
    X_audio = []

    select_audio_id = []
    select_videoimage_id = []
    for idx, data_list in enumerate(tensor):
        if isinstance(data_list[0], dict):
            assert data_list[1] == "video"
            X_audio.append(data_list[0]["audio"])
            select_audio_id.append(True)
            X_video.append((data_list[0]["video"], "video"))
            select_videoimage_id.append(True)
        else:
            if data_list[1] == "audio":
                X_audio.append(data_list[0])
                select_audio_id.append(True)
                select_videoimage_id.append(False)
            elif data_list[1] == "video" or data_list[1] == "image":
                X_video.append(data_list)
                select_videoimage_id.append(True)
                select_audio_id.append(False)
            else:
                raise NotImplementedError

    if len(X_audio) > 0:
        Xa_features = torch.cat(X_audio, dim=0)
        audio_padding_mask = torch.zeros(Xa_features.shape, device=device).bool()
        audio_embedding, T, F = audio_tower.extract_features(
            Xa_features, padding_mask=audio_padding_mask
        )
        Xa_features = mm_projector_a(audio_embedding)
        Xa_features = Xa_features.view(len(X_audio), -1, Xa_features.shape[-1])

    if len(X_video) > 0:
        X_features = encode_images_or_videos(
            vision_tower, mm_projector, X_video, config
        )

    mm_features = []
    idx_a, idx_v = 0, 0
    for audio_idx, videoimage_idx in zip(select_audio_id, select_videoimage_id):
        if audio_idx and videoimage_idx:
            mm_features.append(
                torch.cat([X_features[idx_v], Xa_features[idx_a]], dim=0)
            )
            idx_a += 1
            idx_v += 1
        elif audio_idx:
            mm_features.append(Xa_features[idx_a])
            idx_a += 1
        elif videoimage_idx:
            mm_features.append(X_features[idx_v])
            idx_v += 1
        else:
            raise NotImplementedError

    print("Multimodal features shape:", mm_features[0].shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        default="/kaggle/input/seenx-data/videos/faceless_youtube_channel_ideas.mp4",
        help="Path to the input video file.",
    )

    args = parser.parse_args()
    pipeline_demo(args)
