import argparse
import cv2
import pandas as pd
import numpy as np
from .config import Config
from .shot_segmentation import batch_shot_segmentation
from .logger import Logger
from .extractors import (
    CinematicFeaturePass,
    EmotionFeaturePass,
    FaceScreenRatioFeaturePass,
    MotionSpeedFeaturePass,
    SpeakerProbabilityPass,
    TextProbFeaturePass,
    FrameQualityFeaturePass,
    VideoFeaturePass,
)

logger = Logger(show=True).get_logger()


def run_feature_pipeline(
    video_path: str,
    config: Config,
    passes: list[VideoFeaturePass],
    existing_features: set,
) -> pd.DataFrame:
    context = {}

    vid_cap = cv2.VideoCapture(video_path)
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_cap.release()

    context["data"] = pd.DataFrame({"frame_idx": list(range(total_frames))})
    context["data"]["frame_idx"] = context["data"]["frame_idx"].astype("int32")

    shift = config.get("shot_bound_shift_frames")
    shot_bounds = batch_shot_segmentation(video_path, config.get("shot_segmentor"))
    shot_bounds[:, 0] = np.clip(shot_bounds[:, 0] + shift, 0, total_frames - 1)
    shot_bounds[:, 1] = np.clip(shot_bounds[:, 1] - shift, 0, total_frames - 1)
    context["shift"] = shift
    context["shot_bounds"] = shot_bounds.flatten().tolist()

    for p in passes:
        if p.produces_keys() & existing_features:
            continue

        missing = p.required_keys() - context.keys() - set(context["data"].columns)
        if missing:
            raise RuntimeError(f"Missing deps: {missing}")

        p.run(video_path, context)

    return context["data"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file.",
    )
    args = parser.parse_args()

    config = Config(args.config)
    features_df = run_feature_pipeline(
        args.video,
        config,
        passes=[
            FrameQualityFeaturePass(config),
            SpeakerProbabilityPass(config),
            FaceScreenRatioFeaturePass(config),
            TextProbFeaturePass(config),
            MotionSpeedFeaturePass(config),
            EmotionFeaturePass(config),
            CinematicFeaturePass(config),
        ],
        existing_features=set(),
    )

    print(features_df)
    features_df.to_csv("features.csv", index=False)
