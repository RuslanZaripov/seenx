import numpy as np
import argparse
import pandas as pd
import os
from logger import Logger
from speaker_features import speaker_features_pipeline
from sound_features import sound_features_pipeline, get_vocal_music_features
from src.speaker_feature_pass import (
    CinematicFeaturePass,
    EmotionFeaturePass,
    FaceScreenRatioFeaturePass,
    MotionSpeedFeaturePass,
    SpeakerProbabilityPass,
    TextProbFeaturePass,
    run_feature_pipeline,
)
from zoom_features_2 import zoom_features_pipeline
from parse_retention import parse_retention
from transcribe import collect_wps
from seenx_utils import get_video_duration
from config import Config

logger = Logger(show=True).get_logger()


def get_retention(video_path: str, html_path: str = None) -> pd.DatetimeIndex:
    video_duration = get_video_duration(video_path)
    if html_path is None:
        # create retention data frame with index of sec freq
        logger.info("Creating empty audience retention data")
        retention_index = pd.to_timedelta(
            np.arange(0, int(video_duration) + 1, 1), unit="s"
        )
        retention = pd.DataFrame(index=retention_index)
    else:
        logger.info("Parsing audience retention data")
        retention = parse_retention(
            html_file_path=html_path,
        )
    return retention


def aggregate(
    video_path: str,
    audio_path: str,
    output_path: str,
    config: Config,
    html_path: str = None,
):
    existing_features = []
    existing_df = pd.DataFrame()
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, index_col=0)
        existing_features = existing_df.columns.tolist()
        logger.info(f"Existing columns in output: {existing_features}")

    if "retention" not in existing_features:
        retention = get_retention(video_path=video_path, html_path=html_path)
        # take only until 2 second included
        # retention = retention[retention.index <= pd.to_timedelta("4s")]
        if retention.index.name is None:
            retention.index.name = "time"
    else:
        retention = existing_df[["retention"]]

    logger.info("Extracting speaker features")
    # speaker_features = speaker_features_pipeline(
    #     video_path=video_path,
    #     config=config,
    #     existing_features=existing_features,
    # )
    speaker_features = run_feature_pipeline(
        video_path,
        config,
        passes=[
            SpeakerProbabilityPass(config),
            FaceScreenRatioFeaturePass(config),
            TextProbFeaturePass(config),
            MotionSpeedFeaturePass(config),
            EmotionFeaturePass(config),
            CinematicFeaturePass(config),
        ],
        existing_features=existing_features,
    )

    logger.info("Extracting sound features")
    sound_features = sound_features_pipeline(
        audio_file_path=audio_path, existing_features=existing_features
    )

    logger.info("Extracting music features")
    music_features, vocal_features = get_vocal_music_features(
        audio_path=audio_path, config=config, existing_features=existing_features
    )

    logger.info("Extracting zoom features")
    zoom_features = zoom_features_pipeline(
        argparse.Namespace(
            model=config.get("optical_flow_model"),
            video=video_path,
            small=True,
            mixed_precision=True,
            alternate_corr=False,
        )
    )

    logger.info("Collecting words per second data")
    wps_features = collect_wps(
        video_path=video_path,
        config=config,
        existing_features=existing_features,
    )

    def map_features_to_retention_index(retention, features):
        if features.empty:
            return pd.DataFrame(index=retention.index)
        mapped = pd.DataFrame(index=retention.index)
        for col in features.columns:
            mapped[col] = np.interp(
                np.linspace(0, len(features.index), len(retention.index)),
                np.arange(len(features.index)),
                features[col].values,
            )
        return mapped

    speaker_features_mapped = map_features_to_retention_index(
        retention, speaker_features
    )

    sound_features_mapped = map_features_to_retention_index(retention, sound_features)

    music_features_mapped = map_features_to_retention_index(retention, music_features)

    vocal_features_mapped = map_features_to_retention_index(retention, vocal_features)

    zoom_features_mapped = map_features_to_retention_index(retention, zoom_features)

    wps_features_mapped = map_features_to_retention_index(retention, wps_features)

    aggregated = pd.concat(
        [
            retention,
            speaker_features_mapped,
            sound_features_mapped,
            music_features_mapped,
            vocal_features_mapped,
            zoom_features_mapped,
            wps_features_mapped,
        ],
        axis=1,
    )
    aggregated = pd.concat([existing_df, aggregated], axis=1)

    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--retention_path", type=str, required=False)
    parser.add_argument("-v", "--video_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument("-c", "--config_path", type=str, required=False)

    args = parser.parse_args()

    config = Config(config_path=args.config_path)

    aggregated_df = aggregate(
        video_path=args.video_path,
        audio_path=args.video_path,
        output_path=args.output_path,
        config=config,
        html_path=args.retention_path,
    )

    aggregated_df.to_csv(args.output_path, index=True)
