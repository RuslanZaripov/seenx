import numpy as np
import pandas as pd
from logger import Logger
from speaker_features import speaker_features_pipeline, get_frame_features
from music_features import get_vocal_music_features
from sound_features import sound_features_pipeline
from zoom_features import zoom_features_pipeline
from parse_retention import parse_retention
from transcribe import collect_wps
from config import Config

logger = Logger(show=True).get_logger()


def aggregate(
    html_path: str,
    video_path: str,
    audio_path: str,
    config: Config,
):
    logger.info("Parsing audience retention data")
    retention = parse_retention(
        html_file_path=html_path,
    )

    # take only until 2 second included
    # retention = retention[retention.index <= pd.to_timedelta("2s")]

    if retention.index.name is None:
        retention.index.name = "time"

    logger.info("Extracting speaker features")
    speaker_features = speaker_features_pipeline(
        video_path=video_path,
        config=config,
    )

    logger.info("Extracting sound features")
    sound_features = sound_features_pipeline(audio_file_path=audio_path)

    logger.info("Extracting music features")
    music_features, vocal_features = get_vocal_music_features(config=config)

    logger.info("Extracting zoom features")
    zoom_features = zoom_features_pipeline(
        video_file_path=video_path,
        show=False,
        gpu=False,
    )

    logger.info("Collecting words per second data")
    wps_df = collect_wps(
        video_path=video_path,
        config=config,
    )
    retention = retention.merge(
        wps_df,
        how="left",
        left_index=True,
        right_index=True,
    )

    def map_features_to_retention_index(retention, features):
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

    aggregated = pd.concat(
        [
            retention,
            speaker_features_mapped,
            sound_features_mapped,
            music_features_mapped,
            vocal_features_mapped,
            zoom_features_mapped,
        ],
        axis=1,
    )

    return aggregated


if __name__ == "__main__":
    config = Config("configs/local.json")

    aggregated_df = aggregate(
        html_path="static/htmls/faceless_youtube_channel_ideas.html",
        video_path="static/test.mp4",
        audio_path="static/test.mp4",
        config=config,
    )

    aggregated_df.to_csv("static/data.csv", index=True)
