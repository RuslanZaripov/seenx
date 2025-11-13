import numpy as np
import pandas as pd
from logger import Logger
from speaker_features import speaker_features_pipeline, get_frame_features
from sound_features import sound_features_pipeline
from zoom_features import zoom_features_pipeline
from parse_retention import parse_retention
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

    if retention.index.name is None:
        retention.index.name = "time"

    logger.info("Extracting speaker features")
    speaker_features = speaker_features_pipeline(
        speaker_image_path=config.get("speaker_image_path"),
        video_path=video_path,
        yolo_model_path=config.get("face_detector"),
        arcface_weight_file=config.get("face_embedder"),
        transnet_weights_path=config.get("shot_segmentor"),
    )

    logger.info("Extracting sound features")
    sound_features = sound_features_pipeline(audio_file_path=audio_path)

    logger.info("Extracting zoom features")
    zoom_features = zoom_features_pipeline(
        video_file_path=video_path,
        show=False,
        gpu=False,
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

    zoom_features_mapped = map_features_to_retention_index(retention, zoom_features)

    aggregated = pd.concat(
        [
            retention,
            speaker_features_mapped,
            sound_features_mapped,
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
