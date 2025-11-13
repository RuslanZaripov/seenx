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

    logger.info("Extracting speaker features")
    speaker_features = speaker_features_pipeline(
        speaker_image_path=config["speaker_image_path"],
        video_path=video_path,
        yolo_model_path=config["face_detector"],
        arcface_weight_file=config["face_embedder"],
        transnet_weights_path=config["shot_segmentor"],
    )

    logger.info("Extracting sound features")
    sound_features = sound_features_pipeline(audio_file_path=audio_path)

    logger.info("Extracting zoom features")
    zoom_features = zoom_features_pipeline(
        video_file_path=video_path,
        show=False,
        gpu=False,
    )

    # See how much seconds retention data consist of

    # speaker features are framewise, match speaker features to retention timepoints
    # sound features are also framewise, match sound features to retention timepoints
    # zoom features are framewise, match zoom features to retention timepoints
    # add then to retention dataframe

    # map speaker_features columns by windows of length len(speaker_features.index) / total_duration_seconds
    # i want speaker features column equal to length of retention index
    speaker_features_mapped = pd.DataFrame(index=retention.index)
    for col in speaker_features.columns:
        speaker_features_mapped[col] = np.interp(
            np.linspace(0, len(speaker_features.index), len(retention.index)),
            np.arange(len(speaker_features.index)),
            speaker_features[col].values,
        )

    sound_features_mapped = pd.DataFrame(index=retention.index)
    for col in sound_features.columns:
        sound_features_mapped[col] = np.interp(
            np.linspace(0, len(sound_features.index), len(retention.index)),
            np.arange(len(sound_features.index)),
            sound_features[col].values,
        )
    # same for zoom features
    zoom_features_mapped = pd.DataFrame(index=retention.index)
    for col in zoom_features.columns:
        zoom_features_mapped[col] = np.interp(
            np.linspace(0, len(zoom_features.index), len(retention.index)),
            np.arange(len(zoom_features.index)),
            zoom_features[col].values,
        )

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
