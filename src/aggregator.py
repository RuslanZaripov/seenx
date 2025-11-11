import pandas as pd
from logger import Logger
from speaker_features import speaker_features_pipeline
from sound_features import sound_features_pipeline
from zoom_features import zoom_features_pipeline
from parse_retention import parse_retention


logger = Logger(show=True).get_logger('seenx')

def aggregate():
    logger.info("Starting aggregation of features")
    retention = parse_retention(
        html_file_path = '/kaggle/input/seenx-videos/faceless_youtube_channel_ideas.html'
    )
    
    logger.info("Extracting speaker features")
    speaker_features = speaker_features_pipeline(
        speaker_image_path = '/kaggle/input/seenx-videos/speaker_face.png', 
        video_path = '/kaggle/input/seenx-videos/youtube-video.mp4',
        yolo_model_path = '/kaggle/working/yolov12l-face.pt',
        arcface_weight_file = '/kaggle/working/arcface_weights.h5',
        transnet_weights_path = '/kaggle/input/seenx-videos/transnetv2-pytorch-weights.pth',
    )
    
    logger.info("Extracting sound features")
    sound_features = sound_features_pipeline(
        video_path = '/kaggle/input/seenx-videos/youtube-video.mp4'
    )

    logger.info("Extracting zoom features")
    zoom_features = zoom_features_pipeline(
        video_file_path = '/kaggle/input/seenx-videos/youtube-video.mp4',
        show=False, 
        gpu=False
    )

    # See how much seconds retention data consist of

    # speaker features are framewise, match speaker features to retention timepoints
    # sound features are also framewise, match sound features to retention timepoints
    # zoom features are framewise, match zoom features to retention timepoints
    # add then to retention dataframe

    total_duration_seconds = retention.index[-1].total_seconds()
    speaker_features_resampled = speaker_features.reindex(
        index = pd.timedelta_range(
            start = pd.Timedelta(seconds=0),
            end = pd.Timedelta(seconds=total_duration_seconds),
            freq = '1s'
        )
    ).interpolate()

    sound_features_resampled = sound_features.reindex(
        index = pd.timedelta_range(
            start = pd.Timedelta(seconds=0),
            end = pd.Timedelta(seconds=total_duration_seconds),
            freq = '1s'
        )
    ).interpolate()

    zoom_features_resampled = zoom_features.reindex(
        index = pd.timedelta_range(
            start = pd.Timedelta(seconds=0),
            end = pd.Timedelta(seconds=total_duration_seconds),
            freq = '1s'
        )
    ).interpolate()

    aggregated = retention.join(speaker_features_resampled, how='left', rsuffix='_speaker')
    aggregated = aggregated.join(sound_features_resampled, how='left', rsuffix='_sound')
    aggregated = aggregated.join(zoom_features_resampled, how='left', rsuffix='_zoom')
    
    aggregated.to_csv('data.csv', index=True)    
