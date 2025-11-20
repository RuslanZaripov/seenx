from sound_features import sound_features_pipeline
from logger import Logger
from config import Config

logger = Logger(show=True).get_logger()


def get_vocal_music_features(config: Config):
    vocal_features = sound_features_pipeline(
        config.get("source_separation_dir") + "/vocals.mp3", fps=1, prefix="vocal_"
    )

    music_features = sound_features_pipeline(
        config.get("source_separation_dir") + "/mixed.mp3", fps=1, prefix="music_"
    )

    return music_features, vocal_features
