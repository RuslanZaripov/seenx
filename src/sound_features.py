import os
import shutil
import librosa
import pandas as pd
from logger import Logger
from config import Config
from source_separation import mp4_to_wav, separate, combine

logger = Logger(show=True).get_logger()


def get_vocal_music_features(
    audio_path: str, config: Config, existing_features: list = []
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wav_file_path = "output.wav"

    logger.info(f"Converting mp4 {audio_path} to wav {wav_file_path}")
    mp4_to_wav(audio_path, wav_file_path)

    outp = config.get("source_separation_dir")
    logger.info(f"Separating {wav_file_path} into music and vocals to {outp}")
    separate([wav_file_path], outp=outp)

    filename, _ = os.path.splitext(os.path.basename(wav_file_path))
    sepearated_folder = f"{outp}/htdemucs/{filename}"
    music_path, vocal_path = combine(sepearated_folder)

    vocal_features = sound_features_pipeline(
        vocal_path, fps=1, prefix="vocal_", existing_features=existing_features
    )
    music_features = sound_features_pipeline(
        music_path, fps=1, prefix="music_", existing_features=existing_features
    )

    os.remove(wav_file_path)
    shutil.rmtree(sepearated_folder)

    return music_features, vocal_features


def sound_features_pipeline(
    audio_file_path: str, fps: int = 1, prefix: str = "", existing_features: list = []
) -> pd.DataFrame:
    feature_names = [
        f"{prefix}rms",
        f"{prefix}zcr",
        f"{prefix}centroid",
        f"{prefix}rolloff",
    ]
    if all(feature in existing_features for feature in feature_names):
        logger.info(f"Sound features for {prefix} already exist, skipping extraction")
        return pd.DataFrame()

    y, sr = librosa.load(audio_file_path, sr=None)
    logger.info(f"Audio file: {audio_file_path} shape: {y.shape}, sample rate: {sr}")

    frame_length = sr // fps
    hop_length = sr // fps
    logger.info(f"Extracting features with {frame_length=}, {hop_length=}")

    # Root Mean Square (RMS) Energy: A measure of the signal’s loudness over time.
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

    # Zero Crossing Rate (ZCR): A measure of the frequency of sign changes in the signal.
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=frame_length, hop_length=hop_length
    )

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)

    features = pd.DataFrame(
        {
            f"{prefix}rms": rms.flatten(),
            f"{prefix}zcr": zcr.flatten(),
            f"{prefix}centroid": centroid.flatten(),
            f"{prefix}rolloff": rolloff.flatten(),
        }
    )

    return features
