import librosa
import numpy as np
import pandas as pd
from IPython.display import Audio
import matplotlib.pyplot as plt
from tqdm import tqdm
from logger import Logger

logger = Logger(show=True).get_logger()


def sound_features_pipeline(audio_file_path: str, fps: int = 1) -> pd.DataFrame:
    y, sr = librosa.load(audio_file_path, sr=None)
    print(f"Audio shape: {y.shape}, Sample rate: {sr}")

    frame_length = sr // fps
    hop_length = sr // fps

    # Root Mean Square (RMS) Energy: A measure of the signal’s loudness over time.
    rms = librosa.feature.rms(
        y=y, 
        frame_length=frame_length,
        hop_length=hop_length
    )

    # Zero Crossing Rate (ZCR): A measure of the frequency of sign changes in the signal.
    zcr = librosa.feature.zero_crossing_rate(
        y=y, 
        frame_length=frame_length, 
        hop_length=hop_length
    )

    centroid = librosa.feature.spectral_centroid(
        y=y, 
        sr=sr, 
        hop_length=hop_length
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y, 
        sr=sr, 
        hop_length=hop_length
    )

    # Combine features into a DataFrame
    features = pd.DataFrame({
        'frame_index': np.arange(rms.shape[1]),
        'rms': rms.flatten(),
        'zcr': zcr.flatten(),
        'centroid': centroid.flatten(),
        'rolloff': rolloff.flatten()
    })

    return features
