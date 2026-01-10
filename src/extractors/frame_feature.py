import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from feature_extractor import VideoFeaturePass
from ..config import Config
from ..logger import Logger

logger = Logger(show=True).get_logger()


class FrameFeaturePass(VideoFeaturePass):
    def __init__(self, config: Config):
        self.config = config

    def required_keys(self):
        return set()

    def produces_keys(self):
        return {"brightness", "sharpness"}

    def run(self, video_path, context):
        frame_features = {"brightness": [], "sharpness": []}
        feature_names = list(frame_features.keys())

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            exit()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar = tqdm(total=total_frames, desc="Extracting Frame Quality Features")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            frame_features["brightness"].append(brightness)

            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = float(laplacian.var())
            frame_features["sharpness"].append(sharpness)

            pbar.update(1)

        for feature_name in feature_names:
            context["data"][feature_name] = pd.Series(
                frame_features[feature_name],
                index=context["data"].index,
                dtype="float64",
            )

        pbar.close()
        cap.release()
