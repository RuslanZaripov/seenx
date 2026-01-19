import argparse
import os
import catboost as cb
import pandas as pd
import matplotlib.pyplot as plt
from ..config import Config
from ..logger import Logger
from ..aggregator import aggregate

logger = Logger(show=True).get_logger()


class Predictor:
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config = Config(config_path=config_path)
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} does not exist.")

        model = cb.CatBoostClassifier()
        model.load_model(self.model_path)
        return model

    def predict(self, video_path):
        # features = aggregate(
        #     video_path=video_path,
        #     audio_path=video_path,
        #     output_path="",
        #     config=self.config,
        # )
        features = pd.read_csv("/kaggle/working/faceless_youtube_channel_ideas.csv")
        features["time"] = pd.to_timedelta(features["time"]).dt.total_seconds()
        logger.info(f"Extracted features shape: {features.shape}")
        # logger.log("Model feature count:", self.model.feature_count_)
        # logger.log("Input feature count:", features.drop(columns=["frame"]).shape[1])
        logger.log("Model feature names:", self.model.feature_names_)
        logger.log(
            "Input feature names:", features.drop(columns=["frame"]).columns.tolist()
        )
        predictions = self.model.predict(features.drop(columns=["frame"]))
        # plot retention figure
        self.draw_retention_plot(
            features, predictions, output_path="retention_plot.png"
        )
        return predictions

    def draw_retention_plot(
        self, features, predictions, output_path="retention_plot.png"
    ):
        plt.figure(figsize=(10, 6))
        # convert features["frame"] to int
        frames = features["time"].astype(int)
        plt.plot(frames, predictions, marker="o")
        plt.title("Predicted Retention over Frames")
        plt.xlabel("Frame")
        plt.ylabel("Predicted Retention")
        plt.grid()
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()

    predictor = Predictor(model_path=args.model_path, config_path=args.config_path)
    predictor.predict(args.video_path)
    logger.info("Prediction completed.")
