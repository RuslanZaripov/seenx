import argparse
import os
import catboost as cb
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
        features = aggregate(
            video_path=video_path,
            audio_path=video_path,
            output_path=None,
            config=self.config,
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
        plt.plot(features["frame"], predictions, marker="o")
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
