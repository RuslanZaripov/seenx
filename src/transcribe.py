import math
import pandas as pd
from seenx_utils import get_video_duration
from logger import Logger
from config import Config
import whisper

logger = Logger(show=True).get_logger()


def count_words(text):
    return len(text.split())


def collect_wps(
    video_path: str, config: Config, existing_features: list = []
) -> pd.DataFrame:
    feature_name = "wps"
    if feature_name in existing_features:
        logger.info(f"WPS feature already exists, skipping extraction")
        return pd.DataFrame()

    model = whisper.load_model(config.get("whisper_model_size"))
    result = model.transcribe(video_path)

    wps_frame = []
    for seg in result["segments"]:
        duration = seg["end"] - seg["start"]
        word_count = count_words(seg["text"])
        wps = word_count / duration if duration > 0 else 0

        start = math.ceil(seg["start"])
        end = math.ceil(seg["end"])

        for second in range(start, end):
            wps_frame.append({"time": second, "wps": wps})

    # if wps_frame is empty, add rows for each second of the video with wps = 0
    if not wps_frame:
        video_duration = math.ceil(get_video_duration(video_path))
        for second in range(0, video_duration):
            wps_frame.append({"time": second, "wps": 0})

    wps_frame = pd.DataFrame(wps_frame)
    # wps_frame["time"] = pd.to_timedelta(wps_frame["time"], unit="s")
    # wps_frame = wps_frame.set_index("time")
    wps_frame = wps_frame.drop(columns=["time"])

    return wps_frame


if __name__ == "__main__":
    config = Config("configs/local.json")
    video_path = "static/test.mp4"
    wps_df = collect_wps(video_path=video_path, config=config)
    print(wps_df)
