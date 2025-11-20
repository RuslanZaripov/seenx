import math
import pandas as pd
from config import Config
import whisper


def count_words(text):
    return len(text.split())


def collect_wps(video_path: str, config: Config) -> pd.DataFrame:
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

    wps_frame = pd.DataFrame(wps_frame)
    wps_frame["time"] = pd.to_timedelta(wps_frame["time"], unit="s")
    wps_frame = wps_frame.set_index("time")

    return wps_frame


if __name__ == "__main__":
    config = Config("configs/local.json")
    video_path = "static/test.mp4"
    # print(config.get("whisper_model_size"))
    wps_df = collect_wps(video_path=video_path, config=config)
    print(wps_df)
