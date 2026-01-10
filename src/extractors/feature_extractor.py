class VideoFeaturePass:
    def required_keys(self) -> set[str]:
        return set()

    def produces_keys(self) -> set[str]:
        return set()

    def run(self, video_path: str, context: dict): ...
