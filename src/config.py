import os
import json
from typing import Any, Dict


class Config:
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")

        with open(config_path, "r") as f:
            self.config_data: Dict[str, Any] = json.load(f)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config_data.get(key, default)


if __name__ == "__main__":
    print(f"pwd='{os.getcwd()}'")
    config = Config("configs/init.json")
