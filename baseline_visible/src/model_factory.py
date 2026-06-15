from __future__ import annotations

import os
from pathlib import Path


ONLY_VISIBLE_DIR = Path(__file__).resolve().parents[1]
ULTRALYTICS_CONFIG_DIR = ONLY_VISIBLE_DIR / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_CONFIG_DIR)

from ultralytics import YOLO
from ultralytics.nn import tasks
from ultralytics.nn.modules import CBAM


def register_custom_modules() -> None:
    """Expose supported Ultralytics modules to its YAML parser."""
    tasks.CBAM = CBAM


def build_model(model_yaml: Path, pretrained_weights: Path | None = None) -> YOLO:
    register_custom_modules()
    model = YOLO(str(model_yaml), task="obb")
    if pretrained_weights is not None:
        if not pretrained_weights.exists():
            raise FileNotFoundError(pretrained_weights)
        model.load(str(pretrained_weights))
    return model


def load_model(weights: Path) -> YOLO:
    register_custom_modules()
    if not weights.exists():
        raise FileNotFoundError(weights)
    return YOLO(str(weights), task="obb")

