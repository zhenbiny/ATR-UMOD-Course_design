from __future__ import annotations

from pathlib import Path

import torch

from model_factory import ONLY_VISIBLE_DIR, build_model


PROJECT_DIR = ONLY_VISIBLE_DIR.parent
MODEL_YAML = ONLY_VISIBLE_DIR / "models" / "yolov8s-obb-p2-cbam.yaml"
PRETRAINED = PROJECT_DIR / "yolov8s-obb.pt"


def main() -> None:
    model = build_model(MODEL_YAML, PRETRAINED)
    network = model.model
    cbam_count = sum(1 for module in network.modules() if module.__class__.__name__ == "CBAM")
    detect_head = network.model[-1]
    print(f"model_yaml={MODEL_YAML}")
    print(f"parameters={sum(parameter.numel() for parameter in network.parameters())}")
    print(f"cbam_blocks={cbam_count}")
    print(f"obb_feature_levels={len(detect_head.stride)}")
    print(f"obb_strides={[int(value) for value in detect_head.stride]}")

    with torch.inference_mode():
        outputs = network(torch.zeros(1, 3, 640, 640))
    print(f"forward_output_type={type(outputs).__name__}")
    print("model_validation=OK")


if __name__ == "__main__":
    main()

