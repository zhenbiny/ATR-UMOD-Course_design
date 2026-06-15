from __future__ import annotations

import argparse
from pathlib import Path

from model_factory import ONLY_VISIBLE_DIR, load_model


CONFIG_PATH = ONLY_VISIBLE_DIR / "config" / "visible_90_10.yaml"
WORKSPACE_DIR = ONLY_VISIBLE_DIR / "workspace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RGB-only OBB checkpoint.")
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to best.pt or last.pt.",
    )
    parser.add_argument("--imgsz", type=int, default=832)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)

    output_dir = WORKSPACE_DIR / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.weights)
    metrics = model.val(
        data=str(CONFIG_PATH),
        imgsz=args.imgsz,
        device=0,
        workers=0,
        project=str(output_dir),
        name=args.weights.parent.parent.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    print(f"mAP50={metrics.box.map50:.6f}")
    print(f"mAP50-95={metrics.box.map:.6f}")
    print(f"precision={metrics.box.mp:.6f}")
    print(f"recall={metrics.box.mr:.6f}")


if __name__ == "__main__":
    main()
