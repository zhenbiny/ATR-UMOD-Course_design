from __future__ import annotations

import argparse
import csv
from pathlib import Path

from model_factory import ONLY_VISIBLE_DIR, build_model, load_model


PROJECT_DIR = ONLY_VISIBLE_DIR.parent
CONFIG_PATH = ONLY_VISIBLE_DIR / "config" / "visible_90_10.yaml"
WORKSPACE_DIR = ONLY_VISIBLE_DIR / "workspace"
DEFAULT_MODEL_YAML = ONLY_VISIBLE_DIR / "models" / "yolov8s-obb-p2-cbam.yaml"
DEFAULT_PRETRAINED = PROJECT_DIR / "yolov8s-obb.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one RGB-only YOLOv8s-OBB-P2-CBAM model on a fixed 90/10 split."
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--model-yaml", type=Path, default=DEFAULT_MODEL_YAML)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--name", default="visible_p2_cbam_90_10")
    return parser.parse_args()


def completed_epochs(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CONFIG_PATH}. Run prepare_visible.py before training."
        )

    run_dir = WORKSPACE_DIR / args.name
    last_checkpoint = run_dir / "weights" / "last.pt"
    trained_epochs = completed_epochs(run_dir / "results.csv")
    print(f"RGB-only single model: {trained_epochs}/{args.epochs} epochs logged")
    if trained_epochs >= args.epochs:
        print(f"Skipping completed run: {run_dir}")
        return

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    if last_checkpoint.exists() and trained_epochs > 0:
        print(f"Resuming interrupted run from {last_checkpoint}")
        model = load_model(last_checkpoint)
        checkpoint_pretrained = (model.ckpt or {}).get("train_args", {}).get("pretrained")
        if isinstance(checkpoint_pretrained, (str, Path)):
            raise RuntimeError(
                "This legacy checkpoint stores a pretrained weight path and cannot be "
                "resumed safely. Its best.pt remains valid. Start a new named run if "
                "additional fine-tuning is needed."
            )
        model.train(resume=str(last_checkpoint))
        return

    print(f"Building custom model from {args.model_yaml}")
    print(f"Loading compatible pretrained weights from {args.pretrained}")
    model = build_model(args.model_yaml, args.pretrained)
    model.train(
        data=str(CONFIG_PATH),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=4,
        project=str(WORKSPACE_DIR),
        name=args.name,
        exist_ok=True,
        task="obb",
        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=8,
        close_mosaic=5,
        mosaic=0.7,
        mixup=0.05,
        degrees=10.0,
        translate=0.1,
        scale=0.4,
        flipud=0.2,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.55,
        hsv_v=0.35,
        amp=True,
        seed=42,
        deterministic=True,
        plots=True,
        save=True,
        val=True,
        # Compatible pretrained weights were already loaded by build_model().
        # Persisting the original path here can overwrite custom-model weights on resume.
        pretrained=False,
    )


if __name__ == "__main__":
    main()
