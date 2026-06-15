from __future__ import annotations

import argparse
from pathlib import Path

from func.metrics import match_predictions, parse_iou_thresholds, summarize, write_outputs
from func.parsers import (
    DEFAULT_CLASSES,
    image_set_from_gt,
    image_set_from_preds,
    load_ground_truth,
    load_predictions,
    parse_classes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch-based standalone evaluator for XML GT and predict_std.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--iou-mode", choices=["obb", "box"], default="obb")
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--conf-thres", type=float, default=None)
    parser.add_argument("--iou-thres-list", default="0.50:0.05:0.95")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path("./gt")              #真值所在路径，内容为验证集的真值xml文件
    pred_path = Path("./pred")          #预测所在路径，内容为验证集的预测结果txt文件
    names, class_to_id = parse_classes(args.classes)
    gt = load_ground_truth(gt_path, class_to_id)
    preds = load_predictions(pred_path, class_to_id, conf_thres=args.conf_thres)
    thresholds = parse_iou_thresholds(args.iou_thres_list)
    correct, conf, pred_cls, target_cls = match_predictions(gt, preds, thresholds, args.iou_mode)
    summary, per_class = summarize(correct, conf, pred_cls, target_cls, names)
    gt_images = image_set_from_gt(gt)
    pred_images = image_set_from_preds(preds)
    payload = {
        "evaluator": "torch_eval",
        "iou_mode": args.iou_mode,
        "iou_thresholds": [float(x) for x in thresholds],
        "summary": summary,
        "per_class": per_class,
        "consistency": {
            "gt_images": len(gt_images),
            "pred_images": len(pred_images),
            "matched_images": len(gt_images & pred_images),
            "missing_prediction_images": len(gt_images - pred_images),
            "prediction_only_images": len(pred_images - gt_images),
        },
    }
    write_outputs(args.output, payload, per_class)


if __name__ == "__main__":
    main()
