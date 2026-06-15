from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from func.geometry import iou_matrix
from func.parsers import ObjectLabel, Prediction


def parse_iou_thresholds(text: str) -> torch.Tensor:
    if ":" in text:
        start, step, stop = (float(x) for x in text.split(":"))
        count = int(round((stop - start) / step)) + 1
        return torch.tensor([start + i * step for i in range(count)], dtype=torch.float32)
    return torch.tensor([float(x) for x in text.split(",") if x.strip()], dtype=torch.float32)


def match_predictions(
    gt: list[ObjectLabel], preds: list[Prediction], thresholds: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    correct = torch.zeros((len(preds), len(thresholds)), dtype=torch.bool)
    conf = torch.tensor([p.conf for p in preds], dtype=torch.float32)
    pred_cls = torch.tensor([p.cls for p in preds], dtype=torch.long)
    target_cls = torch.tensor([g.cls for g in gt], dtype=torch.long)

    gt_by_image: dict[str, list[ObjectLabel]] = {}
    pred_by_image: dict[str, list[tuple[int, Prediction]]] = {}
    for item in gt:
        gt_by_image.setdefault(item.image_id, []).append(item)
    for idx, item in enumerate(preds):
        pred_by_image.setdefault(item.image_id, []).append((idx, item))

    for image_id, pred_items in pred_by_image.items():
        gt_items = gt_by_image.get(image_id, [])
        if not gt_items:
            continue
        ious = iou_matrix([g.points for g in gt_items], [p.points for _, p in pred_items], mode)
        gt_classes = torch.tensor([g.cls for g in gt_items], dtype=torch.long)
        pred_classes = torch.tensor([p.cls for _, p in pred_items], dtype=torch.long)
        class_ok = gt_classes[:, None] == pred_classes[None, :]
        for t_idx, threshold in enumerate(thresholds):
            pairs = torch.where((ious >= threshold) & class_ok)
            if not pairs[0].numel():
                continue
            matches = torch.cat((torch.stack(pairs, dim=1), ious[pairs[0], pairs[1]][:, None]), dim=1).numpy()
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            for _, pred_local, _ in matches:
                correct[pred_items[int(pred_local)][0], t_idx] = True
    return correct, conf, pred_cls, target_cls


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    if trapz_fn is None:
        raise AttributeError("Neither np.trapezoid nor np.trapz is found in numpy.")
    return float(trapz_fn(np.interp(x, mrec, mpre), x))


def ap_per_class(
    correct: torch.Tensor, conf: torch.Tensor, pred_cls: torch.Tensor, target_cls: torch.Tensor, num_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = torch.argsort(-conf)
    correct = correct[order].numpy()
    pred_cls_np = pred_cls[order].numpy()
    target_cls_np = target_cls.numpy()
    precision = np.zeros(num_classes, dtype=np.float32)
    recall = np.zeros(num_classes, dtype=np.float32)
    ap = np.zeros((num_classes, correct.shape[1]), dtype=np.float32)
    target_counts = np.array([(target_cls_np == c).sum() for c in range(num_classes)], dtype=np.int32)

    for cls_id in range(num_classes):
        pred_mask = pred_cls_np == cls_id
        n_gt = int(target_counts[cls_id])
        n_pred = int(pred_mask.sum())
        if n_gt == 0 or n_pred == 0:
            continue
        cls_correct = correct[pred_mask]
        fp = (~cls_correct).cumsum(axis=0)
        tp = cls_correct.cumsum(axis=0)
        rec_curve = tp / (n_gt + 1e-16)
        prec_curve = tp / (tp + fp + 1e-16)
        precision[cls_id] = float(prec_curve[-1, 0]) if len(prec_curve) else 0.0
        recall[cls_id] = float(rec_curve[-1, 0]) if len(rec_curve) else 0.0
        for t in range(correct.shape[1]):
            ap[cls_id, t] = compute_ap(rec_curve[:, t], prec_curve[:, t])
    return precision, recall, ap, target_counts


def summarize(
    correct: torch.Tensor,
    conf: torch.Tensor,
    pred_cls: torch.Tensor,
    target_cls: torch.Tensor,
    names: list[str],
) -> tuple[dict, list[dict]]:
    precision, recall, ap, target_counts = ap_per_class(correct, conf, pred_cls, target_cls, len(names))
    pred_cls_np = pred_cls.numpy()
    valid = target_counts > 0
    per_class = []
    for cls_id, name in enumerate(names):
        per_class.append(
            {
                "class_id": cls_id,
                "class_name": name,
                "precision": float(precision[cls_id]),
                "recall": float(recall[cls_id]),
                "AP50": float(ap[cls_id, 0]),
                "AP50-95": float(ap[cls_id].mean()),
                "predictions": int((pred_cls_np == cls_id).sum()),
                "targets": int(target_counts[cls_id]),
            }
        )
    summary = {
        "precision": float(precision[valid].mean()) if valid.any() else 0.0,
        "recall": float(recall[valid].mean()) if valid.any() else 0.0,
        "AP50": float(ap[valid, 0].mean()) if valid.any() else 0.0,
        "AP50-95": float(ap[valid].mean()) if valid.any() else 0.0,
        "predictions": int(len(pred_cls_np)),
        "targets": int(len(target_cls)),
    }
    return summary, per_class


def write_outputs(output: Path, payload: dict, per_class: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["class_id", "class_name", "precision", "recall", "AP50", "AP50-95", "predictions", "targets"]
        )
        writer.writeheader()
        writer.writerows(per_class)

    s = payload["summary"]
    c = payload["consistency"]
    lines = [
        "Evaluator  : torch_eval",
        f"IoU mode   : {payload['iou_mode']}",
        f"GT images  : {c['gt_images']}",
        f"Pred images: {c['pred_images']}",
        f"Matched img: {c['matched_images']}",
        f"Missing pred images : {c['missing_prediction_images']}",
        f"Prediction-only imgs: {c['prediction_only_images']}",
        f"Predictions: {s['predictions']}",
        f"Targets    : {s['targets']}",
        f"Precision  : {s['precision']:.4f}",
        f"Recall     : {s['recall']:.4f}",
        f"AP50       : {s['AP50']:.4f}",
        f"AP50-95    : {s['AP50-95']:.4f}",
        "",
        "Per class:",
    ]
    for item in per_class:
        lines.append(
            f"{item['class_name']:>12s} P {item['precision']:.4f} R {item['recall']:.4f} "
            f"AP50 {item['AP50']:.4f} AP50-95 {item['AP50-95']:.4f} "
            f"pred {item['predictions']} gt {item['targets']}"
        )
    text = "\n".join(lines) + "\n"
    (output / "metrics.txt").write_text(text, encoding="utf-8")
    print(text)
