from __future__ import annotations

import cv2
import numpy as np
import torch


def points_to_xyxy(points: list[np.ndarray]) -> torch.Tensor:
    if not points:
        return torch.zeros((0, 4), dtype=torch.float32)
    pts = torch.as_tensor(np.stack(points), dtype=torch.float32)
    x = pts[..., 0]
    y = pts[..., 1]
    return torch.stack([x.min(1).values, y.min(1).values, x.max(1).values, y.max(1).values], dim=1)


def box_iou_matrix(gt_points: list[np.ndarray], pred_points: list[np.ndarray]) -> torch.Tensor:
    if not gt_points or not pred_points:
        return torch.zeros((len(gt_points), len(pred_points)), dtype=torch.float32)
    a = points_to_xyxy(gt_points)
    b = points_to_xyxy(pred_points)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
    return inter / (area_a + area_b - inter + 1e-7)


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def min_area_rect(points: np.ndarray):
    rect = cv2.minAreaRect(points.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    return (float(cx), float(cy)), (max(float(w), 1e-6), max(float(h), 1e-6)), float(angle)


def obb_iou(a: np.ndarray, b: np.ndarray) -> float:
    rect_a = min_area_rect(a)
    rect_b = min_area_rect(b)
    area_a = rect_a[1][0] * rect_a[1][1]
    area_b = rect_b[1][0] * rect_b[1][1]
    status, inter_pts = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    if status == cv2.INTERSECT_NONE or inter_pts is None:
        inter = 0.0
    elif status == cv2.INTERSECT_FULL:
        inter = min(area_a, area_b)
    else:
        inter = polygon_area(inter_pts.reshape(-1, 2))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def obb_iou_matrix(gt_points: list[np.ndarray], pred_points: list[np.ndarray]) -> torch.Tensor:
    out = torch.zeros((len(gt_points), len(pred_points)), dtype=torch.float32)
    for i, gt in enumerate(gt_points):
        for j, pred in enumerate(pred_points):
            out[i, j] = obb_iou(gt, pred)
    return out


def iou_matrix(gt_points: list[np.ndarray], pred_points: list[np.ndarray], mode: str) -> torch.Tensor:
    return box_iou_matrix(gt_points, pred_points) if mode == "box" else obb_iou_matrix(gt_points, pred_points)
