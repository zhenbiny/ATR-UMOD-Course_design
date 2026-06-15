import os
import cv2
import numpy as np
import sys
import pickle
import itertools
from tqdm import tqdm
from pathlib import Path

# Add torch_eval to path to import func
MOCK_DIR = r"c:\Users\17638\Desktop\NUDT\智能图像处理\mock_evaluation_closed_loop"
sys.path.append(os.path.join(MOCK_DIR, "torch_eval"))

from func.geometry import obb_iou
from func.metrics import match_predictions, parse_iou_thresholds, summarize
from func.parsers import parse_classes, load_ground_truth, Prediction

CLASSES = [
    'car', 'suv', 'van', 'bus', 'freight_car', 
    'truck', 'motorcycle', 'trailer', 'excavator', 
    'crane', 'tank_truck'
]
NAMES, CLASS_TO_ID = parse_classes(",".join(CLASSES))

# Caching file path
CACHE_PATH = os.path.join(MOCK_DIR, "raw_preds_cache.pkl")

# ==============================================================================
# Helper Functions for alignment and IoU
# ==============================================================================
def align_vertices(vertices, reference_box):
    """
    通过寻找顶点排列的最小二乘距离，将 vertices 4个角点与 reference_box 对齐。
    彻底解决旋转矩形加权平均时的角点顺序模糊性问题。
    """
    best_perm = None
    min_dist = float('inf')
    ref_pts = reference_box.reshape(4, 2)
    cand_pts = vertices.reshape(4, 2)
    
    # 4个顶点的全排列共 24 种情况
    for perm in itertools.permutations(range(4)):
        dist = np.sum((ref_pts - cand_pts[list(perm)]) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_perm = perm
            
    return cand_pts[list(best_perm)].flatten()

# ==============================================================================
# Fusion Strategies
# ==============================================================================

def run_aabb_nms(all_preds, nms_threshold=0.5, conf_cutoff=0.20):
    """策略 A: Baseline AABB NMS"""
    final_preds = []
    classes = np.array([p["class_idx"] for p in all_preds])
    scores = np.array([p["conf"] for p in all_preds])
    coords = np.array([p["coords"] for p in all_preds])
    
    for class_idx in range(len(CLASSES)):
        mask = (classes == class_idx)
        if not np.any(mask):
            continue
            
        cls_scores = scores[mask]
        cls_coords = coords[mask]
        
        # Calculate AABB
        x_coords = cls_coords[:, ::2]
        y_coords = cls_coords[:, 1::2]
        x_min, y_min = np.min(x_coords, axis=1), np.min(y_coords, axis=1)
        x_max, y_max = np.max(x_coords, axis=1), np.max(y_coords, axis=1)
        aabb_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
        
        # Standard AABB NMS
        if len(aabb_boxes) == 0:
            continue
            
        # NMS loop
        x1, y1, x2, y2 = aabb_boxes[:, 0], aabb_boxes[:, 1], aabb_boxes[:, 2], aabb_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = cls_scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            ovr = inter / np.maximum(union, 1e-6)
            
            inds = np.where(ovr <= nms_threshold)[0]
            order = order[inds + 1]
            
        for idx in keep:
            if cls_scores[idx] >= conf_cutoff:
                final_preds.append({
                    "class_idx": class_idx,
                    "conf": float(cls_scores[idx]),
                    "coords": cls_coords[idx]
                })
    return final_preds

def run_obb_nms(all_preds, nms_threshold=0.5, conf_cutoff=0.20):
    """策略 B: 严格 OBB NMS"""
    final_preds = []
    classes = np.array([p["class_idx"] for p in all_preds])
    scores = np.array([p["conf"] for p in all_preds])
    coords = np.array([p["coords"] for p in all_preds])
    
    for class_idx in range(len(CLASSES)):
        mask = (classes == class_idx)
        if not np.any(mask):
            continue
            
        cls_scores = scores[mask]
        cls_coords = coords[mask]
        order = cls_scores.argsort()[::-1]
        
        pts = [c.reshape(4, 2) for c in cls_coords]
        keep = []
        visited = np.zeros(len(cls_scores), dtype=bool)
        
        for i in range(len(cls_scores)):
            idx_i = order[i]
            if visited[idx_i]:
                continue
            keep.append(idx_i)
            visited[idx_i] = True
            
            for j in range(i + 1, len(cls_scores)):
                idx_j = order[j]
                if visited[idx_j]:
                    continue
                iou = obb_iou(pts[idx_i], pts[idx_j])
                if iou > nms_threshold:
                    visited[idx_j] = True
                    
        for idx in keep:
            if cls_scores[idx] >= conf_cutoff:
                final_preds.append({
                    "class_idx": class_idx,
                    "conf": float(cls_scores[idx]),
                    "coords": cls_coords[idx]
                })
    return final_preds

def run_obb_wbf_align(all_preds, iou_threshold=0.5, conf_cutoff=0.20, use_boosting=False):
    """
    策略 C: 基于顶点对齐的 OBB 加权框融合 (OBB-WBF-Align)
    如果使用 use_boosting=True，则启用模型投票置信度奖励机制
    """
    final_preds = []
    classes = np.array([p["class_idx"] for p in all_preds])
    scores = np.array([p["conf"] for p in all_preds])
    coords = np.array([p["coords"] for p in all_preds])
    
    for class_idx in range(len(CLASSES)):
        mask = (classes == class_idx)
        if not np.any(mask):
            continue
            
        cls_scores = scores[mask]
        cls_coords = coords[mask]
        
        # Sort by score descending
        order = cls_scores.argsort()[::-1]
        cls_scores = cls_scores[order]
        cls_coords = cls_coords[order]
        
        pts = [c.reshape(4, 2) for c in cls_coords]
        visited = np.zeros(len(cls_scores), dtype=bool)
        
        for i in range(len(cls_scores)):
            if visited[i]:
                continue
                
            overlap_indices = [i]
            for j in range(i + 1, len(cls_scores)):
                if visited[j]:
                    continue
                iou = obb_iou(pts[i], pts[j])
                if iou >= iou_threshold:
                    overlap_indices.append(j)
                    
            visited[overlap_indices] = True
            
            box_coords = cls_coords[overlap_indices]
            box_scores = cls_scores[overlap_indices]
            
            # Align vertices of all overlapping boxes to the reference (highest conf) box
            ref_box = box_coords[0]
            aligned_coords = [ref_box]
            for j in range(1, len(box_coords)):
                aligned_coords.append(align_vertices(box_coords[j], ref_box))
                
            aligned_coords = np.array(aligned_coords)
            
            # Weighted average coordinates
            sum_scores = np.sum(box_scores)
            if sum_scores <= 0:
                weighted_coords = ref_box
            else:
                weighted_coords = np.sum(aligned_coords * box_scores[:, None], axis=0) / sum_scores
            
            # Confidence calculation
            max_conf = box_scores[0]
            if use_boosting:
                # 投票机制：被越多模型预测到，置信度适当获得奖励（每多一个模型 +2% 置信度，最高+10%）
                boost = min(0.10, 0.02 * (len(overlap_indices) - 1))
                fused_conf = min(1.0, max_conf * (1.0 + boost))
            else:
                fused_conf = max_conf
                
            if fused_conf >= conf_cutoff:
                final_preds.append({
                    "class_idx": class_idx,
                    "conf": float(fused_conf),
                    "coords": weighted_coords
                })
    return final_preds

def run_aabb_nms_with_boosting(all_preds, nms_threshold=0.5, conf_cutoff=0.20, boost_factor=0.02, penalty_factor=0.90):
    """
    策略 D: AABB NMS + 多模型共识置信度调整 (NMS-Boosting)
    对于 NMS 保留下来的框，统计其在原始预测中重叠框的个数（即有多少个模型同时预测到此目标）。
    根据模型个数，对最终置信度进行奖励（consensus）或惩罚（lone detection）。
    """
    final_preds = []
    classes = np.array([p["class_idx"] for p in all_preds])
    scores = np.array([p["conf"] for p in all_preds])
    coords = np.array([p["coords"] for p in all_preds])
    
    for class_idx in range(len(CLASSES)):
        mask = (classes == class_idx)
        if not np.any(mask):
            continue
            
        cls_scores = scores[mask]
        cls_coords = coords[mask]
        
        # Calculate AABB
        x_coords = cls_coords[:, ::2]
        y_coords = cls_coords[:, 1::2]
        x_min, y_min = np.min(x_coords, axis=1), np.min(y_coords, axis=1)
        x_max, y_max = np.max(x_coords, axis=1), np.max(y_coords, axis=1)
        aabb_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
        
        if len(aabb_boxes) == 0:
            continue
            
        # NMS loop with overlap count tracking
        x1, y1, x2, y2 = aabb_boxes[:, 0], aabb_boxes[:, 1], aabb_boxes[:, 2], aabb_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = cls_scores.argsort()[::-1]
        keep = []
        counts = [] # Records how many model predictions overlapped with this kept box
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            ovr = inter / np.maximum(union, 1e-6)
            
            # Count how many boxes overlap with this one (including itself)
            overlapping_boxes_count = 1 + int(np.sum(ovr > nms_threshold))
            counts.append(overlapping_boxes_count)
            
            inds = np.where(ovr <= nms_threshold)[0]
            order = order[inds + 1]
            
        for idx, overlap_cnt in zip(keep, counts):
            conf = cls_scores[idx]
            # Adjust confidence based on consensus
            if overlap_cnt >= 2:
                # 获得共识奖励
                boost = min(0.12, boost_factor * (overlap_cnt - 1))
                adjusted_conf = min(1.0, conf * (1.0 + boost))
            else:
                # 孤立预测，进行惩罚
                adjusted_conf = conf * penalty_factor
                
            if adjusted_conf >= conf_cutoff:
                final_preds.append({
                    "class_idx": class_idx,
                    "conf": float(adjusted_conf),
                    "coords": cls_coords[idx]
                })
    return final_preds


def run_category_specific_nms(all_preds, conf_cutoff=0.20):
    """
    策略 E: 类别特异性 NMS 阈值 (Category-Specific NMS)
    对于密集排列的小车辆类别（car, suv），采用较宽松的 NMS（0.55/0.6）防止邻车吞噬。
    对于孤立的大型目标（bus, crane, trailer），采用较严格的 NMS（0.35/0.4）压榨虚警。
    """
    # 类别特定超参配置
    cat_thresholds = {
        'car': 0.55,
        'suv': 0.55,
        'van': 0.50,
        'bus': 0.35,
        'freight_car': 0.45,
        'truck': 0.45,
        'motorcycle': 0.50,
        'trailer': 0.35,
        'excavator': 0.40,
        'crane': 0.35,
        'tank_truck': 0.40
    }
    
    final_preds = []
    classes = np.array([p["class_idx"] for p in all_preds])
    scores = np.array([p["conf"] for p in all_preds])
    coords = np.array([p["coords"] for p in all_preds])
    
    for class_idx in range(len(CLASSES)):
        class_name = CLASSES[class_idx]
        nms_threshold = cat_thresholds[class_name]
        
        mask = (classes == class_idx)
        if not np.any(mask):
            continue
            
        cls_scores = scores[mask]
        cls_coords = coords[mask]
        
        # Calculate AABB
        x_coords = cls_coords[:, ::2]
        y_coords = cls_coords[:, 1::2]
        x_min, y_min = np.min(x_coords, axis=1), np.min(y_coords, axis=1)
        x_max, y_max = np.max(x_coords, axis=1), np.max(y_coords, axis=1)
        aabb_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
        
        if len(aabb_boxes) == 0:
            continue
            
        # Standard NMS loop
        x1, y1, x2, y2 = aabb_boxes[:, 0], aabb_boxes[:, 1], aabb_boxes[:, 2], aabb_boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = cls_scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            ovr = inter / np.maximum(union, 1e-6)
            
            inds = np.where(ovr <= nms_threshold)[0]
            order = order[inds + 1]
            
        for idx in keep:
            if cls_scores[idx] >= conf_cutoff:
                final_preds.append({
                    "class_idx": class_idx,
                    "conf": float(cls_scores[idx]),
                    "coords": cls_coords[idx]
                })
    return final_preds

# ==============================================================================
# Evaluation Harness
# ==============================================================================

def evaluate_predictions(preds_dict, gt):
    eval_preds = []
    for item in preds_dict:
        pts = item["coords"].reshape(4, 2)
        eval_preds.append(Prediction(item["img_id"], item["class_idx"], item["conf"], pts))
        
    thresholds = parse_iou_thresholds("0.50:0.05:0.95")
    correct, conf, pred_cls, target_cls = match_predictions(gt, eval_preds, thresholds, "obb")
    summary, per_class = summarize(correct, conf, pred_cls, target_cls, NAMES)
    return summary["AP50"] * 100.0, summary["AP50-95"] * 100.0

def main():
    print("=== Testing Fusion Strategies Optimization ===")
    
    # 1. Check/Load raw predictions cache
    if os.path.exists(CACHE_PATH):
        print(f"Loading raw predictions cache from {CACHE_PATH}...")
        with open(CACHE_PATH, "rb") as f:
            raw_by_img, test_files = pickle.load(f)
        print(f"Loaded {len(raw_by_img)} images' raw predictions.")
    else:
        print("No cache found. Running model inference on 150 images to construct cache...")
        from ultralytics import YOLO
        workspace_dir = r"c:\Users\17638\Desktop\NUDT\智能图像处理\multimodal_detection\workspace"
        model_paths = []
        for run_dir in os.listdir(workspace_dir):
            if run_dir.startswith("yolov8_fold"):
                best_pt = os.path.join(workspace_dir, run_dir, "weights", "best.pt")
                if os.path.exists(best_pt):
                    model_paths.append(best_pt)
                    
        models = [YOLO(mp) for mp in model_paths]
        
        test_images_dir = os.path.join(MOCK_DIR, "mock_test_dataset", "images")
        test_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(".jpg")])[:150]
        
        raw_by_img = {}
        for filename in tqdm(test_files):
            img_id = os.path.splitext(filename)[0]
            rgb_path = os.path.join(MOCK_DIR, "mock_test_dataset", "images", filename)
            ir_path = os.path.join(MOCK_DIR, "mock_test_dataset", "images_ir", filename)
            
            rgb_img = cv2.imread(rgb_path)
            ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            if rgb_img is None or ir_img is None:
                continue
                
            fused_img = cv2.merge([ir_img, rgb_img[:, :, 1], rgb_img[:, :, 2]])
            
            img_raw_preds = []
            for model in models:
                results = model.predict(fused_img, conf=0.1, imgsz=640, device=0, verbose=False)
                for result in results:
                    if result.obb is not None:
                        obb_coords = result.obb.xyxyxyxy.cpu().numpy()
                        confs = result.obb.conf.cpu().numpy()
                        classes = result.obb.cls.cpu().numpy()
                        
                        for i in range(len(classes)):
                            img_raw_preds.append({
                                "class_idx": int(classes[i]),
                                "conf": float(confs[i]),
                                "coords": obb_coords[i].flatten()
                            })
            raw_by_img[img_id] = img_raw_preds
            
        print(f"Caching raw predictions to {CACHE_PATH}...")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump((raw_by_img, test_files), f)
            
    # 2. Load Ground Truths
    gt_all = load_ground_truth(Path(os.path.join(MOCK_DIR, "torch_eval", "gt")), CLASS_TO_ID)
    selected_img_ids = {os.path.splitext(f)[0] for f in test_files}
    gt = [g for g in gt_all if g.image_id in selected_img_ids]
    print(f"Loaded {len(gt)} ground truth targets.")
    
    # 3. Define and run experiments
    experiments = [
        # (Name, function, kwargs)
        ("A1: Baseline (AABB NMS 0.50, conf 0.20)", run_aabb_nms, {"nms_threshold": 0.50, "conf_cutoff": 0.20}),
        ("A2: Baseline (AABB NMS 0.50, conf 0.25)", run_aabb_nms, {"nms_threshold": 0.50, "conf_cutoff": 0.25}),
        
        ("B1: OBB Rotated NMS 0.50", run_obb_nms, {"nms_threshold": 0.50, "conf_cutoff": 0.20}),
        ("B2: OBB Rotated NMS 0.45", run_obb_nms, {"nms_threshold": 0.45, "conf_cutoff": 0.20}),
        
        ("C1: OBB WBF-Align 0.50", run_obb_wbf_align, {"iou_threshold": 0.50, "conf_cutoff": 0.20, "use_boosting": False}),
        ("C2: OBB WBF-Align + Vote Boost", run_obb_wbf_align, {"iou_threshold": 0.50, "conf_cutoff": 0.20, "use_boosting": True}),
        
        ("D1: AABB NMS + Vote Boost (b=0.02, p=0.90)", run_aabb_nms_with_boosting, {"nms_threshold": 0.50, "conf_cutoff": 0.20, "boost_factor": 0.02, "penalty_factor": 0.90}),
        ("D2: AABB NMS + Vote Boost (b=0.03, p=0.85)", run_aabb_nms_with_boosting, {"nms_threshold": 0.50, "conf_cutoff": 0.20, "boost_factor": 0.03, "penalty_factor": 0.85}),
        ("D3: AABB NMS + Vote Boost (b=0.04, p=0.80)", run_aabb_nms_with_boosting, {"nms_threshold": 0.50, "conf_cutoff": 0.20, "boost_factor": 0.04, "penalty_factor": 0.80}),
        
        ("E1: Category-Specific NMS", run_category_specific_nms, {"conf_cutoff": 0.20}),
        ("E2: Category-Specific NMS + Vote Boost", run_aabb_nms_with_boosting, {"nms_threshold": 0.45, "conf_cutoff": 0.20, "boost_factor": 0.03, "penalty_factor": 0.85}), # Using NMS 0.45 + boost
    ]
    
    print("\n==================== Running Experiments ====================")
    results = []
    for name, func, kwargs in experiments:
        preds = []
        for img_id, raw_preds in raw_by_img.items():
            fused = func(raw_preds, **kwargs)
            for p in fused:
                preds.append({
                    "img_id": img_id,
                    "class_idx": p["class_idx"],
                    "conf": p["conf"],
                    "coords": p["coords"]
                })
        ap50, ap50_95 = evaluate_predictions(preds, gt)
        print(f"| {name:<42s} | AP50: {ap50:.4f}% | AP50-95: {ap50_95:.4f}% |")
        results.append((name, ap50, ap50_95))
        
    print("\n==================== Leaderboard (Sorted by AP50) ====================")
    results.sort(key=lambda x: x[1], reverse=True)
    print("| Rank | Strategy Name                              | AP50       | AP50-95    |")
    print("|------|--------------------------------------------|------------|------------|")
    for i, (name, ap50, ap50_95) in enumerate(results, start=1):
        print(f"| {i:<4d} | {name:<42s} | {ap50:.4f}% | {ap50_95:.4f}% |")
    print("======================================================================")

if __name__ == "__main__":
    main()
