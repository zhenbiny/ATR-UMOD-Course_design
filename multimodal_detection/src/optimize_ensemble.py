import os
import cv2
import numpy as np
import sys
import pickle
import itertools
from tqdm import tqdm
from pathlib import Path
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
MOCK_DIR = os.path.join(PROJECT_DIR, "other", "mock_evaluation")
sys.path.append(os.path.join(MOCK_DIR, "torch_eval"))

from func.metrics import match_predictions, parse_iou_thresholds, summarize
from func.parsers import parse_classes, load_ground_truth, Prediction

CLASSES = [
    'car', 'suv', 'van', 'bus', 'freight_car', 
    'truck', 'motorcycle', 'trailer', 'excavator', 
    'crane', 'tank_truck'
]
NAMES, CLASS_TO_ID = parse_classes(",".join(CLASSES))

CACHE_PATH = os.path.join(MOCK_DIR, "raw_preds_cache_300.pkl")
TARGET_NUM_IMAGES = 300

def run_general_nms_with_boosting(all_preds, nms_threshold, conf_cutoff=0.20, boost_factor=0.03, penalty_factor=0.85, max_boost=0.12):
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
        
        # Get threshold value for this class
        if isinstance(nms_threshold, dict):
            t_val = nms_threshold[class_idx]
        else:
            t_val = nms_threshold
            
        # Calculate AABB bounding box for NMS
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
        counts = []
        
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
            
            # Count overlapping boxes
            overlapping_boxes_count = 1 + int(np.sum(ovr > t_val))
            counts.append(overlapping_boxes_count)
            
            inds = np.where(ovr <= t_val)[0]
            order = order[inds + 1]
            
        for idx, overlap_cnt in zip(keep, counts):
            conf = cls_scores[idx]
            if overlap_cnt >= 2:
                # Consensus boost
                boost = min(max_boost, boost_factor * (overlap_cnt - 1))
                adjusted_conf = min(1.0, conf * (1.0 + boost))
            else:
                # Lone prediction penalty
                adjusted_conf = conf * penalty_factor
                
            if adjusted_conf >= conf_cutoff:
                final_preds.append({
                    "class_idx": class_idx,
                    "conf": float(adjusted_conf),
                    "coords": cls_coords[idx]
                })
    return final_preds

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
    print(f"=== Starting Grid Search for Multi-Fold Ensemble Fusion on {TARGET_NUM_IMAGES} Images ===")
    
    # 1. Load or create cache for TARGET_NUM_IMAGES (300)
    if os.path.exists(CACHE_PATH):
        print(f"Loading predictions from cache: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            raw_by_img, test_files = pickle.load(f)
        if len(raw_by_img) < TARGET_NUM_IMAGES:
            print(f"Cache size ({len(raw_by_img)}) is smaller than target ({TARGET_NUM_IMAGES}). Regenerating...")
            run_inference_to_cache = True
        else:
            print(f"Successfully loaded predictions for {len(raw_by_img)} images.")
            run_inference_to_cache = False
    else:
        run_inference_to_cache = True
        
    if run_inference_to_cache:
        print(f"Cache not found or incomplete. Re-running model inference on {TARGET_NUM_IMAGES} images...")
        from ultralytics import YOLO
        workspace_dir = os.path.join(PROJECT_DIR, "multimodal_detection", "workspace")
        model_paths = []
        for run_dir in os.listdir(workspace_dir):
            if run_dir.startswith("yolov8_fold"):
                best_pt = os.path.join(workspace_dir, run_dir, "weights", "best.pt")
                if os.path.exists(best_pt):
                    model_paths.append(best_pt)
                    
        print(f"Loading {len(model_paths)} models...")
        models = [YOLO(mp) for mp in model_paths]
        
        test_images_dir = os.path.join(MOCK_DIR, "mock_test_dataset", "images")
        test_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(".jpg")])[:TARGET_NUM_IMAGES]
        
        raw_by_img = {}
        for filename in tqdm(test_files, desc="Running Inference"):
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
            
        print(f"Saving predictions to cache: {CACHE_PATH}...")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump((raw_by_img, test_files), f)
            
    # 2. Load Ground Truths
    gt_all = load_ground_truth(Path(os.path.join(MOCK_DIR, "torch_eval", "gt")), CLASS_TO_ID)
    selected_img_ids = {os.path.splitext(f)[0] for f in test_files}
    gt = [g for g in gt_all if g.image_id in selected_img_ids]
    print(f"Loaded {len(gt)} ground truth targets for evaluation.")
    
    # 3. Define threshold groups
    cs_dict_1 = {CLASSES.index(name): (0.55 if name in ['car', 'suv', 'motorcycle'] else (0.35 if name in ['bus', 'trailer', 'crane', 'tank_truck'] else 0.45)) for name in CLASSES}
    cs_dict_2 = {CLASSES.index(name): (0.60 if name in ['car', 'suv', 'motorcycle'] else (0.40 if name in ['bus', 'trailer', 'crane', 'tank_truck'] else 0.50)) for name in CLASSES}
    cs_dict_3 = {CLASSES.index(name): (0.50 if name in ['car', 'suv', 'motorcycle'] else (0.30 if name in ['bus', 'trailer', 'crane', 'tank_truck'] else 0.40)) for name in CLASSES}
    
    T_OPTIONS = [
        {"name": "Fixed_0.45", "val": 0.45},
        {"name": "Fixed_0.50", "val": 0.50},
        {"name": "Fixed_0.55", "val": 0.55},
        {"name": "CS_0.55_0.45_0.35", "val": cs_dict_1},
        {"name": "CS_0.60_0.50_0.40", "val": cs_dict_2},
        {"name": "CS_0.50_0.40_0.30", "val": cs_dict_3}
    ]
    
    BOOST_FACTORS = [0.0, 0.02, 0.03, 0.04, 0.05]
    PENALTY_FACTORS = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    CONF_CUTOFFS = [0.20, 0.25]
    
    # Total combinations
    combinations = list(itertools.product(T_OPTIONS, BOOST_FACTORS, PENALTY_FACTORS, CONF_CUTOFFS))
    print(f"Total parameter combinations to search: {len(combinations)}")
    
    print("\nRunning grid search...")
    results = []
    
    start_time = time.time()
    for t_opt, bf, pf, cutoff in tqdm(combinations, desc="Grid Search Progress"):
        preds = []
        for img_id, raw_preds in raw_by_img.items():
            fused = run_general_nms_with_boosting(raw_preds, t_opt["val"], conf_cutoff=cutoff, boost_factor=bf, penalty_factor=pf)
            for p in fused:
                preds.append({
                    "img_id": img_id,
                    "class_idx": p["class_idx"],
                    "conf": p["conf"],
                    "coords": p["coords"]
                })
        ap50, ap50_95 = evaluate_predictions(preds, gt)
        results.append({
            "t_name": t_opt["name"],
            "t_val": t_opt["val"],
            "bf": bf,
            "pf": pf,
            "cutoff": cutoff,
            "ap50": ap50,
            "ap50_95": ap50_95
        })
        
    search_duration = time.time() - start_time
    print(f"Grid search completed in {search_duration:.2f} seconds.")
    
    # Sort and display top 15
    results.sort(key=lambda x: x["ap50"], reverse=True)
    
    print("\n==================== Top 15 Parameter Combinations ====================")
    print("| Rank | NMS Threshold Type   | Boost Factor | Penalty Factor | Cutoff | AP50       | AP50-95    |")
    print("|------|----------------------|--------------|----------------|--------|------------|------------|")
    for i, res in enumerate(results[:15], start=1):
        print(f"| {i:<4d} | {res['t_name']:<20s} | {res['bf']:<12.2f} | {res['pf']:<14.2f} | {res['cutoff']:<6.2f} | {res['ap50']:.4f}% | {res['ap50_95']:.4f}% |")
    print("=======================================================================")
    
    # Run a secondary fine-grained search around the best neighborhood
    best = results[0]
    print(f"\nBest configuration from coarse search: NMS={best['t_name']}, bf={best['bf']}, pf={best['pf']}, cutoff={best['cutoff']} -> AP50={best['ap50']:.4f}%")
    
    # Fine search range
    fine_bfs = [best['bf'] - 0.01, best['bf'], best['bf'] + 0.01]
    fine_pfs = [best['pf'] - 0.02, best['pf'], best['pf'] + 0.02]
    # Filter out invalid values
    fine_bfs = [x for x in fine_bfs if 0.0 <= x <= 0.10]
    fine_pfs = [x for x in fine_pfs if 0.50 <= x <= 1.00]
    
    # If the best threshold type was class-specific, let's keep it, but we can also search slightly higher or lower values
    if "CS" in best["t_name"]:
        # Find which of the 3 CS sets was best, and also try small adjustments:
        # e.g., CS offset by +0.02 or -0.02
        base_dict = best["t_val"]
        fine_ts = [
            {"name": f"{best['t_name']}_offset_-0.02", "val": {k: max(0.20, v - 0.02) for k, v in base_dict.items()}},
            {"name": best["t_name"], "val": base_dict},
            {"name": f"{best['t_name']}_offset_+0.02", "val": {k: min(0.80, v + 0.02) for k, v in base_dict.items()}},
        ]
    else:
        # Fixed threshold, search ±0.02
        base_t = best["t_val"]
        fine_ts = [
            {"name": f"Fixed_{base_t-0.02:.2f}", "val": base_t - 0.02},
            {"name": f"Fixed_{base_t:.2f}", "val": base_t},
            {"name": f"Fixed_{base_t+0.02:.2f}", "val": base_t + 0.02},
        ]
        
    fine_cutoffs = [best['cutoff'] - 0.02, best['cutoff'], best['cutoff'] + 0.02]
    
    fine_combinations = list(itertools.product(fine_ts, fine_bfs, fine_pfs, fine_cutoffs))
    print(f"\nRunning fine-grained grid search ({len(fine_combinations)} combinations)...")
    
    fine_results = []
    for t_opt, bf, pf, cutoff in fine_combinations:
        preds = []
        for img_id, raw_preds in raw_by_img.items():
            fused = run_general_nms_with_boosting(raw_preds, t_opt["val"], conf_cutoff=cutoff, boost_factor=bf, penalty_factor=pf)
            for p in fused:
                preds.append({
                    "img_id": img_id,
                    "class_idx": p["class_idx"],
                    "conf": p["conf"],
                    "coords": p["coords"]
                })
        ap50, ap50_95 = evaluate_predictions(preds, gt)
        fine_results.append({
            "t_name": t_opt["name"],
            "t_val": t_opt["val"],
            "bf": bf,
            "pf": pf,
            "cutoff": cutoff,
            "ap50": ap50,
            "ap50_95": ap50_95
        })
        
    fine_results.sort(key=lambda x: x["ap50"], reverse=True)
    
    print("\n==================== Top 10 Fine-Grained Search Results ====================")
    print("| Rank | NMS Threshold Type   | Boost Factor | Penalty Factor | Cutoff | AP50       | AP50-95    |")
    print("|------|----------------------|--------------|----------------|--------|------------|------------|")
    for i, res in enumerate(fine_results[:10], start=1):
        print(f"| {i:<4d} | {res['t_name']:<20s} | {res['bf']:<12.3f} | {res['pf']:<14.3f} | {res['cutoff']:<6.3f} | {res['ap50']:.4f}% | {res['ap50_95']:.4f}% |")
    print("============================================================================")
    
    absolute_best = fine_results[0]
    print(f"\n🎉 Optimization Complete! Absolute Best Configuration:")
    print(f" - AP50: {absolute_best['ap50']:.6f}% (AP50-95: {absolute_best['ap50_95']:.6f}%)")
    print(f" - NMS Threshold Type: {absolute_best['t_name']}")
    print(f" - Boost Factor: {absolute_best['bf']:.3f}")
    print(f" - Penalty Factor: {absolute_best['pf']:.3f}")
    print(f" - Confidence Cutoff: {absolute_best['cutoff']:.3f}")

if __name__ == "__main__":
    main()
