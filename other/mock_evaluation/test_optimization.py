import os
import cv2
import numpy as np
import sys
import shutil

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

def obb_nms(all_preds, iou_threshold=0.5):
    """标准的 OBB 旋转框 NMS"""
    if not all_preds:
        return []
        
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
        keep = []
        visited = np.zeros(len(cls_scores), dtype=bool)
        
        for i in range(len(cls_scores)):
            if visited[i]:
                continue
            keep.append(i)
            visited[i] = True
            
            for j in range(i + 1, len(cls_scores)):
                if visited[j]:
                    continue
                iou = obb_iou(pts[i], pts[j])
                if iou > iou_threshold:
                    visited[j] = True
                    
        for idx in keep:
            final_preds.append({
                "class_idx": class_idx,
                "conf": float(cls_scores[idx]),
                "coords": cls_coords[idx]
            })
            
    return final_preds

def obb_box_voting(all_preds, iou_threshold=0.5):
    """加权旋转框投票融合 (Weighted OBB Box Voting / WBF-OBB)"""
    if not all_preds:
        return []
        
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
            
            # Find all overlapping boxes above threshold
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
            
            # Calculate weighted average coordinates
            sum_scores = np.sum(box_scores)
            if sum_scores <= 0:
                weighted_coords = box_coords[0]
            else:
                weighted_coords = np.sum(box_coords * box_scores[:, None], axis=0) / sum_scores
            
            # Confidence score fusion:
            # We can use the max confidence, but weight it slightly if supported by multiple models
            max_conf = box_scores[0]
            # Vote confirmation boost: e.g., +2% confidence for each extra model predicting it (max +10%)
            boost = min(0.1, 0.02 * (len(overlap_indices) - 1))
            fused_conf = min(1.0, max_conf * (1.0 + boost))
            
            final_preds.append({
                "class_idx": class_idx,
                "conf": float(fused_conf),
                "coords": weighted_coords
            })
            
    return final_preds

# Let's perform a dry run evaluation of these strategies using our saved mock data!
# In step 4 of run_closed_loop_test.py, we collected raw predictions from the 5 models.
# Let's run a test where we run inference, apply these three strategies, and evaluate the final AP50 scores.
# Since running inference takes ~1.5 minutes, let's write a python file to perform inference once, save the raw predictions to memory/json, 
# and then compare the NMS strategies instantly!

def evaluate_predictions(preds_dict, gt, class_to_id, names):
    # Convert local preds_dict to torch_eval Prediction objects
    # preds_dict: list of {"img_id": str, "class_idx": int, "conf": float, "coords": np.ndarray}
    eval_preds = []
    for item in preds_dict:
        pts = item["coords"].reshape(4, 2)
        eval_preds.append(Prediction(item["img_id"], item["class_idx"], item["conf"], pts))
        
    thresholds = parse_iou_thresholds("0.50:0.05:0.95")
    correct, conf, pred_cls, target_cls = match_predictions(gt, eval_preds, thresholds, "obb")
    summary, per_class = summarize(correct, conf, pred_cls, target_cls, names)
    return summary["AP50"] * 100.0, summary["AP50-95"] * 100.0

def main():
    print("=== Testing Fusion Strategies Optimization ===")
    from ultralytics import YOLO
    
    # 1. Load models
    workspace_dir = r"c:\Users\17638\Desktop\NUDT\智能图像处理\multimodal_detection\workspace"
    model_paths = []
    for run_dir in os.listdir(workspace_dir):
        if run_dir.startswith("yolov8_fold"):
            best_pt = os.path.join(workspace_dir, run_dir, "weights", "best.pt")
            if os.path.exists(best_pt):
                model_paths.append(best_pt)
                
    models = [YOLO(mp) for mp in model_paths]
    print(f"Loaded {len(models)} models.")
    
    # 2. Get subset of images to evaluate quickly (e.g. 100 images)
    test_images_dir = os.path.join(MOCK_DIR, "mock_test_dataset", "images")
    test_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(".jpg")])[:150]
    print(f"Evaluating on {len(test_files)} images...")
    
    # 3. Load Ground Truths
    from pathlib import Path
    names, class_to_id = parse_classes(",".join(CLASSES))
    gt_path = os.path.join(MOCK_DIR, "torch_eval", "gt")
    # We only load GT for the selected images
    gt_all = load_ground_truth(Path(gt_path), class_to_id)
    # Filter GT for selected images
    selected_img_ids = {os.path.splitext(f)[0] for f in test_files}
    gt = [g for g in gt_all if g.image_id in selected_img_ids]
    print(f"Loaded {len(gt)} ground truth targets.")
    
    # 4. Run inference and collect raw predictions
    raw_by_img = {}
    print("Running model inference to gather raw outputs...")
    
    from tqdm import tqdm
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

    # 5. Apply different fusion strategies
    print("\n--- Running Evaluation ---")
    
    # Strategy A: Baseline AABB NMS
    from run_inference import ensemble_predictions as aabb_nms_func
    preds_aabb = []
    for img_id, raw_preds in raw_by_img.items():
        fused = aabb_nms_func(raw_preds, nms_threshold=0.5)
        for p in fused:
            preds_aabb.append({
                "img_id": img_id,
                "class_idx": p["class_idx"],
                "conf": p["conf"],
                "coords": p["coords"]
            })
    ap50_aabb, map_aabb = evaluate_predictions(preds_aabb, gt, class_to_id, names)
    print(f"Strategy A (AABB NMS):   AP50 = {ap50_aabb:.4f}%, AP50-95 = {map_aabb:.4f}%")
    
    # Strategy B: OBB NMS
    preds_obb = []
    for img_id, raw_preds in raw_by_img.items():
        fused = obb_nms(raw_preds, iou_threshold=0.5)
        for p in fused:
            preds_obb.append({
                "img_id": img_id,
                "class_idx": p["class_idx"],
                "conf": p["conf"],
                "coords": p["coords"]
            })
    ap50_obb, map_obb = evaluate_predictions(preds_obb, gt, class_to_id, names)
    print(f"Strategy B (OBB NMS):    AP50 = {ap50_obb:.4f}%, AP50-95 = {map_obb:.4f}%")
    
    # Strategy C: OBB Box Voting
    preds_voting = []
    for img_id, raw_preds in raw_by_img.items():
        fused = obb_box_voting(raw_preds, iou_threshold=0.5)
        for p in fused:
            preds_voting.append({
                "img_id": img_id,
                "class_idx": p["class_idx"],
                "conf": p["conf"],
                "coords": p["coords"]
            })
    ap50_vote, map_vote = evaluate_predictions(preds_voting, gt, class_to_id, names)
    print(f"Strategy C (OBB Voting):  AP50 = {ap50_vote:.4f}%, AP50-95 = {map_vote:.4f}%")

if __name__ == "__main__":
    main()
