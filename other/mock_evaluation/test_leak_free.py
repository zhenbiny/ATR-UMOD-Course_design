import os
import cv2
import numpy as np
import sys
import pickle
import shutil
from tqdm import tqdm
from pathlib import Path

MOCK_DIR = r"c:\Users\17638\Desktop\NUDT\智能图像处理\mock_evaluation_closed_loop"
sys.path.append(os.path.join(MOCK_DIR, "torch_eval"))

from func.metrics import match_predictions, parse_iou_thresholds, summarize
from func.parsers import parse_classes, load_ground_truth, Prediction

CLASSES = [
    'car', 'suv', 'van', 'bus', 'freight_car', 
    'truck', 'motorcycle', 'trailer', 'excavator', 
    'crane', 'tank_truck'
]
NAMES, CLASS_TO_ID = parse_classes(",".join(CLASSES))

PROJECT_DIR = r"c:\Users\17638\Desktop\NUDT\智能图像处理"
VAL_SPLIT_PATH = os.path.join(PROJECT_DIR, "multimodal_detection", "data_splits", "val_fold0.txt")
SOURCE_DATA_DIR = os.path.join(PROJECT_DIR, "ATR-UMOD", "train")

LEAK_FREE_GT_DIR = os.path.join(MOCK_DIR, "leak_free_gt")
CACHE_PATH = os.path.join(MOCK_DIR, "leak_free_raw_preds.pkl")
TARGET_NUM_IMAGES = 300

def evaluate_predictions(preds_dict, gt):
    eval_preds = []
    for item in preds_dict:
        pts = item["coords"].reshape(4, 2)
        eval_preds.append(Prediction(item["img_id"], item["class_idx"], item["conf"], pts))
        
    thresholds = parse_iou_thresholds("0.50:0.05:0.95")
    correct, conf, pred_cls, target_cls = match_predictions(gt, eval_preds, thresholds, "obb")
    summary, per_class = summarize(correct, conf, pred_cls, target_cls, NAMES)
    return summary

def nms_boxes(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
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
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def run_single_model_nms(all_preds, nms_threshold=0.50, conf_cutoff=0.20):
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
        
        keep = nms_boxes(aabb_boxes, cls_scores, iou_threshold=nms_threshold)
        for idx in keep:
            if cls_scores[idx] >= conf_cutoff:
                final_preds.append({
                    "class_idx": class_idx,
                    "conf": float(cls_scores[idx]),
                    "coords": cls_coords[idx]
                })
    return final_preds

def main():
    print("=== Step 1: Loading Leak-Free Validation Images List ===")
    with open(VAL_SPLIT_PATH, "r", encoding="utf-8") as f:
        val_paths = [line.strip() for line in f if line.strip()]
        
    selected_paths = val_paths[:TARGET_NUM_IMAGES]
    selected_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in selected_paths]
    print(f"Selected {len(selected_img_ids)} images from Fold 0 validation set.")
    
    # Copy xml annotations to temporary folder
    os.makedirs(LEAK_FREE_GT_DIR, exist_ok=True)
    for img_id in selected_img_ids:
        src_xml = os.path.join(SOURCE_DATA_DIR, "labels", f"{img_id}.xml")
        dst_xml = os.path.join(LEAK_FREE_GT_DIR, f"{img_id}.xml")
        if os.path.exists(src_xml):
            shutil.copy2(src_xml, dst_xml)
            
    # Load Ground Truths
    gt = load_ground_truth(Path(LEAK_FREE_GT_DIR), CLASS_TO_ID)
    print(f"Loaded {len(gt)} ground truths for leak-free evaluation.")
    
    # 2. Run model predictions or load from cache
    if os.path.exists(CACHE_PATH):
        print(f"Loading predictions from cache: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            raw_by_img = pickle.load(f)
    else:
        print("Cache not found. Running Fold 0 model inference on 300 validation images...")
        from ultralytics import YOLO
        model_path = os.path.join(PROJECT_DIR, "multimodal_detection", "workspace", "yolov8_fold0", "weights", "best.pt")
        if not os.path.exists(model_path):
            print(f"Error: Fold 0 model weight not found at {model_path}")
            return
            
        model = YOLO(model_path)
        
        raw_by_img = {}
        for img_id in tqdm(selected_img_ids, desc="Fold 0 Inference"):
            rgb_path = os.path.join(SOURCE_DATA_DIR, "images", f"{img_id}.jpg")
            ir_path = os.path.join(SOURCE_DATA_DIR, "images_ir", f"{img_id}.jpg")
            
            rgb_img = cv2.imread(rgb_path)
            ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            if rgb_img is None or ir_img is None:
                continue
                
            fused_img = cv2.merge([ir_img, rgb_img[:, :, 1], rgb_img[:, :, 2]])
            
            results = model.predict(fused_img, conf=0.1, imgsz=640, device=0, verbose=False)
            img_raw_preds = []
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
            pickle.dump(raw_by_img, f)
            
    # 3. Evaluate different confidence cutoffs for a single model (LEAK-FREE)
    print("\n==================== Leak-Free Single Model Evaluation ====================")
    print("| Cutoff | Precision | Recall   | AP50       | AP50-95    | Total Detections |")
    print("|--------|-----------|----------|------------|------------|------------------|")
    
    cutoffs = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    for cutoff in cutoffs:
        preds = []
        total_dts = 0
        for img_id, raw_preds in raw_by_img.items():
            fused = run_single_model_nms(raw_preds, nms_threshold=0.50, conf_cutoff=cutoff)
            for p in fused:
                preds.append({
                    "img_id": img_id,
                    "class_idx": p["class_idx"],
                    "conf": p["conf"],
                    "coords": p["coords"]
                })
                total_dts += 1
                
        metrics = evaluate_predictions(preds, gt)
        print(f"| {cutoff:<6.2f} | {metrics['precision']*100:<9.2f}% | {metrics['recall']*100:<8.2f}% | {metrics['AP50']*100:<10.4f}% | {metrics['AP50-95']*100:<10.4f}% | {total_dts:<16d} |")
    print("===========================================================================")
    
    # 4. Consensus-filtering analysis on 300-image mock ensemble cache
    ENSEMBLE_CACHE = os.path.join(MOCK_DIR, "raw_preds_cache_300.pkl")
    if os.path.exists(ENSEMBLE_CACHE):
        print("\n=== Running Consensus-Filtering Simulation on 300-Image Mock Ensemble Cache ===")
        with open(ENSEMBLE_CACHE, "rb") as f:
            ensemble_raw_by_img, test_files = pickle.load(f)
            
        gt_all = load_ground_truth(Path(os.path.join(MOCK_DIR, "torch_eval", "gt")), CLASS_TO_ID)
        selected_img_ids_ens = {os.path.splitext(f)[0] for f in test_files}
        gt_ens = [g for g in gt_all if g.image_id in selected_img_ids_ens]
        
        def run_consensus_nms(all_preds, nms_threshold=0.47, conf_cutoff=0.25, min_agreement=2, pf=0.0):
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
                    
                    overlapping_boxes_count = 1 + int(np.sum(ovr > nms_threshold))
                    counts.append(overlapping_boxes_count)
                    
                    inds = np.where(ovr <= nms_threshold)[0]
                    order = order[inds + 1]
                    
                for idx, overlap_cnt in zip(keep, counts):
                    conf = cls_scores[idx]
                    if overlap_cnt >= min_agreement:
                        # Shared detection: keep confidence
                        adjusted_conf = conf
                    else:
                        # Lone detection: apply penalty (pf)
                        adjusted_conf = conf * pf
                        
                    if adjusted_conf >= conf_cutoff:
                        final_preds.append({
                            "class_idx": class_idx,
                            "conf": float(adjusted_conf),
                            "coords": cls_coords[idx]
                        })
            return final_preds

        print("\nConsensus NMS parameter tests:")
        print("| Config | Min Agreement | Penalty | Cutoff | AP50       | AP50-95    | Total Detections |")
        print("|--------|---------------|---------|--------|------------|------------|------------------|")
        
        tests = [
            ("No Consensus (Baseline)", 1, 1.0, 0.20),
            ("No Consensus (Low Cutoff)", 1, 1.0, 0.25),
            ("Mild Penalty (pf=0.85)", 1, 0.85, 0.25),
            ("Strong Penalty (pf=0.50)", 1, 0.50, 0.25),
            ("Strong Penalty (pf=0.50)", 1, 0.50, 0.30),
            ("Strict Consensus (K>=2, pf=0.0)", 2, 0.0, 0.20),
            ("Strict Consensus (K>=2, pf=0.0)", 2, 0.0, 0.25),
            ("Strict Consensus (K>=2, pf=0.0)", 2, 0.0, 0.30),
        ]
        
        for name, min_agree, pf, cutoff in tests:
            preds = []
            total_dts = 0
            for img_id, raw_preds in ensemble_raw_by_img.items():
                fused = run_consensus_nms(raw_preds, nms_threshold=0.47, conf_cutoff=cutoff, min_agreement=min_agree, pf=pf)
                for p in fused:
                    preds.append({
                        "img_id": img_id,
                        "class_idx": p["class_idx"],
                        "conf": p["conf"],
                        "coords": p["coords"]
                    })
                    total_dts += 1
            metrics = evaluate_predictions(preds, gt_ens)
            print(f"| {name:<30s} | {min_agree:<13d} | {pf:<7.2f} | {cutoff:<6.2f} | {metrics['AP50']*100:<10.4f}% | {metrics['AP50-95']*100:<10.4f}% | {total_dts:<16d} |")
        print("==========================================================================================")

if __name__ == "__main__":
    main()
