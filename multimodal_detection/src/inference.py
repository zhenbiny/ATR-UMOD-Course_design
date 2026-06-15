import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# 11个类别定义
CLASSES = [
    'car', 'suv', 'van', 'bus', 'freight_car', 
    'truck', 'motorcycle', 'trailer', 'excavator', 
    'crane', 'tank_truck'
]

def cv2_imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """支持中文路径的图片读取"""
    try:
        nparr = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(nparr, flags)
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def nms_boxes(boxes, scores, iou_threshold=0.5):
    """使用 numpy 实现的标准 NMS"""
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
        
        # 防止除以0
        union = areas[i] + areas[order[1:]] - inter
        union = np.maximum(union, 1e-6)
        ovr = inter / union
        
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

def ensemble_predictions(all_preds, nms_threshold=0.5):
    """对所有模型预测出来的框进行融合与 NMS 去重"""
    if not all_preds:
        return []
        
    # 按类别进行分组 NMS
    final_preds = []
    
    # 转换为 numpy 数组便于操作
    classes = np.array([p["class_idx"] for p in all_preds])
    scores = np.array([p["conf"] for p in all_preds])
    coords = np.array([p["coords"] for p in all_preds]) # Shape: (N, 8)
    
    for class_idx in range(len(CLASSES)):
        mask = (classes == class_idx)
        if not np.any(mask):
            continue
            
        cls_scores = scores[mask]
        cls_coords = coords[mask]
        
        # 计算每个旋转框的轴对齐外接矩形 (AABB)，用于计算 IoU NMS
        # cls_coords shape: (M, 8) -> x1, y1, x2, y2, x3, y3, x4, y4
        x_coords = cls_coords[:, ::2]
        y_coords = cls_coords[:, 1::2]
        
        x_min = np.min(x_coords, axis=1)
        y_min = np.min(y_coords, axis=1)
        x_max = np.max(x_coords, axis=1)
        y_max = np.max(y_coords, axis=1)
        
        aabb_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
        
        # 运行 NMS
        keep_indices = nms_boxes(aabb_boxes, cls_scores, iou_threshold=nms_threshold)
        
        for idx in keep_indices:
            final_preds.append({
                "class_idx": class_idx,
                "conf": cls_scores[idx],
                "coords": cls_coords[idx]
            })
            
    return final_preds

def main():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    DEV_DIR = os.path.join(PROJECT_DIR, "multimodal_detection")
    
    default_workspace = os.path.join(DEV_DIR, "workspace")
    default_output = os.path.join(DEV_DIR, "submission_results")

    parser = argparse.ArgumentParser(description="ATR-UMOD 测试集多折模型集成推理打包脚本")
    parser.add_argument("--test_dir", type=str, required=True, help="测试集根目录（需含 images/ 和 images_ir/）")
    parser.add_argument("--workspace", type=str, default=default_workspace, help="工作空间 runs 目录")
    parser.add_argument("--output_dir", type=str, default=default_output, help="结果输出保存文件夹")
    parser.add_argument("--conf", type=float, default=0.1, help="推理置信度阈值 (默认 0.1)")
    parser.add_argument("--nms_thresh", type=float, default=0.5, help="集成 NMS 阈值 (默认 0.5)")
    
    args = parser.parse_args()
    
    # 1. 查找所有可用的 fold 模型
    model_paths = []
    if os.path.exists(args.workspace):
        for run_dir in os.listdir(args.workspace):
            if run_dir.startswith("yolov8_fold"):
                best_pt = os.path.join(args.workspace, run_dir, "weights", "best.pt")
                if os.path.exists(best_pt):
                    model_paths.append(best_pt)
                    
    if not model_paths:
        print(f"警告：未在 {args.workspace} 下找到任何已训练好的 yolov8_foldX 模型权重！")
        return
        
    print(f"共发现 {len(model_paths)} 个 Fold 模型进行集成推理：")
    for mp in model_paths:
        print(f" - {mp}")
        
    # 加载所有模型
    models = [YOLO(mp) for mp in model_paths]
    
    # 2. 扫描测试集文件
    test_images_dir = os.path.join(args.test_dir, "images")
    if not os.path.exists(test_images_dir):
        print(f"错误：测试集图像目录不存在: {test_images_dir}")
        return
        
    test_files = [f for f in os.listdir(test_images_dir) if f.endswith(".jpg")]
    print(f"共发现 {len(test_files)} 张待推理的测试图像。")
    
    # 3. 创建输出文件夹及 11 类 TXT 文件
    os.makedirs(args.output_dir, exist_ok=True)
    txt_files = {}
    for name in CLASSES:
        txt_path = os.path.join(args.output_dir, f"{name}.txt")
        txt_files[name] = open(txt_path, "w", encoding="utf-8")
        
    # 4. 循环遍历推理
    print("开始进行多模态图像合成与集成推理...")
    for filename in tqdm(test_files):
        img_id = os.path.splitext(filename)[0]
        
        # 双模态路径
        rgb_path = os.path.join(args.test_dir, "images", filename)
        ir_path = os.path.join(args.test_dir, "images_ir", filename)
        
        if not os.path.exists(ir_path):
            # 如果对应红外图像不存在，打印警告并跳过
            print(f"警告：找不到对应的红外图像: {ir_path}")
            continue
            
        # 通道合成 B = IR, G = RGB_G, R = RGB_R
        rgb_img = cv2_imread_unicode(rgb_path)
        ir_img = cv2_imread_unicode(ir_path, cv2.IMREAD_GRAYSCALE)
        
        if rgb_img is None or ir_img is None:
            continue
            
        fused_img = cv2.merge([ir_img, rgb_img[:, :, 1], rgb_img[:, :, 2]])
        
        # 收集所有模型的预测
        raw_predictions = []
        for model in models:
            # verbose=False 禁用控制台打印
            results = model.predict(fused_img, conf=args.conf, imgsz=640, device=0, verbose=False)
            for result in results:
                if result.obb is not None:
                    # YOLOv8-OBB 的 xyxyxyxy 是 OBB 顺时针四顶点像素坐标，Shape: (N, 4, 2)
                    obb_coords = result.obb.xyxyxyxy.cpu().numpy()
                    confs = result.obb.conf.cpu().numpy()
                    classes = result.obb.cls.cpu().numpy()
                    
                    for i in range(len(classes)):
                        # 将坐标展平为 x1 y1 x2 y2 x3 y3 x4 y4 (8个数)
                        flat_coords = obb_coords[i].flatten()
                        raw_predictions.append({
                            "class_idx": int(classes[i]),
                            "conf": float(confs[i]),
                            "coords": flat_coords
                        })
                        
        # 集成融合与 NMS 去重
        fused_predictions = ensemble_predictions(raw_predictions, nms_threshold=args.nms_thresh)
        
        # 写入对应的 txt 文件
        for pred in fused_predictions:
            class_name = CLASSES[pred["class_idx"]]
            c = pred["coords"]
            # 格式: image_id confidence x1 y1 x2 y2 x3 y3 x4 y4
            coord_str = " ".join([f"{val:.2f}" for val in c])
            txt_files[class_name].write(f"{img_id} {pred['conf']:.6f} {coord_str}\n")
            
    # 5. 关闭所有文件
    for f in txt_files.values():
        f.close()
        
    print(f"推理打包完毕！所有结果文件已成功输出至: {args.output_dir}")

if __name__ == "__main__":
    main()
