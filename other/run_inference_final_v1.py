import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# ==============================================================================
# 【填写区域】请在下方分别填入您的可见光 (RGB) 与红外 (IR) 测试集图像文件夹的绝对路径
# (例如: TEST_RGB_DIR = r"D:\ATR-UMOD\test\images", TEST_IR_DIR = r"D:\ATR-UMOD\test\images_ir")
TEST_RGB_DIR = r""
TEST_IR_DIR = r""
# ==============================================================================

# 项目默认路径配置
PROJECT_DIR = r"c:\Users\17638\Desktop\NUDT\智能图像处理"
DEV_DIR = os.path.join(PROJECT_DIR, "multimodal_detection")
WORKSPACE_DIR = os.path.join(DEV_DIR, "workspace")
OUTPUT_DIR = os.path.join(DEV_DIR, "submission_results")

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
        
        # 计算轴对齐外接矩形 (AABB)，用于 NMS
        x_coords = cls_coords[:, ::2]
        y_coords = cls_coords[:, 1::2]
        
        x_min = np.min(x_coords, axis=1)
        y_min = np.min(y_coords, axis=1)
        x_max = np.max(x_coords, axis=1)
        y_max = np.max(y_coords, axis=1)
        
        aabb_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
        keep_indices = nms_boxes(aabb_boxes, cls_scores, iou_threshold=nms_threshold)
        
        for idx in keep_indices:
            final_preds.append({
                "class_idx": class_idx,
                "conf": cls_scores[idx],
                "coords": cls_coords[idx]
            })
            
    return final_preds

def main():
    if not TEST_RGB_DIR or not TEST_IR_DIR:
        print("==================================================================")
        print("错误：请先在代码第 9-10 行分别填入您的可见光与红外测试集绝对路径后再运行！")
        print("==================================================================")
        return
        
    # 1. 查找所有可用的 fold 模型
    model_paths = []
    if os.path.exists(WORKSPACE_DIR):
        for run_dir in os.listdir(WORKSPACE_DIR):
            if run_dir.startswith("yolov8_fold"):
                best_pt = os.path.join(WORKSPACE_DIR, run_dir, "weights", "best.pt")
                if os.path.exists(best_pt):
                    model_paths.append(best_pt)
                    
    if not model_paths:
        print(f"错误：未在 {WORKSPACE_DIR} 下找到任何已训练好的 5折模型权重！")
        return
        
    print(f"共发现 {len(model_paths)} 个 Fold 模型参与集成预测：")
    for mp in model_paths:
        print(f" - {mp}")
        
    print("正在载入模型权重，请稍候...")
    models = [YOLO(mp) for mp in model_paths]
    
    # 2. 扫描测试集文件
    if not os.path.exists(TEST_RGB_DIR):
        print(f"错误：可见光测试集图像目录不存在: {TEST_RGB_DIR}")
        return
    if not os.path.exists(TEST_IR_DIR):
        print(f"错误：红外测试集图像目录不存在: {TEST_IR_DIR}")
        return
        
    test_files = [f for f in os.listdir(TEST_RGB_DIR) if f.endswith(".jpg")]
    print(f"成功扫描测试集，共发现 {len(test_files)} 张待推理图像。")
    
    # 3. 创建输出文件夹及 11 类 TXT 文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_files = {}
    for name in CLASSES:
        txt_path = os.path.join(OUTPUT_DIR, f"{name}.txt")
        txt_files[name] = open(txt_path, "w", encoding="utf-8")
        
    # 4. 推理主循环
    print("开始融合通道并执行 5 折模型集成推理（AABB-NMS）...")
    for filename in tqdm(test_files):
        img_id = os.path.splitext(filename)[0]
        
        # 路径组装
        rgb_path = os.path.join(TEST_RGB_DIR, filename)
        ir_path = os.path.join(TEST_IR_DIR, filename)
        
        if not os.path.exists(ir_path):
            print(f"\n警告：未找到配对的红外图像: {ir_path}，已跳过。")
            continue
            
        # 读取并合成为 3 通道 BGR 图像
        rgb_img = cv2_imread_unicode(rgb_path)
        ir_img = cv2_imread_unicode(ir_path, cv2.IMREAD_GRAYSCALE)
        
        if rgb_img is None or ir_img is None:
            continue
            
        fused_img = cv2.merge([ir_img, rgb_img[:, :, 1], rgb_img[:, :, 2]])
        
        # 汇总全部 5 个模型的预测框
        raw_predictions = []
        for model in models:
            results = model.predict(fused_img, conf=0.15, imgsz=640, device=0, verbose=False)
            for result in results:
                if result.obb is not None:
                    obb_coords = result.obb.xyxyxyxy.cpu().numpy() # (N, 4, 2)
                    confs = result.obb.conf.cpu().numpy()
                    classes = result.obb.cls.cpu().numpy()
                    
                    for i in range(len(classes)):
                        flat_coords = obb_coords[i].flatten()
                        raw_predictions.append({
                            "class_idx": int(classes[i]),
                            "conf": float(confs[i]),
                            "coords": flat_coords
                        })
                        
        # 执行跨模型预测融合去重
        fused_predictions = ensemble_predictions(raw_predictions, nms_threshold=0.5)
        
        # 输出至类别对应的 txt 文件
        for pred in fused_predictions:
            if pred["conf"] < 0.20:
                continue
            class_name = CLASSES[pred["class_idx"]]
            c = pred["coords"]
            coord_str = " ".join([f"{val:.2f}" for val in c])
            # 输出格式符合大作业规范: image_id confidence x1 y1 x2 y2 x3 y3 x4 y4
            txt_files[class_name].write(f"{img_id} {pred['conf']:.6f} {coord_str}\n")
            
    # 5. 关闭所有文件
    for f in txt_files.values():
        f.close()
        
    print("\n==================================================================")
    print(f"🎉 推理打包顺利完成！")
    print(f"所有结果文件已成功保存至：")
    print(f"📁 {OUTPUT_DIR}")
    print("==================================================================")

if __name__ == "__main__":
    main()
