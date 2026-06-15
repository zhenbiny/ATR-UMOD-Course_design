import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# ==============================================================================
# 【配置区域】
# 1. 请在下方分别填入您的可见光 (RGB) 与红外 (IR) 测试集图像文件夹的绝对路径
# (例如: TEST_RGB_DIR = r"D:\ATR-UMOD\test\images", TEST_IR_DIR = r"D:\ATR-UMOD\test\images_ir")
TEST_RGB_DIR = r""
TEST_IR_DIR = r""

# 2. 推理模式配置:
#    - "single": 使用最优的单折模型 (Fold 0)，速度极快，定位极准，直接调用 GPU OBB NMS。
#    - "ensemble": 自动加载 5 个 Fold 模型进行预测合并，并调用基于 OpenCV 的旋转框 OBB-NMS 融合去重。
INFERENCE_MODE = "single"
# ==============================================================================

# 项目默认路径配置（动态相对路径定位）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEV_DIR = os.path.join(PROJECT_DIR, "multimodal_detection")
WORKSPACE_DIR = os.path.join(DEV_DIR, "workspace")
OUTPUT_DIR = os.path.join(DEV_DIR, "submission_results")

# 置信度阈值与 NMS IoU 阈值配置
CONF_CUTOFF = 0.30
NMS_IOU = 0.50

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

def obb_iou_opencv(box1, box2):
    """
    计算两个 OBB (4顶点格式: x1, y1, x2, y2, x3, y3, x4, y4) 的 IoU。
    使用 OpenCV 的 rotatedRectangleIntersection 计算。
    """
    pts1 = box1.reshape(4, 2).astype(np.float32)
    pts2 = box2.reshape(4, 2).astype(np.float32)
    
    r1 = cv2.minAreaRect(pts1)
    r2 = cv2.minAreaRect(pts2)
    
    # 获取精确的 OBB 框面积
    area1 = r1[1][0] * r1[1][1]
    area2 = r2[1][0] * r2[1][1]
    
    if area1 <= 0 or area2 <= 0:
        return 0.0
        
    ret, inter = cv2.rotatedRectangleIntersection(r1, r2)
    if ret == cv2.INTERSECT_NONE or inter is None:
        return 0.0
        
    inter_area = cv2.contourArea(inter)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
        
    return inter_area / union_area

def nms_obb(coords, scores, classes, iou_threshold=0.5):
    """
    使用 OpenCV rotatedRectangleIntersection 实现的面向旋转框 (OBB) 的按类别分组 NMS。
    coords: shape (M, 8) -> x1, y1, x2, y2, x3, y3, x4, y4
    scores: shape (M,)
    classes: shape (M,)
    """
    if len(coords) == 0:
        return []
        
    keep = []
    unique_classes = np.unique(classes)
    
    for cls in unique_classes:
        cls_mask = (classes == cls)
        cls_indices = np.where(cls_mask)[0]
        
        cls_coords = coords[cls_mask]
        cls_scores = scores[cls_mask]
        
        order = cls_scores.argsort()[::-1]
        cls_keep = []
        
        while order.size > 0:
            i = order[0]
            cls_keep.append(cls_indices[i])
            
            if order.size == 1:
                break
                
            ious = []
            for j in order[1:]:
                iou = obb_iou_opencv(cls_coords[i], cls_coords[j])
                ious.append(iou)
                
            ious = np.array(ious)
            inds = np.where(ious <= iou_threshold)[0]
            order = order[inds + 1]
            
        keep.extend(cls_keep)
        
    return keep

def main():
    if not TEST_RGB_DIR or not TEST_IR_DIR:
        print("==================================================================")
        print("错误：请先在代码第 10-11 行分别填入您的可见光与红外测试集绝对路径后再运行！")
        print("==================================================================")
        return
        
    # 扫描测试集文件
    if not os.path.exists(TEST_RGB_DIR):
        print(f"错误：可见光测试集图像目录不存在: {TEST_RGB_DIR}")
        return
    if not os.path.exists(TEST_IR_DIR):
        print(f"错误：红外测试集图像目录不存在: {TEST_IR_DIR}")
        return
        
    test_files = [f for f in os.listdir(TEST_RGB_DIR) if f.endswith(".jpg")]
    print(f"成功扫描测试集，共发现 {len(test_files)} 张待推理图像。")
    
    # 1. 载入模型
    models = []
    if INFERENCE_MODE == "single":
        model_path = os.path.join(WORKSPACE_DIR, "yolov8_fold0", "weights", "best.pt")
        if not os.path.exists(model_path):
            print(f"错误：未在 {model_path} 找到 Fold 0 模型权重！")
            return
        print(f"正在载入单折最优模型 (Fold 0)：{model_path} ...")
        models.append(YOLO(model_path))
    elif INFERENCE_MODE == "ensemble":
        print("正在搜索所有已训练的 Fold 模型...")
        if os.path.exists(WORKSPACE_DIR):
            for run_dir in os.listdir(WORKSPACE_DIR):
                if run_dir.startswith("yolov8_fold"):
                    best_pt = os.path.join(WORKSPACE_DIR, run_dir, "weights", "best.pt")
                    if os.path.exists(best_pt):
                        models.append(best_pt)
        if not models:
            print(f"错误：未在 {WORKSPACE_DIR} 下找到任何已训练好的 yolov8_foldX 模型权重！")
            return
        print(f"共发现 {len(models)} 个 Fold 模型参与集成预测，正在载入...")
        models = [YOLO(mp) for mp in models]
    else:
        print(f"错误：不支持的推理模式: {INFERENCE_MODE}。请在配置中填入 'single' 或 'ensemble'。")
        return
        
    # 2. 创建输出文件夹及 11 类 TXT 文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    txt_files = {}
    for name in CLASSES:
        txt_path = os.path.join(OUTPUT_DIR, f"{name}.txt")
        txt_files[name] = open(txt_path, "w", encoding="utf-8")
        
    # 3. 推理主循环
    print(f"开始融合通道并执行【{INFERENCE_MODE}】推理 (置信度阈值: {CONF_CUTOFF}, OBB-NMS IoU: {NMS_IOU}) ...")
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
        
        # 收集所有模型对该图的预测结果
        raw_coords = []
        raw_confs = []
        raw_classes = []
        
        for model in models:
            results = model.predict(
                fused_img, 
                conf=CONF_CUTOFF, 
                iou=NMS_IOU, # 单模型模式下直接让 YOLO 内部执行 OBB NMS
                imgsz=640, 
                device=0, 
                verbose=False
            )
            
            for result in results:
                if result.obb is not None:
                    obb_coords = result.obb.xyxyxyxy.cpu().numpy() # (N, 4, 2)
                    confs = result.obb.conf.cpu().numpy()
                    classes = result.obb.cls.cpu().numpy()
                    
                    for i in range(len(classes)):
                        raw_coords.append(obb_coords[i].flatten())
                        raw_confs.append(float(confs[i]))
                        raw_classes.append(int(classes[i]))
                        
        if len(raw_confs) == 0:
            continue
            
        raw_coords = np.array(raw_coords)
        raw_confs = np.array(raw_confs)
        raw_classes = np.array(raw_classes)
        
        # 如果是多模型集成模式，执行精确的面向旋转框的 OBB-NMS 融合
        if INFERENCE_MODE == "ensemble" and len(models) > 1:
            keep_indices = nms_obb(raw_coords, raw_confs, raw_classes, iou_threshold=NMS_IOU)
            final_coords = raw_coords[keep_indices]
            final_confs = raw_confs[keep_indices]
            final_classes = raw_classes[keep_indices]
        else:
            # 单模型模式（或者只有一个模型有效加载时）不需要再次 NMS
            final_coords = raw_coords
            final_confs = raw_confs
            final_classes = raw_classes
            
        # 写入 txt 文件
        for i in range(len(final_classes)):
            class_name = CLASSES[int(final_classes[i])]
            conf_val = float(final_confs[i])
            coords = final_coords[i]
            coord_str = " ".join([f"{val:.2f}" for val in coords])
            # 输出格式符合大作业规范: image_id confidence x1 y1 x2 y2 x3 y3 x4 y4
            txt_files[class_name].write(f"{img_id} {conf_val:.6f} {coord_str}\n")
            
    # 5. 关闭所有文件
    for f in txt_files.values():
        f.close()
        
    print("\n==================================================================")
    print(f"[SUCCESS] 【{INFERENCE_MODE}】推理打包顺利完成！")
    print(f"所有结果文件已成功保存至：")
    print(f"目录: {OUTPUT_DIR}")
    print("==================================================================")

if __name__ == "__main__":
    main()
