import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# 11个类别定义
CLASSES = [
    'car', 'suv', 'van', 'bus', 'freight_car', 
    'truck', 'motorcycle', 'trailer', 'excavator', 
    'crane', 'tank_truck'
]

# 配色盘，为每个类别分配一个唯一的颜色 (BGR)
COLORS = [
    (0, 255, 0),    # car: 绿色
    (255, 0, 0),    # suv: 蓝色
    (0, 0, 255),    # van: 红色
    (0, 255, 255),  # bus: 黄色
    (255, 0, 255),  # freight_car: 品红
    (255, 255, 0),  # truck: 青色
    (0, 165, 255),  # motorcycle: 橙色
    (128, 0, 128),  # trailer: 紫色
    (128, 128, 0),  # excavator: 橄榄绿
    (0, 128, 128),  # crane: 深青色
    (128, 0, 0)     # tank_truck: 深蓝色
]

def cv2_imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """支持中文路径的图片读取"""
    try:
        nparr = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(nparr, flags)
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def cv2_imwrite_unicode(path, img, params=None):
    """支持中文路径的图片保存"""
    try:
        ext = os.path.splitext(path)[1]
        result, nparr = cv2.imencode(ext, img, params)
        if result:
            with open(path, "wb") as f:
                nparr.tofile(f)
            return True
        return False
    except Exception as e:
        print(f"Error writing image {path}: {e}")
        return False

def visualize_predictions(model, val_txt_path, output_dir, num_samples=10, conf_thresh=0.25):
    """从验证集随机读取图片并进行预测与 OBB 画框可视化保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 读取验证集图像路径列表
    if not os.path.exists(val_txt_path):
        print(f"错误：找不到验证集 txt 路径列表文件 {val_txt_path}")
        return
        
    with open(val_txt_path, "r", encoding="utf-8") as f:
        img_paths = [line.strip() for line in f if line.strip()]
        
    if not img_paths:
        print("警告：验证集路径列表为空！")
        return
        
    print(f"开始可视化预测... 正在选择最多 {num_samples} 张样本图像。")
    # 随机挑选一些样本
    np.random.seed(42)
    selected_paths = np.random.choice(img_paths, min(len(img_paths), num_samples), replace=False)
    
    for filepath in selected_paths:
        filename = os.path.basename(filepath)
        img_id = os.path.splitext(filename)[0]
        
        # 读取合成后的 3 通道图像
        img = cv2_imread_unicode(filepath)
        if img is None:
            continue
            
        # 复制一份用于画框
        draw_img = img.copy()
        
        # 2. 预测
        results = model.predict(filepath, conf=conf_thresh, imgsz=640, device=0, verbose=False)
        
        for result in results:
            if result.obb is not None:
                obb_coords = result.obb.xyxyxyxy.cpu().numpy() # Shape: (N, 4, 2)
                confs = result.obb.conf.cpu().numpy()
                classes = result.obb.cls.cpu().numpy()
                
                for i in range(len(classes)):
                    class_idx = int(classes[i])
                    class_name = CLASSES[class_idx]
                    score = confs[i]
                    pts = obb_coords[i].astype(np.int32) # Shape: (4, 2)
                    
                    # 选择颜色
                    color = COLORS[class_idx % len(COLORS)]
                    
                    # 3. 绘制旋转框多边形
                    cv2.polylines(draw_img, [pts.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)
                    
                    # 绘制类别文字
                    label_str = f"{class_name} {score:.2f}"
                    # 选择一个顶点放置文字
                    txt_x, txt_y = pts[0][0], pts[0][1] - 5
                    cv2.putText(draw_img, label_str, (txt_x, max(txt_y, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
        # 4. 保存可视化图片
        out_path = os.path.join(output_dir, f"val_{img_id}_pred.jpg")
        cv2_imwrite_unicode(out_path, draw_img)
        print(f"可视化图像已保存至: {out_path}")

def main():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    DEV_DIR = os.path.join(PROJECT_DIR, "multimodal_detection")
    
    default_workspace = os.path.join(DEV_DIR, "workspace")
    default_splits = os.path.join(DEV_DIR, "data_splits")
    default_output = os.path.join(default_workspace, "visualizations")

    parser = argparse.ArgumentParser(description="YOLO OBB 模型本地评估与可视化控制脚本")
    parser.add_argument("--fold", type=int, default=0, help="要评估的 Fold 编号 (0-4)")
    parser.add_argument("--workspace", type=str, default=default_workspace, help="训练工作区 runs 路径")
    parser.add_argument("--splits_dir", type=str, default=default_splits, help="数据集划分 txt 存放路径")
    parser.add_argument("--output_dir", type=str, default=default_output, help="可视化保存输出路径")
    parser.add_argument("--samples", type=int, default=10, help="随机画框可视化的样本张数 (默认 10)")
    parser.add_argument("--conf", type=float, default=0.25, help="画框置信度阈值 (默认 0.25)")
    
    args = parser.parse_args()
    
    # 查找模型路径
    model_dir = os.path.join(args.workspace, f"yolov8_fold{args.fold}")
    model_pt = os.path.join(model_dir, "weights", "best.pt")
    
    if not os.path.exists(model_pt):
        print(f"错误：在 {model_dir} 下未找到 best.pt 权重文件，请确认是否已完成 Fold {args.fold} 的训练。")
        return
        
    print(f"正在加载 Fold {args.fold} 最优模型: {model_pt}")
    model = YOLO(model_pt)
    
    # 1. 运行验证集评估 (动态生成包含本机绝对路径的配置文件)
    try:
        from train_kfold import generate_yaml
        yaml_config_path = generate_yaml(args.fold)
    except ImportError:
        yaml_config_path = os.path.join(args.workspace, "..", "config", f"fold{args.fold}.yaml")
        
    print(f"开始在 Fold {args.fold} 验证集上运行官方评估指标...")
    metrics = model.val(data=yaml_config_path, device=0, verbose=True, workers=0)
    
    print("\n-------------------- 评估结果摘要 --------------------")
    print(f"Fold {args.fold} 验证集 mAP@50 (OBB): {metrics.box.map50 * 100:.2f}%")
    print(f"Fold {args.fold} 验证集 mAP@50-95 (OBB): {metrics.box.map * 100:.2f}%")
    print("------------------------------------------------------\n")
    
    # 2. 运行样本预测与框可视化保存
    val_txt_path = os.path.join(args.splits_dir, f"val_fold{args.fold}.txt")
    visualize_predictions(model, val_txt_path, args.output_dir, num_samples=args.samples, conf_thresh=args.conf)

if __name__ == "__main__":
    main()
