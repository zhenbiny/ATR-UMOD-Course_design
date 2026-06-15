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

def main():
    if not TEST_RGB_DIR or not TEST_IR_DIR:
        print("==================================================================")
        print("错误：请先在代码第 9-10 行分别填入您的可见光与红外测试集绝对路径后再运行！")
        print("==================================================================")
        return
        
    # 1. 载入最佳的单折模型 (Fold 0)
    model_path = os.path.join(WORKSPACE_DIR, "yolov8_fold0", "weights", "best.pt")
    if not os.path.exists(model_path):
        print(f"错误：未在 {model_path} 找到 Fold 0 模型权重！")
        return
        
    print(f"正在载入单折最优模型 (Fold 0)：{model_path} ...")
    model = YOLO(model_path)
    
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
    print(f"开始融合通道并执行单折模型推理 (置信度阈值: {CONF_CUTOFF}, NMS IoU: {NMS_IOU}) ...")
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
        
        # 单模型推理，让 YOLO 内部执行 GPU/CUDA OBB NMS，速度最快、精度最高
        results = model.predict(
            fused_img, 
            conf=CONF_CUTOFF, 
            iou=NMS_IOU, 
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
                    class_name = CLASSES[int(classes[i])]
                    conf_val = float(confs[i])
                    coords = obb_coords[i].flatten()
                    coord_str = " ".join([f"{val:.2f}" for val in coords])
                    # 输出格式符合大作业规范: image_id confidence x1 y1 x2 y2 x3 y3 x4 y4
                    txt_files[class_name].write(f"{img_id} {conf_val:.6f} {coord_str}\n")
            
    # 5. 关闭所有文件
    for f in txt_files.values():
        f.close()
        
    print("\n==================================================================")
    print(f"[SUCCESS] 推理打包顺利完成！")
    print(f"所有结果文件已成功保存至：")
    print(f"目录: {OUTPUT_DIR}")
    print("==================================================================")

if __name__ == "__main__":
    main()
