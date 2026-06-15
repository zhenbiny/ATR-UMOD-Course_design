import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 动态定位项目根目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DATASET_DIR = os.path.join(PROJECT_DIR, "ATR-UMOD", "train")
DEV_DIR = os.path.join(PROJECT_DIR, "multimodal_detection")
OUTPUT_IMAGES_DIR = os.path.join(DEV_DIR, "data", "images")
OUTPUT_LABELS_DIR = os.path.join(DEV_DIR, "data", "labels")
SPLITS_DIR = os.path.join(DEV_DIR, "data_splits")

# 确保文件夹存在
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

# 11个类别定义
CLASSES = [
    'car', 'suv', 'van', 'bus', 'freight_car', 
    'truck', 'motorcycle', 'trailer', 'excavator', 
    'crane', 'tank_truck'
]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}

def parse_xml(xml_path):
    """解析XML标注文件"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size_elem = root.find("size")
        width = int(size_elem.find("width").text)
        height = int(size_elem.find("height").text)
        
        location = root.find("location").text if root.find("location") is not None else "0"
        
        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASS_MAP:
                continue
            
            polygon = obj.find("polygon")
            if polygon is not None:
                # 读取旋转框四角点坐标
                try:
                    coords = [
                        float(polygon.find("x1").text), float(polygon.find("y1").text),
                        float(polygon.find("x2").text), float(polygon.find("y2").text),
                        float(polygon.find("x3").text), float(polygon.find("y3").text),
                        float(polygon.find("x4").text), float(polygon.find("y4").text)
                    ]
                    import math
                    if any(math.isnan(x) for x in coords):
                        continue
                    objects.append({
                        "class_idx": CLASS_MAP[name],
                        "coords": coords
                    })
                except AttributeError:
                    # 某些标注文件可能缺失部分顶点，跳过
                    continue
        return width, height, location, objects
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None

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

def process_single_sample(img_name):
    """处理单个样本：图像通道融合 + 标签转换"""
    img_id = os.path.splitext(img_name)[0]
    
    # 路径定义
    rgb_path = os.path.join(DATASET_DIR, "images", f"{img_id}.jpg")
    ir_path = os.path.join(DATASET_DIR, "images_ir", f"{img_id}.jpg")
    xml_path = os.path.join(DATASET_DIR, "labels", f"{img_id}.xml")
    
    out_img_path = os.path.join(OUTPUT_IMAGES_DIR, f"{img_id}.jpg")
    out_lbl_path = os.path.join(OUTPUT_LABELS_DIR, f"{img_id}.txt")
    
    # 1. 解析XML
    xml_data = parse_xml(xml_path)
    if xml_data is None:
        return None
    width, height, location, objects = xml_data
    
    # 2. 转换标签格式并写入 YOLO OBB label
    with open(out_lbl_path, "w", encoding="utf-8") as f:
        for obj in objects:
            c = obj["coords"]
            # 限制坐标在 [0.0, 1.0] 范围内，防止 YOLO 报错忽略整个图像
            x1_n = min(max(c[0] / width, 0.0), 1.0)
            y1_n = min(max(c[1] / height, 0.0), 1.0)
            x2_n = min(max(c[2] / width, 0.0), 1.0)
            y2_n = min(max(c[3] / height, 0.0), 1.0)
            x3_n = min(max(c[4] / width, 0.0), 1.0)
            y3_n = min(max(c[5] / height, 0.0), 1.0)
            x4_n = min(max(c[6] / width, 0.0), 1.0)
            y4_n = min(max(c[7] / height, 0.0), 1.0)
            
            f.write(f"{obj['class_idx']} {x1_n:.6f} {y1_n:.6f} {x2_n:.6f} {y2_n:.6f} {x3_n:.6f} {y3_n:.6f} {x4_n:.6f} {y4_n:.6f}\n")
            
    # 3. 图像合并通道 (Channel Concatenation)
    rgb_img = cv2_imread_unicode(rgb_path)
    ir_img = cv2_imread_unicode(ir_path, cv2.IMREAD_GRAYSCALE)
    
    if rgb_img is None or ir_img is None:
        print(f"Error reading images for {img_id}")
        return None
        
    # 合成 3 通道 BGR 图像
    fused_img = cv2.merge([ir_img, rgb_img[:, :, 1], rgb_img[:, :, 2]])
    cv2_imwrite_unicode(out_img_path, fused_img)
    
    return {"img_id": img_id, "location": location, "img_path": out_img_path}

def main():
    print("开始扫描数据集...")
    all_image_files = [f for f in os.listdir(os.path.join(DATASET_DIR, "images")) if f.endswith(".jpg")]
    print(f"共发现 {len(all_image_files)} 对样本。")
    
    # 使用多进程加速图像融合与标注转换
    num_workers = min(cpu_count(), 8)
    print(f"启动多进程处理，工作进程数: {num_workers}...")
    
    results = []
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_single_sample, all_image_files), total=len(all_image_files)):
            if res is not None:
                results.append(res)
                
    print("数据图像通道融合与标签格式转换完成！")
    
    # 4. 基于场景位置的分层 5 折划分
    img_paths = [r["img_path"] for r in results]
    locations = [r["location"] for r in results]
    img_ids = [r["img_id"] for r in results]
    
    # 将列表转换为numpy数组便于索引
    img_paths_arr = np.array(img_paths)
    locations_arr = np.array(locations)
    
    # 使用 StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("开始进行基于 Location 属性的分层 5 折划分...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(img_paths_arr, locations_arr)):
        train_files = img_paths_arr[train_idx]
        val_files = img_paths_arr[val_idx]
        
        # 写入 txt 文件
        train_txt_path = os.path.join(SPLITS_DIR, f"train_fold{fold}.txt")
        val_txt_path = os.path.join(SPLITS_DIR, f"val_fold{fold}.txt")
        
        with open(train_txt_path, "w", encoding="utf-8") as f:
            for filepath in train_files:
                # 规范化路径分隔符为正斜杠，以防 YOLO 在 Windows/Linux 平台混淆
                f.write(filepath.replace("\\", "/") + "\n")
                
        with open(val_txt_path, "w", encoding="utf-8") as f:
            for filepath in val_files:
                f.write(filepath.replace("\\", "/") + "\n")
                
        print(f"Fold {fold} 划分完成: 训练集 {len(train_files)} 张, 验证集 {len(val_files)} 张")
        
    print("所有预处理与 K 折划分文件生成完毕！")

if __name__ == "__main__":
    main()
