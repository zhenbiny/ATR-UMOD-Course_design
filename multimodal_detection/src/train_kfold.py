import os
import argparse
from ultralytics import YOLO

# 基础路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEV_DIR = os.path.join(PROJECT_DIR, "multimodal_detection")
CONFIG_DIR = os.path.join(DEV_DIR, "config")
SPLITS_DIR = os.path.join(DEV_DIR, "data_splits")
WORKSPACE_DIR = os.path.join(DEV_DIR, "workspace")

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# 11个类别定义
CLASSES = [
    'car', 'suv', 'van', 'bus', 'freight_car', 
    'truck', 'motorcycle', 'trailer', 'excavator', 
    'crane', 'tank_truck'
]

def generate_yaml(fold):
    """自动生成对应 Fold 的 YOLO 数据配置文件"""
    yaml_path = os.path.join(CONFIG_DIR, f"fold{fold}.yaml")
    
    train_txt = os.path.join(SPLITS_DIR, f"train_fold{fold}.txt").replace("\\", "/")
    val_txt = os.path.join(SPLITS_DIR, f"val_fold{fold}.txt").replace("\\", "/")
    
    # YOLO 格式的 yaml 配置内容
    yaml_content = f"""# YOLOv8-OBB dataset config for Fold {fold}
train: {train_txt}
val: {val_txt}

nc: {len(CLASSES)}
names:
"""
    for idx, name in enumerate(CLASSES):
        yaml_content += f"  {idx}: {name}\n"
        
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(f"YAML 配置文件生成成功: {yaml_path}")
    return yaml_path

def train_fold(fold, model_name, epochs, batch, imgsz):
    """训练指定的 Fold"""
    print(f"\n==================== 开始训练 Fold {fold} ====================")
    
    fold_dir = os.path.join(WORKSPACE_DIR, f"yolov8_fold{fold}")
    results_csv = os.path.join(fold_dir, "results.csv")
    last_pt = os.path.join(fold_dir, "weights", "last.pt")
    
    # 检查已完成的 epoch 数
    completed_epochs = 0
    if os.path.exists(results_csv):
        try:
            with open(results_csv, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # 扣除表头行
                completed_epochs = len(lines) - 1
        except Exception as e:
            print(f"读取 results.csv 失败: {e}")
            
    if completed_epochs >= epochs:
        print(f"Fold {fold} 已经完成 {completed_epochs}/{epochs} 轮训练，自动跳过。")
        print(f"==================== Fold {fold} 训练完成 ====================\n")
        return
        
    # 1. 确保 YAML 配置文件存在
    yaml_path = generate_yaml(fold)
    
    # 2. 开启训练
    if os.path.exists(last_pt) and completed_epochs > 0:
        print(f"检测到断点！已完成 {completed_epochs}/{epochs} 轮。正在从 {last_pt} 恢复训练...")
        model = YOLO(last_pt)
        model.train(resume=True)
    else:
        print(f"从头开始训练 Fold {fold}，正在加载模型: {model_name}...")
        model = YOLO(model_name)
        model.train(
            data=yaml_path,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=0,                      # 使用第一个 GPU (RTX 5060)
            project=WORKSPACE_DIR,         # 保存项目的根目录
            name=f"yolov8_fold{fold}",     # 当前训练的文件夹名字
            save=True,                     # 保存模型权重
            val=True,                      # 训练时同步在 val 集上评估
            workers=4,                     # 数据加载线程数
            exist_ok=True                  # 如果文件夹存在则覆盖
        )
    print(f"==================== Fold {fold} 训练完成 ====================\n")

def main():
    parser = argparse.ArgumentParser(description="ATR-UMOD 多模态 5折交叉验证训练控制脚本")
    parser.add_argument("--fold", type=str, default="0", help="要训练的折数 (0-4，或 'all' 代表训练所有折)")
    parser.add_argument("--model", type=str, default="yolov8s-obb.pt", help="基础 YOLO OBB 模型 (如 yolov8n-obb.pt, yolov8s-obb.pt)")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数 (默认 20)")
    parser.add_argument("--batch", type=int, default=16, help="Batch Size (默认 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像分辨率大小 (默认 640)")
    
    args = parser.parse_args()
    
    if args.fold.lower() == "all":
        # 训练所有折
        print(f"【所有折训练模式启动】基础模型: {args.model}, 总轮数: {args.epochs}")
        for f in range(5):
            train_fold(f, args.model, args.epochs, args.batch, args.imgsz)
    else:
        # 仅训练指定折
        try:
            fold_idx = int(args.fold)
            if fold_idx < 0 or fold_idx > 4:
                raise ValueError
            train_fold(fold_idx, args.model, args.epochs, args.batch, args.imgsz)
        except ValueError:
            print("错误：--fold 参数必须是 0-4 之间的整数，或 'all'。")

if __name__ == "__main__":
    main()
