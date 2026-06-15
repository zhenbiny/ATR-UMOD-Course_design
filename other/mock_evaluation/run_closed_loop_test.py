import os
import shutil
import subprocess

PROJECT_DIR = r"c:\Users\17638\Desktop\NUDT\智能图像处理"
MOCK_DIR = os.path.join(PROJECT_DIR, "mock_evaluation_closed_loop")
SOURCE_DATA_DIR = os.path.join(PROJECT_DIR, "ATR-UMOD", "train")
SOURCE_EVAL_DIR = os.path.join(PROJECT_DIR, "大作业测试说明", "torch_eval")
SOURCE_INFERENCE_SCRIPT = os.path.join(PROJECT_DIR, "multimodal_detection", "src", "run_inference_final.py")

MOCK_DATASET_DIR = os.path.join(MOCK_DIR, "mock_test_dataset")
MOCK_EVAL_DIR = os.path.join(MOCK_DIR, "torch_eval")
PYTHON_EXEC = r"D:\Environment\Anaconda3\envs\study\python.exe"

def setup_workspace():
    print("=== Step 1: Setting up mock evaluation workspace directories ===")
    os.makedirs(MOCK_DIR, exist_ok=True)
    os.makedirs(os.path.join(MOCK_DATASET_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(MOCK_DATASET_DIR, "images_ir"), exist_ok=True)
    
    # Copy torch_eval folder if it doesn't exist, or clear out gt and pred if it does
    if os.path.exists(MOCK_EVAL_DIR):
        print("Clearing existing mock torch_eval folder...")
        shutil.rmtree(MOCK_EVAL_DIR)
    
    print("Copying torch_eval evaluator...")
    shutil.copytree(SOURCE_EVAL_DIR, MOCK_EVAL_DIR)
    
    # Clear gt and pred directories
    gt_dir = os.path.join(MOCK_EVAL_DIR, "gt")
    pred_dir = os.path.join(MOCK_EVAL_DIR, "pred")
    for d in (gt_dir, pred_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

def copy_mock_data():
    print("=== Step 2: Selecting and copying 1000 pairs for mock evaluation ===")
    src_images_dir = os.path.join(SOURCE_DATA_DIR, "images")
    src_ir_dir = os.path.join(SOURCE_DATA_DIR, "images_ir")
    src_labels_dir = os.path.join(SOURCE_DATA_DIR, "labels")
    
    # Get sorted image files
    all_images = sorted([f for f in os.listdir(src_images_dir) if f.endswith(".jpg")])
    if len(all_images) < 1000:
        print(f"Warning: Only {len(all_images)} pairs available. Using all available pairs.")
        selected_images = all_images
    else:
        selected_images = all_images[:1000]
        
    print(f"Copying {len(selected_images)} pairs (RGB + IR + XML)...")
    for img_name in selected_images:
        img_id = os.path.splitext(img_name)[0]
        
        # Paths
        src_rgb = os.path.join(src_images_dir, f"{img_id}.jpg")
        src_ir = os.path.join(src_ir_dir, f"{img_id}.jpg")
        src_xml = os.path.join(src_labels_dir, f"{img_id}.xml")
        
        dst_rgb = os.path.join(MOCK_DATASET_DIR, "images", f"{img_id}.jpg")
        dst_ir = os.path.join(MOCK_DATASET_DIR, "images_ir", f"{img_id}.jpg")
        dst_xml = os.path.join(MOCK_EVAL_DIR, "gt", f"{img_id}.xml")
        
        # Copy files
        if os.path.exists(src_rgb):
            shutil.copy2(src_rgb, dst_rgb)
        if os.path.exists(src_ir):
            shutil.copy2(src_ir, dst_ir)
        if os.path.exists(src_xml):
            shutil.copy2(src_xml, dst_xml)
            
    print(f"Successfully copied {len(selected_images)} mock pairs.")

def prepare_inference_script():
    print("=== Step 3: Preparing customized run_inference.py ===")
    with open(SOURCE_INFERENCE_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Replace TEST_DIR path
    # Target line: TEST_DIR = r"" -> TEST_DIR = r"..."
    old_test_line = 'TEST_DIR = r""'
    new_test_line = f'TEST_DIR = r"{MOCK_DATASET_DIR}"'
    if old_test_line in content:
        content = content.replace(old_test_line, new_test_line)
    else:
        # Fallback search/replace
        content = content.replace('TEST_DIR = ""', f'TEST_DIR = r"{MOCK_DATASET_DIR}"')
        
    # Replace OUTPUT_DIR path to go into torch_eval/pred
    old_output_line = 'OUTPUT_DIR = os.path.join(DEV_DIR, "submission_results")'
    new_output_line = f'OUTPUT_DIR = r"{os.path.join(MOCK_EVAL_DIR, "pred")}"'
    content = content.replace(old_output_line, new_output_line)
    
    # Save as run_inference.py in mock evaluation folder
    inference_dst_path = os.path.join(MOCK_DIR, "run_inference.py")
    with open(inference_dst_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Inference script customized and saved to: {inference_dst_path}")

def run_inference():
    print("=== Step 4: Running 5-fold ensemble inference on mock dataset ===")
    script_path = os.path.join(MOCK_DIR, "run_inference.py")
    
    # Run python run_inference.py
    cmd = [PYTHON_EXEC, script_path]
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=MOCK_DIR, capture_output=True)
    
    # Decode robustly
    stdout = result.stdout.decode('utf-8', errors='ignore')
    stderr = result.stderr.decode('utf-8', errors='ignore')
    
    import sys
    enc = sys.stdout.encoding or 'gbk'
    for line in stdout.splitlines():
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode(enc, errors='replace').decode(enc))
            
    if result.returncode != 0:
        print("Error during inference execution:")
        print(stderr)
        return False
    return True

def run_evaluation():
    print("=== Step 5: Running torch_eval/Eval.py to calculate mAP scores ===")
    eval_script = os.path.join(MOCK_EVAL_DIR, "Eval.py")
    
    # Run python Eval.py
    cmd = [PYTHON_EXEC, eval_script]
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=MOCK_EVAL_DIR, capture_output=True)
    
    # Decode robustly
    stdout = result.stdout.decode('utf-8', errors='ignore')
    stderr = result.stderr.decode('utf-8', errors='ignore')
    
    import sys
    enc = sys.stdout.encoding or 'gbk'
    for line in stdout.splitlines():
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode(enc, errors='replace').decode(enc))
            
    if result.returncode != 0:
        print("Error during evaluation execution:")
        print(stderr)
        return False
    return True

def main():
    print("=================== Closed Loop Test Pipeline Starting ===================")
    setup_workspace()
    copy_mock_data()
    # prepare_inference_script() # 注释掉，避免覆盖已优化的推理逻辑
    if run_inference():
        run_evaluation()
    print("=================== Closed Loop Test Pipeline Finished ===================")

if __name__ == "__main__":
    main()
