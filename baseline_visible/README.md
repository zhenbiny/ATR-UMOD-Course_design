# baseline_visible RGB-Only Small-Object OBB Training

This directory contains an RGB-only oriented object detection experiment for
ATR-UMOD. It is isolated from the existing multimodal fusion experiment.

The main experiment trains one model on a fixed stratified 90/10 split. It does
not use K-fold training.

## Model

The custom model is `YOLOv8s-OBB-P2-CBAM`:

- RGB visible-light images only.
- A P2/4 detection head for small aerial targets.
- The original P3/8, P4/16, and P5/32 detection heads.
- Lightweight CBAM attention blocks on shallow P2 and P3 features.
- Eleven ATR-UMOD oriented-box classes.

The model loads compatible layers from `yolov8s-obb.pt` before fine-tuning.

## Environment

Always run the scripts with the locked `study` environment:

```powershell
& 'D:\Environment\Anaconda3\envs\study\python.exe' -B .\baseline_visible\src\prepare_visible.py
```

## Workflow

1. Prepare RGB image hard links, YOLO OBB labels, and the fixed 90/10 split:

```powershell
& 'D:\Environment\Anaconda3\envs\study\python.exe' -B .\baseline_visible\src\prepare_visible.py
```

2. Validate model construction without starting training:

```powershell
& 'D:\Environment\Anaconda3\envs\study\python.exe' -B .\baseline_visible\src\validate_model.py
```

3. Train the RGB-only model for 25 epochs:

```powershell
& 'D:\Environment\Anaconda3\envs\study\python.exe' -B .\baseline_visible\src\train_single.py
```

4. Summarize completed runs:

```powershell
& 'D:\Environment\Anaconda3\envs\study\python.exe' -B .\baseline_visible\src\summarize_results.py
```

## Notes

- `prepare_visible.py` writes only inside `baseline_visible/`.
- RGB images are hard-linked where possible to avoid duplicating the dataset.
- Visible-light XML labels are converted independently instead of copying the
  previous fusion experiment's generated labels.
- The split is stratified by the source dataset's `location` attribute with a
  fixed random seed of `42`.
- Historical K-fold files or smoke-test output may still exist locally from
  earlier experiments. The main single-model workflow does not reference them.

