# Real-dataset lane detection baseline

This project provides a small U-Net binary lane-marking segmentation baseline for **real TuSimple or CULane data**. It is designed to be straightforward to run locally and to produce auditable training logs, checkpoints, pixel-mask metrics, per-image metrics and qualitative overlays.

## Important metric distinction
The script reports pixel-level mask precision, recall, F1 and IoU. These are useful engineering diagnostics, but **they are not official TuSimple accuracy or official CULane lane-instance F1**. Do not put them in a table labelled as official benchmark results. A later evaluation stage should convert outputs to each benchmark's expected format and run its official evaluator.

## 1. Environment

Python 3.10 or 3.11 is recommended. Use a virtual environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build compatible with your computer and CUDA version if you have an NVIDIA GPU. The CPU version also works, but training can be slow.

## 2. TuSimple layout

Extract the dataset so the root contains the label JSON files and `clips/`, for example:

```text
data/tusimple/
├── clips/
│   └── ...
├── label_data_0313.json
├── label_data_0531.json
└── label_data_0601.json
```

The script reads all `label_data*.json` files under the root. Each line is a JSON annotation with `raw_file`, `lanes`, and `h_samples`. It makes a deterministic image-level train/validation split from the available labelled records. For rigorous benchmark comparisons, use clip/group-aware splitting and the official test protocol rather than treating this internal validation split as a leaderboard result.

Run:

```bash
python train_lane_segmentation.py --dataset tusimple --data-root data/tusimple --output-dir runs/tusimple --epochs 20 --batch-size 4
```

## 3. CULane layout

The extracted root should contain the image sequence directories, `list/`, and each image's `.lines.txt` annotation. Typical structure:

```text
data/CULane/
├── driver_23_30frame/
├── driver_37_30frame/
├── driver_100_30frame/
├── driver_161_90frame/
├── driver_182_30frame/
├── driver_193_90frame/
└── list/
    ├── train.txt or train_gt.txt
    ├── val.txt or val_gt.txt
    └── test.txt or test_gt.txt
```

The code attempts to use `list/train.txt` (or `train_gt.txt`) and `list/val.txt` (or `val_gt.txt`). If the list names a separate `.lines.txt` file, it uses it; otherwise it infers the annotation file by replacing the image suffix with `.lines.txt`. If your extracted distribution uses different names, provide `--list-file` for training and adjust the annotation resolver in the script to match the archive exactly.

Run:

```bash
python train_lane_segmentation.py --dataset culane --data-root data/CULane --output-dir runs/culane --epochs 20 --batch-size 4
```

If no validation list is found, the script reserves a deterministic fraction of the training records for validation. For a final benchmark claim, use the official CULane train/validation/test lists and official evaluator.

## 4. Outputs

Each run directory contains:
- `history.json`: training loss and validation metrics for each epoch.
- `best_model.pt`: checkpoint selected by validation F1.
- `results.json`: final validation pixel metrics and run configuration.
- `per_image_metrics.jsonl`: one metrics record per validation image.
- `visualizations/`: overlays; green pixels indicate overlap with the ground truth and red pixels indicate predicted positives outside the ground truth.

Keep the complete run directory with the report. Record the dataset version, exact split, hardware, software versions, command, random seed, epoch count, best epoch, and any failures.

## 5. Reproducibility and limitations
- The seed is fixed by default to 42.
- Augmentation currently applies mild brightness/contrast changes and Gaussian noise to training images only.
- The output is a binary mask, not separate lane instances. Crossing or closely spaced lane lines may merge.
- The provided CULane loader supports common `.lines.txt` annotations; inspect several generated overlays before committing to a full run.
- Do not tune thresholds on the held-out test set.
- The original report's synthetic results must remain identified as synthetic; they cannot be replaced by these results until this code has actually run and the new values are captured.
