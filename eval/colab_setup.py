# === Nightfall Colab Setup ===
# Paste each cell below into its own Colab cell, in order.
# Assumes GPU runtime: Runtime > Change runtime type > T4 GPU (or better).

# --- Cell 1: Mount Drive (checkpoints and dataset persist here) ---
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_ROOT = '/content/drive/MyDrive/nightfall'
os.makedirs(f'{DRIVE_ROOT}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_ROOT}/data', exist_ok=True)
print(f"Drive mounted. Nightfall root: {DRIVE_ROOT}")


# --- Cell 2: Get the Nightfall source code ---
# Replace <your-github-username> with your actual GitHub username.
# If the repo is private, you'll be prompted for a GitHub token instead
# of a password when git asks for credentials.

!git clone https://github.com/<your-github-username>/nightfall.git /content/nightfall
%cd /content/nightfall


# --- Cell 3: Install dependencies ---
%cd /content/nightfall
!pip install -e ".[data]" --quiet
!pip install scipy scikit-image scikit-learn --quiet

# Sanity check: confirm GPU is actually visible to torch before a long run
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no GPU detected -- check Runtime > Change runtime type")


# --- Cell 4: Dataset directory (no manual download needed) ---
# train_all_categories.py downloads each category automatically via
# anomalib's MVTecAD datamodule the first time it's needed -- no
# registration link required. This cell just sets up the persistent
# path on Drive so downloaded data survives a runtime disconnect.
MVTEC_DIR = f'{DRIVE_ROOT}/data/mvtec_ad'
os.makedirs(MVTEC_DIR, exist_ok=True)
print(f"Dataset will be auto-downloaded per-category into: {MVTEC_DIR}")
print("(If the official MVTec download 404s, see train_all_categories.py's "
      "ensure_mvtec_downloaded() docstring for the HuggingFace mirror fallback.)")


# --- Cell 5: Quick test run on 1-2 categories before committing to all 15 ---
!python scripts/train_all_categories.py \
    --data-root {MVTEC_DIR} \
    --output-dir {DRIVE_ROOT}/checkpoints \
    --categories bottle


# --- Cell 6: Full run, all 15 categories ---
# Safe to re-run this cell if Colab disconnects -- already-trained
# categories are skipped automatically (see train_all_categories.py).
!python scripts/train_all_categories.py \
    --data-root {MVTEC_DIR} \
    --output-dir {DRIVE_ROOT}/checkpoints


# --- Cell 7: Check progress / manifest at any point ---
import json
manifest_path = f'{DRIVE_ROOT}/checkpoints/training_manifest.json'
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    for category, info in manifest.items():
        print(f"{category:15s} {info.get('status'):10s} {info}")
else:
    print("No manifest yet -- training hasn't started.")