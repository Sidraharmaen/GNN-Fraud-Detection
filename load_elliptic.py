"""
load_elliptic.py — Download & Explore the Elliptic Bitcoin Dataset
Run: python load_elliptic.py
Requires: kaggle API credentials at ~/.kaggle/kaggle.json
"""

import os
import subprocess
import pandas as pd

# ── 1. Download from Kaggle ───────────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
]

already_downloaded = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in FILES)

if not already_downloaded:
    print("📥 Downloading Elliptic Bitcoin Dataset from Kaggle...")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "ellipticco/elliptic-data-set",
            "-p", DATA_DIR,
            "--unzip",
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("❌ Download failed:")
        print(result.stderr)
        exit(1)
    print("✅ Download complete!\n")
else:
    print("✅ Files already present — skipping download.\n")

# ── 2. Load CSVs ──────────────────────────────────────────────────────────────

# Node features (166 columns: txId + 1 time-step + 93 local + 72 aggregate features)
features_path = os.path.join(DATA_DIR, "elliptic_txs_features.csv")
df_features = pd.read_csv(features_path, header=None)

# Rename first two columns for clarity; rest are feature_1..feature_164
col_names = ["txId", "time_step"] + [f"feature_{i}" for i in range(1, df_features.shape[1] - 1)]
df_features.columns = col_names

# Labels: 1=illicit (fraud), 2=licit (legit), unknown=unlabeled
classes_path = os.path.join(DATA_DIR, "elliptic_txs_classes.csv")
df_classes = pd.read_csv(classes_path)

# Edge list: directed transaction graph
edgelist_path = os.path.join(DATA_DIR, "elliptic_txs_edgelist.csv")
df_edges = pd.read_csv(edgelist_path)

# ── 3. Summary ────────────────────────────────────────────────────────────────

SEPARATOR = "=" * 60

print(SEPARATOR)
print("📄 elliptic_txs_features.csv")
print(SEPARATOR)
print(f"Shape : {df_features.shape}  ({df_features.shape[0]:,} transactions × {df_features.shape[1]} columns)")
print("\nFirst 5 rows:")
print(df_features.head().to_string())

print("\n" + SEPARATOR)
print("🏷️  elliptic_txs_classes.csv")
print(SEPARATOR)
print(f"Shape : {df_classes.shape}")
label_map = {"1": "Illicit (Fraud)", "2": "Licit (Legit)", "unknown": "Unlabeled"}
counts = df_classes["class"].astype(str).map(label_map).value_counts()
print(f"\nLabel distribution:\n{counts.to_string()}")
print("\nFirst 5 rows:")
print(df_classes.head().to_string())

print("\n" + SEPARATOR)
print("🔗 elliptic_txs_edgelist.csv")
print(SEPARATOR)
print(f"Shape : {df_edges.shape}  ({df_edges.shape[0]:,} edges)")
print("\nFirst 5 rows:")
print(df_edges.head().to_string())

print("\n" + SEPARATOR)
print("✅ All 3 files loaded successfully!")
print(SEPARATOR)
