# ============================================================
# AeroGuard — Train Statistical Anomaly Detector
#
# Yeh script:
#   1. Prepared dataset load karta hai
#   2. Sirf healthy (label=0) flights select karta hai
#   3. StatisticalAnomalyDetector fit karta hai
#   4. artifacts/ mein save karta hai
#
# Run:
#   python train_anomaly_detector.py
# ============================================================

import numpy as np
import os
from src.logger import logger
from src.anomaly.statistical import StatisticalAnomalyDetector

# ── Paths ─────────────────────────────────────────────────────
DATASET_DIR  = './data/prepared_datasets/dl_dataset'
ARTIFACT_DIR = './artifacts'

# ── Step 1: Load data ─────────────────────────────────────────
logger.info("Loading prepared dataset...")

X_train = np.load(
    os.path.join(DATASET_DIR, 'X_train.npy'),
    mmap_mode='r'
)
y_train = np.load(
    os.path.join(DATASET_DIR, 'y_train.npy')
)

logger.info(f"X_train shape : {X_train.shape}")
logger.info(f"y_train shape : {y_train.shape}")

# ── Step 2: Healthy flights select karo ───────────────────────
# label=0 = safe/post-maintenance = healthy baseline
healthy_mask  = y_train == 0
X_healthy     = X_train[healthy_mask]

logger.info(f"Total train flights  : {len(y_train):,}")
logger.info(f"Healthy flights (0)  : {X_healthy.shape[0]:,}")
logger.info(f"At-risk flights (1)  : {(y_train==1).sum():,}")

# X shape is (N, T, C) — already correct for detector
# X_train: (N, 4096, 31) — (timesteps, channels) per flight
# But numpy mmap loads as (N, 4096, 31)
# Detector expects (N, T, C) — same format ✅

logger.info(f"X_healthy shape : {X_healthy.shape}")

# ── Step 3: Fit detector ──────────────────────────────────────
detector = StatisticalAnomalyDetector(z_threshold=3.0)

# Convert mmap to regular array for fitting
# Use subset if memory is tight — 2000 flights enough
N_FIT = min(2000, X_healthy.shape[0])
logger.info(f"Using {N_FIT:,} healthy flights for fitting...")

X_fit = np.array(X_healthy[:N_FIT])  # Load into RAM
detector.fit(X_fit)

# ── Step 4: Save ──────────────────────────────────────────────
save_path = os.path.join(
    ARTIFACT_DIR, 'statistical_detector.json'
)
detector.save(save_path)

logger.info("=" * 50)
logger.info("ANOMALY DETECTOR TRAINING COMPLETE")
logger.info(f"Saved to: {save_path}")
logger.info("=" * 50)

# ── Step 5: Quick test ────────────────────────────────────────
logger.info("Quick test on one flight...")

# Test on a healthy flight
test_flight = np.array(X_healthy[0])  # (4096, 31)
result      = detector.detect(test_flight)

logger.info(f"Test (healthy flight):")
logger.info(f"  Anomaly score   : {result['anomaly_score']}")
logger.info(f"  Flagged sensors : {result['flagged_sensors']}")
logger.info(f"  Top anomalies   : {result['top_anomalies']}")

# Test on an at-risk flight
X_atrisk   = X_train[y_train == 1]
test_atrisk = np.array(X_atrisk[0])
result2     = detector.detect(test_atrisk)

logger.info(f"Test (at-risk flight):")
logger.info(f"  Anomaly score   : {result2['anomaly_score']}")
logger.info(f"  Flagged sensors : {result2['flagged_sensors']}")
logger.info(f"  Top anomalies   : {result2['top_anomalies']}")

logger.info("✅ Done")