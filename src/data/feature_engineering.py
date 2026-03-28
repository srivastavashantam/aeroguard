# ============================================================
# AeroGuard — Feature Engineering Pipeline
#
# Strategy: DL approach only
# Input  : Raw sensor data (Dask) + header_labeled (pandas)
# Output : (16,359, 4096, 31) float32 numpy arrays
#
# Channels:
#   Original 23 : Raw sensor readings as-is
#   Novel    8  : Physics-informed per-timestep features
#                 CHT_spread, CHT_mean, CHT4_minus_CHT1,
#                 EGT_spread, EGT_mean, EGT_CHT_divergence,
#                 FQty_imbalance, is_cruise
#
# Pipeline:
#   Step 1 : Batch load flights from Dask (500 at a time)
#   Step 2 : Sort by timestep per flight
#   Step 3 : Physical clipping
#   Step 4 : ffill → bfill → zero fill NaN
#   Step 5 : Add 8 novel channels
#   Step 6 : Truncate/pad to 4096 timesteps
#   Step 7 : Train/Val/Test split (70/20/10)
#   Step 8 : Z-score normalize (fit on train only)
#   Step 9 : Save as numpy arrays
# ============================================================

import os
import gc
import json
import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.model_selection import train_test_split
from src.logger import logger
from src.exception import DataTransformationException


# ============================================================
# CONSTANTS
# ============================================================

# Original 23 sensor columns — exact order matters
SENSOR_COLS = [
    'volt1', 'volt2', 'amp1', 'amp2',
    'FQtyL', 'FQtyR', 'E1 FFlow',
    'E1 OilT', 'E1 OilP', 'E1 RPM',
    'E1 CHT1', 'E1 CHT2', 'E1 CHT3', 'E1 CHT4',
    'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4',
    'OAT', 'IAS', 'VSpd', 'NormAc', 'AltMSL'
]

# Novel channel names
NOVEL_COLS = [
    'CHT_spread',
    'CHT_mean',
    'CHT4_minus_CHT1',
    'EGT_spread',
    'EGT_mean',
    'EGT_CHT_divergence',
    'FQty_imbalance',
    'is_cruise',
]

# All 31 channels in final order
ALL_CHANNELS = SENSOR_COLS + NOVEL_COLS

N_TIMESTEPS = 4096
N_CHANNELS  = 31  # 23 original + 8 novel

# Physical clipping bounds — EDA confirmed values
CLIP_BOUNDS = {
    'volt1'   : (0,    35),
    'volt2'   : (0,    35),
    'amp1'    : (-100, 150),
    'amp2'    : (-50,   50),
    'FQtyL'   : (0,    35),
    'FQtyR'   : (0,    35),
    'E1 FFlow': (0,    20),
    'E1 OilT' : (0,   300),
    'E1 OilP' : (0,   115),
    'E1 RPM'  : (0,  2800),
    'E1 CHT1' : (0,   500),
    'E1 CHT2' : (0,   500),
    'E1 CHT3' : (0,   500),
    'E1 CHT4' : (0,   500),
    'E1 EGT1' : (0,  1800),
    'E1 EGT2' : (0,  1800),
    'E1 EGT3' : (0,  1800),
    'E1 EGT4' : (0,  1800),
    'OAT'     : (-60,  60),
    'IAS'     : (0,   163),
    'VSpd'    : (-3000, 3000),
    'NormAc'  : (-5,    5),
    'AltMSL'  : (-500, 20000),
}

# Cruise phase thresholds — EDA confirmed
CRUISE_IAS_MIN = 70.0
CRUISE_ALT_MIN = 1500.0

# Batch size — memory safe
BATCH_SIZE = 500


# ============================================================
# STEP 1-6: PROCESS SINGLE FLIGHT
# ============================================================

def process_flight(
    flight_id: int,
    flight_df: pd.DataFrame
) -> np.ndarray | None:
    """
    Ek flight ko process karke (4096, 31) array banata hai.

    Steps:
      1. Sort by timestep
      2. Physical clipping
      3. ffill → bfill → zero fill
      4. Add 8 novel channels
      5. Truncate last 4096 / zero pad if short

    Args:
        flight_id : Master Index
        flight_df : raw flight sensor data (unsorted pandas)

    Returns:
        np.ndarray: shape (4096, 31) float32
        None if flight has < 10 valid rows
    """
    try:
        # ── Sort by timestep ──────────────────────────────────
        df = (flight_df
              .sort_values('timestep')
              .reset_index(drop=True))

        # ── Keep only sensor columns ──────────────────────────
        available = [c for c in SENSOR_COLS if c in df.columns]
        df = df[available].copy()

        # Add missing sensor columns as zero
        for col in SENSOR_COLS:
            if col not in df.columns:
                df[col] = 0.0

        # Reorder to exact SENSOR_COLS order
        df = df[SENSOR_COLS]

        if len(df) < 10:
            return None

        # ── Physical clipping ─────────────────────────────────
        # Negative initialization values aur extreme spikes
        # remove karo — EDA confirmed
        for col, (lo, hi) in CLIP_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)

        # ── NaN treatment: ffill → bfill → zero ──────────────
        # ffill  : last known value carry forward
        # bfill  : start of flight NaN handle
        # zero   : agar poora column null ho
        df = df.ffill().bfill().fillna(0.0)

        # ── Add 8 novel channels ──────────────────────────────

        # CHT columns — use available ones
        cht_cols = ['E1 CHT1', 'E1 CHT2', 'E1 CHT3', 'E1 CHT4']
        egt_cols = ['E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4']

        cht_data = df[cht_cols].values   # (T, 4)
        egt_data = df[egt_cols].values   # (T, 4)

        # CHT_spread : max - min across 4 cylinders per timestep
        cht_spread = (cht_data.max(axis=1)
                      - cht_data.min(axis=1))

        # CHT_mean : mean across 4 cylinders per timestep
        cht_mean = cht_data.mean(axis=1)

        # CHT4_minus_CHT1 : directional gradient
        # Baffle fault indicator — EDA confirmed
        cht4_minus_cht1 = df['E1 CHT4'].values - df['E1 CHT1'].values

        # EGT_spread : max - min across 4 cylinders per timestep
        egt_spread = (egt_data.max(axis=1)
                      - egt_data.min(axis=1))

        # EGT_mean : mean across 4 cylinders per timestep
        egt_mean = egt_data.mean(axis=1)

        # EGT_CHT_divergence : EGT_mean - CHT_mean per timestep
        # Intake gasket fault signature — EDA confirmed
        egt_cht_divergence = egt_mean - cht_mean

        # FQty_imbalance : FQtyL - FQtyR per timestep
        fqty_imbalance = (df['FQtyL'].values
                          - df['FQtyR'].values)

        # is_cruise : binary flag per timestep
        # IAS > 70 Kts AND AltMSL > 1500 Ft
        is_cruise = (
            (df['IAS'].values > CRUISE_IAS_MIN) &
            (df['AltMSL'].values > CRUISE_ALT_MIN)
        ).astype(np.float32)

        # Stack novel channels — shape (T, 8)
        novel = np.stack([
            cht_spread,
            cht_mean,
            cht4_minus_cht1,
            egt_spread,
            egt_mean,
            egt_cht_divergence,
            fqty_imbalance,
            is_cruise,
        ], axis=1).astype(np.float32)

        # Original sensors — shape (T, 23)
        original = df.values.astype(np.float32)

        # Combine — shape (T, 31)
        arr = np.concatenate([original, novel], axis=1)

        # ── Truncate or pad to 4096 ───────────────────────────
        T = len(arr)
        if T >= N_TIMESTEPS:
            # Last 4096 timesteps rakho
            # Most recent readings zyada predictive hain
            arr = arr[-N_TIMESTEPS:]
        else:
            # Zero pad at start — same as researchers
            pad = np.zeros(
                (N_TIMESTEPS - T, N_CHANNELS),
                dtype=np.float32
            )
            arr = np.vstack([pad, arr])

        return arr  # (4096, 31)

    except Exception as e:
        logger.warning(
            f"Flight {flight_id}: processing failed "
            f"— {str(e)[:80]}"
        )
        return None


# ============================================================
# STEP 1-6: BATCH PROCESSING — ALL FLIGHTS
# ============================================================

def build_dataset(
    sensor_dask: dd.DataFrame,
    header_labeled: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Saare flights ko batch mein process karke
    X, y, ids arrays banata hai.

    Memory safe — 500 flights at a time load karta hai.

    Args:
        sensor_dask    : filtered Dask sensor DataFrame
        header_labeled : cleaned + labeled header

    Returns:
        tuple:
          X   : (N, 4096, 31) float32
          y   : (N,) int8
          ids : (N,) int32 — Master Index
    """
    try:
        logger.info("=" * 55)
        logger.info("DATASET BUILD SHURU")
        logger.info("=" * 55)

        all_ids = header_labeled.index.tolist()
        label_map = header_labeled['label_binary'].to_dict()

        total = len(all_ids)
        logger.info(f"Total flights to process: {total:,}")
        logger.info(f"Batch size: {BATCH_SIZE}")
        logger.info(
            f"Total batches: {total // BATCH_SIZE + 1}"
        )

        X_list   = []
        y_list   = []
        ids_list = []
        missing  = []

        # Batch processing
        batches = [
            all_ids[i:i + BATCH_SIZE]
            for i in range(0, total, BATCH_SIZE)
        ]

        for batch_idx, batch_ids in enumerate(batches):

            logger.info(
                f"Batch {batch_idx + 1}/{len(batches)} "
                f"— loading {len(batch_ids)} flights..."
            )

            # Dask se batch load karo
            batch_set = set(batch_ids)

            batch_raw = (
                sensor_dask
                .reset_index()
                .loc[lambda df:
                     df['Master Index'].isin(batch_set)]
                .compute()
                .set_index('Master Index')
            )

            # Add timestep if not present as column
            if 'timestep' not in batch_raw.columns:
                logger.warning(
                    "timestep column missing in batch"
                )

            # Process each flight in batch
            for fid in batch_ids:
                flight_rows = batch_raw[
                    batch_raw.index == fid
                ].copy()

                if len(flight_rows) == 0:
                    missing.append(fid)
                    continue

                arr = process_flight(fid, flight_rows)

                if arr is None:
                    missing.append(fid)
                    continue

                X_list.append(arr)
                y_list.append(label_map[fid])
                ids_list.append(fid)

            # Memory cleanup
            del batch_raw
            gc.collect()

            # Progress log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"  Progress: {len(X_list):,} flights "
                    f"processed, {len(missing):,} missing"
                )

        # Convert to numpy
        logger.info("Converting to numpy arrays...")

        X   = np.array(X_list,   dtype=np.float32)
        y   = np.array(y_list,   dtype=np.int8)
        ids = np.array(ids_list, dtype=np.int32)

        logger.info(f"X shape  : {X.shape}")
        logger.info(f"y shape  : {y.shape}")
        logger.info(f"Missing  : {len(missing):,} flights")
        logger.info(
            f"Memory   : {X.nbytes / 1e9:.2f} GB"
        )

        # Label distribution
        unique, counts = np.unique(y, return_counts=True)
        logger.info("Label distribution:")
        for u, c in zip(unique, counts):
            logger.info(
                f"  label={u} : {c:,} ({c/len(y)*100:.1f}%)"
            )

        return X, y, ids

    except Exception as e:
        raise DataTransformationException(
            e, context="Dataset build — batch processing"
        )


# ============================================================
# STEP 7: TRAIN/VAL/TEST SPLIT
# ============================================================

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
) -> tuple:
    """
    70/20/10 stratified split.

    Stratified — label distribution preserve karo.
    Aircraft-aware grouping ideal hai lekin
    header mein tail_num available nahi (privacy removed).
    Stratified split best available option hai.

    Args:
        X   : (N, 4096, 31)
        y   : (N,)
        ids : (N,)

    Returns:
        tuple: X_train, X_val, X_test,
               y_train, y_val, y_test,
               ids_train, ids_val, ids_test
    """
    try:
        logger.info("=" * 55)
        logger.info("TRAIN/VAL/TEST SPLIT")
        logger.info("=" * 55)

        indices = np.arange(len(X))

        # First split: 70% train, 30% temp
        idx_train, idx_temp = train_test_split(
            indices,
            test_size=0.30,
            random_state=42,
            stratify=y
        )

        # Second split: temp → 20% val, 10% test
        # 20/30 = 0.667 of temp = val
        idx_val, idx_test = train_test_split(
            idx_temp,
            test_size=0.333,
            random_state=42,
            stratify=y[idx_temp]
        )

        logger.info(
            f"Train : {len(idx_train):,} "
            f"({len(idx_train)/len(X)*100:.1f}%)"
        )
        logger.info(
            f"Val   : {len(idx_val):,} "
            f"({len(idx_val)/len(X)*100:.1f}%)"
        )
        logger.info(
            f"Test  : {len(idx_test):,} "
            f"({len(idx_test)/len(X)*100:.1f}%)"
        )

        # Label distribution per split
        for name, idx in [
            ('Train', idx_train),
            ('Val',   idx_val),
            ('Test',  idx_test)
        ]:
            y_s = y[idx]
            n1  = int(y_s.sum())
            n0  = len(y_s) - n1
            logger.info(
                f"  {name}: label=0: {n0:,} "
                f"({n0/len(y_s)*100:.1f}%)  "
                f"label=1: {n1:,} "
                f"({n1/len(y_s)*100:.1f}%)"
            )

        return (
            X[idx_train], X[idx_val],   X[idx_test],
            y[idx_train], y[idx_val],   y[idx_test],
            ids[idx_train], ids[idx_val], ids[idx_test]
        )

    except Exception as e:
        raise DataTransformationException(
            e, context="Train/Val/Test split"
        )


# ============================================================
# STEP 8: Z-SCORE NORMALIZATION
# ============================================================

def normalize_dataset(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> tuple:
    """
    Z-score normalization per channel.

    Fit SIRF train pe — val/test pe sirf apply karo.
    Data leakage prevent karna zaroori hai.

    Per channel stats:
      mean : (31,) — har channel ka train mean
      std  : (31,) — har channel ka train std

    Args:
        X_train : (N_train, 4096, 31)
        X_val   : (N_val,   4096, 31)
        X_test  : (N_test,  4096, 31)

    Returns:
        tuple: X_train_norm, X_val_norm, X_test_norm,
               train_mean, train_std
    """
    try:
        logger.info("=" * 55)
        logger.info("Z-SCORE NORMALIZATION")
        logger.info("=" * 55)
        logger.info("Fitting on TRAIN only...")

        # Reshape to (N*T, C) for per-channel stats
        N, T, C = X_train.shape
        X_2d = X_train.reshape(-1, C)

        train_mean = X_2d.mean(axis=0)   # (31,)
        train_std  = X_2d.std(axis=0)    # (31,)

        # Zero std replace with 1 — division by zero prevent
        train_std[train_std == 0] = 1.0

        # Log per-channel stats
        logger.info(
            f"  {'Channel':<25} {'Mean':>10} {'Std':>10}"
        )
        logger.info(f"  {'─'*47}")
        for i, ch in enumerate(ALL_CHANNELS):
            logger.info(
                f"  {ch:<25} "
                f"{train_mean[i]:>10.3f} "
                f"{train_std[i]:>10.3f}"
            )

        # Apply normalization
        def zscore(X, mean, std):
            return (
                (X - mean[np.newaxis, np.newaxis, :])
                / std[np.newaxis, np.newaxis, :]
            )

        logger.info("Applying normalization...")

        X_train_norm = zscore(X_train, train_mean, train_std)
        X_val_norm   = zscore(X_val,   train_mean, train_std)
        X_test_norm  = zscore(X_test,  train_mean, train_std)

        # Verify
        logger.info(
            f"X_train: mean={X_train_norm.mean():.4f} "
            f"std={X_train_norm.std():.4f}"
        )
        logger.info(
            f"X_val  : mean={X_val_norm.mean():.4f} "
            f"std={X_val_norm.std():.4f}"
        )
        logger.info(
            f"X_test : mean={X_test_norm.mean():.4f} "
            f"std={X_test_norm.std():.4f}"
        )

        return (
            X_train_norm, X_val_norm, X_test_norm,
            train_mean, train_std
        )

    except Exception as e:
        raise DataTransformationException(
            e, context="Z-score normalization"
        )


# ============================================================
# STEP 9: SAVE DATASET
# ============================================================

def save_dataset(
    X_train: np.ndarray, X_val: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray, y_val: np.ndarray,
    y_test:  np.ndarray,
    ids_train: np.ndarray, ids_val: np.ndarray,
    ids_test:  np.ndarray,
    norm_mean: np.ndarray, norm_std: np.ndarray,
    output_dir: str,
) -> None:
    """
    Saare arrays disk pe save karta hai.

    Args:
        output_dir: path where files will be saved
    """
    try:
        logger.info("=" * 55)
        logger.info("SAVING DATASET")
        logger.info(f"Output dir: {output_dir}")
        logger.info("=" * 55)

        os.makedirs(output_dir, exist_ok=True)

        save_map = {
            'X_train.npy'       : X_train,
            'X_val.npy'         : X_val,
            'X_test.npy'        : X_test,
            'y_train.npy'       : y_train,
            'y_val.npy'         : y_val,
            'y_test.npy'        : y_test,
            'ids_train.npy'     : ids_train,
            'ids_val.npy'       : ids_val,
            'ids_test.npy'      : ids_test,
            'norm_mean.npy'     : norm_mean,
            'norm_std.npy'      : norm_std,
            'channel_names.npy' : np.array(ALL_CHANNELS),
        }

        for fname, arr in save_map.items():
            fpath = os.path.join(output_dir, fname)
            np.save(fpath, arr)
            size  = os.path.getsize(fpath)
            if size >= 1e9:
                size_str = f"{size/1e9:.2f} GB"
            elif size >= 1e6:
                size_str = f"{size/1e6:.2f} MB"
            else:
                size_str = f"{size/1e3:.1f} KB"
            logger.info(f"  Saved: {fname:<22} {size_str}")

        # Dataset info JSON
        info = {
            'dataset'         : 'AeroGuard DL Dataset',
            'total_flights'   : int(len(y_train)
                                    + len(y_val)
                                    + len(y_test)),
            'train_flights'   : int(len(y_train)),
            'val_flights'     : int(len(y_val)),
            'test_flights'    : int(len(y_test)),
            'timesteps'       : N_TIMESTEPS,
            'n_channels'      : N_CHANNELS,
            'channels'        : ALL_CHANNELS,
            'original_sensors': SENSOR_COLS,
            'novel_channels'  : NOVEL_COLS,
            'array_shape'     : f'(N, {N_TIMESTEPS}, {N_CHANNELS})',
            'label_threshold' : 'date_diff <= -2 → label=1',
            'padding'         : 'zero padding at start',
            'normalization'   : 'Z-score (fit on train only)',
            'split'           : '70/20/10 stratified',
            'label_dist_train': {
                'label_0': int((y_train == 0).sum()),
                'label_1': int((y_train == 1).sum()),
            },
            'label_dist_val'  : {
                'label_0': int((y_val == 0).sum()),
                'label_1': int((y_val == 1).sum()),
            },
            'label_dist_test' : {
                'label_0': int((y_test == 0).sum()),
                'label_1': int((y_test == 1).sum()),
            },
            'norm_mean'       : norm_mean.tolist(),
            'norm_std'        : norm_std.tolist(),
        }

        info_path = os.path.join(output_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        logger.info(f"  Saved: dataset_info.json")
        logger.info("=" * 55)
        logger.info("DATASET SAVE COMPLETE ✓")
        logger.info("=" * 55)

    except Exception as e:
        raise DataTransformationException(
            e, context="Dataset save"
        )


# ============================================================
# MAIN RUNNER
# ============================================================

def run_feature_engineering(
    sensor_dask:    dd.DataFrame,
    header_labeled: pd.DataFrame,
    output_dir:     str,
) -> None:
    """
    Poora feature engineering pipeline run karta hai.

    Args:
        sensor_dask    : filtered Dask sensor DataFrame
        header_labeled : cleaned + labeled header (16,359)
        output_dir     : where to save prepared dataset
    """
    try:
        logger.info("=" * 55)
        logger.info("AEROGUARD FEATURE ENGINEERING START")
        logger.info("=" * 55)
        logger.info(f"Flights    : {len(header_labeled):,}")
        logger.info(f"Timesteps  : {N_TIMESTEPS}")
        logger.info(f"Channels   : {N_CHANNELS}")
        logger.info(
            f"Shape      : "
            f"({len(header_labeled):,}, "
            f"{N_TIMESTEPS}, {N_CHANNELS})"
        )
        logger.info(f"Output dir : {output_dir}")

        # Step 1-6: Build dataset
        X, y, ids = build_dataset(sensor_dask, header_labeled)

        # Step 7: Split
        (X_train, X_val,   X_test,
         y_train, y_val,   y_test,
         ids_train, ids_val, ids_test) = split_dataset(
            X, y, ids
        )

        # Free full X — no longer needed
        del X
        gc.collect()

        # Step 8: Normalize
        (X_train, X_val, X_test,
         norm_mean, norm_std) = normalize_dataset(
            X_train, X_val, X_test
        )

        # Step 9: Save
        save_dataset(
            X_train, X_val,   X_test,
            y_train, y_val,   y_test,
            ids_train, ids_val, ids_test,
            norm_mean, norm_std,
            output_dir
        )

        logger.info("=" * 55)
        logger.info("FEATURE ENGINEERING COMPLETE ✓")
        logger.info("=" * 55)

    except Exception as e:
        raise DataTransformationException(
            e, context="Feature engineering pipeline"
        )