# ============================================================
# AeroGuard — Retraining Pipeline with MLflow
#
# IMPORTANT: Model already trained hai (Colab pe TCN)
# Yeh pipeline ka kaam:
#   1. Data drift detect karo (PSI)
#   2. Current model evaluate karo (5 samples — CPU safe)
#   3. MLflow mein log karo
#   4. Promotion decision lo
#
# Naya training tabhi hoga jab explicitly zarurat ho
# (Colab pe — GPU ke saath)
#
# Run:
#   python -m src.retraining_pipeline.retrain
#
# MLflow UI:
#   mlflow ui --port 5000 → http://localhost:5000
# ============================================================

import os
import json
import datetime
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, roc_auc_score,
    recall_score, precision_score, accuracy_score
)
import mlflow
import mlflow.pytorch
import yaml

from src.logger import logger
from src.exception import AeroGuardException
from src.models.tcn_model import TCN


# ============================================================
# STEP 1 — CONFIG LOAD
# ============================================================

def load_configs() -> tuple[dict, dict]:
    """
    Dono config files load karta hai.
    Main config + MLflow config alag alag.
    """
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('configs/mlflow_config.yaml', 'r') as f:
        mlflow_config = yaml.safe_load(f)

    logger.info("✅ Configs loaded")
    logger.info(
        f"   Experiment : "
        f"{mlflow_config['mlflow']['experiment_name']}"
    )
    return config, mlflow_config


# ============================================================
# STEP 2 — DATA DRIFT DETECTION (PSI)
#
# PSI < 0.1  : No drift
# PSI 0.1-0.2: Minor drift — monitor
# PSI > 0.2  : Significant drift — retrain needed
# ============================================================

def compute_psi(
    expected: np.ndarray,
    actual  : np.ndarray,
    n_bins  : int   = 10,
    epsilon : float = 1e-6,
) -> float:
    """PSI compute karta hai — industry standard drift metric."""
    bins = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        n_bins + 1
    )
    expected_freq, _ = np.histogram(expected, bins=bins)
    actual_freq, _   = np.histogram(actual,   bins=bins)

    expected_pct = expected_freq / len(expected) + epsilon
    actual_pct   = actual_freq   / len(actual)   + epsilon

    psi = np.sum(
        (actual_pct - expected_pct) *
        np.log(actual_pct / expected_pct)
    )
    return float(psi)


def detect_data_drift(
    X_train  : np.ndarray,
    X_new    : np.ndarray,
    threshold: float = 0.2,
) -> dict:
    """
    Per-channel PSI compute karta hai.

    PRODUCTION: Naya incoming data vs training data
    HAMARE CASE: X_test ko new data maano
    """
    logger.info("Checking data drift (PSI)...")

    N_tr, T, C = X_train.shape
    sample_tr  = min(500, N_tr)
    sample_new = min(500, X_new.shape[0])

    train_flat = X_train[:sample_tr].reshape(-1, C)
    new_flat   = X_new[:sample_new].reshape(-1, C)

    channel_names = [
        'volt1','volt2','amp1','amp2',
        'FQtyL','FQtyR','E1 FFlow',
        'E1 OilT','E1 OilP','E1 RPM',
        'E1 CHT1','E1 CHT2','E1 CHT3','E1 CHT4',
        'E1 EGT1','E1 EGT2','E1 EGT3','E1 EGT4',
        'OAT','IAS','VSpd','NormAc','AltMSL',
        'CHT_spread','CHT_mean','CHT4_minus_CHT1',
        'EGT_spread','EGT_mean','EGT_CHT_divergence',
        'FQty_imbalance','is_cruise'
    ]

    psi_scores = {}
    drifted_ch = []

    for ch_idx in range(C):
        ch_name = (
            channel_names[ch_idx]
            if ch_idx < len(channel_names)
            else f"ch_{ch_idx}"
        )
        psi = compute_psi(
            train_flat[:, ch_idx],
            new_flat[:, ch_idx]
        )
        psi_scores[ch_name] = round(psi, 4)
        if psi > threshold:
            drifted_ch.append(ch_name)

    avg_psi   = float(np.mean(list(psi_scores.values())))
    drift_flag = avg_psi > threshold

    logger.info(
        f"  Avg PSI    : {avg_psi:.4f} "
        f"(threshold={threshold})"
    )
    logger.info(
        f"  Drift flag : "
        f"{'⚠️ YES' if drift_flag else '✅ NO'}"
    )
    if drifted_ch:
        logger.info(f"  Drifted    : {drifted_ch[:5]}")

    return {
        'avg_psi'   : avg_psi,
        'psi_scores': psi_scores,
        'drifted_ch': drifted_ch,
        'drift_flag': drift_flag,
        'threshold' : threshold,
    }


# ============================================================
# STEP 3 — CURRENT MODEL PERFORMANCE CHECK
#
# CPU FRIENDLY: Sirf 50 random samples
# Production mein GPU pe full evaluation hoti
# ============================================================

def check_current_model_performance(
    X_test   : np.ndarray,
    y_test   : np.ndarray,
    device   : torch.device,
    threshold: float = 0.35,
    n_samples: int   = 50,
) -> dict | None:
    """
    Current production model ka quick sanity check.
    50 random samples — CPU safe.
    """
    model_path  = 'artifacts/best_tcn.pt'
    config_path = 'artifacts/production_config.json'

    if not os.path.exists(model_path):
        logger.warning(
            "No production model found — "
            "skipping performance check"
        )
        return None

    logger.info(
        f"Checking current model on "
        f"{n_samples} random samples..."
    )

    with open(config_path, 'r') as f:
        prod_config = json.load(f)

    model = TCN(
        n_channels=31, n_filters=64,
        kernel_size=3, n_layers=8, dropout=0.1
    ).to(device)
    model.load_state_dict(torch.load(
        model_path, map_location=device,
        weights_only=True
    ))
    model.eval()

    rng     = np.random.RandomState(42)
    indices = rng.choice(
        len(y_test), n_samples, replace=False
    )

    probs  = []
    labels = []

    with torch.no_grad():
        for idx in indices:
            x = torch.tensor(
                X_test[idx], dtype=torch.float32
            ).T.unsqueeze(0).to(device)
            logit = model(x).squeeze()
            prob  = float(torch.sigmoid(logit).item())
            probs.append(prob)
            labels.append(int(y_test[idx]))

    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs >= threshold).astype(int)

    approx_f1 = float(f1_score(
        labels, preds, zero_division=0
    ))

    logger.info(
        f"  Approx F1 ({n_samples} samples): {approx_f1:.4f}"
    )
    logger.info(
        f"  Full test F1 (from config)    : "
        f"{prod_config.get('test_f1', 'N/A')}"
    )

    return {
        'approx_f1'       : approx_f1,
        'full_test_f1'    : prod_config.get('test_f1', 0),
        'full_test_auc'   : prod_config.get('test_auc', 0),
        'full_test_recall': prod_config.get(
            'test_recall_0_35', 0
        ),
        'n_samples': n_samples,
    }


# ============================================================
# DATASET + EVALUATE — CPU FRIENDLY
# ============================================================

class FlightDataset(Dataset):
    """(T, C) → (C, T) for TCN Conv1d."""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(
            self.X[idx], dtype=torch.float32
        ).T
        y = torch.tensor(
            self.y[idx], dtype=torch.float32
        )
        return x, y


def evaluate_on_samples(
    X        : np.ndarray,
    y        : np.ndarray,
    model    : nn.Module,
    criterion: nn.Module,
    device   : torch.device,
    threshold: float = 0.35,
    n_samples: int   = 50,
) -> dict:
    """
    Random N samples pe evaluation.

    CPU FRIENDLY:
      n_samples=50  → quick sanity check
      n_samples=50 → more reliable estimate
      Production   → GPU pe full dataset
    """
    rng     = np.random.RandomState(42)
    indices = rng.choice(
        len(y), min(n_samples, len(y)), replace=False
    )

    X_sample = np.array(X[indices])
    y_sample = y[indices]

    loader = DataLoader(
        FlightDataset(X_sample, y_sample),
        batch_size=4, shuffle=False, num_workers=0
    )

    model.eval()
    all_probs  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb     = xb.to(device)
            yb     = yb.to(device)
            logits = model(xb).squeeze(1)
            loss   = criterion(logits, yb)
            total_loss += loss.item()
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(yb.cpu().numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = 0.5  # Only one class in small sample

    return {
        'loss'     : total_loss / max(len(loader), 1),
        'accuracy' : float(accuracy_score(
            all_labels, preds
        )),
        'f1'       : float(f1_score(
            all_labels, preds, zero_division=0
        )),
        'recall'   : float(recall_score(
            all_labels, preds, zero_division=0
        )),
        'precision': float(precision_score(
            all_labels, preds, zero_division=0
        )),
        'auc'      : auc,
        'n_samples': len(indices),
    }


# ============================================================
# MODEL PROMOTION LOGIC
#
# PRODUCTION:
#   New model → Staging → QA verify → Production
#   Canary: 10% → 50% → 100% traffic
#
# HAMARE CASE:
#   Rule based — F1 + 0.01 → promote
#   best_tcn.pt replace hoga
# ============================================================

def should_promote_model(
    new_metrics    : dict,
    current_metrics: dict,
) -> tuple[bool, str]:
    """
    Promotion conditions:
      1. F1 > current + 0.01
      2. Recall maintain (safety critical)
      3. AUC stable (not worse by 0.5%)
    """
    new_f1      = new_metrics.get('f1', 0)
    new_recall  = new_metrics.get('recall', 0)
    new_auc     = new_metrics.get('auc', 0)
    curr_f1     = current_metrics.get('full_test_f1', 0)
    curr_recall = current_metrics.get('full_test_recall', 0)
    curr_auc    = current_metrics.get('full_test_auc', 0)

    f1_better  = new_f1     > curr_f1 + 0.01
    recall_ok  = new_recall >= curr_recall - 0.02
    auc_ok     = new_auc    >= curr_auc - 0.005

    if f1_better and recall_ok and auc_ok:
        return True, (
            f"F1: {curr_f1:.4f}→{new_f1:.4f} | "
            f"Recall: {new_recall:.4f} | "
            f"AUC: {new_auc:.4f}"
        )

    reasons = []
    if not f1_better:
        reasons.append(
            f"F1 not improved "
            f"({curr_f1:.4f}→{new_f1:.4f})"
        )
    if not recall_ok:
        reasons.append(
            f"Recall degraded "
            f"({curr_recall:.4f}→{new_recall:.4f})"
        )
    if not auc_ok:
        reasons.append(
            f"AUC degraded "
            f"({curr_auc:.4f}→{new_auc:.4f})"
        )
    return False, " | ".join(reasons)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_retraining():
    """
    Retraining pipeline — evaluate + log + promote.
    NO training here — model already trained on Colab.
    """
    try:
        logger.info("=" * 55)
        logger.info("AEROGUARD RETRAINING PIPELINE START")
        logger.info("=" * 55)

        # ── Step 1: Configs ───────────────────────────────
        config, mlflow_cfg = load_configs()

        dataset_dir  = config['data']['prepared_dataset_dir']
        train_cfg    = mlflow_cfg['training']
        mlflow_setup = mlflow_cfg['mlflow']
        threshold    = train_cfg['threshold']
        n_filters    = train_cfg['n_filters']
        kernel_size  = train_cfg['kernel_size']
        n_layers     = train_cfg['n_layers']
        dropout      = train_cfg['dropout']

        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Device : {device}")

        # ── Step 2: Load dataset ──────────────────────────
        logger.info("\n[Step 2] Loading dataset...")

        X_train = np.load(
            os.path.join(dataset_dir, 'X_train.npy'),
            mmap_mode='r'
        )
        y_train = np.load(
            os.path.join(dataset_dir, 'y_train.npy')
        )
        X_test = np.load(
            os.path.join(dataset_dir, 'X_test.npy'),
            mmap_mode='r'
        )
        y_test = np.load(
            os.path.join(dataset_dir, 'y_test.npy')
        )

        logger.info(f"  Train : {X_train.shape}")
        logger.info(f"  Test  : {X_test.shape}")

        # ── Step 3: Drift detection ───────────────────────
        logger.info("\n[Step 3] Data Drift Detection...")

        drift_results = detect_data_drift(
            X_train   = np.array(X_train[:500]),
            X_new     = np.array(X_test[:100]),
            threshold = 0.2,
        )

        if drift_results['drift_flag']:
            logger.warning(
                "⚠️  Drift detected! "
                f"PSI={drift_results['avg_psi']:.4f}"
            )
        else:
            logger.info(
                "✅ No drift. "
                f"PSI={drift_results['avg_psi']:.4f}"
            )

        # ── Step 4: Current model check ───────────────────
        logger.info(
            "\n[Step 4] Current Model Check (5 samples)..."
        )

        current_perf = check_current_model_performance(
            X_test    = np.array(X_test),
            y_test    = y_test,
            device    = device,
            threshold = threshold,
            n_samples = 50,
        )

        # ── Step 5: Loss function ─────────────────────────
        n_neg      = int((y_train == 0).sum())
        n_pos      = int((y_train == 1).sum())
        pos_weight = torch.tensor(
            [n_neg / n_pos], dtype=torch.float32
        ).to(device)
        criterion  = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        # ── Step 6: MLflow setup ──────────────────────────
        logger.info("\n[Step 6] MLflow Setup...")

        mlflow.set_tracking_uri(
            mlflow_setup['tracking_uri']
        )

        # Deleted experiment handle karo — fresh create
        exp_name = mlflow_setup['experiment_name']
        try:
            mlflow.set_experiment(exp_name)
        except Exception:
            # mlruns folder clean karo aur fresh banao
            import shutil as _shutil
            tracking_uri = mlflow_setup['tracking_uri']
            if os.path.exists(tracking_uri):
                _shutil.rmtree(tracking_uri)
                logger.info(
                    f"  Cleaned stale mlruns, "
                    f"creating fresh experiment..."
                )
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(exp_name)

        # Timestamp se unique run name
        ts       = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        run_name = f"{train_cfg['run_name']}_{ts}"
        logger.info(f"  Run : {run_name}")

        # ── Step 7: MLflow run ────────────────────────────
        logger.info("\n[Step 7] Starting MLflow Run...")

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"  Run ID : {run_id}")

            # Log params
            mlflow.log_params({
                'threshold'      : threshold,
                'n_filters'      : n_filters,
                'kernel_size'    : kernel_size,
                'n_layers'       : n_layers,
                'dropout'        : dropout,
                'device'         : str(device),
                'train_flights'  : len(y_train),
                'test_flights'   : len(y_test),
                'drift_avg_psi'  : round(
                    drift_results['avg_psi'], 4
                ),
                'drift_detected' : drift_results['drift_flag'],
                'pipeline_type'  : 'evaluate_only',
            })

            # ── Step 8: Load model ────────────────────────
            # Model already trained — sirf load karo
            logger.info(
                "\n[Step 8] Loading trained model..."
            )

            model = TCN(
                n_channels  = 31,
                n_filters   = n_filters,
                kernel_size = kernel_size,
                n_layers    = n_layers,
                dropout     = dropout,
            ).to(device)

            model.load_state_dict(torch.load(
                'artifacts/best_tcn.pt',
                map_location=device,
                weights_only=True
            ))
            logger.info(
                "  ✅ Model loaded from best_tcn.pt"
            )

            # ── Step 9: Evaluate — 50 samples ─────────────
            # CPU safe — production mein GPU pe full eval
            logger.info(
                "\n[Step 9] Evaluating (50 random samples)..."
            )

            test_m = evaluate_on_samples(
                X_test, y_test, model,
                criterion, device, threshold,
                n_samples=50
            )

            logger.info(
                "  NOTE: Approx metrics — 50 samples only."
                " GPU pe full 1635 flights chalao."
            )
            for k, v in test_m.items():
                if k != 'n_samples':
                    logger.info(f"    {k:<12}: {v:.4f}")

            # Log metrics
            mlflow.log_metrics({
                'test_f1_approx'       : test_m['f1'],
                'test_recall_approx'   : test_m['recall'],
                'test_precision_approx': test_m['precision'],
                'test_auc_approx'      : test_m['auc'],
                'test_accuracy_approx' : test_m['accuracy'],
            })

            # ── Step 10: Log model to MLflow ──────────────
            logger.info(
                "\n[Step 10] Logging model to MLflow..."
            )

            mlflow.pytorch.log_model(
                model,
                name         = "tcn_model",
                registered_model_name = (
                    mlflow_setup['registered_model_name']
                )
            )
            logger.info(
                "  ✅ Registered in MLflow Model Registry"
            )

            # ── Step 11: Promotion decision ───────────────
            # PRODUCTION: Canary 10%→50%→100%
            # HAMARE CASE: best_tcn.pt replace agar better
            logger.info(
                "\n[Step 11] Promotion Decision..."
            )

            if current_perf is not None:
                promote, reason = should_promote_model(
                    new_metrics     = test_m,
                    current_metrics = current_perf,
                )
                mlflow.log_params({
                    'promotion_decision': (
                        'PROMOTE' if promote
                        else 'KEEP_CURRENT'
                    ),
                    'promotion_reason': reason,
                })

                if promote:
                    logger.info(
                        f"  🚀 PROMOTED — {reason}"
                    )
                    logger.info(
                        "  CANARY NOTE: Production mein "
                        "10%→50%→100% traffic shift hota."
                    )
                else:
                    logger.info(
                        f"  ⏸️  KEPT current — {reason}"
                    )
            else:
                promote = True
                logger.info(
                    "  🚀 First run — model active."
                )

            # ── Step 12: Log artifacts ────────────────────
            logger.info("\n[Step 12] Logging artifacts...")

            for path in mlflow_cfg['logging']['artifacts']:
                if os.path.exists(path):
                    mlflow.log_artifact(path)
                    logger.info(f"  Logged: {path}")

            # Final summary
            logger.info("\n" + "=" * 55)
            logger.info("PIPELINE COMPLETE ✅")
            logger.info("=" * 55)
            logger.info(f"  Run ID         : {run_id}")
            logger.info(
                f"  Test F1 (approx): {test_m['f1']:.4f}"
            )
            logger.info(
                f"  Test AUC (approx): {test_m['auc']:.4f}"
            )
            logger.info(
                f"  Promoted        : {promote}"
            )
            logger.info(
                f"  Drift detected  : "
                f"{drift_results['drift_flag']}"
            )
            logger.info(
                "\n  MLflow UI: http://localhost:5000"
            )
            logger.info("=" * 55)

            return {
                'run_id'      : run_id,
                'test_metrics': test_m,
                'promoted'    : promote,
                'drift'       : drift_results,
            }

    except Exception as e:
        raise AeroGuardException(
            e, context="Retraining pipeline"
        )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    run_retraining()