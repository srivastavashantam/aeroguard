# ============================================================
# AeroGuard — Statistical Anomaly Detection (Layer 1a)
#
# Approach: Per-sensor rolling z-score per flight phase
#
# IMPORTANT: Data already Z-score normalized hai
# Isliye phase thresholds bhi normalized values mein hain
#
# Normalization stats (from feature engineering):
#   IAS  : mean=47.355, std=42.682
#   AltMSL: mean=1944.660, std=1606.085
#
# Normalized phase thresholds:
#   Cruise  : IAS_norm > 0.531  AND AltMSL_norm > -0.277
#             (raw: IAS > 70 Kts AND AltMSL > 1500 Ft)
#   Taxi    : IAS_norm < -0.407 AND AltMSL_norm < -0.899
#             (raw: IAS < 30 Kts AND AltMSL < 500 Ft)
#   Takeoff : IAS_norm >= -0.407 AND AltMSL_norm < -0.277
#   Descent : remainder
#
# Z-score threshold = 3.0 (FAA 3-sigma exceedance standard)
# Flags 0.3% of normal data — industry standard
#
# Output per flight:
#   anomaly_score    : float (0-1)
#   flagged_sensors  : list
#   phase_anomalies  : dict
#   anomaly_timeline : per-timestep binary
#   top_anomalies    : top 5 by z-score
# ============================================================

import numpy as np
import json
import os
from src.logger import logger
from src.exception import AnomalyDetectionException


# ============================================================
# CONSTANTS
# ============================================================

CHANNEL_NAMES = [
    'volt1', 'volt2', 'amp1', 'amp2',
    'FQtyL', 'FQtyR', 'E1 FFlow',
    'E1 OilT', 'E1 OilP', 'E1 RPM',
    'E1 CHT1', 'E1 CHT2', 'E1 CHT3', 'E1 CHT4',
    'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4',
    'OAT', 'IAS', 'VSpd', 'NormAc', 'AltMSL',
    'CHT_spread', 'CHT_mean', 'CHT4_minus_CHT1',
    'EGT_spread', 'EGT_mean', 'EGT_CHT_divergence',
    'FQty_imbalance', 'is_cruise'
]
# ── Channels to SKIP for z-score ─────────────────────────────
# Binary ya categorical channels pe z-score meaningful nahi
# is_cruise : binary 0/1
# volt2     : highly correlated with volt1, mostly zero
SKIP_ZSCORE_CHANNELS = {
    'is_cruise',
    'volt2',
}

# Channel index lookup
CH = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}

# ── Normalized phase thresholds ───────────────────────────────
# Data Z-score normalized hai (fit on train only)
# Raw → Normalized: (raw - mean) / std
#
# IAS  : mean=47.355, std=42.682
# AltMSL: mean=1944.660, std=1606.085
#
# Cruise  raw: IAS > 70,  AltMSL > 1500
# Cruise  norm: IAS > 0.531, AltMSL > -0.277
CRUISE_IAS_NORM  =  0.531
CRUISE_ALT_NORM  = -0.277

# Taxi raw: IAS < 30, AltMSL < 500
# Taxi norm: IAS < -0.407, AltMSL < -0.899
TAXI_IAS_NORM    = -0.407
TAXI_ALT_NORM    = -0.899

# Z-score threshold = 3.0
# FAA Flight Data Monitoring standard — 3-sigma exceedance
# Flags only 0.3% of normally distributed data
# References:
#   - FAA Advisory Circular AC 120-82 (Flight Operational Quality)
#   - 3-sigma rule: Chebyshev's theorem + Normal distribution
Z_THRESHOLD = 3.0

# Minimum timesteps per phase to compute stats
MIN_PHASE_TIMESTEPS = 30


# ============================================================
# FLIGHT PHASE DETECTION
# ============================================================

def detect_flight_phases(flight: np.ndarray) -> np.ndarray:
    """
    Har timestep ke liye flight phase assign karta hai.

    Phases:
      0 = taxi    : IAS < 30 Kts, AltMSL < 500 Ft
      1 = takeoff : IAS >= 30 Kts, AltMSL < 1500 Ft
      2 = cruise  : IAS > 70 Kts, AltMSL >= 1500 Ft
      3 = descent : baaki sab

    NOTE: Normalized thresholds use kiye hain kyunki
    data already z-score normalized hai.

    Args:
        flight: (T, 31) normalized flight array

    Returns:
        np.ndarray: (T,) int8 phase labels
    """
    try:
        ias    = flight[:, CH['IAS']]
        alt    = flight[:, CH['AltMSL']]
        phases = np.full(len(flight), 3, dtype=np.int8)

        # Cruise: highest priority
        cruise_mask = (
            (ias > CRUISE_IAS_NORM) &
            (alt > CRUISE_ALT_NORM)
        )
        # Taxi
        taxi_mask = (
            (ias < TAXI_IAS_NORM) &
            (alt < TAXI_ALT_NORM)
        )
        # Takeoff: IAS >= taxi threshold, below cruise alt
        takeoff_mask = (
            (ias >= TAXI_IAS_NORM) &
            (alt <= CRUISE_ALT_NORM) &
            ~cruise_mask
        )

        phases[cruise_mask]  = 2
        phases[taxi_mask]    = 0
        phases[takeoff_mask] = 1
        # Descent = default 3

        return phases

    except Exception as e:
        raise AnomalyDetectionException(
            e, context="Flight phase detection"
        )


# ============================================================
# STATISTICAL ANOMALY DETECTOR
# ============================================================

class StatisticalAnomalyDetector:
    """
    Per-sensor, per-phase statistical anomaly detector.

    Fit   : Healthy flights ka per-sensor per-phase
            mean + std compute karo
    Detect: Naye flight pe z-score nikalo
            |z| > 3 → anomaly flagged (FAA 3-sigma standard)
    """

    def __init__(self, z_threshold: float = Z_THRESHOLD):
        self.z_threshold  = z_threshold
        self.stats        = {}
        self.is_fitted    = False
        self.sensor_names = CHANNEL_NAMES
        self.phase_names  = {
            0: 'taxi',
            1: 'takeoff',
            2: 'cruise',
            3: 'descent'
        }

    def fit(self, X_healthy: np.ndarray) -> None:
        """
        Healthy (post-maintenance) flights pe fit karo.

        Args:
            X_healthy: (N, T, C) normalized arrays
                       Sirf label=0 (safe) flights pass karo
        """
        try:
            logger.info("=" * 50)
            logger.info("STATISTICAL DETECTOR — FIT")
            logger.info("=" * 50)

            N, T, C = X_healthy.shape
            logger.info(f"Healthy flights : {N:,}")
            logger.info(f"Timesteps       : {T:,}")
            logger.info(f"Channels        : {C}")

            for phase_id, phase_name in \
                    self.phase_names.items():
                self.stats[phase_name] = {}
                phase_data = []

                for i in range(N):
                    flight = X_healthy[i]    # (T, C)
                    phases = detect_flight_phases(flight)
                    mask   = phases == phase_id
                    if mask.sum() >= MIN_PHASE_TIMESTEPS:
                        phase_data.append(flight[mask])

                if not phase_data:
                    logger.warning(
                        f"Phase {phase_name}: "
                        f"insufficient data"
                    )
                    continue

                phase_arr = np.vstack(phase_data)  # (M, C)

                for ch_idx, sensor in \
                        enumerate(self.sensor_names):
                    if sensor in SKIP_ZSCORE_CHANNELS:
                        continue
                    vals = phase_arr[:, ch_idx]
                    std  = float(vals.std())
                    # Avoid division by zero
                    if std < 1e-6:
                        std = 1.0
                    self.stats[phase_name][sensor] = {
                        'mean': float(vals.mean()),
                        'std' : std,
                    }

                logger.info(
                    f"Phase {phase_name:8s} : "
                    f"{phase_arr.shape[0]:>8,} timesteps "
                    f"from {len(phase_data):,} flights"
                )

            self.is_fitted = True
            logger.info("✅ Statistical detector fitted")

        except Exception as e:
            raise AnomalyDetectionException(
                e, context="Statistical detector fit"
            )

    def detect(self, flight: np.ndarray) -> dict:
        """
        Ek flight mein anomalies detect karta hai.

        Args:
            flight: (T, C) normalized flight array

        Returns:
            dict:
              anomaly_score    : float (0-1)
                                 fraction of flagged timesteps
              flagged_sensors  : sensors with anomalies
              phase_anomalies  : per phase flagged count
              anomaly_timeline : (T,) binary flags
              top_anomalies    : top 5 by max z-score
        """
        try:
            if not self.is_fitted:
                raise ValueError(
                    "Call fit() before detect()"
                )

            T, C       = flight.shape
            phases     = detect_flight_phases(flight)
            anom_flags = np.zeros(T, dtype=np.int8)
            phase_anom = {
                p: 0 for p in self.phase_names.values()
            }
            flagged    = set()
            top_list   = []

            for ch_idx, sensor in \
                    enumerate(self.sensor_names):
                if sensor in SKIP_ZSCORE_CHANNELS:
                    continue

                for phase_id, phase_name in \
                        self.phase_names.items():

                    if phase_name not in self.stats:
                        continue
                    if sensor not in \
                            self.stats[phase_name]:
                        continue

                    mask = phases == phase_id
                    if mask.sum() < MIN_PHASE_TIMESTEPS:
                        continue

                    mean = self.stats[phase_name][sensor]['mean']
                    std  = self.stats[phase_name][sensor]['std']

                    vals = flight[mask, ch_idx]
                    z    = np.abs((vals - mean) / std)
                    z_mask = z > self.z_threshold

                    if z_mask.any():
                        # Flag timesteps
                        ts_indices = np.where(mask)[0]
                        anom_flags[ts_indices[z_mask]] = 1
                        phase_anom[phase_name] += \
                            int(z_mask.sum())
                        flagged.add(sensor)
                        top_list.append({
                            'sensor'      : sensor,
                            'phase'       : phase_name,
                            'max_z'       : round(
                                float(z.max()), 2
                            ),
                            'pct_flagged' : round(
                                float(z_mask.mean()) * 100,
                                1
                            ),
                        })

            anomaly_score = float(anom_flags.mean())

            # Top 5 by max z-score
            top_list = sorted(
                top_list,
                key=lambda x: x['max_z'],
                reverse=True
            )[:5]

            return {
                'anomaly_score'   : round(anomaly_score, 4),
                'flagged_sensors' : sorted(list(flagged)),
                'phase_anomalies' : phase_anom,
                'anomaly_timeline': anom_flags.tolist(),
                'top_anomalies'   : top_list,
            }

        except Exception as e:
            raise AnomalyDetectionException(
                e, context="Statistical anomaly detect"
            )

    def save(self, path: str) -> None:
        """Detector stats save karo."""
        try:
            os.makedirs(
                os.path.dirname(path),
                exist_ok=True
            )
            with open(path, 'w') as f:
                json.dump({
                    'stats'      : self.stats,
                    'z_threshold': self.z_threshold,
                    'is_fitted'  : self.is_fitted,
                }, f, indent=2)
            logger.info(f"Detector saved: {path}")
        except Exception as e:
            raise AnomalyDetectionException(
                e, context="Saving detector"
            )

    def load(self, path: str) -> None:
        """Saved detector load karo."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.stats        = data['stats']
            self.z_threshold  = data['z_threshold']
            self.is_fitted    = data['is_fitted']
            self.sensor_names = CHANNEL_NAMES
            logger.info(f"Detector loaded: {path}")
        except Exception as e:
            raise AnomalyDetectionException(
                e, context="Loading detector"
            )