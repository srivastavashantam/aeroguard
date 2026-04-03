# ============================================================
# AeroGuard — Statistical Anomaly Detection (Layer 1a)
#
# FIXES v2:
#   1. Weighted anomaly score — sirf maintenance-relevant
#      sensors count karo, cruise phase ko 2x weight
#   2. Persistence threshold — sensor tabhi flagged maano
#      jab >= 5% timesteps us phase mein flagged hon
#   3. Pilot-action sensors display se remove karo
#
# Normalized phase thresholds (Z-score normalized data):
#   IAS  : mean=47.355, std=42.682
#   AltMSL: mean=1944.660, std=1606.085
#
#   Cruise  raw: IAS > 70,  AltMSL > 1500
#   Cruise  norm: IAS > 0.531, AltMSL > -0.277
#   Taxi    raw: IAS < 30, AltMSL < 500
#   Taxi    norm: IAS < -0.407, AltMSL < -0.899
#
# Z-score threshold = 2.5 (FAA 2.5-sigma standard)
# ============================================================
# v1 vs v2 key differences:
#   v1: anomaly_score = simple fraction of flagged timesteps (any sensor)
#       Problem: NormAc, VSpd jaise pilot-action sensors bhi count ho rahe the
#                1-2 random spikes bhi score inflate kar rahe the
#   v2: anomaly_score = weighted sum (maintenance sensors only, cruise 2x)
#       + persistence threshold (>= 5% of phase timesteps flagged tabhi count)
#       Result: zyada signal, kam noise — maintenance-meaningful score

import numpy as np
import json
import os
from src.logger import logger
from src.exception import AnomalyDetectionException


# ============================================================
# CONSTANTS
# ============================================================

# Exact same channel order as all other AeroGuard files —
# koi bhi reordering = wrong channel ko wrong index assign hoga
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

# CH: naam → column index lookup — magic numbers se bachao
CH = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}

# ── Normalized phase thresholds ───────────────────────────────
# Raw thresholds → normalized: (raw - mean) / std
# IAS  mean=47.355, std=42.682
# AltMSL mean=1944.660, std=1606.085
CRUISE_IAS_NORM  =  0.531   # (70 - 47.355) / 42.682
CRUISE_ALT_NORM  = -0.277   # (1500 - 1944.660) / 1606.085
TAXI_IAS_NORM    = -0.407   # (30 - 47.355) / 42.682
TAXI_ALT_NORM    = -0.899   # (500 - 1944.660) / 1606.085

# Z-score threshold — FAA AC 120-82 Flight Data Monitoring standard
# |z| > 2.5 → 0.6% false positive rate on normally distributed data
Z_THRESHOLD = 2.5

# Minimum timesteps required per phase to compute meaningful stats
# 30 ≈ 30 seconds of data — isse kam hoga toh mean/std unreliable
MIN_PHASE_TIMESTEPS = 30

# ── Channels to SKIP for z-score ─────────────────────────────
# is_cruise : binary 0/1 — z-score meaningless
# volt2     : mostly zero, std ≈ 0 → z = val/~0 = infinity → false alarms
SKIP_ZSCORE_CHANNELS = {
    'is_cruise', 'volt2',
}

# ── Maintenance-relevant sensors ─────────────────────────────
# v2 FIX #1: sirf yeh sensors weighted anomaly score mein contribute karenge
# WHY? v1 mein NormAc (normal acceleration), VSpd (vertical speed), IAS
# bhi score mein count ho rahe the — yeh pilot behavior se driven hain,
# mechanical failure ka signal nahi hain
# Maintenance engineer ko actionable sensors chahiye — pilot inputs nahi
MAINTENANCE_SENSORS = {
    'E1 OilT', 'E1 OilP', 'E1 RPM',              # Engine core
    'E1 CHT1', 'E1 CHT2', 'E1 CHT3', 'E1 CHT4',  # Cylinder head temps
    'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4',  # Exhaust gas temps
    'E1 FFlow', 'FQtyL', 'FQtyR',                 # Fuel system
    'CHT_spread', 'CHT_mean', 'CHT4_minus_CHT1',  # Engineered CHT features
    'EGT_spread', 'EGT_mean', 'EGT_CHT_divergence', # Engineered EGT features
    'FQty_imbalance',                              # Fuel asymmetry
}

# ── Pilot-action sensors — display se hide karo ───────────────
# v2 FIX #3: yeh sensors flagged_sensors list mein nahi aayenge
# NormAc : normal acceleration — maneuvers se naturally high hoga
# VSpd   : vertical speed — climb/descent mein naturally unusual
# IAS    : airspeed — pilot throttle/pitch se driven
# AltMSL : altitude — flight profile se driven
# OAT    : outside air temperature — environment, not aircraft health
# volt1, amp1, amp2: electrical — pilot avionics load se driven
# Maintenance report mein dikhne se confusion hoga — filter karo
PILOT_ACTION_SENSORS = {
    'NormAc', 'VSpd', 'IAS', 'AltMSL',
    'OAT', 'volt1', 'amp1', 'amp2',
}

# ── Phase weights for anomaly score ──────────────────────────
# v2 FIX #1 (part 2): cruise phase ko 2x weight kyun?
# Cruise mein engine steady-state operations mein hota hai —
# temperature, pressure, RPM sab stable hone chahiye
# Cruise mein koi bhi anomaly = genuinely unusual = zyada concerning
# Taxi mein engine warmup pe hi hota hai — 0.5x weight (less critical)
PHASE_WEIGHTS = {
    'taxi'    : 0.5,  # Warmup phase — kuch variation normal hai
    'takeoff' : 1.0,  # Normal weight
    'cruise'  : 2.0,  # Steady state — anomalies yahan zyada meaningful
    'descent' : 1.0,  # Normal weight
}

# ── Persistence threshold ─────────────────────────────────────
# v2 FIX #2: sensor tabhi flagged maano jab >= 5% timesteps flagged hon
# WHY? v1 mein ek bhi spike (1 timestep) poore sensor ko flag karta tha
# Real maintenance issues persistent hote hain —
# 1-2 random outliers sensor noise ya turbulence ho sakta hai
# 5% of 4096 timesteps ≈ 205 consecutive/scattered anomalous readings —
# yeh genuinely concerning pattern hai, not random spike
PERSISTENCE_THRESHOLD = 0.05


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

    NOTE: Normalized thresholds — data already z-score normalized.
    """
    # v1 se same — phase detection logic nahi badla
    # Sirf detect() mein weighting logic aur persistence check add hua hai
    try:
        ias    = flight[:, CH['IAS']]
        alt    = flight[:, CH['AltMSL']]
        # Default phase = 3 (descent) — selective overwrite
        phases = np.full(len(flight), 3, dtype=np.int8)

        cruise_mask = (
            (ias > CRUISE_IAS_NORM) &
            (alt > CRUISE_ALT_NORM)
        )
        taxi_mask = (
            (ias < TAXI_IAS_NORM) &
            (alt < TAXI_ALT_NORM)
        )
        # Takeoff: speed building lekin altitude cruise se neeche
        # ~cruise_mask: overlap avoid karo
        takeoff_mask = (
            (ias >= TAXI_IAS_NORM) &
            (alt <= CRUISE_ALT_NORM) &
            ~cruise_mask
        )

        # Priority: cruise > taxi > takeoff > descent (default)
        phases[cruise_mask]  = 2
        phases[taxi_mask]    = 0
        phases[takeoff_mask] = 1

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

    v2 improvements:
      - Weighted anomaly score (maintenance sensors only)
      - Persistence threshold (>= 5% flagged to count)
      - Pilot-action sensors excluded from display
    """

    def __init__(self, z_threshold: float = Z_THRESHOLD):
        self.z_threshold  = z_threshold
        self.stats        = {}           # nested: phase → sensor → {mean, std}
        self.is_fitted    = False        # guard: detect() fit() ke baad hi call ho
        self.sensor_names = CHANNEL_NAMES
        self.phase_names  = {
            0: 'taxi',
            1: 'takeoff',
            2: 'cruise',
            3: 'descent'
        }

    def fit(self, X_healthy: np.ndarray) -> None:
        """
        Healthy flights pe fit karo.

        Args:
            X_healthy: (N, T, C) normalized arrays
        """
        # fit() logic v1 se same — baseline stats compute karo
        # Weighted scoring sirf detect() mein apply hota hai
        # Stats ek baar compute karo, har detection pe reuse karo
        try:
            logger.info("=" * 50)
            logger.info("STATISTICAL DETECTOR — FIT v2")
            logger.info("=" * 50)

            N, T, C = X_healthy.shape
            logger.info(f"Healthy flights : {N:,}")
            logger.info(f"Timesteps       : {T:,}")
            logger.info(f"Channels        : {C}")

            for phase_id, phase_name in \
                    self.phase_names.items():
                self.stats[phase_name] = {}
                phase_data = []

                # Har healthy flight se is phase ka data collect karo
                for i in range(N):
                    flight = X_healthy[i]
                    phases = detect_flight_phases(flight)
                    mask   = phases == phase_id
                    if mask.sum() >= MIN_PHASE_TIMESTEPS:
                        phase_data.append(flight[mask])

                if not phase_data:
                    logger.warning(
                        f"Phase {phase_name}: insufficient data"
                    )
                    continue

                # List of arrays → single (M, C) matrix
                phase_arr = np.vstack(phase_data)

                for ch_idx, sensor in \
                        enumerate(self.sensor_names):
                    if sensor in SKIP_ZSCORE_CHANNELS:
                        continue
                    vals = phase_arr[:, ch_idx]
                    std  = float(vals.std())
                    # Degenerate channel guard: std ≈ 0 → z = val/~0 = infinity
                    if std < 1e-6:
                        std = 1.0
                    self.stats[phase_name][sensor] = {
                        'mean': float(vals.mean()),
                        'std' : std,
                    }

                logger.info(
                    f"Phase {phase_name:8s} : "
                    f"{phase_arr.shape[0]:>8,} timesteps"
                )

            self.is_fitted = True
            logger.info("✅ Statistical detector v2 fitted")

        except Exception as e:
            raise AnomalyDetectionException(
                e, context="Statistical detector fit"
            )

    def detect(self, flight: np.ndarray) -> dict:
        """
        Flight mein anomalies detect karta hai.

        v2 changes:
          - Weighted score (maintenance sensors + phase weight)
          - Persistence threshold applied
          - Pilot sensors excluded from flagged list

        Returns:
            dict:
              anomaly_score    : float (0-1) weighted score
              flagged_sensors  : maintenance sensors only
              phase_anomalies  : per phase weighted count
              anomaly_timeline : (T,) binary flags
              top_anomalies    : top 5 persistent anomalies
        """
        try:
            if not self.is_fitted:
                raise ValueError("Call fit() before detect()")

            T, C       = flight.shape
            phases     = detect_flight_phases(flight)
            # anom_flags: (T,) binary — visualization ke liye (timeline plot)
            # NOTE: yeh raw flags hain — persistence/weighting se unaffected
            # Streamlit timeline chart mein har anomalous timestep dikhana hai
            anom_flags = np.zeros(T, dtype=np.int8)

            # phase_anom: float now (v1 mein int tha) — weighted counts store karo
            phase_anom = {
                p: 0.0
                for p in self.phase_names.values()
            }

            flagged   = set()  # maintenance sensors jo persistent anomaly dikha rahe hain
            top_list  = []     # (sensor, phase, max_z, pct_flagged) records

            # v2 weighted score ke liye running numerator + denominator
            # Final score = weighted_sum / weighted_total
            # WHY running sum aur ek baar compute nahi?
            # Har sensor-phase combo ka apna weight hai — aggregation loop mein hi hoti hai
            weighted_sum   = 0.0
            weighted_total = 0.0

            # Double loop: har sensor × har phase
            for ch_idx, sensor in \
                    enumerate(self.sensor_names):

                if sensor in SKIP_ZSCORE_CHANNELS:
                    continue

                # v2: maintenance sensor hai ya nahi — weighted score ke liye check
                is_maintenance = sensor in MAINTENANCE_SENSORS

                for phase_id, phase_name in \
                        self.phase_names.items():

                    if phase_name not in self.stats:
                        continue
                    if sensor not in self.stats[phase_name]:
                        continue

                    mask    = phases == phase_id
                    n_phase = mask.sum()
                    if n_phase < MIN_PHASE_TIMESTEPS:
                        continue

                    # Healthy baseline se z-score compute karo
                    mean = self.stats[phase_name][sensor]['mean']
                    std  = self.stats[phase_name][sensor]['std']
                    vals = flight[mask, ch_idx]
                    z    = np.abs((vals - mean) / std)
                    z_mask = z > self.z_threshold

                    # v2 FIX #2: Persistence check
                    # pct_flagged = is phase mein kitne fraction timesteps anomalous hain
                    # PERSISTENCE_THRESHOLD = 0.05 → 5% minimum
                    pct_flagged   = float(z_mask.mean())
                    is_persistent = (
                        pct_flagged >= PERSISTENCE_THRESHOLD
                    )

                    # anom_flags: ALWAYS update — persistence se independent
                    # Timeline visualization mein saare spikes dikhne chahiye
                    # Sirf score + flagged_sensors mein persistence filter apply hoga
                    if z_mask.any():
                        ts_indices = np.where(mask)[0]
                        anom_flags[ts_indices[z_mask]] = 1

                    # v2: Score + flagged list mein sirf tab add karo jab:
                    #   1. is_persistent = True (>= 5% flagged)
                    #   2. is_maintenance = True (pilot sensor nahi)
                    if is_persistent and is_maintenance:
                        # Phase weight: cruise = 2x, taxi = 0.5x
                        phase_weight = PHASE_WEIGHTS[phase_name]
                        # Weighted contribution: pct_flagged × phase_weight
                        # Matlab: cruise mein 10% flagged = taxi mein 10% flagged se 4x zyada score
                        weighted_sum += (
                            pct_flagged * phase_weight
                        )
                        # Denominator mein sirf phase weight add karo — normalization ke liye
                        weighted_total += phase_weight

                        # Phase anomaly count: weighted integer count
                        phase_anom[phase_name] += (
                            int(z_mask.sum()) * phase_weight
                        )

                        # v2 FIX #3: PILOT_ACTION_SENSORS bhi filter karo
                        # MAINTENANCE_SENSORS mein kuch overlap ho sakta hai —
                        # double check karo ki pilot sensor toh nahi hai
                        if sensor not in PILOT_ACTION_SENSORS:
                            flagged.add(sensor)
                            top_list.append({
                                'sensor'      : sensor,
                                'phase'       : phase_name,
                                'max_z'       : round(
                                    float(z.max()), 2
                                ),
                                # pct_flagged × 100: percentage format for display
                                'pct_flagged' : round(
                                    pct_flagged * 100, 1
                                ),
                            })

            # ── Weighted anomaly score compute karo ──────────────
            # weighted_sum / weighted_total = weighted average of pct_flagged
            # weighted_total = 0 → koi bhi maintenance sensor persistent anomaly nahi dikha
            # min(score, 1.0): theoretically > 1 aa sakta hai agar multiple sensors
            # same phase mein high pct_flagged dikhayein — clip karo
            if weighted_total > 0:
                anomaly_score = float(
                    weighted_sum / weighted_total
                )
                anomaly_score = float(
                    min(anomaly_score, 1.0)
                )
            else:
                anomaly_score = 0.0

            # phase_anom float → int: display ke liye (weighted float counts readable nahi)
            # Fractional weighted counts ko nearest int mein round karo
            phase_anom_int = {
                p: int(v)
                for p, v in phase_anom.items()
            }

            # max_z se descending sort → top 5 worst sensor-phase combos
            top_list = sorted(
                top_list,
                key=lambda x: x['max_z'],
                reverse=True
            )[:5]

            return {
                'anomaly_score'   : round(anomaly_score, 4),
                'flagged_sensors' : sorted(list(flagged)),   # sorted: consistent order
                'phase_anomalies' : phase_anom_int,
                'anomaly_timeline': anom_flags.tolist(),     # numpy → list (JSON serializable)
                'top_anomalies'   : top_list,
            }

        except Exception as e:
            raise AnomalyDetectionException(
                e, context="Statistical anomaly detect v2"
            )

    def save(self, path: str) -> None:
        # JSON save — same as v1
        # Stats (floats + strings) human-readable JSON mein fit hote hain
        try:
            os.makedirs(
                os.path.dirname(path), exist_ok=True
            )
            with open(path, 'w') as f:
                json.dump({
                    'stats'      : self.stats,        # per-phase per-sensor mean + std
                    'z_threshold': self.z_threshold,  # detect() mein use hoga
                    'is_fitted'  : self.is_fitted,    # load ke baad detect() directly call ho sake
                }, f, indent=2)
            logger.info(f"Detector saved: {path}")
        except Exception as e:
            raise AnomalyDetectionException(
                e, context="Saving detector"
            )

    def load(self, path: str) -> None:
        # JSON load — stats restore karo
        # PHASE_WEIGHTS, PERSISTENCE_THRESHOLD, MAINTENANCE_SENSORS:
        # yeh module-level constants hain — JSON mein save nahi kiye
        # ASSUMPTION: load ke waqt same constants hain jo fit ke waqt the
        # Agar v2 constants change karo (e.g., PERSISTENCE_THRESHOLD = 0.03),
        # detector ko re-fit karna padega — saved JSON outdated ho jaayega
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.stats        = data['stats']
            self.z_threshold  = data['z_threshold']
            self.is_fitted    = data['is_fitted']
            # sensor_names JSON mein nahi — CHANNEL_NAMES se restore
            self.sensor_names = CHANNEL_NAMES
            logger.info(f"Detector loaded: {path}")
        except Exception as e:
            raise AnomalyDetectionException(
                e, context="Loading detector"
            )