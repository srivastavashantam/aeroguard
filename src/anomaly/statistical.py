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

# numpy: array slicing, z-score math, phase masking — sab yahan se
import numpy as np
# json: save/load ke liye — stats dict JSON mein serialize karta hai
# WHY JSON aur pickle nahi? Stats sirf floats aur dicts hain — JSON human-readable hai
# Debugging mein file directly open karke check kar sakte ho stats
import json
# os: output directory create karne ke liye (os.makedirs)
import os
# Custom structured logger
from src.logger import logger
# Custom exception wrapper — context string ke saath anomaly errors wrap karta hai
from src.exception import AnomalyDetectionException


# ============================================================
# CONSTANTS
# ============================================================
# CHANNEL_NAMES: (T, 31) flight array ka exact column order
# Isolation Forest file (isolation_forest.py) se same list —
# dono files synchronize rehni chahiye — ek mein change = dono mein change

CHANNEL_NAMES = [
    'volt1', 'volt2', 'amp1', 'amp2',            # Electrical: voltage aur current
    'FQtyL', 'FQtyR', 'E1 FFlow',                # Fuel: left/right tank qty + flow rate
    'E1 OilT', 'E1 OilP', 'E1 RPM',              # Engine: oil temp, oil pressure, RPM
    'E1 CHT1', 'E1 CHT2', 'E1 CHT3', 'E1 CHT4', # Cylinder Head Temps (4 cylinders)
    'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4', # Exhaust Gas Temps (4 cylinders)
    'OAT', 'IAS', 'VSpd', 'NormAc', 'AltMSL',   # Flight: outside air temp, airspeed, vertical speed, normal accel, altitude
    'CHT_spread', 'CHT_mean', 'CHT4_minus_CHT1', # Engineered: CHT imbalance features
    'EGT_spread', 'EGT_mean', 'EGT_CHT_divergence', # Engineered: EGT imbalance + combustion signal
    'FQty_imbalance', 'is_cruise'                # Engineered: fuel asymmetry + cruise flag
]

# ── Channels to SKIP for z-score ─────────────────────────────
# Binary ya categorical channels pe z-score meaningful nahi
# is_cruise : binary 0/1 — z-score ka koi statistical sense nahi
# volt2     : highly correlated with volt1, mostly zero —
#             std ≈ 0 hoga, z = val/~0 → infinity — dangerous division
# Set use kiya hai list ki jagah — O(1) lookup "sensor in SKIP_ZSCORE_CHANNELS"
SKIP_ZSCORE_CHANNELS = {
    'is_cruise',
    'volt2',
}

# Channel index lookup — naam se column number: CH['IAS'] = 19
# Magic numbers se bachao — flight[:, 19] nahi, flight[:, CH['IAS']] likho
CH = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}

# ── Normalized phase thresholds ───────────────────────────────
# CONTEXT: Dataset pipeline ne IAS aur AltMSL ko z-score normalize kiya tha
# (fit on training set only — no leakage)
# Toh raw thresholds convert karne padenge normalized space mein:
#
# Formula: norm = (raw - mean) / std
#
# IAS  : mean=47.355, std=42.682
#   Cruise threshold raw=70 Kts  → norm=(70-47.355)/42.682 =  0.531
#   Taxi   threshold raw=30 Kts  → norm=(30-47.355)/42.682 = -0.407
#
# AltMSL: mean=1944.660, std=1606.085
#   Cruise threshold raw=1500 Ft → norm=(1500-1944.660)/1606.085 = -0.277
#   Taxi   threshold raw=500 Ft  → norm=(500-1944.660)/1606.085  = -0.899
CRUISE_IAS_NORM  =  0.531   # IAS > 70 Kts normalized
CRUISE_ALT_NORM  = -0.277   # AltMSL > 1500 Ft normalized

TAXI_IAS_NORM    = -0.407   # IAS < 30 Kts normalized
TAXI_ALT_NORM    = -0.899   # AltMSL < 500 Ft normalized

# Z-score threshold = 3.0
# FAA Flight Data Monitoring (AC 120-82) standard — 3-sigma exceedance rule
# Statistical basis: Normal distribution mein 99.7% data ±3σ ke andar hota hai
# |z| > 3 → sirf 0.3% normally distributed data flag hoga = very low false positive rate
# Aviation domain mein widely accepted — agar Cessna 172 ka OilT 3σ se zyada jaaye
# toh woh genuinely unusual event hai — flag karna zaroori hai
Z_THRESHOLD = 3.0

# Minimum timesteps required per phase for meaningful stats computation
# 30 = approximately 30 seconds of data (1 Hz recording)
# Isse kam data hoga toh mean/std unreliable honge — skip karo
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
    # WHY PHASE-BASED ANALYSIS?
    # Ek hi sensor ka "normal" range alag-alag flight phases mein dramatically different hota hai:
    #   RPM cruise mein ~2400, taxi mein ~800 — ek hi threshold dono pe lagaana = wrong
    #   OilT takeoff pe high hona normal hai, cruise mein high hona abnormal hai
    # Phase-wise stats fit karne se har phase ka apna baseline hota hai —
    # zyada sensitive aur zyada accurate anomaly detection

    try:
        # IAS aur AltMSL — phase detection ke liye sirf yahi do channels chahiye
        ias    = flight[:, CH['IAS']]
        alt    = flight[:, CH['AltMSL']]
        # Default phase = 3 (descent) — "else" category
        # np.full: poora array ek value se fill karo — phir selective overwrite
        phases = np.full(len(flight), 3, dtype=np.int8)

        # PRIORITY ORDER:
        # Cruise check pehle hota hai — agar cruise conditions satisfy hain
        # toh wo timestep cruise hai, chahe takeoff conditions bhi partially satisfy karein
        cruise_mask = (
            (ias > CRUISE_IAS_NORM) &   # IAS > 70 Kts (normalized)
            (alt > CRUISE_ALT_NORM)     # AltMSL > 1500 Ft (normalized)
        )
        # Taxi: ground pe, low speed
        taxi_mask = (
            (ias < TAXI_IAS_NORM) &     # IAS < 30 Kts (normalized)
            (alt < TAXI_ALT_NORM)       # AltMSL < 500 Ft (normalized)
        )
        # Takeoff: speed build ho rahi hai lekin altitude abhi cruise se neeche hai
        # ~cruise_mask: cruise already assign nahi hua — overlap avoid karo
        takeoff_mask = (
            (ias >= TAXI_IAS_NORM) &
            (alt <= CRUISE_ALT_NORM) &
            ~cruise_mask
        )

        # Assignment order = priority order
        # Baad mein assign kiya toh overwrite hoga — cruise sabse last = highest priority
        phases[cruise_mask]  = 2
        phases[taxi_mask]    = 0
        phases[takeoff_mask] = 1
        # Phase 3 (descent) = default — already set tha np.full mein

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
    # ALGORITHM OVERVIEW:
    # ┌────────────────────────────────────────────────────────┐
    # │  FIT (offline — ek baar):                             │
    # │    Healthy flights → phase segments → per-sensor      │
    # │    mean + std per phase → stats dict                  │
    # │                                                        │
    # │  DETECT (online — har flight pe):                     │
    # │    New flight → phase detection → per-sensor z-score  │
    # │    |z| > 3.0 → flag that timestep                     │
    # │    anomaly_score = flagged_timesteps / total_timesteps │
    # └────────────────────────────────────────────────────────┘
    #
    # WHY STATISTICAL aur ML nahi?
    # Interpretability: "OilT cruise phase mein 3.8σ above normal" — maintenance engineer
    # seedha samjhega. ML model ka output sirf probability hai — black box.
    # Complementary: Statistical = per-timestep spikes, Isolation Forest = overall pattern
    # Dono milake Layer 1 banate hain — phir TCN Layer 2 hai (supervised)

    def __init__(self, z_threshold: float = Z_THRESHOLD):
        # z_threshold: anomaly flag karne ke liye z-score cutoff
        # Default Z_THRESHOLD=3.0 (FAA standard) — caller override kar sakta hai
        self.z_threshold  = z_threshold
        # stats: nested dict — stats[phase_name][sensor_name] = {'mean': x, 'std': y}
        # fit() mein populate hoga
        self.stats        = {}
        # is_fitted: guard flag — detect() fit() ke baad hi call ho
        self.is_fitted    = False
        # sensor_names: channel order reference — detect loop mein use hota hai
        self.sensor_names = CHANNEL_NAMES
        # phase_names: int → string mapping — logging aur output dict keys ke liye
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
        # CRITICAL: Sirf healthy (label=0) flights pass karo
        # Agar at-risk flights bhi include kiye toh unhealthy patterns
        # "normal" stats mein absorb ho jayenge — detector blind ho jaayega
        # Yeh assumption hai: post-maintenance flights = healthy baseline

        try:
            logger.info("=" * 50)
            logger.info("STATISTICAL DETECTOR — FIT")
            logger.info("=" * 50)

            N, T, C = X_healthy.shape
            logger.info(f"Healthy flights : {N:,}")
            logger.info(f"Timesteps       : {T:,}")
            logger.info(f"Channels        : {C}")

            # Har phase ke liye alag stats compute karo
            for phase_id, phase_name in \
                    self.phase_names.items():
                self.stats[phase_name] = {}
                # Saare healthy flights ka is phase ka data collect karo
                phase_data = []

                for i in range(N):
                    flight = X_healthy[i]           # (T, C) — ek flight
                    phases = detect_flight_phases(flight)  # (T,) phase labels
                    mask   = phases == phase_id     # boolean mask for this phase
                    # Minimum timesteps check — bahut kam data hai toh stats unreliable honge
                    if mask.sum() >= MIN_PHASE_TIMESTEPS:
                        phase_data.append(flight[mask])  # (phase_timesteps, C)

                if not phase_data:
                    # Yeh phase kisi bhi healthy flight mein nahi mila — skip
                    # Detect mein bhi skip ho jaayega (sensor not in stats check)
                    logger.warning(
                        f"Phase {phase_name}: "
                        f"insufficient data"
                    )
                    continue

                # Saare flights ka phase data ek matrix mein stack karo
                # phase_data: list of (phase_ts, C) arrays → vstack → (M, C)
                # M = total timesteps across all healthy flights for this phase
                phase_arr = np.vstack(phase_data)  # (M, C)

                # Har sensor ke liye mean aur std compute karo
                for ch_idx, sensor in \
                        enumerate(self.sensor_names):
                    # Binary/degenerate channels skip karo
                    if sensor in SKIP_ZSCORE_CHANNELS:
                        continue
                    vals = phase_arr[:, ch_idx]
                    std  = float(vals.std())
                    # Degenerate channel guard: agar std ≈ 0 (constant sensor)
                    # toh z = (x - mean) / ~0 → infinity → sab kuch flag hoga falsely
                    # std=1.0 set karo — effectively z = deviation from mean (no scaling)
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
            # Har timestep ka phase label compute karo
            phases     = detect_flight_phases(flight)
            # anom_flags: (T,) binary — 1 = anomalous timestep, 0 = normal
            # int8 dtype: memory efficient (sirf 0 ya 1 store hai)
            anom_flags = np.zeros(T, dtype=np.int8)
            # phase_anom: har phase mein total flagged timesteps count
            # Downstream visualizations ke liye useful (e.g., "cruise mein 45 anomalies")
            phase_anom = {
                p: 0 for p in self.phase_names.values()
            }
            # flagged: set of sensor names jinhone koi bhi anomaly di
            # Set: duplicate sensors avoid karna (ek sensor multiple phases mein flag ho sakta hai)
            flagged    = set()
            # top_list: individual (sensor, phase, max_z) records — baad mein sort karenge
            top_list   = []

            # Double loop: har sensor × har phase combination check karo
            # N_sensors × N_phases = ~29 × 4 = ~116 combinations — fast hai
            for ch_idx, sensor in \
                    enumerate(self.sensor_names):
                if sensor in SKIP_ZSCORE_CHANNELS:
                    continue

                for phase_id, phase_name in \
                        self.phase_names.items():

                    # Stats exist karte hain is phase ke liye?
                    # (fit mein insufficient data tha toh stats nahi honge)
                    if phase_name not in self.stats:
                        continue
                    # Is sensor ke stats hain is phase ke liye?
                    if sensor not in \
                            self.stats[phase_name]:
                        continue

                    # Is phase ke timesteps ka mask
                    mask = phases == phase_id
                    # Minimum timesteps check — bahut kam data pe z-score unreliable
                    if mask.sum() < MIN_PHASE_TIMESTEPS:
                        continue

                    # Training se computed healthy baseline
                    mean = self.stats[phase_name][sensor]['mean']
                    std  = self.stats[phase_name][sensor]['std']

                    # Is phase ke sensor values extract karo
                    vals = flight[mask, ch_idx]
                    # Z-score: kitne standard deviations is sensor ka value
                    # healthy baseline se dur hai
                    # np.abs: direction matter nahi (too high ya too low dono anomalous)
                    z    = np.abs((vals - mean) / std)
                    # FAA 3-sigma threshold apply karo
                    z_mask = z > self.z_threshold

                    if z_mask.any():
                        # np.where(mask)[0]: phase ke timestep indices original array mein
                        # z_mask: unme se kaun se anomalous hain
                        # Combined → original array mein exact anomalous positions
                        ts_indices = np.where(mask)[0]
                        anom_flags[ts_indices[z_mask]] = 1
                        # Phase mein total flagged count increment karo
                        phase_anom[phase_name] += \
                            int(z_mask.sum())
                        # Sensor ko flagged set mein daalo
                        flagged.add(sensor)
                        # Record: max_z = worst single timestep is sensor/phase combo mein
                        # pct_flagged = is phase ke kitne % timesteps flagged hue
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

            # anomaly_score = flagged timesteps / total timesteps
            # 0.0 = fully normal flight, 1.0 = every timestep anomalous
            # Simple aur interpretable metric — dashboard pe directly show kar sakte hain
            anomaly_score = float(anom_flags.mean())

            # max_z se descending sort → top 5 worst sensor/phase combos
            # Maintenance engineer ke liye most actionable information
            top_list = sorted(
                top_list,
                key=lambda x: x['max_z'],
                reverse=True
            )[:5]

            return {
                'anomaly_score'   : round(anomaly_score, 4),
                'flagged_sensors' : sorted(list(flagged)),  # sorted list — consistent output order
                'phase_anomalies' : phase_anom,
                'anomaly_timeline': anom_flags.tolist(),    # numpy → Python list (JSON serializable)
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
            # JSON mein save karo — pickle nahi
            # WHY JSON? stats = nested dicts of floats — human readable
            # Text editor mein khol ke directly inspect kar sakte hain
            # "cruise → E1 OilT → mean: 0.423, std: 0.187" — meaningful
            # indent=2: pretty-print — debugging ke liye readable formatting
            with open(path, 'w') as f:
                json.dump({
                    'stats'      : self.stats,       # nested dict: phase → sensor → {mean, std}
                    'z_threshold': self.z_threshold, # threshold jo detect mein use hoga
                    'is_fitted'  : self.is_fitted,   # True hoga — load ke baad detect() call ho sake
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
            # Saare attributes restore karo
            self.stats        = data['stats']
            self.z_threshold  = data['z_threshold']
            self.is_fitted    = data['is_fitted']
            # sensor_names JSON mein save nahi kiya — CHANNEL_NAMES se restore karo
            # ASSUMPTION: load ke waqt CHANNEL_NAMES list wahi hai jo fit ke waqt thi
            # Agar channels change kiye toh purana saved detector incompatible ho jaayega
            self.sensor_names = CHANNEL_NAMES
            logger.info(f"Detector loaded: {path}")
        except Exception as e:
            raise AnomalyDetectionException(
                e, context="Loading detector"
            )