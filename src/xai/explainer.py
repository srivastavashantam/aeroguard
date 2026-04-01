# ============================================================
# AeroGuard — XAI Engine (Explainability)
#
# Approach: SHAP GradientExplainer for TCN
#
# Kyun SHAP?
#   - Model-agnostic explanation framework
#   - GradientExplainer: deep learning ke liye
#     backpropagation se feature attributions nikalta hai
#   - Output: per-channel, per-timestep importance scores
#
# 3 Types of Explanations:
#   1. Channel Importance  : kaunse sensors important the
#   2. Temporal Importance : kaunse timesteps important the
#   3. Plain Language      : human readable summary
#
# Usage:
#   from src.xai.explainer import AeroGuardExplainer
#   explainer = AeroGuardExplainer(model, background_data)
#   explanation = explainer.explain(flight_array)
# ============================================================

# numpy: array math — gradient × input, axis-wise mean, argsort
import numpy as np
# torch: forward pass aur backward pass ke liye — gradients w.r.t. input nikalne ke liye
import torch
# torch.nn: model type hint ke liye (nn.Module)
import torch.nn as nn
# Custom structured logger
from src.logger import logger
# Custom exception wrapper — context string ke saath XAI errors wrap karta hai
from src.exception import AeroGuardException

# ============================================================
# CHANNEL NAMES
# ============================================================
# Yeh list exact same order mein honi chahiye jis order mein
# (T, 31) flight array ke columns arranged hain —
# isolation_forest.py aur statistical.py se same list
# Koi bhi reordering = wrong channel ko wrong importance assign hogi

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

# Human readable sensor names for plain language
# CHANNEL_NAMES mein short codes hain (e.g., 'E1 OilT') —
# maintenance engineers ke liye full description zaroori hai
# Dict lookup O(1) — sensor code → readable string
# .get(key, key) fallback: agar sensor description nahi hai toh raw name return karo
SENSOR_DESCRIPTIONS = {
    'E1 OilT'            : 'Engine oil temperature',
    'E1 OilP'            : 'Engine oil pressure',
    'E1 RPM'             : 'Engine RPM',
    'E1 CHT1'            : 'Cylinder 1 head temperature',
    'E1 CHT2'            : 'Cylinder 2 head temperature',
    'E1 CHT3'            : 'Cylinder 3 head temperature',
    'E1 CHT4'            : 'Cylinder 4 head temperature',
    'E1 EGT1'            : 'Cylinder 1 exhaust temperature',
    'E1 EGT2'            : 'Cylinder 2 exhaust temperature',
    'E1 EGT3'            : 'Cylinder 3 exhaust temperature',
    'E1 EGT4'            : 'Cylinder 4 exhaust temperature',
    'E1 FFlow'           : 'Fuel flow rate',
    'FQtyL'              : 'Left fuel tank quantity',
    'FQtyR'              : 'Right fuel tank quantity',
    'volt1'              : 'Main bus voltage',
    'amp1'               : 'Main battery current',
    'CHT_spread'         : 'Cylinder temperature spread',
    'CHT_mean'           : 'Average cylinder temperature',
    'CHT4_minus_CHT1'    : 'Cylinder temperature gradient',
    'EGT_spread'         : 'Exhaust temperature spread',
    'EGT_mean'           : 'Average exhaust temperature',
    'EGT_CHT_divergence' : 'Exhaust-cylinder temp divergence',
    'FQty_imbalance'     : 'Fuel tank imbalance',
    'OAT'                : 'Outside air temperature',
    'IAS'                : 'Indicated airspeed',
    'AltMSL'             : 'Altitude MSL',
}


# ============================================================
# GRADIENT-BASED IMPORTANCE (No SHAP dependency)
# ============================================================
# WHY GRADIENT x INPUT aur SHAP nahi?
# SHAP GradientExplainer ke issues:
#   1. Background dataset chahiye — alag se manage karna padta
#   2. Slow — multiple forward passes per explanation
#   3. TCN jaise models pe setup tricky hai
#
# Gradient x Input (Saliency method):
#   Formula: importance[c, t] = |gradient[c, t] × input[c, t]|
#   Intuition: "Agar yeh pixel/sensor slightly change ho,
#              toh output kitna change hoga?" × "Kitna active tha yeh sensor?"
#   Single backward pass = fast
#   No background data needed
#   Equally interpretable for aviation domain
#
# Reference: Simonyan et al. (2013) "Deep Inside Convolutional Networks"

def compute_gradient_importance(
    model    : nn.Module,
    flight   : np.ndarray,
    device   : str = 'cpu',
) -> dict:
    """
    Gradient x Input method se feature importance nikalta hai.

    Kyun Gradient x Input?
      - SHAP GradientExplainer requires specific setup
      - Gradient x Input equally interpretable hai
      - Fast — single backward pass
      - Works on any differentiable model
      - Formula: importance = |gradient * input|

    Args:
        model  : trained TCN model
        flight : (T, C) = (4096, 31) normalized array
        device : 'cpu' or 'cuda'

    Returns:
        dict:
          channel_importance : (31,) per channel importance
          temporal_importance: (4096,) per timestep importance
          top_channels       : top 5 channels by importance
    """
    try:
        # eval() mode: dropout off, batchnorm running stats use
        # Explanation ke time training behavior nahi chahiye
        model.eval()

        # (T, C) → transpose → (C, T) → newaxis → (1, C, T)
        # requires_grad=True: CRITICAL — input tensor ka gradient chahiye
        # normally sirf parameters ka gradient compute hota hai
        # requires_grad=True se PyTorch input tensor ke liye bhi gradient track karta hai
        x = torch.tensor(
            flight.T[np.newaxis, ...],
            dtype=torch.float32,
            requires_grad=True
        ).to(device)

        # Forward pass: (1, 31, 4096) → (1, 1) logit
        logit = model(x).squeeze()
        # Logit → probability (0-1) — gradient probability w.r.t. input nikalna zyada interpretable hai
        prob  = torch.sigmoid(logit)

        # Backward pass — probability ka gradient input tensor ke w.r.t. compute karo
        # model.zero_grad(): pehle ke kisi bhi accumulated gradients clear karo
        # prob.backward(): chain rule se x.grad populate hoga — shape (1, 31, 4096)
        model.zero_grad()
        prob.backward()

        # .detach(): gradient computation graph se alag karo — numpy conversion ke liye zaroori
        # .cpu(): agar CUDA pe tha toh RAM mein laao
        # .numpy()[0]: (1, C, T) → (C, T) — batch dimension remove
        grad = x.grad.detach().cpu().numpy()[0]  # (C, T) = (31, 4096)
        inp  = x.detach().cpu().numpy()[0]        # (C, T) = (31, 4096)

        # Gradient x Input: element-wise multiply, phir absolute value
        # WHY ABSOLUTE? Negative gradient = "isse kam karo prediction badhegi" — dono directions important
        # |grad × inp| = "yeh sensor kitna influential tha prediction mein"
        importance = np.abs(grad * inp)           # (C, T) = (31, 4096)

        # Channel importance: har channel ke liye time dimension pe average
        # (31, 4096) → mean(axis=1) → (31,)
        # axis=1 = time axis — poore flight mein is channel ki average importance
        channel_imp = importance.mean(axis=1)     # (31,)
        # Normalize to 0-1: max channel ko 1.0 banao, baaki relative
        # Isse different flights ke explanations comparable hote hain
        if channel_imp.max() > 0:
            channel_imp = channel_imp / channel_imp.max()

        # Temporal importance: har timestep ke liye channel dimension pe average
        # (31, 4096) → mean(axis=0) → (4096,)
        # axis=0 = channel axis — is timestep pe sabhi sensors ki average importance
        temporal_imp = importance.mean(axis=0)    # (4096,)
        if temporal_imp.max() > 0:
            temporal_imp = temporal_imp / temporal_imp.max()

        # Top 5 most important channels
        # np.argsort: ascending order mein sorted indices
        # [::-1]: reverse → descending (highest importance pehle)
        # [:5]: top 5 indices
        top_idx = np.argsort(channel_imp)[::-1][:5]
        top_channels = [
            {
                'channel'    : CHANNEL_NAMES[i],
                # .get(key, key): description nahi hai toh raw channel name fallback
                'description': SENSOR_DESCRIPTIONS.get(
                    CHANNEL_NAMES[i], CHANNEL_NAMES[i]
                ),
                'importance' : round(float(channel_imp[i]), 4),
            }
            for i in top_idx
        ]

        return {
            'channel_importance' : channel_imp.tolist(),   # numpy → list (JSON serializable)
            'temporal_importance': temporal_imp.tolist(),  # (4096,) list — Streamlit timeline plot ke liye
            'top_channels'       : top_channels,           # top 5 dicts — plain language mein use honge
        }

    except Exception as e:
        raise AeroGuardException(
            e, context="Gradient importance computation"
        )


# ============================================================
# PLAIN LANGUAGE EXPLANATION
# ============================================================
# WHY PLAIN LANGUAGE?
# TCN output = probability (e.g., 0.73) — maintenance engineer ke liye actionable nahi
# "Engine oil temperature strongly influenced this prediction" — actionable hai
# XAI ka real value tab hai jab technical output human decision mein translate ho
# Yeh function wahi karta hai: numbers → sentences → actions

def generate_plain_explanation(
    prediction    : dict,
    gradient_info : dict,
    anomaly_stat  : dict = None,
) -> dict:
    """
    Model output ko plain language mein convert karta hai.

    Args:
        prediction   : output from predict_single_flight()
        gradient_info: output from compute_gradient_importance()
        anomaly_stat : output from StatisticalAnomalyDetector

    Returns:
        dict:
          summary          : one-line summary
          driving_factors  : what drove the prediction
          sensor_insights  : per sensor plain text
          recommended_action: what mechanic should do
    """
    try:
        prob     = prediction['probability']
        severity = prediction['severity']
        # top_ch: top 5 channels list — [{channel, description, importance}, ...]
        top_ch   = gradient_info['top_channels']

        # ── Summary ───────────────────────────────────────────────
        # Severity → one-line summary string
        # Dict lookup cleaner hai if-elif chain se
        # prob*100: percentage format zyada intuitive hai (73% vs 0.73)
        severity_summary = {
            'CRITICAL': (
                f"CRITICAL ALERT: Aircraft shows {prob*100:.1f}% "
                f"probability of maintenance within 2 days. "
                f"Immediate grounding recommended."
            ),
            'HIGH': (
                f"HIGH RISK: {prob*100:.1f}% maintenance "
                f"probability detected. Inspect before next flight."
            ),
            'MEDIUM': (
                f"ELEVATED RISK: {prob*100:.1f}% maintenance "
                f"probability. Monitor closely."
            ),
            'NORMAL': (
                f"Aircraft appears airworthy. "
                f"{prob*100:.1f}% maintenance probability — "
                f"below alert threshold."
            ),
        }
        summary = severity_summary.get(severity, "")

        # ── Driving factors ───────────────────────────────────────
        # Top 3 channels (top 5 mein se) ko English sentences mein convert karo
        # importance score → qualitative level → sentence
        # "strongly/moderately/slightly" — maintenance engineer ke liye intuitive gradation
        driving_factors = []
        for ch_info in top_ch[:3]:
            ch   = ch_info['channel']
            desc = ch_info['description']
            imp  = ch_info['importance']

            # Importance thresholds: 0.5 = dominant factor, 0.2 = secondary, <0.2 = minor
            if imp > 0.5:
                level = "strongly"
            elif imp > 0.2:
                level = "moderately"
            else:
                level = "slightly"

            driving_factors.append(
                f"{desc} {level} influenced this prediction "
                f"(importance: {imp:.2f})"
            )

        # ── Sensor insights ───────────────────────────────────────
        # Top channels ko sensor groups mein categorize karo
        # Group-based insights: agar koi engine sensor top mein hai
        # toh engine-specific recommendation generate karo
        # Sets use kiye hain: & (intersection) se O(1) membership check
        sensor_insights = []

        # Engine core sensors: oil system + RPM
        engine_sensors = {
            'E1 OilT', 'E1 OilP', 'E1 RPM'
        }
        # Cylinder head: individual CHTs + engineered spread/mean/gradient
        cylinder_sensors = {
            'E1 CHT1', 'E1 CHT2', 'E1 CHT3', 'E1 CHT4',
            'CHT_spread', 'CHT_mean', 'CHT4_minus_CHT1'
        }
        # Combustion: EGTs + EGT-CHT relationship
        combustion_sensors = {
            'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4',
            'EGT_spread', 'EGT_mean', 'EGT_CHT_divergence'
        }
        # Fuel system: flow + tank quantities + imbalance
        fuel_sensors = {
            'E1 FFlow', 'FQtyL', 'FQtyR', 'FQty_imbalance'
        }

        # top_ch_names: set of channel names in top 5 — intersection check ke liye
        top_ch_names = {c['channel'] for c in top_ch}

        # Set intersection (&): koi bhi engine sensor top 5 mein hai?
        if top_ch_names & engine_sensors:
            sensor_insights.append(
                "Engine oil system parameters show unusual "
                "patterns — check oil level and pressure."
            )
        if top_ch_names & cylinder_sensors:
            sensor_insights.append(
                "Cylinder temperature patterns suggest "
                "potential compression or cooling issue — "
                "inspect cylinder heads."
            )
        if top_ch_names & combustion_sensors:
            sensor_insights.append(
                "Exhaust temperature imbalance detected — "
                "possible intake gasket or fuel mixture issue."
            )
        if top_ch_names & fuel_sensors:
            sensor_insights.append(
                "Fuel system anomaly detected — "
                "check fuel flow and tank balance."
            )

        # Statistical detector output bhi incorporate karo — agar available hai
        # anomaly_score > 0.05: sirf meaningful statistical anomaly pe insight add karo
        # (very low score pe unnecessary noise nahi dena)
        if anomaly_stat and \
                anomaly_stat.get('anomaly_score', 0) > 0.05:
            flagged = anomaly_stat.get('flagged_sensors', [])
            if flagged:
                # Top 3 flagged sensors list karo — output verbose mat karo
                sensor_insights.append(
                    f"Statistical anomalies detected in: "
                    f"{', '.join(flagged[:3])}."
                )

        # ── Recommended action ────────────────────────────────────
        # Severity + top channel group → specific actionable instruction
        # Nested ternary: top_ch_names intersection se primary system identify karo
        # phir severity se action urgency decide karo
        # Maintenance engineer ko clearly batao: "kya karna hai aur kahan dekhna hai"
        action_map = {
            'CRITICAL': (
                "Ground aircraft immediately. Do not fly "
                "until full inspection completed. "
                "Priority: " + (
                    "engine/oil system"
                    if top_ch_names & engine_sensors
                    else "cylinder/exhaust system"
                    if top_ch_names & cylinder_sensors
                    else "fuel system"
                    if top_ch_names & fuel_sensors
                    else "full system check"   # fallback: koi known group nahi
                )
            ),
            'HIGH': (
                "Complete pre-flight inspection before "
                "next flight. Focus on: " + (
                    "oil level, pressure, and temperature"
                    if top_ch_names & engine_sensors
                    else "cylinder compression and cooling"
                    if top_ch_names & cylinder_sensors
                    else "fuel system and flow"
                    if top_ch_names & fuel_sensors
                    else "full visual inspection"
                )
            ),
            'MEDIUM': (
                "Monitor for next 2 flights. "
                "Log any unusual sounds or performance changes."
            ),
            'NORMAL': (
                "Continue normal operations. "
                "Schedule routine maintenance as planned."
            ),
        }

        return {
            'summary'           : summary,
            'driving_factors'   : driving_factors,    # list of strings — top 3 sensors ka English explanation
            'sensor_insights'   : sensor_insights,    # list of strings — group-based actionable insights
            'recommended_action': action_map.get(
                severity, ""                          # fallback empty string — unknown severity pe crash nahi
            ),
        }

    except Exception as e:
        raise AeroGuardException(
            e, context="Plain language explanation"
        )


# ============================================================
# MAIN EXPLAINER CLASS
# ============================================================
# AeroGuardExplainer: thin wrapper class jo dono pieces combine karta hai
#   compute_gradient_importance() → numbers
#   generate_plain_explanation()  → language
# FastAPI /explain endpoint ya Streamlit dashboard yahi class use karega
# Caller ko internals nahi jaanne — sirf .explain() call karo

class AeroGuardExplainer:
    """
    AeroGuard XAI Engine.

    Combines:
      1. Gradient x Input importance (channel + temporal)
      2. Statistical anomaly context
      3. Plain language explanation
    """

    def __init__(
        self,
        model : nn.Module,
        device: str = 'cpu',
    ):
        # model: trained TCN — explain() mein gradient backward pass ke liye chahiye
        # device: model aur input tensor same device pe hone chahiye
        self.model  = model
        self.device = device
        logger.info("AeroGuardExplainer initialized")

    def explain(
        self,
        flight       : np.ndarray,
        prediction   : dict,
        anomaly_stat : dict = None,
    ) -> dict:
        """
        Poora explanation generate karta hai.

        Args:
            flight      : (4096, 31) normalized flight
            prediction  : from predict_single_flight()
            anomaly_stat: from StatisticalAnomalyDetector

        Returns:
            dict: complete explanation
        """
        # anomaly_stat optional hai — Layer 1 detectors available nahi hain toh None pass karo
        # Plain language function gracefully handle karta hai None case
        try:
            logger.info("Generating explanation...")

            # Step 1: Gradient × Input — numerical importance scores
            # channel_importance (31,), temporal_importance (4096,), top_channels list
            grad_info = compute_gradient_importance(
                self.model, flight, self.device
            )

            # Step 2: Numbers → English sentences + recommended actions
            # gradient_info + prediction + anomaly_stat → human readable dict
            plain = generate_plain_explanation(
                prediction, grad_info, anomaly_stat
            )

            # Final explanation dict: downstream consumers (FastAPI, Streamlit) ke liye
            # gradient_importance: visualization ke liye (bar chart, heatmap)
            # plain_language     : display ke liye (text cards, alerts)
            # anomaly_context    : statistical detector ka raw output — passthrough
            explanation = {
                'gradient_importance': grad_info,
                'plain_language'     : plain,
                'anomaly_context'    : anomaly_stat or {},  # None → empty dict (JSON serializable)
            }

            logger.info(
                f"Explanation generated — "
                f"top channel: "
                f"{grad_info['top_channels'][0]['channel']}"
            )

            return explanation

        except Exception as e:
            raise AeroGuardException(
                e, context="AeroGuard explanation"
            )