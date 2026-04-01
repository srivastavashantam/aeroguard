# ============================================================
# AeroGuard — FastAPI Backend
#
# Endpoints:
#   GET  /health          — API health check
#   GET  /model-info      — Model metadata
#   POST /predict         — Prediction + Anomaly + XAI
#
# Usage:
#   uvicorn api.main:app --reload --port 8000
# ============================================================
# NOTE: Yeh v2 hai original main.py ka — upgrades:
#   v1: sirf TCN prediction
#   v2: TCN + Statistical Anomaly Detection + XAI explanation
# Teen components ek hi /predict endpoint mein integrate hain —
# caller ko sirf ek API call karni hai, teen layers ka result milta hai

# os: environment variable access ke liye (future config management)
import os
# numpy: list[list[float]] → np.ndarray conversion aur shape validation
import numpy as np
# FastAPI: async web framework — automatic OpenAPI docs, request/response validation
from fastapi import FastAPI, HTTPException
# CORSMiddleware: browser se cross-origin requests allow karo (Streamlit dashboard ke liye)
from fastapi.middleware.cors import CORSMiddleware
# BaseModel: Pydantic — JSON parsing + type validation + serialization
# Field: schema metadata (description, defaults) — /docs pe dikhta hai
from pydantic import BaseModel, Field
# Optional: field present bhi ho sakta hai ya None — flight_id required nahi hai
from typing import Optional
# Custom logger — structured logs with context
from src.logger import logger
# TCN model loading aur single flight inference
from src.models.tcn_model import load_tcn_model, predict_single_flight
# Layer 1a: per-sensor per-phase statistical anomaly detector
from src.anomaly.statistical import StatisticalAnomalyDetector
# XAI engine: Gradient × Input importance + plain language explanation
from src.xai.explainer import AeroGuardExplainer
# Custom exception class
from src.exception import ModelPredictionException

# ── App setup ─────────────────────────────────────────────────
# FastAPI app instantiate karo — metadata Swagger UI (/docs) pe dikhta hai
app = FastAPI(
    title       = "AeroGuard API",
    description = "Aircraft Health Monitoring & "
                  "Explainable Maintenance Alert System",
    version     = "1.0.0",
)

# CORS: sabhi origins allow karo — development ke liye
# Production mein specific domains whitelist karo:
# allow_origins=["http://localhost:8501", "https://aeroguard.yourdomain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Globals ───────────────────────────────────────────────────
# Teeno components module-level globals mein rakhe hain —
# startup pe ek baar load hote hain, har request pe share hote hain
#
# MODEL    : trained TCN (torch.nn.Module) — inference ke liye
# CONFIG   : production_config.json dict — threshold, channels, metrics
# DETECTOR : fitted StatisticalAnomalyDetector — JSON se loaded
# EXPLAINER: AeroGuardExplainer — model ka reference hold karta hai
# DEVICE   : inference device — "cuda" agar GPU available ho
#
# WHY FOUR GLOBALS aur ek dict nahi?
# Har component independently None-check hota hai —
# agar DETECTOR load fail ho toh sirf anomaly skip hoga,
# baaki pipeline kaam karti rahegi (graceful degradation)
MODEL      = None
CONFIG     = None
DETECTOR   = None
EXPLAINER  = None
DEVICE     = "cpu"


@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup pe automatically call hota hai.
    Teeno components sequentially load hote hain.
    Koi bhi fail hua toh app start nahi hogi — fail fast.
    """
    # global keyword: module-level variables update karne ke liye zaroori
    # warna Python local variables banayega aur globals None rahenge
    global MODEL, CONFIG, DETECTOR, EXPLAINER
    try:
        logger.info("AeroGuard API starting up...")

        # ── Component 1: TCN Model ────────────────────────────────
        # artifacts/best_tcn.pt + production_config.json load karo
        # MODEL.eval() already set hai load_tcn_model ke andar
        MODEL, CONFIG = load_tcn_model(
            model_path  = "artifacts/best_tcn.pt",
            config_path = "artifacts/production_config.json",
            device      = DEVICE
        )
        logger.info("✅ TCN model loaded")

        # ── Component 2: Statistical Anomaly Detector ─────────────
        # StatisticalAnomalyDetector() sirf object banata hai —
        # .load() se artifacts/statistical_detector.json se fitted stats restore hote hain
        # (per-phase per-sensor mean + std jo healthy flights pe compute hua tha)
        DETECTOR = StatisticalAnomalyDetector()
        DETECTOR.load("artifacts/statistical_detector.json")
        logger.info("✅ Anomaly detector loaded")

        # ── Component 3: XAI Explainer ────────────────────────────
        # AeroGuardExplainer ko fitted MODEL reference chahiye —
        # isliye MODEL pehle load karna zaroori tha
        # EXPLAINER apne andar model store karta hai — gradient backward pass ke liye
        EXPLAINER = AeroGuardExplainer(MODEL, device=DEVICE)
        logger.info("✅ XAI explainer initialized")

        logger.info("✅ AeroGuard API ready")

    except Exception as e:
        # Koi bhi component fail hua → log karo aur re-raise karo
        # App crash ho jaaye startup pe — broken state mein silently chalna dangerous hai
        logger.error(f"Startup failed: {e}")
        raise


# ── Schemas ───────────────────────────────────────────────────
# Pydantic models teen kaam karte hain:
#   1. JSON → Python object automatic parsing
#   2. Type validation — galat type aaye toh 422 auto-return
#   3. /docs pe OpenAPI schema generate hota hai automatically


class FlightDataInput(BaseModel):
    """
    Input schema for /predict endpoint.
    flight_data : (4096, 31) normalized sensor array
    flight_id   : Optional Master Index
    explain     : Whether to include XAI explanation
    """
    # list[list[float]]: JSON array of arrays — Pydantic type check karta hai
    # Shape (4096×31) validation Pydantic nahi karta — endpoint mein manually kiya hai
    flight_data : list[list[float]] = Field(
        ...,
        description="(4096, 31) normalized sensor array"
    )
    # Optional flight identifier — NGAFID Master Index
    # Response mein echo hota hai taaki caller track kar sake
    flight_id   : Optional[int] = Field(
        None,
        description="Master Index of the flight"
    )
    # explain=True default: XAI by default on hai
    # Heavy flights ke liye caller explain=False pass kar sakta hai
    # (XAI mein backward pass hota hai — thoda slower)
    explain     : bool = Field(
        True,
        description="Include XAI explanation in response"
    )


class AnomalyResult(BaseModel):
    # Statistical detector ka output — /predict response mein nested
    anomaly_score   : float       # 0-1: fraction of flagged timesteps
    flagged_sensors : list[str]   # sensor names jo 3σ se bahar gaye
    phase_anomalies : dict        # per phase flagged timestep counts
    top_anomalies   : list[dict]  # top 5 by max z-score


class ChannelImportance(BaseModel):
    # XAI top_channels list ka ek element
    channel     : str    # 'E1 OilT'
    description : str    # 'Engine oil temperature'
    importance  : float  # 0-1 normalized gradient × input score


class XAIResult(BaseModel):
    # XAI explainer ka flattened output — gradient info + plain language combined
    top_channels        : list[ChannelImportance]  # top 5 sensors by importance
    summary             : str                       # one-line alert summary
    driving_factors     : list[str]                 # top 3 sensors ka English explanation
    sensor_insights     : list[str]                 # group-based actionable insights
    recommended_action  : str                       # severity-specific action


class PredictionResponse(BaseModel):
    # /predict ka complete response — teeno layers ka combined output
    flight_id   : Optional[int]
    probability : float           # TCN sigmoid output (0-1)
    prediction  : int             # 0=safe, 1=at-risk
    severity    : str             # NORMAL/MEDIUM/HIGH/CRITICAL
    threshold   : float           # classification boundary (from config)
    message     : str             # human readable severity message
    # Optional fields: agar respective component unavailable tha ya fail hua toh None
    anomaly     : Optional[AnomalyResult]   # Layer 1a statistical results
    explanation : Optional[XAIResult]       # XAI gradient + language output


class HealthResponse(BaseModel):
    status          : str   # "ok"
    model_loaded    : bool  # MODEL is not None
    detector_loaded : bool  # DETECTOR is not None — v2 mein add hua
    version         : str


class ModelInfoResponse(BaseModel):
    model_name  : str
    n_channels  : int
    n_timesteps : int
    threshold   : float
    test_auc    : float
    test_f1     : float
    test_recall : float  # test_recall_0_35 — threshold=0.35 pe computed


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check."""
    # v2 mein detector_loaded bhi add kiya — monitoring systems ko pata chale
    # sirf MODEL nahi, DETECTOR bhi ready hai ya nahi
    return HealthResponse(
        status          = "ok",
        model_loaded    = MODEL is not None,
        detector_loaded = DETECTOR is not None,
        version         = "1.0.0"
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Model metadata aur performance metrics."""
    if CONFIG is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    return ModelInfoResponse(
        model_name  = CONFIG['model_name'],
        n_channels  = CONFIG['n_channels'],
        n_timesteps = CONFIG['n_timesteps'],
        threshold   = CONFIG['threshold'],
        test_auc    = CONFIG['test_auc'],
        test_f1     = CONFIG['test_f1'],
        # test_recall_0_35: threshold=0.35 pe evaluated recall
        # Aviation safety ke liye recall priority pe hai — false negatives costly
        test_recall = CONFIG['test_recall_0_35'],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: FlightDataInput):
    """
    Ek flight ke liye complete analysis.

    Returns:
      - TCN maintenance probability + severity
      - Statistical anomaly detection results
      - XAI explanation (optional, default=True)
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        # ── Step 1: Input convert ─────────────────────────────────
        # Pydantic ne list[list[float]] validate kiya — numpy mein convert karo
        # float32: model weights ke saath consistent dtype
        flight_array = np.array(
            input_data.flight_data,
            dtype=np.float32
        )

        # Pydantic nested list ka shape validate nahi karta — manually karo
        # 422 Unprocessable Entity: client side data error — retry karo correct shape se
        expected = (
            CONFIG['n_timesteps'],
            CONFIG['n_channels']
        )
        if flight_array.shape != expected:
            raise HTTPException(
                status_code=422,
                detail=f"Expected shape {expected}, "
                       f"got {flight_array.shape}"
            )

        # ── Step 2: TCN Prediction ────────────────────────────────
        # Layer 2: supervised TCN — maintenance probability nikalo
        # result dict: {probability, prediction, severity, threshold}
        result = predict_single_flight(
            MODEL, flight_array, CONFIG, DEVICE
        )

        # Severity → actionable human message mapping
        severity_messages = {
            "CRITICAL": "Ground aircraft immediately — "
                        "critical maintenance required",
            "HIGH"    : "Inspect before next flight — "
                        "high risk detected",
            "MEDIUM"  : "Monitor closely — "
                        "elevated maintenance risk",
            "NORMAL"  : "Aircraft appears airworthy — "
                        "no immediate action required",
        }

        logger.info(
            f"Flight {input_data.flight_id} — "
            f"Prob: {result['probability']:.4f} | "
            f"Severity: {result['severity']}"
        )

        # ── Step 3: Anomaly Detection ─────────────────────────────
        # Layer 1a: Statistical detector — per-sensor per-phase z-score
        # anomaly_result: Pydantic model → response mein jaayega
        # anomaly_dict  : raw dict → XAI ko pass kiya jaayega (plain language context ke liye)
        anomaly_result = None
        anomaly_dict   = None

        if DETECTOR is not None:
            try:
                anomaly_dict   = DETECTOR.detect(flight_array)
                # Raw dict → Pydantic model — response serialization ke liye
                anomaly_result = AnomalyResult(
                    anomaly_score   = anomaly_dict['anomaly_score'],
                    flagged_sensors = anomaly_dict['flagged_sensors'],
                    phase_anomalies = anomaly_dict['phase_anomalies'],
                    top_anomalies   = anomaly_dict['top_anomalies'],
                )
                logger.info(
                    f"Anomaly score: "
                    f"{anomaly_dict['anomaly_score']:.4f}"
                )
            except Exception as e:
                # GRACEFUL DEGRADATION: anomaly detection fail hua toh sirf log karo
                # HTTPException raise mat karo — TCN prediction still valid hai
                # anomaly_result = None rahega → response mein anomaly field None hoga
                logger.warning(f"Anomaly detection failed: {e}")

        # ── Step 4: XAI Explanation ───────────────────────────────
        # explain flag: caller control karta hai XAI on/off
        # Agar EXPLAINER None hai (startup fail tha) toh gracefully skip karo
        xai_result = None

        if input_data.explain and EXPLAINER is not None:
            try:
                # EXPLAINER.explain(): gradient backward pass + plain language generation
                # anomaly_dict pass karo: statistical context plain language mein include hoga
                explanation = EXPLAINER.explain(
                    flight_array, result, anomaly_dict
                )
                plain = explanation['plain_language']
                grad  = explanation['gradient_importance']

                # grad['top_channels']: list of dicts → list of ChannelImportance Pydantic models
                # **ch: dict unpacking — keys match karne chahiye ChannelImportance fields se
                xai_result = XAIResult(
                    top_channels = [
                        ChannelImportance(**ch)
                        for ch in grad['top_channels']
                    ],
                    summary            = plain['summary'],
                    driving_factors    = plain['driving_factors'],
                    sensor_insights    = plain['sensor_insights'],
                    recommended_action = plain['recommended_action'],
                )
                logger.info(
                    f"XAI top channel: "
                    f"{grad['top_channels'][0]['channel']}"
                )
            except Exception as e:
                # GRACEFUL DEGRADATION: XAI fail hua toh warning log karo
                # Core prediction already done hai — XAI optional enrichment hai
                # xai_result = None → response mein explanation field None hoga
                logger.warning(f"XAI failed: {e}")

        # ── Step 5: Response assemble karo ───────────────────────
        # Teeno layers ka combined response — koi bhi None ho sakta hai
        # Pydantic Optional fields automatically None serialize karta hai
        return PredictionResponse(
            flight_id   = input_data.flight_id,
            probability = result['probability'],
            prediction  = result['prediction'],
            severity    = result['severity'],
            threshold   = result['threshold'],
            message     = severity_messages[result['severity']],
            anomaly     = anomaly_result,    # None agar DETECTOR unavailable/failed
            explanation = xai_result,        # None agar explain=False ya EXPLAINER failed
        )

    except HTTPException:
        # HTTPException re-raise karo bina wrap kiye —
        # 422 (shape mismatch) ko 500 mein convert hone se bachao
        raise
    except Exception as e:
        # Unexpected errors → 500 Internal Server Error
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )