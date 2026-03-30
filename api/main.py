# ============================================================
# AeroGuard — FastAPI Backend
#
# Endpoints:
#   GET  /health          — API health check
#   POST /predict         — Single flight prediction
#   GET  /model-info      — Model metadata
#
# Usage:
#   uvicorn api.main:app --reload --port 8000
# ============================================================

# os: environment variables padhne ke liye (agar kabhi MODEL_PATH env se lena ho)
# Abhi direct string use ho rahi hai, lekin future-proofing ke liye import rakha hai
import os
# numpy: JSON list → numpy array conversion ke liye /predict endpoint mein
# flight_data list[list[float]] aata hai — model np.ndarray expect karta hai
import numpy as np
# FastAPI: ASGI web framework — async endpoints, automatic OpenAPI docs, request validation sab built-in
from fastapi import FastAPI, HTTPException
# CORSMiddleware: Cross-Origin Resource Sharing — browser se alag port pe React/Streamlit dashboard
# agar CORS na ho toh browser request block kar dega (security policy)
from fastapi.middleware.cors import CORSMiddleware
# BaseModel: Pydantic ka base class — automatic JSON parsing, type validation, aur serialization
# Field: schema mein extra metadata add karna (description, constraints) OpenAPI docs ke liye
from pydantic import BaseModel, Field
# Optional: type hint — field present bhi ho sakta hai, None bhi — flight_id required nahi hai
from typing import Optional
# Custom structured logger — loguru based, har log mein timestamp + level + context
from src.logger import logger
# Model loading aur inference functions — tcn_model.py se
from src.models.tcn_model import load_tcn_model, predict_single_flight
# Custom exception class — TCN inference errors ko wrap karta hai context ke saath
from src.exception import ModelPredictionException

# ── App setup ─────────────────────────────────────────────────
# FastAPI app instantiate karo with metadata
# title/description/version → /docs pe Swagger UI mein dikhta hai
# /docs aur /redoc automatically generate hote hain — koi extra setup nahi chahiye
app = FastAPI(
    title       = "AeroGuard API",
    description = "Aircraft Health Monitoring & "
                  "Explainable Maintenance Alert System",
    version     = "1.0.0",
)

# CORS — dashboard ke liye
# allow_origins=["*"]: sabhi domains ko allow karo
# Production mein yeh ["http://localhost:3000", "https://aeroguard.yourdomain.com"] hona chahiye
# Development mein wildcard theek hai — faster iteration
# allow_methods=["*"]: GET, POST, PUT, DELETE, OPTIONS sab allow
# allow_headers=["*"]: Authorization, Content-Type wagerah sab allow
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Model load at startup ─────────────────────────────────────
# Global variables — module level pe rakhe hain taaki saare endpoints access kar sakein
# None initialize karo — startup_event mein populate honge
# WHY GLOBAL?
# FastAPI stateless nahi hai in the sense that ML models are expensive to load.
# Har request pe model load karna = ~2-5 seconds latency per call — unacceptable.
# Ek baar startup pe load karo, memory mein rakho, sab requests share karein.
MODEL      = None   # TCN model instance (torch.nn.Module)
CONFIG     = None   # production_config.json ka dict (threshold, channels, metrics)
DEVICE     = "cpu"  # "cuda" agar GPU available ho — inference device


@app.on_event("startup")
async def startup_event():
    """
    FastAPI application start hone pe automatically call hota hai.
    Model aur config ek baar yahan load hote hain — har request pe nahi.
    Agar model load fail ho toh app start hi nahi hogi (raise karta hai) —
    better than starting with a broken state silently.
    """
    # global keyword zaroori hai — warna Python local variable samjhega
    # aur module-level MODEL/CONFIG update nahi honge
    global MODEL, CONFIG
    try:
        logger.info("AeroGuard API starting up...")
        # load_tcn_model: artifacts se model weights aur config load karta hai
        # DEVICE="cpu" — production mein "cuda" se swap karo agar GPU ho
        MODEL, CONFIG = load_tcn_model(
            model_path  = "artifacts/best_tcn.pt",
            config_path = "artifacts/production_config.json",
            device      = DEVICE
        )
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        # Error log karo aur re-raise karo — app crash ho jaaye startup pe
        # WHY CRASH? Agar model load nahi hua aur app chalta rahe,
        # toh /predict pe 503 ya worse — random errors aayenge.
        # Fail fast at startup = cleaner production behavior.
        logger.error(f"Model load failed: {e}")
        raise


# ── Schemas ───────────────────────────────────────────────────
# Pydantic models do kaam karte hain:
#   1. Request validation — galat shape/type aaye toh 422 Unprocessable Entity auto-return
#   2. OpenAPI documentation — /docs pe schema automatically dikhta hai
# Ek baar define karo, FastAPI baaki sab handle karta hai.


class FlightDataInput(BaseModel):
    """
    Input schema for /predict endpoint.

    flight_data: 2D array of shape (4096, 31)
                 Normalized sensor readings
                 Row = timestep, Col = channel
    flight_id  : Optional flight identifier
    """
    # list[list[float]]: JSON array of arrays — Python type hint se Pydantic validate karega
    # Field(...): ... matlab required field — default value nahi hai
    # description: /docs pe dikhta hai — API consumers ke liye helpful
    # NOTE: Shape validation (4096x31) yahan nahi hoti — Pydantic sirf type check karta hai
    # Actual shape validation /predict endpoint mein manually ki gayi hai
    flight_data : list[list[float]] = Field(
        ...,
        description="(4096, 31) normalized sensor array"
    )
    # Optional[int]: flight_id present hona zaroori nahi — None allowed hai
    # Master Index = NGAFID dataset ka unique flight identifier
    # Logging aur traceability ke liye useful — response mein bhi echo hota hai
    flight_id   : Optional[int] = Field(
        None,
        description="Master Index of the flight"
    )


class PredictionResponse(BaseModel):
    """
    Output schema for /predict endpoint.
    """
    # flight_id wapas echo karo — caller ko pata chale kaunsi flight ka result hai
    # Especially useful agar future mein batch processing implement ho
    flight_id   : Optional[int]
    # 0.0 to 1.0 — sigmoid output, rounded to 4 decimal places
    probability : float
    # 0 = safe (airworthy), 1 = at-risk (maintenance needed)
    prediction  : int
    # "NORMAL" | "MEDIUM" | "HIGH" | "CRITICAL" — human-readable bucketing
    severity    : str
    # Jo threshold use hua classification ke liye — transparency ke liye include kiya
    threshold   : float
    # Human-readable action message — severity se map hota hai
    message     : str


class HealthResponse(BaseModel):
    # "ok" ya "degraded" — load balancer health checks ke liye
    status      : str
    # True agar MODEL global None nahi hai — startup successful tha ya nahi
    model_loaded: bool
    # Semantic version string — API versioning track karne ke liye
    version     : str


class ModelInfoResponse(BaseModel):
    # "TCN" — model architecture name
    model_name  : str
    # 31 — input sensor channels count
    n_channels  : int
    # 4096 — input timesteps count
    n_timesteps : int
    # Classification boundary — tune kiya gaya hai class imbalance ke liye
    threshold   : float
    # Area Under ROC Curve — overall discrimination ability
    test_auc    : float
    # F1 Score — precision aur recall ka harmonic mean
    test_f1     : float
    # Recall at threshold=0.35 — at-risk flights kitne pakde (safety-critical metric)
    # Aviation mein false negatives bahut costly hain — recall priority pe hai
    test_recall : float


# ── Endpoints ─────────────────────────────────────────────────
# FastAPI mein har endpoint ek async function hai.
# async: uvicorn event loop pe run hota hai — I/O operations block nahi karta
# response_model: return dict ko is schema mein validate + serialize karta hai automatically


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check."""
    # Simple health check — koi computation nahi, sirf status return karo
    # Kubernetes/Docker health probes yahi endpoint hit karte hain typically
    # model_loaded: MODEL None hai ya nahi — startup fail hua tha toh False hoga
    return HealthResponse(
        status       = "ok",
        model_loaded = MODEL is not None,
        version      = "1.0.0"
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Model metadata aur performance metrics."""
    # CONFIG None check — agar startup fail hua toh CONFIG bhi None hoga
    # 503 Service Unavailable: server up hai lekin model ready nahi — caller retry kar sakta hai
    if CONFIG is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    # CONFIG dict se fields extract karo — keys production_config.json se match karne chahiye
    # test_recall_0_35: threshold=0.35 pe computed recall — Colab evaluation notebook mein yahi tha
    return ModelInfoResponse(
        model_name  = CONFIG['model_name'],
        n_channels  = CONFIG['n_channels'],
        n_timesteps = CONFIG['n_timesteps'],
        threshold   = CONFIG['threshold'],
        test_auc    = CONFIG['test_auc'],
        test_f1     = CONFIG['test_f1'],
        test_recall = CONFIG['test_recall_0_35'],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: FlightDataInput):
    """
    Ek flight ke liye maintenance prediction.

    Input  : (4096, 31) normalized sensor array
    Output : probability, severity, prediction
    """
    # Guard clause: MODEL None hai toh predict karna impossible
    # 503: model temporarily unavailable — client ko retry karne ka signal
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    try:
        # ── STEP 1: List → numpy array ───────────────────────────────
        # Pydantic ne list[list[float]] validate kar diya — ab numpy mein convert karo
        # dtype=np.float32: model weights float32 mein hain — match karo
        # float64 doge toh torch.tensor conversion pe implicit cast hoga — better explicit karo
        flight_array = np.array(
            input_data.flight_data,
            dtype=np.float32
        )

        # ── STEP 2: Shape validation ─────────────────────────────────
        # Pydantic nested list ka shape validate nahi karta — manually karo
        # (4096, 31) expected: 4096 timesteps, 31 channels
        # Galat shape aaye toh 422 Unprocessable Entity — client side bug hai yeh
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

        # ── STEP 3: Inference ────────────────────────────────────────
        # predict_single_flight: numpy → tensor → model → probability + severity
        # Internals: transpose, sigmoid, thresholding — sab encapsulated hai tcn_model.py mein
        result = predict_single_flight(
            MODEL, flight_array, CONFIG, DEVICE
        )

        # ── STEP 4: Human readable message ──────────────────────────
        # Severity → actionable message mapping
        # Aviation domain ke liye specific language:
        #   CRITICAL: Ground the aircraft — fly mat karo
        #   HIGH: Inspect karo pehle — risk hai
        #   MEDIUM: Monitor karo — watch karo
        #   NORMAL: Airworthy — sab theek
        # Dict lookup O(1) hai — if-elif chain se cleaner
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

        # Structured log — flight ID + probability + severity ek line mein
        # Downstream log aggregation (ELK, CloudWatch) ke liye parseable format
        logger.info(
            f"Flight {input_data.flight_id} — "
            f"Prob: {result['probability']:.4f} | "
            f"Severity: {result['severity']}"
        )

        # ── STEP 5: Response assemble karo ──────────────────────────
        # Pydantic PredictionResponse schema validate karega return value ko
        # Keys result dict se map ho rahi hain + flight_id aur message add ho rahe hain
        return PredictionResponse(
            flight_id   = input_data.flight_id,
            probability = result['probability'],
            prediction  = result['prediction'],
            severity    = result['severity'],
            threshold   = result['threshold'],
            message     = severity_messages[result['severity']],
        )

    except HTTPException:
        # HTTPException ko re-raise karo bina wrap kiye
        # WHY? Neeche wala broad `except Exception` HTTPException ko bhi catch kar lega
        # aur 422 ko 500 mein convert kar dega — galat status code jaayega client ko
        raise
    except Exception as e:
        # Koi bhi unexpected error (ModelPredictionException, numpy error, etc.)
        # 500 Internal Server Error mein convert karo
        # str(e) detail mein daalo — debugging ke liye helpful
        # NOTE: Production mein sensitive internal details str(e) mein expose mat karo
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )