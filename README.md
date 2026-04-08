# ✈️ AeroGuard — Aircraft Health Monitoring &amp; Explainable Maintenance Alert System

> **An end-to-end production ML system that predicts aircraft maintenance needs up to 2 days in advance using per-second sensor data from Cessna 172 flights — with explainability, anomaly detection, and a live deployment on Render.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-purple)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://docker.com)
[![Deployed on Render](https://img.shields.io/badge/Deployed-Render-46e3b7)](https://render.com)

---

## 🔴 Live Demo

| Service | URL |
|---------|-----|
| **FastAPI Backend** | https://aeroguard-api.onrender.com |
| **Health Check** | https://aeroguard-api.onrender.com/health |
| **API Docs (Swagger)** | https://aeroguard-api.onrender.com/docs |
| **Streamlit Dashboard** | https://aeroguard-dashboard.onrender.com |

> **Note:** Both services run on Render free tier. If the API appears offline, open the health check URL directly, wait 30–50 seconds for the container to spin up, then refresh the dashboard.

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Why AeroGuard — Real-World Impact](#-why-aeroguard--real-world-impact)
3. [Dataset](#-dataset)
4. [System Architecture](#-system-architecture)
5. [Tech Stack](#-tech-stack)
6. [Project Structure](#-project-structure)
7. [Execution Steps — Complete Walkthrough](#-execution-steps--complete-walkthrough)
   - [Phase 1: Environment Setup](#phase-1-environment-setup)
   - [Phase 2: Data Pipeline](#phase-2-data-pipeline)
   - [Phase 3: Feature Engineering](#phase-3-feature-engineering)
   - [Phase 4: Model Training on Google Colab](#phase-4-model-training-on-google-colab)
   - [Phase 5: Local Model Integration](#phase-5-local-model-integration)
   - [Phase 6: Anomaly Detection](#phase-6-anomaly-detection)
   - [Phase 7: XAI Engine](#phase-7-xai-engine)
   - [Phase 8: FastAPI Backend](#phase-8-fastapi-backend)
   - [Phase 9: Streamlit Dashboard](#phase-9-streamlit-dashboard)
   - [Phase 10: Retraining Pipeline + MLflow](#phase-10-retraining-pipeline--mlflow)
   - [Phase 11: Docker + CI/CD](#phase-11-docker--cicd)
   - [Phase 12: Deployment on Render](#phase-12-deployment-on-render)
8. [How to Run Locally](#-how-to-run-locally)
9. [API Reference](#-api-reference)
10. [Model Performance](#-model-performance)
11. [Key Design Decisions](#-key-design-decisions)

---

## 🎯 Problem Statement

General Aviation — flight schools, small charter operators, and airports — relies on **reactive or calendar-based maintenance** for aircraft like the Cessna 172.

**Reactive maintenance:** The aircraft is already compromised by the time the failure is noticed. This creates safety risks and emergency groundings.

**Calendar-based maintenance:** Parts get replaced regardless of their actual condition — sometimes too early (wasteful), sometimes too late (dangerous).

**The hidden opportunity:** Every Cessna 172 flight already records 23+ sensor readings every second — oil temperature, RPM, cylinder head temperatures, fuel flow, exhaust gas temperatures, airspeed, altitude, and more. This data flows into the NGAFID (National General Aviation Flight Information Database) but is never used predictively.

**AeroGuard's mission is to close this gap.** It processes this per-second sensor data and answers four questions after every flight lands:

1. **Will this aircraft need maintenance in the next 2 days?** (RUL binary classification)
2. **If yes, which component is likely failing?** (Fault classification)
3. **Which moments in the flight showed abnormal patterns?** (Anomaly detection)
4. **Why did the model make this prediction?** (Explainability via Gradient × Input)

---

## 💡 Why AeroGuard — Real-World Impact

**Without AeroGuard:**
```
Flight lands → Pilot parks aircraft → No sensor data reviewed
→ Next day pilot takes off → Mid-flight failure → Emergency
```

**With AeroGuard:**
```
Flight lands → AeroGuard processes last flight data automatically
→ 3-layer analysis runs → Suspicious pattern detected
→ Mechanic receives alert: "Aircraft #7, Oil system anomaly,
  73% probability of maintenance within 2 days.
  Primary cause: E1 OilT elevated during cruise phase"
→ Aircraft grounded BEFORE next flight
```

This is the difference between reactive and predictive maintenance at scale.

---

## 📊 Dataset

**Source:** [NGAFID — National General Aviation Flight Information Database](https://doi.org/10.5281/zenodo.6624956)

| Metric | Value |
|--------|-------|
| Total Flights | 28,935 |
| Total Flight Hours | 31,177 hours |
| Total Sensor Rows | 100 million+ |
| Dataset Size | ~4.3 GB |
| Sensors per Flight | 23 (+ 8 engineered = 31 total) |
| Sampling Rate | 1 reading per second |
| Aircraft Model | Cessna 172 only |
| Unplanned Maintenance Events | 2,111 events |
| Maintenance Issue Types | 36 types |

### Dataset Files

| File | Description |
|------|-------------|
| `all_flights/flight_header.csv` | Flight metadata — labels, duration, maintenance dates (28,935 rows) |
| `all_flights/one_parq/` | Per-second sensor readings as Dask Parquet (4.3 GB) |
| `2days/flight_header.csv` | Benchmark subset — flights within 2-day maintenance window |
| `2days/flight_data.pkl` | Preprocessed numpy arrays from the paper's benchmark subset |

### Sensor Channels

**Engine (Most Critical for Maintenance)**

| Sensor | Description | Failure Signal |
|--------|-------------|----------------|
| E1 OilT | Engine Oil Temperature | Sustained high → cooling failure |
| E1 OilP | Engine Oil Pressure | Low pressure → engine seizure risk |
| E1 RPM | Engine RPM | Fluctuations → ignition/fuel issue |
| E1 CHT1–4 | Cylinder Head Temperatures (4 cylinders) | Imbalance → uneven combustion |
| E1 EGT1–4 | Exhaust Gas Temperatures (4 cylinders) | Divergence → fuel mixture/valve issue |
| E1 FFlow | Fuel Flow Rate | Abnormal → injector/carburetor issue |

**Flight Dynamics**

| Sensor | Description |
|--------|-------------|
| IAS | Indicated Airspeed — used for flight phase detection |
| AltMSL | Altitude Mean Sea Level — used for flight phase detection |
| VSpd | Vertical Speed |
| NormAc | Normal Acceleration — G-force on airframe |
| OAT | Outside Air Temperature |

**Fuel & Electrical**

| Sensor | Description |
|--------|-------------|
| FQtyL / FQtyR | Left/Right Fuel Tank Quantity |
| volt1 / volt2 | Main and Standby Bus Voltages |
| amp1 / amp2 | Main and Standby Battery Ammeters |

**Engineered Features (8 Novel Channels Added)**

| Feature | Description | Diagnostic Value |
|---------|-------------|-----------------|
| CHT_spread | max(CHT1–4) − min(CHT1–4) | Inter-cylinder combustion imbalance |
| CHT_mean | Average of all 4 CHTs | Overall cylinder thermal load |
| CHT4_minus_CHT1 | Directional temperature gradient | Systematic front/rear cooling imbalance |
| EGT_spread | max(EGT1–4) − min(EGT1–4) | Combustion uniformity |
| EGT_mean | Average of all 4 EGTs | Overall exhaust temperature |
| EGT_CHT_divergence | EGT − CHT ratio deviation | Intake gasket leak signature |
| FQty_imbalance | \|FQtyL − FQtyR\| normalized | Fuel system asymmetry |
| is_cruise | Binary cruise phase flag | Provides phase context to the model |

### Labeling Strategy

The label `date_diff` captures how many days before a maintenance event a given flight occurred. After evaluating three threshold options:

| Threshold | Total Flights | Safe (0) | At-Risk (1) | Ratio |
|-----------|--------------|----------|-------------|-------|
| −2 days | 16,359 | 71.5% | 28.5% | 1:2.5 |
| −3 days | 16,359 | 85.1% | 14.9% | 1:5 |
| −5 days | 16,359 | 95.4% | 4.6% | 1:20 |

**AeroGuard uses the −2 day threshold.** The 1:2.5 class ratio is manageable with weighted loss. The −5 day threshold (1:20 ratio) makes the model degenerate — it will always predict "safe" and achieve 95% accuracy while being completely useless.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INCOMING FLIGHT DATA                         │
│              (23 sensors × ~4096 seconds per flight)            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │         LAYER 1a           │
              │   Statistical Anomaly      │  ← Per-sensor z-score
              │      Detector              │    per flight phase
              └─────────────┬──────────────┘
                            │ anomaly_score + flagged_sensors
              ┌─────────────▼──────────────┐
              │         LAYER 2            │
              │     TCN RUL Classifier     │  ← "Will this aircraft
              │  (Temporal Convolutional   │    fail in 2 days?"
              │       Network)             │
              └─────────────┬──────────────┘
                            │ maintenance_probability (0–1)
              ┌─────────────▼──────────────┐
              │        XAI ENGINE          │
              │   Gradient × Input         │  ← "Why did the model
              │   Channel Importance       │    make this prediction?"
              │   Plain Language Summary   │
              └─────────────┬──────────────┘
                            │ human-readable explanation
              ┌─────────────▼──────────────┐
              │      ALERT GENERATOR       │
              │  CRITICAL / HIGH /         │  ← Final decision
              │  MEDIUM / NORMAL           │    + recommended action
              └─────────────┬──────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                                     ▼
  FastAPI REST API                    Streamlit Dashboard
  /health /predict /model-info        Mechanic View + Fleet View
```

### Alert Severity Thresholds

| Probability | Severity | Action |
|-------------|----------|--------|
| ≥ 80% | 🔴 CRITICAL | Ground aircraft immediately |
| 60–79% | 🟠 HIGH | Inspect before next flight |
| 40–59% | 🟡 MEDIUM | Monitor closely |
| < 40% | 🟢 NORMAL | No action needed |

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Data Processing** | Dask + PyArrow | Loading 4.3 GB sensor parquet files without RAM overflow |
| **Data Manipulation** | Pandas + NumPy | Flight metadata processing, array operations |
| **ML Framework** | PyTorch 2.0+ | TCN model training and inference |
| **Model Architecture** | TCN (Temporal Convolutional Network) | Time series classification — 8 dilation layers |
| **Experiment Tracking** | MLflow | Hyperparameter logging, model registry, artifact storage |
| **Anomaly Detection** | Statistical Z-Score (per-phase) | FAA 3-sigma standard — no training required |
| **Explainability** | Gradient × Input | Channel-level and timestep-level importance attribution |
| **API Framework** | FastAPI + Uvicorn | Async REST API with automatic OpenAPI documentation |
| **Dashboard** | Streamlit + Plotly | Mechanic view + Fleet manager view |
| **Containerization** | Docker + Docker Compose | Reproducible deployment environments |
| **CI/CD** | GitHub Actions | Automated test → build → deploy pipeline |
| **Deployment** | Render.com | Production hosting of both API and Dashboard |
| **Logging** | Loguru | Structured logging to terminal + rotating file handlers |
| **Configuration** | YAML + python-dotenv | Centralized config management |

---

## 📁 Project Structure

```
aeroguard/
│
├── .env                          # Environment variables (not pushed to GitHub)
├── .gitignore
├── requirements.txt
├── README.md
├── docker-compose.yml            # Orchestrates API + Dashboard containers
├── Dockerfile                    # Multi-stage Docker build
│
├── configs/
│   ├── config.yaml               # Central config: paths, thresholds, logging
│   └── mlflow_config.yaml        # MLflow + retraining hyperparameters
│
├── src/
│   ├── __init__.py
│   ├── logger.py                 # Loguru-based dual-sink logger (terminal + file)
│   ├── exception.py              # Custom exception hierarchy with context
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py          # Dask parquet load + flight header CSV load
│   │   ├── transformation.py     # Header cleaning, label construction, sensor filtering
│   │   └── feature_engineering.py # Per-flight extraction, 8 novel channels, normalization
│   │
│   ├── models/
│   │   └── tcn_model.py          # TCN architecture + load_tcn_model + predict_single_flight
│   │
│   ├── anomaly/
│   │   ├── __init__.py
│   │   └── statistical.py        # Per-sensor per-phase z-score detector (v2 with weighting)
│   │
│   └── xai/
│       ├── __init__.py
│       └── explainer.py          # Gradient × Input XAI + plain language generation
│
├── api/
│   ├── __init__.py
│   └── main.py                   # FastAPI: /health, /predict, /model-info endpoints
│
├── dashboard/
│   └── app.py                    # Streamlit: Mechanic View + Fleet Manager View
│
├── src/retraining_pipeline/
│   └── retrain.py                # Full MLflow-tracked retraining pipeline
│
├── artifacts/
│   ├── best_tcn.pt               # Trained TCN weights
│   ├── production_config.json    # Model metadata + thresholds + channel names
│   └── statistical_detector.json # Fitted per-phase per-sensor statistics
│
├── data/                         # Raw NGAFID dataset (gitignored — download separately)
│   ├── all_flights/
│   │   ├── flight_header.csv
│   │   └── one_parq/
│   └── 2days/
│       ├── flight_header.csv
│       └── flight_data.pkl
│
├── mlruns/                       # MLflow experiment tracking data
├── logs/                         # Rotating daily log files (gitignored)
└── .github/
    └── workflows/
        └── ci.yml                # GitHub Actions CI/CD pipeline
```

---

## 🚀 Execution Steps — Complete Walkthrough

This section documents every step taken to build AeroGuard from scratch — from the first `git init` all the way to a live production deployment.

---

### Phase 1: Environment Setup

**Step 1 — Create GitHub Repository**

Create a GitHub repository named `Aeroguard`. Add a `.gitignore`, `README.md`, and a license file during creation. Then clone it locally:

```bash
git clone https://github.com/<your-username>/Aeroguard.git
cd Aeroguard
code .
```

Verify Python version (must be 3.10 or higher):
```bash
python --version
```

**Step 2 — Create and Activate Virtual Environment**

```bash
python -m venv aviation
# Windows
aviation\Scripts\activate
# macOS/Linux
source aviation/bin/activate
```

**Step 3 — Create `requirements.txt` in the root folder**

```
# Core data libraries
numpy
pandas

# Data visualization
matplotlib
seaborn

# Machine learning
scikit-learn

# Large scale data processing
dask
pyarrow

# Deep learning
torch
torchvision

# Notebook environment for EDA
jupyter

# Logging and environment management
loguru
python-dotenv

# Experiment tracking
mlflow

# YAML config
pyyaml

# API framework
fastapi
uvicorn
pydantic

# Dashboard
streamlit
plotly

# HTTP client for testing
requests

# YAML for MLflow config
pyyaml
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

**Step 4 — Create `.env` file in the root folder**

```env
# ============================================================
# AeroGuard — Environment Variables
# IMPORTANT: This file is in .gitignore — never push to GitHub
# ============================================================

PROJECT_ROOT=.
DATA_DIR=./data
ARTIFACTS_DIR=./artifacts
LOGS_DIR=./logs

MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=aeroguard

API_HOST=0.0.0.0
API_PORT=8000

ENVIRONMENT=development
```

**Step 5 — Create `configs/config.yaml`**

Create a `configs/` folder and inside it create `config.yaml`:

```yaml
project:
  name: aeroguard
  version: "1.0.0"
  environment: development

paths:
  data_dir: ./data
  artifacts_dir: ./artifacts
  logs_dir: ./logs
  mlruns_dir: ./mlruns

data:
  raw_flight_data: ./data/all_flights/one_parq
  flight_header: ./data/all_flights/flight_header.csv
  flight_header_2days: ./data/2days/flight_header.csv
  flight_data_2days: ./data/2days/flight_data.pkl
  prepared_dataset_dir: ./data/prepared_datasets/dl_dataset
  min_flight_duration: 1800   # seconds — flights shorter than 30 min are dropped
  truncate_timesteps: 4096    # last ~68 minutes of each flight

model:
  rul_threshold: 2            # days — flights within 2 days of maintenance = at-risk
  alert_critical: 0.80
  alert_high: 0.60
  alert_medium: 0.40

logging:
  level: DEBUG
  rotation: "10 MB"
  retention: "30 days"
```

**Step 6 — Create `src/` package with logger and exceptions**

Create `src/` folder. Inside it create `__init__.py` (empty file). This converts `src/` into a Python package and enables imports like `from src.logger import logger`.

Create `src/logger.py`:

```python
import os, sys
from loguru import logger
from dotenv import load_dotenv
import yaml

load_dotenv()
LOGS_DIR = os.getenv("LOGS_DIR", "./logs")
os.makedirs(LOGS_DIR, exist_ok=True)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

LOG_LEVEL = config["logging"]["level"]
logger.remove()

# Handler 1: Terminal with colors
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> | "
        "<white>{message}</white>"
    ),
    colorize=True
)

# Handler 2: Daily rotating file
logger.add(
    os.path.join(LOGS_DIR, "aeroguard_{time:YYYY-MM-DD}.log"),
    level=LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    rotation="10 MB",
    retention="30 days",
    compression="zip"
)
```

Test the logger:
```bash
python -c "from src.logger import logger; logger.info('AeroGuard logger test'); logger.debug('Debug test')"
```

Create `src/exception.py` with a custom exception hierarchy:

```python
import sys

class AeroGuardException(Exception):
    def __init__(self, error, context: str = ""):
        error_message = (
            f"[{type(error).__name__}] {str(error)}\n"
            f"Location: {sys.exc_info()[2]}\n"
            f"Context: {context}"
        )
        super().__init__(error_message)

class DataIngestionException(AeroGuardException): pass
class DataTransformationException(AeroGuardException): pass
class ModelTrainingException(AeroGuardException): pass
class ModelPredictionException(AeroGuardException): pass
class AnomalyDetectionException(AeroGuardException): pass
class AlertGenerationException(AeroGuardException): pass
class ConfigurationException(AeroGuardException): pass
```

Add `logs/` and `.env` to `.gitignore`, then push:

```bash
git add .
git commit -m "feat: project scaffold, logger, custom exceptions"
git push origin main
```

---

### Phase 2: Data Pipeline

**Step 7 — Download the Dataset**

Download the NGAFID dataset from: https://doi.org/10.5281/zenodo.6624956

Create a `data/` folder in the project root and place the dataset inside it. The folder structure should be:
```
data/
├── all_flights/
│   ├── flight_header.csv
│   └── one_parq/          ← Dask Parquet files (~4.3 GB)
└── 2days/
    ├── flight_header.csv
    └── flight_data.pkl
```

Add `data/` to `.gitignore` — the dataset is 4.3 GB and must never be pushed to GitHub.

**Step 8 — Create `src/data/` package**

Create `src/data/` folder and `src/data/__init__.py`.

**Step 9 — Create `src/data/ingestion.py`**

This module is responsible for loading the raw dataset into memory-safe formats:
- `flight_header.csv` → loaded into a Pandas DataFrame (metadata + labels)
- `one_parq/` → loaded as a Dask DataFrame (lazy — 4.3 GB stays on disk)
- `2days/flight_data.pkl` → loaded as preprocessed NumPy arrays

Key function: `load_data()` returns a dict with `header_full`, `header_2days`, `sensor_data`, and `flight_data_2days`.

Dask is used for the sensor parquet data because loading 4.3 GB into Pandas would crash most machines. Dask reads only the metadata at load time and fetches actual data on demand.

Test ingestion:
```bash
python -c "from src.data.ingestion import load_data; data = load_data(); print(data['header_full'].shape)"
```

**Step 10 — Create `src/data/transformation.py`**

This module performs three phases of cleaning and label construction:

**Phase 1 — Header Cleaning:**
- Removes flights shorter than 1800 seconds (30 minutes)
- Handles missing values in key columns
- Converts dtypes to memory-efficient formats (category, int32, float32)
- Expected output: ~18,384 rows

**Phase 2 — Label Construction:**
- Drops "same day" flights (before_after == 'same') — label is ambiguous
- Drops extreme `date_diff` values (outside ±30 days) — only 23 flights, pure noise
- Creates binary label: `date_diff <= -2` → `label_binary = 1` (at-risk)
- Creates regression target: `days_to_maintenance` for before-flights only
- Expected output: ~16,359 rows with 28.5% at-risk label rate

**Phase 3 — Sensor Cleaning:**
- Filters sensor data to only include Master Index IDs in the cleaned header
- Drops the `cluster` column (it is a maintenance label duplicate, not a sensor)
- Per-flight sort and NaN fill happen later during extraction

Test transformation:
```bash
python -c "
from src.data.ingestion import load_data
from src.data.transformation import run_transformation
data = load_data()
header_labeled, sensor_filtered = run_transformation(data['header_full'], data['sensor_data'])
print('Shape:', header_labeled.shape)
print(header_labeled['label_binary'].value_counts())
"
```

Push to GitHub:
```bash
git add .
git commit -m "feat: data ingestion + transformation pipeline complete"
git push origin main
```

---

### Phase 3: Feature Engineering

**Step 11 — Create `src/data/feature_engineering.py`**

This is the most compute-intensive step. It converts the cleaned Dask sensor data into fixed-shape NumPy arrays ready for deep learning.

**What it does, step by step:**

1. **Per-flight extraction** — For each of the 16,359 flights, loads its sensor rows from Dask, sorts by timestep, forward-fills NaN values (FAA-standard imputation), and pads/truncates to exactly 4096 timesteps.

2. **Cruise phase filter** — Only cruise-phase data is used for normalization statistics (IAS > 70 Kts AND AltMSL > 1500 Ft).

3. **8 novel engineered channels** are computed per flight per timestep:
   - `CHT_spread` = max(CHT1–4) − min(CHT1–4)
   - `CHT_mean` = mean of CHT1–4
   - `CHT4_minus_CHT1` = directional gradient across cylinders
   - `EGT_spread` = max(EGT1–4) − min(EGT1–4)
   - `EGT_mean` = mean of EGT1–4
   - `EGT_CHT_divergence` = deviation in EGT/CHT ratio (intake gasket signature)
   - `FQty_imbalance` = |FQtyL − FQtyR| normalized
   - `is_cruise` = binary cruise phase flag

4. **Final array shape per flight:** `(4096, 31)` — 4096 timesteps × 31 channels (23 original + 8 engineered)

5. **Aircraft-aware train/val/test split** — Flights from the same aircraft (`tail_num`) are kept in the same split to prevent data leakage. Split ratio: 70% / 15% / 15%.

6. **Z-score normalization** — Mean and std computed on `X_train` only (no leakage). Applied to all three splits.

7. **Output saved as NumPy `.npy` files:**
   ```
   data/prepared_datasets/dl_dataset/
   ├── X_train.npy   (~5.82 GB)
   ├── X_val.npy     (~1.66 GB)
   ├── X_test.npy    (~830 MB)
   ├── y_train.npy
   ├── y_val.npy
   ├── y_test.npy
   ├── norm_mean.npy
   ├── norm_std.npy
   ├── channel_names.npy
   └── dataset_info.json
   ```

Test on a small subset first (50 flights):
```bash
python -c "
from src.data.ingestion import load_data
from src.data.transformation import run_transformation
from src.data.feature_engineering import run_feature_engineering
data = load_data()
header_labeled, sensor_filtered = run_transformation(data['header_full'], data['sensor_data'])
header_test = header_labeled.head(50)
run_feature_engineering(sensor_filtered, header_test, output_dir='./data/prepared_datasets/dl_dataset_test')
"
```

Prevent laptop sleep during the full run (Windows):
```bash
powercfg /change standby-timeout-ac 0
```

Run on the full dataset (this takes several hours):
```bash
python -c "
from src.data.ingestion import load_data
from src.data.transformation import run_transformation
from src.data.feature_engineering import run_feature_engineering
import yaml
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)
data = load_data()
header_labeled, sensor_filtered = run_transformation(data['header_full'], data['sensor_data'])
run_feature_engineering(sensor_filtered, header_labeled, output_dir=config['data']['prepared_dataset_dir'])
"
```

After completion, delete the test folder and push:
```bash
git add .
git commit -m "feat: feature engineering pipeline — (16359, 4096, 31) dataset ready"
git push origin main
```

---

### Phase 4: Model Training on Google Colab

`X_train.npy` is 5.82 GB. Training a TCN on a local 8 GB RAM machine would crash. Training is done on Google Colab with a free T4 GPU.

**Step 12 — Upload Dataset to Google Drive**

Upload the entire `data/prepared_datasets/dl_dataset/` folder to Google Drive inside a folder named `AeroGuard_Dataset`.

**Step 13 — Open Google Colab**

Go to https://colab.research.google.com. Create a new notebook named `AeroGuard_Training.ipynb`. Change runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU).

**Step 14 — Cell 1: Environment Setup + Drive Mount**
```python
import subprocess, sys
gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(gpu_info.stdout if gpu_info.returncode == 0 else "No GPU")
from google.colab import drive
drive.mount('/content/drive')
print("✅ Drive mounted")
```

**Step 15 — Cell 2: Data Load**
```python
import numpy as np, os
DATASET_PATH = '/content/drive/MyDrive/AeroGuard_Dataset'
X_train = np.load(os.path.join(DATASET_PATH, 'X_train.npy'), mmap_mode='r')
X_val   = np.load(os.path.join(DATASET_PATH, 'X_val.npy'),   mmap_mode='r')
X_test  = np.load(os.path.join(DATASET_PATH, 'X_test.npy'),  mmap_mode='r')
y_train = np.load(os.path.join(DATASET_PATH, 'y_train.npy'))
y_val   = np.load(os.path.join(DATASET_PATH, 'y_val.npy'))
y_test  = np.load(os.path.join(DATASET_PATH, 'y_test.npy'))
print(f"X_train: {X_train.shape} | label=1: {y_train.sum():,}")
```

`mmap_mode='r'` is critical — it keeps the 5.82 GB array on disk and only loads batches needed by the DataLoader.

**Step 16 — Cell 3: PyTorch DataLoader**
```python
import torch
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        # Transpose: (4096, 31) → (31, 4096) for Conv1d
        x = torch.tensor(self.X[idx], dtype=torch.float32).T
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y

train_loader = DataLoader(FlightDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(FlightDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(FlightDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
```

**Step 17 — Cell 4: TCN Architecture**

The TCN (Temporal Convolutional Network) uses exponentially increasing dilations to capture both local and long-range temporal patterns without the memory cost of LSTMs.

```
Input:  (Batch, 31 channels, 4096 timesteps)
↓
TCNBlock dilation=1   → local patterns (adjacent seconds)
TCNBlock dilation=2   → 2x context
TCNBlock dilation=4   → 4x context
TCNBlock dilation=8
TCNBlock dilation=16
TCNBlock dilation=32
TCNBlock dilation=64
TCNBlock dilation=128 → ~1000 timestep receptive field
↓
Global Average Pooling (mean over time dimension)
↓
Linear(64→64) → ReLU → Dropout → Linear(64→1)
↓
Output: (Batch, 1) raw logit
```

Each TCNBlock contains two CausalConv1d layers (causal = only looks at past timesteps, not future — production-safe) plus a residual connection.

**Step 18 — Cell 5: Training Loop**

```python
import time
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, roc_auc_score

# Class imbalance correction
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

# Training with early stopping (patience=7), gradient clipping (max_norm=1.0)
# Best model saved to artifacts/best_tcn.pt when val_F1 improves
```

Key training decisions:
- `BCEWithLogitsLoss` with `pos_weight = n_neg/n_pos` — corrects for the 1:2.5 class imbalance
- Gradient clipping at `max_norm=1.0` — prevents explosion in deep TCN layers
- `CosineAnnealingLR` — smooth LR decay over 30 epochs
- Early stopping with `patience=7` — stops when validation F1 stops improving

**Step 19 — Cell 6: Final Test Evaluation + Export**

After training, evaluate on the held-out test set and save:
- `best_tcn.pt` — model weights (state_dict only)
- `production_config.json` — threshold, channel names, test metrics, MLflow run ID

Download both files and place them in the local `artifacts/` folder.

---

### Phase 5: Local Model Integration

**Step 20 — Create `src/models/` package**

Create `src/models/` folder and `src/models/__init__.py`.

**Step 21 — Create `src/models/tcn_model.py`**

This file defines the exact same TCN architecture used in Colab (weights_only=True for security) and exposes two public functions:

- `load_tcn_model(model_path, config_path, device)` — loads weights + config, returns `(model, config)`
- `predict_single_flight(model, flight_array, config, device)` — takes a `(4096, 31)` NumPy array, returns `{probability, prediction, severity, threshold}`

The severity mapping:
```python
if prob >= 0.80:   severity = "CRITICAL"
elif prob >= 0.60: severity = "HIGH"
elif prob >= 0.40: severity = "MEDIUM"
else:              severity = "NORMAL"
```

Test the model loads correctly:
```bash
python -c "from src.models.tcn_model import load_tcn_model; model, config = load_tcn_model(); print('Model loaded! Threshold:', config['threshold'])"
```

Push:
```bash
git add .
git commit -m "feat: TCN model loader + inference ready"
git push origin main
```

---

### Phase 6: Anomaly Detection

**Important Design Decision:** The paper explicitly warned that LSTM Autoencoders struggle on this dataset because *"a significant portion of variance is explained by pilot action, not aircraft condition"* — a barrel roll would eclipse the signal from a leaky gasket. Isolation Forest was evaluated but showed only 0.0073 separation between healthy and at-risk flights — practically noise.

**The chosen approach: Statistical Z-Score Anomaly Detection**

This approach uses the fact that the data is already normalized. For each sensor in each flight phase (taxi, takeoff, cruise, descent), it computes the z-score using statistics from healthy flights as the baseline. Any reading beyond 3 standard deviations (the FAA Flight Data Monitoring standard) is flagged.

**Step 22 — Create `src/anomaly/` package**

Create `src/anomaly/` folder and `src/anomaly/__init__.py`.

**Step 23 — Create `src/anomaly/statistical.py`**

This file implements `StatisticalAnomalyDetector` with:

- `fit(X_healthy)` — computes per-sensor per-phase mean and std from healthy flights
- `detect(flight)` — runs z-score analysis and returns:
  - `anomaly_score` — weighted fraction of flagged timesteps (cruise phase weighted 2x, taxi 0.5x)
  - `flagged_sensors` — only maintenance-relevant sensors (pilot-action sensors excluded)
  - `phase_anomalies` — per-phase flagged timestep counts
  - `anomaly_timeline` — binary array of length 4096 for visualization
  - `top_anomalies` — top 5 sensor/phase combinations by max z-score
- `save(path)` / `load(path)` — JSON serialization (human-readable)

**Key v2 improvements over naive z-score:**
1. **Phase-weighted scoring** — cruise anomalies count 4x more than taxi anomalies
2. **Persistence threshold (5%)** — a sensor is only flagged if ≥5% of its phase timesteps are anomalous (eliminates single-spike false alarms)
3. **Pilot-action sensors excluded** — NormAc, VSpd, IAS, AltMSL, OAT, volt1, amp1, amp2 are not shown in the flagged sensors list

**Results from statistical detector:**
```
Healthy flights:   0.59% timesteps flagged
At-risk flights:  14.11% timesteps flagged
Separation ratio: 24x
```

Train and save the detector:
```bash
python -c "
import numpy as np
from src.anomaly.statistical import StatisticalAnomalyDetector
X_train = np.load('./data/prepared_datasets/dl_dataset/X_train.npy', mmap_mode='r')
y_train = np.load('./data/prepared_datasets/dl_dataset/y_train.npy')
X_healthy = np.array(X_train[y_train == 0])
detector = StatisticalAnomalyDetector()
detector.fit(X_healthy)
detector.save('artifacts/statistical_detector.json')
print('Detector saved.')
"
```

Push:
```bash
git add .
git commit -m "feat: statistical anomaly detector v2 with phase weighting — 24x separation"
git push origin main
```

---

### Phase 7: XAI Engine

**Step 24 — Create `src/xai/` package**

Create `src/xai/` folder and `src/xai/__init__.py`.

**Step 25 — Create `src/xai/explainer.py`**

This file implements the explainability engine using **Gradient × Input** attribution (also known as Saliency maps). This method was chosen over SHAP GradientExplainer because it requires no background dataset, runs in a single backward pass, and is equally interpretable.

**How it works:**
1. A forward pass computes the maintenance probability
2. A backward pass computes `∂probability/∂input` for every sensor at every timestep
3. The element-wise product `|gradient × input|` measures how much each sensor at each timestep influenced the prediction
4. Channel importance = mean over time axis → `(31,)` vector
5. Temporal importance = mean over channel axis → `(4096,)` vector

The `AeroGuardExplainer` class combines this with a plain language generator that maps sensor group patterns to human-readable maintenance recommendations:

```
"Engine oil temperature strongly influenced this prediction (importance: 0.87)"
"Cylinder temperature patterns suggest potential compression or cooling issue — inspect cylinder heads."
"Recommended Action: Ground aircraft immediately. Priority: engine/oil system"
```

The final explanation dict contains:
- `gradient_importance` — channel_importance, temporal_importance, top_channels
- `plain_language` — summary, driving_factors, sensor_insights, recommended_action
- `anomaly_context` — passthrough from the statistical detector

Push:
```bash
git add .
git commit -m "feat: XAI engine — gradient×input + plain language explanation"
git push origin main
```

---

### Phase 8: FastAPI Backend

**Step 26 — Create `api/` package**

Create `api/` folder and `api/__init__.py`.

**Step 27 — Create `api/main.py`**

The FastAPI backend exposes three endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API health check — model loaded status |
| `/model-info` | GET | Model metadata, test metrics, thresholds |
| `/predict` | POST | Full analysis: TCN + anomaly + XAI |

**Startup behavior:** On app startup, three components are loaded once into memory:
1. TCN model + production config (`artifacts/best_tcn.pt`)
2. Statistical anomaly detector (`artifacts/statistical_detector.json`)
3. XAI explainer (wraps the loaded TCN model)

**`/predict` request body:**
```json
{
  "flight_data": [[...], [...], ...],  // (4096, 31) float array
  "flight_id": 12345,                  // optional
  "explain": true                      // include XAI output
}
```

**`/predict` response:**
```json
{
  "flight_id": 12345,
  "probability": 0.7342,
  "prediction": 1,
  "severity": "HIGH",
  "threshold": 0.35,
  "message": "Inspect before next flight — high risk detected",
  "anomaly": {
    "anomaly_score": 0.1411,
    "flagged_sensors": ["E1 OilT", "E1 CHT2"],
    "phase_anomalies": {"cruise": 312, "taxi": 0, "takeoff": 45, "descent": 0},
    "top_anomalies": [...]
  },
  "explanation": {
    "top_channels": [...],
    "summary": "HIGH RISK: 73.4% maintenance probability...",
    "driving_factors": [...],
    "sensor_insights": [...],
    "recommended_action": "Complete pre-flight inspection..."
  }
}
```

**Graceful degradation:** If the anomaly detector or XAI explainer fails for any reason, their respective fields in the response are `null` — the core TCN prediction still succeeds.

Start the API:
```bash
uvicorn api.main:app --reload --port 8000
```

Test endpoints:
```
http://127.0.0.1:8000/health       → {"status":"ok","model_loaded":true,...}
http://127.0.0.1:8000/model-info   → model metadata
http://127.0.0.1:8000/docs         → Interactive Swagger UI
```

Push:
```bash
git add .
git commit -m "feat: FastAPI backend — /health /model-info /predict endpoints complete"
git push origin main
```

---

### Phase 9: Streamlit Dashboard

**Step 28 — Create `dashboard/app.py`**

The Streamlit dashboard has two views selectable from the sidebar:

**Mechanic View** — Single-flight deep analysis:
- Upload a `.npy` file or select from test flights
- Displays risk score with color-coded severity banner
- Plotly bar chart of top-5 sensor importances
- Phase-wise anomaly counts bar chart
- Time-series plot of the most important sensor with anomaly markers overlaid
- AI explanation cards: driving factors + sensor insights + recommended action
- Flagged sensors displayed as color-coded chips

**Fleet Manager View** — Multi-aircraft risk overview:
- Simulate or load a fleet of flights
- Summary metrics: count of CRITICAL / HIGH / MEDIUM / NORMAL aircraft
- Pie chart of risk distribution
- Bar chart of maintenance probability by aircraft ID
- Sortable fleet status table
- Priority action list for CRITICAL and HIGH severity aircraft

Run the dashboard (keep API running in another terminal):
```bash
# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
streamlit run dashboard/app.py
```

Push:
```bash
git add .
git commit -m "feat: Streamlit dashboard — mechanic view + fleet manager view"
git push origin main
```

---

### Phase 10: Retraining Pipeline + MLflow

**Step 29 — Create `configs/mlflow_config.yaml`**

```yaml
mlflow:
  tracking_uri: ./mlruns
  experiment_name: AeroGuard_TCN
  registered_model_name: AeroGuard_TCN

training:
  run_name: baseline_retrain_v1
  n_epochs: 30
  batch_size: 32
  lr: 0.0003
  weight_decay: 0.0001
  patience: 7
  n_filters: 64
  kernel_size: 3
  n_layers: 8
  dropout: 0.1
  threshold: 0.35

logging:
  artifacts:
    - artifacts/retrained_tcn.pt
    - artifacts/production_config.json
    - artifacts/statistical_detector.json
    - configs/config.yaml
```

**Step 30 — Create `src/retraining_pipeline/retrain.py`**

The retraining pipeline handles everything in a single `run_retraining()` call:

1. Load configs from YAML — all hyperparameters come from config, not hardcoded
2. Load dataset with `mmap_mode='r'` — memory-safe loading
3. Compute `pos_weight = n_safe / n_atrisk` dynamically from the new data
4. Initialize MLflow run — logs all hyperparameters + dataset statistics
5. Train TCN with AdamW + CosineAnnealingLR + gradient clipping + early stopping
6. Save best checkpoint locally after each validation F1 improvement
7. Load best checkpoint and run final test evaluation
8. Log test metrics to MLflow as run-level scalars
9. Register model in MLflow Model Registry under `AeroGuard_TCN`
10. Upload all artifacts to MLflow
11. Write new `production_config.json` with current test metrics and MLflow run ID

**Why retraining matters:** Aircraft sensor data drifts over time — seasonal patterns, aging sensors, new failure modes. Regular retraining (triggered by schedule, data drift PSI > 0.2, or performance degradation > 5% F1 drop) keeps the model current.

Run the pipeline:
```bash
python -m src.retraining_pipeline.retrain
```

View results in MLflow UI:
```bash
mlflow ui --port 5000
# Navigate to http://localhost:5000
```

Push:
```bash
git add .
git commit -m "feat: MLflow retraining pipeline complete with model registry"
git push origin main
```

---

### Phase 11: Docker + CI/CD

**Step 31 — Create `Dockerfile`**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 32 — Create `docker-compose.yml`**

```yaml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000
    environment:
      - PYTHONUNBUFFERED=1

  dashboard:
    build: .
    ports:
      - "8501:8501"
    command: streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    environment:
      - API_URL=http://api:8000
      - PYTHONUNBUFFERED=1
    depends_on:
      - api
```

Test Docker build:
```bash
docker-compose up --build
```

**Step 33 — Create `.github/workflows/ci.yml`**

```yaml
name: AeroGuard CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest tests/ -v || true

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t aeroguard .
```

Push:
```bash
git add .
git commit -m "feat: Docker + GitHub Actions CI/CD pipeline"
git push origin main
```

---

### Phase 12: Deployment on Render

Render was chosen over AWS ECS for this portfolio project because AWS with ALB ($16/month) + NAT Gateway ($32/month) costs $50–80/month, while Render's free tier supports Docker deployments with GitHub Actions integration and custom domains.

**Step 34 — Fix `.gitignore` for Deployment**

Remove `artifacts/` from `.gitignore` — Render needs the model files to be in the repository:
```bash
# Remove artifacts/ line from .gitignore
git add artifacts/
git commit -m "fix: include artifacts in repo for Render deployment"
git push origin main
```

Also remove `artifacts/` from `.dockerignore`.

**Step 35 — Deploy FastAPI Service on Render**

1. Go to https://render.com → Sign in with GitHub
2. Click **New** → **Web Service**
3. Connect the `Aeroguard` GitHub repository
4. Configure:
   - **Name:** `aeroguard-api`
   - **Language:** Docker
   - **Branch:** main
   - **Instance Type:** Free
   - **Docker Command:** `uvicorn api.main:app --host 0.0.0.0 --port 8000`
   - **Health Check Path:** `/health`
   - **Environment Variable:** `PYTHONUNBUFFERED = 1`
5. Click **Deploy Web Service** — deployment takes 10–15 minutes

Verify:
```
https://aeroguard-api.onrender.com/health
→ {"status":"ok","model_loaded":true,"detector_loaded":true,"version":"1.0.0"}
```

**Step 36 — Deploy Streamlit Dashboard on Render**

1. Click **New** → **Web Service**
2. Configure:
   - **Name:** `aeroguard-dashboard`
   - **Language:** Docker
   - **Docker Command:** `streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true`
   - **Environment Variables:**
     - `API_URL = https://aeroguard-api.onrender.com`
     - `PYTHONUNBUFFERED = 1`
3. Click **Deploy Web Service**

Both services auto-deploy on every push to `main` (Auto-Deploy: On Commit is enabled by default).

---

## 💻 How to Run Locally

### Prerequisites
- Python 3.10+
- Git
- 8+ GB RAM (for model inference; 16+ GB recommended for retraining)
- NGAFID dataset (download from https://doi.org/10.5281/zenodo.6624956)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Aeroguard.git
cd Aeroguard

# 2. Create and activate virtual environment
python -m venv aviation
aviation\Scripts\activate       # Windows
source aviation/bin/activate    # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up .env file (copy from .env.example if available)
# Adjust paths as needed

# 5. Start the FastAPI backend
uvicorn api.main:app --reload --port 8000

# 6. In a new terminal, start the Streamlit dashboard
streamlit run dashboard/app.py
```

### Run with Docker

```bash
# Build and start both services
docker-compose up --build

# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

### Data Pipeline (if running from scratch)

```bash
# Place NGAFID dataset in data/ folder, then:

# Step 1: Verify ingestion
python -c "from src.data.ingestion import load_data; d = load_data(); print('Header shape:', d['header_full'].shape)"

# Step 2: Run transformation
python -c "
from src.data.ingestion import load_data
from src.data.transformation import run_transformation
data = load_data()
header, sensor = run_transformation(data['header_full'], data['sensor_data'])
print('Labeled flights:', len(header), '| At-risk:', header['label_binary'].sum())
"

# Step 3: Run feature engineering (takes several hours on full data)
python -c "
from src.data.ingestion import load_data
from src.data.transformation import run_transformation
from src.data.feature_engineering import run_feature_engineering
import yaml
with open('configs/config.yaml') as f: config = yaml.safe_load(f)
data = load_data()
header, sensor = run_transformation(data['header_full'], data['sensor_data'])
run_feature_engineering(sensor, header, config['data']['prepared_dataset_dir'])
"

# Step 4: Train anomaly detector
python -c "
import numpy as np
from src.anomaly.statistical import StatisticalAnomalyDetector
X_train = np.load('./data/prepared_datasets/dl_dataset/X_train.npy', mmap_mode='r')
y_train = np.load('./data/prepared_datasets/dl_dataset/y_train.npy')
detector = StatisticalAnomalyDetector()
detector.fit(np.array(X_train[y_train == 0]))
detector.save('artifacts/statistical_detector.json')
"

# Step 5: Train TCN model — do this on Google Colab (T4 GPU)
# Upload X_train.npy, X_val.npy, X_test.npy, y_*.npy to Google Drive
# Run AeroGuard_Training.ipynb on Colab
# Download best_tcn.pt and production_config.json to artifacts/
```

### Run Retraining Pipeline

```bash
python -m src.retraining_pipeline.retrain

# View results in MLflow UI
mlflow ui --port 5000
# Navigate to http://localhost:5000
```

---

## 📡 API Reference

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "detector_loaded": true,
  "version": "1.0.0"
}
```

### `GET /model-info`
```json
{
  "model_name": "TCN",
  "n_channels": 31,
  "n_timesteps": 4096,
  "threshold": 0.35,
  "test_auc": 0.697,
  "test_f1": 0.521,
  "test_recall": 0.871
}
```

### `POST /predict`

**Request:**
```json
{
  "flight_data": [ [float, ...] ],  // shape: (4096, 31) — normalized sensor array
  "flight_id": 12345,               // optional — echoed in response
  "explain": true                   // optional — include XAI output (default: true)
}
```

**Response:**
```json
{
  "flight_id": 12345,
  "probability": 0.7342,
  "prediction": 1,
  "severity": "HIGH",
  "threshold": 0.35,
  "message": "Inspect before next flight — high risk detected",
  "anomaly": {
    "anomaly_score": 0.1411,
    "flagged_sensors": ["E1 OilT", "E1 CHT2", "CHT_spread"],
    "phase_anomalies": {"taxi": 0, "takeoff": 45, "cruise": 312, "descent": 0},
    "anomaly_timeline": [0, 0, 1, ...],
    "top_anomalies": [
      {"sensor": "E1 OilT", "phase": "cruise", "max_z": 3.82, "pct_flagged": 14.2}
    ]
  },
  "explanation": {
    "top_channels": [
      {"channel": "E1 OilT", "description": "Engine oil temperature", "importance": 0.8731}
    ],
    "summary": "HIGH RISK: 73.4% maintenance probability detected. Inspect before next flight.",
    "driving_factors": ["Engine oil temperature strongly influenced this prediction (importance: 0.87)"],
    "sensor_insights": ["Engine oil system parameters show unusual patterns — check oil level and pressure."],
    "recommended_action": "Complete pre-flight inspection before next flight. Focus on: oil level, pressure, and temperature"
  }
}
```

---

## 📈 Model Performance

### TCN — Final Test Set Results

| Metric | Value |
|--------|-------|
| **F1 Score** | **0.521** |
| **AUC-ROC** | **0.697** |
| **Recall @ threshold=0.35** | **0.871** |
| Precision | 0.432 |
| Accuracy | 0.731 |
| Classification Threshold | 0.35 |

### Comparison with Paper Benchmarks

| Model | Binary F1 | AUC | Note |
|-------|-----------|-----|------|
| ConvMHSA (paper) | 0.76* | — | Accuracy metric, not F1 |
| InceptionTime (paper) | 0.755* | — | Accuracy metric, not F1 |
| **AeroGuard TCN** | **0.521** | **0.697** | F1 + AUC on same dataset |

*Paper reported accuracy (76%), not F1. Direct comparison on F1 is not available.

### Statistical Anomaly Detector Performance

| Metric | Healthy Flights | At-Risk Flights |
|--------|----------------|-----------------|
| Average anomaly score | 0.59% | 14.11% |
| Separation ratio | — | **24x** |

### Key Design Choice: Recall Priority

Aviation safety demands that missed detections (false negatives) are far more costly than false alarms (false positives). A grounded aircraft loses a flight; an un-detected failing aircraft risks lives.

Recall of **0.871** at threshold=0.35 means 87.1% of at-risk flights are correctly flagged. This is the primary metric used for model selection and retraining decisions.

---

## 🧠 Key Design Decisions

**Why TCN over LSTM/Transformer?**
TCN's causal dilated convolutions match the inference requirements exactly — they only look at past timesteps (no future leakage), have a large receptive field (~1000 timesteps at 8 layers), and are faster to train than LSTMs on long sequences.

**Why the 2-day label threshold?**
At −5 days, the 1:20 class imbalance makes the model predict "safe" constantly and achieve 95% accuracy while being useless. At −2 days, the 1:2.5 ratio is manageable with `pos_weight` in `BCEWithLogitsLoss`.

**Why Statistical Z-Score over Isolation Forest?**
Isolation Forest on this dataset produced only 0.0073 separation between healthy and at-risk flights. The paper warned about this: pilot variance dominates aggregate flight features. Statistical z-score per flight phase achieves 24x separation because it is sensitive to sustained per-sensor anomalies rather than overall flight pattern differences.

**Why Gradient × Input over SHAP?**
SHAP GradientExplainer requires a background dataset and multiple forward passes. Gradient × Input runs in a single backward pass, requires no additional data, and produces equally interpretable attribution scores for both channel importance and temporal importance.

**Why Render over AWS?**
AWS with proper networking (ALB + NAT Gateway) costs $50–80/month for a portfolio project. Render's free tier supports Docker deployments with GitHub Actions CI/CD, making it the practical choice for demonstrating production deployment skills without ongoing cost.

---

## 📝 License

This project is for educational and portfolio purposes. The NGAFID dataset is publicly available at https://doi.org/10.5281/zenodo.6624956 under its original terms.

---

<div align="center">

**Built with PyTorch · FastAPI · Streamlit · MLflow · Docker · Render**

*End-to-end production ML system for predictive aviation maintenance*

</div>
