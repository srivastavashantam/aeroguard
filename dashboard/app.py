# ============================================================
# AeroGuard — Streamlit Dashboard v2
#
# Views:
#   1. Mechanic View  — Single flight deep analysis
#   2. Fleet View     — All aircraft risk overview
#
# Run:
#   streamlit run dashboard/app.py
# ============================================================

import os
import streamlit as st
import numpy as np
import requests
import plotly.graph_objects as go
import pandas as pd

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "AeroGuard",
    page_icon  = "✈️",
    layout     = "wide",
)

# ── Constants ─────────────────────────────────────────────────
import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

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

SEVERITY_COLORS = {
    'CRITICAL': '#FF4B4B',
    'HIGH'    : '#FF8C00',
    'MEDIUM'  : '#FFD700',
    'NORMAL'  : '#00CC44',
}

SEVERITY_EMOJI = {
    'CRITICAL': '🔴',
    'HIGH'    : '🟠',
    'MEDIUM'  : '🟡',
    'NORMAL'  : '🟢',
}

# Maintenance-relevant sensors only — pilot-action excluded
MAINTENANCE_SENSORS_ORDERED = [
    'E1 OilT', 'E1 OilP', 'E1 RPM',
    'E1 CHT1', 'E1 CHT2', 'E1 CHT3', 'E1 CHT4',
    'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4',
    'E1 FFlow', 'CHT_spread', 'EGT_spread',
    'EGT_CHT_divergence', 'FQty_imbalance',
]

DEFAULT_DATASET_PATH = (
    './data/prepared_datasets/dl_dataset/X_test.npy'
)

# Per-channel anomaly z-threshold for visualization
# 2.5 = slightly relaxed vs detector's 3.0
# Better sensitivity for dashboard display
VIZ_Z_THRESHOLD = 2.5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(ttl=30)
def check_api_health():
    """API health check — 30s cache."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json()
    except:
        return None


def predict_flight(flight_data, flight_id=None):
    """
    FastAPI /predict endpoint call karta hai.
    Returns full dict: prediction + anomaly + explanation.
    """
    try:
        payload = {
            "flight_data": flight_data.tolist(),
            "flight_id"  : flight_id,
            "explain"    : True
        }
        r = requests.post(
            f"{API_URL}/predict",
            json    = payload,
            timeout = 30
        )
        return r.json()
    except Exception as e:
        st.error(f"API call failed: {e}")
        return None


def compute_channel_anomaly(
    signal   : np.ndarray,
    z_thresh : float = VIZ_Z_THRESHOLD
) -> np.ndarray:
    """
    Ek channel ka per-timestep anomaly flag compute karta hai.

    Method: Simple z-score vs flight's own mean/std.
    Flags timesteps where |z| > z_thresh.

    Note: Yeh visualization ke liye hai — detector ke
    training-data-based stats se alag hai.
    Per-channel deviation clearly dikhata hai.

    Args:
        signal  : (T,) normalized channel values
        z_thresh: z-score threshold (default 2.5)

    Returns:
        np.ndarray: (T,) binary anomaly flags
    """
    mean = float(signal.mean())
    std  = float(signal.std())
    if std < 1e-6:
        std = 1.0
    z    = np.abs((signal - mean) / std)
    return (z > z_thresh).astype(np.int8)


def generate_realistic_flight(severity: str) -> np.ndarray:
    """
    Realistic Cessna 172 flight simulate karta hai.

    TCN model real NGAFID data pe trained hai isliye
    anomaly injection strong honi chahiye taaki model
    detect kar sake.

    Anomaly strategy:
      CRITICAL : Multiple sensors, sustained, strong
                 OilP -4σ, OilT +3σ, CHT_spread +4σ
                 RPM +2σ, EGT_spread +3σ
                 50% of cruise phase affected
      HIGH     : OilP -2.5σ, CHT_spread +2.5σ
                 30% of cruise phase
      MEDIUM   : OilP -1.5σ, CHT_spread +1.5σ
                 15% of cruise phase
      NORMAL   : No anomaly
    """
    
    T   = 4096
    C   = 31
    arr = np.zeros((T, C), dtype=np.float32)

    noise = np.random.randn(T, C).astype(np.float32) * 0.1

    taxi_end    = 300
    takeoff_end = 600
    cruise_end  = 3200

    # ── IAS (idx=19) ──────────────────────────────────────────
    ias = np.zeros(T)
    ias[:taxi_end]              = -0.40 + \
        np.random.randn(taxi_end) * 0.05
    ias[taxi_end:takeoff_end]   = np.linspace(
        -0.40, 0.55, takeoff_end - taxi_end
    )
    ias[takeoff_end:cruise_end] = 0.55 + \
        np.random.randn(cruise_end - takeoff_end) * 0.05
    ias[cruise_end:]            = np.linspace(
        0.55, -0.30, T - cruise_end
    )
    arr[:, 19] = ias + noise[:, 19]

    # ── AltMSL (idx=22) ───────────────────────────────────────
    alt = np.zeros(T)
    alt[:taxi_end]              = -1.20 + \
        np.random.randn(taxi_end) * 0.05
    alt[taxi_end:takeoff_end]   = np.linspace(
        -1.20, 0.30, takeoff_end - taxi_end
    )
    alt[takeoff_end:cruise_end] = 0.30 + \
        np.random.randn(cruise_end - takeoff_end) * 0.03
    alt[cruise_end:]            = np.linspace(
        0.30, -1.10, T - cruise_end
    )
    arr[:, 22] = alt + noise[:, 22]

    # ── RPM (idx=9) ───────────────────────────────────────────
    rpm = np.zeros(T)
    rpm[:taxi_end]              = -0.60 + \
        np.random.randn(taxi_end) * 0.05
    rpm[taxi_end:takeoff_end]   = np.linspace(
        -0.60, 0.45, takeoff_end - taxi_end
    )
    rpm[takeoff_end:cruise_end] = 0.45 + \
        np.random.randn(cruise_end - takeoff_end) * 0.04
    rpm[cruise_end:]            = np.linspace(
        0.45, -0.40, T - cruise_end
    )
    arr[:, 9]  = rpm + noise[:, 9]

    # Engine sensors
    arr[:, 7]  = rpm * 0.6 + noise[:, 7] * 0.1
    arr[:, 8]  = rpm * 0.7 + noise[:, 8] * 0.1
    arr[:, 6]  = rpm * 0.4 + noise[:, 6] * 0.1

    # Cylinder temps
    for i in range(10, 14):
        arr[:, i] = rpm * 0.5 + noise[:, i] * 0.15
    for i in range(14, 18):
        arr[:, i] = rpm * 0.8 + noise[:, i] * 0.15

    # Fuel
    fuel = np.linspace(0.3, -0.3, T)
    arr[:, 4] = fuel + noise[:, 4] * 0.05
    arr[:, 5] = fuel + noise[:, 5] * 0.05

    # Electrical
    arr[:, 0] = 1.0 + noise[:, 0] * 0.1
    arr[:, 2] = 0.5 + noise[:, 2] * 0.1

    # Novel channels
    arr[:, 23] = np.abs(noise[:, 23]) * 0.3
    arr[:, 24] = arr[:, 10:14].mean(axis=1)
    arr[:, 25] = arr[:, 13] - arr[:, 10]
    arr[:, 26] = np.abs(noise[:, 26]) * 0.3
    arr[:, 27] = arr[:, 14:18].mean(axis=1)
    arr[:, 28] = arr[:, 27] - arr[:, 24]
    arr[:, 29] = arr[:, 4] - arr[:, 5] + \
        noise[:, 29] * 0.05
    arr[:, 30] = (
        (arr[:, 19] > 0.531) & (arr[:, 22] > -0.277)
    ).astype(np.float32)

    # ── Anomaly injection ─────────────────────────────────────
    cruise_start = takeoff_end
    cruise_len   = cruise_end - cruise_start

    if severity == "CRITICAL":
        # Severe multi-sensor failure — 50% cruise affected
        s = cruise_start + int(cruise_len * 0.2)
        e = cruise_start + int(cruise_len * 0.7)

        # OilP severe drop — oil system failure
        arr[s:e, 8]  -= 4.0
        # OilT severe rise — overheating
        arr[s:e, 7]  += 3.0
        # RPM erratic — engine stress
        arr[s:e, 9]  += 2.0 + \
            np.random.randn(e-s) * 0.5
        # CHT_spread — multi cylinder issue
        arr[s:e, 23] += 4.0
        # EGT_spread — combustion imbalance
        arr[s:e, 26] += 3.0
        # EGT_CHT_divergence — intake gasket
        arr[s:e, 28] += 3.0
        # CHT individual cylinders
        arr[s:e, 10] += 2.0
        arr[s:e, 13] -= 1.5   # CHT4 different — imbalance
        # FQty_imbalance — fuel asymmetry
        arr[s:e, 29] += 2.5

    elif severity == "HIGH":
        # Moderate oil + cylinder issue — 30% cruise
        s = cruise_start + int(cruise_len * 0.3)
        e = cruise_start + int(cruise_len * 0.6)
        arr[s:e, 8]  -= 2.5
        arr[s:e, 7]  += 1.8
        arr[s:e, 23] += 2.5
        arr[s:e, 26] += 1.8
        arr[s:e, 28] += 1.5

    elif severity == "MEDIUM":
        # Early warning signs — 15% cruise
        s = cruise_start + int(cruise_len * 0.45)
        e = cruise_start + int(cruise_len * 0.60)
        arr[s:e, 8]  -= 1.5
        arr[s:e, 23] += 1.5
        arr[s:e, 26] += 1.0

    return arr

def load_real_flight(
    dataset_path : str,
    flight_label : str,
    seed         : int,
) -> tuple | None:
    """
    X_test.npy se ek real flight randomly load karta hai.

    Args:
        dataset_path : path to X_test.npy
        flight_label : "Any" | "Safe (label=0)" |
                       "At-Risk (label=1)"
        seed         : random seed

    Returns:
        (flight_array, flight_id, true_label) or None
    """
    try:
        y_path   = dataset_path.replace(
            'X_test.npy', 'y_test.npy'
        )
        ids_path = dataset_path.replace(
            'X_test.npy', 'ids_test.npy'
        )

        if not os.path.exists(dataset_path):
            st.error(f"File not found: {dataset_path}")
            return None

        X_test = np.load(dataset_path, mmap_mode='r')
        y_test = np.load(y_path)
        ids    = (
            np.load(ids_path)
            if os.path.exists(ids_path)
            else np.arange(len(y_test))
        )

        if flight_label == "Safe (label=0)":
            indices = np.where(y_test == 0)[0]
        elif flight_label == "At-Risk (label=1)":
            indices = np.where(y_test == 1)[0]
        else:
            indices = np.arange(len(y_test))

        if len(indices) == 0:
            st.error("No flights found for selected label")
            return None

        rng    = np.random.RandomState(seed)
        idx    = rng.choice(indices)
        flight = np.array(X_test[idx])
        fid    = int(ids[idx])
        label  = int(y_test[idx])

        return flight, fid, label

    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    st.sidebar.image(
        "https://img.icons8.com/color/96/"
        "airplane-take-off.png"
    )
    st.sidebar.title("AeroGuard")
    st.sidebar.caption("Aircraft Health Monitoring System")

    health = check_api_health()
    if health and health.get('status') == 'ok':
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Offline — start uvicorn")

    st.sidebar.divider()

    view = st.sidebar.radio(
        "Select View",
        ["🔧 Mechanic View", "📊 Fleet Manager View"],
        label_visibility="collapsed"
    )

    st.sidebar.divider()
    st.sidebar.caption("Model: TCN | Threshold: 0.35")
    st.sidebar.caption("Dataset: NGAFID Cessna 172")
    st.sidebar.caption("Flights: 16,359 | AUC: 0.697")

    return view


# ============================================================
# MECHANIC VIEW
# ============================================================

def render_mechanic_view():
    st.title("🔧 Mechanic View — Single Flight Analysis")
    st.subheader("Flight Data Input")

    input_method = st.radio(
        "Input method",
        ["Upload .npy file", "Generate / Load test flight"],
        horizontal=True
    )

    flight_array = None
    flight_id    = None

    # ── Option 1: Upload ──────────────────────────────────────
    if input_method == "Upload .npy file":
        uploaded = st.file_uploader(
            "Upload normalized flight array (.npy)",
            type=['npy']
        )
        if uploaded:
            try:
                flight_array = np.load(uploaded)
                flight_id    = st.number_input(
                    "Flight ID (optional)",
                    value=0, step=1
                )
                st.success(
                    f"✅ Flight loaded — "
                    f"shape: {flight_array.shape}"
                )
            except Exception as e:
                st.error(f"Failed to load: {e}")

    # ── Option 2: Simulate or load real ──────────────────────
    else:
        test_mode = st.radio(
            "Test mode",
            [
                "🎲 Simulate realistic flight",
                "📂 Load real flight from dataset"
            ],
            horizontal=True
        )

        if test_mode == "🎲 Simulate realistic flight":
            col_a, col_b = st.columns(2)
            with col_a:
                severity_sim = st.selectbox(
                    "Simulate severity",
                    ["NORMAL", "MEDIUM", "HIGH", "CRITICAL"]
                )
            with col_b:
                flight_id = st.number_input(
                    "Flight ID", value=99999, step=1
                )

            if st.button(
                "🎲 Generate Realistic Flight",
                type="primary"
            ):
                flight_array = generate_realistic_flight(
                    severity_sim
                )
                st.session_state['flight_array'] = \
                    flight_array
                st.session_state['flight_id'] = \
                    int(flight_id)
                st.success(
                    f"✅ Realistic {severity_sim} flight "
                    f"generated"
                )

        else:
            dataset_path = st.text_input(
                "Dataset path (X_test.npy)",
                value=DEFAULT_DATASET_PATH
            )
            col_a, col_b = st.columns(2)
            with col_a:
                flight_label = st.selectbox(
                    "Flight type",
                    [
                        "Any",
                        "Safe (label=0)",
                        "At-Risk (label=1)"
                    ]
                )
            with col_b:
                random_seed = st.number_input(
                    "Random seed",
                    value=42, step=1
                )

            if st.button(
                "📂 Load Real Flight",
                type="primary"
            ):
                res = load_real_flight(
                    dataset_path,
                    flight_label,
                    int(random_seed)
                )
                if res:
                    flight_array, fid, true_label = res
                    st.session_state['flight_array'] = \
                        flight_array
                    st.session_state['flight_id'] = fid
                    label_str = (
                        "⚠️ At-Risk"
                        if true_label == 1
                        else "✅ Safe"
                    )
                    st.success(
                        f"✅ Real flight loaded — "
                        f"ID: {fid} | "
                        f"True label: {label_str}"
                    )

        if 'flight_array' in st.session_state:
            flight_array = st.session_state['flight_array']
            flight_id    = st.session_state.get(
                'flight_id', 99999
            )

    # ── Analyze ───────────────────────────────────────────────
    if flight_array is not None:
        if flight_array.shape != (4096, 31):
            st.error(
                f"Expected shape (4096, 31), "
                f"got {flight_array.shape}"
            )
            return

        if st.button("🚀 Analyze Flight", type="primary"):
            with st.spinner("Analyzing flight data..."):
                result = predict_flight(
                    flight_array, int(flight_id)
                )
            if result:
                render_analysis_results(
                    result, flight_array
                )


# ============================================================
# ANALYSIS RESULTS
# ============================================================

def render_analysis_results(result, flight_array):
    """
    Full analysis output render karta hai.

    Sections:
      1. Alert banner
      2. Key metrics (3 cards)
      3. Sensor timelines — top 3 channels
         Per-channel anomaly computed locally (z > 2.5)
         Combined with global anomaly timeline from API
      4. Channel importance + phase anomalies
      5. AI explanation
      6. Flagged maintenance sensors
    """
    severity = result['severity']
    color    = SEVERITY_COLORS[severity]
    emoji    = SEVERITY_EMOJI[severity]

    # Global anomaly timeline from API
    global_anom = np.array(
        result.get('anomaly', {}).get(
            'anomaly_timeline', [0] * 4096
        )
    )

    st.divider()

    # ── 1. Alert banner ───────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background-color:{color}22;
            border-left:6px solid {color};
            padding:18px 24px;
            border-radius:6px;
            margin-bottom:24px;
        ">
            <h2 style="color:{color}; margin:0;">
                {emoji} {severity} ALERT
            </h2>
            <p style="margin:6px 0 0 0; font-size:16px;">
                {result['message']}
            </p>
            <p style="
                margin:4px 0 0 0;
                font-size:14px; opacity:0.8;
            ">
                Maintenance probability:
                <strong>
                    {result['probability']*100:.1f}%
                </strong>
                &nbsp;|&nbsp;
                Alert threshold: 35%
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── 2. Key metrics ────────────────────────────────────────
    # Compute meaningful anomaly metrics locally
    # API anomaly score uses training-data z-scores
    # Local per-flight z-score shows relative deviations

    # Per-flight anomaly score — fraction of timesteps
    # where ANY maintenance sensor exceeds 2.5σ
    maint_indices = [
        CHANNEL_NAMES.index(s)
        for s in MAINTENANCE_SENSORS_ORDERED
        if s in CHANNEL_NAMES
    ]
    local_anom_flags = np.zeros(4096, dtype=np.int8)
    local_flagged    = set()

    for idx in maint_indices:
        ch_anom = compute_channel_anomaly(
            flight_array[:, idx], VIZ_Z_THRESHOLD
        )
        # Persistence check — >= 3% of flight
        if ch_anom.mean() >= 0.03:
            local_anom_flags = np.maximum(
                local_anom_flags, ch_anom
            )
            local_flagged.add(CHANNEL_NAMES[idx])

    local_score = float(local_anom_flags.mean())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🎯 Maintenance Probability",
            f"{result['probability']*100:.1f}%",
        )

    with col2:
        if local_score > 0.20:
            anom_label = "High anomaly activity"
            delta_col  = "inverse"
        elif local_score > 0.08:
            anom_label = "Moderate anomaly activity"
            delta_col  = "inverse"
        else:
            anom_label = "Low anomaly activity"
            delta_col  = "normal"
        st.metric(
            "⚠️ Flight Anomaly Score",
            f"{local_score*100:.1f}%",
            delta       = anom_label,
            delta_color = delta_col,
        )

    with col3:
        maint_flagged_list = sorted(list(local_flagged))
        st.metric(
            "🔧 Maintenance Sensors Flagged",
            f"{len(maint_flagged_list)} / "
            f"{len(MAINTENANCE_SENSORS_ORDERED)}",
        )

    st.divider()

    # ── 3. Sensor timelines ───────────────────────────────────
    st.subheader(
        "📈 Sensor Timelines — Top 3 Contributing Channels"
    )
    st.caption(
        "🔴 Red markers = anomalous readings (>2.5σ from "
        "flight mean) | "
        "Shaded red = anomaly duration"
    )

    if result.get('explanation'):
        top_channels = \
            result['explanation']['top_channels'][:3]
        timesteps    = np.arange(4096)

        for ch_info in top_channels:
            ch_name = ch_info['channel']
            if ch_name not in CHANNEL_NAMES:
                continue

            ch_idx  = CHANNEL_NAMES.index(ch_name)
            signal  = flight_array[:, ch_idx]

            # Smoothed signal for clarity
            window        = 20
            signal_smooth = np.convolve(
                signal,
                np.ones(window) / window,
                mode='same'
            )

            # Per-channel anomaly — local z-score
            ch_anom = compute_channel_anomaly(
                signal, VIZ_Z_THRESHOLD
            )

            # Combine with global API anomaly
            combined    = np.maximum(ch_anom, global_anom)
            anom_points = np.where(combined == 1)[0]
            anom_dur    = int(combined.sum())
            anom_pct    = float(combined.mean()) * 100

            fig = go.Figure()

            # Smoothed main signal
            fig.add_trace(go.Scatter(
                x    = timesteps,
                y    = signal_smooth,
                mode = 'lines',
                name = f"{ch_name} (smoothed)",
                line = dict(color=color, width=2),
            ))

            # Raw signal faint overlay
            fig.add_trace(go.Scatter(
                x       = timesteps,
                y       = signal,
                mode    = 'lines',
                name    = 'Raw signal',
                line    = dict(
                    color=color, width=0.5, dash='dot'
                ),
                opacity = 0.2,
            ))

            # Red X markers at anomaly points
            if len(anom_points) > 0:
                fig.add_trace(go.Scatter(
                    x      = anom_points,
                    y      = signal[anom_points],
                    mode   = 'markers',
                    name   = 'Anomaly (>2.5σ)',
                    marker = dict(
                        color='red', size=5, symbol='x'
                    ),
                ))

                # Shade contiguous anomaly segments
                segments = []
                start    = anom_points[0]
                end      = anom_points[0]
                for pt in anom_points[1:]:
                    if pt - end <= 15:
                        end = pt
                    else:
                        segments.append((start, end))
                        start = pt
                        end   = pt
                segments.append((start, end))

                for seg_s, seg_e in segments[:15]:
                    fig.add_vrect(
                        x0         = seg_s,
                        x1         = seg_e + 1,
                        fillcolor  = "red",
                        opacity    = 0.12,
                        line_width = 0,
                    )

            fig.update_layout(
                title = (
                    f"{ch_name} — "
                    f"importance: {ch_info['importance']:.3f}"
                    f" | anomalous: {anom_dur} timesteps"
                    f" ({anom_pct:.1f}% of flight)"
                ),
                height        = 300,
                margin        = dict(l=0, r=0, t=45, b=0),
                xaxis_title   = "Timestep (seconds)",
                yaxis_title   = "Normalized Value",
                plot_bgcolor  = 'rgba(0,0,0,0)',
                paper_bgcolor = 'rgba(0,0,0,0)',
                legend        = dict(
                    orientation='h',
                    yanchor='bottom', y=1.02,
                    xanchor='right', x=1
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 4. Channel importance + phase anomalies ───────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("🔍 Top Contributing Sensors")
        if result.get('explanation'):
            top_ch  = result['explanation']['top_channels']
            fig_bar = go.Figure(go.Bar(
                x            = [
                    c['importance'] for c in top_ch
                ],
                y            = [
                    c['channel'] for c in top_ch
                ],
                orientation  = 'h',
                text         = [
                    f"{c['importance']:.3f}"
                    for c in top_ch
                ],
                textposition = 'outside',
                marker_color = color,
                hovertext    = [
                    c['description'] for c in top_ch
                ],
            ))
            fig_bar.update_layout(
                height        = 260,
                margin        = dict(l=0, r=60, t=10, b=0),
                xaxis_title   = "Importance Score",
                yaxis         = dict(autorange='reversed'),
                plot_bgcolor  = 'rgba(0,0,0,0)',
                paper_bgcolor = 'rgba(0,0,0,0)',
            )
            st.plotly_chart(
                fig_bar, use_container_width=True
            )

    with col_r:
        st.subheader("⚠️ Anomalies by Flight Phase")
        if result.get('anomaly'):
            phase_anom = result['anomaly']['phase_anomalies']
            phases     = list(phase_anom.keys())
            counts     = list(phase_anom.values())
            total      = max(sum(counts), 1)

            fig_phase = go.Figure(go.Bar(
                x    = phases,
                y    = counts,
                text = [
                    f"{c}<br>({c/total*100:.1f}%)"
                    for c in counts
                ],
                textposition = 'outside',
                marker_color = [
                    color if c > 0 else '#444444'
                    for c in counts
                ],
            ))
            fig_phase.update_layout(
                height        = 260,
                margin        = dict(l=0, r=0, t=10, b=0),
                yaxis_title   = "Weighted Anomalous Timesteps",
                plot_bgcolor  = 'rgba(0,0,0,0)',
                paper_bgcolor = 'rgba(0,0,0,0)',
            )
            st.plotly_chart(
                fig_phase, use_container_width=True
            )

    st.divider()

    # ── 5. AI Explanation ─────────────────────────────────────
    if result.get('explanation'):
        st.subheader("💡 AI Explanation")
        plain = result['explanation']

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Driving Factors:**")
            for f in plain['driving_factors']:
                st.markdown(f"• {f}")
        with col_b:
            st.markdown("**Sensor Insights:**")
            for ins in plain['sensor_insights']:
                st.markdown(f"• {ins}")

        st.markdown(
            f"""
            <div style="
                background:{color}11;
                border:1px solid {color};
                padding:12px 16px;
                border-radius:5px;
                margin-top:10px;
            ">
            <strong>✅ Recommended Action:</strong><br>
            {plain['recommended_action']}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── 6. Flagged maintenance sensors ────────────────────────
    if maint_flagged_list:
        st.divider()
        st.subheader(
            f"🚨 Flagged Maintenance Sensors "
            f"({len(maint_flagged_list)})"
        )
        st.caption(
            "Sensors with persistent anomalies (>3% of flight "
            "timesteps exceed 2.5σ). "
            "Pilot-action sensors excluded."
        )

        # Top anomaly details from API
        top_anom     = result.get(
            'anomaly', {}
        ).get('top_anomalies', [])
        top_anom_map = {
            a['sensor']: a for a in top_anom
        }

        cols = st.columns(min(len(maint_flagged_list), 4))
        for i, sensor in enumerate(maint_flagged_list):
            with cols[i % 4]:
                # Local anomaly stats for this sensor
                s_idx    = CHANNEL_NAMES.index(sensor) \
                    if sensor in CHANNEL_NAMES else -1
                pct_local = 0.0
                if s_idx >= 0:
                    ch_a = compute_channel_anomaly(
                        flight_array[:, s_idx],
                        VIZ_Z_THRESHOLD
                    )
                    pct_local = float(ch_a.mean()) * 100

                # API detail if available
                detail = top_anom_map.get(sensor, {})
                phase  = detail.get('phase', '')
                maxz   = detail.get('max_z', 0)

                st.markdown(
                    f"""
                    <div style="
                        background:{color}22;
                        border:1px solid {color}88;
                        padding:8px 10px;
                        border-radius:6px;
                        margin:4px 0;
                        font-size:13px;
                    ">
                    <strong>{sensor}</strong><br>
                    <span style="opacity:0.8;">
                        {pct_local:.1f}% anomalous
                        {f'| peak in {phase}' if phase else ''}
                        {f'| z={maxz}' if maxz else ''}
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# FLEET MANAGER VIEW
# ============================================================

def render_fleet_view():
    """
    Fleet Manager View — Real X_test.npy flights use karta hai.
    """
    st.title("📊 Fleet Manager View — Risk Overview")

    col1, col2 = st.columns([2, 1])
    with col1:
        dataset_path = st.text_input(
            "Dataset path (X_test.npy)",
            value=DEFAULT_DATASET_PATH
        )
    with col2:
        n_flights = st.slider(
            "Number of aircraft",
            min_value=5, max_value=30, value=10
        )

    col_a, col_b = st.columns(2)
    with col_a:
        flight_filter = st.selectbox(
            "Flight type",
            ["Any", "Safe (label=0)", "At-Risk (label=1)"]
        )
    with col_b:
        random_seed = st.number_input(
            "Random seed", value=42, step=1
        )

    if st.button("📂 Load Fleet from Dataset",
                 type="primary"):
        with st.spinner(
            f"Loading and analyzing {n_flights} real flights..."
        ):
            fleet = load_fleet_from_dataset(
                dataset_path, n_flights,
                flight_filter, int(random_seed)
            )
            if fleet:
                st.session_state['fleet_results'] = fleet
                st.success(
                    f"✅ {len(fleet)} real flights analyzed"
                )

    if 'fleet_results' in st.session_state:
        render_fleet_results(
            st.session_state['fleet_results']
        )


def load_fleet_from_dataset(
    dataset_path : str,
    n_flights    : int,
    flight_filter: str,
    seed         : int,
) -> list:
    """
    X_test.npy se N real flights load karke analyze karta hai.
    """
    try:
        y_path   = dataset_path.replace(
            'X_test.npy', 'y_test.npy'
        )
        ids_path = dataset_path.replace(
            'X_test.npy', 'ids_test.npy'
        )

        if not os.path.exists(dataset_path):
            st.error(f"File not found: {dataset_path}")
            return []

        X_test = np.load(dataset_path, mmap_mode='r')
        y_test = np.load(y_path)
        ids    = (
            np.load(ids_path)
            if os.path.exists(ids_path)
            else np.arange(len(y_test))
        )

        # Filter by label
        if flight_filter == "Safe (label=0)":
            indices = np.where(y_test == 0)[0]
        elif flight_filter == "At-Risk (label=1)":
            indices = np.where(y_test == 1)[0]
        else:
            indices = np.arange(len(y_test))

        # Random selection
        rng      = np.random.RandomState(seed)
        selected = rng.choice(
            indices,
            size    = min(n_flights, len(indices)),
            replace = False
        )

        results = []
        progress = st.progress(0)

        for i, idx in enumerate(selected):
            flight = np.array(X_test[idx])
            fid    = int(ids[idx])
            label  = int(y_test[idx])

            r = predict_flight(flight, flight_id=fid)
            if r:
                r['aircraft_id'] = f"N{1000+i}"
                r['flight_id']   = fid
                r['true_label']  = label

                # Local anomaly score
                maint_idx = [
                    CHANNEL_NAMES.index(s)
                    for s in MAINTENANCE_SENSORS_ORDERED
                    if s in CHANNEL_NAMES
                ]
                local_flags = np.zeros(4096, dtype=np.int8)
                local_count = 0
                for ch_idx in maint_idx:
                    ch_a = compute_channel_anomaly(
                        flight[:, ch_idx], VIZ_Z_THRESHOLD
                    )
                    if ch_a.mean() >= 0.03:
                        local_flags = np.maximum(
                            local_flags, ch_a
                        )
                        local_count += 1

                r['local_anomaly_score']   = \
                    float(local_flags.mean())
                r['local_sensors_flagged'] = local_count
                r['flight_array']          = flight
                results.append(r)

            progress.progress((i + 1) / len(selected))

        progress.empty()
        return results

    except Exception as e:
        st.error(f"Failed to load fleet: {e}")
        return []

def simulate_fleet(n_flights: int) -> list:
    """
    N aircraft simulate karta hai.
    Har aircraft ka alag seed — different results.
    """
    results = []
    master_rng = np.random.RandomState(123)
    sevs   = ['NORMAL', 'MEDIUM', 'HIGH', 'CRITICAL']
    probs  = [0.4, 0.3, 0.2, 0.1]

    for i in range(n_flights):
        sev        = master_rng.choice(sevs, p=probs)
        # Har aircraft ka alag seed
        per_seed   = int(master_rng.randint(0, 99999))
        np.random.seed(per_seed)

        flight = generate_realistic_flight(sev)

        r = predict_flight(flight, flight_id=1000 + i)
        if r:
            r['aircraft_id']   = f"N{1000+i}"
            r['simulated_sev'] = sev

            # Local anomaly score
            maint_idx = [
                CHANNEL_NAMES.index(s)
                for s in MAINTENANCE_SENSORS_ORDERED
                if s in CHANNEL_NAMES
            ]
            local_flags = np.zeros(4096, dtype=np.int8)
            local_count = 0
            for idx in maint_idx:
                ch_a = compute_channel_anomaly(
                    flight[:, idx], VIZ_Z_THRESHOLD
                )
                if ch_a.mean() >= 0.03:
                    local_flags = np.maximum(
                        local_flags, ch_a
                    )
                    local_count += 1

            r['local_anomaly_score']   = \
                float(local_flags.mean())
            r['local_sensors_flagged'] = local_count
            results.append(r)

    return results

def render_fleet_results(fleet: list):
    """Fleet results render karta hai."""

    st.divider()

    severity_counts = {}
    for r in fleet:
        s = r['severity']
        severity_counts[s] = severity_counts.get(s, 0) + 1

    total = len(fleet)

    # ── Summary metrics ───────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n = severity_counts.get('CRITICAL', 0)
        st.metric(
            "🔴 Critical", f"{n} / {total}",
            delta="Ground immediately" if n else None,
            delta_color="inverse" if n else "normal"
        )
    with col2:
        n = severity_counts.get('HIGH', 0)
        st.metric(
            "🟠 High Risk", f"{n} / {total}",
            delta="Inspect before flight" if n else None,
            delta_color="inverse" if n else "normal"
        )
    with col3:
        st.metric(
            "🟡 Medium",
            f"{severity_counts.get('MEDIUM', 0)} / {total}"
        )
    with col4:
        st.metric(
            "🟢 Normal",
            f"{severity_counts.get('NORMAL', 0)} / {total}"
        )

    # Fleet health banner
    critical_high = (
        severity_counts.get('CRITICAL', 0) +
        severity_counts.get('HIGH', 0)
    )
    if critical_high > 0:
        st.warning(
            f"⚠️ {critical_high} aircraft require "
            f"immediate attention before next flight."
        )
    else:
        st.success(
            "✅ No critical or high-risk aircraft detected."
        )

    st.divider()

    # ── Charts ────────────────────────────────────────────────
    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.subheader("Fleet Risk Distribution")
        ordered = [
            s for s in
            ['CRITICAL', 'HIGH', 'MEDIUM', 'NORMAL']
            if s in severity_counts
        ]
        fig_pie = go.Figure(go.Pie(
            labels       = ordered,
            values       = [severity_counts[s] for s in ordered],
            marker_colors= [SEVERITY_COLORS[s] for s in ordered],
            hole         = 0.45,
            textinfo     = 'label+percent',
        ))
        fig_pie.update_layout(
            height        = 300,
            margin        = dict(l=0, r=0, t=20, b=0),
            paper_bgcolor = 'rgba(0,0,0,0)',
            showlegend    = False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.subheader("Maintenance Probability by Aircraft")
        sorted_fleet = sorted(
            fleet, key=lambda x: x['probability'],
            reverse=True
        )
        ids    = [r['aircraft_id'] for r in sorted_fleet]
        probs  = [r['probability']*100 for r in sorted_fleet]
        colors = [
            SEVERITY_COLORS[r['severity']]
            for r in sorted_fleet
        ]

        fig_bar = go.Figure(go.Bar(
            x            = ids,
            y            = probs,
            marker_color = colors,
            text         = [f"{p:.0f}%" for p in probs],
            textposition = 'outside',
        ))
        fig_bar.add_hline(
            y=35, line_dash="dash",
            line_color="rgba(255,255,255,0.4)",
            annotation_text="Alert threshold (35%)",
            annotation_font_color="rgba(255,255,255,0.6)"
        )
        fig_bar.update_layout(
            height        = 300,
            margin        = dict(l=0, r=0, t=20, b=40),
            yaxis_title   = "Maintenance Probability (%)",
            yaxis_range   = [0, 115],
            plot_bgcolor  = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)',
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Anomaly scores ────────────────────────────────────────
    st.subheader("📊 Flight Anomaly Scores by Aircraft")
    st.caption(
        "Fraction of timesteps where maintenance sensors "
        "exceed 2.5σ from flight mean."
    )

    local_scores = [
        r.get('local_anomaly_score', 0)*100
        for r in sorted_fleet
    ]
    fig_anom = go.Figure(go.Bar(
        x            = ids,
        y            = local_scores,
        marker_color = colors,
        text         = [f"{s:.1f}%" for s in local_scores],
        textposition = 'outside',
    ))
    fig_anom.update_layout(
        height        = 250,
        margin        = dict(l=0, r=0, t=10, b=40),
        yaxis_title   = "Anomaly Score (%)",
        yaxis_range   = [0, max(local_scores)*1.4+2],
        plot_bgcolor  = 'rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    st.divider()

    # ── Fleet table ───────────────────────────────────────────
    st.subheader("Fleet Status Table")

    rows = []
    for r in sorted(
        fleet, key=lambda x: x['probability'], reverse=True
    ):
        emoji      = SEVERITY_EMOJI[r['severity']]
        top_sensor = (
            r.get('explanation', {})
             .get('top_channels', [{}])[0]
             .get('channel', 'N/A')
        )
        phase_anom  = r.get('anomaly', {}).get(
            'phase_anomalies', {}
        )
        worst_phase = (
            max(phase_anom, key=phase_anom.get)
            if phase_anom and any(phase_anom.values())
            else 'N/A'
        )
        true_label = r.get('true_label', -1)
        label_str  = (
            "⚠️ At-Risk" if true_label == 1
            else "✅ Safe" if true_label == 0
            else "—"
        )

        rows.append({
            'Aircraft'       : r['aircraft_id'],
            'Flight ID'      : r.get('flight_id', '—'),
            'True Label'     : label_str,
            'Severity'       : f"{emoji} {r['severity']}",
            'Maint. Prob'    : f"{r['probability']*100:.1f}%",
            'Anomaly Score'  : f"{r.get('local_anomaly_score',0)*100:.1f}%",
            'Sensors Flagged': f"{r.get('local_sensors_flagged',0)} / 16",
            'Top Sensor'     : top_sensor,
            'Worst Phase'    : worst_phase,
            'Action'         : r['message'],
        })

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )

    # ── Priority actions ──────────────────────────────────────
    priority = [
        r for r in fleet
        if r['severity'] in ['CRITICAL', 'HIGH']
    ]
    if priority:
        st.divider()
        st.subheader("🚨 Priority Actions Required")
        for r in sorted(
            priority,
            key=lambda x: x['probability'], reverse=True
        ):
            c          = SEVERITY_COLORS[r['severity']]
            e          = SEVERITY_EMOJI[r['severity']]
            top_sensor = (
                r.get('explanation', {})
                 .get('top_channels', [{}])[0]
                 .get('channel', 'N/A')
            )
            st.markdown(
                f"""
                <div style="
                    border-left:5px solid {c};
                    padding:12px 18px; margin:6px 0;
                    background:{c}15; border-radius:5px;
                ">
                <strong style="font-size:15px;">
                    {e} {r['aircraft_id']}
                </strong>
                &nbsp;—&nbsp;
                <span style="color:{c};">
                    {r['severity']}
                </span>
                &nbsp;|&nbsp;
                Prob: <strong>
                    {r['probability']*100:.1f}%
                </strong>
                &nbsp;|&nbsp;
                Top: <strong>{top_sensor}</strong>
                &nbsp;|&nbsp;
                Flight ID: {r.get('flight_id','—')}
                <br>
                <span style="font-size:13px; opacity:0.85;">
                    📋 {r['message']}
                </span>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.success(
            "✅ No priority actions required."
        )

# ============================================================
# MAIN
# ============================================================

def main():
    view = render_sidebar()
    if "Mechanic" in view:
        render_mechanic_view()
    else:
        render_fleet_view()


if __name__ == "__main__":
    main()