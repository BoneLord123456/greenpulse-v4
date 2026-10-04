"""
GreenPulse - single-page Streamlit frontend.

Run:  streamlit run app.py

The ESP32 sensor data below is TEMPORARY SAMPLE DATA. Search this file for
"BACKEND INTEGRATION POINT" to find the two places to replace with FastAPI calls.
Open-Meteo data is REAL (fetched live from Open-Meteo for the sensor's coordinates).
"""
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="GreenPulse", page_icon="🌿", layout="wide")

# =============================================================================
# 0. STYLING  (one CSS block - edit ACCENT / CSS here to restyle the whole page)
# =============================================================================
ACCENT = "#2e9e6b"        # brand green, used for headings, metric cards, buttons, charts


def get_theme_mode() -> str:
    return "dark"


def apply_custom_css():
    bg = "#0b1220"
    bg_alt = "#111827"
    text = "#e5eef9"
    muted = "#bfd1ea"
    border = "rgba(148, 163, 184, 0.24)"
    card = "rgba(15, 23, 42, 0.78)"
    button_bg = "linear-gradient(135deg, #1f2937 0%, #334155 100%)"
    button_text = "#f8fafc"

    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --gp-accent: {ACCENT};
  --gp-bg: {bg};
  --gp-bg-alt: {bg_alt};
  --gp-text: {text};
  --gp-muted: {muted};
  --gp-card-bg: {card};
  --gp-border: {border};
  --gp-button-bg: {button_bg};
  --gp-button-text: {button_text};
  --gp-shadow: rgba(15, 23, 42, 0.24);
}}

html, body, [data-testid="stAppViewContainer"], .stApp, .main {{
  background: var(--gp-bg) !important;
  color: var(--gp-text) !important;
}}

.block-container, [data-testid="stMainBlockContainer"] {{
  max-width: 1400px;
  padding-top: 1.5rem;
  padding-bottom: 3rem;
  background: var(--gp-bg);
  color: var(--gp-text);
}}
footer {{ visibility: hidden; }}

h1, h2, h3, p, li,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"],
[data-testid="stCaptionContainer"] {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  color: var(--gp-text);
}}
.stApp h1 {{
  font-size: 2.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  padding-bottom: 0;
  color: var(--gp-accent);
}}
.stApp h2 {{
  font-size: 1.4rem;
  font-weight: 650;
  letter-spacing: -0.01em;
  border-left: 4px solid var(--gp-accent);
  padding: 0.1rem 0 0.1rem 0.7rem !important;
  margin-top: 2rem !important;
  color: var(--gp-text);
}}
.stApp h3 {{
  font-size: 0.95rem;
  font-weight: 600;
  opacity: 0.8;
  padding-bottom: 0.2rem;
  margin-top: 0.6rem;
  color: var(--gp-text);
}}
[data-testid="stCaptionContainer"] {{ opacity: 0.8; color: var(--gp-muted); }}

[data-testid="stMetric"] {{
  background: var(--gp-card-bg);
  border: 1px solid var(--gp-border);
  border-left: 3px solid var(--gp-accent);
  border-radius: 12px;
  padding: 0.8rem 0.9rem;
  box-shadow: 0 10px 24px var(--gp-shadow);
}}
[data-testid="stMetricLabel"] {{ opacity: 0.75; font-size: 0.8rem; color: var(--gp-muted); }}
[data-testid="stMetricValue"] {{ font-size: 1.55rem; font-weight: 600; color: var(--gp-text); }}
[data-testid="stMetricValue"] > div {{
  overflow: visible; text-overflow: clip; white-space: normal;
}}

[data-testid="stExpander"] details {{
  border: 1px solid var(--gp-border);
  border-radius: 12px;
  background: var(--gp-card-bg);
  box-shadow: 0 8px 22px var(--gp-shadow);
}}
[data-testid="stPlotlyChart"] {{
  border: 1px solid var(--gp-border);
  border-radius: 12px;
  padding: 4px;
  background: var(--gp-card-bg);
  box-shadow: 0 10px 22px var(--gp-shadow);
}}
[data-testid="stMap"] {{
  border: 1px solid var(--gp-border);
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 10px 22px var(--gp-shadow);
  min-height: 520px;
  height: 100% !important;
}}
[data-testid="stMap"] iframe {{
  height: 100% !important;
  min-height: 520px;
}}
[data-testid="stDataFrame"] {{
  border: 1px solid var(--gp-border);
  border-radius: 10px;
  overflow: hidden;
}}
[data-testid="stDeckGlJsonChart"] {{
  border: 1px solid var(--gp-border);
  border-radius: 12px;
  overflow: hidden;
}}
[data-testid="stAlert"] {{ border-radius: 10px; }}

.stButton > button {{
  border-radius: 999px;
  font-weight: 600;
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: var(--gp-button-bg);
  color: var(--gp-button-text);
  box-shadow: 0 8px 18px rgba(16, 24, 40, 0.18);
  padding: 0.45rem 0.9rem;
}}
.stButton > button:hover {{ background: var(--gp-accent); color: #ffffff; border-color: var(--gp-accent); }}

.theme-toggle-row {{
  display: flex;
  justify-content: flex-end;
  margin: 0 0 0.5rem 0;
}}

@media (max-width: 820px) {{
  [data-testid="stHorizontalBlock"] {{
    display: block !important;
    width: 100% !important;
  }}
  [data-testid="stHorizontalBlock"] > div {{
    width: 100% !important;
    max-width: 100% !important;
    flex: 0 0 100% !important;
  }}
  [data-testid="stMap"] {{
    min-height: 380px;
  }}
  [data-testid="stMap"] iframe {{
    min-height: 380px;
  }}
}}

@media (max-width: 640px) {{
  .stApp h1 {{ font-size: 1.9rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
}}
</style>
"""
    flat = "\n".join(line.strip() for line in css.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


# =============================================================================
# 1. SETTINGS & SESSION STATE
# =============================================================================
BACKEND_URL = "http://localhost:8000"    # localhost for development; in Streamlit Cloud use tunnel URL

if "using_sample_data" not in st.session_state:
    st.session_state.using_sample_data = False

SENSOR_COLUMNS = [
    "time", "temperature", "humidity_percent", "co2", "soil_moisture", "rainfall",
    "pressure", "pm2_5", "pm10", "aqi", "soil_score",
    "sound_activity", "sound_score", "pir_activity", "pir_score", "ems",
    "latitude", "longitude", "advisory",
]


# =============================================================================
# 2. SENSOR DATA  (ESP32 -> backend)         *** DEMO DATA ***
# =============================================================================
def _make_sample_telemetry() -> dict:
    """Generate sample telemetry data matching backend structure."""
    import random
    now = datetime.now()
    
    # Simulate variation in EMS score
    base_ems = 61
    ems_variation = random.uniform(-5, 5)
    
    return {
        "ems": {
            "score": max(0, min(100, base_ems + ems_variation)),
            "ground_stability": random.randint(35, 50),
            "human_pressure": random.randint(70, 85),
            "environmental_quality": random.randint(65, 78),
            "alert_level": random.choice(["NORMAL", "WATCH", "WARNING"]),
            "primary_driver": "ground_stability",
            "advisory": "Demo data - Backend connection unavailable. Recent rainfall is increasing ground stress. Monitor drainage conditions."
        },
        "local_telemetry": {
            "temperature_c": 22 + random.uniform(-2, 3),
            "humidity_percent": 75 + random.uniform(-10, 15),
            "soil_score": max(0, min(100, 50 + random.uniform(-20, 20))),
            "sound_score": random.uniform(60, 99),
            "pir_score": random.uniform(10, 85),
            "co2_ppm": 420 + random.uniform(-50, 100)
        },
        "environmental_context": {
            "rainfall_1h": max(0, 2.0 + random.uniform(-1, 3)),
            "rainfall_24h": max(0, 25.0 + random.uniform(-5, 10)),
            "rainfall_score": random.randint(60, 90),
            "aqi": random.randint(40, 70),
            "surface_pressure_hpa": 1008 + random.uniform(-2, 2)
        },
        "latitude": 25.6121,
        "longitude": 91.8977
    }


@st.cache_data(ttl=60)
def _make_sample_history() -> pd.DataFrame:
    """Generate 48 demo readings in nested telemetry structure, 30 minutes apart."""
    n = 48
    x = np.arange(n)
    rng = np.random.default_rng(1)
    times = pd.date_range(end=pd.Timestamp.now().floor("s"), periods=n, freq="30min")
    
    # Generate demo readings as nested telemetry, then flatten
    readings = []
    for i, t in enumerate(times):
        demo = _make_sample_telemetry()
        demo["time"] = t  # Override with specific time
        flat = _flatten_telemetry(demo)
        readings.append(flat)
    
    return pd.DataFrame(readings)


def get_sensor_data() -> dict:
    """Return the LATEST reading as a flattened dict.
    
    Attempts to fetch from backend. On failure, returns demo data and sets
    session state to show warning.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/latest", timeout=5)
        response.raise_for_status()
        telemetry = response.json()
        st.session_state.using_sample_data = False
        return _flatten_telemetry(telemetry)
    except requests.exceptions.RequestException:
        # Backend unavailable - use demo data
        st.session_state.using_sample_data = True
        demo = _make_sample_telemetry()
        return _flatten_telemetry(demo)


def get_sensor_history() -> pd.DataFrame:
    """Return historical readings as a DataFrame (one row per reading).
    
    Attempts to fetch from backend. On failure, returns demo history.
    """
    try:
        response = requests.get(f"{BACKEND_URL}/readings/all", timeout=5)
        response.raise_for_status()
        data = response.json()
        # Handle both list of readings or single reading
        if isinstance(data, dict):
            data = [data]
        # Flatten each reading
        flat_readings = [_flatten_telemetry(reading) for reading in data]
        st.session_state.using_sample_data = False
        return pd.DataFrame(flat_readings)
    except requests.exceptions.RequestException:
        # Backend unavailable - use demo history
        st.session_state.using_sample_data = True
        return _make_sample_history()


def _flatten_telemetry(telemetry: dict) -> dict:
    """Convert nested telemetry structure to flat dict for display.
    
    Input: {"ems": {...}, "local_telemetry": {...}, "environmental_context": {...}}
    Output: flat dict with renamed fields for display compatibility
    """
    flat = {"time": datetime.now()}
    
    # EMS section
    ems = telemetry.get("ems") or {}
    flat["ems"] = ems.get("score")
    flat["ems_alert_level"] = ems.get("alert_level")
    flat["ems_primary_driver"] = ems.get("primary_driver")
    flat["ems_ground_stability"] = ems.get("ground_stability")
    flat["ems_human_pressure"] = ems.get("human_pressure")
    flat["ems_environmental_quality"] = ems.get("environmental_quality")
    flat["advisory"] = ems.get("advisory")
    
    # Local telemetry section
    local = telemetry.get("local_telemetry") or {}
    flat["temperature"] = local.get("temperature_c")
    flat["humidity_percent"] = local.get("humidity_percent")
    flat["soil_score"] = local.get("soil_score")
    flat["sound_score"] = local.get("sound_score")
    flat["pir_score"] = local.get("pir_score")
    flat["co2"] = local.get("co2_ppm")
    
    # Environmental context section
    env = telemetry.get("environmental_context") or {}
    flat["rainfall"] = env.get("rainfall_1h")
    flat["rainfall_24h"] = env.get("rainfall_24h")
    flat["rainfall_score"] = env.get("rainfall_score")
    flat["aqi"] = env.get("aqi")
    flat["pressure"] = env.get("surface_pressure_hpa")
    
    # Preserve optional fields (latitude, longitude, etc.)
    for key in ["latitude", "longitude", "pm2_5", "pm10"]:
        if key in telemetry:
            flat[key] = telemetry[key]
    
    return flat


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    """Make any backend DataFrame safe to plot: correct columns, numbers, time order."""
    df = df.copy()
    for col in SENSOR_COLUMNS:
        if col not in df:
            df[col] = np.nan                      # e.g. PIR not available yet
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    for col in SENSOR_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[SENSOR_COLUMNS].reset_index(drop=True)


# =============================================================================
# 3. OPEN-METEO DATA  (external, modelled - NOT from the ESP32)
# =============================================================================
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


@st.cache_data(ttl=600, show_spinner=False)   # cache 10 min: be kind to the free API
def _fetch_json(url: str, params: dict):
    """GET a URL and return (json, retrieval_time). Raises on any failure
    (failures are not cached)."""
    response = requests.get(url, params=params, timeout=(3.05, 10))
    if response.status_code >= 400:
        try:
            reason = response.json().get("reason", "")
        except ValueError:
            reason = ""
        raise RuntimeError(f"HTTP {response.status_code} {reason}".strip())
    return response.json(), datetime.now()


def get_open_meteo_data(latitude, longitude) -> dict:
    """
    Fetch rainfall, pressure, PM2.5, PM10 and AQI for the given coordinates.

    Never raises. Returns:
      {"current":  {"rainfall", "pressure", "pm2_5", "pm10", "aqi"},   (values may be None)
       "history":  DataFrame [time, rainfall, pressure, pm2_5, pm10, aqi]  (past 2 days),
       "retrieved_at": datetime or None,
       "errors":   [str, ...]}
    """
    result = {"current": {}, "history": pd.DataFrame(), "retrieved_at": None, "errors": []}

    # --- validate coordinates -------------------------------------------------
    try:
        lat, lon = round(float(latitude), 3), round(float(longitude), 3)
    except (TypeError, ValueError):
        lat = lon = float("nan")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):      # NaN fails this too
        result["errors"].append("Invalid or missing sensor coordinates - cannot query Open-Meteo.")
        return result

    # --- (name, url, {open-meteo variable: our name}) ---------------------------
    sources = [
        ("Weather API", WEATHER_URL, {"rain": "rainfall", "surface_pressure": "pressure"}),
        ("Air Quality API", AIR_QUALITY_URL,
         {"pm2_5": "pm2_5", "pm10": "pm10", "us_aqi": "aqi"}),
    ]
    frames, now_times, fetch_times = [], [], []

    for name, url, variables in sources:
        params = {
            "latitude": lat, "longitude": lon,
            "current": ",".join(variables), "hourly": ",".join(variables),
            "past_days": 2, "forecast_days": 1, "timezone": "auto",
        }
        try:
            data, fetched_at = _fetch_json(url, params)
            fetch_times.append(fetched_at)

            current = data.get("current") or {}
            for api_name, our_name in variables.items():
                result["current"][our_name] = current.get(api_name)
            if current.get("time"):
                now_times.append(pd.to_datetime(current["time"]))

            hourly = data.get("hourly") or {}
            frame = pd.DataFrame(hourly).rename(columns=variables)
            frame["time"] = pd.to_datetime(frame["time"])
            frames.append(frame)
        except requests.exceptions.Timeout:
            result["errors"].append(f"{name}: the request timed out.")
        except requests.exceptions.RequestException as exc:
            result["errors"].append(f"{name}: network error ({type(exc).__name__}).")
        except Exception as exc:                   # HTTP error text, bad JSON, missing keys...
            result["errors"].append(f"{name}: {exc}")

    if frames:
        history = reduce(lambda a, b: a.merge(b, on="time", how="outer"), frames).sort_values("time")
        if now_times:                              # hourly arrays also contain future hours
            history = history[history["time"] <= max(now_times)]
        result["history"] = history.reset_index(drop=True)
    if fetch_times:
        result["retrieved_at"] = max(fetch_times)
    return result


# =============================================================================
# 4. SMALL HELPERS
# =============================================================================
def fmt(value, decimals=1, unit=""):
    """Format a number for display; 'N/A' if missing."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}{(' ' + unit) if unit else ''}"


def delta_vs_previous(history: pd.DataFrame, column: str, decimals=1):
    """Change between the last two readings, for st.metric(delta=...)."""
    if len(history) < 2:
        return None
    a, b = history[column].iloc[-1], history[column].iloc[-2]
    if pd.isna(a) or pd.isna(b):
        return None
    return f"{a - b:+.{decimals}f}"


def show_plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                              # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True)


def line_chart(df: pd.DataFrame, column: str, title: str, y_label: str, height: int = 300):
    data = df[["time", column]].dropna()
    if data.empty:
        st.info(f"No data available for: {title}")
        return
    fig = px.line(
        data,
        x="time",
        y=column,
        title=title,
        labels={"time": "", column: y_label},
        color_discrete_sequence=[ACCENT],
        template="plotly_dark",
    )
    fig.update_traces(line_width=2.2)
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        title_font_size=15,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    show_plot(fig)


def chart_grid(df: pd.DataFrame, charts: list):
    """Draw (column, title, y_label) charts two per row."""
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for col, (column, title, y_label) in zip(cols, charts[i:i + 2]):
            with col:
                line_chart(df, column, title, y_label)


# =============================================================================
# 5. PAGE SECTIONS
# =============================================================================
def show_header(latest: dict):
    """Display header with EMS dashboard in box format."""
    # Header with LIVE status
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("<h1 style='text-align: center; color: #2e9e6b; font-size: 3rem; margin: 0;'>🌿 GREENPULSE</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #bfd1ea; font-size: 1.1rem; margin: 0;'>Environmental Monitoring & Stability System</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h2 style='text-align: center; color: #ff6b6b; font-size: 2.5rem; margin: 0;'>🔴 LIVE</h2>", unsafe_allow_html=True)
    
    # Divider
    st.divider()
    
    if st.session_state.using_sample_data:
        st.warning("⚠️ Backend is not connected. Showing DEMO DATA. Start your backend at " + BACKEND_URL)
    
    t = pd.to_datetime(latest.get("time"), errors="coerce")
    if pd.notna(t):
        st.caption(f"Last update: {t:%Y-%m-%d %H:%M:%S}")
    
    if st.button("🔄 Refresh"):
        st.rerun()


def show_sensor_data(latest: dict, history: pd.DataFrame):
    st.header("Sensor Data")
    st.caption("Source: backend-processed ESP32 sensor data.")

    st.metric("EMS", fmt(latest.get("ems"), 1), delta_vs_previous(history, "ems"))

    st.subheader("Environmental sensors")
    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature", fmt(latest.get("temperature"), 1, "°C"),
              delta_vs_previous(history, "temperature"))
    c2.metric("Humidity", fmt(latest.get("humidity_percent"), 0, "%"),
              delta_vs_previous(history, "humidity_percent", 0))
    c3.metric("CO2", fmt(latest.get("co2"), 0, "ppm"), delta_vs_previous(history, "co2", 0))

    st.subheader("Sensor scores")
    c1, c2, c3 = st.columns(3)
    c1.metric("Soil Score", fmt(latest.get("soil_score"), 1), delta_vs_previous(history, "soil_score"))
    c2.metric("Sound Score", fmt(latest.get("sound_score"), 1), delta_vs_previous(history, "sound_score"))
    c3.metric("PIR Score", fmt(latest.get("pir_score"), 1), delta_vs_previous(history, "pir_score"))

    st.subheader("Weather and air quality")
    row1 = st.columns(2)

    row1[0].metric("Rainfall", fmt(latest.get("rainfall"), 1, "mm"),
                   delta_vs_previous(history, "rainfall"))
    row1[1].metric("AQI", fmt(latest.get("aqi"), 0), delta_vs_previous(history, "aqi", 0))

    st.caption("Air quality and environmental indicators are shown here for the latest reading.")


def show_ems_dashboard(latest: dict):
    """Display EMS dashboard in structured box format."""
    
    ems_score = latest.get("ems")
    alert_level = latest.get("ems_alert_level", "UNKNOWN")
    primary_driver = latest.get("ems_primary_driver", "unknown")
    ground_stability = latest.get("ems_ground_stability")
    human_pressure = latest.get("ems_human_pressure")
    env_quality = latest.get("ems_environmental_quality")
    advisory = latest.get("advisory", "No advisory available.")
    
    # Alert level color coding
    alert_colors = {
        "CRITICAL": "🔴",
        "WARNING": "🟠",
        "WATCH": "🟡",
        "NORMAL": "🟢",
    }
    alert_emoji = alert_colors.get(alert_level, "⚪")
    
    # Main EMS Score Display - centered
    st.markdown("<h2 style='text-align: center; color: #bfd1ea; font-size: 1.4rem; letter-spacing: 0.05em;'>ENVIRONMENTAL MONITORING SCORE</h2>", unsafe_allow_html=True)
    score_col = st.columns([1, 2, 1])[1]
    with score_col:
        st.markdown(f"<h1 style='text-align: center; font-size: 3rem; color: #2e9e6b;'>{fmt(ems_score, 1)}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 1.2rem;'>{alert_emoji} {alert_level}</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Component Breakdown
    st.markdown("<h2 style='text-align: left; color: #bfd1ea; font-size: 1.4rem; letter-spacing: 0.05em;'>COMPONENT BREAKDOWN</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ground Stability", fmt(ground_stability, 1))
    with col2:
        st.metric("Human Pressure", fmt(human_pressure, 1))
    with col3:
        st.metric("Environmental Quality", fmt(env_quality, 1))
    
    st.divider()
    
    # Primary Driver
    st.markdown("<h2 style='text-align: left; color: #bfd1ea; font-size: 1.4rem; letter-spacing: 0.05em;'>PRIMARY DRIVER</h2>", unsafe_allow_html=True)
    st.markdown(f"**{primary_driver.replace('_', ' ').title()}**")
    
    st.divider()
    
    # Advisory
    st.markdown("<h2 style='text-align: left; color: #bfd1ea; font-size: 1.4rem; letter-spacing: 0.05em;'>ADVISORY</h2>", unsafe_allow_html=True)
    st.info(advisory)
    
    st.divider()


def show_local_telemetry(latest: dict):
    """Display local sensor telemetry as a table."""
    st.markdown("<h2 style='text-align: left; color: #bfd1ea; font-size: 1.4rem; letter-spacing: 0.05em;'>LOCAL TELEMETRY</h2>", unsafe_allow_html=True)
    
    telemetry_data = {
        "Temp": fmt(latest.get("temperature"), 1, "°C"),
        "Humidity": fmt(latest.get("humidity_percent"), 0, "%"),
        "Soil": fmt(latest.get("soil_score"), 0),
        "Sound": fmt(latest.get("sound_score"), 0),
        "PIR": fmt(latest.get("pir_score"), 0),
        "CO2": fmt(latest.get("co2"), 0, "ppm"),
    }
    
    # Display as metrics in a row
    cols = st.columns(6)
    for i, (label, value) in enumerate(telemetry_data.items()):
        with cols[i]:
            st.metric(label, value)


def show_external_data(meteo: dict, latitude, longitude):
    st.header("Open-Meteo Environmental Data")
    st.caption("Source: Open-Meteo (modelled data for the sensor's coordinates) - "
               "NOT measured by the ESP32.")
    if meteo["retrieved_at"]:
        st.caption(f"Retrieved: {meteo['retrieved_at']:%Y-%m-%d %H:%M:%S} "
                   f"for latitude {fmt(latitude, 5)}, longitude {fmt(longitude, 5)}")
    for message in meteo["errors"]:
        st.warning(f"Open-Meteo unavailable - {message}")

    cur = meteo["current"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rainfall (preceding hour)", fmt(cur.get("rainfall"), 1, "mm"))
    c2.metric("Surface pressure", fmt(cur.get("pressure"), 0, "hPa"))
    c3.metric("PM2.5", fmt(cur.get("pm2_5"), 1, "µg/m³"))
    c4.metric("PM10", fmt(cur.get("pm10"), 1, "µg/m³"))
    c5.metric("AQI", fmt(cur.get("aqi"), 0))


def show_sensor_graphs(history: pd.DataFrame):
    st.header("Sensor Graphs")
    with st.expander("Key environmental trends", expanded=True):
        left_col, right_col = st.columns([1.15, 1.4])

        with left_col:
            line_chart(history, "ems", "EMS vs Time", "EMS", height=700)

        with right_col:
            row1_left, row1_right = st.columns(2)
            with row1_left:
                line_chart(history, "temperature", "Temperature vs Time", "°C", height=220)
            with row1_right:
                line_chart(history, "humidity_percent", "Humidity vs Time", "%", height=220)

            row2_left, row2_right = st.columns(2)
            with row2_left:
                line_chart(history, "sound_score", "Sound Score vs Time", "Score", height=220)
            with row2_right:
                line_chart(history, "soil_score", "Soil Score vs Time", "Score", height=220)

            row3 = st.columns([1])
            with row3[0]:
                line_chart(history, "pir_score", "PIR Score vs Time", "Score", height=220)


def show_external_graphs(meteo: dict):
    st.header("External Data Graphs")
    st.caption("Open-Meteo hourly data for the past 2 days (modelled, not ESP32 measurements).")
    history = meteo["history"]
    if history.empty:
        st.info("No Open-Meteo historical data available.")
        return
    with st.expander("Open-Meteo graphs", expanded=True):
        chart_grid(history, [
            ("rainfall", "Rainfall vs Time", "mm"),
            ("pressure", "Surface Pressure vs Time", "hPa"),
            ("pm2_5", "PM2.5 vs Time", "µg/m³"),
            ("pm10", "PM10 vs Time", "µg/m³"),
            ("aqi", "AQI vs Time", "AQI"),
        ])


def show_location_and_context(latitude, longitude, latest: dict, history: pd.DataFrame):
    """Display location map and environmental context side by side."""
    
    col1, col2 = st.columns([3, 2])
    
    # LEFT: Location/Map
    with col1:
        st.markdown("<h2 style='text-align: left; color: #bfd1ea; font-size: 1.4rem; letter-spacing: 0.05em;'>LOCATION / MAP</h2>", unsafe_allow_html=True)
        try:
            point = pd.DataFrame({"latitude": [float(latitude)], "longitude": [float(longitude)]})
            if not (point.notna().all().all() and -90 <= point.latitude[0] <= 90
                    and -180 <= point.longitude[0] <= 180):
                raise ValueError
            st.map(point, zoom=14)
        except (TypeError, ValueError):
            st.info("No valid GPS position available.")
    
    # RIGHT: Environmental Context
    with col2:
        st.markdown("<h2 style='text-align: left; color: #bfd1ea; font-size: 1.4rem; letter-spacing: 0.05em;'>ENVIRONMENTAL CONTEXT</h2>", unsafe_allow_html=True)
        
        aqi = latest.get("aqi")
        # Determine AQI status
        if aqi is None or pd.isna(aqi):
            aqi_status = "UNKNOWN"
        elif aqi <= 50:
            aqi_status = "GOOD"
        elif aqi <= 100:
            aqi_status = "SATISFACTORY"
        elif aqi <= 200:
            aqi_status = "MODERATELY POLLUTED"
        elif aqi <= 300:
            aqi_status = "HEAVILY POLLUTED"
        else:
            aqi_status = "SEVERELY POLLUTED"
        
        st.metric("AQI", fmt(aqi, 0))
        st.markdown(f"**{aqi_status}**")
        
        st.divider()
        
        # Rainfall data
        st.markdown("**RAINFALL**")
        rainfall_1h = latest.get("rainfall")
        rainfall_24h = latest.get("rainfall_24h")
        rainfall_72h = latest.get("rainfall_score")  # Using rainfall_score as proxy for 72h
        
        st.write(f"**1h:** {fmt(rainfall_1h, 1, 'mm')}")
        st.write(f"**24h:** {fmt(rainfall_24h, 1, 'mm')}")
        st.write(f"**72h:** {fmt(rainfall_72h, 1, 'mm')}")
        
        st.divider()
        
        # Surface Pressure
        st.markdown("**SURFACE PRESSURE**")
        pressure = latest.get("pressure")
        st.write(f"{fmt(pressure, 1, 'hPa')}")



# =============================================================================
# 6. MAIN
# =============================================================================
def main():
    apply_custom_css()

    try:
        latest = get_sensor_data()
        history = prepare_history(get_sensor_history())
    except Exception as exc:                       # backend down, bad JSON, ...
        st.title("GreenPulse")
        st.error(f"Could not load sensor data: {exc}")
        st.stop()

    # Header with LIVE indicator
    show_header(latest)
    
    # EMS Dashboard - Main Telemetry Display
    show_ems_dashboard(latest)
    
    # Location Map & Environmental Context
    lat, lon = latest.get("latitude"), latest.get("longitude")
    show_location_and_context(lat, lon, latest, history)
    
    st.divider()
    
    # Local Telemetry
    show_local_telemetry(latest)
    
    st.divider()
    
    # Sensor Graphs (collapsible)
    show_sensor_graphs(history)


main()
