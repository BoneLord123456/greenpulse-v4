"""
GreenPulse - Streamlit frontend (Overview + three exploration pages).

Run:  streamlit run app.py

Pages (sidebar navigation):
  Overview              score, three dimensions, alert, advisory, map
  Ground Stability      soil + rainfall   (70% / 30%)
  Human Pressure        PIR + sound + CO2 (50% / 30% / 20%)
  Environmental Quality AQI + temperature + humidity (50% / 25% / 25%)

Data comes from the FastAPI backend. Set its public tunnel URL as BACKEND_URL
(Streamlit Cloud: App settings -> Secrets; locally: environment variable or
.streamlit/secrets.toml). If the backend is unreachable the app falls back to
DEMO DATA and says so. Search for "BACKEND INTEGRATION POINT" to find where
the backend is called.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="GreenPulse", page_icon="🌿", layout="wide")

# =============================================================================
# 0. CONSTANTS
# =============================================================================
ACCENT = "#2e9e6b"
# Node position, used when the backend payload has no latitude/longitude.
NODE_LATITUDE = 25.5788
NODE_LONGITUDE = 91.8933


def get_backend_url() -> str:
    """Backend base URL.

    Streamlit Cloud runs this script on Streamlit's servers, not in the viewer's
    browser, so "localhost" there is NOT your laptop. Put your tunnel's public
    URL (e.g. https://xxxx.trycloudflare.com) in the app's Secrets:
        BACKEND_URL = "https://xxxx.trycloudflare.com"
    Order: Streamlit secret -> environment variable -> http://localhost:8000.
    """
    try:
        url = st.secrets["BACKEND_URL"]
    except Exception:                      # no secrets file / key not set
        url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    return str(url).strip().rstrip("/")


BACKEND_URL = get_backend_url()
# ngrok's free tier serves a browser-warning page unless this header is sent; harmless elsewhere.
BACKEND_HEADERS = {"ngrok-skip-browser-warning": "true", "Accept": "application/json"}
BACKEND_TIMEOUT = 10                       # tunnels add latency

PAGES = ["Overview", "Ground Stability", "Human Pressure", "Environmental Quality"]

# Weights used by the backend EMS sub-scores (shown on each page).
GROUND_WEIGHTS = {"Soil": 0.70, "Rainfall": 0.30}
HUMAN_WEIGHTS = {"PIR": 0.50, "Sound": 0.30, "CO₂": 0.20}
ENV_WEIGHTS = {"AQI": 0.50, "Temperature": 0.25, "Humidity": 0.25}

ALERT_COLORS = {
    "NORMAL": "#2e9e6b",
    "WATCH": "#e3b341",
    "WARNING": "#f0883e",
    "CRITICAL": "#f85149",
}

SENSOR_COLUMNS = [
    "time", "temperature", "humidity_percent", "co2", "soil_moisture", "rainfall",
    "rainfall_24h", "rainfall_72h", "rainfall_score", "pressure", "pm2_5", "pm10", "aqi",
    "soil_score", "sound_score", "pir_score",
    "co2_score",
    "ems", "ems_ground_stability", "ems_human_pressure", "ems_environmental_quality",
    "latitude", "longitude",
]

if "using_sample_data" not in st.session_state:
    st.session_state.using_sample_data = False
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]


# =============================================================================
# 1. STYLING
# =============================================================================
def apply_custom_css():
    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{
  --gp-accent: {ACCENT};
  --gp-bg: #0b1220;
  --gp-text: #e5eef9;
  --gp-muted: #9fb3cf;
  --gp-card: rgba(15, 23, 42, 0.78);
  --gp-border: rgba(148, 163, 184, 0.24);
}}
html, body, [data-testid="stAppViewContainer"], .stApp, .main {{
  background: var(--gp-bg) !important; color: var(--gp-text) !important;
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container, [data-testid="stMainBlockContainer"] {{
  max-width: 1300px; padding-top: 1.5rem; padding-bottom: 3rem;
}}
footer {{ visibility: hidden; }}
h1, h2, h3, p, li, label, [data-testid="stCaptionContainer"] {{
  font-family: 'Inter', system-ui, sans-serif; color: var(--gp-text);
}}
.stApp h2 {{
  font-size: 1.2rem; font-weight: 650; letter-spacing: -0.01em;
  border-left: 4px solid var(--gp-accent); padding: 0.1rem 0 0.1rem 0.7rem !important;
  margin-top: 1.8rem !important;
}}
[data-testid="stCaptionContainer"] {{ color: var(--gp-muted); }}

/* sidebar */
[data-testid="stSidebar"] {{ background: #0f172a; border-right: 1px solid var(--gp-border); }}
.gp-brand {{ font-size: 1.35rem; font-weight: 700; color: var(--gp-accent); margin-bottom: 0.2rem; }}
.gp-status {{ font-size: 0.85rem; color: var(--gp-muted); padding-top: 0.6rem;
  border-top: 1px solid var(--gp-border); margin-top: 1rem; }}

/* header */
.gp-header {{ display: flex; justify-content: space-between; align-items: flex-end;
  border-bottom: 1px solid var(--gp-border); padding-bottom: 0.8rem; margin-bottom: 1rem; }}
.gp-title {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; color: var(--gp-accent); line-height: 1.1; }}
.gp-sub {{ color: var(--gp-muted); font-size: 0.95rem; }}
.gp-live {{ font-weight: 600; font-size: 0.9rem; display: flex; align-items: center; gap: 0.4rem; }}
.gp-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}

/* hero */
.gp-hero {{ text-align: center; padding: 2.2rem 1rem; border: 1px solid var(--gp-border);
  border-radius: 16px; background: var(--gp-card); }}
.gp-hero-label {{ color: var(--gp-muted); font-size: 0.95rem; letter-spacing: 0.04em; }}
.gp-hero-score {{ font-size: 5.5rem; font-weight: 700; line-height: 1.05; letter-spacing: -0.03em; }}
.gp-hero-score span {{ font-size: 1.6rem; font-weight: 500; color: var(--gp-muted); margin-left: 0.3rem; }}
.gp-hero-level {{ font-size: 1.3rem; font-weight: 700; letter-spacing: 0.12em; }}

/* score cards */
.gp-card {{ border: 1px solid var(--gp-border); border-radius: 12px; background: var(--gp-card);
  padding: 1rem 1.1rem; }}
.gp-card-label {{ color: var(--gp-muted); font-size: 0.85rem; }}
.gp-card-score {{ font-size: 2.3rem; font-weight: 700; line-height: 1.15; }}
.gp-card-score span {{ font-size: 0.95rem; font-weight: 500; color: var(--gp-muted); margin-left: 0.25rem; }}
.gp-bar {{ height: 6px; border-radius: 3px; background: rgba(148,163,184,0.18); margin-top: 0.5rem; overflow: hidden; }}
.gp-bar > div {{ height: 100%; border-radius: 3px; }}
.gp-card-note {{ color: var(--gp-muted); font-size: 0.78rem; margin-top: 0.4rem; }}

/* alert / advisory */
.gp-callout {{ border: 1px solid var(--gp-border); border-left: 4px solid var(--gp-accent);
  border-radius: 10px; background: var(--gp-card); padding: 0.9rem 1.1rem; }}
.gp-callout-title {{ font-weight: 650; font-size: 0.9rem; color: var(--gp-muted); margin-bottom: 0.2rem; }}

/* page score */
.gp-page-score {{ font-size: 4rem; font-weight: 700; line-height: 1; letter-spacing: -0.03em; }}
.gp-page-score span {{ font-size: 1.3rem; font-weight: 500; color: var(--gp-muted); margin-left: 0.3rem; }}

[data-testid="stMetric"] {{ background: var(--gp-card); border: 1px solid var(--gp-border);
  border-left: 3px solid var(--gp-accent); border-radius: 12px; padding: 0.8rem 0.9rem; }}
[data-testid="stMetricLabel"] {{ color: var(--gp-muted); font-size: 0.8rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.5rem; font-weight: 600; }}
[data-testid="stPlotlyChart"] {{ border: 1px solid var(--gp-border); border-radius: 12px; padding: 4px;
  background: var(--gp-card); }}
[data-testid="stMap"] {{ border: 1px solid var(--gp-border); border-radius: 12px; overflow: hidden; }}
.stButton > button {{ border-radius: 999px; font-weight: 600; border: 1px solid var(--gp-border);
  background: #1f2937; color: #f8fafc; }}
.stButton > button:hover {{ background: var(--gp-accent); border-color: var(--gp-accent); color: #fff; }}

@media (max-width: 820px) {{
  [data-testid="stHorizontalBlock"] {{ display: block !important; }}
  [data-testid="stHorizontalBlock"] > div {{ width: 100% !important; max-width: 100% !important; flex: 0 0 100% !important; }}
  .gp-hero-score {{ font-size: 4rem; }}
  .gp-header {{ flex-direction: column; align-items: flex-start; gap: 0.4rem; }}
}}
</style>
"""
    st.markdown("\n".join(line.strip() for line in css.splitlines() if line.strip()),
                unsafe_allow_html=True)


# =============================================================================
# 2. SMALL HELPERS
# =============================================================================
def fmt(value, decimals=1, unit=""):
    """Format a number for display; 'N/A' if missing."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}{(' ' + unit) if unit else ''}"


def score_color(score) -> str:
    if score is None or pd.isna(score):
        return "#64748b"
    if score >= 75:
        return ACCENT
    if score >= 50:
        return "#e3b341"
    return "#f85149"


def aqi_category(aqi) -> str:
    if aqi is None or pd.isna(aqi):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderately polluted"
    if aqi <= 300:
        return "Heavily polluted"
    return "Severely polluted"


def pretty(name) -> str:
    return str(name or "unknown").replace("_", " ").title()


def goto(page: str):
    st.session_state.page = page


def _clip(x) -> float:
    return float(max(0.0, min(100.0, x)))


# Fallback scoring - used ONLY when the backend does not send these sub-scores.
def derive_co2_score(co2):
    return None if co2 is None or pd.isna(co2) else _clip(100 - (co2 - 600) / 14)       # 600ppm=100, 2000ppm=0


def sub_score(latest: dict, key: str, derived):
    """Return (value, is_derived): backend value if present, else the fallback."""
    v = latest.get(key)
    if v is not None and not pd.isna(v):
        return float(v), False
    return derived, derived is not None


# =============================================================================
# 3. DATA  (backend -> flat dicts / DataFrames)
# =============================================================================
def _make_sample_telemetry() -> dict:
    """DEMO telemetry matching the backend structure (used when the backend is down)."""
    import random
    soil = random.uniform(55, 95)
    rain_score = random.uniform(55, 90)
    pir = random.uniform(40, 90)
    sound = random.uniform(50, 95)
    co2 = 420 + random.uniform(-20, 250)
    aqi = random.randint(30, 110)
    temp = 22 + random.uniform(-4, 8)
    hum = 65 + random.uniform(-20, 20)

    ground = 0.7 * soil + 0.3 * rain_score
    human = 0.5 * pir + 0.3 * sound + 0.2 * derive_co2_score(co2)
    env = random.uniform(60, 90)
    comps = {"ground_stability": ground, "human_pressure": human, "environmental_quality": env}

    return {
        "ems": {
            "score": sum(comps.values()) / 3,
            "ground_stability": ground,
            "human_pressure": human,
            "environmental_quality": env,
            "alert_level": random.choice(["NORMAL", "WATCH", "WATCH", "WARNING"]),
            "primary_driver": min(comps, key=comps.get),
            "advisory": "Demo data - backend unavailable. Inspect drainage and avoid disturbing saturated ground.",
        },
        "local_telemetry": {
            "temperature_c": temp,
            "humidity_percent": hum,
            "soil_score": soil,
            "soil_moisture_percent": 30 + soil / 4 + random.uniform(-2, 2),
            "sound_score": sound,
            "pir_score": pir,
            "co2_ppm": co2,
        },
        "environmental_context": {
            "rainfall_1h": max(0, random.uniform(-1, 3)),
            "rainfall_24h": max(0, 4.2 + random.uniform(-3, 6)),
            "rainfall_72h": max(0, 12.7 + random.uniform(-5, 10)),
            "rainfall_score": rain_score,
            "aqi": aqi,
            "pm2_5": aqi * 0.45 + random.uniform(-3, 3),
            "pm10": aqi * 0.8 + random.uniform(-4, 4),
            "surface_pressure_hpa": 1008 + random.uniform(-2, 2),
        },
        "latitude": 25.5788,
        "longitude": 91.8933,
    }


@st.cache_data(ttl=60)
def _make_sample_history() -> pd.DataFrame:
    """48 demo readings, 30 minutes apart."""
    times = pd.date_range(end=pd.Timestamp.now().floor("s"), periods=48, freq="30min")
    rows = []
    for t in times:
        demo = _make_sample_telemetry()
        demo["time"] = t
        rows.append(_flatten_telemetry(demo))
    return pd.DataFrame(rows)


def _flatten_telemetry(telemetry: dict, default_now: bool = True) -> dict:
    """Nested backend telemetry -> flat dict used by the pages.

    The backend JSON currently carries no timestamp. For the latest reading we
    use the fetch time; for history rows (default_now=False) a missing
    timestamp stays empty and prepare_history() falls back to reading order.
    """
    stamp = next((telemetry[k] for k in ("time", "timestamp", "created_at", "recorded_at")
                  if telemetry.get(k)), None)
    flat = {"time": stamp if stamp is not None else (datetime.now() if default_now else None)}

    ems = telemetry.get("ems") or {}
    flat["ems"] = ems.get("score")
    flat["ems_alert_level"] = ems.get("alert_level")
    flat["ems_primary_driver"] = ems.get("primary_driver")
    flat["ems_ground_stability"] = ems.get("ground_stability")
    flat["ems_human_pressure"] = ems.get("human_pressure")
    flat["ems_environmental_quality"] = ems.get("environmental_quality")
    flat["advisory"] = ems.get("advisory")

    local = telemetry.get("local_telemetry") or {}
    flat["temperature"] = local.get("temperature_c")
    flat["humidity_percent"] = local.get("humidity_percent")
    flat["soil_score"] = local.get("soil_score")
    flat["soil_moisture"] = local.get("soil_moisture_percent", local.get("soil_moisture"))
    flat["sound_score"] = local.get("sound_score")
    flat["pir_score"] = local.get("pir_score")
    flat["co2"] = local.get("co2_ppm")
    flat["co2_score"] = local.get("co2_score")

    env = telemetry.get("environmental_context") or {}
    flat["rainfall"] = env.get("rainfall_1h")
    flat["rainfall_24h"] = env.get("rainfall_24h")
    flat["rainfall_72h"] = env.get("rainfall_72h")
    flat["rainfall_score"] = env.get("rainfall_score")
    flat["aqi"] = env.get("aqi")
    flat["pm2_5"] = env.get("pm2_5", telemetry.get("pm2_5"))
    flat["pm10"] = env.get("pm10", telemetry.get("pm10"))
    flat["pressure"] = env.get("surface_pressure_hpa")

    flat["latitude"] = telemetry.get("latitude", NODE_LATITUDE)
    flat["longitude"] = telemetry.get("longitude", NODE_LONGITUDE)
    return flat


def get_sensor_data() -> dict:
    """LATEST reading (flat dict). BACKEND INTEGRATION POINT 1."""
    try:
        response = requests.get(f"{BACKEND_URL}/latest", headers=BACKEND_HEADERS, timeout=BACKEND_TIMEOUT)
        response.raise_for_status()
        st.session_state.using_sample_data = False
        return _flatten_telemetry(response.json())
    except (requests.exceptions.RequestException, ValueError):
        st.session_state.using_sample_data = True
        return _flatten_telemetry(_make_sample_telemetry())


def get_sensor_history() -> pd.DataFrame:
    """All readings (DataFrame). BACKEND INTEGRATION POINT 2."""
    try:
        response = requests.get(f"{BACKEND_URL}/readings/all", headers=BACKEND_HEADERS, timeout=BACKEND_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame([_flatten_telemetry(r, default_now=False) for r in data])
    except (requests.exceptions.RequestException, ValueError):
        st.session_state.using_sample_data = True
        return _make_sample_history()


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    """Make any backend DataFrame safe to plot: all columns, numeric, time-ordered."""
    df = df.copy()
    for col in SENSOR_COLUMNS:
        if col not in df:
            df[col] = np.nan
    if df["time"].isna().all():
        df["time"] = np.arange(len(df))           # no timestamps from backend: use reading order
    else:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
    for col in SENSOR_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[SENSOR_COLUMNS].reset_index(drop=True)


# =============================================================================
# 4. CHARTS
# =============================================================================
def _style(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10), title_font_size=15,
        hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"), legend_title_text="",
    )
    return fig


def show_plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                              # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True)


def line_chart(df, column, title, y_label, height=280):
    data = df[["time", column]].dropna()
    if data.empty:
        st.info(f"No data yet for: {title}")
        return
    fig = px.line(data, x="time", y=column, title=title,
                  labels={"time": "", column: y_label},
                  color_discrete_sequence=[ACCENT], template="plotly_dark")
    fig.update_traces(line_width=2.2)
    show_plot(_style(fig, height))


def multi_line_chart(df, columns: dict, title, y_label, height=280):
    """columns = {column: legend name}"""
    data = df[["time"] + list(columns)].dropna(subset=list(columns), how="all")
    if data.empty:
        st.info(f"No data yet for: {title}")
        return
    long = data.melt("time", var_name="series", value_name="value").dropna()
    long["series"] = long["series"].map(columns)
    fig = px.line(long, x="time", y="value", color="series", title=title,
                  labels={"time": "", "value": y_label}, template="plotly_dark",
                  color_discrete_sequence=[ACCENT, "#58a6ff", "#e3b341"])
    fig.update_traces(line_width=2.2)
    show_plot(_style(fig, height))


def scatter_chart(df, x, y, title, x_label, y_label, height=280):
    data = df[["time", x, y]].dropna()
    if len(data) < 3:
        st.info(f"Not enough data yet for: {title}")
        return
    fig = px.scatter(data, x=x, y=y, color="time", title=title,
                     labels={x: x_label, y: y_label, "time": "Time"},
                     template="plotly_dark", color_continuous_scale="Viridis")
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(marker=dict(size=7, opacity=0.85))
    show_plot(_style(fig, height).update_layout(hovermode="closest"))
    r = data[x].corr(data[y])
    if pd.notna(r):
        st.caption(f"Correlation: {r:+.2f}")


def rainfall_bars(latest, height=280):
    data = pd.DataFrame({
        "Window": ["1 h", "24 h", "72 h"],
        "mm": [latest.get("rainfall"), latest.get("rainfall_24h"), latest.get("rainfall_72h")],
    }).dropna()
    if data.empty:
        st.info("No rainfall data yet.")
        return
    fig = px.bar(data, x="Window", y="mm", title="Rainfall by window",
                 labels={"mm": "mm", "Window": ""}, color_discrete_sequence=[ACCENT],
                 template="plotly_dark", text="mm")
    fig.update_traces(texttemplate="%{text:.1f}")
    show_plot(_style(fig, height).update_layout(hovermode="closest"))


def chart_row(*draw_fns):
    for col, fn in zip(st.columns(len(draw_fns)), draw_fns):
        with col:
            fn()


# =============================================================================
# 5. SHARED UI PIECES
# =============================================================================
def section(title: str):
    st.header(title)


def score_card_html(label, score, note="", href=None):
    color = score_color(score)
    width = 0 if score is None or pd.isna(score) else int(score)
    return (f"<div class='gp-card'><div class='gp-card-label'>{label}</div>"
            f"<div class='gp-card-score' style='color:{color}'>{fmt(score, 0)}<span>/ 100</span></div>"
            f"<div class='gp-bar'><div style='width:{width}%;background:{color}'></div></div>"
            f"{f'<div class=gp-card-note>{note}</div>' if note else ''}</div>")


def value_card_html(label, value_text, note=""):
    return (f"<div class='gp-card'><div class='gp-card-label'>{label}</div>"
            f"<div class='gp-card-score'>{value_text}</div>"
            f"{f'<div class=gp-card-note>{note}</div>' if note else ''}</div>")


def page_title(title: str, score, caption: str):
    st.title(title)
    st.caption(caption)
    color = score_color(score)
    st.markdown(f"<div class='gp-page-score' style='color:{color}'>{fmt(score, 0)}<span>/ 100</span></div>",
                unsafe_allow_html=True)


def contributing_factors(items):
    """items = [(label, score, weight, derived_flag)]"""
    section("Contributing factors")
    cols = st.columns(len(items))
    for col, (label, score, weight, derived) in zip(cols, items):
        note = f"Weight {weight:.0%}" + (" · derived in dashboard" if derived else "")
        with col:
            st.markdown(score_card_html(f"{label} score", score, note), unsafe_allow_html=True)
    if any(d for *_, d in items):
        st.caption("Sub-scores marked “derived” are calculated here because the backend does not send them yet.")


# =============================================================================
# 6. PAGES
# =============================================================================
def show_header(latest):
    live = not st.session_state.using_sample_data
    dot_color = "#f85149" if live else "#e3b341"
    label = "LIVE" if live else "DEMO DATA"
    st.markdown(
        f"<div class='gp-header'><div><div class='gp-title'>🌿 GreenPulse</div>"
        f"<div class='gp-sub'>Environmental Monitoring System</div></div>"
        f"<div class='gp-live'><span class='gp-dot' style='background:{dot_color}'></span>{label}</div></div>",
        unsafe_allow_html=True)
    if not live:
        st.warning("Backend not reachable - showing demo data. Check that the backend and its tunnel "
                   "are running and that BACKEND_URL points to the tunnel's public address.")
    t = pd.to_datetime(latest.get("time"), errors="coerce")
    if pd.notna(t):
        st.caption(f"Last update: {t:%Y-%m-%d %H:%M:%S}")


def show_overview(latest):
    ems = latest.get("ems")
    level = (latest.get("ems_alert_level") or "UNKNOWN").upper()
    level_color = ALERT_COLORS.get(level, "#64748b")
    driver = pretty(latest.get("ems_primary_driver"))

    st.markdown(
        f"<div class='gp-hero'><div class='gp-hero-label'>Environmental monitoring score</div>"
        f"<div class='gp-hero-score' style='color:{level_color}'>{fmt(ems, 0)}<span>/ 100</span></div>"
        f"<div class='gp-hero-level' style='color:{level_color}'>{level}</div></div>",
        unsafe_allow_html=True)

    st.write("")
    dims = [
        ("Ground Stability", latest.get("ems_ground_stability")),
        ("Human Pressure", latest.get("ems_human_pressure")),
        ("Environmental Quality", latest.get("ems_environmental_quality")),
    ]
    for col, (name, score) in zip(st.columns(3), dims):
        with col:
            st.markdown(score_card_html(name, score), unsafe_allow_html=True)
            st.button(f"Explore {name}", key=f"go_{name}", on_click=goto, args=(name,),
                      use_container_width=True)

    st.write("")
    st.markdown(
        f"<div class='gp-callout' style='border-left-color:{level_color}'>"
        f"<div class='gp-callout-title'>Alert</div>"
        f"<b style='color:{level_color}'>{level}</b> - {driver} is currently the primary driver.</div>",
        unsafe_allow_html=True)
    st.write("")
    st.markdown(
        f"<div class='gp-callout'><div class='gp-callout-title'>GreenPulse advisory</div>"
        f"{latest.get('advisory') or 'No advisory available.'}</div>",
        unsafe_allow_html=True)

    section("Monitoring map")
    lat, lon = latest.get("latitude"), latest.get("longitude")
    try:
        point = pd.DataFrame({"latitude": [float(lat)], "longitude": [float(lon)]})
        if not (-90 <= point.latitude[0] <= 90 and -180 <= point.longitude[0] <= 180):
            raise ValueError
        st.map(point, zoom=14, height=380)
        st.caption(f"Node at {abs(point.latitude[0]):.4f}° {'N' if point.latitude[0] >= 0 else 'S'}, "
                   f"{abs(point.longitude[0]):.4f}° {'E' if point.longitude[0] >= 0 else 'W'}")
    except (TypeError, ValueError):
        st.info("No valid GPS position available for this node.")


def show_ground_stability(latest, history):
    page_title("Ground Stability", latest.get("ems_ground_stability"),
               "How likely the ground around the node is to stay stable, from soil condition and rainfall.")
    contributing_factors([
        ("Soil", latest.get("soil_score"), GROUND_WEIGHTS["Soil"], False),
        ("Rainfall", latest.get("rainfall_score"), GROUND_WEIGHTS["Rainfall"], False),
    ])

    section("Ground stability history")
    line_chart(history, "ems_ground_stability", "Ground stability over time", "Score", height=320)

    section("Rainfall")
    c1, c2, c3 = st.columns(3)
    c1.metric("Last hour", fmt(latest.get("rainfall"), 1, "mm"))
    c2.metric("24 hours", fmt(latest.get("rainfall_24h"), 1, "mm"))
    c3.metric("72 hours", fmt(latest.get("rainfall_72h"), 1, "mm"))
    chart_row(lambda: rainfall_bars(latest),
              lambda: line_chart(history, "rainfall_score", "Rainfall score over time", "Score"))

    section("Soil")
    c1, c2 = st.columns(2)
    c1.metric("Soil moisture", fmt(latest.get("soil_moisture"), 1, "%"))
    c2.metric("Soil score", fmt(latest.get("soil_score"), 0, "/ 100"))
    chart_row(lambda: line_chart(history, "soil_score", "Soil score over time", "Score"),
              lambda: line_chart(history, "soil_moisture", "Soil moisture over time", "%"))


def show_human_pressure(latest, history):
    co2_score, co2_derived = sub_score(latest, "co2_score", derive_co2_score(latest.get("co2")))
    page_title("Human Pressure", latest.get("ems_human_pressure"),
               "How much human presence and activity the node is picking up.")
    contributing_factors([
        ("PIR", latest.get("pir_score"), HUMAN_WEIGHTS["PIR"], False),
        ("Sound", latest.get("sound_score"), HUMAN_WEIGHTS["Sound"], False),
        ("CO₂", co2_score, HUMAN_WEIGHTS["CO₂"], co2_derived),
    ])

    section("Human pressure history")
    line_chart(history, "ems_human_pressure", "Human pressure over time", "Score", height=320)

    section("Local activity")
    c1, c2, c3 = st.columns(3)
    c1.metric("PIR activity", fmt(latest.get("pir_score"), 0, "/ 100"))
    c2.metric("Sound activity", fmt(latest.get("sound_score"), 0, "/ 100"))
    c3.metric("CO₂ concentration", fmt(latest.get("co2"), 0, "ppm"))
    chart_row(lambda: line_chart(history, "pir_score", "PIR score over time", "Score"),
              lambda: line_chart(history, "sound_score", "Sound score over time", "Score"))
    chart_row(lambda: line_chart(history, "co2", "CO₂ over time", "ppm"),
              lambda: scatter_chart(history, "pir_score", "sound_score", "PIR vs sound",
                                    "PIR score", "Sound score"))


@st.cache_data(ttl=600, show_spinner=False)
def _open_meteo_air(lat: float, lon: float):
    """Open-Meteo PM2.5 / PM10 (modelled) - fallback if the backend sends none."""
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", timeout=(3.05, 10), params={
        "latitude": round(lat, 3), "longitude": round(lon, 3),
        "current": "pm2_5,pm10", "hourly": "pm2_5,pm10",
        "past_days": 2, "forecast_days": 1, "timezone": "auto"})
    r.raise_for_status()
    j = r.json()
    hist = pd.DataFrame(j.get("hourly") or {})
    hist["time"] = pd.to_datetime(hist["time"])
    now = pd.to_datetime((j.get("current") or {}).get("time"))
    if pd.notna(now):
        hist = hist[hist["time"] <= now]
    return j.get("current") or {}, hist


def show_environmental_quality(latest, history):
    page_title("Environmental Quality", latest.get("ems_environmental_quality"),
               "Air quality and microclimate at the node.")
    section("Contributing factors")
    aqi = latest.get("aqi")
    cards = [
        ("AQI", fmt(aqi, 0), f"{aqi_category(aqi)} · weight {ENV_WEIGHTS['AQI']:.0%}"),
        ("Temperature", fmt(latest.get("temperature"), 1, "°C"), f"Weight {ENV_WEIGHTS['Temperature']:.0%}"),
        ("Humidity", fmt(latest.get("humidity_percent"), 0, "%"), f"Weight {ENV_WEIGHTS['Humidity']:.0%}"),
    ]
    for col, (label, value, note) in zip(st.columns(3), cards):
        with col:
            st.markdown(value_card_html(label, value, note), unsafe_allow_html=True)

    section("Environmental quality history")
    line_chart(history, "ems_environmental_quality", "Environmental quality over time", "Score", height=320)

    # PM2.5 / PM10: backend first, Open-Meteo as fallback
    pm25, pm10 = latest.get("pm2_5"), latest.get("pm10")
    pm_history, pm_source = history, None
    if history[["pm2_5", "pm10"]].isna().all().all() or pd.isna(pm25) or pd.isna(pm10):
        try:
            cur, om_hist = _open_meteo_air(float(latest.get("latitude")), float(latest.get("longitude")))
            pm25 = pm25 if pd.notna(pm25) else cur.get("pm2_5")
            pm10 = pm10 if pd.notna(pm10) else cur.get("pm10")
            if history[["pm2_5", "pm10"]].isna().all().all():
                pm_history, pm_source = om_hist, "Open-Meteo (modelled, not measured by the node)"
        except Exception:
            pm_source = "unavailable"

    section("Air quality")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AQI", fmt(aqi, 0))
    c2.metric("Category", aqi_category(aqi))
    c3.metric("PM2.5", fmt(pm25, 1, "µg/m³"))
    c4.metric("PM10", fmt(pm10, 1, "µg/m³"))
    if pm_source:
        st.caption(f"PM2.5 / PM10 source: {pm_source}.")
    chart_row(lambda: line_chart(history, "aqi", "AQI over time", "AQI"),
              lambda: multi_line_chart(pm_history, {"pm2_5": "PM2.5", "pm10": "PM10"},
                                       "PM2.5 / PM10", "µg/m³"))

    section("Microclimate")
    c1, c2 = st.columns(2)
    c1.metric("Temperature", fmt(latest.get("temperature"), 1, "°C"))
    c2.metric("Humidity", fmt(latest.get("humidity_percent"), 0, "%"))
    chart_row(lambda: line_chart(history, "temperature", "Temperature over time", "°C"),
              lambda: line_chart(history, "humidity_percent", "Humidity over time", "%"))
    scatter_chart(history, "temperature", "humidity_percent", "Temperature vs humidity",
                  "Temperature (°C)", "Humidity (%)")


# =============================================================================
# 7. NAVIGATION + MAIN
# =============================================================================
def show_sidebar():
    with st.sidebar:
        st.markdown("<div class='gp-brand'>🌿 GreenPulse</div>", unsafe_allow_html=True)
        st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        if st.button("Refresh data", use_container_width=True):
            st.rerun()
        online = not st.session_state.using_sample_data
        color = ACCENT if online else "#e3b341"
        text = "Online" if online else "Demo data"
        st.markdown(f"<div class='gp-status'>System: <span style='color:{color}'>●</span> {text}</div>",
                    unsafe_allow_html=True)


def main():
    apply_custom_css()

    try:
        latest = get_sensor_data()
        history = prepare_history(get_sensor_history())
    except Exception as exc:                       # bad JSON, etc.
        st.title("GreenPulse")
        st.error(f"Could not load sensor data: {exc}")
        st.stop()

    show_sidebar()
    show_header(latest)

    page = st.session_state.page
    if page == "Ground Stability":
        show_ground_stability(latest, history)
    elif page == "Human Pressure":
        show_human_pressure(latest, history)
    elif page == "Environmental Quality":
        show_environmental_quality(latest, history)
    else:
        show_overview(latest)


main()
