from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================
# OPTIONAL ADVANCED HOOKS
# =========================
try:
    from GreenLightPlus import GreenhouseGeometry  # type: ignore

    GLP_AVAILABLE = True
    GLP_ERROR = ""
except Exception as exc:  # pragma: no cover - optional package
    GreenhouseGeometry = None
    GLP_AVAILABLE = False
    GLP_ERROR = str(exc)

try:
    import pcse  # type: ignore

    PCSE_AVAILABLE = True
    PCSE_ERROR = ""
except Exception as exc:  # pragma: no cover - optional package
    pcse = None
    PCSE_AVAILABLE = False
    PCSE_ERROR = str(exc)

try:
    import mlagents_envs  # type: ignore  # noqa: F401

    MLAGENTS_AVAILABLE = True
    MLAGENTS_ERROR = ""
except Exception as exc:  # pragma: no cover - optional package
    MLAGENTS_AVAILABLE = False
    MLAGENTS_ERROR = str(exc)


# =========================
# CONFIG
# =========================
INDONESIAN_LOCATIONS = {
    "Jakarta": (-6.2088, 106.8456),
    "Bandung": (-6.9175, 107.6191),
    "Surabaya": (-7.2575, 112.7521),
    "Yogyakarta": (-7.7956, 110.3695),
    "Denpasar": (-8.6705, 115.2126),
    "Medan": (3.5952, 98.6722),
    "Makassar": (-5.1477, 119.4327),
    "Palembang": (-2.9761, 104.7754),
    "Bogor": (-6.5971, 106.8060),
    "Malang": (-7.9666, 112.6326),
}

DEFAULT_CROP = {
    "name": "Tomato",
    "base_temp_c": 10.0,
    "optimal_temp_c": 25.0,
    "max_temp_c": 36.0,
    "target_soil_min": 55.0,
    "target_soil_max": 72.0,
    "target_humidity_min": 62.0,
    "target_humidity_max": 82.0,
}


class WeatherProvider(str, Enum):
    OPENWEATHERMAP = "OpenWeatherMap"
    METEOSTAT = "Meteostat"
    SYNTHETIC = "Synthetic Fallback"


@dataclass
class WeatherSnapshot:
    provider: str
    location: str
    lat: float
    lon: float
    temperature_c: float
    humidity_pct: float
    wind_mps: float
    cloud_pct: float
    rain_mm_h: float
    pressure_hpa: float
    fetched_at: str
    source_status: str

    @property
    def radiation_w_m2(self) -> float:
        cloud_factor = 1.0 - np.clip(self.cloud_pct, 0, 100) / 100.0
        daylight_hint = 0.75 if 6 <= datetime.now().hour <= 17 else 0.15
        return float(np.clip(900 * cloud_factor * daylight_hint, 35, 950))


@dataclass
class TwinState:
    hour: int
    inside_temp_c: float = 25.0
    outside_temp_c: float = 30.0
    humidity_pct: float = 70.0
    soil_moisture_pct: float = 45.0
    co2_ppm: float = 430.0
    biomass_g_m2: float = 35.0
    leaf_area_index: float = 0.35
    crop_stage_pct: float = 5.0
    water_used_l: float = 0.0
    energy_used_kwh: float = 0.0
    stress_index: float = 0.0
    disease_risk_pct: float = 0.0
    yield_forecast_kg_m2: float = 0.2


@dataclass
class ControlAction:
    irrigation_l: float = 0.0
    ventilation_pct: float = 20.0
    fan_pct: float = 0.0
    shade_pct: float = 0.0
    heating_kw: float = 0.0
    co2_injection_ppm: float = 0.0
    reason: str = "Idle"


class AutomationMode(str, Enum):
    RULE_BASED = "Rule-based expert"
    OPTIMIZER = "Predictive optimizer"
    RL_READY = "RL/ML-Agents ready"


# =========================
# WEATHER CLIENT
# =========================
class WeatherClient:
    def __init__(
        self,
        provider: WeatherProvider,
        location: str,
        lat: float,
        lon: float,
        openweather_api_key: str | None = None,
    ):
        self.provider = provider
        self.location = location
        self.lat = lat
        self.lon = lon
        self.openweather_api_key = openweather_api_key

    def fetch(self) -> WeatherSnapshot:
        if self.provider == WeatherProvider.OPENWEATHERMAP and self.openweather_api_key:
            return self._fetch_openweathermap()
        if self.provider == WeatherProvider.METEOSTAT:
            return self._fetch_meteostat()
        return self._synthetic_snapshot("Offline synthetic weather forcing")

    def _fetch_openweathermap(self) -> WeatherSnapshot:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "appid": self.openweather_api_key,
            "units": "metric",
        }
        try:
            response = requests.get(url, params=params, timeout=12)
            response.raise_for_status()
            payload = response.json()
            rain = payload.get("rain", {})
            return WeatherSnapshot(
                provider=WeatherProvider.OPENWEATHERMAP.value,
                location=self.location,
                lat=self.lat,
                lon=self.lon,
                temperature_c=float(payload["main"]["temp"]),
                humidity_pct=float(payload["main"]["humidity"]),
                wind_mps=float(payload.get("wind", {}).get("speed", 0.0)),
                cloud_pct=float(payload.get("clouds", {}).get("all", 0.0)),
                rain_mm_h=float(rain.get("1h", rain.get("3h", 0.0))),
                pressure_hpa=float(payload["main"].get("pressure", 1010.0)),
                fetched_at=datetime.now(timezone.utc).isoformat(),
                source_status="Live OpenWeatherMap data",
            )
        except Exception as exc:
            return self._synthetic_snapshot(f"OpenWeatherMap fallback: {exc}")

    def _fetch_meteostat(self) -> WeatherSnapshot:
        try:
            import meteostat as ms  # type: ignore

            now = datetime.utcnow()
            start = now - timedelta(hours=6)
            point = ms.Point(self.lat, self.lon)
            if hasattr(ms, "Hourly"):
                data = ms.Hourly(point, start, now).fetch()
            else:
                data = ms.hourly(point, start, now).fetch()
            if data.empty:
                return self._synthetic_snapshot("Meteostat returned no nearby hourly data")

            latest = data.dropna(how="all").tail(1).iloc[0]
            return WeatherSnapshot(
                provider=WeatherProvider.METEOSTAT.value,
                location=self.location,
                lat=self.lat,
                lon=self.lon,
                temperature_c=float(latest.get("temp", np.nan) or 28.0),
                humidity_pct=float(latest.get("rhum", np.nan) or 75.0),
                wind_mps=float((latest.get("wspd", 0.0) or 0.0) / 3.6),
                cloud_pct=float(latest.get("coco", 3.0) or 3.0) * 12.5,
                rain_mm_h=float(latest.get("prcp", 0.0) or 0.0),
                pressure_hpa=float(latest.get("pres", 1010.0) or 1010.0),
                fetched_at=datetime.now(timezone.utc).isoformat(),
                source_status="Nearest Meteostat hourly observation",
            )
        except Exception as exc:
            return self._synthetic_snapshot(f"Meteostat fallback: {exc}")

    def _synthetic_snapshot(self, status: str) -> WeatherSnapshot:
        hour = datetime.now().hour
        daily_wave = math.sin((hour - 6) / 24 * 2 * math.pi)
        temp = 28.5 + 4.0 * daily_wave + np.random.uniform(-0.8, 0.8)
        humidity = 76.0 - 12.0 * daily_wave + np.random.uniform(-3.0, 3.0)
        cloud = np.clip(45 + np.random.normal(0, 12), 5, 95)
        rain = float(max(0.0, np.random.normal(0.2, 0.6)))
        return WeatherSnapshot(
            provider=WeatherProvider.SYNTHETIC.value,
            location=self.location,
            lat=self.lat,
            lon=self.lon,
            temperature_c=float(temp),
            humidity_pct=float(np.clip(humidity, 35, 98)),
            wind_mps=float(np.clip(np.random.normal(2.5, 0.8), 0, 10)),
            cloud_pct=float(cloud),
            rain_mm_h=rain,
            pressure_hpa=float(np.clip(np.random.normal(1010, 5), 995, 1020)),
            fetched_at=datetime.now(timezone.utc).isoformat(),
            source_status=status,
        )


# =========================
# DIGITAL TWIN SIMULATOR
# =========================
class GreenhouseDigitalTwin:
    def __init__(self, weather: WeatherSnapshot, crop: dict[str, float | str]):
        self.weather = weather
        self.crop = crop
        self.state = TwinState(
            hour=0,
            inside_temp_c=weather.temperature_c - 1.2,
            outside_temp_c=weather.temperature_c,
            humidity_pct=weather.humidity_pct,
            soil_moisture_pct=48.0,
        )
        self.history: list[dict[str, Any]] = []

    def reset_weather(self, weather: WeatherSnapshot) -> None:
        self.weather = weather
        self.state.outside_temp_c = weather.temperature_c

    def step(self, action: ControlAction) -> TwinState:
        s = self.state
        w = self.weather

        solar_gain = w.radiation_w_m2 / 900 * 1.4
        ventilation_cooling = (action.ventilation_pct / 100) * (s.inside_temp_c - w.temperature_c) * 0.12
        fan_cooling = action.fan_pct / 100 * 0.55
        shade_cooling = action.shade_pct / 100 * solar_gain * 0.75
        heating = action.heating_kw * 0.18
        thermal_noise = np.random.uniform(-0.12, 0.12)

        s.inside_temp_c += (
            0.08 * (w.temperature_c - s.inside_temp_c)
            + solar_gain
            - ventilation_cooling
            - fan_cooling
            - shade_cooling
            + heating
            + thermal_noise
        )

        evapotranspiration = max(0.25, 0.8 + 0.09 * (s.inside_temp_c - 24) + w.wind_mps * 0.05)
        rain_capture = min(w.rain_mm_h * 0.18, 2.0)
        s.soil_moisture_pct += action.irrigation_l * 0.42 + rain_capture - evapotranspiration
        s.soil_moisture_pct = float(np.clip(s.soil_moisture_pct, 0, 100))

        humidity_delta = (
            0.09 * (w.humidity_pct - s.humidity_pct)
            + action.irrigation_l * 0.2
            - action.ventilation_pct * 0.035
            - action.fan_pct * 0.015
        )
        s.humidity_pct = float(np.clip(s.humidity_pct + humidity_delta, 25, 100))

        s.co2_ppm += action.co2_injection_ppm * 0.35 - action.ventilation_pct * 0.55 - 3.5
        s.co2_ppm = float(np.clip(s.co2_ppm, 350, 1200))

        temp_score = triangular_score(
            s.inside_temp_c,
            float(self.crop["base_temp_c"]),
            float(self.crop["optimal_temp_c"]),
            float(self.crop["max_temp_c"]),
        )
        soil_score = band_score(
            s.soil_moisture_pct,
            float(self.crop["target_soil_min"]),
            float(self.crop["target_soil_max"]),
        )
        humidity_score = band_score(
            s.humidity_pct,
            float(self.crop["target_humidity_min"]),
            float(self.crop["target_humidity_max"]),
        )
        co2_score = band_score(s.co2_ppm, 420, 900)
        growth_factor = np.mean([temp_score, soil_score, humidity_score, co2_score])

        s.stress_index = float(np.clip((1 - growth_factor) * 100, 0, 100))
        s.disease_risk_pct = float(
            np.clip((s.humidity_pct - 72) * 1.6 + max(0, 24 - s.inside_temp_c) * 2.3, 0, 100)
        )
        s.biomass_g_m2 += 2.6 * growth_factor * (1 - s.crop_stage_pct / 125)
        s.leaf_area_index = float(np.clip(0.25 + s.biomass_g_m2 / 190, 0.1, 5.5))
        s.crop_stage_pct = float(np.clip(s.crop_stage_pct + 0.22 * growth_factor, 0, 100))
        s.yield_forecast_kg_m2 = float(np.clip(s.biomass_g_m2 * 0.018 * growth_factor, 0, 14))

        s.water_used_l += action.irrigation_l
        s.energy_used_kwh += action.fan_pct / 100 * 0.12 + action.heating_kw * 0.08
        s.hour += 1

        self.history.append(
            {
                "hour": s.hour,
                "inside_temp_c": s.inside_temp_c,
                "outside_temp_c": w.temperature_c,
                "humidity_pct": s.humidity_pct,
                "soil_moisture_pct": s.soil_moisture_pct,
                "co2_ppm": s.co2_ppm,
                "biomass_g_m2": s.biomass_g_m2,
                "leaf_area_index": s.leaf_area_index,
                "crop_stage_pct": s.crop_stage_pct,
                "water_used_l": s.water_used_l,
                "energy_used_kwh": s.energy_used_kwh,
                "stress_index": s.stress_index,
                "disease_risk_pct": s.disease_risk_pct,
                "yield_forecast_kg_m2": s.yield_forecast_kg_m2,
                "irrigation_l": action.irrigation_l,
                "ventilation_pct": action.ventilation_pct,
                "fan_pct": action.fan_pct,
                "shade_pct": action.shade_pct,
                "heating_kw": action.heating_kw,
                "co2_injection_ppm": action.co2_injection_ppm,
                "decision": action.reason,
            }
        )
        return s


def triangular_score(value: float, low: float, optimal: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if value == optimal:
        return 1.0
    if value < optimal:
        return float((value - low) / (optimal - low))
    return float((high - value) / (high - optimal))


def band_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    return float(np.clip(1.0 - distance / 35.0, 0, 1))


# =========================
# AI / AUTOMATION CONTROLLER
# =========================
def ai_controller(state: TwinState, weather: WeatherSnapshot, mode: AutomationMode) -> ControlAction:
    action = ControlAction()
    reasons: list[str] = []

    if state.soil_moisture_pct < 38:
        action.irrigation_l = 14.0
        reasons.append("soil critical")
    elif state.soil_moisture_pct < 55:
        action.irrigation_l = 7.0
        reasons.append("soil below target")

    if state.inside_temp_c > 31:
        action.ventilation_pct = 85.0
        action.fan_pct = 70.0
        action.shade_pct = 45.0
        reasons.append("heat control")
    elif state.inside_temp_c > 28:
        action.ventilation_pct = 60.0
        action.fan_pct = 35.0
        action.shade_pct = 25.0
        reasons.append("mild cooling")

    if state.inside_temp_c < 19:
        action.heating_kw = 3.0
        action.ventilation_pct = min(action.ventilation_pct, 25.0)
        reasons.append("night heating")

    if state.humidity_pct > 86:
        action.ventilation_pct = max(action.ventilation_pct, 70.0)
        action.fan_pct = max(action.fan_pct, 45.0)
        reasons.append("fungal risk ventilation")

    if state.co2_ppm < 430 and action.ventilation_pct < 60:
        action.co2_injection_ppm = 90.0
        reasons.append("CO2 enrichment")

    if mode == AutomationMode.OPTIMIZER:
        action = optimizer_adjustment(action, state, weather)
        reasons.append("predictive optimizer")
    elif mode == AutomationMode.RL_READY:
        action = rl_ready_policy(action, state, weather)
        reasons.append("rl observation policy")

    action.reason = ", ".join(reasons) if reasons else "stable"
    return action


def optimizer_adjustment(action: ControlAction, state: TwinState, weather: WeatherSnapshot) -> ControlAction:
    expected_heat = weather.radiation_w_m2 > 650 or weather.temperature_c > 31
    expected_rain = weather.rain_mm_h > 1.0

    if expected_heat and state.soil_moisture_pct < 62:
        action.irrigation_l = max(action.irrigation_l, 9.0)
    if expected_rain:
        action.irrigation_l *= 0.35
    if expected_heat:
        action.shade_pct = max(action.shade_pct, 35.0)
        action.ventilation_pct = max(action.ventilation_pct, 70.0)
    if state.disease_risk_pct > 65:
        action.fan_pct = max(action.fan_pct, 70.0)
        action.ventilation_pct = max(action.ventilation_pct, 80.0)
    return action


def rl_ready_policy(action: ControlAction, state: TwinState, weather: WeatherSnapshot) -> ControlAction:
    obs = unity_observation_vector(state, weather)
    heat_pressure = obs[0] - 0.7
    water_deficit = 0.58 - obs[2]
    action.irrigation_l = float(np.clip(action.irrigation_l + water_deficit * 12, 0, 18))
    action.ventilation_pct = float(np.clip(action.ventilation_pct + heat_pressure * 70, 10, 100))
    action.fan_pct = float(np.clip(action.fan_pct + max(0, heat_pressure) * 55, 0, 100))
    return action


# =========================
# CONNECTOR BRIDGES
# =========================
def pcse_bridge_payload(twin: GreenhouseDigitalTwin) -> dict[str, Any]:
    df = pd.DataFrame(twin.history)
    if df.empty:
        weather_rows: list[dict[str, Any]] = []
    else:
        weather_rows = (
            df[["hour", "outside_temp_c", "inside_temp_c", "humidity_pct"]]
            .tail(24)
            .round(3)
            .to_dict(orient="records")
        )

    return {
        "status": "ready" if PCSE_AVAILABLE else "adapter_only",
        "pcse_available": PCSE_AVAILABLE,
        "model_target": "WOFOST/LINGRA/LINTUL via PCSE",
        "note": "PCSE production run still needs crop, soil, site, and agromanagement parameter sets.",
        "weather_hint": weather_rows,
    }


def farmvibes_payload(twin: GreenhouseDigitalTwin, weather: WeatherSnapshot) -> dict[str, Any]:
    return {
        "farm": {
            "country": "Indonesia",
            "location": weather.location,
            "lat": weather.lat,
            "lon": weather.lon,
        },
        "weather": asdict(weather),
        "latest_greenhouse_state": asdict(twin.state),
        "workflows": [
            "farm_ai/agriculture/weather",
            "farm_ai/agriculture/ndvi",
            "farm_ai/agriculture/heatmap",
        ],
        "integration_mode": "Export payload for FarmVibes.AI workflow client or cluster API",
    }


def unity_observation_vector(state: TwinState, weather: WeatherSnapshot) -> list[float]:
    return [
        normalize(state.inside_temp_c, 15, 40),
        normalize(state.humidity_pct, 20, 100),
        normalize(state.soil_moisture_pct, 0, 100),
        normalize(state.co2_ppm, 350, 1200),
        normalize(weather.radiation_w_m2, 0, 950),
        normalize(state.disease_risk_pct, 0, 100),
        normalize(state.crop_stage_pct, 0, 100),
    ]


def unity_training_config() -> str:
    return """behaviors:
  GreenhouseIrrigation:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 0.0003
      beta: 0.005
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    network_settings:
      normalize: true
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000
"""


def glp_status() -> dict[str, Any]:
    if not GLP_AVAILABLE:
        return {
            "available": False,
            "status": "GreenLightPlus belum terpasang",
            "detail": GLP_ERROR or "Install package GLP sesuai dokumentasi proyek.",
        }

    return {
        "available": True,
        "status": "GreenLightPlus detected",
        "detail": "Ready for geometry/model/RL coupling. Use GLP weather files for high-fidelity greenhouse simulation.",
        "geometry_class": str(GreenhouseGeometry),
    }


def normalize(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0, 1))


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="AI Greenhouse Digital Twin",
    page_icon="AI",
    layout="wide",
)

st.title("AI Greenhouse Digital Twin Indonesia")
st.caption("Real-time weather forcing, self-automation, crop simulation bridge, and GreenLightPlus-ready adapter.")

with st.sidebar:
    st.header("Runtime")
    location = st.selectbox("Lokasi Indonesia", list(INDONESIAN_LOCATIONS.keys()), index=0)
    lat, lon = INDONESIAN_LOCATIONS[location]
    provider = WeatherProvider(
        st.selectbox(
            "Weather provider",
            [provider.value for provider in WeatherProvider],
            index=0,
        )
    )
    api_key = st.text_input(
        "OpenWeatherMap API key",
        value=os.getenv("OPENWEATHER_API_KEY", ""),
        type="password",
        help="Bisa juga pakai environment variable OPENWEATHER_API_KEY.",
    )
    mode = AutomationMode(
        st.selectbox(
            "Automation mode",
            [mode.value for mode in AutomationMode],
            index=1,
        )
    )
    steps = st.slider("Simulation steps (hours)", 1, 240, 48)
    auto_refresh_weather = st.toggle("Refresh weather before run", value=True)
    delay = st.slider("Execution delay per step", 0.0, 0.25, 0.02, 0.01)

    st.divider()
    st.header("Crop")
    crop = DEFAULT_CROP.copy()
    crop["name"] = st.selectbox("Crop profile", ["Tomato", "Chili", "Cucumber", "Melon"], index=0)
    crop["optimal_temp_c"] = st.slider("Optimal temp (C)", 20.0, 32.0, float(crop["optimal_temp_c"]), 0.5)
    crop["target_soil_min"] = st.slider("Target soil min (%)", 35.0, 70.0, float(crop["target_soil_min"]), 1.0)
    crop["target_soil_max"] = st.slider("Target soil max (%)", 55.0, 90.0, float(crop["target_soil_max"]), 1.0)

    st.divider()
    if st.button("Reset twin"):
        st.session_state.pop("weather", None)
        st.session_state.pop("twin", None)
        st.rerun()


def load_weather() -> WeatherSnapshot:
    client = WeatherClient(provider, location, lat, lon, api_key or None)
    return client.fetch()


if "weather" not in st.session_state:
    st.session_state.weather = load_weather()

if "twin" not in st.session_state:
    st.session_state.twin = GreenhouseDigitalTwin(st.session_state.weather, crop)

weather: WeatherSnapshot = st.session_state.weather
twin: GreenhouseDigitalTwin = st.session_state.twin
twin.crop = crop

top1, top2, top3, top4 = st.columns(4)
top1.metric("Weather Source", weather.provider, weather.source_status)
top2.metric("Outdoor Temp", f"{weather.temperature_c:.1f} C")
top3.metric("Humidity", f"{weather.humidity_pct:.0f}%")
top4.metric("Rain", f"{weather.rain_mm_h:.1f} mm/h")

run_col, weather_col, export_col = st.columns([1, 1, 1])
run = run_col.button("Run Self-Automation", type="primary", use_container_width=True)
refresh = weather_col.button("Refresh Weather", use_container_width=True)
export_json = export_col.button("Export Connector Payloads", use_container_width=True)

if refresh:
    st.session_state.weather = load_weather()
    twin.reset_weather(st.session_state.weather)
    st.rerun()

if run:
    if auto_refresh_weather:
        st.session_state.weather = load_weather()
        twin.reset_weather(st.session_state.weather)
        weather = st.session_state.weather

    progress = st.progress(0)
    status = st.empty()
    for idx in range(steps):
        action = ai_controller(twin.state, weather, mode)
        twin.step(action)
        progress.progress((idx + 1) / steps)
        status.write(f"Step {idx + 1}/{steps}: {action.reason}")
        if delay:
            time.sleep(delay)
    st.success("Self-automation execution complete")


st.subheader("Digital Twin Summary")
state = twin.state
summary_cols = st.columns(6)
summary_cols[0].metric("Inside Temp", f"{state.inside_temp_c:.1f} C")
summary_cols[1].metric("Soil", f"{state.soil_moisture_pct:.1f}%")
summary_cols[2].metric("CO2", f"{state.co2_ppm:.0f} ppm")
summary_cols[3].metric("Stress", f"{state.stress_index:.0f}%")
summary_cols[4].metric("Disease Risk", f"{state.disease_risk_pct:.0f}%")
summary_cols[5].metric("Yield Forecast", f"{state.yield_forecast_kg_m2:.2f} kg/m2")

history = pd.DataFrame(twin.history)
chart_tab, action_tab, connector_tab, raw_tab = st.tabs(
    ["Analytics", "Automation Decisions", "Advanced Connectors", "Raw Data"]
)

with chart_tab:
    if history.empty:
        st.info("Run simulation untuk mengisi grafik.")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        axes[0, 0].plot(history["hour"], history["inside_temp_c"], label="Inside")
        axes[0, 0].plot(history["hour"], history["outside_temp_c"], label="Outside")
        axes[0, 0].set_title("Temperature")
        axes[0, 0].legend()

        axes[0, 1].plot(history["hour"], history["soil_moisture_pct"], label="Soil")
        axes[0, 1].plot(history["hour"], history["humidity_pct"], label="Humidity")
        axes[0, 1].set_title("Water Balance")
        axes[0, 1].legend()

        axes[1, 0].plot(history["hour"], history["biomass_g_m2"], label="Biomass")
        axes[1, 0].plot(history["hour"], history["yield_forecast_kg_m2"], label="Yield forecast")
        axes[1, 0].set_title("Crop Growth")
        axes[1, 0].legend()

        axes[1, 1].plot(history["hour"], history["stress_index"], label="Stress")
        axes[1, 1].plot(history["hour"], history["disease_risk_pct"], label="Disease risk")
        axes[1, 1].set_title("Risk")
        axes[1, 1].legend()

        fig.tight_layout()
        st.pyplot(fig)

with action_tab:
    if history.empty:
        st.info("Automation decision log belum ada.")
    else:
        columns = [
            "hour",
            "irrigation_l",
            "ventilation_pct",
            "fan_pct",
            "shade_pct",
            "heating_kw",
            "co2_injection_ppm",
            "decision",
        ]
        st.dataframe(history[columns].tail(80), use_container_width=True)

with connector_tab:
    glp = glp_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("GreenLightPlus", "Ready" if glp["available"] else "Adapter")
    c2.metric("PCSE", "Ready" if PCSE_AVAILABLE else "Adapter")
    c3.metric("Unity ML-Agents", "Ready" if MLAGENTS_AVAILABLE else "Adapter")

    with st.expander("GreenLightPlus bridge", expanded=True):
        st.json(glp)
        st.write(
            "GLP bridge disiapkan untuk coupling high-fidelity greenhouse model. "
            "Saat package tersedia, adapter bisa diarahkan ke geometry/model/RL runner GLP."
        )

    with st.expander("PCSE crop simulation bridge"):
        st.json(pcse_bridge_payload(twin))

    with st.expander("FarmVibes.AI payload"):
        st.json(farmvibes_payload(twin, weather))

    with st.expander("Unity ML-Agents observation + PPO config"):
        st.json(
            {
                "mlagents_available": MLAGENTS_AVAILABLE,
                "observation_vector": unity_observation_vector(state, weather),
                "action_space": [
                    "irrigation_l",
                    "ventilation_pct",
                    "fan_pct",
                    "shade_pct",
                    "heating_kw",
                    "co2_injection_ppm",
                ],
            }
        )
        st.code(unity_training_config(), language="yaml")

with raw_tab:
    st.json({"weather": asdict(weather), "state": asdict(state)})
    if not history.empty:
        st.dataframe(history.tail(120), use_container_width=True)

if export_json:
    payload = {
        "weather": asdict(weather),
        "current_state": asdict(state),
        "pcse": pcse_bridge_payload(twin),
        "farmvibes": farmvibes_payload(twin, weather),
        "unity": {
            "observation_vector": unity_observation_vector(state, weather),
            "training_config": unity_training_config(),
        },
        "greenlightplus": glp_status(),
    }
    st.download_button(
        "Download digital_twin_payload.json",
        data=json.dumps(payload, indent=2),
        file_name="digital_twin_payload.json",
        mime="application/json",
    )
