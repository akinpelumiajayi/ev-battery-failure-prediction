"""
Interactive demo for the EV Battery Failure Prediction API.

A fleet-maintenance view: set a vehicle's telemetry, get the failure risk,
the decision (inspect / don't inspect) and the top factors driving it.
Talks to the FastAPI service over HTTP — the UI holds no model logic.

    API_URL=http://localhost:8000 streamlit run demo/streamlit_app.py
"""

from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
# Generous because free-tier hosts (e.g. Render) sleep idle services; the
# first request after a nap can take 30-60 s while the API container boots.
TIMEOUT = 60

st.set_page_config(page_title="EV Battery Failure Risk", page_icon="🔋", layout="wide")

PRESETS = {
    "Healthy commuter (2022, low mileage)": dict(
        vehicle_brand="Hyundai",
        vehicle_type="Crossover",
        battery_chemistry="NMC",
        manufacturing_year=2022,
        vehicle_age_years=2.5,
        odometer_km=40_000,
        cycle_count=420,
        state_of_health=95.0,
        battery_health_percent=95.5,
        internal_resistance=0.25,
        cell_temperature_avg=24.0,
        cell_temperature_max=33.0,
        fast_charge_ratio=0.15,
        overcharge_events=0,
        BMS_warning_count=0,
        cooling_system_health=95.0,
        capacity_loss_percent=4.0,
        previous_faults=0,
    ),
    "Hard-worked fleet van (2017, hot climate)": dict(
        vehicle_brand="Nissan",
        vehicle_type="Van",
        battery_chemistry="LMO",
        manufacturing_year=2017,
        vehicle_age_years=8.5,
        odometer_km=330_000,
        cycle_count=2900,
        state_of_health=62.0,
        battery_health_percent=61.0,
        internal_resistance=0.85,
        cell_temperature_avg=41.0,
        cell_temperature_max=62.0,
        fast_charge_ratio=0.75,
        overcharge_events=4,
        BMS_warning_count=5,
        cooling_system_health=70.0,
        capacity_loss_percent=36.0,
        previous_faults=4,
    ),
}


WAKE_DEADLINE = int(
    os.environ.get("API_WAKE_DEADLINE", "180")
)  # seconds; a free-tier API can take 1-3 min to wake (image pull + startup)


class APIUnavailable(Exception):
    """The API did not return valid model info before the wake deadline."""


@st.cache_data(ttl=60)
def _fetch_model_info() -> dict:
    # While a sleeping host spins up, its proxy can drop the connection, answer
    # 5xx/4xx, or serve a non-JSON placeholder page. Treat anything other than
    # valid model JSON as "not ready yet" and retry until the deadline. Raises on
    # final failure; st.cache_data does not cache exceptions, so the next page
    # load (or Retry) tries again.
    deadline = time.monotonic() + WAKE_DEADLINE
    attempts = 0
    while True:
        attempts += 1
        try:
            resp = requests.get(f"{API_URL}/model", timeout=TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "model_name" in data:
                    return data
                last = f"HTTP 200 without model info ({resp.headers.get('content-type', '?')})"
            else:
                last = f"HTTP {resp.status_code}"
        except (requests.RequestException, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if time.monotonic() >= deadline:
            raise APIUnavailable(f"{last} (after {attempts} attempts)")
        time.sleep(3)


def model_info() -> tuple[dict | None, str]:
    try:
        return _fetch_model_info(), ""
    except APIUnavailable as exc:
        return None, str(exc)


with st.spinner("Waking up the prediction API (free hosting sleeps when idle; this can take 1-3 minutes)..."):
    info, api_error = model_info()

st.title("🔋 EV Battery Failure Risk")
st.caption(
    "Predicts whether a battery pack is at risk of failure from BMS telemetry, charging and driving "
    "behaviour. Part of an end-to-end MLOps project — "
    "[source on GitHub](https://github.com/akinpelumiajayi/ev-battery-failure-prediction)."
)

if info is None:
    st.warning(
        "The prediction API is still waking up. It runs on free hosting that sleeps after ~15 minutes "
        "idle. Give it a few seconds and press **Retry**."
    )
    if st.button("Retry", type="primary"):
        st.rerun()
    st.caption(f"API: `{API_URL}` · last response: {api_error}")
    st.caption("Running locally? Start the API with `make serve` or `make docker-up`.")
    st.stop()

with st.sidebar:
    st.subheader("Model in production")
    st.metric("Model", info["model_name"].replace("_", " ").title())
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC (test)", f"{info['test_metrics'].get('test_roc_auc', 0):.3f}")
    c2.metric("Recall (test)", f"{info['test_metrics'].get('test_recall', 0):.1%}")
    st.metric(
        "Decision threshold",
        f"{info['decision_threshold']:.2f}",
        help="Chosen on validation data to minimise expected cost (missed failure = 10x an inspection).",
    )
    st.caption(f"Version `{info['model_version']}` · trained {(info.get('trained_at') or '?')[:10]}")

preset_name = st.selectbox("Start from a preset", list(PRESETS))
p = PRESETS[preset_name]

st.subheader("Vehicle telemetry")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Vehicle**")
    brand = st.selectbox(
        "Brand",
        ["Tesla", "BYD", "Hyundai", "Kia", "Nissan", "Ford", "BMW", "Volkswagen"],
        index=["Tesla", "BYD", "Hyundai", "Kia", "Nissan", "Ford", "BMW", "Volkswagen"].index(
            p["vehicle_brand"]
        ),
    )
    vtype = st.selectbox(
        "Type",
        ["Sedan", "SUV", "Crossover", "Hatchback", "Truck", "Van"],
        index=["Sedan", "SUV", "Crossover", "Hatchback", "Truck", "Van"].index(p["vehicle_type"]),
    )
    chem = st.selectbox(
        "Chemistry",
        ["NMC", "LFP", "NCA", "LMO", "LTO"],
        index=["NMC", "LFP", "NCA", "LMO", "LTO"].index(p["battery_chemistry"]),
    )
    year = st.slider("Manufacturing year", 2015, 2024, p["manufacturing_year"])
    odo = st.slider("Odometer (km)", 0, 450_000, p["odometer_km"], step=5_000)
    cycles = st.slider("Charge cycles", 150, 3400, p["cycle_count"], step=10)
with col2:
    st.markdown("**Battery health**")
    soh = st.slider("State of health (%)", 45.0, 100.0, p["state_of_health"], 0.5)
    ir = st.slider("Internal resistance (Ω)", 0.03, 1.12, p["internal_resistance"], 0.01)
    cap_loss = st.slider("Capacity loss (%)", 0.0, 53.0, p["capacity_loss_percent"], 0.5)
    t_avg = st.slider("Avg cell temperature (°C)", -10.0, 65.0, p["cell_temperature_avg"], 0.5)
    t_max = st.slider("Max cell temperature (°C)", -5.0, 87.0, p["cell_temperature_max"], 0.5)
    cooling = st.slider("Cooling system health", 57.0, 100.0, p["cooling_system_health"], 0.5)
with col3:
    st.markdown("**Usage & events**")
    fast = st.slider("Fast-charge ratio", 0.0, 1.0, p["fast_charge_ratio"], 0.01)
    over = st.slider("Overcharge events", 0, 5, p["overcharge_events"])
    bms = st.slider("BMS warnings", 0, 7, p["BMS_warning_count"])
    faults = st.slider("Previous faults", 0, 6, p["previous_faults"])

payload = dict(
    vehicle_brand=brand,
    vehicle_type=vtype,
    battery_chemistry=chem,
    manufacturing_year=year,
    vehicle_age_years=max(2025 - year - 0.5, 0.5),
    odometer_km=odo,
    cycle_count=cycles,
    state_of_health=soh,
    battery_health_percent=soh,
    internal_resistance=ir,
    capacity_loss_percent=cap_loss,
    cell_temperature_avg=t_avg,
    cell_temperature_max=t_max,
    cooling_system_health=cooling,
    fast_charge_ratio=fast,
    overcharge_events=over,
    BMS_warning_count=bms,
    previous_faults=faults,
)

try:
    resp = requests.post(f"{API_URL}/explain", json=payload, params={"top_k": 8}, timeout=TIMEOUT)
    resp.raise_for_status()
    result = resp.json()
except requests.RequestException as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

st.divider()
left, right = st.columns([1, 2])
with left:
    prob = result["failure_probability"]
    band = result["risk_band"]
    colour = {"low": "🟢", "medium": "🟠", "high": "🔴"}[band]
    st.metric("Failure probability", f"{prob:.1%}")
    st.markdown(f"### {colour} {band.upper()} risk")
    if result["prediction"]:
        st.error("**Recommendation: schedule a battery inspection.**")
    else:
        st.success("**Recommendation: no action needed.**")
    st.caption(
        f"Flag threshold {result['decision_threshold']:.2f} · fields not shown are imputed from "
        "training medians."
    )
with right:
    st.markdown("**What's driving this prediction** (SHAP contribution, log-odds)")
    factors = pd.DataFrame(result["top_factors"])
    factors["direction"] = factors["contribution"].map(lambda c: "raises risk" if c > 0 else "lowers risk")
    factors["value"] = factors["value"].map(lambda v: "missing (imputed)" if v is None else str(v))
    st.bar_chart(factors.set_index("feature")["contribution"].sort_values(), horizontal=True)
    st.dataframe(factors[["feature", "value", "contribution", "direction"]], hide_index=True, width="stretch")

with st.expander("Raw API request / response"):
    st.code(
        f"curl -X POST {API_URL}/explain -H 'Content-Type: application/json' -d '{payload}'", language="bash"
    )
    st.json(result)
