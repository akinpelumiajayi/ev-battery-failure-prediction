# Dataset card — EV battery telemetry

| | |
|---|---|
| **File** | `data/raw/ev_battery_data.csv` (7.6 MB; DVC pointer `data/raw/ev_battery_data.csv.dvc`, md5 `5b638704…ffea`) |
| **Public mirror** | [GitHub release `data-v1`](https://github.com/akinpelumiajayi/ev-battery-failure-prediction/releases/tag/data-v1). Fetched and md5-verified by `python -m src.data.fetch_data` |
| **Rows / columns** | 20,000 vehicles × 70 columns (1 ID, 1 serial, 67 features, 1 target) |
| **Target** | `battery_failure` (0/1), positive rate **6.92 %** |
| **Missingness** | ~3.8 % of cells, spread across most sensor columns |
| **Source** | [EV Battery Health Prediction Dataset (20K)](https://www.kaggle.com/datasets/srisyra02/ev-battery-health-prediction-dataset-20k) by Srimathy Sivanessan on Kaggle (version 1) |
| **Licence** | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The GitHub release mirror redistributes the file unmodified under the same licence, with attribution to the original author. |

## Feature groups

| Group | Examples | Count |
|---|---|---:|
| Vehicle & pack | brand, model, type, manufacturing year, battery manufacturer, chemistry, capacity (kWh), drive type, odometer, age, fleet vs. private | 11 |
| Battery state (BMS) | cycle count, state of health / charge, depth of discharge, cell voltage mean/std, pack voltage, cell temperature avg/max, internal resistance, charge/discharge efficiency, remaining capacity, capacity loss | 15 |
| Charging behaviour | fast/slow/overnight/home charging ratios, charge power, charging time, interruptions, overcharge events, cycles last month | 9 |
| Driving behaviour | average speed, trip distance, aggressive acceleration, hard braking, regen usage, highway/city ratio, daily distance | 8 |
| Environment | ambient / max / min temperature, humidity, altitude, terrain, dust exposure | 7 |
| Maintenance & faults | days since service, cooling-system health, firmware updates, previous faults, sensor faults, BMS warnings, abnormal voltage events, voltage imbalance, temperature variance | 9 |
| Derived scores | thermal-runaway risk, battery stress index, aging score, thermal health, charging quality, driving stress, maintenance score, predicted remaining life cycles | 8 |

## Data contract

Enforced by [`src/data/validate.py`](../src/data/validate.py) before anything else runs:
unique `vehicle_id`; binary target; percentages in [0, 100]; ratios in [0, 1]; cell voltage in (0, 5] V;
positive pack voltage; manufacturing year 1990–2030; at least 1,000 rows; positive rate in (1 %, 50 %);
no fully empty column.

## Observations: synthetic origin

The Kaggle description confirms the records are **simulated** ("20,000 simulated electric vehicle battery
telemetry records"). The data itself shows this too, which is a useful sanity check on the audit:

* `vehicle_type` is spread uniformly across every brand (Audi, BMW and Ford each have ~150 Trucks, Vans,
  Sedans, …), and model/type pairs are often implausible (a *Mustang Mach-E* labelled *Truck*).
* The failure rate is nearly identical across chemistries (6.0–7.2 %), although real LFP, NMC and LTO
  packs age very differently.
* The best single features (thermal-runaway risk, state of health, internal resistance) each separate
  the classes cleanly (AUC ≈ 0.81–0.85), and a linear model beats gradient boosting. Both are typical of
  a generator built on additive logistic rules.

The *pipeline* (contracts, leakage checks, threshold optimisation, drift monitoring) is built to be
data-agnostic. The *metrics* should not be read as real-world battery failure-prediction performance.

## Splits

Stratified 70 / 15 / 15 train / validation / test (`random_state=42`). The training fold is also
snapshotted to `data/reference/reference.parquet` as the drift-monitoring baseline.
