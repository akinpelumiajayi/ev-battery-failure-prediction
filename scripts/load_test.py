"""
Tiny dependency-free traffic generator for the API (stdlib only).

Sends a mix of healthy / degraded vehicles so Grafana shows a realistic
risk-band split, and reports latency percentiles at the end.

    python scripts/load_test.py --url http://localhost:8000 --rps 10 --duration 60
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request


def vehicle(rng: random.Random) -> dict:
    wear = rng.random()  # 0 = new, 1 = worn out
    return {
        "vehicle_brand": rng.choice(["Tesla", "BYD", "Hyundai", "Kia", "Nissan", "Ford"]),
        "vehicle_type": rng.choice(["Sedan", "SUV", "Crossover", "Van"]),
        "battery_chemistry": rng.choice(["NMC", "LFP", "NCA"]),
        "manufacturing_year": int(2024 - 9 * wear),
        "vehicle_age_years": round(1 + 10 * wear, 2),
        "odometer_km": round(5_000 + 400_000 * wear * rng.uniform(0.6, 1.0)),
        "cycle_count": round(200 + 3000 * wear),
        "state_of_health": round(99 - 45 * wear * rng.uniform(0.7, 1.0), 1),
        "internal_resistance": round(0.15 + 0.9 * wear * rng.uniform(0.6, 1.0), 3),
        "cell_temperature_max": round(rng.uniform(25, 45) + 25 * wear, 1),
        "fast_charge_ratio": round(rng.random(), 2),
        "BMS_warning_count": int(7 * wear * rng.random()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rps", type=float, default=5)
    parser.add_argument("--duration", type=float, default=60, help="seconds")
    args = parser.parse_args()

    rng = random.Random(42)
    latencies, bands = [], {"low": 0, "medium": 0, "high": 0}
    errors = 0
    end = time.time() + args.duration
    while time.time() < end:
        body = json.dumps(vehicle(rng)).encode()
        req = urllib.request.Request(
            f"{args.url}/predict", data=body, headers={"Content-Type": "application/json"}
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                bands[json.loads(resp.read())["risk_band"]] += 1
            latencies.append(time.perf_counter() - start)
        except OSError:
            errors += 1
        time.sleep(max(0.0, 1 / args.rps - (time.perf_counter() - start)))

    if latencies:
        qs = statistics.quantiles(latencies, n=100)
        print(f"requests={len(latencies)} errors={errors} bands={bands}")
        print(f"latency p50={qs[49] * 1000:.1f}ms p95={qs[94] * 1000:.1f}ms p99={qs[98] * 1000:.1f}ms")
    else:
        print(f"no successful requests (errors={errors})")


if __name__ == "__main__":
    main()
