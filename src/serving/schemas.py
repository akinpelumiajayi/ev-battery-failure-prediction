"""Request / response contracts for the inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Percent = Field(default=None, ge=0, le=100)
Ratio = Field(default=None, ge=0, le=1)
NonNeg = Field(default=None, ge=0)


class VehicleFeatures(BaseModel):
    """
    Raw telemetry for one vehicle — mirrors the raw CSV schema (minus id/target).

    Every field is optional: real fleets have missing sensor readings, and the
    same imputer fitted at training time fills them at inference time.
    Bounds mirror the data contract in `src/data/validate.py`.
    """

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "vehicle_brand": "Tesla",
                "vehicle_model": "Model 3",
                "vehicle_type": "Sedan",
                "manufacturing_year": 2019,
                "battery_manufacturer": "Panasonic",
                "battery_chemistry": "NCA",
                "battery_capacity_kwh": 75.0,
                "drive_type": "AWD",
                "odometer_km": 182000,
                "vehicle_age_years": 6.1,
                "fleet_or_private": "Fleet",
                "cycle_count": 1650,
                "battery_health_percent": 68.0,
                "state_of_charge": 48.0,
                "state_of_health": 67.5,
                "cell_voltage_avg": 3.52,
                "pack_voltage": 305.0,
                "cell_temperature_avg": 38.0,
                "cell_temperature_max": 58.0,
                "internal_resistance": 0.72,
                "fast_charge_ratio": 0.7,
                "overcharge_events": 3,
                "BMS_warning_count": 4,
                "cooling_system_health": 78.0,
            }
        },
    )

    vehicle_brand: str | None = None
    vehicle_model: str | None = None
    vehicle_type: str | None = None
    manufacturing_year: float | None = Field(default=None, ge=1990, le=2030)
    battery_manufacturer: str | None = None
    battery_chemistry: str | None = None
    battery_capacity_kwh: float | None = Field(default=None, gt=0)
    drive_type: str | None = None
    odometer_km: float | None = NonNeg
    vehicle_age_years: float | None = NonNeg
    fleet_or_private: str | None = None
    cycle_count: float | None = NonNeg
    battery_health_percent: float | None = Percent
    state_of_charge: float | None = Percent
    depth_of_discharge: float | None = Percent
    state_of_health: float | None = Percent
    cell_voltage_avg: float | None = Field(default=None, gt=0, le=5)
    cell_voltage_std: float | None = NonNeg
    pack_voltage: float | None = Field(default=None, gt=0)
    cell_temperature_avg: float | None = None
    cell_temperature_max: float | None = None
    internal_resistance: float | None = NonNeg
    charge_efficiency: float | None = None
    discharge_efficiency: float | None = None
    remaining_capacity: float | None = None
    capacity_loss_percent: float | None = None
    charging_cycles_last_month: float | None = NonNeg
    fast_charge_ratio: float | None = Ratio
    slow_charge_ratio: float | None = Ratio
    average_charge_power_kw: float | None = NonNeg
    average_charging_time: float | None = NonNeg
    overnight_charging_ratio: float | None = Ratio
    home_charging_ratio: float | None = Ratio
    charging_interruptions: float | None = NonNeg
    overcharge_events: float | None = NonNeg
    average_speed: float | None = NonNeg
    average_trip_distance: float | None = NonNeg
    aggressive_acceleration_score: float | None = None
    hard_braking_score: float | None = None
    regenerative_braking_usage: float | None = None
    highway_driving_ratio: float | None = Ratio
    city_driving_ratio: float | None = Ratio
    daily_distance: float | None = NonNeg
    average_ambient_temperature: float | None = None
    maximum_temperature: float | None = None
    minimum_temperature: float | None = None
    humidity: float | None = Percent
    altitude: float | None = None
    terrain_type: str | None = None
    dust_exposure: float | None = None
    last_service_days: float | None = NonNeg
    cooling_system_health: float | None = None
    firmware_updates: float | None = NonNeg
    previous_faults: float | None = NonNeg
    maintenance_score: float | None = None
    thermal_runaway_risk: float | None = None
    voltage_imbalance: float | None = None
    temperature_variance: float | None = None
    sensor_fault_count: float | None = NonNeg
    BMS_warning_count: float | None = NonNeg
    abnormal_voltage_events: float | None = NonNeg
    battery_stress_index: float | None = None
    aging_score: float | None = None
    thermal_health_score: float | None = None
    charging_quality_score: float | None = None
    driving_stress_score: float | None = None
    predicted_remaining_life_cycles: float | None = None


RiskBand = Literal["low", "medium", "high"]


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = flag for inspection (predicted failure), 0 = healthy")
    failure_probability: float = Field(..., ge=0, le=1)
    risk_band: RiskBand = Field(..., description="low / medium / high, relative to the decision threshold")
    decision_threshold: float = Field(..., description="cost-optimised threshold used for `prediction`")
    model_version: str


class BatchRequest(BaseModel):
    vehicles: list[VehicleFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    n_flagged: int


class FeatureContribution(BaseModel):
    feature: str
    value: float | str | None = Field(..., description="raw input value (None = missing, imputed)")
    contribution: float = Field(..., description="SHAP value in log-odds; > 0 pushes towards failure")


class ExplanationResponse(PredictionResponse):
    base_value: float = Field(..., description="model output (log-odds) for an average vehicle")
    top_factors: list[FeatureContribution]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    decision_threshold: float
    risk_bands: dict[str, float]
    trained_at: str | None = None
    git_sha: str | None = None
    test_metrics: dict[str, float] = {}
    n_features: int
