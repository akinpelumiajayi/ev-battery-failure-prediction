from typing import Optional
from pydantic import BaseModel, Field


class VehicleFeatures(BaseModel):
    """
    Raw feature payload — mirrors the raw CSV schema (minus id/target).
    All fields optional to tolerate real-world missing sensor readings;
    the same imputer used at training time handles them at inference time.
    """
    vehicle_brand: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_type: Optional[str] = None
    manufacturing_year: Optional[float] = None
    battery_manufacturer: Optional[str] = None
    battery_chemistry: Optional[str] = None
    battery_capacity_kwh: Optional[float] = None
    drive_type: Optional[str] = None
    odometer_km: Optional[float] = None
    vehicle_age_years: Optional[float] = None
    fleet_or_private: Optional[str] = None
    cycle_count: Optional[float] = None
    battery_health_percent: Optional[float] = None
    state_of_charge: Optional[float] = None
    depth_of_discharge: Optional[float] = None
    state_of_health: Optional[float] = None
    cell_voltage_avg: Optional[float] = None
    cell_voltage_std: Optional[float] = None
    pack_voltage: Optional[float] = None
    cell_temperature_avg: Optional[float] = None
    cell_temperature_max: Optional[float] = None
    internal_resistance: Optional[float] = None
    charge_efficiency: Optional[float] = None
    discharge_efficiency: Optional[float] = None
    remaining_capacity: Optional[float] = None
    capacity_loss_percent: Optional[float] = None
    charging_cycles_last_month: Optional[float] = None
    fast_charge_ratio: Optional[float] = None
    slow_charge_ratio: Optional[float] = None
    average_charge_power_kw: Optional[float] = None
    average_charging_time: Optional[float] = None
    overnight_charging_ratio: Optional[float] = None
    home_charging_ratio: Optional[float] = None
    charging_interruptions: Optional[float] = None
    overcharge_events: Optional[float] = None
    average_speed: Optional[float] = None
    average_trip_distance: Optional[float] = None
    aggressive_acceleration_score: Optional[float] = None
    hard_braking_score: Optional[float] = None
    regenerative_braking_usage: Optional[float] = None
    highway_driving_ratio: Optional[float] = None
    city_driving_ratio: Optional[float] = None
    daily_distance: Optional[float] = None
    average_ambient_temperature: Optional[float] = None
    maximum_temperature: Optional[float] = None
    minimum_temperature: Optional[float] = None
    humidity: Optional[float] = None
    altitude: Optional[float] = None
    terrain_type: Optional[str] = None
    dust_exposure: Optional[float] = None
    last_service_days: Optional[float] = None
    cooling_system_health: Optional[float] = None
    firmware_updates: Optional[float] = None
    previous_faults: Optional[float] = None
    maintenance_score: Optional[float] = None
    thermal_runaway_risk: Optional[float] = None
    voltage_imbalance: Optional[float] = None
    temperature_variance: Optional[float] = None
    sensor_fault_count: Optional[float] = None
    BMS_warning_count: Optional[float] = None
    abnormal_voltage_events: Optional[float] = None
    battery_stress_index: Optional[float] = None
    aging_score: Optional[float] = None
    thermal_health_score: Optional[float] = None
    charging_quality_score: Optional[float] = None
    driving_stress_score: Optional[float] = None
    predicted_remaining_life_cycles: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_brand": "Tesla", "vehicle_model": "Model 3", "vehicle_type": "Sedan",
                "manufacturing_year": 2021, "battery_manufacturer": "CATL",
                "battery_chemistry": "NMC", "battery_capacity_kwh": 75.0, "drive_type": "AWD",
                "odometer_km": 45000, "vehicle_age_years": 3.2, "fleet_or_private": "Private",
                "cycle_count": 650, "battery_health_percent": 88.5, "state_of_charge": 62.0,
                "state_of_health": 87.9, "cell_voltage_avg": 3.7, "pack_voltage": 370.5,
                "cell_temperature_avg": 25.0, "cell_temperature_max": 32.0,
                "internal_resistance": 0.45,
            }
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = predicted failure, 0 = healthy")
    failure_probability: float = Field(..., description="Model's predicted probability of failure")
    risk_band: str = Field(..., description="low / medium / high, derived from probability")
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
