---
title: Micro-Grid Energy Arbitrator OpenEnv V2
emoji: ⚡
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - energy
  - micro-grid
  - solar
  - physics-v2
  - iit-mandi
---

# Micro-Grid Energy Arbitrator — OpenEnv V2

**Team RauResh — IIT Mandi**

An OpenEnv-compliant environment where an AI agent manages energy flows across a solar PV array, lithium battery bank, and main electricity grid. 

**V2 Physics Engine Updates:**
- **Inverter Standby Loss**: Constant 45W draw (parasitic load).
- **Thermal Derating**: Battery discharge efficiency drops at high temperatures (>38°C).
- **Precision Sensing**: Modelled on ESP32 + ADS1115 ADC noise profiles.

---

## Why This Domain

Grid-scale energy management is one of the most consequential optimisation problems of the next decade. This environment forces an agent to reason about:
- **Physics V2 Constraints**: Understanding that even "idling" has a cost (standby loss).
- **Arbitrage**: Buying cheap at midnight, selling expensive at peak.
- **Safety**: Preventing blackouts (SoC = 0%) under noisy sensor data.

---

## Environment Description

One episode = 24 hours. One timestep = 1 hour.

| Action | Description |
|--------|-------------|
| `buy_energy:<kw>` | Purchase kW from grid → charges battery |
| `sell_energy:<kw>` | Export kW from battery → revenue |
| `store_energy:0` | Route solar to battery (Standard) |
| `idle:0` | Standby mode (45W loss still applies) |
| `action:shed_<tier>`| Strategic load-shedding (deferrable/essential/critical) |

---

## Observation Space (V2)

```json
{
  "episode_id": "a3f9bc12",
  "task_id": "task_easy",
  "hour": 9,
  "battery": {
    "state_of_charge_pct": 62.4,
    "energy_content_kwh": 31.2,
    "health_factor": 0.98,
    "temperature_c": 28.5
  },
  "current_solar_output_kw": 14.2,
  "pricing": {
    "buy_price_per_kwh": 0.1536,
    "sell_price_per_kwh": 0.0922,
    "is_peak_hour": true
  },
  "load": {
    "total_demand_kw": 11.6,
    "critical_kw": 1.0,
    "deferrable_kw": 5.8
  },
  "net_cost_usd": 0.42,
  "blackout_count": 0,
  "cumulative_reward": 1.24
}