"""
Inference Script — Micro-Grid Energy Arbitrator OpenEnv V2
========================================================
Team RauResh — IIT Mandi

Updates:
- System prompt tuned for Physics V2 (Thermal awareness)
- Fixed data mapping for net_cost_usd
- Enhanced history tracking for better decision context
"""

import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from env.environment import MicroGridEnv
from env.models import ActionType, GridAction, LoadTier
from graders.graders import get_grader

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

MAX_STEPS   = 24
TEMPERATURE = 0.1
MAX_TOKENS  = 200
SEED        = 42
TASK_IDS    = ["task_easy", "task_medium", "task_hard"]

#System prompt (Physics V2 Optimized)
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Energy Management System (EMS) for a micro-grid. 
Goal: Minimize net cost while preventing blackouts.

PHYSICS V2 CONSTRAINTS:
1. STANDBY LOSS: The system loses ~45W/hr regardless of activity. Factor this in!
2. THERMAL DERATING: In high heat or storms, battery efficiency drops.
3. BLACKOUT: If SoC hits 0%, the episode ends immediately with a massive penalty.

AVAILABLE ACTIONS:
- buy_energy:<kw>              (Charge battery using grid power)
- sell_energy:<kw>             (Discharge battery to grid for profit)
- store_energy:0               (Standard solar charging mode)
- idle:0                       (Minimal activity, standby loss still applies)
- buy_energy:<kw>:shed_tier    (Buy power while shedding 'deferrable' or 'essential' loads)

STRATEGY:
- BUY during off-peak/cheap hours.
- SELL during peak hours ONLY if SoC > 40%.
- SHED loads if SoC < 20% to avoid blackout risk.
- Keep kW values between 0.0 and 20.0.

Output ONLY the action string (e.g., buy_energy:5.5). No prose.
""").strip()

def build_prompt(obs: Dict[str, Any], step: int, history: List[str]) -> str:
    # Use .get() with defaults to prevent crashing on missing keys
    battery  = obs.get("battery", {})
    pricing  = obs.get("pricing", {})
    load     = obs.get("load", {})
    forecast = obs.get("forecast", {})
    solar    = obs.get("current_solar_output_kw", 0)

    history_str = "\n".join(f"  {h}" for h in history[-3:]) if history else "  (Genesis step)"

    return textwrap.dedent(f"""
    --- STEP {obs.get('hour', step)}/23 | TASK: {obs.get('task_id')} ---

    BATTERY STATUS:
    - SoC: {battery.get('state_of_charge_pct', 0):.1f}%
    - Health/Efficiency: {battery.get('health_factor', 1.0):.2f}x
    - Available Energy: {battery.get('energy_content_kwh', 0):.2f} kWh

    ENVIRONMENT:
    - Weather: {obs.get('weather', 'unknown')} | Irradiance: {obs.get('irradiance_wm2', 0):.0f} W/m²
    - Solar Output: {solar:.2f} kW
    - Buy Price: ${pricing.get('buy_price_per_kwh', 0):.4f} | Sell Price: ${pricing.get('sell_price_per_kwh', 0):.4f}

    DEMAND:
    - Total Load: {load.get('total_demand_kw', 0):.2f} kW
    - Breakdown: Crit({load.get('critical_kw', 0):.1f}) Ess({load.get('essential_kw', 0):.1f}) Def({load.get('deferrable_kw', 0):.1f})

    FORECAST (Next 3h Prices):
    - {forecast.get('price_forecast', [])[:3]}

    FINANCIALS/SAFETY:
    - Net Cost: ${obs.get('net_cost_usd', 0):.3f}
    - Blackouts: {obs.get('blackout_count', 0)}

    RECENT HISTORY:
    {history_str}

    DECISION:
    """).strip()

def parse_action(text: str) -> GridAction:
    raw = text.strip().lower().splitlines()[0].replace("action:", "").strip()
    try:
        parts = raw.split(":")
        atype = ActionType(parts[0])
        qty = float(parts[1]) if len(parts) > 1 else 0.0
        
        shed = None
        if len(parts) > 2:
            # Handle "shed_deferrable" or just "deferrable"
            tier_str = parts[2].replace("shed_", "")
            try:
                shed = LoadTier(tier_str)
            except: pass
            
        return GridAction(action_type=atype, quantity_kw=max(0.0, min(qty, 20.0)), shed_tier=shed)
    except:
        return GridAction(action_type=ActionType.IDLE, quantity_kw=0.0)

def run_episode(client: OpenAI, task_id: str, seed: int, verbose: bool = True) -> Dict[str, Any]:
    env = MicroGridEnv(task_id=task_id, max_steps=MAX_STEPS, seed=seed)
    obs = env.reset()
    obs_dict = obs.model_dump()

    history: List[str] = []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        prompt = build_prompt(obs_dict, step, history)
        
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw_text = resp.choices[0].message.content or "idle:0"
        except Exception as e:
            print(f"API Error: {e}")
            raw_text = "idle:0"

        action = parse_action(raw_text)
        result = env.step(action)
        
        obs_dict = result.observation.model_dump()
        total_reward += result.reward
        
        status = f"H{step:02d}: {action.action_type.value}({action.quantity_kw}kW) -> SoC {result.observation.battery.state_of_charge_pct:.1f}%"
        history.append(status)
        if verbose: print(status)

        if result.done: break

    # Final Grading
    state = env.state()
    grader = get_grader(task_id)
    grade = grader.grade(state)

    return {
        "task_id": task_id,
        "score": grade.score,
        "passed": grade.passed,
        "blackouts": state["blackout_count"],
        "net_cost": state["net_cost_usd"],
        "feedback": grade.feedback
    }

def main(task_filter: Optional[str] = None, seed: int = SEED):
    if not API_KEY:
        print("Set HF_TOKEN or API_KEY env var.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = [task_filter] if task_filter else TASK_IDS
    
    results = []
    for tid in tasks:
        print(f"\n--- Evaluating {tid} ---")
        res = run_episode(client, tid, seed)
        results.append(res)
        print(f"RESULT: Score={res['score']:.3f} | Pass={res['passed']}")

    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    main(task_filter=parser.parse_args().task, seed=parser.parse_args().seed)