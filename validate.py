#!/usr/bin/env python3
"""
Pre-submission validation script — Micro-Grid Energy Arbitrator V2
Team RauResh — IIT Mandi

Updates: 
- Synchronized with Physics V2 (Standby Loss & Thermal Derating)
- Validates float-based Battery cycles
- Matches new env.state() key schema
"""

import argparse
import subprocess
import sys
import traceback
import os
from typing import List, Tuple

# Terminal Colors
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   return f"{GREEN}  ✓ PASS{RESET}  {msg}"
def fail(msg): return f"{RED}  ✗ FAIL{RESET}  {msg}"
def warn(msg): return f"{YELLOW}  ⚠ WARN{RESET}  {msg}"

Results = List[Tuple[str, str]]

def check_files(results: Results) -> None:
    required = [
        ("inference.py",       "Mandatory inference script"),
        ("openenv.yaml",       "OpenEnv spec"),
        ("Dockerfile",         "Container definition"),
        ("requirements.txt",   "Python dependencies"),
        ("README.md",          "Documentation"),
        ("app.py",             "FastAPI server"),
        ("env/models.py",      "Typed models"),
        ("env/physics.py",     "Physics simulator"),
        ("env/environment.py", "Core environment"),
        ("env/reward.py",      "Reward function"),
        ("graders/graders.py", "Task graders"),
        ("tests/test_env.py",  "Unit tests"),
    ]
    for path, desc in required:
        if os.path.exists(path):
            results.append(("pass", ok(f"file: {path} ({desc})")))
        else:
            results.append(("fail", fail(f"file: {path} MISSING ({desc})")))

def check_yaml(results: Results) -> None:
    import yaml
    try:
        with open("openenv.yaml") as f:
            spec = yaml.safe_load(f)
        for field in ["name", "version", "observation_space", "action_space", "tasks"]:
            if field in spec:
                results.append(("pass", ok(f"yaml: field '{field}' present")))
            else:
                results.append(("fail", fail(f"yaml: missing field '{field}'")))

        tasks = spec.get("tasks", [])
        if len(tasks) >= 3:
            results.append(("pass", ok(f"yaml: {len(tasks)} tasks defined")))
        else:
            results.append(("fail", fail(f"yaml: {len(tasks)} task(s) found (need 3)")))
            
    except Exception as e:
        results.append(("fail", fail(f"yaml: parse error — {e}")))

def check_imports(results: Results) -> None:
    mods = [
        ("env.environment", "MicroGridEnv"),
        ("env.models",      "GridAction, GridObservation, EpisodeResult, ActionType, BatteryState"),
        ("env.physics",     "BlackoutError, apply_charge, apply_discharge, attenuate_for_weather"),
        ("env.reward",      "compute_reward"),
        ("graders.graders", "get_grader, GRADER_REGISTRY"),
    ]
    for mod, symbols in mods:
        try:
            m = __import__(mod, fromlist=symbols.split(", "))
            for sym in [s.strip() for s in symbols.split(",")]:
                getattr(m, sym)
            results.append(("pass", ok(f"import: {mod}")))
        except Exception as e:
            results.append(("fail", fail(f"import: {mod} — {e}")))

def check_interface(results: Results) -> None:
    from env.environment import MicroGridEnv
    from env.models import ActionType, EpisodeResult, GridAction, GridObservation

    for task_id in ["task_easy", "task_medium", "task_hard"]:
        try:
            env = MicroGridEnv(task_id=task_id, seed=42)
            obs = env.reset()

            # 1. Check Observation Structure
            assert isinstance(obs, GridObservation), "reset() must return GridObservation"
            assert hasattr(obs, 'hour'), "Observation missing 'hour' attribute"
            assert obs.hour == 0, f"Expected hour 0, got {obs.hour}"
            results.append(("pass", ok(f"interface [{task_id}]: reset() returns valid V2 Observation")))

            # 2. Check State Dictionary Keys (Matches env.py V2)
            s = env.state()
            required_keys = [
                "episode_id", "net_cost_usd", "battery_soc_pct", 
                "blackout_count", "cumulative_reward"
            ]
            for key in required_keys:
                assert key in s, f"State missing key: {key}"
            results.append(("pass", ok(f"interface [{task_id}]: state() matches V2 schema")))

            # 3. Check Step Execution
            action = GridAction(action_type=ActionType.IDLE, quantity_kw=0.0)
            res = env.step(action)
            assert isinstance(res, EpisodeResult), "step() must return EpisodeResult"
            assert isinstance(res.reward, float), "Reward must be a float"
            results.append(("pass", ok(f"interface [{task_id}]: step() handles V2 physics cycle")))

        except Exception as e:
            results.append(("fail", fail(f"interface [{task_id}]: {str(e)}")))

def check_graders(results: Results) -> None:
    from graders.graders import get_grader, GRADER_REGISTRY
    
    # Check score normalization [0, 1]
    test_state = {
        "blackout_count": 0, 
        "net_cost_usd": 1.5, 
        "battery_soc_pct": 85.0, 
        "cumulative_reward": 10.0,
        "hour": 24
    }
    
    for tid in ["task_easy", "task_medium", "task_hard"]:
        try:
            grader = get_grader(tid)
            score = grader.grade(test_state).score
            if 0.0 <= score <= 1.0:
                results.append(("pass", ok(f"grader [{tid}]: score {score:.2f} is normalized")))
            else:
                results.append(("fail", fail(f"grader [{tid}]: score {score} out of [0,1] range")))
        except Exception as e:
            results.append(("fail", fail(f"grader [{tid}]: {e}")))

def check_dockerfile(results: Results) -> None:
    if not os.path.exists("Dockerfile"):
        results.append(("fail", fail("Dockerfile missing")))
        return
    
    with open("Dockerfile") as f:
        content = f.read()
    
    checks = [
        ("7860", "Exposes port 7860"),
        ("requirements.txt", "Installs dependencies"),
        ("python", "Uses Python base image")
    ]
    for pattern, desc in checks:
        if pattern in content:
            results.append(("pass", ok(f"Dockerfile: {desc}")))
        else:
            results.append(("warn", warn(f"Dockerfile: maybe missing {desc}?")))

def main(strict: bool = False) -> None:
    sys.path.insert(0, ".")
    print(f"\n{BOLD}{'═'*70}")
    print("  Micro-Grid Energy Arbitrator — Pre-Submission Validation V2")
    print(f"  Team RauResh — IIT Mandi")
    print(f"{'═'*70}{RESET}\n")

    results: Results = []
    checks = [
        ("Filesystem", check_files),
        ("YAML Spec", check_yaml),
        ("Imports", check_imports),
        ("Environment Interface", check_interface),
        ("Graders", check_graders),
        ("Docker", check_dockerfile),
    ]

    for name, func in checks:
        print(f"{BOLD}── {name}{RESET}")
        before = len(results)
        try:
            func(results)
        except Exception as e:
            results.append(("fail", fail(f"{name} critical error: {e}")))
        
        for _, msg in results[before:]:
            print(msg)
        print()

    failures = sum(1 for s, _ in results if s == "fail")
    warnings = sum(1 for s, _ in results if s == "warn")

    print(f"{BOLD}{'═'*70}")
    print(f"  SUMMARY: {failures} Failures | {warnings} Warnings")
    print(f"{'═'*70}{RESET}\n")

    if failures == 0:
        if warnings > 0 and strict:
            print(f"{RED}{BOLD}  ✗ BLOCKED BY WARNINGS (STRICT MODE){RESET}\n")
            sys.exit(1)
        print(f"{GREEN}{BOLD}  ✓ SYSTEM READY FOR SUBMISSION{RESET}\n")
        sys.exit(0)
    else:
        print(f"{RED}{BOLD}  ✗ VALIDATION FAILED — FIX FAILURES BEFORE SUBMITTING{RESET}\n")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    main(strict=parser.parse_args().strict)