"""
FastAPI server — Micro-Grid Energy Arbitrator OpenEnv V2
Team RauResh — IIT Mandi

Updates:
- Unified endpoints (Removed duplicate /reset)
- Optional ResetRequest for grader compatibility
- Pydantic V2 model_dump integration
"""

import os
import uuid
import logging
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env.environment import MicroGridEnv
from env.models import ActionType, GridAction, LoadTier
from graders.graders import GRADER_REGISTRY, get_grader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MicroGridAPI")

app = FastAPI(
    title="Micro-Grid Energy Arbitrator OpenEnv",
    description="Team RauResh — IIT Mandi | Physics V2: Thermal Derating & Standby Loss",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# In-memory session storage
_sessions: Dict[str, MicroGridEnv] = {}

# --- Request Models ---

class ResetRequest(BaseModel):
    task_id: str = Field(default="task_easy", description="ID of the task to initialize")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    max_steps: Optional[int] = Field(default=24, description="Duration of episode in hours")

class StepRequest(BaseModel):
    session_id: str
    action_type: str = Field(..., description="buy_energy | sell_energy | store_energy | idle")
    quantity_kw: float = 0.0
    shed_tier: Optional[str] = Field(default=None, description="deferrable | essential | critical")

class GradeRequest(BaseModel):
    session_id: str

# --- Endpoints ---

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "microgrid-energy-arbitrator-v2",
        "version": "2.0.0",
        "team": "RauResh — IIT Mandi",
        "status": "ready",
        "tasks": list(GRADER_REGISTRY.keys()),
        "physics_engine": "V2 (Thermal + Standby)"
    }

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "sessions_active": len(_sessions)}

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    """Initializes a new simulation session."""
    if req.task_id not in GRADER_REGISTRY:
        raise HTTPException(400, f"Unknown task_id {req.task_id}. Valid: {list(GRADER_REGISTRY.keys())}")
    
    session_id = str(uuid.uuid4())[:8]
    env = MicroGridEnv(task_id=req.task_id, max_steps=req.max_steps or 24, seed=req.seed)
    obs = env.reset()
    
    _sessions[session_id] = env
    logger.info(f"New session created: {session_id} for task {req.task_id}")
    
    return {
        "session_id": session_id, 
        "observation": obs.model_dump() 
    }

@app.post("/step")
def step(req: StepRequest) -> Dict[str, Any]:
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, f"Session {req.session_id} not found.")

    try:
        atype = ActionType(req.action_type.lower())
    except ValueError:
        valid_actions = [a.value for a in ActionType]
        raise HTTPException(400, f"Invalid action_type. Must be one of: {valid_actions}")

    shed = None
    if req.shed_tier:
        try:
            shed = LoadTier(req.shed_tier.lower())
        except ValueError:
            valid_tiers = [t.value for t in LoadTier]
            raise HTTPException(400, f"Invalid shed_tier. Must be one of: {valid_tiers}")

    action = GridAction(action_type=atype, quantity_kw=req.quantity_kw, shed_tier=shed)
    
    try:
        result = env.step(action)
    except Exception as e:
        logger.error(f"Error in step(): {e}")
        raise HTTPException(500, "Internal simulator failure")

    return result.model_dump()

@app.get("/state")
def get_state(session_id: str = Query(...)) -> Dict[str, Any]:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, f"Session {session_id} not found.")
    return env.state()

@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    return {tid: g.describe() for tid, g in GRADER_REGISTRY.items()}

@app.post("/grade")
def grade(req: GradeRequest) -> Dict[str, Any]:
    """Grades the current session performance."""
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, f"Session {req.session_id} not found.")
    
    s = env.state()
    try:
        grader = get_grader(s.get("task_id", "task_easy"))
        result = grader.grade(s)
    except Exception as e:
        logger.error(f"Grading failed: {e}")
        raise HTTPException(500, f"Grading logic error: {e}")

    return {
        "session_id": req.session_id,
        "score": result.score,
        "passed": result.passed,
        "feedback": result.feedback,
        "breakdown": {
            "uptime": result.uptime_score,
            "economic": result.economic_score,
            "reserve": result.reserve_score,
            "safety": result.blackout_score
        }
    }