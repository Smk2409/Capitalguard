from __future__ import annotations

import datetime as dt
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_simulator import generate_synthetic_market, apply_shock, asset_names, sector_map
from optimizer import optimize_portfolio
from risk_engine import check_constraints, decide_action, DEFAULT_LIMITS

app = FastAPI(title="CapitalGuard AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- in-memory "book" state (a real deployment would back this with a DB) ---
MARKET = generate_synthetic_market()
STATE = {
    "weights": {t: round(1 / len(asset_names()), 4) for t in asset_names()},
    "audit_log": [],
}


def log_event(kind: str, payload: dict):
    STATE["audit_log"].append({
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "kind": kind,
        **payload,
    })
    # keep the log bounded for a demo session
    STATE["audit_log"] = STATE["audit_log"][-100:]


class OptimizeRequest(BaseModel):
    views: Optional[dict[str, float]] = None
    risk_aversion: float = 2.5
    max_weight: float = 0.35
    sector_caps: Optional[dict[str, float]] = None
    max_turnover: float = 0.25
    apply: bool = True  # if true, updates STATE["weights"] to the new allocation


class RiskCheckRequest(BaseModel):
    weights: Optional[dict[str, float]] = None  # defaults to current STATE weights


class ScenarioRequest(BaseModel):
    shock_type: str = "equity_selloff"
    magnitude: float = 0.10
    weights: Optional[dict[str, float]] = None


@app.get("/health")
def health():
    return {"status": "ok", "time": dt.datetime.utcnow().isoformat() + "Z"}


@app.get("/market/prices")
def market_prices(days: int = 120):
    tail = MARKET.tail(days)
    cum = (1 + tail).cumprod()
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in cum.index],
        "series": {col: [round(v, 4) for v in cum[col].tolist()] for col in cum.columns},
        "sectors": sector_map(),
    }


@app.get("/portfolio")
def portfolio():
    report = check_constraints(STATE["weights"], MARKET)
    action = decide_action(report)
    return {"weights": STATE["weights"], "risk": report, "action": action}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    result = optimize_portfolio(
        MARKET,
        current_weights=STATE["weights"],
        views=req.views,
        risk_aversion=req.risk_aversion,
        max_weight=req.max_weight,
        sector_caps=req.sector_caps,
        max_turnover=req.max_turnover,
    )
    # Gate the proposal through risk control before it's allowed to apply
    risk_report = check_constraints(result["weights"], MARKET)
    action = decide_action(risk_report)

    if req.apply and action["action"] in ("PROCEED", "AUTO_HEDGE"):
        STATE["weights"] = result["weights"]

    log_event("OPTIMIZE", {
        "views": req.views, "result": result, "risk_action": action,
        "applied": req.apply and action["action"] in ("PROCEED", "AUTO_HEDGE"),
    })

    return {"optimization": result, "risk_check": risk_report, "action": action}


@app.post("/risk-check")
def risk_check(req: RiskCheckRequest):
    weights = req.weights or STATE["weights"]
    report = check_constraints(weights, MARKET)
    action = decide_action(report)
    log_event("RISK_CHECK", {"weights": weights, "risk_action": action})
    return {"risk_check": report, "action": action}


@app.post("/scenario")
def scenario(req: ScenarioRequest):
    weights = req.weights or STATE["weights"]
    shocked_market = apply_shock(MARKET, req.shock_type, req.magnitude)
    report = check_constraints(weights, shocked_market, drawdown_lookback=5)
    action = decide_action(report)
    log_event("SCENARIO", {
        "shock_type": req.shock_type, "magnitude": req.magnitude,
        "weights": weights, "risk_action": action,
    })
    return {"shock_type": req.shock_type, "magnitude": req.magnitude,
            "risk_check": report, "action": action}


@app.get("/audit-log")
def audit_log():
    return {"events": list(reversed(STATE["audit_log"]))}


@app.get("/limits")
def limits():
    return DEFAULT_LIMITS
