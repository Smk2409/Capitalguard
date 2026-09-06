
from __future__ import annotations

import numpy as np
import pandas as pd

from data_simulator import sector_map

DEFAULT_LIMITS = {
    "var_95_limit": 0.025,       # max acceptable 1-day 95% VaR, as a fraction of book value
    "cvar_95_limit": 0.04,       # max acceptable 1-day 95% CVaR
    "max_drawdown_limit": 0.15,  # max acceptable trailing drawdown
    "single_asset_limit": 0.35,  # max weight in any one asset
    "sector_limits": {"Equity": 0.55, "FixedIncome": 0.55, "RealAsset": 0.25},
    "liquidity_floor": 0.03,     # min cash / T-bill weight
}


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    tickers = list(returns.columns)
    w = np.array([weights.get(t, 0.0) for t in tickers])
    return returns[tickers].values @ w


def calc_var_cvar(port_returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Historical-simulation VaR/CVaR (1-day, positive numbers = loss)."""
    losses = -port_returns
    var = float(np.percentile(losses, confidence * 100))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var
    return round(var, 4), round(cvar, 4)


def calc_max_drawdown(port_returns: np.ndarray) -> float:
    cum = np.cumprod(1 + port_returns)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    return round(float(drawdown.min()), 4)


def check_constraints(weights: dict[str, float], returns: pd.DataFrame,
                       limits: dict = None, drawdown_lookback: int = 60) -> dict:
    """Runs every check and returns a structured breach report.

    VaR/CVaR use the full history (they need enough tail observations to be
    statistically meaningful); trailing drawdown intentionally uses a
    shorter recent window (`drawdown_lookback` trading days) since that is
    what a risk officer actually means by "how far off its recent peak is
    the book right now" — a two-year-old drawdown shouldn't freeze today's
    trading.
    """
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    sectors = sector_map()

    port_ret = portfolio_returns(returns, weights)
    var95, cvar95 = calc_var_cvar(port_ret, 0.95)
    max_dd = calc_max_drawdown(port_ret[-drawdown_lookback:])

    breaches = []

    if var95 > limits["var_95_limit"]:
        breaches.append({
            "type": "VAR_BREACH", "severity": "HIGH",
            "detail": f"1-day 95% VaR {var95*100:.2f}% exceeds limit {limits['var_95_limit']*100:.2f}%",
        })
    if cvar95 > limits["cvar_95_limit"]:
        breaches.append({
            "type": "CVAR_BREACH", "severity": "HIGH",
            "detail": f"1-day 95% CVaR {cvar95*100:.2f}% exceeds limit {limits['cvar_95_limit']*100:.2f}%",
        })
    if max_dd < -limits["max_drawdown_limit"]:
        breaches.append({
            "type": "DRAWDOWN_BREACH", "severity": "CRITICAL",
            "detail": f"Trailing drawdown {max_dd*100:.2f}% exceeds limit {-limits['max_drawdown_limit']*100:.2f}%",
        })

    for ticker, w in weights.items():
        if w > limits["single_asset_limit"]:
            breaches.append({
                "type": "CONCENTRATION_BREACH", "severity": "MEDIUM",
                "detail": f"{ticker} weight {w*100:.1f}% exceeds single-asset limit {limits['single_asset_limit']*100:.0f}%",
            })

    sector_totals: dict[str, float] = {}
    for ticker, w in weights.items():
        sec = sectors.get(ticker, "Other")
        sector_totals[sec] = sector_totals.get(sec, 0.0) + w
    for sec, cap in limits["sector_limits"].items():
        if sector_totals.get(sec, 0.0) > cap:
            breaches.append({
                "type": "SECTOR_CONCENTRATION_BREACH", "severity": "MEDIUM",
                "detail": f"{sec} allocation {sector_totals.get(sec,0)*100:.1f}% exceeds sector cap {cap*100:.0f}%",
            })

    cash_weight = weights.get("CASH", 0.0)
    if cash_weight < limits["liquidity_floor"]:
        breaches.append({
            "type": "LIQUIDITY_BREACH", "severity": "HIGH",
            "detail": f"Cash buffer {cash_weight*100:.1f}% below liquidity floor {limits['liquidity_floor']*100:.1f}%",
        })

    return {
        "metrics": {
            "var_95": var95,
            "cvar_95": cvar95,
            "max_drawdown": max_dd,
            "sector_exposure": {k: round(v, 4) for k, v in sector_totals.items()},
        },
        "breaches": breaches,
        "status": "OK" if not breaches else ("CRITICAL" if any(b["severity"] == "CRITICAL" for b in breaches) else "BREACH"),
    }


ACTION_LADDER = {
    "CRITICAL": ("FREEZE", "Freeze all new allocation changes and escalate to Risk Officer immediately."),
    "HIGH": ("AUTO_HEDGE", "Auto-hedge: shift the flagged exposure toward cash/gov bonds within pre-approved limits."),
    "MEDIUM": ("ESCALATE", "Flag for Risk Officer review before the next rebalance executes."),
}


def decide_action(risk_report: dict) -> dict:
    """Collapses a breach report into a single recommended action —
    the most severe breach present determines the response."""
    if risk_report["status"] == "OK":
        return {"action": "PROCEED", "message": "All risk checks passed — allocation may execute as proposed.",
                "triggered_by": []}

    severities_present = {b["severity"] for b in risk_report["breaches"]}
    for sev in ("CRITICAL", "HIGH", "MEDIUM"):
        if sev in severities_present:
            action, message = ACTION_LADDER[sev]
            triggers = [b["detail"] for b in risk_report["breaches"] if b["severity"] == sev]
            return {"action": action, "message": message, "triggered_by": triggers}
    return {"action": "PROCEED", "message": "No actionable breach found.", "triggered_by": []}


if __name__ == "__main__":
    from data_simulator import generate_synthetic_market, apply_shock
    import json

    rets = generate_synthetic_market()
    weights = {"US_EQ": 0.22, "INTL_EQ": 0.12, "EM_EQ": 0.10, "GOV_BD": 0.15,
               "CORP_BD": 0.15, "HY_BD": 0.08, "REIT": 0.08, "CASH": 0.10}

    print("=== Normal market (compliant book) ===")
    report = check_constraints(weights, rets)
    print(json.dumps(report, indent=2))
    print(json.dumps(decide_action(report), indent=2))

    print("\n=== Concentrated, low-liquidity book under equity sell-off shock ===")
    risky_weights = {"US_EQ": 0.30, "INTL_EQ": 0.15, "EM_EQ": 0.18, "GOV_BD": 0.08,
                      "CORP_BD": 0.08, "HY_BD": 0.10, "REIT": 0.09, "CASH": 0.02}
    shocked = apply_shock(rets, "equity_selloff", magnitude=0.12)
    report2 = check_constraints(risky_weights, shocked, drawdown_lookback=5)
    print(json.dumps(report2, indent=2))
    print(json.dumps(decide_action(report2), indent=2))
