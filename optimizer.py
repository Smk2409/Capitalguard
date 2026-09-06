
from __future__ import annotations

import numpy as np
import cvxpy as cp
import pandas as pd

from data_simulator import asset_names, sector_map


def _implied_equilibrium_returns(cov: np.ndarray, weights: np.ndarray, risk_aversion: float) -> np.ndarray:
    """Reverse-optimization step of Black-Litterman: the expected returns
    implied by the current (market) weights, given a risk-aversion level."""
    return risk_aversion * cov @ weights


def _blend_views(pi: np.ndarray, cov: np.ndarray, views: dict[str, float] | None,
                  tickers: list[str], tau: float = 0.05, view_confidence: float = 0.7) -> np.ndarray:
    """Simplified Black-Litterman posterior: blend equilibrium returns (pi)
    with discretionary views using a confidence weight. A full BL
    implementation solves a linear system with a view-uncertainty matrix;
    this simplified version blends per-asset for transparency/speed, which
    is sufficient for a real-time control loop."""
    if not views:
        return pi
    posterior = pi.copy()
    for ticker, view_return in views.items():
        if ticker in tickers:
            idx = tickers.index(ticker)
            posterior[idx] = (1 - view_confidence) * pi[idx] + view_confidence * view_return
    return posterior


def optimize_portfolio(
    returns: pd.DataFrame,
    current_weights: dict[str, float] | None = None,
    views: dict[str, float] | None = None,
    risk_aversion: float = 2.5,
    max_weight: float = 0.35,
    min_weight: float = 0.0,
    cash_floor: float = 0.03,
    sector_caps: dict[str, float] | None = None,
    max_turnover: float = 0.25,
    turnover_penalty: float = 0.02,
) -> dict:
    """Returns the recommended new allocation plus a rationale trail.

    All inputs are plain dict/DataFrame so this function can be called from
    the API layer, a notebook, or a batch job with the same signature.
    """
    tickers = asset_names()
    n = len(tickers)
    sectors = sector_map()
    cov = returns.cov().values * 252  # annualize
    mu_hist = returns.mean().values * 252

    if current_weights is None:
        current_weights = {t: 1 / n for t in tickers}
    w_current = np.array([current_weights.get(t, 0.0) for t in tickers])
    w_current = w_current / w_current.sum()

    # --- Black-Litterman-style expected returns ---
    pi = _implied_equilibrium_returns(cov, w_current, risk_aversion)
    mu = _blend_views(pi, cov, views, tickers)

    # --- Decision variable & objective ---
    w = cp.Variable(n)
    turnover = cp.norm1(w - w_current)
    ret = mu @ w
    risk = cp.quad_form(w, cp.psd_wrap(cov))
    objective = cp.Maximize(ret - risk_aversion * risk - turnover_penalty * turnover)

    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
        turnover <= max_turnover,
    ]
    cash_idx = tickers.index("CASH")
    constraints.append(w[cash_idx] >= cash_floor)

    if sector_caps:
        for sector, cap in sector_caps.items():
            idx = [i for i, t in enumerate(tickers) if sectors[t] == sector]
            if idx:
                constraints.append(cp.sum(w[idx]) <= cap)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    if w.value is None:
        raise RuntimeError("Optimizer failed to converge — constraints may be infeasible")

    new_weights = {t: float(max(0, round(v, 4))) for t, v in zip(tickers, w.value)}
    # renormalize tiny numerical drift
    total = sum(new_weights.values())
    new_weights = {t: round(v / total, 4) for t, v in new_weights.items()}

    delta = {t: round(new_weights[t] - current_weights.get(t, 0.0), 4) for t in tickers}
    moved = {t: d for t, d in delta.items() if abs(d) >= 0.005}

    expected_return = float(mu @ w.value)
    expected_vol = float(np.sqrt(w.value @ cov @ w.value))
    sharpe = (expected_return - 0.02) / expected_vol if expected_vol > 0 else 0.0

    rationale = _build_rationale(moved, views, sector_caps, sectors)

    return {
        "weights": new_weights,
        "previous_weights": current_weights,
        "changes": moved,
        "expected_return": round(expected_return, 4),
        "expected_volatility": round(expected_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "turnover": round(float(np.sum(np.abs(w.value - w_current))), 4),
        "rationale": rationale,
    }


def _build_rationale(moved: dict[str, float], views: dict | None, sector_caps: dict | None, sectors: dict) -> list[str]:
    lines = []
    for ticker, d in sorted(moved.items(), key=lambda kv: -abs(kv[1])):
        direction = "increased" if d > 0 else "decreased"
        reason = ""
        if views and ticker in views:
            reason = f" — reflects desk view of {views[ticker]*100:.1f}% expected return"
        lines.append(f"{ticker} {direction} by {abs(d)*100:.1f}%{reason}")
    if sector_caps:
        lines.append(f"Sector caps enforced: {', '.join(f'{k} \u2264 {v*100:.0f}%' for k, v in sector_caps.items())}")
    if not lines:
        lines.append("No material reallocation — current book already near-optimal under constraints")
    return lines


if __name__ == "__main__":
    from data_simulator import generate_synthetic_market
    rets = generate_synthetic_market()
    result = optimize_portfolio(
        rets,
        views={"EM_EQ": 0.16, "HY_BD": 0.02},
        sector_caps={"Equity": 0.55, "FixedIncome": 0.45},
    )
    import json
    print(json.dumps(result, indent=2))
