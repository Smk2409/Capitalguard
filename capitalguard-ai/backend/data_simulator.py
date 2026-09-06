"""
data_simulator.py
------------------
Generates synthetic multi-asset market data so the optimization and risk
engines can be demoed / tested without a paid market-data feed.

In production this module would be swapped for a real ingestion layer
(e.g. a Kafka consumer or REST poller hitting a market-data vendor),
but it exposes the same shape of output (a returns DataFrame + metadata)
so nothing downstream needs to change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASSET_UNIVERSE = [
    {"ticker": "US_EQ",  "name": "US Equities",        "sector": "Equity",     "mu": 0.09, "vol": 0.16},
    {"ticker": "INTL_EQ","name": "International Equities","sector": "Equity",  "mu": 0.07, "vol": 0.18},
    {"ticker": "EM_EQ",  "name": "Emerging Mkt Equities","sector": "Equity",   "mu": 0.10, "vol": 0.24},
    {"ticker": "GOV_BD", "name": "Government Bonds",    "sector": "FixedIncome","mu": 0.03,"vol": 0.05},
    {"ticker": "CORP_BD","name": "Corporate Bonds",     "sector": "FixedIncome","mu": 0.045,"vol": 0.08},
    {"ticker": "HY_BD",  "name": "High-Yield Bonds",    "sector": "FixedIncome","mu": 0.06,"vol": 0.11},
    {"ticker": "REIT",   "name": "Real Estate (REITs)", "sector": "RealAsset", "mu": 0.075,"vol": 0.19},
    {"ticker": "CASH",   "name": "Cash / T-Bills",      "sector": "Cash",      "mu": 0.02, "vol": 0.005},
]

def asset_names() -> list[str]:
    return [a["ticker"] for a in ASSET_UNIVERSE]

def sector_map() -> dict[str, str]:
    return {a["ticker"]: a["sector"] for a in ASSET_UNIVERSE}


def generate_synthetic_market(n_days: int = 504, seed: int = 42) -> pd.DataFrame:
    """Daily log-returns for the asset universe, with a plausible correlation
    structure (equities correlated with each other, bonds correlated with each
    other, REIT partially correlated with both, cash near-zero-vol)."""
    rng = np.random.default_rng(seed)
    n = len(ASSET_UNIVERSE)
    mu_daily = np.array([a["mu"] for a in ASSET_UNIVERSE]) / 252
    vol_daily = np.array([a["vol"] for a in ASSET_UNIVERSE]) / np.sqrt(252)

    # Build a block-ish correlation matrix
    corr = np.eye(n)
    equity_idx = [0, 1, 2]
    bond_idx = [3, 4, 5]
    for i in equity_idx:
        for j in equity_idx:
            if i != j:
                corr[i, j] = 0.7
    for i in bond_idx:
        for j in bond_idx:
            if i != j:
                corr[i, j] = 0.6
    corr[6, equity_idx] = corr[equity_idx, 6] = 0.4   # REIT vs equity
    corr[6, bond_idx] = corr[bond_idx, 6] = 0.2        # REIT vs bonds
    corr[7, :] = corr[:, 7] = 0.0
    np.fill_diagonal(corr, 1.0)

    cov_daily = np.outer(vol_daily, vol_daily) * corr
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    returns = rng.multivariate_normal(mu_daily, cov_daily, size=len(dates))
    return pd.DataFrame(returns, index=dates, columns=asset_names())


def apply_shock(returns: pd.DataFrame, shock_type: str, magnitude: float = 0.08) -> pd.DataFrame:
    """Append a synthetic shock day to a returns series for scenario testing.
    shock_type in {"equity_selloff", "rate_spike", "liquidity_crunch", "credit_event"}.
    """
    shocked = returns.copy()
    shock_row = pd.Series(0.0, index=returns.columns)

    if shock_type == "equity_selloff":
        shock_row[["US_EQ", "INTL_EQ", "EM_EQ"]] = -magnitude
        shock_row[["REIT"]] = -magnitude * 0.6
        shock_row[["GOV_BD"]] = magnitude * 0.3       # flight to quality
        shock_row[["CASH"]] = 0.0
    elif shock_type == "rate_spike":
        shock_row[["GOV_BD", "CORP_BD"]] = -magnitude * 0.8
        shock_row[["HY_BD"]] = -magnitude * 0.5
        shock_row[["REIT"]] = -magnitude * 0.4
        shock_row[["US_EQ", "INTL_EQ"]] = -magnitude * 0.2
    elif shock_type == "liquidity_crunch":
        shock_row[["HY_BD", "EM_EQ", "REIT"]] = -magnitude
        shock_row[["US_EQ", "INTL_EQ", "CORP_BD"]] = -magnitude * 0.5
        shock_row[["CASH", "GOV_BD"]] = magnitude * 0.1
    elif shock_type == "credit_event":
        shock_row[["HY_BD", "CORP_BD"]] = -magnitude * 1.1
        shock_row[["EM_EQ"]] = -magnitude * 0.6
        shock_row[["GOV_BD"]] = magnitude * 0.25
    else:
        raise ValueError(f"Unknown shock_type: {shock_type}")

    shock_date = returns.index[-1] + pd.tseries.offsets.BDay(1)
    shocked.loc[shock_date] = shock_row
    return shocked
