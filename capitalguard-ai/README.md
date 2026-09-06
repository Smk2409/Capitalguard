# CapitalGuard AI

An automated capital-management and optimization engine for the **FinTech — Asset & Capital Management / Optimization Controls** hackathon track. It continuously proposes an optimal portfolio allocation and gates every decision through a real-time risk-control layer before it's allowed to execute.

Built to answer the three questions in the problem statement directly:

| Problem statement asks... | Where it lives |
|---|---|
| Optimization strategy for risk-adjusted returns under constraints | `backend/optimizer.py` |
| Detect / prevent / respond to risk breaches in real time | `backend/risk_engine.py` |
| Dashboard to visualize exposure, decisions, and run scenarios | `frontend/index.html` |

## Architecture

```
data_simulator.py  →  optimizer.py  →  risk_engine.py  →  main.py (FastAPI)  →  index.html (dashboard)
   (market data)      (proposes a       (validates the                            (visualizes state,
                        new allocation)   proposal, decides                        lets you optimize
                                          PROCEED / AUTO_HEDGE /                   and run scenarios)
                                          ESCALATE / FREEZE)
```

The optimizer and risk engine are **deliberately independent modules**. The optimizer's only job is to propose the best allocation given return assumptions and constraints; the risk engine's only job is to decide whether any given allocation (proposed or actual) is safe to hold. Every optimizer output is re-checked by the risk engine before `main.py` applies it — a bug or an extreme market input in the optimizer can never itself cause a risk-limit breach, because the control layer is the final gate, not a suggestion.

### 1. Optimization engine (`optimizer.py`)

- Starts from **market-implied equilibrium returns** (reverse-optimized from current weights — the standard Black-Litterman starting point), then blends in optional desk **views** (e.g. "we think EM equities return 16%") with a confidence weight to get posterior expected returns.
- Solves a convex **mean-variance QP** (via `cvxpy`) that maximizes expected return minus a risk penalty minus a turnover penalty, subject to:
  - fully invested (`sum(weights) == 1`)
  - per-asset bounds (no single line dominates the book)
  - per-sector caps (e.g. Equity ≤ 55%)
  - a liquidity floor (minimum cash / T-bill weight)
  - a turnover budget (limits how much of the book can trade in one cycle, so the optimizer doesn't churn the book for a marginal gain)
- Returns not just the new weights, but a **plain-language rationale** for every material change — this is what makes the decision explainable and audit-ready rather than a black box.

A convex QP was chosen deliberately over a black-box ML model: every output can be traced back to its inputs (views, constraints, risk aversion), which matters far more for a compliance-sensitive capital-allocation tool than a small accuracy gain from a fancier model.

### 2. Risk control & safeguard system (`risk_engine.py`)

Computes, on every check:
- **VaR / CVaR** (historical simulation, 95% confidence, 1-day horizon)
- **Trailing max drawdown** (configurable lookback — recent window, not the whole history, since a two-year-old drawdown shouldn't freeze today's trading)
- **Concentration** (single-asset and sector limits)
- **Liquidity buffer** (minimum cash allocation)

Breaches are collapsed into a single **action ladder** so downstream systems get one instruction instead of a wall of alerts:

| Severity | Action | Meaning |
|---|---|---|
| none | `PROCEED` | Allocation may execute as proposed |
| MEDIUM | `ESCALATE` | Flag for risk-officer review before the next rebalance |
| HIGH | `AUTO_HEDGE` | Automatically shift the flagged exposure toward cash/gov bonds within pre-approved limits |
| CRITICAL | `FREEZE` | Freeze all allocation changes and escalate immediately |

### 3. Decision dashboard (`frontend/index.html`)

A single-file dashboard (vanilla JS + Chart.js, no build step) that:
- Visualizes current allocation (donut chart) and sector exposure vs. caps (bar chart)
- Shows live VaR / CVaR / drawdown against their limits, and any active breaches
- Lets a risk manager **re-run the optimizer** with a custom view (e.g. "what if EM equities return 16%?") and see the resulting allocation, its rationale, and the risk engine's verdict on it
- Includes a **scenario simulator**: pick a shock (equity sell-off, rate spike, liquidity crunch, credit event) and magnitude, and see the resulting risk state and action instantly
- Shows a running **audit log** of every decision made this session, for compliance traceability

If the backend isn't running, the dashboard falls back to a static demo dataset so it's still viewable stand-alone.

## Running it

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (no server needed for the frontend — it's a static file that calls the API at `http://localhost:8000`).

### Try it from the command line too

```bash
# current portfolio + risk state
curl http://localhost:8000/portfolio

# re-optimize with a view that EM equities will return 16%
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"views": {"EM_EQ": 0.16}, "sector_caps": {"Equity": 0.55, "FixedIncome": 0.55}}'

# stress-test the current book against an equity sell-off
curl -X POST http://localhost:8000/scenario \
  -H "Content-Type: application/json" \
  -d '{"shock_type": "equity_selloff", "magnitude": 0.12}'
```

Or run the modules directly to see console output without the API at all:

```bash
python3 optimizer.py     # sample optimization run with a printed rationale
python3 risk_engine.py   # compliant book (OK) vs. a concentrated book under shock (AUTO_HEDGE)
```

## Design trade-offs

- **Synthetic data, real math.** `data_simulator.py` generates a plausible multi-asset return series (correlated equities, correlated bonds, a REIT partially correlated with both) so the optimizer and risk engine can be demoed without a paid data feed. Every downstream module consumes a plain `pandas.DataFrame` of returns, so swapping in a real market-data feed only touches this one file.
- **In-memory state.** `main.py` keeps the current book in a module-level dict rather than a database, since this is a hackathon-scope demo. The API surface (`/portfolio`, `/optimize`, `/risk-check`, `/scenario`, `/audit-log`) is exactly what a real deployment would expose — swapping in Postgres/Redis for `STATE` wouldn't change the routes.
- **Simplified Black-Litterman.** The full Black-Litterman model solves a linear system involving a view-uncertainty covariance matrix. This implementation blends per-asset views into the equilibrium return with a single confidence scalar — less rigorous, but transparent and fast enough for a real-time control loop, which matters more here than matching the academic model exactly.
- **Action ladder over a raw alert list.** Risk systems that surface every breach as an equal-weight alert train operators to ignore them. Collapsing to one recommended action (the most severe breach wins) keeps the system opinionated and actionable.
