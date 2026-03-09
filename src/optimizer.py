"""CVaR optimizer and hedge frontier generator."""

import numpy as np
from scipy.optimize import minimize

import config
from src.options import all_instruments
from src.portfolio import cvar


def _hedged_returns(terminal_ret, instrument, hedge_ratio, spot):
    """Apply hedge payoff and cost to raw terminal returns."""
    terminal_prices = spot * (1.0 + terminal_ret)
    payoff = instrument["payoff"](terminal_prices) / spot
    cost = instrument["premium_pct"] * hedge_ratio
    return terminal_ret + hedge_ratio * payoff - cost


def optimize_instrument(terminal_ret, instrument, spot,
                        confidence=0.95, budget=config.HEDGE_BUDGET):
    """Find optimal hedge ratio for one instrument to minimize CVaR."""
    base_cvar = cvar(terminal_ret, confidence)
    prem_pct = instrument["premium_pct"]

    # max ratio we can afford
    if prem_pct > 0:
        max_ratio = min(budget / prem_pct, 1.0)
    else:
        # collar can be net credit — still cap at 50% to avoid
        # collapsing the entire return distribution
        max_ratio = 0.50

    if max_ratio <= 0:
        return {
            "instrument": instrument["name"],
            "hedge_ratio": 0.0,
            "cost_pct": 0.0,
            "unhedged_cvar": base_cvar,
            "hedged_cvar": base_cvar,
            "cvar_reduction_pct": 0.0,
        }

    def objective(x):
        hedged = _hedged_returns(terminal_ret, instrument, x[0], spot)
        return cvar(hedged, confidence)

    res = minimize(
        objective,
        x0=[max_ratio * 0.5],
        method="SLSQP",
        bounds=[(0.0, max_ratio)],
        constraints=[{"type": "ineq", "fun": lambda x: budget - x[0] * max(prem_pct, 0)}],
        options={"ftol": 1e-8},
    )

    opt_ratio = float(res.x[0])
    opt_cvar = float(res.fun)
    reduction = (base_cvar - opt_cvar) / base_cvar * 100 if base_cvar > 0 else 0.0

    return {
        "instrument": instrument["name"],
        "hedge_ratio": opt_ratio,
        "cost_pct": opt_ratio * prem_pct,
        "unhedged_cvar": base_cvar,
        "hedged_cvar": opt_cvar,
        "cvar_reduction_pct": reduction,
    }


def hedge_frontier(terminal_ret, instrument, spot,
                   confidence=0.95, n_points=config.FRONTIER_POINTS,
                   max_budget=0.05):
    """Sweep hedge ratio from 0 → max affordable, record cost vs CVaR."""
    base_cvar = cvar(terminal_ret, confidence)
    prem_pct = instrument["premium_pct"]

    if prem_pct > 0:
        max_ratio = min(max_budget / prem_pct, 1.0)
    else:
        max_ratio = 0.50

    ratios = np.linspace(0, max_ratio, n_points)
    points = []

    for ratio in ratios:
        hedged = _hedged_returns(terminal_ret, instrument, ratio, spot)
        c = cvar(hedged, confidence)
        cost = ratio * prem_pct
        reduction = (base_cvar - c) / base_cvar * 100 if base_cvar > 0 else 0.0
        points.append({
            "cost_pct": cost,
            "cvar": c,
            "cvar_reduction_pct": reduction,
            "hedge_ratio": float(ratio),
        })

    return {"instrument": instrument["name"], "points": points}


def optimize_all(terminal_ret, spot, sigma, confidence=0.95, budget=config.HEDGE_BUDGET):
    """Optimize all four instruments, return comparison list."""
    instruments = all_instruments(spot=spot, sigma=sigma)
    return [optimize_instrument(terminal_ret, inst, spot, confidence, budget)
            for inst in instruments]


def all_frontiers(terminal_ret, spot, sigma, confidence=0.95, n_points=config.FRONTIER_POINTS):
    """Compute hedge frontiers for all instruments."""
    instruments = all_instruments(spot=spot, sigma=sigma)
    return [hedge_frontier(terminal_ret, inst, spot, confidence, n_points)
            for inst in instruments]
