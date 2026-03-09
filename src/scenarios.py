"""Stress scenarios — historical replays + forward-looking hypothetical shocks."""

import numpy as np
import pandas as pd

import config
from src.portfolio import cvar, var, max_drawdown


def run_historical(returns, weights, name, start, end):
    """Portfolio performance during a historical stress window."""
    w = np.asarray(weights)
    mask = (returns.index >= start) & (returns.index <= end)
    sub = returns.loc[mask]

    if len(sub) < 2:
        return None

    port_ret = sub.values @ w
    cum_ret = float(np.prod(1 + port_ret) - 1)

    return {
        "name": name,
        "period": f"{start} to {end}",
        "return": cum_ret,
        "max_drawdown": max_drawdown(port_ret),
        "var_95": var(port_ret, 0.95),
        "cvar_95": cvar(port_ret, 0.95),
        "worst_day": float(np.min(port_ret)),
        "best_day": float(np.max(port_ret)),
        "n_days": len(port_ret),
        "hypothetical": False,
    }


def run_all_historical(returns, weights, scenarios=None):
    """Run all configured historical scenarios."""
    scenarios = scenarios or config.STRESS_SCENARIOS
    results = []
    for name, dates in scenarios.items():
        r = run_historical(returns, weights, name, dates["start"], dates["end"])
        if r is not None:
            results.append(r)
    return results


def run_hypothetical(name, tickers, weights, custom_shocks=None):
    """Apply a hypothetical shock to current portfolio."""
    w = np.asarray(weights)

    if custom_shocks:
        shocks = custom_shocks
    elif name in config.HYPOTHETICAL_SCENARIOS:
        shocks = config.HYPOTHETICAL_SCENARIOS[name]
    else:
        raise ValueError(f"Unknown scenario: {name}")

    asset_ret = np.array([shocks.get(t, 0.0) for t in tickers])
    port_ret = float(w @ asset_ret)

    return {
        "name": name,
        "period": "Hypothetical",
        "return": port_ret,
        "max_drawdown": abs(min(port_ret, 0.0)),
        "var_95": abs(min(port_ret, 0.0)),
        "cvar_95": abs(min(port_ret, 0.0)),
        "worst_day": port_ret,
        "best_day": port_ret,
        "n_days": 1,
        "hypothetical": True,
        "severity": shocks.get("severity", 0.5),
    }


def run_all_hypothetical(tickers, weights):
    """Run all hypothetical scenarios."""
    return [run_hypothetical(name, tickers, weights)
            for name in config.HYPOTHETICAL_SCENARIOS]


def hedge_in_scenario(returns, weights, instrument, hedge_ratio,
                      name, start, end, spot):
    """Compare hedged vs unhedged in a historical stress window."""
    w = np.asarray(weights)
    mask = (returns.index >= start) & (returns.index <= end)
    sub = returns.loc[mask]

    if len(sub) < 2:
        return None

    port_ret = sub.values @ w
    unhedged_ret = float(np.prod(1 + port_ret) - 1)
    unhedged_cvar_val = cvar(port_ret, 0.95)

    # hedge payoff on the scenario outcome
    terminal_price = spot * (1 + unhedged_ret)
    payoff = float(instrument["payoff"](np.array([terminal_price]))[0]) / spot
    cost = instrument["premium_pct"] * hedge_ratio
    hedged_ret = unhedged_ret + hedge_ratio * payoff - cost

    # shift daily returns by net hedge for CVaR estimate
    hedge_net = hedge_ratio * payoff - cost
    hedged_port_ret = port_ret + hedge_net / len(port_ret)
    hedged_cvar_val = cvar(hedged_port_ret, 0.95)

    improvement = ((unhedged_cvar_val - hedged_cvar_val) / unhedged_cvar_val * 100
                   if unhedged_cvar_val > 0 else 0.0)

    return {
        "scenario": name,
        "unhedged_return": unhedged_ret,
        "hedged_return": hedged_ret,
        "hedge_benefit": hedged_ret - unhedged_ret,
        "unhedged_cvar": unhedged_cvar_val,
        "hedged_cvar": hedged_cvar_val,
        "cvar_improvement_pct": improvement,
    }
