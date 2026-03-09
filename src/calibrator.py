"""
Portfolio calibration — optimal weight allocation for different risk profiles.

Runs mean-variance optimization (Markowitz), min-CVaR, max-Sharpe, risk parity,
and packages them into named strategies (aggressive, balanced, conservative, etc.).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config
from src.data_loader import fetch_prices, log_returns
from src.portfolio import cvar, var, portfolio_stats, max_drawdown
from src.simulation import run_gbm


STRATEGY_PROFILES = {
    "max_sharpe": {
        "label": "Maximum Sharpe",
        "description": "Best risk-adjusted return. Classic Markowitz tangency portfolio.",
        "style": "Aggressive-Balanced",
    },
    "min_vol": {
        "label": "Minimum Volatility",
        "description": "Lowest possible portfolio variance. Flight-to-quality allocation.",
        "style": "Conservative",
    },
    "min_cvar": {
        "label": "Minimum CVaR",
        "description": "Minimizes expected loss in the worst 5% of scenarios. Tail-risk focused.",
        "style": "Conservative-Defensive",
    },
    "risk_parity": {
        "label": "Risk Parity",
        "description": "Equal risk contribution from each asset. Bridgewater All-Weather inspired.",
        "style": "Balanced",
    },
    "max_return": {
        "label": "Maximum Return",
        "description": "Highest expected return with no risk constraint. Concentrated and aggressive.",
        "style": "Aggressive",
    },
    "balanced": {
        "label": "Balanced (Target Vol 12%)",
        "description": "Max return subject to 12% annualized vol target. Middle-of-the-road.",
        "style": "Balanced",
    },
    "conservative": {
        "label": "Conservative (Target Vol 8%)",
        "description": "Max return subject to 8% vol target. Lower drawdowns, steadier ride.",
        "style": "Conservative",
    },
    "aggressive": {
        "label": "Aggressive (Target Vol 18%)",
        "description": "Max return subject to 18% vol target. Higher risk, higher potential.",
        "style": "Aggressive",
    },
}


def _ann_ret(w, mu):
    return float(w @ mu)


def _ann_vol(w, cov):
    return float(np.sqrt(w @ cov @ w))


def _neg_sharpe(w, mu, cov, rf):
    ret = _ann_ret(w, mu)
    vol = _ann_vol(w, cov)
    return -(ret - rf) / vol if vol > 1e-8 else 0


def _portfolio_cvar(w, returns, confidence=0.95):
    port_ret = returns @ w
    return cvar(port_ret, confidence)


def optimize_weights(prices, tickers=None, lookback=252, min_weight=0.0, max_weight=0.60):
    """
    Run all optimization strategies on the given assets.
    Returns dict of strategy_name -> {weights, stats, ...}
    """
    tickers = tickers or list(prices.columns)
    n = len(tickers)

    rets = log_returns(prices[tickers])
    # use recent data for forward-looking estimates
    recent = rets.iloc[-lookback:] if len(rets) > lookback else rets

    mu = recent.mean().values * 252
    cov = recent.cov().values * 252
    returns_arr = recent.values

    rf = config.RISK_FREE_RATE
    bounds = [(min_weight, max_weight)] * n
    constraints_sum1 = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    w0 = np.ones(n) / n

    results = {}

    # ── Max Sharpe ──
    res = minimize(_neg_sharpe, w0, args=(mu, cov, rf), method="SLSQP",
                   bounds=bounds, constraints=[constraints_sum1],
                   options={"ftol": 1e-10, "maxiter": 500})
    results["max_sharpe"] = _pack_result("max_sharpe", res.x, rets, tickers)

    # ── Min Volatility ──
    res = minimize(lambda w: _ann_vol(w, cov), w0, method="SLSQP",
                   bounds=bounds, constraints=[constraints_sum1],
                   options={"ftol": 1e-10, "maxiter": 500})
    results["min_vol"] = _pack_result("min_vol", res.x, rets, tickers)

    # ── Min CVaR ──
    res = minimize(lambda w: _portfolio_cvar(w, returns_arr), w0, method="SLSQP",
                   bounds=bounds, constraints=[constraints_sum1],
                   options={"ftol": 1e-10, "maxiter": 500})
    results["min_cvar"] = _pack_result("min_cvar", res.x, rets, tickers)

    # ── Risk Parity (inverse vol) ──
    vols = recent.std().values * np.sqrt(252)
    inv_vol = 1.0 / np.clip(vols, 0.01, None)
    rp_w = inv_vol / inv_vol.sum()
    results["risk_parity"] = _pack_result("risk_parity", rp_w, rets, tickers)

    # ── Max Return ──
    res = minimize(lambda w: -_ann_ret(w, mu), w0, method="SLSQP",
                   bounds=bounds, constraints=[constraints_sum1],
                   options={"ftol": 1e-10, "maxiter": 500})
    results["max_return"] = _pack_result("max_return", res.x, rets, tickers)

    # ── Target vol strategies ──
    for strat_name, target_vol in [("balanced", 0.12), ("conservative", 0.08), ("aggressive", 0.18)]:
        vol_constraint = {"type": "ineq", "fun": lambda w, tv=target_vol: tv - _ann_vol(w, cov)}
        res = minimize(lambda w: -_ann_ret(w, mu), w0, method="SLSQP",
                       bounds=bounds,
                       constraints=[constraints_sum1, vol_constraint],
                       options={"ftol": 1e-10, "maxiter": 500})
        results[strat_name] = _pack_result(strat_name, res.x, rets, tickers)

    return results


def _pack_result(strategy_name, weights, returns, tickers):
    """Package optimized weights with stats and metadata."""
    w = np.array(weights)
    # clean up near-zero weights
    w[np.abs(w) < 1e-4] = 0.0
    w = w / w.sum()

    port_ret = returns.values @ w
    stats = portfolio_stats(port_ret)
    profile = STRATEGY_PROFILES[strategy_name]

    return {
        "strategy": strategy_name,
        "label": profile["label"],
        "description": profile["description"],
        "style": profile["style"],
        "tickers": tickers,
        "weights": {t: float(w[i]) for i, t in enumerate(tickers)},
        "weights_arr": w,
        "stats": stats,
    }


def efficient_frontier(prices, tickers=None, n_points=40, lookback=252,
                       min_weight=0.0, max_weight=0.60):
    """
    Trace out the mean-variance efficient frontier.
    Returns list of (vol, return, weights) tuples.
    """
    tickers = tickers or list(prices.columns)
    n = len(tickers)

    rets = log_returns(prices[tickers])
    recent = rets.iloc[-lookback:] if len(rets) > lookback else rets

    mu = recent.mean().values * 252
    cov = recent.cov().values * 252

    bounds = [(min_weight, max_weight)] * n
    constraints_sum1 = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    w0 = np.ones(n) / n

    # find return range
    min_ret_res = minimize(lambda w: _ann_ret(w, mu), w0, method="SLSQP",
                           bounds=bounds, constraints=[constraints_sum1])
    max_ret_res = minimize(lambda w: -_ann_ret(w, mu), w0, method="SLSQP",
                           bounds=bounds, constraints=[constraints_sum1])

    min_r = _ann_ret(min_ret_res.x, mu)
    max_r = _ann_ret(max_ret_res.x, mu)
    target_rets = np.linspace(min_r, max_r, n_points)

    frontier = []
    for target in target_rets:
        ret_constraint = {"type": "eq", "fun": lambda w, t=target: _ann_ret(w, mu) - t}
        res = minimize(lambda w: _ann_vol(w, cov), w0, method="SLSQP",
                       bounds=bounds,
                       constraints=[constraints_sum1, ret_constraint],
                       options={"ftol": 1e-12, "maxiter": 500})
        if res.success:
            vol = _ann_vol(res.x, cov)
            frontier.append({"vol": vol, "ret": target, "weights": res.x.tolist()})

    return frontier


def mc_strategy_comparison(prices, strategies, n_sims=2000, horizon=config.HORIZON_DAYS,
                           seed=config.RANDOM_SEED):
    """
    Run MC simulation for each strategy to compare forward-looking risk.
    Returns terminal return distributions per strategy.
    """
    comparisons = {}
    for name, strat in strategies.items():
        w = strat["weights_arr"]
        sim = run_gbm(prices, weights=w, n_sims=n_sims, horizon=horizon, seed=seed)
        term = sim["terminal_returns"]
        comparisons[name] = {
            "label": strat["label"],
            "terminal_returns": term,
            "mean_ret": float(np.mean(term)),
            "median_ret": float(np.median(term)),
            "var_95": float(var(term, 0.95)),
            "cvar_95": float(cvar(term, 0.95)),
            "prob_loss": float(np.mean(term < 0)),
            "pct_5": float(np.percentile(term, 5)),
            "pct_95": float(np.percentile(term, 95)),
        }
    return comparisons
