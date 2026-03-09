"""Factor analysis and risk parity — inspired by AQR, Bridgewater, HBK."""

import numpy as np
import pandas as pd

import config
from src.data_loader import fetch_prices, log_returns


# ── Momentum factor (AQR-style cross-sectional + time-series) ────────────────

def momentum_scores(prices, windows=None):
    """
    Cross-sectional momentum: rank assets by trailing return at each window.
    Returns dict of {window: DataFrame of z-scores}.
    """
    windows = windows or config.MOMENTUM_WINDOWS
    rets = log_returns(prices)
    scores = {}

    for w in windows:
        trailing = rets.rolling(w).sum()
        # z-score across assets at each point in time
        mu = trailing.mean(axis=1)
        sigma = trailing.std(axis=1)
        z = trailing.sub(mu, axis=0).div(sigma.replace(0, np.nan), axis=0)
        scores[w] = z.dropna()

    return scores


def current_momentum(prices, windows=None):
    """Latest momentum z-scores for each asset at each lookback."""
    scores = momentum_scores(prices, windows)
    current = {}
    for w, df in scores.items():
        if len(df) > 0:
            current[w] = df.iloc[-1].to_dict()
    return current


# ── Risk Parity (Bridgewater All-Weather inspired) ───────────────────────────

def risk_parity_weights(prices, lookback=252):
    """
    Inverse-vol risk parity: weight each asset by 1/vol, normalized.
    This is the simplest version of what Bridgewater does.
    """
    rets = log_returns(prices)
    recent = rets.iloc[-lookback:]
    vols = recent.std() * np.sqrt(252)

    # inverse vol weights
    inv_vol = 1.0 / vols
    w = inv_vol / inv_vol.sum()

    return {
        "tickers": list(prices.columns),
        "weights": w.values,
        "weights_dict": w.to_dict(),
        "asset_vols": vols.to_dict(),
    }


def compare_portfolios(prices, custom_weights, lookback=252):
    """
    Compare custom portfolio vs risk parity allocation.
    Returns stats for both.
    """
    from src.portfolio import portfolio_stats, cvar

    rets = log_returns(prices)
    custom_w = np.array(custom_weights)

    rp = risk_parity_weights(prices, lookback)
    rp_w = rp["weights"]

    custom_ret = rets.values @ custom_w
    rp_ret = rets.values @ rp_w

    return {
        "custom": {
            "weights": dict(zip(prices.columns, custom_w)),
            "stats": portfolio_stats(custom_ret),
        },
        "risk_parity": {
            "weights": rp["weights_dict"],
            "stats": portfolio_stats(rp_ret),
            "asset_vols": rp["asset_vols"],
        },
    }


# ── Correlation regime detection ─────────────────────────────────────────────

def rolling_correlation(prices, window=63):
    """Rolling pairwise correlation between portfolio assets."""
    rets = log_returns(prices)
    n = len(rets.columns)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = rets.columns[i], rets.columns[j]
            rolling_corr = rets[a].rolling(window).corr(rets[b]).dropna()
            pairs.append({
                "pair": f"{a}/{b}",
                "series": rolling_corr,
                "current": float(rolling_corr.iloc[-1]) if len(rolling_corr) > 0 else 0,
                "mean": float(rolling_corr.mean()),
                "std": float(rolling_corr.std()),
            })

    return pairs


# ── Drawdown analysis (Elliott-style risk monitoring) ─────────────────────────

def drawdown_analysis(prices, weights=None):
    """
    Detailed drawdown decomposition.
    Identifies all drawdown episodes, ranks by severity.
    """
    if weights is None:
        weights = config.DEFAULT_WEIGHTS
    weights = np.array(weights)
    rets = log_returns(prices)

    # portfolio level
    port_ret = rets.values @ weights
    wealth = np.cumprod(1 + port_ret)
    peak = np.maximum.accumulate(wealth)
    dd = (peak - wealth) / peak

    # find drawdown episodes
    in_dd = dd > 0.01  # at least 1% drawdown
    episodes = []
    start = None

    for i in range(len(dd)):
        if in_dd[i] and start is None:
            start = i
        elif not in_dd[i] and start is not None:
            max_dd = float(dd[start:i].max())
            if max_dd > 0.03:  # only report >3% drawdowns
                peak_idx = start + np.argmax(dd[start:i])
                episodes.append({
                    "start_date": str(rets.index[start].date()),
                    "trough_date": str(rets.index[peak_idx].date()),
                    "end_date": str(rets.index[i].date()),
                    "max_drawdown": max_dd,
                    "duration_days": i - start,
                    "recovery_days": i - peak_idx,
                })
            start = None

    # if still in drawdown at end of data
    if start is not None:
        max_dd = float(dd[start:].max())
        if max_dd > 0.03:
            peak_idx = start + np.argmax(dd[start:])
            episodes.append({
                "start_date": str(rets.index[start].date()),
                "trough_date": str(rets.index[peak_idx].date()),
                "end_date": "ongoing",
                "max_drawdown": max_dd,
                "duration_days": len(dd) - start,
                "recovery_days": None,
            })

    # sort by severity
    episodes.sort(key=lambda x: x["max_drawdown"], reverse=True)

    return {
        "drawdown_series": dd,
        "dates": rets.index,
        "wealth": wealth,
        "episodes": episodes[:15],  # top 15
        "current_drawdown": float(dd[-1]),
        "max_ever": float(dd.max()),
    }


# ── Tail dependency (beyond correlation — how do assets co-move in crashes) ──

def tail_dependency(prices, weights=None, threshold_pct=5):
    """
    Measure how assets co-move in the left tail.
    When portfolio is in worst 5% of days, what happens to each asset?
    """
    if weights is None:
        weights = config.DEFAULT_WEIGHTS
    weights = np.array(weights)
    rets = log_returns(prices)
    port_ret = rets.values @ weights

    threshold = np.percentile(port_ret, threshold_pct)
    tail_mask = port_ret <= threshold

    result = {}
    for col in rets.columns:
        asset_ret = rets[col].values
        result[col] = {
            "avg_return_in_tail": float(asset_ret[tail_mask].mean()),
            "avg_return_normal": float(asset_ret[~tail_mask].mean()),
            "tail_vol": float(asset_ret[tail_mask].std() * np.sqrt(252)),
            "normal_vol": float(asset_ret[~tail_mask].std() * np.sqrt(252)),
            # how much worse is tail correlation vs normal?
            "tail_beta": float(
                np.cov(asset_ret[tail_mask], port_ret[tail_mask])[0, 1] /
                np.var(port_ret[tail_mask])
            ) if np.var(port_ret[tail_mask]) > 0 else 0,
        }

    return result
