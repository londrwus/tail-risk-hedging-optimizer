"""Portfolio analytics — VaR, CVaR, risk attribution, stats."""

import numpy as np
import pandas as pd

import config


def var(returns, confidence=0.95):
    """Historical Value at Risk (positive = loss)."""
    return float(-np.percentile(returns, (1 - confidence) * 100))


def cvar(returns, confidence=0.95):
    """CVaR / Expected Shortfall — average loss beyond VaR."""
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else var(returns, confidence)


def max_drawdown(returns):
    """Maximum peak-to-trough drawdown from a return series."""
    wealth = np.cumprod(1 + np.asarray(returns))
    peak = np.maximum.accumulate(wealth)
    dd = (peak - wealth) / peak
    return float(np.max(dd))


def portfolio_stats(returns, rf=config.RISK_FREE_RATE):
    """Full stats suite — Sharpe, Sortino, Calmar, skew, kurtosis, VaR, CVaR."""
    r = np.asarray(returns)

    ann_ret = float(np.mean(r) * 252)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(252))
    excess = ann_ret - rf
    sharpe = excess / ann_vol if ann_vol > 0 else 0.0

    down = r[r < 0]
    down_vol = float(np.std(down, ddof=1) * np.sqrt(252)) if len(down) > 1 else 1e-8
    sortino = excess / down_vol

    mdd = max_drawdown(r)
    calmar = abs(ann_ret) / mdd if mdd > 0 else 0.0

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "skew": float(pd.Series(r).skew()),
        "kurtosis": float(pd.Series(r).kurtosis()),
        "var_95": var(r, 0.95),
        "var_99": var(r, 0.99),
        "cvar_95": cvar(r, 0.95),
        "cvar_99": cvar(r, 0.99),
    }


def euler_risk_attribution(returns, weights, confidence=0.95):
    """
    Marginal CVaR contribution per asset — Euler decomposition.
    Uses direct tail-conditional mean, not finite differences.
    """
    w = np.array(weights)
    arr = returns.values if hasattr(returns, "values") else returns
    port_ret = arr @ w

    threshold = np.percentile(port_ret, (1 - confidence) * 100)
    tail_mask = port_ret <= threshold

    tickers = list(returns.columns) if hasattr(returns, "columns") else [f"asset_{i}" for i in range(len(w))]

    # component CVaR = weight_i * E[r_i | portfolio in tail]
    comp_cvar = {}
    for i, t in enumerate(tickers):
        comp_cvar[t] = float(-arr[tail_mask, i].mean() * w[i])

    total = sum(comp_cvar.values())
    pct = {t: v / total if abs(total) > 1e-12 else 0.0 for t, v in comp_cvar.items()}

    return {
        "tickers": tickers,
        "weights": w,
        "component_cvar": comp_cvar,
        "pct_contribution": pct,
        "total_cvar": total,
    }


def correlation_breakdown(returns, stress_periods=None):
    """Normal vs stress-period correlation matrices."""
    stress_periods = stress_periods or config.STRESS_SCENARIOS
    result = {"full_period": returns.corr()}

    for name, dates in stress_periods.items():
        mask = (returns.index >= dates["start"]) & (returns.index <= dates["end"])
        sub = returns.loc[mask]
        if len(sub) > 5:
            result[name] = sub.corr()

    return result
