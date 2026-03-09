"""Entry timing analysis — should you buy now or scale in gradually?

Compares lump-sum vs DCA strategies using Monte Carlo simulation.
Generates buy-in schedules at different price levels and evaluates
expected cost basis, risk, and probability of better entry points.
"""

import numpy as np
import pandas as pd

import config
from src.data_loader import fetch_prices, log_returns
from src.portfolio import cvar, var


def _simulate_prices(S0, mu, sigma, horizon, n_sims, seed=config.RANDOM_SEED):
    """GBM paths for a single asset or portfolio spot price."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_sims, horizon))
    cum = np.cumsum(drift + diffusion * Z, axis=1)
    # prepend day 0 (no change)
    paths = S0 * np.exp(np.hstack([np.zeros((n_sims, 1)), cum]))
    return paths


def _path_drawdowns(paths, S0):
    """For each simulated path, find the max drawdown from the start price."""
    mins = np.minimum.accumulate(paths, axis=1)
    dd_from_start = (S0 - mins) / S0
    return dd_from_start


def analyze_entry(prices, weights, horizon_days=252, n_sims=5000,
                  seed=config.RANDOM_SEED):
    """
    Full entry timing analysis for the active portfolio.

    Returns dca schedules, lump-sum vs dca comparison, and
    probability of better entry points.
    """
    w = np.array(weights)
    rets = log_returns(prices)
    port_ret = (rets.values @ w)

    mu = float(np.mean(port_ret) * 252)
    sigma = float(np.std(port_ret) * np.sqrt(252))

    # current portfolio spot (weighted sum of latest prices)
    last_prices = prices.iloc[-1].values
    S0 = float((last_prices * w).sum())

    paths = _simulate_prices(S0, mu, sigma, horizon_days, n_sims, seed)

    # ── probability of dips (better entry points) ──
    dip_levels = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    dip_probs = {}
    for dip in dip_levels:
        target = S0 * (1 - dip)
        # did the path ever touch this level?
        hit = np.any(paths <= target, axis=1)
        dip_probs[dip] = float(np.mean(hit))

    # ── lump sum: buy everything at S0 on day 0 ──
    terminal_lump = paths[:, -1]
    lump_ret = terminal_lump / S0 - 1

    # ── DCA schedules ──
    # build a few practical schedules
    schedules = _build_schedules(horizon_days)
    dca_results = {}

    for name, schedule in schedules.items():
        cost_bases = []
        terminal_vals = []

        for sim_i in range(n_sims):
            path = paths[sim_i]
            total_invested = 0.0
            total_shares = 0.0

            for day_idx, pct in schedule:
                if day_idx < len(path):
                    buy_price = path[day_idx]
                    amount = pct  # fraction of total capital
                    shares = amount / buy_price
                    total_invested += amount
                    total_shares += shares

            if total_shares > 0 and total_invested > 0:
                avg_cost = total_invested / total_shares
                final_val = total_shares * path[-1]
                cost_bases.append(avg_cost)
                terminal_vals.append(final_val)
            else:
                cost_bases.append(S0)
                terminal_vals.append(1.0)

        cost_bases = np.array(cost_bases)
        terminal_vals = np.array(terminal_vals)
        dca_ret = terminal_vals / 1.0 - 1  # total capital = 1.0

        dca_results[name] = {
            "label": _schedule_label(name),
            "schedule": schedule,
            "avg_cost_basis": float(np.mean(cost_bases)),
            "median_cost_basis": float(np.median(cost_bases)),
            "terminal_returns": dca_ret,
            "mean_return": float(np.mean(dca_ret)),
            "median_return": float(np.median(dca_ret)),
            "var_95": float(var(dca_ret, 0.95)),
            "cvar_95": float(cvar(dca_ret, 0.95)),
            "prob_loss": float(np.mean(dca_ret < 0)),
            "prob_beat_lump": float(np.mean(dca_ret > lump_ret)),
        }

    # ── price-level buy plan ──
    # "if price drops to X, deploy Y% of remaining capital"
    level_plan = _price_level_plan(S0, sigma)

    # ── timing score: is now a good time? ──
    timing = _timing_assessment(prices, w, mu, sigma, dip_probs)

    return {
        "spot": S0,
        "mu": mu,
        "sigma": sigma,
        "horizon_days": horizon_days,
        "paths": paths,
        "dip_probabilities": dip_probs,
        "lump_sum": {
            "terminal_returns": lump_ret,
            "mean_return": float(np.mean(lump_ret)),
            "median_return": float(np.median(lump_ret)),
            "var_95": float(var(lump_ret, 0.95)),
            "cvar_95": float(cvar(lump_ret, 0.95)),
            "prob_loss": float(np.mean(lump_ret < 0)),
        },
        "dca_results": dca_results,
        "level_plan": level_plan,
        "timing": timing,
    }


def _build_schedules(horizon_days):
    """Build a set of DCA schedules — (day_index, fraction) pairs."""
    # adapt to horizon
    monthly_days = min(21, horizon_days // 4)

    schedules = {}

    # aggressive: 50% now, 25% in 1 month, 15% in 2 months, 10% in 3 months
    schedules["aggressive"] = [
        (0, 0.50), (monthly_days, 0.25),
        (monthly_days * 2, 0.15), (monthly_days * 3, 0.10),
    ]

    # balanced: equal tranches monthly over ~4 months
    n_tranches = min(4, max(2, horizon_days // monthly_days))
    equal_pct = 1.0 / n_tranches
    schedules["balanced"] = [
        (monthly_days * i, equal_pct) for i in range(n_tranches)
    ]

    # conservative: 25% now, then 15% each month for 5 months
    if horizon_days > monthly_days * 4:
        schedules["conservative"] = [
            (0, 0.25), (monthly_days, 0.15), (monthly_days * 2, 0.15),
            (monthly_days * 3, 0.15), (monthly_days * 4, 0.15),
            (monthly_days * 5, 0.15),
        ]
    else:
        n = min(6, max(3, horizon_days // monthly_days))
        first = 0.25
        rest = 0.75 / (n - 1)
        schedules["conservative"] = [(monthly_days * i, first if i == 0 else rest) for i in range(n)]

    # ultra_conservative: 10% now, then spread rest evenly over 6-9 months
    if horizon_days > monthly_days * 6:
        n = min(10, horizon_days // monthly_days)
        first = 0.10
        rest = 0.90 / (n - 1)
        schedules["ultra_conservative"] = [
            (monthly_days * i, first if i == 0 else rest) for i in range(n)
        ]

    return schedules


def _schedule_label(name):
    labels = {
        "aggressive": "Aggressive DCA (50/25/15/10)",
        "balanced": "Equal Monthly DCA",
        "conservative": "Conservative DCA (25% + monthly)",
        "ultra_conservative": "Ultra Conservative (10% + spread)",
    }
    return labels.get(name, name)


def _price_level_plan(S0, sigma):
    """
    Generate a buy-the-dip ladder.
    Deploy capital at specific price levels based on portfolio vol.
    """
    # scale dip triggers to actual portfolio volatility
    # higher vol portfolio = wider spacing
    vol_factor = max(sigma / 0.15, 0.8)  # normalize around 15% vol

    levels = []
    remaining = 1.0
    base_dips = [0.0, 0.03, 0.07, 0.12, 0.18, 0.25]
    base_allocs = [0.30, 0.20, 0.20, 0.15, 0.10, 0.05]

    for dip, alloc in zip(base_dips, base_allocs):
        adj_dip = dip * vol_factor
        price = S0 * (1 - adj_dip)
        pct = min(alloc, remaining)
        if pct < 0.01:
            break
        levels.append({
            "price": round(price, 2),
            "dip_pct": adj_dip,
            "allocation": pct,
            "cumulative": 1.0 - remaining + pct,
            "rationale": _dip_rationale(adj_dip),
        })
        remaining -= pct

    return levels


def _dip_rationale(dip):
    if dip < 0.01:
        return "Initial position — establish exposure"
    elif dip < 0.05:
        return "Minor pullback — normal vol, add on weakness"
    elif dip < 0.10:
        return "Meaningful correction — historically recoverable"
    elif dip < 0.20:
        return "Significant drawdown — high conviction level"
    elif dip < 0.30:
        return "Major selloff — crisis-level entry"
    return "Extreme dislocation — maximum conviction"


def _timing_assessment(prices, w, mu, sigma, dip_probs):
    """Quick assessment: is now a reasonable time to start buying?
    Uses traditional signals + ML crash probability from XGBoost."""
    rets = log_returns(prices)
    port_ret = rets.values @ w

    # recent momentum (3 month)
    recent_ret = float(np.sum(port_ret[-63:]))
    # vol regime
    recent_vol = float(np.std(port_ret[-63:]) * np.sqrt(252))

    # how far from recent highs?
    port_prices = (prices.values * w[None, :]).sum(axis=1)
    peak_252 = float(np.max(port_prices[-252:]))
    current = float(port_prices[-1])
    pct_from_peak = (current - peak_252) / peak_252

    # ML crash probability
    ml_crash_prob = None
    ml_return_pred = None
    try:
        from src.ml_models import train_crash_predictor, train_return_predictor
        crash = train_crash_predictor(prices)
        if crash:
            ml_crash_prob = crash["current_crash_prob"]
        ret_pred = train_return_predictor(prices, forward_days=21)
        if ret_pred:
            ml_return_pred = ret_pred["current_prediction"]
    except Exception:
        pass

    # scoring — higher is more favorable to buy
    score = 50  # neutral start

    # momentum: negative recent returns = slightly better entry
    if recent_ret < -0.05:
        score += 15
    elif recent_ret < 0:
        score += 5
    elif recent_ret > 0.15:
        score -= 10

    # distance from highs: buying dips scores higher
    if pct_from_peak < -0.15:
        score += 20
    elif pct_from_peak < -0.05:
        score += 10
    elif pct_from_peak > -0.02:
        score -= 5

    # vol regime: high vol = uncertainty but also better prices
    if recent_vol > sigma * 1.3:
        score += 5
    elif recent_vol < sigma * 0.7:
        score -= 5

    # probability of 5% dip in horizon — if high, maybe wait a bit
    if dip_probs.get(0.05, 0) > 0.70:
        score -= 10
    elif dip_probs.get(0.05, 0) < 0.30:
        score += 10

    # ML signals — XGBoost crash predictor and return forecast
    if ml_crash_prob is not None:
        if ml_crash_prob > 0.6:
            score -= 15
        elif ml_crash_prob > 0.4:
            score -= 8
        elif ml_crash_prob < 0.15:
            score += 8

    if ml_return_pred is not None:
        if ml_return_pred > 0.03:
            score += 5
        elif ml_return_pred < -0.03:
            score -= 5

    score = max(0, min(100, score))

    if score >= 70:
        verdict = "Favorable — conditions support starting to buy"
        strategy = "aggressive"
    elif score >= 50:
        verdict = "Neutral — no strong signal either way, DCA recommended"
        strategy = "balanced"
    elif score >= 35:
        verdict = "Cautious — consider scaling in slowly"
        strategy = "conservative"
    else:
        verdict = "Unfavorable — high risk of near-term drawdown, be patient"
        strategy = "ultra_conservative"

    factors = []
    if recent_ret < -0.05:
        factors.append(f"Recent 3-month return is negative ({recent_ret:.1%}), potential mean reversion")
    if recent_ret > 0.10:
        factors.append(f"Strong recent rally ({recent_ret:.1%}), risk of pullback")
    if pct_from_peak < -0.10:
        factors.append(f"Trading {abs(pct_from_peak):.1%} below 52-week high")
    if pct_from_peak > -0.02:
        factors.append("Near all-time highs — less margin of safety")
    if recent_vol > sigma * 1.2:
        factors.append(f"Elevated volatility ({recent_vol:.1%} vs {sigma:.1%} historical)")
    prob5 = dip_probs.get(0.05, 0)
    factors.append(f"{prob5:.0%} probability of a 5% dip within the horizon")

    if ml_crash_prob is not None:
        factors.append(f"XGBoost crash probability: {ml_crash_prob:.0%} (next 21 days)")
    if ml_return_pred is not None:
        direction = "positive" if ml_return_pred > 0 else "negative"
        factors.append(f"XGBoost return forecast: {ml_return_pred:.2%} ({direction}, 21-day)")

    return {
        "score": score,
        "verdict": verdict,
        "recommended_strategy": strategy,
        "factors": factors,
        "recent_return_3m": recent_ret,
        "recent_vol": recent_vol,
        "pct_from_peak": pct_from_peak,
        "ml_crash_prob": ml_crash_prob,
        "ml_return_pred": ml_return_pred,
    }
