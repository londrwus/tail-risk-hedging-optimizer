"""Rolling historical backtester — estimate vol → price hedge → evaluate OOS."""

import numpy as np
import pandas as pd

import config
from src.data_loader import log_returns
from src.options import all_instruments
from src.portfolio import cvar, var


def run_backtest(prices, weights=None, est_window=config.BACKTEST_EST_WINDOW,
                 hold_period=config.BACKTEST_HOLD_PERIOD,
                 budget=config.HEDGE_BUDGET, instrument_type="put"):
    """
    Rolling backtest for one instrument type.

    Walk through history: estimate vol → price hedge → hold → evaluate.
    """
    if weights is None:
        weights = config.DEFAULT_WEIGHTS
    w = np.array(weights)
    rets = log_returns(prices)
    n_obs = len(rets)

    inst_idx = {"put": 0, "collar": 1, "put_spread": 2, "vix_call": 3}
    idx = inst_idx.get(instrument_type, 0)

    steps = []
    t = est_window

    while t + hold_period <= n_obs:
        est_rets = rets.iloc[t - est_window:t]
        est_vol = float(est_rets.std().mean() * np.sqrt(config.TRADING_DAYS_YEAR))

        # current portfolio value as weighted price
        current_prices = prices.iloc[t]
        spot = float((current_prices * w).sum())

        # OOS portfolio return
        oos_rets = rets.iloc[t:t + hold_period]
        port_oos = float((oos_rets.values @ w).sum())

        # price the hedge
        T = hold_period / config.TRADING_DAYS_YEAR
        instruments = all_instruments(spot=spot, sigma=est_vol, T=T)
        inst = instruments[idx]

        prem_pct = inst["premium_pct"]
        if prem_pct > 0:
            ratio = min(budget / prem_pct, 1.0)
        else:
            ratio = 1.0

        terminal_price = spot * (1 + port_oos)
        payoff = float(inst["payoff"](np.array([terminal_price]))[0]) / spot
        cost = prem_pct * ratio
        hedged_ret = port_oos + ratio * payoff - cost

        steps.append({
            "date": rets.index[t],
            "est_vol": est_vol,
            "unhedged_ret": port_oos,
            "hedged_ret": hedged_ret,
            "hedge_cost": cost,
            "hedge_payoff": payoff * ratio,
        })

        t += hold_period

    # pack into result dict
    dates = [s["date"] for s in steps]
    unhedged = np.array([s["unhedged_ret"] for s in steps])
    hedged = np.array([s["hedged_ret"] for s in steps])

    return {
        "instrument": instrument_type,
        "steps": steps,
        "dates": dates,
        "unhedged_returns": unhedged,
        "hedged_returns": hedged,
        "cum_unhedged": np.cumprod(1 + unhedged) - 1 if len(unhedged) > 0 else np.array([]),
        "cum_hedged": np.cumprod(1 + hedged) - 1 if len(hedged) > 0 else np.array([]),
        "unhedged_cvar": cvar(unhedged, 0.95) if len(unhedged) > 0 else 0.0,
        "hedged_cvar": cvar(hedged, 0.95) if len(hedged) > 0 else 0.0,
    }


def run_all_backtests(prices, weights=None):
    """Backtest all four instrument types."""
    return {
        itype: run_backtest(prices, weights, instrument_type=itype)
        for itype in ["put", "collar", "put_spread", "vix_call"]
    }
