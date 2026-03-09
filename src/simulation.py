"""Monte Carlo engine — correlated GBM with regime-conditional vol + block bootstrap."""

import numpy as np
import pandas as pd

import config
from src.data_loader import log_returns


def detect_regime(returns, lookback=config.REGIME_LOOKBACK, use_ml=True):
    """Classify current vol regime — uses KMeans clustering when possible,
    falls back to simple vol thresholds if not enough data."""
    if use_ml:
        try:
            from src.ml_models import detect_regime_ml
            result = detect_regime_ml(returns)
            return result["regime"]
        except Exception:
            pass

    # fallback: simple threshold-based detection
    port_ret = returns.mean(axis=1)
    recent = port_ret.iloc[-lookback:]
    rvol = recent.std() * np.sqrt(config.TRADING_DAYS_YEAR)

    if rvol < config.REGIME_THRESHOLDS["low"]:
        return "low"
    elif rvol > config.REGIME_THRESHOLDS["high"]:
        return "high"
    return "mid"


def run_gbm(prices, weights=None, n_sims=config.N_SIMULATIONS,
            horizon=config.HORIZON_DAYS, seed=config.RANDOM_SEED,
            regime_override=None):
    """
    Correlated GBM with regime-conditional vol scaling.

    Returns dict with paths, terminal_returns, portfolio_paths, regime, sim_vol.
    """
    if weights is None:
        weights = config.DEFAULT_WEIGHTS
    w = np.array(weights)
    rng = np.random.default_rng(seed)

    rets = log_returns(prices)
    n_assets = len(prices.columns)

    mu = rets.mean().values * config.TRADING_DAYS_YEAR
    sigma = rets.std().values * np.sqrt(config.TRADING_DAYS_YEAR)

    regime = regime_override or detect_regime(rets)
    vol_scale = config.REGIME_VOL_SCALE[regime]
    sigma_adj = sigma * vol_scale

    # cholesky on correlation matrix
    corr = rets.corr().values
    # small jitter to avoid cholesky failing on near-singular matrices
    L = np.linalg.cholesky(corr + 1e-8 * np.eye(n_assets))

    dt = 1.0 / config.TRADING_DAYS_YEAR
    drift = (mu - 0.5 * sigma_adj ** 2) * dt  # Ito correction
    diffusion = sigma_adj * np.sqrt(dt)

    Z = rng.standard_normal((n_sims, horizon, n_assets))
    Z_corr = Z @ L.T

    lr = drift[None, None, :] + diffusion[None, None, :] * Z_corr
    cum_lr = np.cumsum(lr, axis=1)

    s0 = prices.iloc[-1].values
    paths = s0[None, None, :] * np.exp(cum_lr)

    # portfolio-level paths
    price_ratios = paths / s0[None, None, :]
    port_paths = np.sum(price_ratios * w[None, None, :], axis=2)
    terminal_ret = port_paths[:, -1] - 1.0

    return {
        "paths": paths,
        "terminal_returns": terminal_ret,
        "portfolio_paths": port_paths,
        "regime": regime,
        "sim_vol": sigma_adj,
    }


def run_bootstrap(prices, weights=None, n_sims=config.N_SIMULATIONS,
                  horizon=config.HORIZON_DAYS, block_size=config.BLOCK_SIZE,
                  seed=config.RANDOM_SEED):
    """Block bootstrap — preserves vol clustering and cross-asset dependence."""
    if weights is None:
        weights = config.DEFAULT_WEIGHTS
    w = np.array(weights)
    rng = np.random.default_rng(seed)

    rets = log_returns(prices)
    arr = rets.values
    n_obs, n_assets = arr.shape

    n_blocks = int(np.ceil(horizon / block_size))
    paths = np.zeros((n_sims, horizon, n_assets))

    for i in range(n_sims):
        blocks = []
        for _ in range(n_blocks):
            start = rng.integers(0, n_obs - block_size)
            blocks.append(arr[start:start + block_size])
        sim_ret = np.concatenate(blocks, axis=0)[:horizon]
        paths[i] = prices.iloc[-1].values * np.exp(np.cumsum(sim_ret, axis=0))

    s0 = prices.iloc[-1].values
    price_ratios = paths / s0[None, None, :]
    port_paths = np.sum(price_ratios * w[None, None, :], axis=2)
    terminal_ret = port_paths[:, -1] - 1.0

    regime = detect_regime(rets)

    return {
        "paths": paths,
        "terminal_returns": terminal_ret,
        "portfolio_paths": port_paths,
        "regime": regime,
        "sim_vol": rets.std().values * np.sqrt(config.TRADING_DAYS_YEAR),
    }
