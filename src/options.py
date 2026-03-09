"""Black-Scholes pricing, Greeks, and hedge instrument payoffs."""

import numpy as np
from scipy.stats import norm

import config


# ── BS primitives ─────────────────────────────────────────────────────────────

def _d1d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_call(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def put_greeks(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    return {
        "delta": norm.cdf(d1) - 1.0,
        "gamma": norm.pdf(d1) / (S * sigma * sqrt_T),
        "vega": S * norm.pdf(d1) * sqrt_T / 100,  # per 1% vol move
        "theta": (-(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365,
        "rho": -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100,
    }


def call_greeks(S, K, T, r, sigma):
    d1, d2 = _d1d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    return {
        "delta": norm.cdf(d1),
        "gamma": norm.pdf(d1) / (S * sigma * sqrt_T),
        "vega": S * norm.pdf(d1) * sqrt_T / 100,
        "theta": (-(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365,
        "rho": K * T * np.exp(-r * T) * norm.cdf(d2) / 100,
    }


# ── Hedge instruments ─────────────────────────────────────────────────────────
# Each returns a dict with name, premium, payoff(S_T), greeks, premium_pct

def _make_instrument(name, spot, premium_val, payoff_fn, greeks_dict):
    return {
        "name": name,
        "spot": spot,
        "premium": premium_val,
        "premium_pct": premium_val / spot,
        "payoff": payoff_fn,
        "greeks": greeks_dict,
    }


def protective_put(spot, sigma, T=None, r=None):
    """Long put at 95% moneyness."""
    T = T or config.HORIZON_DAYS / config.TRADING_DAYS_YEAR
    r = r or config.RISK_FREE_RATE
    K = spot * config.PUT_MONEYNESS
    prem = bs_put(spot, K, T, r, sigma)
    payoff_fn = lambda S_T: np.maximum(K - S_T, 0.0)
    g = put_greeks(spot, K, T, r, sigma)
    return _make_instrument("Protective Put", spot, prem, payoff_fn, {"put": g})


def collar(spot, sigma, T=None, r=None):
    """Long put + short OTM call — cost-reduced downside protection."""
    T = T or config.HORIZON_DAYS / config.TRADING_DAYS_YEAR
    r = r or config.RISK_FREE_RATE
    K_put = spot * config.COLLAR_PUT_MONEY
    K_call = spot * config.COLLAR_CALL_MONEY
    prem = bs_put(spot, K_put, T, r, sigma) - bs_call(spot, K_call, T, r, sigma)

    def payoff_fn(S_T):
        return np.maximum(K_put - S_T, 0.0) - np.maximum(S_T - K_call, 0.0)

    g = {
        "long_put": put_greeks(spot, K_put, T, r, sigma),
        "short_call": call_greeks(spot, K_call, T, r, sigma),
    }
    return _make_instrument("Collar", spot, prem, payoff_fn, g)


def put_spread(spot, sigma, T=None, r=None):
    """Long put K1 + short put K2 — cheaper, capped protection."""
    T = T or config.HORIZON_DAYS / config.TRADING_DAYS_YEAR
    r = r or config.RISK_FREE_RATE
    K_long = spot * config.SPREAD_LONG_MONEY
    K_short = spot * config.SPREAD_SHORT_MONEY
    prem = bs_put(spot, K_long, T, r, sigma) - bs_put(spot, K_short, T, r, sigma)

    def payoff_fn(S_T):
        return np.maximum(K_long - S_T, 0.0) - np.maximum(K_short - S_T, 0.0)

    g = {
        "long_put": put_greeks(spot, K_long, T, r, sigma),
        "short_put": put_greeks(spot, K_short, T, r, sigma),
    }
    return _make_instrument("Put Spread", spot, prem, payoff_fn, g)


def vix_call(spot, sigma, T=None, r=None):
    """VIX call proxy — convex payoff on vol spikes."""
    T = T or config.HORIZON_DAYS / config.TRADING_DAYS_YEAR
    r = r or config.RISK_FREE_RATE
    vix_now = config.VIX_CURRENT
    K = config.VIX_STRIKE
    prem = bs_call(vix_now, K, T, r, sigma)

    def payoff_fn(S_T):
        # rough proxy: VIX scales inversely with equity returns
        port_ret = S_T / spot - 1.0
        implied_vix = vix_now * (1.0 - 5.0 * port_ret)
        implied_vix = np.clip(implied_vix, 0, None)
        return np.maximum(implied_vix - K, 0.0)

    g = {"vix_call": call_greeks(vix_now, K, T, r, sigma)}
    return _make_instrument("VIX Call", spot, prem, payoff_fn, g)


def all_instruments(spot, sigma, T=None, r=None):
    """Create all four hedge instruments."""
    return [
        protective_put(spot, sigma, T, r),
        collar(spot, sigma, T, r),
        put_spread(spot, sigma, T, r),
        vix_call(spot, sigma, T, r),
    ]
