"""Monte Carlo price predictions — dynamic ticker, timeframe, risk analysis."""

import numpy as np
import pandas as pd

import config
from src.data_loader import fetch_prices, log_returns
from src.portfolio import cvar, var


# wider universe of tickers people might want to predict
TICKER_UNIVERSE = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "GLD": "Gold ETF",
    "SLV": "Silver ETF",
    "USO": "Crude Oil ETF",
    "TLT": "20+ Year Treasury",
    "EFA": "Developed Markets ex-US",
    "EEM": "Emerging Markets",
    "IWM": "Russell 2000 Small Cap",
    "XLE": "Energy Sector",
    "XLF": "Financial Sector",
    "XLK": "Technology Sector",
    "HYG": "High Yield Corporate Bonds",
    "DBA": "Agriculture Commodities",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "TON11419-USD": "Toncoin",
}

TIMEFRAMES = {
    "3m": {"label": "3 Months", "days": 63},
    "6m": {"label": "6 Months", "days": 126},
    "1y": {"label": "1 Year", "days": 252},
    "2y": {"label": "2 Years", "days": 504},
    "3y": {"label": "3 Years", "days": 756},
    "5y": {"label": "5 Years", "days": 1260},
}


def predict_asset(ticker, horizon_days=252, n_sims=config.PREDICTION_SIMS,
                  seed=config.RANDOM_SEED):
    """
    GBM price prediction for a single asset.
    Returns percentile bands, risk metrics, stress analysis.
    """
    # fetch can fail on invalid tickers or network issues
    prices = fetch_prices([ticker], start="2010-01-01")
    if prices is None or prices.empty:
        return None

    if ticker not in prices.columns:
        return None

    series = prices[ticker].dropna()
    if len(series) < 60:
        return None

    rets = np.log(series / series.shift(1)).dropna().values

    mu = float(np.mean(rets) * 252)
    sigma = float(np.std(rets) * np.sqrt(252))

    rng = np.random.default_rng(seed)
    dt = 1 / 252
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    S0 = float(series.iloc[-1])
    Z = rng.standard_normal((n_sims, horizon_days))
    log_ret = drift + diffusion * Z
    cum = np.cumsum(log_ret, axis=1)
    paths = S0 * np.exp(cum)

    pcts = [5, 10, 25, 50, 75, 90, 95]
    bands = {p: np.percentile(paths, p, axis=0).tolist() for p in pcts}

    terminal = paths[:, -1]
    terminal_ret = terminal / S0 - 1

    # per-path max drawdown
    max_dds = []
    for i in range(n_sims):
        path = paths[i]
        peak = np.maximum.accumulate(path)
        dd = (peak - path) / peak
        max_dds.append(float(dd.max()))
    max_dds = np.array(max_dds)

    # stress scenarios on this specific asset
    # what if vol doubles? what if a 3-sigma shock happens?
    stress_paths_high_vol = S0 * np.exp(
        np.cumsum((drift + diffusion * 2.0 * rng.standard_normal((500, horizon_days))), axis=1)
    )
    stress_terminal = stress_paths_high_vol[:, -1]

    # price levels of interest
    last_date = series.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days)

    # key support/resistance from recent history
    recent = series.iloc[-252:] if len(series) > 252 else series
    hist_high = float(recent.max())
    hist_low = float(recent.min())

    return {
        "ticker": ticker,
        "label": TICKER_UNIVERSE.get(ticker, ticker),
        "current_price": S0,
        "mu": mu,
        "sigma": sigma,
        "horizon_days": horizon_days,
        "bands": bands,
        "dates": future_dates.tolist(),

        # terminal stats
        "terminal_mean": float(np.mean(terminal)),
        "terminal_median": float(np.median(terminal)),
        "terminal_std": float(np.std(terminal)),
        "pct_5_price": float(np.percentile(terminal, 5)),
        "pct_25_price": float(np.percentile(terminal, 25)),
        "pct_75_price": float(np.percentile(terminal, 75)),
        "pct_95_price": float(np.percentile(terminal, 95)),

        # return probabilities
        "prob_up": float(np.mean(terminal > S0)),
        "prob_down_10": float(np.mean(terminal < S0 * 0.9)),
        "prob_down_20": float(np.mean(terminal < S0 * 0.8)),
        "prob_up_20": float(np.mean(terminal > S0 * 1.2)),
        "prob_up_50": float(np.mean(terminal > S0 * 1.5)),
        "expected_return": float(np.mean(terminal_ret)),
        "pct_5_return": float(np.percentile(terminal_ret, 5)),
        "pct_95_return": float(np.percentile(terminal_ret, 95)),

        # risk metrics
        "var_95": float(var(terminal_ret, 0.95)),
        "cvar_95": float(cvar(terminal_ret, 0.95)),
        "avg_max_drawdown": float(np.mean(max_dds)),
        "worst_max_drawdown": float(np.max(max_dds)),
        "prob_drawdown_gt_20": float(np.mean(max_dds > 0.20)),

        # stress scenario: doubled vol
        "stress_high_vol_median": float(np.median(stress_terminal)),
        "stress_high_vol_5pct": float(np.percentile(stress_terminal, 5)),
        "stress_high_vol_prob_down": float(np.mean(stress_terminal < S0)),

        # historical context
        "hist_52w_high": hist_high,
        "hist_52w_low": hist_low,
        "pct_from_high": float((S0 - hist_high) / hist_high),

        # risk concerns — flag anything worrying
        "risk_flags": _risk_flags(S0, sigma, mu, terminal_ret, max_dds),

        # ML-enhanced prediction (XGBoost)
        "ml_prediction": _ml_predict(ticker, horizon_days),
    }


def _ml_predict(ticker, horizon_days):
    """Run XGBoost prediction for the asset. Returns None if it fails."""
    try:
        from src.ml_models import train_asset_predictor
        result = train_asset_predictor(ticker, forward_days=min(horizon_days, 63))
        if result is None:
            return None
        return {
            "predicted_return": result["predicted_return"],
            "predicted_price": result["predicted_price"],
            "direction_accuracy": result["avg_direction_acc"],
            "top_features": result["feature_importance"][:10],
            "forward_days": result["forward_days"],
        }
    except Exception:
        return None


def _risk_flags(S0, sigma, mu, terminal_ret, max_dds):
    """Generate human-readable risk warnings based on the prediction."""
    flags = []

    if sigma > 0.35:
        flags.append(f"High volatility ({sigma:.0%} annualized) — wide prediction range")
    if sigma > 0.50:
        flags.append(f"Extreme volatility ({sigma:.0%}) — treat predictions with caution")
    if mu < 0:
        flags.append(f"Negative historical drift ({mu:.1%}) — asset has been declining")
    if float(np.mean(max_dds > 0.30)) > 0.25:
        flags.append(f"25%+ of simulated paths show >30% drawdown")
    if float(np.percentile(terminal_ret, 5)) < -0.40:
        flags.append(f"5th percentile return below -40% — significant tail risk")
    if float(np.mean(terminal_ret < 0)) > 0.45:
        flags.append(f"Near coin-flip odds of losing money ({np.mean(terminal_ret < 0):.0%} probability)")

    if not flags:
        flags.append("No major risk flags detected for this horizon")

    return flags


def fetch_rates():
    """Pull current interest rate data."""
    import yfinance as yf

    rates = {}
    for ticker, label in config.RATE_TICKERS.items():
        # skip tickers that fail to download — rates are supplementary
        data = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if data is None or len(data) == 0:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].iloc[:, 0]
        else:
            close = data["Close"]
        rates[ticker] = {
            "label": label,
            "current": float(close.iloc[-1]),
            "history": close,
            "change_1m": float(close.iloc[-1] - close.iloc[-21]) if len(close) > 21 else 0,
            "change_1y": float(close.iloc[-1] - close.iloc[-252]) if len(close) > 252 else 0,
        }

    return rates
