"""yfinance data with parquet caching + return calculations."""

import hashlib
import time

import numpy as np
import pandas as pd
import yfinance as yf

import config


def _cache_path(tickers, start, end):
    key = f"{'_'.join(sorted(tickers))}_{start}_{end or 'latest'}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return config.CACHE_DIR / f"prices_{h}.parquet"


def _cache_ok(path):
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600
    return age_h < config.CACHE_EXPIRY_HOURS


def fetch_prices(tickers=None, start=None, end=None, force_refresh=False):
    """Pull adj close, cache to parquet."""
    tickers = tickers or config.DEFAULT_TICKERS
    start = start or config.DATA_START_DATE
    end = end or config.DATA_END_DATE

    cp = _cache_path(tickers, start, end)

    if not force_refresh and _cache_ok(cp):
        return pd.read_parquet(cp)

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna()
    prices.index = pd.DatetimeIndex(prices.index)
    prices.columns = tickers

    prices.to_parquet(cp, engine="pyarrow")
    return prices


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def simple_returns(prices):
    return prices.pct_change().dropna()


def rolling_vol(returns, window=config.REGIME_LOOKBACK, annualize=True):
    rv = returns.rolling(window=window).std()
    if annualize:
        rv *= np.sqrt(config.TRADING_DAYS_YEAR)
    return rv.dropna()


def portfolio_returns(returns, weights=None):
    if weights is None:
        weights = config.DEFAULT_WEIGHTS
    w = np.array(weights)
    return returns @ w


def correlation_matrix(returns, start=None, end=None):
    sub = returns.copy()
    if start:
        sub = sub.loc[start:]
    if end:
        sub = sub.loc[:end]
    return sub.corr()
