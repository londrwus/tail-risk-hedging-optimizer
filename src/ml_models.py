"""ML models — scikit-learn + XGBoost for regime detection, return/vol forecasting."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb

import config
from src.data_loader import fetch_prices, log_returns


# ── feature engineering ──────────────────────────────────────────────────────

def build_features(prices, lookbacks=(5, 10, 21, 63, 126, 252)):
    """Build a rich feature matrix from price data.

    Features per asset and portfolio-level:
    - returns at multiple horizons
    - rolling vol at multiple windows
    - rolling skew, kurtosis
    - distance from high/low
    - momentum z-scores
    - rolling correlation (mean pairwise)
    """
    rets = log_returns(prices)
    n_assets = len(prices.columns)
    features = {}

    for lb in lookbacks:
        if len(rets) < lb + 10:
            continue

        tag = f"{lb}d"

        # per-asset trailing return
        trailing = rets.rolling(lb).sum()
        for col in rets.columns:
            features[f"ret_{col}_{tag}"] = trailing[col]

        # per-asset rolling vol
        rvol = rets.rolling(lb).std() * np.sqrt(252)
        for col in rets.columns:
            features[f"vol_{col}_{tag}"] = rvol[col]

        # per-asset rolling skew
        if lb >= 21:
            rskew = rets.rolling(lb).skew()
            for col in rets.columns:
                features[f"skew_{col}_{tag}"] = rskew[col]

        # per-asset rolling kurtosis
        if lb >= 21:
            rkurt = rets.rolling(lb).kurt()
            for col in rets.columns:
                features[f"kurt_{col}_{tag}"] = rkurt[col]

    # distance from 252-day high and low per asset
    for col in prices.columns:
        rolling_high = prices[col].rolling(252, min_periods=63).max()
        rolling_low = prices[col].rolling(252, min_periods=63).min()
        features[f"dist_high_{col}"] = (prices[col] - rolling_high) / rolling_high
        features[f"dist_low_{col}"] = (prices[col] - rolling_low) / rolling_low

    # mean pairwise correlation (rolling 63d)
    if n_assets > 1:
        corr_series = []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                pair_corr = rets.iloc[:, i].rolling(63).corr(rets.iloc[:, j])
                corr_series.append(pair_corr)
        features["mean_corr_63d"] = pd.concat(corr_series, axis=1).mean(axis=1)

    # portfolio-level features
    equal_w = np.ones(n_assets) / n_assets
    port_ret = (rets.values * equal_w).sum(axis=1)
    port_ret = pd.Series(port_ret, index=rets.index)

    for lb in [21, 63, 126]:
        if len(port_ret) < lb:
            continue
        features[f"port_vol_{lb}d"] = port_ret.rolling(lb).std() * np.sqrt(252)
        features[f"port_ret_{lb}d"] = port_ret.rolling(lb).sum()

    df = pd.DataFrame(features, index=rets.index).dropna()
    return df


# ── KMeans regime detection ──────────────────────────────────────────────────

def detect_regime_ml(returns, n_regimes=3, lookback=config.REGIME_LOOKBACK):
    """KMeans-based regime detection on vol + skew + kurtosis features.

    More sophisticated than simple vol thresholds — captures volatility
    clustering, skew shifts, and correlation regime changes.
    Returns: regime label, all regime labels for the history, cluster centers.
    """
    port_ret = returns.mean(axis=1)
    n = len(port_ret)

    # build rolling features for clustering
    windows = [21, 63]
    feat_dict = {}
    for w in windows:
        feat_dict[f"vol_{w}"] = port_ret.rolling(w).std() * np.sqrt(252)
        feat_dict[f"skew_{w}"] = port_ret.rolling(w).skew()
        feat_dict[f"kurt_{w}"] = port_ret.rolling(w).kurt()

    # mean pairwise correlation if multiple assets
    if returns.shape[1] > 1:
        corr_vals = []
        for i in range(returns.shape[1]):
            for j in range(i + 1, returns.shape[1]):
                c = returns.iloc[:, i].rolling(63).corr(returns.iloc[:, j])
                corr_vals.append(c)
        feat_dict["mean_corr"] = pd.concat(corr_vals, axis=1).mean(axis=1)

    feat_df = pd.DataFrame(feat_dict, index=port_ret.index).dropna()
    if len(feat_df) < 50:
        # not enough data — fall back to simple regime
        return _fallback_regime(port_ret, lookback)

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values)

    km = KMeans(n_clusters=n_regimes, random_state=config.RANDOM_SEED, n_init=20)
    labels = km.fit_predict(X)

    # sort regimes by average vol so 0=low, 1=mid, 2=high
    vol_col = list(feat_dict.keys()).index("vol_21")
    center_vols = km.cluster_centers_[:, vol_col]
    order = np.argsort(center_vols)
    label_map = {old: new for new, old in enumerate(order)}
    labels = np.array([label_map[l] for l in labels])

    regime_names = {0: "low", 1: "mid", 2: "high"}
    current_regime = regime_names[labels[-1]]

    # unscale centers for interpretability
    centers_raw = scaler.inverse_transform(km.cluster_centers_)
    center_info = {}
    feat_names = list(feat_dict.keys())
    for regime_idx in range(n_regimes):
        mapped_idx = order[regime_idx]
        center_info[regime_names[regime_idx]] = {
            feat_names[k]: float(centers_raw[mapped_idx, k]) for k in range(len(feat_names))
        }

    return {
        "regime": current_regime,
        "labels": pd.Series(labels, index=feat_df.index),
        "regime_names": regime_names,
        "centers": center_info,
        "n_regimes": n_regimes,
        "feature_names": feat_names,
    }


def _fallback_regime(port_ret, lookback):
    recent = port_ret.iloc[-lookback:]
    rvol = float(recent.std() * np.sqrt(252))
    if rvol < config.REGIME_THRESHOLDS["low"]:
        regime = "low"
    elif rvol > config.REGIME_THRESHOLDS["high"]:
        regime = "high"
    else:
        regime = "mid"
    return {"regime": regime, "labels": None, "centers": None, "n_regimes": 0,
            "regime_names": {0: "low", 1: "mid", 2: "high"}, "feature_names": []}


# ── XGBoost return predictor ────────────────────────────────────────────────

def train_return_predictor(prices, forward_days=21):
    """Train XGBoost to predict forward N-day portfolio return.

    Uses trailing returns, vol, skew, kurtosis, correlations as features.
    Trains on expanding window with time-series cross-validation.
    """
    features = build_features(prices)
    rets = log_returns(prices)

    # equal-weight portfolio return as target
    n_assets = len(prices.columns)
    equal_w = np.ones(n_assets) / n_assets
    port_ret = (rets.values * equal_w).sum(axis=1)
    port_ret = pd.Series(port_ret, index=rets.index)

    # forward return as target
    target = port_ret.rolling(forward_days).sum().shift(-forward_days)
    target.name = "forward_return"

    # align features and target
    common = features.index.intersection(target.dropna().index)
    X = features.loc[common]
    y = target.loc[common]

    if len(X) < 200:
        return None

    # time-series split — never look ahead
    tscv = TimeSeriesSplit(n_splits=5)
    X_arr = X.values
    y_arr = y.values

    # train final model on all data
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=config.RANDOM_SEED, verbosity=0,
    )

    # cross-val scores
    cv_scores = []
    for train_idx, test_idx in tscv.split(X_arr):
        model.fit(X_arr[train_idx], y_arr[train_idx])
        pred = model.predict(X_arr[test_idx])
        rmse = float(np.sqrt(mean_squared_error(y_arr[test_idx], pred)))
        # directional accuracy — did we predict the sign right?
        direction_acc = float(np.mean(np.sign(pred) == np.sign(y_arr[test_idx])))
        cv_scores.append({"rmse": rmse, "direction_acc": direction_acc})

    # refit on full dataset
    model.fit(X_arr, y_arr)

    # feature importance
    importances = dict(zip(X.columns, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:20]

    # current prediction
    last_features = X.iloc[-1:].values
    current_pred = float(model.predict(last_features)[0])

    return {
        "model": model,
        "scaler": None,  # XGBoost doesn't need scaling
        "feature_names": list(X.columns),
        "forward_days": forward_days,
        "current_prediction": current_pred,
        "cv_scores": cv_scores,
        "avg_direction_acc": float(np.mean([s["direction_acc"] for s in cv_scores])),
        "avg_rmse": float(np.mean([s["rmse"] for s in cv_scores])),
        "feature_importance": top_features,
        "n_samples": len(X),
    }


# ── XGBoost volatility forecaster ───────────────────────────────────────────

def train_vol_forecaster(prices, forward_days=21):
    """Train XGBoost to predict forward realized vol.

    Realized vol tends to cluster — high vol follows high vol.
    ML captures nonlinear patterns that GARCH misses.
    """
    features = build_features(prices)
    rets = log_returns(prices)

    n_assets = len(prices.columns)
    equal_w = np.ones(n_assets) / n_assets
    port_ret = (rets.values * equal_w).sum(axis=1)
    port_ret = pd.Series(port_ret, index=rets.index)

    # forward realized vol as target
    target = port_ret.rolling(forward_days).std().shift(-forward_days) * np.sqrt(252)
    target.name = "forward_vol"

    common = features.index.intersection(target.dropna().index)
    X = features.loc[common]
    y = target.loc[common]

    if len(X) < 200:
        return None

    tscv = TimeSeriesSplit(n_splits=5)
    X_arr = X.values
    y_arr = y.values

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=config.RANDOM_SEED, verbosity=0,
    )

    cv_scores = []
    for train_idx, test_idx in tscv.split(X_arr):
        model.fit(X_arr[train_idx], y_arr[train_idx])
        pred = model.predict(X_arr[test_idx])
        rmse = float(np.sqrt(mean_squared_error(y_arr[test_idx], pred)))
        # how well does it rank vol periods?
        rank_corr = float(np.corrcoef(pred, y_arr[test_idx])[0, 1])
        cv_scores.append({"rmse": rmse, "rank_corr": rank_corr})

    model.fit(X_arr, y_arr)

    importances = dict(zip(X.columns, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:20]

    current_pred = float(model.predict(X.iloc[-1:].values)[0])

    return {
        "model": model,
        "feature_names": list(X.columns),
        "forward_days": forward_days,
        "current_prediction": current_pred,
        "cv_scores": cv_scores,
        "avg_rank_corr": float(np.mean([s["rank_corr"] for s in cv_scores])),
        "avg_rmse": float(np.mean([s["rmse"] for s in cv_scores])),
        "feature_importance": top_features,
        "n_samples": len(X),
    }


# ── XGBoost crash probability classifier ────────────────────────────────────

def train_crash_predictor(prices, crash_threshold=-0.05, forward_days=21):
    """Predict probability of a >5% drawdown in next N days.

    Binary classification: did the portfolio drop more than threshold
    within the next forward_days trading days?
    """
    features = build_features(prices)
    rets = log_returns(prices)

    n_assets = len(prices.columns)
    equal_w = np.ones(n_assets) / n_assets
    port_ret = (rets.values * equal_w).sum(axis=1)
    port_ret = pd.Series(port_ret, index=rets.index)

    # target: did a crash happen in the next N days?
    target = pd.Series(0, index=port_ret.index, dtype=int)
    cum_ret = port_ret.values
    for i in range(len(cum_ret) - forward_days):
        window = cum_ret[i + 1:i + 1 + forward_days]
        running_sum = np.cumsum(window)
        if np.any(running_sum <= crash_threshold):
            target.iloc[i] = 1

    common = features.index.intersection(target.index)
    # drop the last forward_days rows (no future data)
    common = common[:-forward_days]
    X = features.loc[common]
    y = target.loc[common]

    if len(X) < 200 or y.sum() < 20:
        return None

    tscv = TimeSeriesSplit(n_splits=5)
    X_arr = X.values
    y_arr = y.values

    # handle class imbalance
    pos_weight = float((y_arr == 0).sum() / max((y_arr == 1).sum(), 1))

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=config.RANDOM_SEED, verbosity=0,
        eval_metric="logloss",
    )

    cv_scores = []
    for train_idx, test_idx in tscv.split(X_arr):
        model.fit(X_arr[train_idx], y_arr[train_idx])
        pred = model.predict(X_arr[test_idx])
        pred_proba = model.predict_proba(X_arr[test_idx])[:, 1]
        acc = float(accuracy_score(y_arr[test_idx], pred))

        # precision on crash calls — when we say crash, how often correct?
        crash_calls = pred == 1
        if crash_calls.sum() > 0:
            precision = float(y_arr[test_idx][crash_calls].mean())
        else:
            precision = 0.0

        # recall — what fraction of real crashes did we catch?
        real_crashes = y_arr[test_idx] == 1
        if real_crashes.sum() > 0:
            recall = float(pred[real_crashes].mean())
        else:
            recall = 0.0

        cv_scores.append({"accuracy": acc, "precision": precision, "recall": recall})

    model.fit(X_arr, y_arr)

    importances = dict(zip(X.columns, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:20]

    # current crash probability
    current_proba = float(model.predict_proba(X.iloc[-1:].values)[0, 1])

    return {
        "model": model,
        "feature_names": list(X.columns),
        "forward_days": forward_days,
        "crash_threshold": crash_threshold,
        "current_crash_prob": current_proba,
        "cv_scores": cv_scores,
        "avg_accuracy": float(np.mean([s["accuracy"] for s in cv_scores])),
        "avg_precision": float(np.mean([s["precision"] for s in cv_scores])),
        "avg_recall": float(np.mean([s["recall"] for s in cv_scores])),
        "feature_importance": top_features,
        "crash_rate": float(y.mean()),
        "n_samples": len(X),
    }


# ── XGBoost single-asset price direction predictor ──────────────────────────

def train_asset_predictor(ticker, forward_days=21):
    """Train XGBoost to predict forward return for a single asset.

    Builds features from the asset's own history plus macro context (SPY, TLT, GLD).
    """
    # fetch the asset + context tickers
    context_tickers = ["SPY", "TLT", "GLD"]
    all_tickers = list(set([ticker] + context_tickers))

    prices = fetch_prices(all_tickers, start="2010-01-01")
    if prices is None or prices.empty or ticker not in prices.columns:
        return None

    features = build_features(prices)
    rets = log_returns(prices)

    if ticker not in rets.columns:
        return None

    # target: forward return of the specific asset
    target = rets[ticker].rolling(forward_days).sum().shift(-forward_days)

    common = features.index.intersection(target.dropna().index)
    X = features.loc[common]
    y = target.loc[common]

    if len(X) < 200:
        return None

    tscv = TimeSeriesSplit(n_splits=5)
    X_arr = X.values
    y_arr = y.values

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=config.RANDOM_SEED, verbosity=0,
    )

    cv_scores = []
    for train_idx, test_idx in tscv.split(X_arr):
        model.fit(X_arr[train_idx], y_arr[train_idx])
        pred = model.predict(X_arr[test_idx])
        rmse = float(np.sqrt(mean_squared_error(y_arr[test_idx], pred)))
        direction_acc = float(np.mean(np.sign(pred) == np.sign(y_arr[test_idx])))
        cv_scores.append({"rmse": rmse, "direction_acc": direction_acc})

    model.fit(X_arr, y_arr)

    importances = dict(zip(X.columns, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:15]

    current_pred = float(model.predict(X.iloc[-1:].values)[0])
    current_price = float(prices[ticker].iloc[-1])
    predicted_price = current_price * np.exp(current_pred)

    return {
        "ticker": ticker,
        "model": model,
        "feature_names": list(X.columns),
        "forward_days": forward_days,
        "current_price": current_price,
        "predicted_return": current_pred,
        "predicted_price": predicted_price,
        "cv_scores": cv_scores,
        "avg_direction_acc": float(np.mean([s["direction_acc"] for s in cv_scores])),
        "avg_rmse": float(np.mean([s["rmse"] for s in cv_scores])),
        "feature_importance": top_features,
        "n_samples": len(X),
    }


# ── convenience: run all ML models at once ──────────────────────────────────

def run_all_models(prices, weights=None):
    """Train all ML models on the portfolio data. Returns dict of results."""
    if weights is None:
        weights = config.DEFAULT_WEIGHTS

    results = {}

    # regime detection
    rets = log_returns(prices)
    results["regime"] = detect_regime_ml(rets)

    # portfolio return predictor (21-day forward)
    ret_pred = train_return_predictor(prices, forward_days=21)
    if ret_pred:
        results["return_predictor"] = {k: v for k, v in ret_pred.items() if k != "model"}

    # vol forecaster
    vol_pred = train_vol_forecaster(prices, forward_days=21)
    if vol_pred:
        results["vol_forecaster"] = {k: v for k, v in vol_pred.items() if k != "model"}

    # crash predictor
    crash_pred = train_crash_predictor(prices)
    if crash_pred:
        results["crash_predictor"] = {k: v for k, v in crash_pred.items() if k != "model"}

    return results
