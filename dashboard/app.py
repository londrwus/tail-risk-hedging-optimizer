"""FastAPI backend — serves API endpoints + static files for the dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from scipy.stats import lognorm

import config
from src.data_loader import fetch_prices, log_returns, rolling_vol, portfolio_returns
from src.simulation import run_gbm, detect_regime
from src.options import all_instruments
from src.optimizer import optimize_all, all_frontiers
from src.portfolio import portfolio_stats, euler_risk_attribution, cvar, var, correlation_breakdown
from src.scenarios import run_all_historical, run_all_hypothetical
from src.backtester import run_all_backtests
from src.predictions import predict_asset, fetch_rates, TICKER_UNIVERSE, TIMEFRAMES
from src.factors import (
    current_momentum, risk_parity_weights, compare_portfolios,
    rolling_correlation, drawdown_analysis, tail_dependency,
)
from src.calibrator import (
    optimize_weights, efficient_frontier, mc_strategy_comparison,
    STRATEGY_PROFILES,
)
from src.entry_timing import analyze_entry
from src.ml_models import run_all_models, detect_regime_ml, train_asset_predictor


STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Tail Risk Hedger")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ── colors ───────────────────────────────────────────────────────────────────

C = {
    "bg": "#222", "card": "#2d2d2d", "accent": "#00bc8c",
    "red": "#e74c3c", "blue": "#3498db", "yellow": "#f39c12",
    "purple": "#9b59b6", "text": "#ecf0f1", "muted": "#95a5a6",
}
INST_COLORS = ["#00bc8c", "#3498db", "#f39c12", "#9b59b6"]
ASSET_COLORS = ["#00bc8c", "#3498db", "#f39c12", "#9b59b6",
                "#e74c3c", "#1abc9c", "#e67e22", "#2ecc71",
                "#e84393", "#fd79a8", "#636e72"]
STRAT_COLORS = {
    "max_sharpe": "#00bc8c", "min_vol": "#3498db", "min_cvar": "#e74c3c",
    "risk_parity": "#f39c12", "max_return": "#9b59b6",
    "balanced": "#1abc9c", "conservative": "#2980b9", "aggressive": "#e67e22",
}


# ── layout helpers ───────────────────────────────────────────────────────────

def _dark():
    return dict(
        template="plotly_dark",
        paper_bgcolor=C["card"], plot_bgcolor=C["card"],
        margin=dict(l=40, r=20, t=30, b=40),
        font=dict(color=C["text"]),
    )


def _fig_json(fig):
    """Convert a plotly figure to a dict that Plotly.js can consume directly."""
    d = fig.to_dict()
    return {"data": d["data"], "layout": d["layout"]}


def _clean(obj):
    """Recursively convert numpy/pandas types to JSON-safe primitives."""
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DatetimeIndex):
        return [t.isoformat() for t in obj]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


# ── state & caches ───────────────────────────────────────────────────────────

_active = {
    "tickers": list(config.DEFAULT_TICKERS),
    "weights": list(config.DEFAULT_WEIGHTS),
}
_cache = {}
_rates_cache = {}
_rp_cache = {}
_calib_cache = {}
_entry_cache = {}
_ml_cache = {}


def _clear_all():
    _cache.clear()
    _rates_cache.clear()
    _rp_cache.clear()
    _calib_cache.clear()
    _entry_cache.clear()
    _ml_cache.clear()


def _get_data():
    if "base" in _cache:
        return _cache

    tickers = _active["tickers"]
    w = np.array(_active["weights"])

    prices = fetch_prices(tickers)
    rets = log_returns(prices)
    port_ret = portfolio_returns(rets, w)

    sim = run_gbm(prices, weights=w)

    spot = float((prices.iloc[-1].values * w).sum())
    hist_vol = float(rets.tail(63).std().mean() * np.sqrt(252))
    iv = hist_vol * config.VRP_ADJ

    instruments = all_instruments(spot, iv)
    opt_results = optimize_all(sim["terminal_returns"], spot, iv)
    frontiers = all_frontiers(sim["terminal_returns"], spot, iv)
    stats = portfolio_stats(port_ret)
    attrib = euler_risk_attribution(rets, w)
    hist_scenarios = run_all_historical(rets, w)
    hypo_scenarios = run_all_hypothetical(tickers, w)
    corr_maps = correlation_breakdown(rets)
    dd = drawdown_analysis(prices, w)
    tail_dep = tail_dependency(prices, w)
    mom = current_momentum(prices)
    roll_corr = rolling_correlation(prices)

    _cache.update({
        "base": True,
        "tickers": tickers, "weights": w,
        "prices": prices, "returns": rets, "port_ret": port_ret,
        "sim": sim, "spot": spot, "iv": iv,
        "instruments": instruments, "opt_results": opt_results,
        "frontiers": frontiers, "stats": stats, "attrib": attrib,
        "hist_scenarios": hist_scenarios, "hypo_scenarios": hypo_scenarios,
        "corr_maps": corr_maps, "dd": dd, "tail_dep": tail_dep,
        "momentum": mom, "roll_corr": roll_corr,
    })
    return _cache


def _get_rates():
    if _rates_cache:
        return _rates_cache
    _rates_cache.update(fetch_rates())
    return _rates_cache


def _get_rp():
    if _rp_cache:
        return _rp_cache

    d = _get_data()
    prices = d["prices"]
    w = _active["weights"]
    comparison = compare_portfolios(prices, w)
    rp_w = risk_parity_weights(prices)

    rets = log_returns(prices)
    custom_ret = rets.values @ np.array(w)
    rp_ret = rets.values @ rp_w["weights"]

    _rp_cache.update({
        "comparison": comparison,
        "rp_weights": rp_w,
        "custom_cum": np.cumprod(1 + custom_ret) - 1,
        "rp_cum": np.cumprod(1 + rp_ret) - 1,
        "dates": rets.index,
    })
    return _rp_cache


def _get_calibration():
    if _calib_cache:
        return _calib_cache

    d = _get_data()
    prices = d["prices"]
    tickers = list(prices.columns)

    strategies = optimize_weights(prices, tickers)
    front = efficient_frontier(prices, tickers)
    mc_comp = mc_strategy_comparison(prices, strategies)

    _calib_cache.update({
        "strategies": strategies,
        "frontier": front,
        "mc_comp": mc_comp,
        "tickers": tickers,
    })
    return _calib_cache


def _get_entry():
    if _entry_cache:
        return _entry_cache

    tickers = _active["tickers"]
    w = _active["weights"]
    prices = fetch_prices(tickers)
    result = analyze_entry(prices, w, horizon_days=252, n_sims=5000)
    _entry_cache.update(result)
    return _entry_cache


# ── heatmap helper ───────────────────────────────────────────────────────────

def _corr_heatmap(corr_df, title=""):
    labels = corr_df.columns.tolist()
    vals = corr_df.values.tolist()
    text = [[f"{v:.2f}" for v in row] for row in vals]
    fig = go.Figure(go.Heatmap(
        z=vals, x=labels, y=labels, text=text, texttemplate="%{text}",
        colorscale="RdYlGn", zmin=-1, zmax=1,
    ))
    layout = _dark()
    layout["margin"] = dict(l=40, r=20, t=50, b=40)
    fig.update_layout(**layout, title=title)
    return fig


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def api_config():
    """Initial config for the frontend — ticker universe, defaults, etc."""
    return {
        "tickers": _active["tickers"],
        "weights": _active["weights"],
        "universe": {
            "SPY": "SPY", "QQQ": "QQQ", "TLT": "TLT", "GLD": "GLD",
            "SLV": "SLV", "USO": "USO", "IWM": "IWM", "EFA": "EFA", "EEM": "EEM",
            "XLE": "XLE", "XLF": "XLF", "XLK": "XLK", "XLV": "XLV", "XLC": "XLC",
            "HYG": "HYG", "IEF": "IEF", "DBC": "DBC", "DBA": "DBA",
            "AAPL": "AAPL", "MSFT": "MSFT", "AMZN": "AMZN", "GOOGL": "GOOGL",
            "NVDA": "NVDA", "TSLA": "TSLA", "META": "META", "BRK-B": "BRK-B",
            "JPM": "JPM", "V": "V", "UNH": "UNH", "JNJ": "JNJ", "PG": "PG",
            "XOM": "XOM", "CVX": "CVX",
            "ETH (Ethereum)": "ETH-USD",
            "BTC (Bitcoin)": "BTC-USD",
            "SOL (Solana)": "SOL-USD",
            "TON (Toncoin)": "TON11419-USD",
        },
        "pred_tickers": {k: v for k, v in TICKER_UNIVERSE.items()},
        "pred_timeframes": {k: v["label"] for k, v in TIMEFRAMES.items()},
        "horizon_days": config.HORIZON_DAYS,
        "n_sims": config.N_SIMULATIONS,
    }


@app.post("/api/portfolio")
def api_set_portfolio(data: dict):
    tickers = data.get("tickers", [])
    weights = data.get("weights", [])

    if not tickers or not weights:
        return {"error": "no tickers/weights"}

    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]

    _active["tickers"] = tickers
    _active["weights"] = weights
    _clear_all()

    return {"tickers": tickers, "weights": weights, "status": "ok"}


# ── Tab 1: Portfolio Overview ────────────────────────────────────────────────

@app.get("/api/tab/portfolio")
def api_portfolio():
    d = _get_data()
    s = d["stats"]
    regime = d["sim"]["regime"]

    # stats table
    stats_rows = [
        ["Ann. Return", f"{s['ann_return']:.2%}"],
        ["Ann. Volatility", f"{s['ann_vol']:.2%}"],
        ["Sharpe Ratio", f"{s['sharpe']:.2f}"],
        ["Sortino Ratio", f"{s['sortino']:.2f}"],
        ["Calmar Ratio", f"{s['calmar']:.2f}"],
        ["Max Drawdown", f"{s['max_drawdown']:.2%}"],
        ["Skewness", f"{s['skew']:.2f}"],
        ["Excess Kurtosis", f"{s['kurtosis']:.2f}"],
        ["VaR 95%", f"{s['var_95']:.2%}"],
        ["CVaR 95%", f"{s['cvar_95']:.2%}"],
        ["VaR 99%", f"{s['var_99']:.2%}"],
        ["CVaR 99%", f"{s['cvar_99']:.2%}"],
        ["Vol Regime", regime.upper()],
    ]

    # fan chart
    paths = d["sim"]["portfolio_paths"]
    days = list(range(paths.shape[1]))
    bands = {str(p): np.percentile(paths, p, axis=0).tolist() for p in [5, 25, 50, 75, 95]}

    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(x=days, y=bands["95"], mode="lines", line=dict(width=0), showlegend=False))
    fig_fan.add_trace(go.Scatter(x=days, y=bands["5"], fill="tonexty", mode="lines",
                                  line=dict(width=0), fillcolor="rgba(0,188,140,0.15)", name="5th-95th"))
    fig_fan.add_trace(go.Scatter(x=days, y=bands["75"], mode="lines", line=dict(width=0), showlegend=False))
    fig_fan.add_trace(go.Scatter(x=days, y=bands["25"], fill="tonexty", mode="lines",
                                  line=dict(width=0), fillcolor="rgba(0,188,140,0.3)", name="25th-75th"))
    fig_fan.add_trace(go.Scatter(x=days, y=bands["50"], mode="lines",
                                  line=dict(color=C["accent"], width=2), name="Median"))
    fig_fan.update_layout(**_dark(), yaxis_title="Normalized Portfolio Value", xaxis_title="Trading Days")

    # return distribution
    ret = d["sim"]["terminal_returns"].tolist()
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=ret, nbinsx=80, name="Terminal Returns",
                                     marker_color=C["accent"], opacity=0.7))
    for val, label, color in [(-s["var_95"], "VaR 95%", C["yellow"]),
                               (-s["cvar_95"], "CVaR 95%", C["red"])]:
        fig_dist.add_vline(x=val, line_dash="dash", line_color=color,
                           annotation_text=label, annotation_font_color=color)
    fig_dist.update_layout(**_dark(), xaxis_title="Terminal Return")

    # rolling vol
    rv = rolling_vol(d["returns"], window=63)
    fig_rvol = go.Figure()
    for i, col in enumerate(rv.columns):
        fig_rvol.add_trace(go.Scatter(
            x=[t.isoformat() for t in rv.index], y=rv[col].tolist(),
            mode="lines", name=col, line=dict(color=ASSET_COLORS[i % len(ASSET_COLORS)])))
    for label, thresh in config.REGIME_THRESHOLDS.items():
        fig_rvol.add_hline(y=thresh, line_dash="dot", line_color="gray",
                           annotation_text=f"{label} regime")
    fig_rvol.update_layout(**_dark(), yaxis_title="Annualized Vol")

    # cumulative return
    cum = (1 + d["port_ret"]).cumprod() - 1
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=[t.isoformat() for t in cum.index], y=cum.values.tolist(),
        mode="lines", line=dict(color=C["accent"]), name="Portfolio"))
    for name, dates in config.STRESS_SCENARIOS.items():
        fig_cum.add_vrect(x0=dates["start"], x1=dates["end"],
                          fillcolor="red", opacity=0.08, line_width=0,
                          annotation_text=name.replace("_", " "),
                          annotation_position="top left", annotation_font_size=9)
    fig_cum.update_layout(**_dark(), yaxis_title="Cumulative Return", yaxis_tickformat=".0%")

    # risk attribution
    a = d["attrib"]
    fig_attrib = go.Figure(go.Bar(
        x=a["tickers"],
        y=[a["pct_contribution"][t] for t in a["tickers"]],
        marker_color=ASSET_COLORS[:len(a["tickers"])],
    ))
    fig_attrib.update_layout(**_dark(), yaxis_title="% of Total CVaR", yaxis_tickformat=".0%")

    return _clean({
        "stats": stats_rows,
        "fan_chart": _fig_json(fig_fan),
        "return_dist": _fig_json(fig_dist),
        "rolling_vol": _fig_json(fig_rvol),
        "cum_return": _fig_json(fig_cum),
        "risk_attrib": _fig_json(fig_attrib),
    })


# ── Tab 2: Hedge Analysis ───────────────────────────────────────────────────

@app.get("/api/tab/hedge")
def api_hedge():
    d = _get_data()
    spot = d["spot"]
    instruments = d["instruments"]
    ret = d["sim"]["terminal_returns"]
    opt = d["opt_results"]

    # payoff diagram
    S_T = np.linspace(spot * 0.7, spot * 1.3, 200)
    fig_payoff = go.Figure()
    for i, inst in enumerate(instruments):
        payoff = inst["payoff"](S_T) - inst["premium"]
        fig_payoff.add_trace(go.Scatter(x=S_T.tolist(), y=payoff.tolist(), mode="lines",
                                         name=inst["name"], line=dict(color=INST_COLORS[i])))
    fig_payoff.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_payoff.add_vline(x=spot, line_dash="dot", line_color="white",
                         annotation_text="Spot", annotation_font_color="white")
    fig_payoff.update_layout(**_dark(), xaxis_title="Terminal Value", yaxis_title="Net Payoff")

    # hedged vs unhedged
    fig_hedged = go.Figure()
    fig_hedged.add_trace(go.Histogram(x=ret.tolist(), nbinsx=80, name="Unhedged",
                                       marker_color="gray", opacity=0.5, histnorm="probability density"))
    best = min(opt, key=lambda o: o["hedged_cvar"])
    best_idx = next(i for i, inst in enumerate(instruments) if inst["name"] == best["instrument"])
    inst = instruments[best_idx]
    terminal_prices = spot * (1 + ret)
    payoff = inst["payoff"](terminal_prices) / spot
    cost = inst["premium_pct"] * best["hedge_ratio"]
    hedged_ret = ret + best["hedge_ratio"] * payoff - cost
    fig_hedged.add_trace(go.Histogram(x=hedged_ret.tolist(), nbinsx=80,
                                       name=f"Hedged ({inst['name']})",
                                       marker_color=C["accent"], opacity=0.5,
                                       histnorm="probability density"))
    fig_hedged.update_layout(barmode="overlay", **_dark(),
                              xaxis_title="Terminal Return", yaxis_title="Density")

    # greeks
    greeks_rows = []
    for inst in instruments:
        for leg, g in inst["greeks"].items():
            greeks_rows.append([inst["name"], leg,
                                f"{g['delta']:.4f}", f"{g['gamma']:.6f}",
                                f"{g['vega']:.4f}", f"{g['theta']:.4f}"])

    # premiums
    prem_rows = [[i["name"], f"{i['premium']:.2f}", f"{i['premium_pct']:.4%}"]
                  for i in instruments]

    return _clean({
        "payoff": _fig_json(fig_payoff),
        "hedged_dist": _fig_json(fig_hedged),
        "greeks": {"cols": ["Instrument", "Leg", "Delta", "Gamma", "Vega", "Theta"],
                   "rows": greeks_rows},
        "premiums": {"cols": ["Instrument", "Premium ($)", "Premium (%)"],
                     "rows": prem_rows},
    })


# ── Tab 3: Hedge Frontier ───────────────────────────────────────────────────

@app.get("/api/tab/frontier")
def api_frontier():
    d = _get_data()

    fig = go.Figure()
    for i, f in enumerate(d["frontiers"]):
        costs = [p["cost_pct"] * 100 for p in f["points"]]
        reds = [p["cvar_reduction_pct"] for p in f["points"]]
        fig.add_trace(go.Scatter(x=costs, y=reds, mode="lines+markers",
                                  name=f["instrument"], line=dict(color=INST_COLORS[i]),
                                  marker=dict(size=4)))
    fig.add_vline(x=config.HEDGE_BUDGET * 100, line_dash="dash", line_color=C["red"],
                  annotation_text=f"Budget ({config.HEDGE_BUDGET:.0%})",
                  annotation_font_color=C["red"])
    fig.update_layout(**_dark(), xaxis_title="Hedge Cost (% NAV)", yaxis_title="CVaR Reduction (%)")

    opt_rows = [[o["instrument"], f"{o['hedge_ratio']:.1%}",
                 f"{o['cost_pct']:.3%}", f"{o['hedged_cvar']:.3%}",
                 f"{o['cvar_reduction_pct']:.1f}%"]
                for o in d["opt_results"]]

    return _clean({
        "frontier": _fig_json(fig),
        "opt_table": {"cols": ["Instrument", "Ratio", "Cost", "Hedged CVaR", "Reduction"],
                      "rows": opt_rows},
    })


# ── Tab 4: Stress Testing ───────────────────────────────────────────────────

@app.get("/api/tab/stress")
def api_stress():
    d = _get_data()
    sc_h = d["hist_scenarios"]
    sc_y = d["hypo_scenarios"]

    # historical bar
    names_h = [s["name"].replace("_", " ") for s in sc_h]
    rets_h = [float(s["return"]) * 100 for s in sc_h]
    colors_h = [C["red"] if r < 0 else C["accent"] for r in rets_h]
    fig_hist = go.Figure(go.Bar(x=names_h, y=rets_h, marker_color=colors_h))
    fig_hist.update_layout(**_dark(), yaxis_title="Return (%)")

    # hypothetical bar
    names_y = [s["name"].replace("_", " ") for s in sc_y]
    rets_y = [float(s["return"]) * 100 for s in sc_y]
    sev = [float(s.get("severity", 0.5)) for s in sc_y]
    colors_y = [C["red"] if r < 0 else C["accent"] for r in rets_y]
    fig_hypo = go.Figure(go.Bar(x=names_y, y=rets_y, marker_color=colors_y,
                                 text=[f"Sev: {s:.0%}" for s in sev], textposition="outside"))
    layout_hypo = _dark()
    layout_hypo["margin"] = dict(l=40, r=20, t=30, b=80)
    fig_hypo.update_layout(**layout_hypo, yaxis_title="Return (%)", xaxis_tickangle=-30)

    # detail table
    all_sc = sc_h + sc_y
    detail_rows = [[s["name"].replace("_", " "),
                    "Hypothetical" if s.get("hypothetical") else s.get("period", ""),
                    f"{s['return']:.2%}", f"{s['max_drawdown']:.2%}",
                    f"{s['cvar_95']:.2%}"]
                   for s in all_sc]

    # correlation heatmaps
    fig_corr_n = _corr_heatmap(d["corr_maps"]["full_period"], "Full Period Correlation")
    cm = d["corr_maps"]
    stress_key = "GFC_2008" if "GFC_2008" in cm else next((k for k in cm if k != "full_period"), None)
    fig_corr_s = _corr_heatmap(cm[stress_key], f"Stress Correlation ({stress_key.replace('_', ' ')})") if stress_key else go.Figure()

    return _clean({
        "stress_hist": _fig_json(fig_hist),
        "stress_hypo": _fig_json(fig_hypo),
        "detail": {"cols": ["Scenario", "Type", "Return", "Max DD", "CVaR 95%"],
                   "rows": detail_rows},
        "corr_normal": _fig_json(fig_corr_n),
        "corr_stress": _fig_json(fig_corr_s),
    })


# ── Tab 5: Predictions ──────────────────────────────────────────────────────

@app.get("/api/tab/predictions")
def api_predictions(ticker: str = "SPY", timeframe: str = "1y", n_sims: int = 5000):
    if ticker not in TICKER_UNIVERSE or timeframe not in TIMEFRAMES:
        return {"error": "invalid ticker or timeframe"}

    horizon = TIMEFRAMES[timeframe]["days"]
    tf_label = TIMEFRAMES[timeframe]["label"]

    p = predict_asset(ticker, horizon_days=horizon, n_sims=n_sims)
    if not p:
        return {"error": f"no data for {ticker}"}

    dates = [d.isoformat() for d in p["dates"]]
    bands = p["bands"]

    # fan chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dates, y=bands[95], mode="lines", line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=dates, y=bands[5], fill="tonexty", mode="lines",
                               line=dict(width=0), fillcolor="rgba(52,152,219,0.12)", name="5th-95th"))
    fig1.add_trace(go.Scatter(x=dates, y=bands[90], mode="lines", line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=dates, y=bands[10], fill="tonexty", mode="lines",
                               line=dict(width=0), fillcolor="rgba(52,152,219,0.18)", name="10th-90th"))
    fig1.add_trace(go.Scatter(x=dates, y=bands[75], mode="lines", line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=dates, y=bands[25], fill="tonexty", mode="lines",
                               line=dict(width=0), fillcolor="rgba(52,152,219,0.3)", name="25th-75th"))
    fig1.add_trace(go.Scatter(x=dates, y=bands[50], mode="lines",
                               line=dict(color=C["blue"], width=2), name="Median"))
    fig1.add_hline(y=p["current_price"], line_dash="dot", line_color=C["yellow"],
                   annotation_text=f"Current ${p['current_price']:.2f}")
    fig1.add_hline(y=p["pct_95_price"], line_dash="dot", line_color=C["accent"],
                   annotation_text=f"Bull ${p['pct_95_price']:.0f}", annotation_position="right")
    fig1.add_hline(y=p["pct_5_price"], line_dash="dot", line_color=C["red"],
                   annotation_text=f"Bear ${p['pct_5_price']:.0f}", annotation_position="right")
    fig1.update_layout(**_dark(), title=f"{p['label']} — {tf_label} Forward ({n_sims:,} sims)",
                       yaxis_title="Price ($)")

    # terminal distribution (lognormal approx)
    price_range = np.linspace(p["pct_5_price"] * 0.8, p["pct_95_price"] * 1.2, 200)
    s_param = p["sigma"] * np.sqrt(horizon / 252)
    scale = p["current_price"] * np.exp(p["mu"] * horizon / 252)
    pdf_vals = lognorm.pdf(price_range, s_param, scale=scale)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=price_range.tolist(), y=pdf_vals.tolist(), mode="lines",
                               fill="tozeroy", fillcolor="rgba(52,152,219,0.3)",
                               line=dict(color=C["blue"]), name="Distribution"))
    fig2.add_vline(x=p["current_price"], line_dash="dash", line_color=C["yellow"],
                   annotation_text="Current")
    fig2.add_vline(x=p["terminal_median"], line_dash="dash", line_color=C["accent"],
                   annotation_text="Median")
    fig2.update_layout(**_dark(), xaxis_title="Terminal Price ($)", yaxis_title="Density",
                       showlegend=False)

    # stress indicator
    fig3 = go.Figure()
    fig3.add_trace(go.Indicator(
        mode="number+delta", value=p["stress_high_vol_median"],
        delta={"reference": p["current_price"], "relative": True, "valueformat": ".1%"},
        title={"text": "Median (2x Vol)"}, number={"prefix": "$", "valueformat": ".0f"},
        domain={"x": [0, 1], "y": [0.55, 1]},
    ))
    fig3.add_trace(go.Indicator(
        mode="number", value=p["stress_high_vol_prob_down"] * 100,
        title={"text": "P(Loss) under stress"}, number={"suffix": "%", "valueformat": ".0f"},
        domain={"x": [0, 0.45], "y": [0, 0.45]},
    ))
    fig3.add_trace(go.Indicator(
        mode="number", value=p["stress_high_vol_5pct"],
        title={"text": "5th pct (2x Vol)"}, number={"prefix": "$", "valueformat": ".0f"},
        domain={"x": [0.55, 1], "y": [0, 0.45]},
    ))
    fig3.update_layout(**_dark())

    # ML prediction row for the stats table
    ml_pred = p.get("ml_prediction")
    ml_rows = []
    if ml_pred:
        ml_rows = [
            ["── ML (XGBoost) ──", ""],
            ["ML Predicted Price", f"${ml_pred['predicted_price']:.2f}"],
            ["ML Predicted Return", f"{ml_pred['predicted_return']:.1%}"],
            ["ML Direction Accuracy", f"{ml_pred['direction_accuracy']:.1%}"],
            ["ML Forward Window", f"{ml_pred['forward_days']}d"],
        ]

    stats_rows = [
        ["Expected Price (MC)", f"${p['terminal_mean']:.2f}"],
        ["Median Price (MC)", f"${p['terminal_median']:.2f}"],
        ["Expected Return (MC)", f"{p['expected_return']:.1%}"],
    ] + ml_rows + [
        ["5th Pct Return", f"{p['pct_5_return']:.1%}"],
        ["95th Pct Return", f"{p['pct_95_return']:.1%}"],
        ["P(price > current)", f"{p['prob_up']:.1%}"],
        ["P(loss > 10%)", f"{p['prob_down_10']:.1%}"],
        ["P(loss > 20%)", f"{p['prob_down_20']:.1%}"],
        ["P(gain > 20%)", f"{p['prob_up_20']:.1%}"],
        ["P(gain > 50%)", f"{p['prob_up_50']:.1%}"],
        ["VaR 95%", f"{p['var_95']:.1%}"],
        ["CVaR 95%", f"{p['cvar_95']:.1%}"],
        ["Avg Max Drawdown", f"{p['avg_max_drawdown']:.1%}"],
        ["Worst Max Drawdown", f"{p['worst_max_drawdown']:.1%}"],
        ["P(DD > 20%)", f"{p['prob_drawdown_gt_20']:.1%}"],
        ["Ann. Drift (hist)", f"{p['mu']:.2%}"],
        ["Ann. Vol (hist)", f"{p['sigma']:.2%}"],
    ]

    ml_level_row = []
    if ml_pred:
        ml_level_row = [[f"XGBoost Predicted ({ml_pred['forward_days']}d)",
                          f"${ml_pred['predicted_price']:.2f}"]]

    level_rows = [
        ["Current Price", f"${p['current_price']:.2f}"],
        ["52-Week High", f"${p['hist_52w_high']:.2f}"],
        ["52-Week Low", f"${p['hist_52w_low']:.2f}"],
        ["% From 52w High", f"{p['pct_from_high']:.1%}"],
    ] + ml_level_row + [
        [f"Bull Case (95th, {tf_label})", f"${p['pct_95_price']:.2f}"],
        [f"Base Case (50th, {tf_label})", f"${p['terminal_median']:.2f}"],
        [f"Bear Case (5th, {tf_label})", f"${p['pct_5_price']:.2f}"],
        [f"Stress (2x Vol, median)", f"${p['stress_high_vol_median']:.2f}"],
        [f"Stress (2x Vol, 5th pct)", f"${p['stress_high_vol_5pct']:.2f}"],
    ]

    # rates
    rates = _get_rates()
    rates_rows = []
    fig_rates = go.Figure()
    for i, (tk, info) in enumerate(rates.items()):
        rates_rows.append([info["label"], f"{info['current']:.2f}%",
                           f"{info['change_1m']:+.2f}%", f"{info['change_1y']:+.2f}%"])
        fig_rates.add_trace(go.Scatter(
            x=[t.isoformat() for t in info["history"].index],
            y=info["history"].values.tolist(),
            mode="lines", name=info["label"],
            line=dict(color=ASSET_COLORS[i % len(ASSET_COLORS)])))
    fig_rates.update_layout(**_dark(), yaxis_title="Yield (%)", title="US Treasury Yields")

    return _clean({
        "fan_chart": _fig_json(fig1),
        "terminal_dist": _fig_json(fig2),
        "stress": _fig_json(fig3),
        "stats": {"cols": ["Metric", "Value"], "rows": stats_rows},
        "risk_flags": p["risk_flags"],
        "price_levels": {"cols": ["Level", "Price"], "rows": level_rows},
        "current_info": {
            "label": p["label"], "ticker": ticker,
            "price": p["current_price"],
            "high_52w": p["hist_52w_high"],
            "low_52w": p["hist_52w_low"],
            "pct_from_high": p["pct_from_high"],
        },
        "rates": {"cols": ["Rate", "Current (%)", "1M Change", "1Y Change"],
                  "rows": rates_rows},
        "rates_chart": _fig_json(fig_rates),
    })


# ── Tab 6: Factor & Drawdown ────────────────────────────────────────────────

@app.get("/api/tab/factors")
def api_factors():
    d = _get_data()
    mom = d["momentum"]
    td = d["tail_dep"]
    roll = d["roll_corr"]
    dd = d["dd"]

    # momentum
    fig_mom = go.Figure()
    windows = sorted(mom.keys())
    tickers = list(mom[windows[0]].keys()) if windows else []
    for i, t in enumerate(tickers):
        vals = [mom[w].get(t, 0) for w in windows]
        fig_mom.add_trace(go.Bar(
            x=[f"{w}d" for w in windows], y=vals,
            name=t, marker_color=ASSET_COLORS[i % len(ASSET_COLORS)]))
    fig_mom.update_layout(barmode="group", **_dark(),
                           xaxis_title="Lookback Window", yaxis_title="Momentum Z-Score")

    # tail dependency
    td_tickers = list(td.keys())
    tail_ret = [td[t]["avg_return_in_tail"] * 100 for t in td_tickers]
    normal_ret = [td[t]["avg_return_normal"] * 100 for t in td_tickers]
    tail_beta = [td[t]["tail_beta"] for t in td_tickers]

    fig_tail = go.Figure()
    fig_tail.add_trace(go.Bar(x=td_tickers, y=normal_ret, name="Normal Days", marker_color=C["accent"]))
    fig_tail.add_trace(go.Bar(x=td_tickers, y=tail_ret, name="Tail Days (worst 5%)", marker_color=C["red"]))
    for i, t in enumerate(td_tickers):
        fig_tail.add_annotation(x=t, y=max(tail_ret[i], normal_ret[i]) + 0.05,
                                text=f"β={tail_beta[i]:.2f}", showarrow=False,
                                font=dict(size=10, color=C["muted"]))
    fig_tail.update_layout(barmode="group", **_dark(), yaxis_title="Avg Daily Return (%)")

    # rolling correlation
    fig_rcorr = go.Figure()
    for i, p in enumerate(roll):
        fig_rcorr.add_trace(go.Scatter(
            x=[t.isoformat() for t in p["series"].index],
            y=p["series"].values.tolist(),
            mode="lines", name=p["pair"],
            line=dict(color=ASSET_COLORS[i % len(ASSET_COLORS)], width=1)))
    fig_rcorr.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_rcorr.update_layout(**_dark(), yaxis_title="63-Day Rolling Correlation")

    # drawdown
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=[t.isoformat() for t in dd["dates"]],
        y=(-dd["drawdown_series"] * 100).tolist(),
        mode="lines", fill="tozeroy", fillcolor="rgba(231,76,60,0.3)",
        line=dict(color=C["red"], width=1), name="Drawdown"))
    fig_dd.update_layout(**_dark(), yaxis_title="Drawdown (%)", yaxis_tickformat=".1f")

    dd_rows = [[e["start_date"], e["trough_date"], e["end_date"],
                f"{e['max_drawdown']:.1%}", f"{e['duration_days']}d",
                f"{e['recovery_days']}d" if e["recovery_days"] else "ongoing"]
               for e in dd["episodes"][:10]]

    return _clean({
        "momentum": _fig_json(fig_mom),
        "tail_dep": _fig_json(fig_tail),
        "rolling_corr": _fig_json(fig_rcorr),
        "drawdown": _fig_json(fig_dd),
        "dd_table": {"cols": ["Start", "Trough", "End", "Max DD", "Duration", "Recovery"],
                     "rows": dd_rows},
    })


# ── Tab 7: Risk Parity ──────────────────────────────────────────────────────

@app.get("/api/tab/riskparity")
def api_riskparity():
    rp = _get_rp()
    comp = rp["comparison"]
    tickers = list(comp["custom"]["weights"].keys())

    custom_w = [comp["custom"]["weights"][t] for t in tickers]
    rp_w = [comp["risk_parity"]["weights"].get(t, 0) for t in tickers]

    fig_w = go.Figure()
    fig_w.add_trace(go.Bar(x=tickers, y=[w * 100 for w in custom_w],
                            name="Current", marker_color=C["blue"]))
    fig_w.add_trace(go.Bar(x=tickers, y=[w * 100 for w in rp_w],
                            name="Risk Parity", marker_color=C["accent"]))
    fig_w.update_layout(barmode="group", **_dark(), yaxis_title="Weight (%)")

    cs = comp["custom"]["stats"]
    rs = comp["risk_parity"]["stats"]
    metrics = ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "cvar_95"]
    labels = ["Ann. Return", "Ann. Vol", "Sharpe", "Sortino", "Max DD", "CVaR 95%"]
    fmt = [".2%", ".2%", ".2f", ".2f", ".2%", ".4f"]
    comp_rows = [[label, f"{cs[key]:{f}}", f"{rs[key]:{f}}"]
                 for label, key, f in zip(labels, metrics, fmt)]

    fig_cum = go.Figure()
    dates = [t.isoformat() for t in rp["dates"]]
    fig_cum.add_trace(go.Scatter(x=dates, y=rp["custom_cum"].tolist(),
                                  mode="lines", name="Current", line=dict(color=C["blue"])))
    fig_cum.add_trace(go.Scatter(x=dates, y=rp["rp_cum"].tolist(),
                                  mode="lines", name="Risk Parity", line=dict(color=C["accent"])))
    fig_cum.update_layout(**_dark(), yaxis_title="Cumulative Return", yaxis_tickformat=".0%")

    return _clean({
        "weights": _fig_json(fig_w),
        "comparison": {"cols": ["Metric", "Current Portfolio", "Risk Parity"],
                       "rows": comp_rows},
        "cum_return": _fig_json(fig_cum),
    })


# ── Tab 8: Portfolio Calibration ─────────────────────────────────────────────

@app.get("/api/tab/calibration")
def api_calibration():
    cal = _get_calibration()
    strategies = cal["strategies"]
    front = cal["frontier"]
    mc_comp = cal["mc_comp"]
    tickers = cal["tickers"]

    # weight comparison
    fig_w = go.Figure()
    fig_w.add_trace(go.Bar(
        x=tickers, y=[w * 100 for w in _active["weights"]],
        name="Current", marker_color="#555", opacity=0.6))
    for sname, strat in strategies.items():
        weights = [strat["weights"].get(t, 0) * 100 for t in tickers]
        fig_w.add_trace(go.Bar(
            x=tickers, y=weights, name=strat["label"],
            marker_color=STRAT_COLORS.get(sname, C["text"])))
    fig_w.update_layout(barmode="group", **_dark(), yaxis_title="Weight (%)",
                         legend=dict(font=dict(size=10)))

    # strategy summary
    strat_rows = []
    for sname, strat in strategies.items():
        s = strat["stats"]
        strat_rows.append([strat["label"], strat["style"], f"{s['sharpe']:.2f}",
                           f"{s['ann_vol']:.1%}", f"{s['cvar_95']:.3%}",
                           f"{s['max_drawdown']:.1%}"])

    # efficient frontier
    fig_ef = go.Figure()
    if front:
        vols = [p["vol"] * 100 for p in front]
        rets = [p["ret"] * 100 for p in front]
        fig_ef.add_trace(go.Scatter(x=vols, y=rets, mode="lines",
                                     line=dict(color=C["accent"], width=2), name="Efficient Frontier"))
    for sname, strat in strategies.items():
        s = strat["stats"]
        fig_ef.add_trace(go.Scatter(
            x=[s["ann_vol"] * 100], y=[s["ann_return"] * 100],
            mode="markers+text",
            marker=dict(size=12, color=STRAT_COLORS.get(sname, C["text"])),
            text=[strat["label"]], textposition="top center",
            textfont=dict(size=9, color=C["text"]), name=strat["label"]))
    cs = portfolio_stats(_get_data()["port_ret"])
    fig_ef.add_trace(go.Scatter(
        x=[cs["ann_vol"] * 100], y=[cs["ann_return"] * 100],
        mode="markers+text", marker=dict(size=14, color="white", symbol="star"),
        text=["Current"], textposition="bottom center",
        textfont=dict(size=10, color="white"), name="Current Portfolio"))
    fig_ef.update_layout(**_dark(), xaxis_title="Annualized Vol (%)",
                          yaxis_title="Annualized Return (%)", showlegend=False)

    # MC box plot
    order = ["conservative", "min_vol", "min_cvar", "risk_parity",
             "balanced", "max_sharpe", "aggressive", "max_return"]
    fig_mc = go.Figure()
    mc_rows = []
    for sname in order:
        if sname not in mc_comp:
            continue
        mc = mc_comp[sname]
        fig_mc.add_trace(go.Box(
            y=(mc["terminal_returns"] * 100).tolist(), name=mc["label"],
            marker_color=STRAT_COLORS.get(sname, C["text"]), boxpoints=False))
        mc_rows.append([mc["label"], f"{mc['mean_ret']:.2%}", f"{mc['median_ret']:.2%}",
                        f"{mc['var_95']:.2%}", f"{mc['cvar_95']:.2%}",
                        f"{mc['prob_loss']:.0%}", f"{mc['pct_5']:.2%}", f"{mc['pct_95']:.2%}"])
    fig_mc.update_layout(**_dark(), yaxis_title="Terminal Return (%)", showlegend=False)

    # detail table
    detail_rows = []
    for sname, strat in strategies.items():
        s = strat["stats"]
        top = sorted(strat["weights"].items(), key=lambda x: -x[1])
        top_str = ", ".join([f"{t}:{w:.0%}" for t, w in top if w > 0.01])
        detail_rows.append([strat["label"], top_str, f"{s['ann_return']:.2%}",
                            f"{s['ann_vol']:.2%}", f"{s['sharpe']:.2f}",
                            f"{s['sortino']:.2f}", f"{s['calmar']:.2f}", f"{s['skew']:.2f}"])

    return _clean({
        "weights": _fig_json(fig_w),
        "strategy_table": {"cols": ["Strategy", "Style", "Sharpe", "Ann.Vol", "CVaR 95%", "Max DD"],
                           "rows": strat_rows},
        "eff_frontier": _fig_json(fig_ef),
        "mc_box": _fig_json(fig_mc),
        "mc_table": {"cols": ["Strategy", "Mean Ret", "Median", "VaR 95%", "CVaR 95%", "P(Loss)", "5th pct", "95th pct"],
                     "rows": mc_rows},
        "detail_table": {"cols": ["Strategy", "Weights", "Return", "Vol", "Sharpe", "Sortino", "Calmar", "Skew"],
                         "rows": detail_rows},
    })


# ── Tab 9: Entry Timing ─────────────────────────────────────────────────────

@app.get("/api/tab/entry")
def api_entry():
    e = _get_entry()
    timing = e["timing"]
    lump = e["lump_sum"]
    dca = e["dca_results"]
    dip_probs = e["dip_probabilities"]
    level_plan = e["level_plan"]
    spot = e["spot"]

    # dip probability chart
    dip_pcts = sorted(dip_probs.keys())
    fig_dip = go.Figure()
    fig_dip.add_trace(go.Bar(
        x=[f"-{d:.0%}" for d in dip_pcts],
        y=[dip_probs[d] * 100 for d in dip_pcts],
        marker_color=[C["accent"] if dip_probs[d] < 0.4 else
                      C["yellow"] if dip_probs[d] < 0.65 else C["red"]
                      for d in dip_pcts],
        text=[f"{dip_probs[d]:.0%}" for d in dip_pcts],
        textposition="outside"))
    fig_dip.update_layout(**_dark(), xaxis_title="Dip Level (from current)",
                           yaxis_title="Probability (%)", yaxis_range=[0, 105], showlegend=False)

    # DCA vs lump sum
    fig_dca = go.Figure()
    fig_dca.add_trace(go.Histogram(
        x=lump["terminal_returns"].tolist(), nbinsx=60, name="Lump Sum",
        marker_color="gray", opacity=0.4, histnorm="probability density"))
    dca_colors = [C["accent"], C["blue"], C["yellow"], C["purple"]]
    for i, (name, res) in enumerate(dca.items()):
        fig_dca.add_trace(go.Histogram(
            x=res["terminal_returns"].tolist(), nbinsx=60,
            name=res["label"], marker_color=dca_colors[i % len(dca_colors)],
            opacity=0.35, histnorm="probability density"))
    fig_dca.update_layout(barmode="overlay", **_dark(),
                           xaxis_title="Terminal Return (1Y)", yaxis_title="Density")

    # DCA stats
    recommended = timing["recommended_strategy"]
    dca_rows = [["Lump Sum (100% now)", f"{lump['mean_return']:.2%}", f"{lump['median_return']:.2%}",
                 f"{lump['var_95']:.2%}", f"{lump['cvar_95']:.2%}", f"{lump['prob_loss']:.1%}", "—"]]
    for name, res in dca.items():
        marker = " ★" if name == recommended else ""
        dca_rows.append([res["label"] + marker, f"{res['mean_return']:.2%}",
                         f"{res['median_return']:.2%}", f"{res['var_95']:.2%}",
                         f"{res['cvar_95']:.2%}", f"{res['prob_loss']:.1%}",
                         f"{res['prob_beat_lump']:.1%}"])

    # price level plan
    level_rows = [[f"${lev['price']:,.2f}",
                   f"-{lev['dip_pct']:.1%}" if lev['dip_pct'] > 0.001 else "Current",
                   f"{lev['allocation']:.0%}", f"{lev['cumulative']:.0%}",
                   lev["rationale"]]
                  for lev in level_plan]

    # DCA schedule chart
    fig_sched = go.Figure()
    colors = [C["accent"], C["blue"], C["yellow"], C["purple"]]
    for i, (name, res) in enumerate(dca.items()):
        days = [s[0] for s in res["schedule"]]
        pcts = [s[1] * 100 for s in res["schedule"]]
        cum = np.cumsum(pcts).tolist()
        fig_sched.add_trace(go.Scatter(
            x=days, y=cum, mode="lines+markers", name=res["label"],
            line=dict(color=colors[i % len(colors)]), marker=dict(size=8)))
    fig_sched.add_hline(y=100, line_dash="dash", line_color=C["muted"],
                         annotation_text="100% deployed")
    fig_sched.update_layout(**_dark(), xaxis_title="Trading Day",
                             yaxis_title="% Capital Deployed", yaxis_range=[0, 110])

    return _clean({
        "timing": {
            "score": timing["score"],
            "verdict": timing["verdict"],
            "recommended": recommended,
            "rec_label": dca[recommended]["label"] if recommended in dca else recommended,
            "factors": timing["factors"],
            "spot": spot,
            "sigma": e["sigma"],
            "mu": e["mu"],
        },
        "dip_probs": _fig_json(fig_dip),
        "dca_comparison": _fig_json(fig_dca),
        "dca_table": {"cols": ["Strategy", "Mean Return", "Median Return", "VaR 95%",
                               "CVaR 95%", "P(Loss)", "P(Beat Lump)"],
                      "rows": dca_rows},
        "level_plan": {"cols": ["Price", "Dip", "Deploy", "Cumulative", "Rationale"],
                       "rows": level_rows},
        "schedules": _fig_json(fig_sched),
    })


# ── Tab 10: ML Models ─────────────────────────────────────────────────────────

def _get_ml():
    if _ml_cache:
        return _ml_cache

    d = _get_data()
    prices = d["prices"]
    ml = run_all_models(prices, _active["weights"])
    _ml_cache.update(ml)
    return _ml_cache


@app.get("/api/tab/ml")
def api_ml():
    ml = _get_ml()
    d = _get_data()

    # ── regime detection chart ──
    regime_info = ml.get("regime", {})
    fig_regime = go.Figure()

    if regime_info.get("labels") is not None:
        labels = regime_info["labels"]
        # rgba fill colors for regime bands
        fill_map = {
            "low": "rgba(0,188,140,0.2)",
            "mid": "rgba(243,156,18,0.2)",
            "high": "rgba(231,76,60,0.2)",
        }
        line_map = {"low": C["accent"], "mid": C["yellow"], "high": C["red"]}
        regime_names = regime_info["regime_names"]

        dates = [t.isoformat() for t in labels.index]
        regime_vals = labels.values

        for regime_idx, rname in regime_names.items():
            mask = regime_vals == regime_idx
            if not mask.any():
                continue
            y_vals = [1 if m else None for m in mask]
            fig_regime.add_trace(go.Scatter(
                x=dates, y=y_vals, mode="lines",
                fill="tozeroy", fillcolor=fill_map.get(rname, "rgba(149,165,166,0.2)"),
                line=dict(color=line_map.get(rname, C["muted"]), width=1),
                name=f"{rname.upper()} regime", connectgaps=False))

        fig_regime.update_layout(**_dark(), yaxis_visible=False,
                                  title=f"KMeans Regime Detection — Current: {regime_info['regime'].upper()}")
    else:
        fig_regime.update_layout(**_dark(), title="Regime detection (insufficient data)")

    # regime cluster centers table
    centers = regime_info.get("centers", {})
    center_rows = []
    if centers:
        for rname, feats in centers.items():
            row = [rname.upper()]
            for k, v in feats.items():
                row.append(f"{v:.3f}")
            center_rows.append(row)
    center_cols = ["Regime"] + list(list(centers.values())[0].keys()) if centers else ["Regime"]

    # ── return predictor ──
    ret_pred = ml.get("return_predictor", {})
    fig_ret_imp = go.Figure()
    if ret_pred:
        feats = ret_pred.get("feature_importance", [])
        if feats:
            names = [f[0] for f in feats[:15]][::-1]
            vals = [f[1] for f in feats[:15]][::-1]
            fig_ret_imp.add_trace(go.Bar(
                y=names, x=vals, orientation="h",
                marker_color=C["accent"]))
        layout = _dark()
        layout["margin"] = dict(l=180, r=20, t=40, b=40)
        fig_ret_imp.update_layout(**layout, title="Return Predictor — Feature Importance",
                                   xaxis_title="Importance")

    ret_pred_rows = []
    if ret_pred:
        ret_pred_rows = [
            ["Forward Window", f"{ret_pred['forward_days']} days"],
            ["Current Prediction", f"{ret_pred['current_prediction']:.2%}"],
            ["Direction Accuracy (CV)", f"{ret_pred['avg_direction_acc']:.1%}"],
            ["RMSE (CV)", f"{ret_pred['avg_rmse']:.4f}"],
            ["Training Samples", f"{ret_pred['n_samples']:,}"],
        ]

    # ── vol forecaster ──
    vol_pred = ml.get("vol_forecaster", {})
    fig_vol_imp = go.Figure()
    if vol_pred:
        feats = vol_pred.get("feature_importance", [])
        if feats:
            names = [f[0] for f in feats[:15]][::-1]
            vals = [f[1] for f in feats[:15]][::-1]
            fig_vol_imp.add_trace(go.Bar(
                y=names, x=vals, orientation="h",
                marker_color=C["blue"]))
        layout = _dark()
        layout["margin"] = dict(l=180, r=20, t=40, b=40)
        fig_vol_imp.update_layout(**layout, title="Vol Forecaster — Feature Importance",
                                   xaxis_title="Importance")

    vol_pred_rows = []
    if vol_pred:
        vol_pred_rows = [
            ["Forward Window", f"{vol_pred['forward_days']} days"],
            ["Predicted Vol", f"{vol_pred['current_prediction']:.1%}"],
            ["Rank Correlation (CV)", f"{vol_pred['avg_rank_corr']:.3f}"],
            ["RMSE (CV)", f"{vol_pred['avg_rmse']:.4f}"],
            ["Training Samples", f"{vol_pred['n_samples']:,}"],
        ]

    # ── crash predictor ──
    crash_pred = ml.get("crash_predictor", {})
    fig_crash = go.Figure()
    if crash_pred:
        prob = crash_pred["current_crash_prob"]
        color = C["red"] if prob > 0.5 else C["yellow"] if prob > 0.3 else C["accent"]
        fig_crash.add_trace(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Crash Probability (next 21 days)"},
            number={"suffix": "%", "valueformat": ".0f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "rgba(0,188,140,0.2)"},
                    {"range": [30, 60], "color": "rgba(243,156,18,0.2)"},
                    {"range": [60, 100], "color": "rgba(231,76,60,0.2)"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": prob * 100},
            }
        ))
        fig_crash.update_layout(**_dark())

    crash_rows = []
    if crash_pred:
        crash_rows = [
            ["Crash Threshold", f"{crash_pred['crash_threshold']:.0%}"],
            ["Current Probability", f"{crash_pred['current_crash_prob']:.1%}"],
            ["Historical Crash Rate", f"{crash_pred['crash_rate']:.1%}"],
            ["Accuracy (CV)", f"{crash_pred['avg_accuracy']:.1%}"],
            ["Precision (CV)", f"{crash_pred['avg_precision']:.1%}"],
            ["Recall (CV)", f"{crash_pred['avg_recall']:.1%}"],
            ["Training Samples", f"{crash_pred['n_samples']:,}"],
        ]

    # ── cross-validation detail chart ──
    fig_cv = go.Figure()
    if ret_pred and ret_pred.get("cv_scores"):
        folds = list(range(1, len(ret_pred["cv_scores"]) + 1))
        fig_cv.add_trace(go.Bar(
            x=[f"Fold {f}" for f in folds],
            y=[s["direction_acc"] * 100 for s in ret_pred["cv_scores"]],
            name="Return Direction Acc", marker_color=C["accent"]))
    if crash_pred and crash_pred.get("cv_scores"):
        folds = list(range(1, len(crash_pred["cv_scores"]) + 1))
        fig_cv.add_trace(go.Bar(
            x=[f"Fold {f}" for f in folds],
            y=[s["accuracy"] * 100 for s in crash_pred["cv_scores"]],
            name="Crash Accuracy", marker_color=C["red"]))
    fig_cv.update_layout(barmode="group", **_dark(),
                          title="Time-Series Cross-Validation Performance",
                          yaxis_title="Accuracy (%)")

    # ── summary card data ──
    summary = {
        "regime": regime_info.get("regime", "unknown").upper(),
        "return_pred": ret_pred.get("current_prediction"),
        "return_dir_acc": ret_pred.get("avg_direction_acc"),
        "vol_pred": vol_pred.get("current_prediction"),
        "crash_prob": crash_pred.get("current_crash_prob"),
    }

    return _clean({
        "summary": summary,
        "regime_chart": _fig_json(fig_regime),
        "regime_centers": {"cols": center_cols, "rows": center_rows},
        "return_importance": _fig_json(fig_ret_imp),
        "return_stats": {"cols": ["Metric", "Value"], "rows": ret_pred_rows},
        "vol_importance": _fig_json(fig_vol_imp),
        "vol_stats": {"cols": ["Metric", "Value"], "rows": vol_pred_rows},
        "crash_gauge": _fig_json(fig_crash),
        "crash_stats": {"cols": ["Metric", "Value"], "rows": crash_rows},
        "cv_chart": _fig_json(fig_cv),
    })


@app.get("/api/ml/predict/{ticker}")
def api_ml_predict(ticker: str, forward_days: int = 21):
    """XGBoost prediction for a specific asset."""
    result = train_asset_predictor(ticker, forward_days=forward_days)
    if result is None:
        return {"error": f"Could not train model for {ticker}"}

    return _clean({
        "ticker": ticker,
        "current_price": result["current_price"],
        "predicted_price": result["predicted_price"],
        "predicted_return": result["predicted_return"],
        "direction_accuracy": result["avg_direction_acc"],
        "rmse": result["avg_rmse"],
        "forward_days": result["forward_days"],
        "top_features": result["feature_importance"][:10],
        "n_samples": result["n_samples"],
    })


# ── FAQ ──────────────────────────────────────────────────────────────────────

@app.get("/api/faq")
def api_faq():
    # send the FAQ content for the frontend modals
    return {
        "fan-chart": {"title": "Monte Carlo Fan Chart", "body": "Simulated future portfolio paths using correlated GBM with Cholesky decomposition. Dark band = 25-75th pct, light band = 5-95th pct. Wider = more uncertainty."},
        "return-dist": {"title": "Return Distribution", "body": "Histogram of simulated terminal returns. VaR (yellow) = loss exceeded 5% of the time. CVaR (red) = average loss in worst 5%. CVaR is the preferred tail risk measure."},
        "rolling-vol": {"title": "Rolling Volatility", "body": "Annualized 63-day rolling vol per asset. Horizontal lines show regime thresholds. Simulation engine scales vol based on detected regime."},
        "cum-return": {"title": "Historical Cumulative Return", "body": "Portfolio cumulative return. Red shaded areas = known stress periods. Shows how the portfolio weathered past crises."},
        "risk-attrib": {"title": "Risk Attribution (Euler)", "body": "Each asset's % contribution to portfolio CVaR via Euler decomposition. Uses tail-conditional mean: when portfolio is in worst 5%, what does each asset contribute?"},
        "payoff": {"title": "Hedge Payoff Diagrams", "body": "Net payoff (after premium) for each hedge instrument at different terminal prices. Spot = current portfolio value."},
        "hedged-dist": {"title": "Hedged vs Unhedged", "body": "Overlaid return distributions. Shows how the best hedge instrument reshapes the left tail (reduces worst outcomes)."},
        "frontier": {"title": "Hedge Frontier", "body": "Pareto curve: how much CVaR reduction you get per % of NAV spent on hedging. Red line = budget constraint. Analogous to the efficient frontier."},
        "stress-hist": {"title": "Historical Stress Scenarios", "body": "Portfolio return during known crisis periods (GFC, COVID, etc). Based on actual historical data."},
        "stress-hypo": {"title": "Hypothetical Scenarios", "body": "Forward-looking stress tests: geopolitical shocks, recessions, sector crashes. Returns based on calibrated assumptions."},
        "predictions": {"title": "Price Predictions", "body": "Monte Carlo GBM simulation for individual assets. Fan chart shows percentile bands. Uses historical drift and volatility."},
        "entry-timing": {"title": "Entry Timing & DCA", "body": "Should you invest now or scale in? Timing score (0-100) based on momentum, distance from highs, vol regime. DCA strategies compared to lump sum."},
        "dca-comparison": {"title": "DCA vs Lump Sum", "body": "Return distributions for different entry strategies. Lump sum usually wins 2/3 of the time (upward drift), but DCA reduces timing risk."},
        "dip-probs": {"title": "Dip Probabilities", "body": "MC-simulated probability of portfolio dipping below various levels. If 10% dip has >40% probability, consider scaling in."},
        "calibration": {"title": "Portfolio Calibration", "body": "8 optimization strategies from max Sharpe to minimum CVaR. Each uses scipy SLSQP with different objectives."},
        "eff-frontier": {"title": "Efficient Frontier", "body": "Mean-variance frontier (Markowitz). Each dot = a strategy. Star = your current portfolio. Points below the curve are suboptimal."},
        "mc-strategies": {"title": "MC Strategy Comparison", "body": "Forward-looking MC simulation per strategy. Box plots show return distributions. Compares tail risk across strategies."},
        "risk-parity": {"title": "Risk Parity", "body": "Inverse-vol weighting (Bridgewater All-Weather style). Equal risk contribution from each asset. Compared against your current allocation."},
        "ml-regime": {"title": "ML Regime Detection", "body": "KMeans clustering on rolling vol, skew, kurtosis, and correlations. More sophisticated than simple vol thresholds — captures nonlinear regime shifts."},
        "ml-return": {"title": "XGBoost Return Predictor", "body": "Gradient boosted trees predicting 21-day forward portfolio return. Features: multi-horizon returns, vol, skew, kurtosis, correlations. Validated with time-series cross-validation."},
        "ml-vol": {"title": "XGBoost Vol Forecaster", "body": "Predicts forward 21-day realized vol. Captures vol clustering and mean-reversion patterns that GARCH models miss."},
        "ml-crash": {"title": "XGBoost Crash Predictor", "body": "Binary classifier: will the portfolio drop >5% in the next 21 days? Uses class-weighted XGBoost with precision/recall metrics."},
        "ml-cv": {"title": "Cross-Validation", "body": "Time-series 5-fold CV — each fold trains on all data before the test period. No look-ahead bias. Shows model stability across different market periods."},
    }


# ── run ──────────────────────────────────────────────────────────────────────

def run_dashboard():
    import uvicorn
    print(f"Dashboard at http://{config.DASH_HOST}:{config.DASH_PORT}")
    uvicorn.run(app, host=config.DASH_HOST, port=config.DASH_PORT)


if __name__ == "__main__":
    run_dashboard()
