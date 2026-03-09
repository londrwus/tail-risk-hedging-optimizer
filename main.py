"""CLI entry point — run analysis or launch dashboard."""

import argparse
import sys

import numpy as np

import config
from src.data_loader import fetch_prices, log_returns, portfolio_returns
from src.simulation import run_gbm, detect_regime
from src.options import all_instruments
from src.optimizer import optimize_all
from src.portfolio import portfolio_stats, cvar, var, euler_risk_attribution
from src.scenarios import run_all_historical, run_all_hypothetical
from src.backtester import run_all_backtests


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def run_analysis():
    """Full CLI analysis — stats, simulation, optimization, stress tests."""
    tickers = config.DEFAULT_TICKERS
    weights = np.array(config.DEFAULT_WEIGHTS)

    print_header("TAIL RISK HEDGING OPTIMIZER")
    print(f"Portfolio: {tickers}")
    print(f"Weights:   {list(weights)}")
    print(f"Horizon:   {config.HORIZON_DAYS} trading days")
    print(f"Sims:      {config.N_SIMULATIONS:,}")

    print("\nFetching market data...")
    prices = fetch_prices()
    rets = log_returns(prices)
    port_ret = portfolio_returns(rets)
    print(f"Data range: {rets.index[0].date()} to {rets.index[-1].date()} ({len(rets)} days)")

    print_header("PORTFOLIO STATISTICS")
    stats = portfolio_stats(port_ret)
    for k, v in stats.items():
        if isinstance(v, float):
            if abs(v) < 1:
                print(f"  {k:20s}: {v:>10.4f}")
            else:
                print(f"  {k:20s}: {v:>10.2f}")

    regime = detect_regime(rets)
    print(f"\n  Current vol regime: {regime.upper()}")

    print_header("MONTE CARLO SIMULATION")
    sim = run_gbm(prices)
    term_ret = sim["terminal_returns"]
    print(f"  Regime:         {sim['regime'].upper()}")
    print(f"  Mean return:    {np.mean(term_ret):.4f}")
    print(f"  Median return:  {np.median(term_ret):.4f}")
    print(f"  VaR 95%:        {var(term_ret, 0.95):.4f}")
    print(f"  CVaR 95%:       {cvar(term_ret, 0.95):.4f}")

    print_header("RISK ATTRIBUTION (CVaR 95%)")
    attrib = euler_risk_attribution(rets, weights)
    for t in attrib["tickers"]:
        pct = attrib["pct_contribution"][t]
        print(f"  {t:6s}: {pct:>7.1%} of total CVaR")

    print_header("HEDGE OPTIMIZATION")
    spot = float((prices.iloc[-1] * weights).sum())
    hist_vol = float(rets.tail(63).std().mean() * np.sqrt(252))
    iv = hist_vol * config.VRP_ADJ

    opt = optimize_all(term_ret, spot, iv)
    print(f"  {'Instrument':<16} {'Ratio':>8} {'Cost':>8} {'CVaR':>8} {'Reduction':>10}")
    print(f"  {'-'*52}")
    for o in opt:
        print(f"  {o['instrument']:<16} {o['hedge_ratio']:>7.1%} "
              f"{o['cost_pct']:>7.3%} {o['hedged_cvar']:>7.3%} "
              f"{o['cvar_reduction_pct']:>9.1f}%")

    print_header("STRESS SCENARIOS")
    hist = run_all_historical(rets, weights)
    for s in hist:
        print(f"  {s['name']:25s}: return={s['return']:>8.2%}  maxDD={s['max_drawdown']:>7.2%}")

    hypo = run_all_hypothetical(tickers, weights)
    for s in hypo:
        print(f"  {s['name']:30s}: return={s['return']:>8.2%}")

    print(f"\n  Done.")


def main():
    parser = argparse.ArgumentParser(description="Tail Risk Hedging Optimizer")
    parser.add_argument("--dashboard", action="store_true", help="Launch Dash dashboard")
    parser.add_argument("--analysis", action="store_true", help="Run CLI analysis only")
    parser.add_argument("--refresh", action="store_true", help="Force refresh market data")

    args = parser.parse_args()

    if args.refresh:
        fetch_prices(force_refresh=True)
        print("Cache refreshed.")

    if args.analysis:
        run_analysis()
    else:
        from dashboard.app import run_dashboard
        run_dashboard()


if __name__ == "__main__":
    main()
