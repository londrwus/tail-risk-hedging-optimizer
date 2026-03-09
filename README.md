# Tail risk hedging optimizer

Simulates a multi-asset portfolio, prices derivative hedges (Black-Scholes), and finds the cheapest way to cut tail risk via CVaR optimization. There's also an XGBoost layer for return/vol/crash prediction and a KMeans regime detector that feeds into the Monte Carlo engine.

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## How it works

The core loop: fetch market data, estimate covariance, simulate 10,000 correlated paths (Cholesky GBM with regime-scaled vol), then optimize hedge ratios by minimizing CVaR subject to a cost budget.

Four hedge instruments are priced via Black-Scholes: protective put, collar, put spread, and a VIX call proxy. The optimizer (scipy SLSQP) finds the best ratio for each, and plots a hedge frontier (cost vs CVaR reduction) that's basically an efficient frontier for hedging.

On top of that, XGBoost models predict 21-day forward returns, forward realized vol, and crash probability. KMeans clustering on rolling vol/skew/kurtosis replaces the usual "if vol > 25% then high regime" threshold approach. The ML regime label feeds back into the simulation's vol scaling.

Everything runs through a 10-tab dashboard (FastAPI + Plotly.js). No React, just vanilla HTML/JS.

## Dashboard tabs

1. Portfolio overview -- fan chart, return distribution with VaR/CVaR, rolling vol, Euler risk attribution
2. Hedge analysis -- payoff diagrams for all four instruments, hedged vs unhedged overlay, Greeks
3. Hedge frontier -- cost vs CVaR reduction across instruments
4. Stress testing -- GFC 2008, COVID 2020, dot-com, volmageddon, 2022 rate hikes, plus hypothetical shocks
5. Price predictions -- MC + XGBoost forecasts for 18 assets (equities, bonds, commodities, crypto)
6. Factor & drawdown -- momentum z-scores, tail dependency, rolling correlations, drawdown history
7. Risk parity -- inverse-vol allocation vs yours
8. Portfolio calibration -- 8 strategies (max Sharpe, min vol, min CVaR, target vol variants) on the efficient frontier
9. Entry timing -- lump sum vs DCA, dip probabilities, price-level buy plan
10. ML models -- regime timeline, crash gauge, feature importance, CV performance

You can reconfigure the portfolio (up to 11 assets, including BTC/ETH/SOL/TON) from the header.

## Structure

```
config.py               all parameters in one place
main.py                 CLI or dashboard launcher

src/
  data_loader.py        yfinance + parquet cache
  simulation.py         correlated GBM, block bootstrap, regime detection
  options.py            Black-Scholes: put, collar, put spread, VIX call
  portfolio.py          VaR, CVaR, Sharpe, Sortino, Euler attribution
  optimizer.py          CVaR minimization, hedge frontier
  backtester.py         rolling 252d estimation, 21d hold period
  scenarios.py          historical + hypothetical stress tests
  predictions.py        single-asset MC + XGBoost forecasts
  factors.py            momentum, tail dependency, rolling correlation
  calibrator.py         8 strategies, efficient frontier, MC comparison
  entry_timing.py       DCA vs lump sum, dip probabilities, timing score
  ml_models.py          XGBoost models, KMeans regime detection

dashboard/
  app.py                FastAPI (10 tabs as JSON endpoints)
  static/               HTML + CSS + JS (single-page app, dark theme)
```

## Running it

```bash
pip install -e .
python main.py              # dashboard at http://127.0.0.1:8050
python main.py --analysis   # CLI analysis only
python main.py --refresh    # force refresh cached data
```

## Default portfolio

SPY 40%, QQQ 30%, TLT 20%, GLD 10%. Change it from the dashboard or edit `config.py`.

## ML models

Three XGBoost models, all trained with `TimeSeriesSplit` (5-fold, no look-ahead):

- Return predictor (21-day forward) -- around 54% directional accuracy on cross-validation, which is about what you'd expect for a model with no alternative data
- Vol forecaster (21-day forward realized vol) -- rank correlation around 0.24 against actual. Vol is more predictable than returns, mostly because it clusters
- Crash predictor (P(>5% drawdown in 21 days)) -- binary classifier with class-weighted loss. High accuracy numbers look good but precision is low because crashes are rare events

KMeans (3 clusters) runs on rolling vol, skew, kurtosis, and mean pairwise correlation. The detected regime scales the GBM simulation's volatility parameter.

~95 features for a 4-asset portfolio: multi-horizon returns (5d to 252d), rolling vol/skew/kurtosis at each window, distance from 52-week high/low, and average pairwise correlation.

## Tech stack

Python 3.12, numpy, pandas, scipy, XGBoost, scikit-learn, yfinance, FastAPI, Plotly.js, pyarrow.

## Technical details

### Monte Carlo simulation

Each asset follows geometric Brownian motion. The log-return over one time step $\Delta t$ is:

$$\frac{dS}{S} = \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\;Z$$

where $Z \sim \mathcal{N}(0,1)$. To correlate the assets, I compute the Cholesky decomposition $L$ of the historical correlation matrix $\Sigma$ and transform independent normals:

$$Z_{\text{corr}} = Z \cdot L^\top$$

A small jitter ($10^{-8} \cdot I$) keeps the decomposition from failing on near-singular matrices.

The simulation runs 10,000 paths over a 63-day (1 quarter) horizon by default. Terminal portfolio returns are the weighted sum of per-asset price ratios minus one:

$$r_{\text{port}} = \sum_{i} w_i \left(\frac{S_i^T}{S_i^0} \right) - 1$$

There's also a block bootstrap alternative that resamples 21-day blocks from the historical return matrix. This preserves both volatility clustering and cross-asset dependence without assuming GBM.

### Regime detection

The KMeans approach clusters historical periods on four features computed at 21-day and 63-day windows: rolling volatility, rolling skewness, rolling kurtosis, and mean pairwise correlation. Three clusters get sorted by their average vol so that cluster 0 is always "low vol" and cluster 2 is "high vol."

The detected regime scales the GBM diffusion parameter:

$$\sigma_{\text{adj}} = \sigma_{\text{hist}} \times \lambda_{\text{regime}}, \quad \lambda \in \{0.85,\; 1.0,\; 1.4\}$$

This way the simulation produces fatter tails when the market is already stressed.

### CVaR and risk attribution

VaR at confidence level $\alpha$ is the $\alpha$-quantile loss:

$$\text{VaR}_\alpha = -F^{-1}(1 - \alpha)$$

CVaR (expected shortfall) is the conditional expectation beyond VaR:

$$\text{CVaR}_\alpha = -\mathbb{E}\left[r \;\middle|\; r \leq -\text{VaR}_\alpha\right]$$

CVaR is a coherent risk measure, VaR is not, which matters for optimization since CVaR is convex.

Risk attribution uses Euler decomposition. Each asset's contribution when the portfolio is in its worst $(1-\alpha)$ outcomes:

$$\text{CVaR}_i = w_i \cdot \mathbb{E}\left[r_i \;\middle|\; r_{\text{port}} \leq -\text{VaR}_\alpha\right]$$

These component CVaRs sum exactly to total CVaR: $\sum_i \text{CVaR}_i = \text{CVaR}_{\text{port}}$.

### Black-Scholes pricing

Standard BS formulas. Put and call prices:

$$C = S\,\Phi(d_1) - Ke^{-rT}\Phi(d_2)$$

$$P = Ke^{-rT}\Phi(-d_2) - S\,\Phi(-d_1)$$

where $\Phi$ is the standard normal CDF and:

$$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

Greeks are computed analytically (not finite-differenced): $\Delta$, $\Gamma$, $\mathcal{V}$ (per 1% vol move), $\Theta$ (per calendar day), $\rho$. Each instrument reports Greeks per leg, so the collar shows both the long put and the short call.

The VIX call uses a rough proxy where implied VIX scales as $\text{VIX}_{\text{impl}} = \text{VIX}_0 \cdot (1 - 5 \cdot r_{\text{port}})$, clipped at zero. Not a real vol surface model, but it captures the key behavior that VIX spikes when equities crash.

Implied volatility is estimated as $\sigma_{\text{IV}} = 1.15 \cdot \sigma_{\text{realized}}$. The 15% markup approximates the variance risk premium. A real implementation would pull IV from an options chain.

### Hedge optimization

The optimizer minimizes portfolio CVaR at 95% confidence as a function of hedge ratio $h$, subject to a cost budget:

$$\min_{h} \; \text{CVaR}_{0.95}\!\left(r_{\text{hedged}}\right) \quad \text{s.t.} \quad h \cdot \frac{\pi}{S_0} \leq B$$

where $B = 2\%$ of NAV and $\pi$ is the instrument premium. The hedged return:

$$r_{\text{hedged}} = r_{\text{unhedged}} + h \cdot \frac{\text{payoff}(S_T)}{S_0} - h \cdot \frac{\pi}{S_0}$$

scipy SLSQP handles the constrained optimization. The hedge frontier sweeps $h$ from 0 to the maximum affordable ratio and records (cost, CVaR reduction) at each point.

Collars are capped at $h \leq 0.5$ even when they're net credit, because at $h = 1$ they collapse the return distribution into a narrow band (everything gets clamped between the put and call strikes).

### Portfolio calibration

Eight strategies, all solved with SLSQP on the historical covariance matrix:

- Max Sharpe: $\min_w \; -\frac{w^\top\mu - r_f}{\sqrt{w^\top \Sigma w}}$
- Min vol: $\min_w \; \sqrt{w^\top \Sigma w}$
- Min CVaR: $\min_w \; \text{CVaR}(R \cdot w)$ directly on the return matrix
- Risk parity: $w_i = \frac{1/\sigma_i}{\sum_j 1/\sigma_j}$
- Max return: $\min_w \; -w^\top\mu$
- Target vol (8%, 12%, 18%): $\min_w \; -w^\top\mu$ subject to $\sqrt{w^\top \Sigma w} \leq \sigma_{\text{target}}$

All strategies have per-asset bounds $w_i \in [0, 0.6]$ and $\sum_i w_i = 1$.

### XGBoost models

Features are built per asset and per portfolio, at lookback windows $\tau \in \{5, 10, 21, 63, 126, 252\}$ days:

- Trailing log-returns $\sum_{t-\tau}^{t} r_t$
- Rolling standard deviation $\hat\sigma_\tau \cdot \sqrt{252}$
- Rolling skewness and kurtosis ($\tau \geq 21$ only)
- Distance from 252-day high and low: $(S_t - S_{\max}^{252}) / S_{\max}^{252}$
- Mean pairwise rolling correlation (63-day window)
- Portfolio-level vol and return at 21d/63d/126d

XGBoost hyperparameters: 200 trees, max depth 4, learning rate 0.05, 80% subsample, 80% column sample, $\ell_1 = 0.1$, $\ell_2 = 1.0$. These are conservative settings that avoid overfitting on financial data.

The crash predictor handles class imbalance via $\text{scale\_pos\_weight} = n^- / n^+$. Crashes (>5% drawdown in 21 days) happen roughly 7% of the time, so the model sees about 14x more non-crash samples.

All validation uses `TimeSeriesSplit` with 5 folds. Each fold trains on everything before the test period. No shuffling, no look-ahead.

### Backtester

Rolling window: 252-day estimation period, 21-day hold period. At each step: estimate vol from the trailing window, price the hedge instruments, apply the payoff to the realized out-of-sample returns. Output: realized CVaR with vs without the hedge over the full backtest. GFC 2008 and COVID 2020 windows are included as fixed stress periods.

### Entry timing

The timing score $s \in [0, 100]$ combines:

- Recent 3-month portfolio momentum (negative return = better entry)
- Distance from 52-week high (further below = better entry)
- Vol regime (elevated vol gets a small bonus, low vol gets a small penalty)
- MC-estimated $\mathbb{P}(\text{dip} \geq 5\%)$ over the horizon
- XGBoost crash probability ($p > 0.6$ subtracts 15 points, $p < 0.15$ adds 8)
- XGBoost return forecast (positive/negative adjusts by 5 points)

The DCA analysis simulates 5,000 paths and applies four buy schedules to each: aggressive (50/25/15/10 over 4 months), balanced (equal monthly), conservative (25% + spread), ultra conservative (10% + spread). For each schedule and each simulated path, it tracks the average cost basis and terminal value, then reports mean/median return, VaR, CVaR, and $\mathbb{P}(\text{DCA beats lump sum})$.

## License

MIT
