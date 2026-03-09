"""All tunable parameters in one place."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache"

RANDOM_SEED = 42

# portfolio
DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
DEFAULT_WEIGHTS = [0.40, 0.30, 0.20, 0.10]
INITIAL_NAV = 1_000_000.0

# data
DATA_START_DATE = "2005-01-01"
DATA_END_DATE = None   # None → today
CACHE_EXPIRY_HOURS = 12

# simulation
N_SIMULATIONS = 10_000
HORIZON_DAYS = 63      # 1 quarter
TRADING_DAYS_YEAR = 252
BLOCK_SIZE = 21        # 1-month blocks for bootstrap

# risk
CONFIDENCE_LEVELS = [0.95, 0.99]
RISK_FREE_RATE = 0.05

# hedge
HEDGE_BUDGET = 0.02    # 2% of NAV
FRONTIER_POINTS = 30

# option moneyness
PUT_MONEYNESS = 0.95
COLLAR_PUT_MONEY = 0.95
COLLAR_CALL_MONEY = 1.05
SPREAD_LONG_MONEY = 0.95
SPREAD_SHORT_MONEY = 0.90   # deeper OTM
VIX_STRIKE = 20.0
VIX_CURRENT = 18.0

# IV proxy — realized vol typically underestimates IV by ~15%
VRP_ADJ = 1.15

# regime detection
REGIME_LOOKBACK = 63
REGIME_THRESHOLDS = {"low": 0.12, "high": 0.25}  # annualized vol boundaries
REGIME_VOL_SCALE = {"low": 0.85, "mid": 1.00, "high": 1.40}

# backtest
BACKTEST_EST_WINDOW = 252   # 1yr lookback
BACKTEST_HOLD_PERIOD = 21   # 1 month OOS

# stress scenarios — historical
STRESS_SCENARIOS = {
    "GFC_2008":          {"start": "2008-09-01", "end": "2009-03-31"},
    "COVID_2020":        {"start": "2020-02-15", "end": "2020-04-30"},
    "DOT_COM":           {"start": "2000-03-01", "end": "2002-10-31"},
    "VOLMAGEDDON_2018":  {"start": "2018-01-26", "end": "2018-04-06"},
    "RATE_HIKE_2022":    {"start": "2022-01-03", "end": "2022-10-12"},
}

# hypothetical / forward-looking scenarios
# these represent plausible future shocks, not historical replays
HYPOTHETICAL_SCENARIOS = {
    "GEOPOLITICAL_ENERGY_SHOCK": {
        "SPY": -0.18, "QQQ": -0.22, "TLT": 0.08, "GLD": 0.15,
        "severity": 0.85,
    },
    "GLOBAL_RECESSION": {
        "SPY": -0.30, "QQQ": -0.35, "TLT": 0.15, "GLD": 0.10,
        "severity": 0.90,
    },
    "RATES_SPIKE_INFLATION": {
        "SPY": -0.15, "QQQ": -0.20, "TLT": -0.12, "GLD": 0.05,
        "severity": 0.70,
    },
    "TECH_SECTOR_CRASH": {
        "SPY": -0.20, "QQQ": -0.40, "TLT": 0.10, "GLD": 0.03,
        "severity": 0.75,
    },
    "LIQUIDITY_CRISIS": {
        "SPY": -0.12, "QQQ": -0.15, "TLT": -0.05, "GLD": 0.08,
        "severity": 0.65,
    },
    "STAGFLATION": {
        "SPY": -0.22, "QQQ": -0.25, "TLT": -0.10, "GLD": 0.12,
        "severity": 0.80,
    },
}

# prediction tickers — broader universe for price forecasts
PREDICTION_TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "Crude Oil",
}
PREDICTION_HORIZON_YEARS = 3
PREDICTION_HORIZON_DAYS = 252 * 3  # 3 years
PREDICTION_SIMS = 5_000

# rates — proxies for macro overlay
RATE_TICKERS = {
    "^TNX": "US 10Y Treasury Yield",
    "^IRX": "US 13-Week T-Bill",
    "^FVX": "US 5Y Treasury Yield",
}

# factor analysis
MOMENTUM_WINDOWS = [21, 63, 126, 252]  # 1m, 3m, 6m, 12m
FACTOR_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "SLV", "USO", "EFA", "EEM"]

# risk parity (Bridgewater All-Weather inspired)
RISK_PARITY_TICKERS = ["SPY", "TLT", "GLD", "DBC", "IEF"]
RISK_PARITY_LABELS = ["Equities", "Long Bonds", "Gold", "Commodities", "Interm Bonds"]

# dashboard
DASH_HOST = "127.0.0.1"
DASH_PORT = 8050
DASH_DEBUG = True
