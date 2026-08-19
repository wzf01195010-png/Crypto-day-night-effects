#!/usr/bin/env python
# coding: utf-8

# # BTC / ETH UTC Daytime–Nighttime Cutoff Backtest
#
# This notebook calculates all **25 Night strategy / Day strategy combinations** for BTC and ETH from 2016-01-01 through 2025-12-31.
#
# All timestamps and session boundaries are **UTC**. Every cutoff is a fixed 12-hour / 12-hour split:
#
# - `4-4-4`: Day 04:00–16:00; Night 16:00–04:00
# - `5-5-5`: Day 05:00–17:00; Night 17:00–05:00
# - `6-6-6`: Day 06:00–18:00; Night 18:00–06:00
# - `7-7-7`: Day 07:00–19:00; Night 19:00–07:00
# - `8-8-8`: Day 08:00–20:00; Night 20:00–08:00
# - `9-9-9`: Day 09:00–21:00; Night 21:00–09:00
# - `10-10-10`: Day 10:00–22:00; Night 22:00–10:00
#
# `8-8-8` therefore means the corrected **12h/12h** definition, not the old three-segment 00–08 / 08–16 / 16–24 definition.
#
# Metrics:
#
# - **Final Wealth**: ending value from initial wealth 100.
# - **Yearly MDD**: maximum drawdown within each calendar year, with a fresh 1.0 baseline at the start of that year.
# - **Average MDD**: arithmetic mean of the ten calendar-year MDD values.
# - **Sharpe Ratio**: zero-risk-free-rate Sharpe based on daily net strategy returns, annualized by `sqrt(365)`.
#
# The existing 0, 1, and 2 bps transaction-cost scenarios are retained. Costs apply to position turnover; a long-to-short flip has turnover 2.

# In[1]:


from pathlib import Path
import itertools
import math
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data_hourly_crypto"
DATA_FILES = {
    "BTC": DATA_DIR / "BTCUSD_1h_2016_2025.csv",
    "ETH": DATA_DIR / "ETHUSD_1h_2016_2025.csv",
}

COINS = ["BTC", "ETH"]
CUTOFF_HOURS = [4, 5, 6, 7, 8, 9, 10]
YEARS = list(range(2016, 2026))
START_TS = pd.Timestamp("2016-01-01 00:00:00", tz="UTC")
END_TS = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")

INITIAL_CAPITAL = 100.0
TRANSACTION_COSTS_BPS = [0, 1, 2]
STRATEGIES = ["cash", "long", "short", "trend", "reversal"]
TIMEZONE = "UTC"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MAIN = OUTPUT_DIR / "btc_eth_all_metrics.csv"
OUTPUT_YEARLY_MDD = OUTPUT_DIR / "btc_eth_yearly_mdd.csv"
LEGACY_8_CSV = OUTPUT_DIR / "legacy_cutoff_8_results.csv"

pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda value: f"{value:,.6f}")

assert all(0 <= hour <= 11 for hour in CUTOFF_HOURS)
assert len(STRATEGIES) ** 2 == 25


def print_frame(title: str, frame: pd.DataFrame) -> None:
    """Print a DataFrame without requiring Jupyter or IPython."""
    print(f"\n{title}")
    print(frame.to_string(index=False))


# ## 1. Load and validate BTC/ETH hourly data
#
# `open_time` must agree exactly with the Unix epoch `timestamp` after both are interpreted as UTC. The selected sample must contain every UTC hour from 2016-01-01 00:00 through 2025-12-31 23:00.

# In[2]:


def load_coin_data(coin: str) -> tuple[pd.DataFrame, dict]:
    path = DATA_FILES[coin]
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)
    required = {"open_time", "close", "timestamp"}
    missing_columns = required.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing_columns)}")

    open_time_utc = pd.to_datetime(raw["open_time"], utc=True, errors="coerce")
    epoch_utc = pd.to_datetime(
        pd.to_numeric(raw["timestamp"], errors="coerce"),
        unit="s",
        utc=True,
        errors="coerce",
    )
    close = pd.to_numeric(raw["close"], errors="coerce")

    timestamp_mismatches = int((open_time_utc != epoch_utc).sum())
    out = pd.DataFrame({"timestamp": open_time_utc, "close": close})
    raw_rows = len(out)
    duplicate_timestamps = int(out["timestamp"].duplicated().sum())
    out = out.dropna(subset=["timestamp", "close"])
    out = out[out["close"] > 0]
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    out = out[(out["timestamp"] >= START_TS) & (out["timestamp"] <= END_TS)]
    out = out.set_index("timestamp").sort_index()

    expected_index = pd.date_range(START_TS, END_TS, freq="1h", tz="UTC")
    missing_hours = int(len(expected_index.difference(out.index)))
    extra_hours = int(len(out.index.difference(expected_index)))
    split_like_jumps = int((np.log(out["close"]).diff().abs() > np.log(10)).sum())

    quality = {
        "coin": coin,
        "source_file": str(path),
        "timezone": str(out.index.tz),
        "raw_rows": raw_rows,
        "rows_after_cleaning": int(len(out)),
        "data_start": out.index.min(),
        "data_end": out.index.max(),
        "timestamp_mismatches": timestamp_mismatches,
        "duplicate_timestamps": duplicate_timestamps,
        "missing_hours": missing_hours,
        "extra_hours": extra_hours,
        "split_like_jump_count": split_like_jumps,
    }
    return out, quality


coin_data = {}
quality_rows = []
for coin in COINS:
    data, quality = load_coin_data(coin)
    coin_data[coin] = data
    quality_rows.append(quality)

data_quality = pd.DataFrame(quality_rows)
print_frame("Data quality", data_quality)


# ## 2. Build all seven UTC 12h/12h session definitions
#
# For each trading date, the Night session ends at the cutoff hour and is followed by that date's Day session. Only dates with exactly 12 Night bars and 12 Day bars are retained.

# In[3]:


def window_text(start_hour: int, end_hour: int) -> str:
    return f"{start_hour:02d}:00-{end_hour:02d}:00"


def build_day_night_sessions(
    price_df: pd.DataFrame, day_start_hour: int
) -> pd.DataFrame:
    x = price_df.copy().sort_index().reset_index()
    x["bar_ret"] = x["close"].pct_change()
    x = x.dropna(subset=["bar_ret"]).copy()

    shifted = x["timestamp"] - pd.to_timedelta(day_start_hour, unit="h")
    x["cycle_start"] = shifted.dt.floor("D") + pd.to_timedelta(day_start_hour, unit="h")
    x["hours_since_cycle_start"] = (
        (x["timestamp"] - x["cycle_start"]) / pd.Timedelta(hours=1)
    ).astype(int)
    x["session"] = np.where(x["hours_since_cycle_start"] < 12, "day", "night")
    x["trading_date"] = x["cycle_start"].dt.floor("D")
    night_mask = x["session"].eq("night")
    x.loc[night_mask, "trading_date"] = (
        x.loc[night_mask, "cycle_start"] + pd.Timedelta(days=1)
    ).dt.floor("D")

    grouped = (
        x.groupby(["trading_date", "session"], observed=True)
        .agg(
            session_ret=("bar_ret", lambda values: (1.0 + values).prod() - 1.0),
            n_bars=("bar_ret", "size"),
            first_bar=("timestamp", "min"),
            last_bar=("timestamp", "max"),
        )
        .reset_index()
    )

    returns = grouped.pivot(
        index="trading_date", columns="session", values="session_ret"
    )
    counts = grouped.pivot(index="trading_date", columns="session", values="n_bars")
    first_bars = grouped.pivot(
        index="trading_date", columns="session", values="first_bar"
    )
    last_bars = grouped.pivot(
        index="trading_date", columns="session", values="last_bar"
    )

    out = pd.DataFrame(index=returns.index).sort_index()
    out["night_ret"] = returns.get("night")
    out["day_ret"] = returns.get("day")
    out["n_night_bars"] = counts.get("night")
    out["n_day_bars"] = counts.get("day")
    out["night_first_bar"] = first_bars.get("night")
    out["night_last_bar"] = last_bars.get("night")
    out["day_first_bar"] = first_bars.get("day")
    out["day_last_bar"] = last_bars.get("day")

    complete = (
        out[
            out["night_ret"].notna()
            & out["day_ret"].notna()
            & out["n_night_bars"].eq(12)
            & out["n_day_bars"].eq(12)
        ]
        .reset_index()
        .sort_values("trading_date")
        .reset_index(drop=True)
    )
    return complete


sessions_by_key = {}
session_quality_rows = []
for coin in COINS:
    for cutoff in CUTOFF_HOURS:
        day_end = (cutoff + 12) % 24
        sessions = build_day_night_sessions(coin_data[coin], cutoff)
        sessions_by_key[(coin, cutoff)] = sessions
        session_quality_rows.append(
            {
                "coin": coin,
                "cutoff": cutoff,
                "cutoff_label": f"{cutoff}-{cutoff}-{cutoff}",
                "timezone": TIMEZONE,
                "day_window": window_text(cutoff, day_end),
                "night_window": window_text(day_end, cutoff),
                "complete_trading_dates": len(sessions),
                "first_trading_date": sessions["trading_date"].min(),
                "last_trading_date": sessions["trading_date"].max(),
                "min_day_bars": int(sessions["n_day_bars"].min()),
                "max_day_bars": int(sessions["n_day_bars"].max()),
                "min_night_bars": int(sessions["n_night_bars"].min()),
                "max_night_bars": int(sessions["n_night_bars"].max()),
            }
        )

session_quality = pd.DataFrame(session_quality_rows)
print_frame("Session quality", session_quality)


# ## 3. Strategy and metric definitions
#
# The five session strategies are `cash`, `long`, `short`, `trend`, and `reversal`. Trend and reversal use only the previous return of the same session type, so no future information is used.

# In[4]:


def position_array(
    strategy: str, previous_same_session_returns: pd.Series
) -> np.ndarray:
    signals = previous_same_session_returns.to_numpy(dtype=float)
    signed = np.sign(np.nan_to_num(signals, nan=0.0))
    if strategy == "cash":
        return np.zeros(len(signals), dtype=float)
    if strategy == "long":
        return np.ones(len(signals), dtype=float)
    if strategy == "short":
        return -np.ones(len(signals), dtype=float)
    if strategy == "trend":
        return signed
    if strategy == "reversal":
        return -signed
    raise ValueError(f"Unknown strategy: {strategy}")


def max_drawdown_from_returns(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    running_peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / running_peak - 1.0))


def backtest_combo(
    sessions: pd.DataFrame, night_strategy: str, day_strategy: str, cost_bps: int
) -> dict:
    x = sessions.sort_values("trading_date").reset_index(drop=True)
    previous_night = x["night_ret"].shift(1)
    previous_day = x["day_ret"].shift(1)

    night_position = position_array(night_strategy, previous_night)
    day_position = position_array(day_strategy, previous_day)
    positions = np.column_stack([night_position, day_position]).reshape(-1)
    realized = np.column_stack(
        [
            x["night_ret"].to_numpy(dtype=float),
            x["day_ret"].to_numpy(dtype=float),
        ]
    ).reshape(-1)

    previous_position = np.concatenate(([0.0], positions[:-1]))
    turnover = np.abs(positions - previous_position)
    cost_rate = cost_bps / 10_000.0
    gross_returns = positions * realized
    net_returns = (1.0 + gross_returns) * (1.0 - cost_rate * turnover) - 1.0
    if np.any(1.0 + net_returns <= 0):
        raise ValueError(
            "A strategy return reached -100% or below; multiplicative wealth is undefined."
        )

    wealth = INITIAL_CAPITAL * np.cumprod(1.0 + net_returns)
    daily_returns = np.prod(1.0 + net_returns.reshape(-1, 2), axis=1) - 1.0
    daily_std = float(np.std(daily_returns, ddof=1))
    if daily_std == 0.0:
        sharpe_ratio = 0.0 if float(np.mean(daily_returns)) == 0.0 else np.nan
    else:
        sharpe_ratio = float(np.mean(daily_returns) / daily_std * math.sqrt(365.0))

    session_years = np.repeat(x["trading_date"].dt.year.to_numpy(), 2)
    yearly_mdd = {
        year: max_drawdown_from_returns(net_returns[session_years == year])
        for year in YEARS
    }

    return {
        "final_wealth": float(wealth[-1]),
        "total_return": float(wealth[-1] / INITIAL_CAPITAL - 1.0),
        "full_period_mdd": max_drawdown_from_returns(net_returns),
        "average_mdd": float(np.mean(list(yearly_mdd.values()))),
        "sharpe_ratio": sharpe_ratio,
        "annualized_daily_volatility": float(daily_std * math.sqrt(365.0)),
        "trade_count": int(np.count_nonzero(turnover)),
        "total_turnover": float(turnover.sum()),
        "n_trading_dates": int(len(x)),
        "n_sessions": int(len(net_returns)),
        "yearly_mdd": yearly_mdd,
    }


STRATEGY_COMBINATIONS = [
    {
        "strategy_id": strategy_id,
        "night_strategy": night_strategy,
        "day_strategy": day_strategy,
        "strategy_label": f"Night {night_strategy.title()} / Day {day_strategy.title()}",
    }
    for strategy_id, (night_strategy, day_strategy) in enumerate(
        itertools.product(STRATEGIES, STRATEGIES), start=1
    )
]

print_frame("Strategy combinations", pd.DataFrame(STRATEGY_COMBINATIONS))


# ## 4. Run BTC and ETH across all cutoffs, costs, and 25 strategies
#
# The main wide result contains one row per asset × cutoff × cost × strategy. A second long result stores one row per calendar-year MDD observation.

# In[5]:


result_rows = []
yearly_mdd_rows = []
source_lookup = data_quality.set_index("coin")["source_file"].to_dict()

for coin in COINS:
    for cutoff in CUTOFF_HOURS:
        sessions = sessions_by_key[(coin, cutoff)]
        day_end = (cutoff + 12) % 24
        print(f"Running {coin} cutoff {cutoff}-{cutoff}-{cutoff} UTC...")
        for cost_bps in TRANSACTION_COSTS_BPS:
            for combo in STRATEGY_COMBINATIONS:
                metrics = backtest_combo(
                    sessions,
                    night_strategy=combo["night_strategy"],
                    day_strategy=combo["day_strategy"],
                    cost_bps=cost_bps,
                )
                base = {
                    "coin": coin,
                    "timezone": TIMEZONE,
                    "cutoff": cutoff,
                    "cutoff_label": f"{cutoff}-{cutoff}-{cutoff}",
                    "day_window": window_text(cutoff, day_end),
                    "night_window": window_text(day_end, cutoff),
                    "transaction_cost_bps": cost_bps,
                    "strategy_id": combo["strategy_id"],
                    "strategy_label": combo["strategy_label"],
                    "night_strategy": combo["night_strategy"],
                    "day_strategy": combo["day_strategy"],
                    "initial_wealth": INITIAL_CAPITAL,
                    "source_file": source_lookup[coin],
                }
                row = {
                    **base,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "yearly_mdd"
                    },
                }
                for year, mdd in metrics["yearly_mdd"].items():
                    row[f"yearly_mdd_{year}"] = mdd
                    yearly_mdd_rows.append({**base, "year": year, "yearly_mdd": mdd})
                result_rows.append(row)

results = pd.DataFrame(result_rows)
yearly_mdd_results = pd.DataFrame(yearly_mdd_rows)
results["rank_final_wealth_within_coin_cutoff_cost"] = (
    results.groupby(["coin", "cutoff", "transaction_cost_bps"])["final_wealth"]
    .rank(method="first", ascending=False)
    .astype(int)
)

yearly_columns = [f"yearly_mdd_{year}" for year in YEARS]
ordered_columns = [
    "coin",
    "timezone",
    "cutoff",
    "cutoff_label",
    "day_window",
    "night_window",
    "transaction_cost_bps",
    "strategy_id",
    "strategy_label",
    "night_strategy",
    "day_strategy",
    "initial_wealth",
    "final_wealth",
    "total_return",
    "sharpe_ratio",
    "average_mdd",
    "full_period_mdd",
    *yearly_columns,
    "annualized_daily_volatility",
    "trade_count",
    "total_turnover",
    "n_trading_dates",
    "n_sessions",
    "rank_final_wealth_within_coin_cutoff_cost",
    "source_file",
]
results = (
    results[ordered_columns]
    .sort_values(["coin", "cutoff", "transaction_cost_bps", "strategy_id"])
    .reset_index(drop=True)
)
yearly_mdd_results = yearly_mdd_results.sort_values(
    ["coin", "cutoff", "transaction_cost_bps", "strategy_id", "year"]
).reset_index(drop=True)

results.to_csv(OUTPUT_MAIN, index=False)
yearly_mdd_results.to_csv(OUTPUT_YEARLY_MDD, index=False)

print(f"Saved main result: {OUTPUT_MAIN}")
print(f"Saved yearly MDD: {OUTPUT_YEARLY_MDD}")
print(f"Main rows: {len(results):,}; yearly MDD rows: {len(yearly_mdd_results):,}")


# ## 5. Quality assurance and regression checks

# In[6]:


expected_result_rows = len(COINS) * len(CUTOFF_HOURS) * len(TRANSACTION_COSTS_BPS) * 25
expected_yearly_rows = expected_result_rows * len(YEARS)

assert len(results) == expected_result_rows == 1_050
assert len(yearly_mdd_results) == expected_yearly_rows == 10_500
assert results.groupby(["coin", "cutoff", "transaction_cost_bps"]).size().eq(25).all()
assert results["timezone"].eq("UTC").all()
assert data_quality["timezone"].eq("UTC").all()
assert (
    data_quality[
        [
            "timestamp_mismatches",
            "duplicate_timestamps",
            "missing_hours",
            "extra_hours",
            "split_like_jump_count",
        ]
    ]
    .eq(0)
    .all()
    .all()
)
assert (
    session_quality[
        ["min_day_bars", "max_day_bars", "min_night_bars", "max_night_bars"]
    ]
    .eq(12)
    .all()
    .all()
)
assert session_quality["complete_trading_dates"].eq(3_652).all()
assert np.isfinite(results["final_wealth"]).all()
assert np.isfinite(results["sharpe_ratio"]).all()
assert results[yearly_columns].le(1e-12).all().all()
assert results[yearly_columns].ge(-1.0 - 1e-12).all().all()
assert np.allclose(
    results["average_mdd"], results[yearly_columns].mean(axis=1), rtol=0, atol=1e-12
)

cash_cash = results[
    (results["night_strategy"] == "cash") & (results["day_strategy"] == "cash")
]
assert np.allclose(cash_cash["final_wealth"], INITIAL_CAPITAL, rtol=0, atol=1e-12)
assert np.allclose(cash_cash["average_mdd"], 0.0, rtol=0, atol=1e-12)
assert np.allclose(cash_cash["sharpe_ratio"], 0.0, rtol=0, atol=1e-12)

legacy_tie_max_error = np.nan
if LEGACY_8_CSV.exists():
    legacy = pd.read_csv(LEGACY_8_CSV)
    current_8 = results[results["cutoff"].eq(8)][
        [
            "coin",
            "transaction_cost_bps",
            "day_strategy",
            "night_strategy",
            "final_wealth",
        ]
    ]
    tie = current_8.merge(
        legacy[
            [
                "coin",
                "transaction_cost_bps",
                "day_strategy",
                "night_strategy",
                "final_balance",
            ]
        ],
        on=["coin", "transaction_cost_bps", "day_strategy", "night_strategy"],
        how="inner",
        validate="one_to_one",
    )
    assert len(tie) == 150
    legacy_tie_max_error = float(
        (tie["final_wealth"] - tie["final_balance"]).abs().max()
    )
    assert legacy_tie_max_error < 1e-7

qa_summary = pd.DataFrame(
    {
        "check": [
            "main result rows",
            "yearly MDD rows",
            "strategies per asset/cutoff/cost",
            "session hours per side",
            "timezone",
            "complete dates per cutoff",
            "legacy cutoff-8 max wealth error",
        ],
        "value": [
            len(results),
            len(yearly_mdd_results),
            25,
            12,
            "UTC",
            3_652,
            legacy_tie_max_error,
        ],
    }
)
print_frame("Quality assurance", qa_summary)
print("All quality checks passed.")


# ## 6. Results
#
# The first table shows all 350 zero-cost rows (2 assets × 7 cutoffs × 25 strategies) with the requested metrics. The exported main CSV also contains the 1 bps and 2 bps results.

# In[7]:


requested_metric_columns = [
    "coin",
    "cutoff_label",
    "day_window",
    "night_window",
    "strategy_id",
    "strategy_label",
    "final_wealth",
    *yearly_columns,
    "average_mdd",
    "sharpe_ratio",
]
zero_cost_metrics = results[results["transaction_cost_bps"].eq(0)][
    requested_metric_columns
]
print_frame("Zero-cost metrics", zero_cost_metrics)

best_zero_cost = results.loc[
    results[results["transaction_cost_bps"].eq(0)]
    .groupby(["coin", "cutoff"])["final_wealth"]
    .idxmax(),
    [
        "coin",
        "cutoff_label",
        "day_window",
        "night_window",
        "strategy_label",
        "final_wealth",
        "average_mdd",
        "full_period_mdd",
        "sharpe_ratio",
    ],
].sort_values(["coin", "cutoff_label"])

print("Best zero-cost strategy by asset and cutoff:")
print_frame("Best zero-cost strategy by asset and cutoff", best_zero_cost)

output_files = pd.DataFrame(
    {
        "output": ["All metrics (wide)", "Yearly MDD (long)"],
        "path": [str(OUTPUT_MAIN), str(OUTPUT_YEARLY_MDD)],
        "rows": [len(results), len(yearly_mdd_results)],
    }
)
print_frame("Output files", output_files)
