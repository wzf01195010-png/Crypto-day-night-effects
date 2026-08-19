"""Stationary-bootstrap tests for the BTC/ETH day-night strategy paper.

This script uses the session construction, lagged-signal rules, and
turnover-cost mechanics from ``btc_eth_backtest.py``. It does not modify the
source data or the paper's LaTeX files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data_hourly_crypto"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"
DATA_FILES = {
    "BTC": DATA_DIR / "BTCUSD_1h_2016_2025.csv",
    "ETH": DATA_DIR / "ETHUSD_1h_2016_2025.csv",
}
EXISTING_RESULTS = OUTPUT_DIR / "btc_eth_all_metrics.csv"

START_TS = pd.Timestamp("2016-01-01 00:00:00", tz="UTC")
END_TS = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")
INITIAL_WEALTH = 100.0
BOOTSTRAP_REPLICATIONS = 5_000
EXPECTED_BLOCK_LENGTH_DAYS = 7.0
RANDOM_SEED = 20260818
TRANSACTION_COSTS_BPS = (0, 1, 2)
BOOTSTRAP_BATCH_SIZE = 250

ASSET_SPECIFICATIONS = {
    "BTC": {
        "day_start_hour": 8,
        "preferred_night": "reversal",
        "preferred_day": "reversal",
        "preferred_label": "Reversal/Reversal",
    },
    "ETH": {
        "day_start_hour": 5,
        "preferred_night": "long",
        "preferred_day": "reversal",
        "preferred_label": "Long/Reversal",
    },
}

SAMPLE_PERIODS = {
    "2016-2025": (
        pd.Timestamp("2016-01-01", tz="UTC"),
        pd.Timestamp("2025-12-31", tz="UTC"),
    ),
    "2016-2020": (
        pd.Timestamp("2016-01-01", tz="UTC"),
        pd.Timestamp("2020-12-31", tz="UTC"),
    ),
    "2021-2025": (
        pd.Timestamp("2021-01-01", tz="UTC"),
        pd.Timestamp("2025-12-31", tz="UTC"),
    ),
}

CONDITIONAL_CSV = OUTPUT_DIR / "conditional_predictability_tests.csv"
PREFERRED_CSV = OUTPUT_DIR / "preferred_vs_buyhold_tests.csv"
CONDITIONAL_TEX = OUTPUT_DIR / "conditional_predictability_full_sample.tex"
PREFERRED_TEX = OUTPUT_DIR / "preferred_vs_buyhold_full_sample.tex"
SUBPERIOD_TEX = OUTPUT_DIR / "statistical_tests_subperiods.tex"
VALIDATION_MD = OUTPUT_DIR / "statistical_test_validation.md"


# ---------------------------------------------------------------------------
# Shared data, session, and backtest definitions
# ---------------------------------------------------------------------------


def load_coin_data(asset: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and validate the hourly USD close series exactly as in the notebook."""
    path = DATA_FILES[asset]
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
    quality = {
        "asset": asset,
        "source_file": str(path),
        "raw_rows": raw_rows,
        "rows_after_cleaning": int(len(out)),
        "data_start": out.index.min(),
        "data_end": out.index.max(),
        "timestamp_mismatches": timestamp_mismatches,
        "duplicate_timestamps": duplicate_timestamps,
        "missing_hours": int(len(expected_index.difference(out.index))),
        "extra_hours": int(len(out.index.difference(expected_index))),
        "missing_close_values": int(out["close"].isna().sum()),
    }
    return out, quality


def build_day_night_sessions(
    price_df: pd.DataFrame, day_start_hour: int
) -> pd.DataFrame:
    """Create the notebook's two 12-hour sessions and trading-date alignment."""
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


def position_array(
    strategy: str, previous_same_session_returns: pd.Series
) -> np.ndarray:
    """Use the exact position rules from the main backtest."""
    signals = previous_same_session_returns.to_numpy(dtype=float)
    signed = np.sign(np.nan_to_num(signals, nan=0.0))
    if strategy == "cash":
        return np.zeros(len(signals), dtype=float)
    if strategy == "long":
        return np.ones(len(signals), dtype=float)
    if strategy == "short":
        return -np.ones(len(signals), dtype=float)
    if strategy in {"trend", "momentum"}:
        return signed
    if strategy == "reversal":
        return -signed
    raise ValueError(f"Unknown strategy: {strategy}")


def backtest_daily_returns(
    sessions: pd.DataFrame,
    night_strategy: str,
    day_strategy: str,
    cost_bps: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return the notebook-equivalent daily path and turnover diagnostics."""
    x = sessions.sort_values("trading_date").reset_index(drop=True).copy()
    if x[["night_ret", "day_ret"]].isna().any().any():
        raise ValueError("Missing session returns cannot be replaced with zero.")
    assert_consecutive_daily_dates(x, "backtest sessions")

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
        raise ValueError("A session net return reached -100% or below.")

    session_net = net_returns.reshape(-1, 2)
    daily_returns = np.prod(1.0 + session_net, axis=1) - 1.0
    if np.any(1.0 + daily_returns <= 0):
        raise ValueError("A daily net return reached -100% or below.")

    daily = pd.DataFrame(
        {
            "trading_date": x["trading_date"],
            "night_return": x["night_ret"],
            "day_return": x["day_ret"],
            "previous_night_return": previous_night,
            "previous_day_return": previous_day,
            "night_position": night_position,
            "day_position": day_position,
            "night_net_return": session_net[:, 0],
            "day_net_return": session_net[:, 1],
            "daily_net_return": daily_returns,
        }
    )
    final_wealth = float(INITIAL_WEALTH * np.prod(1.0 + daily_returns))
    diagnostics = {
        "final_wealth": final_wealth,
        "trade_count": float(np.count_nonzero(turnover)),
        "total_turnover": float(turnover.sum()),
    }
    return daily, diagnostics


# ---------------------------------------------------------------------------
# Bootstrap and multiple-testing helpers
# ---------------------------------------------------------------------------


def assert_consecutive_daily_dates(frame: pd.DataFrame, context: str) -> None:
    dates = frame["trading_date"].sort_values().reset_index(drop=True)
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise AssertionError(f"{context}: trading dates are not strictly ordered.")
    if len(dates) > 1:
        gaps = dates.diff().dropna()
        if not gaps.eq(pd.Timedelta(days=1)).all():
            raise AssertionError(
                f"{context}: observations are not consecutive calendar days."
            )


def sample_period(frame: pd.DataFrame, period_label: str) -> pd.DataFrame:
    start, end = SAMPLE_PERIODS[period_label]
    out = (
        frame[frame["trading_date"].between(start, end, inclusive="both")]
        .copy()
        .sort_values("trading_date")
        .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError(f"No sessions in sample period {period_label}.")
    assert_consecutive_daily_dates(out, period_label)
    return out


def stationary_bootstrap_indices(
    n_observations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano stationary bootstrap with expected block length 7."""
    if n_observations < 2:
        raise ValueError("Stationary bootstrap requires at least two observations.")
    restart_probability = 1.0 / EXPECTED_BLOCK_LENGTH_DAYS
    indices = np.empty(
        (BOOTSTRAP_REPLICATIONS, n_observations),
        dtype=np.int32,
    )
    indices[:, 0] = rng.integers(0, n_observations, size=BOOTSTRAP_REPLICATIONS)
    for column in range(1, n_observations):
        indices[:, column] = (indices[:, column - 1] + 1) % n_observations
        restart = rng.random(BOOTSTRAP_REPLICATIONS) < restart_probability
        restart_count = int(restart.sum())
        if restart_count:
            indices[restart, column] = rng.integers(
                0,
                n_observations,
                size=restart_count,
            )
    return indices


def bootstrap_in_batches(
    indices: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    estimates = np.empty(indices.shape[0], dtype=float)
    for start in range(0, indices.shape[0], BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, indices.shape[0])
        estimates[start:stop] = statistic(indices[start:stop])
    if not np.isfinite(estimates).all():
        raise ValueError("A bootstrap replication produced a non-finite statistic.")
    return estimates


def centered_bootstrap_inference(
    estimate: float,
    bootstrap_estimates: np.ndarray,
) -> tuple[float, float, float]:
    lower, upper = np.quantile(bootstrap_estimates, [0.025, 0.975])
    exceedances = int(
        np.count_nonzero(np.abs(bootstrap_estimates - estimate) >= abs(estimate))
    )
    p_value = (1.0 + exceedances) / (BOOTSTRAP_REPLICATIONS + 1.0)
    return float(lower), float(upper), float(p_value)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Holm adjustment received a missing p-value.")
    order = np.argsort(values)
    ordered = values[order]
    m = len(values)
    adjusted_ordered = np.maximum.accumulate(
        np.array([(m - rank) * value for rank, value in enumerate(ordered)])
    )
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted


def add_holm_and_significance(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    out = frame.copy()
    out["p_value_holm"] = np.nan
    for _, group in out.groupby(group_columns, sort=False):
        out.loc[group.index, "p_value_holm"] = holm_adjust(
            group["p_value_raw"].to_numpy(dtype=float)
        )
    out["significance_10pct"] = out["p_value_holm"] < 0.10
    out["significance_5pct"] = out["p_value_holm"] < 0.05
    out["significance_1pct"] = out["p_value_holm"] < 0.01
    return out


# ---------------------------------------------------------------------------
# Test 1: conditional predictability
# ---------------------------------------------------------------------------


def reversal_conditional_test(
    asset: str,
    period_label: str,
    session: str,
    sessions: pd.DataFrame,
    indices: np.ndarray,
) -> dict[str, object]:
    return_column = f"{session}_ret"
    paired = (
        pd.DataFrame(
            {
                "current": sessions[return_column],
                "lag": sessions[return_column].shift(1),
            }
        )
        .dropna()
        .reset_index(drop=True)
    )
    if len(paired) != len(sessions) - 1:
        raise AssertionError("Exactly one unavailable lag observation must be removed.")
    if len(paired) != indices.shape[1]:
        raise AssertionError("Bootstrap indices do not match conditional observations.")

    current = paired["current"].to_numpy(dtype=float)
    lag = paired["lag"].to_numpy(dtype=float)
    positive = lag > 0
    negative = lag < 0
    zero = lag == 0
    if not positive.any() or not negative.any():
        raise ValueError("Both positive and negative lag groups are required.")
    mean_after_positive = float(current[positive].mean())
    mean_after_negative = float(current[negative].mean())
    estimate = mean_after_negative - mean_after_positive

    def statistic(batch_indices: np.ndarray) -> np.ndarray:
        bootstrap_current = current[batch_indices]
        bootstrap_lag = lag[batch_indices]
        positive_mask = bootstrap_lag > 0
        negative_mask = bootstrap_lag < 0
        positive_count = positive_mask.sum(axis=1)
        negative_count = negative_mask.sum(axis=1)
        if np.any(positive_count == 0) or np.any(negative_count == 0):
            raise ValueError("A bootstrap sample omitted one conditional group.")
        positive_mean = (
            np.where(positive_mask, bootstrap_current, 0.0).sum(axis=1) / positive_count
        )
        negative_mean = (
            np.where(negative_mask, bootstrap_current, 0.0).sum(axis=1) / negative_count
        )
        return negative_mean - positive_mean

    bootstrap_estimates = bootstrap_in_batches(indices, statistic)
    ci_lower, ci_upper, p_value = centered_bootstrap_inference(
        estimate,
        bootstrap_estimates,
    )
    print(
        f"[conditional] {asset} {period_label} {session}: "
        f"n={len(paired):,}, lag+={positive.sum():,}, lag-={negative.sum():,}, lag0={zero.sum():,}"
    )
    return {
        "asset": asset,
        "sample_period": period_label,
        "session": "nighttime" if session == "night" else "daytime",
        "preferred_rule": "Reversal",
        "statistic_name": "conditional_reversal_difference",
        "observations": len(paired),
        "lag_positive_observations": int(positive.sum()),
        "lag_negative_observations": int(negative.sum()),
        "lag_zero_observations": int(zero.sum()),
        "mean_after_positive_lag_bps": mean_after_positive * 10_000.0,
        "mean_after_negative_lag_bps": mean_after_negative * 10_000.0,
        "estimate_bps": estimate * 10_000.0,
        "ci_95_lower_bps": ci_lower * 10_000.0,
        "ci_95_upper_bps": ci_upper * 10_000.0,
        "p_value_raw": p_value,
    }


def unconditional_long_test(
    asset: str,
    period_label: str,
    session: str,
    sessions: pd.DataFrame,
    indices: np.ndarray,
) -> dict[str, object]:
    current = sessions[f"{session}_ret"].to_numpy(dtype=float)
    if not np.isfinite(current).all():
        raise ValueError("Missing session returns cannot enter the Long test.")
    if len(current) != indices.shape[1]:
        raise AssertionError("Bootstrap indices do not match Long-test observations.")
    estimate = float(current.mean())
    bootstrap_estimates = bootstrap_in_batches(
        indices,
        lambda batch_indices: current[batch_indices].mean(axis=1),
    )
    ci_lower, ci_upper, p_value = centered_bootstrap_inference(
        estimate,
        bootstrap_estimates,
    )
    print(f"[conditional] {asset} {period_label} {session} Long: n={len(current):,}")
    return {
        "asset": asset,
        "sample_period": period_label,
        "session": "nighttime" if session == "night" else "daytime",
        "preferred_rule": "Long",
        "statistic_name": "unconditional_mean",
        "observations": len(current),
        "lag_positive_observations": np.nan,
        "lag_negative_observations": np.nan,
        "lag_zero_observations": np.nan,
        "mean_after_positive_lag_bps": np.nan,
        "mean_after_negative_lag_bps": np.nan,
        "estimate_bps": estimate * 10_000.0,
        "ci_95_lower_bps": ci_lower * 10_000.0,
        "ci_95_upper_bps": ci_upper * 10_000.0,
        "p_value_raw": p_value,
    }


def run_conditional_tests(
    sessions_by_asset: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period_label in SAMPLE_PERIODS:
        period_sessions = {
            asset: sample_period(sessions, period_label)
            for asset, sessions in sessions_by_asset.items()
        }
        reversal_n = len(period_sessions["BTC"]) - 1
        if len(period_sessions["ETH"]) - 1 != reversal_n:
            raise AssertionError(
                "BTC and ETH conditional samples must align by calendar date."
            )
        reversal_indices = stationary_bootstrap_indices(reversal_n, rng)
        rows.append(
            reversal_conditional_test(
                "BTC", period_label, "night", period_sessions["BTC"], reversal_indices
            )
        )
        rows.append(
            reversal_conditional_test(
                "BTC", period_label, "day", period_sessions["BTC"], reversal_indices
            )
        )
        rows.append(
            reversal_conditional_test(
                "ETH", period_label, "day", period_sessions["ETH"], reversal_indices
            )
        )
        del reversal_indices

        long_indices = stationary_bootstrap_indices(len(period_sessions["ETH"]), rng)
        rows.append(
            unconditional_long_test(
                "ETH", period_label, "night", period_sessions["ETH"], long_indices
            )
        )
        del long_indices

    conditional = add_holm_and_significance(
        pd.DataFrame(rows),
        ["asset", "sample_period"],
    )
    columns = [
        "asset",
        "sample_period",
        "session",
        "preferred_rule",
        "statistic_name",
        "observations",
        "lag_positive_observations",
        "lag_negative_observations",
        "lag_zero_observations",
        "mean_after_positive_lag_bps",
        "mean_after_negative_lag_bps",
        "estimate_bps",
        "ci_95_lower_bps",
        "ci_95_upper_bps",
        "p_value_raw",
        "p_value_holm",
        "significance_10pct",
        "significance_5pct",
        "significance_1pct",
    ]
    return conditional[columns]


# ---------------------------------------------------------------------------
# Test 2: preferred strategy versus buy-and-hold
# ---------------------------------------------------------------------------


def preferred_vs_buyhold_test(
    asset: str,
    period_label: str,
    sessions: pd.DataFrame,
    cost_bps: int,
    indices: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    specification = ASSET_SPECIFICATIONS[asset]
    preferred_daily, preferred_diagnostics = backtest_daily_returns(
        sessions,
        specification["preferred_night"],
        specification["preferred_day"],
        cost_bps,
    )
    # The paper's passive benchmark uses the same turnover-cost function.  Its
    # constant Long/Long position therefore pays only the initial entry cost.
    buyhold_daily, buyhold_diagnostics = backtest_daily_returns(
        sessions,
        "long",
        "long",
        cost_bps,
    )
    paired = preferred_daily[["trading_date", "daily_net_return"]].merge(
        buyhold_daily[["trading_date", "daily_net_return"]],
        on="trading_date",
        how="inner",
        validate="one_to_one",
        suffixes=("_preferred", "_buyhold"),
    )
    if len(paired) != len(sessions) or len(paired) != indices.shape[1]:
        raise AssertionError(
            "Preferred and buy-and-hold daily returns are not jointly aligned."
        )
    preferred_growth = 1.0 + paired["daily_net_return_preferred"].to_numpy(dtype=float)
    buyhold_growth = 1.0 + paired["daily_net_return_buyhold"].to_numpy(dtype=float)
    if np.any(preferred_growth <= 0) or np.any(buyhold_growth <= 0):
        raise ValueError("All 1 + g values must be positive before taking logarithms.")
    differences = np.log(preferred_growth) - np.log(buyhold_growth)
    estimate = float(differences.mean())
    bootstrap_estimates = bootstrap_in_batches(
        indices,
        lambda batch_indices: differences[batch_indices].mean(axis=1),
    )
    ci_lower, ci_upper, p_value = centered_bootstrap_inference(
        estimate,
        bootstrap_estimates,
    )

    preferred_terminal = preferred_diagnostics["final_wealth"]
    buyhold_terminal = buyhold_diagnostics["final_wealth"]
    ratio = preferred_terminal / buyhold_terminal
    log_ratio_error = abs(np.log(ratio) - float(differences.sum()))
    if log_ratio_error > 1e-10:
        raise AssertionError(
            "Daily log differences do not reproduce the terminal-wealth ratio."
        )
    if (
        buyhold_diagnostics["trade_count"] != 1.0
        or buyhold_diagnostics["total_turnover"] != 1.0
    ):
        raise AssertionError(
            "Buy-and-hold must pay only one initial-entry turnover cost."
        )

    print(
        f"[strategy] {asset} {period_label} {cost_bps} bps: n={len(paired):,}, "
        f"preferred wealth={preferred_terminal:,.6f}, buyhold wealth={buyhold_terminal:,.6f}"
    )
    row = {
        "asset": asset,
        "sample_period": period_label,
        "preferred_strategy": specification["preferred_label"],
        "transaction_cost_bps": cost_bps,
        "observations": len(paired),
        "mean_daily_log_difference_bps": estimate * 10_000.0,
        "annualized_log_difference_percent": 365.0 * estimate * 100.0,
        "ci_95_lower_bps": ci_lower * 10_000.0,
        "ci_95_upper_bps": ci_upper * 10_000.0,
        "p_value_raw": p_value,
        "preferred_terminal_wealth": preferred_terminal,
        "buyhold_terminal_wealth": buyhold_terminal,
        "terminal_wealth_ratio": ratio,
    }
    checks = [
        {
            "asset": asset,
            "sample_period": period_label,
            "cost_bps": cost_bps,
            "path": "preferred",
            "positive_growth_factors": bool(np.all(preferred_growth > 0)),
            "log_ratio_identity_error": log_ratio_error,
        },
        {
            "asset": asset,
            "sample_period": period_label,
            "cost_bps": cost_bps,
            "path": "buyhold",
            "positive_growth_factors": bool(np.all(buyhold_growth > 0)),
            "log_ratio_identity_error": log_ratio_error,
        },
    ]
    return row, checks


def run_preferred_tests(
    sessions_by_asset: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    for period_label in SAMPLE_PERIODS:
        period_sessions = {
            asset: sample_period(sessions, period_label)
            for asset, sessions in sessions_by_asset.items()
        }
        n_observations = len(period_sessions["BTC"])
        if len(period_sessions["ETH"]) != n_observations:
            raise AssertionError(
                "BTC and ETH strategy samples must align by calendar date."
            )
        indices = stationary_bootstrap_indices(n_observations, rng)
        for asset in ("BTC", "ETH"):
            for cost_bps in TRANSACTION_COSTS_BPS:
                row, row_checks = preferred_vs_buyhold_test(
                    asset,
                    period_label,
                    period_sessions[asset],
                    cost_bps,
                    indices,
                )
                rows.append(row)
                checks.extend(row_checks)
        del indices

    preferred = add_holm_and_significance(
        pd.DataFrame(rows),
        ["asset", "sample_period"],
    )
    columns = [
        "asset",
        "sample_period",
        "preferred_strategy",
        "transaction_cost_bps",
        "observations",
        "mean_daily_log_difference_bps",
        "annualized_log_difference_percent",
        "ci_95_lower_bps",
        "ci_95_upper_bps",
        "p_value_raw",
        "p_value_holm",
        "preferred_terminal_wealth",
        "buyhold_terminal_wealth",
        "terminal_wealth_ratio",
        "significance_10pct",
        "significance_5pct",
        "significance_1pct",
    ]
    return preferred[columns], checks


# ---------------------------------------------------------------------------
# Validation and generated LaTeX tables
# ---------------------------------------------------------------------------


def validate_against_existing_results(
    preferred: pd.DataFrame,
) -> pd.DataFrame:
    if not EXISTING_RESULTS.exists():
        raise FileNotFoundError(EXISTING_RESULTS)
    existing = pd.read_csv(EXISTING_RESULTS)
    validation_rows: list[dict[str, object]] = []
    full = preferred[preferred["sample_period"].eq("2016-2025")]
    for row in full.itertuples(index=False):
        specification = ASSET_SPECIFICATIONS[row.asset]
        cutoff = specification["day_start_hour"]
        for path, night_rule, day_rule, calculated in (
            (
                "preferred",
                specification["preferred_night"],
                specification["preferred_day"],
                row.preferred_terminal_wealth,
            ),
            ("buyhold", "long", "long", row.buyhold_terminal_wealth),
        ):
            match = existing[
                existing["coin"].eq(row.asset)
                & existing["cutoff"].eq(cutoff)
                & existing["transaction_cost_bps"].eq(row.transaction_cost_bps)
                & existing["night_strategy"].eq(night_rule)
                & existing["day_strategy"].eq(day_rule)
            ]
            if len(match) != 1:
                raise AssertionError(
                    "Could not identify one existing terminal-wealth result."
                )
            existing_wealth = float(match.iloc[0]["final_wealth"])
            difference = float(calculated - existing_wealth)
            validation_rows.append(
                {
                    "asset": row.asset,
                    "cost_bps": int(row.transaction_cost_bps),
                    "path": path,
                    "calculated_terminal_wealth": float(calculated),
                    "existing_terminal_wealth": existing_wealth,
                    "difference": difference,
                    "status": "PASS" if abs(difference) < 1e-7 else "FAIL",
                }
            )
    validation = pd.DataFrame(validation_rows)
    if not validation["status"].eq("PASS").all():
        raise AssertionError(
            "A terminal-wealth path failed to reproduce the existing backtest."
        )
    return validation


def significance_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def latex_number(value: float, decimals: int = 2, stars: str = "") -> str:
    exponent = f"^{{{stars}}}" if stars else ""
    return f"${value:.{decimals}f}{exponent}$"


def latex_p_value(value: float) -> str:
    return f"${value:.3f}$"


def conditional_latex_rows(frame: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for row in frame.itertuples(index=False):
        stars = significance_stars(float(row.p_value_holm))
        ci = f"[{row.ci_95_lower_bps:.2f}, {row.ci_95_upper_bps:.2f}]"
        rows.append(
            " & ".join(
                [
                    row.asset,
                    row.sample_period,
                    row.session.title(),
                    row.preferred_rule,
                    f"{int(row.observations):,}",
                    latex_number(float(row.estimate_bps), 2, stars),
                    f"${ci}$",
                    latex_p_value(float(row.p_value_raw)),
                    latex_p_value(float(row.p_value_holm)),
                ]
            )
            + r" \\"
        )
    return rows


def preferred_latex_rows(frame: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for row in frame.itertuples(index=False):
        stars = significance_stars(float(row.p_value_holm))
        ci = f"[{row.ci_95_lower_bps:.2f}, {row.ci_95_upper_bps:.2f}]"
        rows.append(
            " & ".join(
                [
                    row.asset,
                    row.sample_period,
                    row.preferred_strategy,
                    str(int(row.transaction_cost_bps)),
                    f"{int(row.observations):,}",
                    latex_number(float(row.mean_daily_log_difference_bps), 2, stars),
                    latex_number(float(row.annualized_log_difference_percent), 2),
                    f"${ci}$",
                    latex_p_value(float(row.p_value_raw)),
                    latex_p_value(float(row.p_value_holm)),
                    latex_number(float(row.terminal_wealth_ratio), 2),
                ]
            )
            + r" \\"
        )
    return rows


def conditional_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    body = "\n".join(conditional_latex_rows(frame))
    return rf"""% Auto-generated by statistical_tests.py from {CONDITIONAL_CSV.name}; do not edit manually.
\begin{{table}}[!htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\begin{{tabular}}{{llllrrrrr}}
\toprule
Asset & Period & Session & Rule & $N$ & Estimate (bps) & 95\% CI & Raw $p$ & Holm $p$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\begin{{minipage}}{{0.96\textwidth}}\footnotesize
Notes: Reversal estimates equal the mean return after a negative same-session lag minus the mean return after a positive same-session lag. ETH nighttime reports its unconditional mean. Stationary bootstrap with 5,000 replications, expected block length seven calendar days, and seed 20260818. Stars use Holm-adjusted p-values: $^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$.
\end{{minipage}}
\end{{table}}
"""


def preferred_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    body = "\n".join(preferred_latex_rows(frame))
    return rf"""% Auto-generated by statistical_tests.py from {PREFERRED_CSV.name}; do not edit manually.
\begin{{table}}[!htbp]
\centering
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lllrrrrrrrr}}
\toprule
Asset & Period & Preferred strategy & Cost & $N$ & Mean log diff. (bps) & Annualized (\%) & 95\% CI & Raw $p$ & Holm $p$ & Wealth ratio \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\begin{{minipage}}{{0.96\textwidth}}\footnotesize
Notes: Cost is in basis points per unit of turnover. The paired statistic is the preferred-strategy daily log wealth increment minus buy-and-hold. Buy-and-hold pays only its initial entry cost. Stationary bootstrap with 5,000 replications, expected block length seven calendar days, and seed 20260818. Stars use Holm-adjusted p-values: $^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$.
\end{{minipage}}
\end{{table}}
"""


def write_latex_outputs() -> None:
    # Read the saved CSVs back so the LaTeX files are generated from the
    # machine-readable results rather than from manually entered values.
    conditional = pd.read_csv(CONDITIONAL_CSV)
    preferred = pd.read_csv(PREFERRED_CSV)
    conditional_full = conditional[conditional["sample_period"].eq("2016-2025")]
    preferred_full = preferred[preferred["sample_period"].eq("2016-2025")]
    conditional_sub = conditional[~conditional["sample_period"].eq("2016-2025")]
    preferred_sub = preferred[~preferred["sample_period"].eq("2016-2025")]

    CONDITIONAL_TEX.write_text(
        conditional_table(
            conditional_full,
            "Conditional Predictability in Preferred BTC and ETH Sessions, 2016--2025",
            "tab:conditional_predictability_full",
        ),
        encoding="utf-8",
    )
    PREFERRED_TEX.write_text(
        preferred_table(
            preferred_full,
            "Preferred Day--Night Strategies versus Buy-and-Hold, 2016--2025",
            "tab:preferred_vs_buyhold_full",
        ),
        encoding="utf-8",
    )
    SUBPERIOD_TEX.write_text(
        "% Auto-generated appendix tables; do not edit manually.\n\n"
        + conditional_table(
            conditional_sub,
            "Conditional Predictability by Subperiod",
            "tab:conditional_predictability_subperiods",
        )
        + "\n"
        + preferred_table(
            preferred_sub,
            "Preferred Strategies versus Buy-and-Hold by Subperiod",
            "tab:preferred_vs_buyhold_subperiods",
        ),
        encoding="utf-8",
    )


def interpretation_lines(
    conditional: pd.DataFrame,
    preferred: pd.DataFrame,
) -> list[str]:
    lines = [
        "## Concise interpretation",
        "",
        "A result is described as statistically significant only when its percentile confidence interval excludes zero and its Holm-adjusted p-value is below the stated level.",
        "",
    ]
    full_conditional = conditional[conditional["sample_period"].eq("2016-2025")]
    for row in full_conditional.itertuples(index=False):
        ci_excludes_zero = row.ci_95_lower_bps > 0 or row.ci_95_upper_bps < 0
        significant = ci_excludes_zero and row.p_value_holm < 0.05
        lines.append(
            f"- {row.asset} {row.session} {row.preferred_rule}: estimate {row.estimate_bps:.2f} bps per session "
            f"(95% CI [{row.ci_95_lower_bps:.2f}, {row.ci_95_upper_bps:.2f}], Holm p={row.p_value_holm:.3f}); "
            f"{'significant at 5%' if significant else 'not significant at 5%'} after the stated joint criterion."
        )
    lines.append("")
    full_preferred = preferred[preferred["sample_period"].eq("2016-2025")]
    for row in full_preferred.itertuples(index=False):
        ci_excludes_zero = row.ci_95_lower_bps > 0 or row.ci_95_upper_bps < 0
        significant = ci_excludes_zero and row.p_value_holm < 0.05
        lines.append(
            f"- {row.asset} {row.preferred_strategy}, {int(row.transaction_cost_bps)} bps: mean daily log difference "
            f"{row.mean_daily_log_difference_bps:.2f} bps (95% CI [{row.ci_95_lower_bps:.2f}, "
            f"{row.ci_95_upper_bps:.2f}], Holm p={row.p_value_holm:.3f}); "
            f"{'significant at 5%' if significant else 'not significant at 5%'} after the stated joint criterion."
        )
    return lines


def write_validation_report(
    quality_rows: list[dict[str, object]],
    sessions_by_asset: dict[str, pd.DataFrame],
    conditional: pd.DataFrame,
    preferred: pd.DataFrame,
    terminal_validation: pd.DataFrame,
    path_checks: list[dict[str, object]],
) -> None:
    max_terminal_difference = float(terminal_validation["difference"].abs().max())
    checks_frame = pd.DataFrame(path_checks)
    if not checks_frame["positive_growth_factors"].all():
        raise AssertionError("A strategy path contains a nonpositive growth factor.")
    max_log_identity_error = float(checks_frame["log_ratio_identity_error"].max())

    lines = [
        "# Statistical-test validation",
        "",
        "## Method",
        "",
        "- Session construction, same-session lagged signals, and turnover costs match `btc_eth_backtest.py`.",
        "- Timestamps and session cutoffs are UTC. BTC uses 08:00--20:00 daytime; ETH uses 05:00--17:00 daytime.",
        "- Stationary bootstrap: 5,000 replications, expected block length seven calendar days, seed 20260818.",
        "- Conditional return/lag pairs are formed before resampling, so bootstrap block boundaries do not create artificial lags.",
        "- Conditional reversal tests drop the first unavailable lag. Strategy paths preserve the notebook initialization: a dynamic rule holds cash when its first lag is unavailable.",
        "- Buy-and-hold is Long/Long under the original turnover function and pays one initial-entry cost only.",
        "",
        "## Data quality",
        "",
        "| Asset | Hourly rows | Missing hours | Duplicate timestamps | Complete session dates | First date | Last date |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    quality_lookup = {row["asset"]: row for row in quality_rows}
    for asset in ("BTC", "ETH"):
        quality = quality_lookup[asset]
        sessions = sessions_by_asset[asset]
        lines.append(
            f"| {asset} | {quality['rows_after_cleaning']:,} | {quality['missing_hours']} | "
            f"{quality['duplicate_timestamps']} | {len(sessions):,} | "
            f"{sessions['trading_date'].min().date()} | {sessions['trading_date'].max().date()} |"
        )

    lines.extend(
        [
            "",
            "## Backtest reproduction",
            "",
            "| Asset | Cost (bps) | Path | Calculated wealth | Existing wealth | Difference | Status |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in terminal_validation.itertuples(index=False):
        lines.append(
            f"| {row.asset} | {row.cost_bps} | {row.path} | {row.calculated_terminal_wealth:.9f} | "
            f"{row.existing_terminal_wealth:.9f} | {row.difference:.3e} | {row.status} |"
        )
    lines.extend(
        [
            "",
            f"Maximum absolute terminal-wealth difference: `{max_terminal_difference:.3e}`. No material difference from the existing backtest tables was found.",
            f"Maximum daily-log-difference/terminal-wealth-ratio identity error: `{max_log_identity_error:.3e}`.",
            "",
            "## Observation counts",
            "",
            "The script prints every test's observation count while running. The generated CSV files retain those counts, including positive, negative, and zero lag counts for each reversal test.",
            "",
        ]
    )
    lines.extend(interpretation_lines(conditional, preferred))
    lines.append("")
    VALIDATION_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    sessions_by_asset: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, object]] = []
    for asset, specification in ASSET_SPECIFICATIONS.items():
        hourly, quality = load_coin_data(asset)
        if any(
            quality[key] != 0
            for key in (
                "timestamp_mismatches",
                "duplicate_timestamps",
                "missing_hours",
                "extra_hours",
                "missing_close_values",
            )
        ):
            raise AssertionError(
                f"{asset} hourly data failed quality validation: {quality}"
            )
        sessions = build_day_night_sessions(hourly, specification["day_start_hour"])
        if sessions[["night_ret", "day_ret"]].isna().any().any():
            raise AssertionError("Missing session returns are not permitted.")
        if not sessions[["n_night_bars", "n_day_bars"]].eq(12).all().all():
            raise AssertionError("Each session must contain exactly 12 hourly returns.")
        assert_consecutive_daily_dates(sessions, f"{asset} full sessions")
        sessions_by_asset[asset] = sessions
        quality_rows.append(quality)
        print(
            f"[data] {asset}: {len(hourly):,} hourly prices, {len(sessions):,} complete trading dates, "
            f"{sessions['trading_date'].min().date()} to {sessions['trading_date'].max().date()}"
        )

    if not sessions_by_asset["BTC"]["trading_date"].equals(
        sessions_by_asset["ETH"]["trading_date"]
    ):
        raise AssertionError("BTC and ETH complete trading dates do not align.")

    conditional = run_conditional_tests(sessions_by_asset, rng)
    conditional.to_csv(CONDITIONAL_CSV, index=False)
    print(f"Saved {CONDITIONAL_CSV} ({len(conditional)} rows)")

    preferred, path_checks = run_preferred_tests(sessions_by_asset, rng)
    preferred.to_csv(PREFERRED_CSV, index=False)
    print(f"Saved {PREFERRED_CSV} ({len(preferred)} rows)")

    terminal_validation = validate_against_existing_results(preferred)
    write_latex_outputs()
    write_validation_report(
        quality_rows,
        sessions_by_asset,
        conditional,
        preferred,
        terminal_validation,
        path_checks,
    )

    expected_outputs = [
        CONDITIONAL_CSV,
        PREFERRED_CSV,
        CONDITIONAL_TEX,
        PREFERRED_TEX,
        SUBPERIOD_TEX,
        VALIDATION_MD,
    ]
    missing_outputs = [
        path
        for path in expected_outputs
        if not path.exists() or path.stat().st_size == 0
    ]
    if missing_outputs:
        raise AssertionError(f"Missing or empty output files: {missing_outputs}")
    print("All statistical tests and validation checks passed.")
    for path in expected_outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
