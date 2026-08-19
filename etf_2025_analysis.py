from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"
INITIAL_WEALTH = 100.0
ANNUALIZATION_DAYS = 252.0

ASSETS = {
    "IBIT": {
        "source": PROJECT_DIR
        / "data_hourly_crypto"
        / "IBIT_1h_2024_07_05_to_2026_07_08.csv",
        "primary_label": "Reversal / Reversal",
        "night_strategy": "reversal",
        "day_strategy": "reversal",
    },
    "ETHA": {
        "source": PROJECT_DIR
        / "data_hourly_crypto"
        / "ETHA_1h_2024_07_23_to_2026_07_06.csv",
        "primary_label": "Long / Reversal",
        "night_strategy": "long",
        "day_strategy": "reversal",
    },
}


def position(strategy: str, previous_returns: pd.Series) -> np.ndarray:
    if strategy == "long":
        return np.ones(len(previous_returns), dtype=float)
    if strategy == "reversal":
        return -np.sign(np.nan_to_num(previous_returns.to_numpy(dtype=float), nan=0.0))
    raise ValueError(f"Unsupported strategy: {strategy}")


def max_drawdown_session_level(
    night_returns: np.ndarray, day_returns: np.ndarray
) -> float:
    session_returns = np.column_stack([night_returns, day_returns]).reshape(-1)
    wealth = INITIAL_WEALTH * np.cumprod(1.0 + session_returns)
    wealth_with_initial = np.concatenate(([INITIAL_WEALTH], wealth))
    running_peak = np.maximum.accumulate(wealth_with_initial)
    return float(np.min(wealth_with_initial / running_peak - 1.0))


def compute_metrics(
    sessions: pd.DataFrame,
    night_strategy: str,
    day_strategy: str,
    cost_bps: int,
) -> dict:
    previous_night = sessions["night_return"].shift(1)
    previous_day = sessions["day_return"].shift(1)
    evaluation_mask = sessions["trading_date"].dt.year.eq(2025).to_numpy()

    night_positions = position(night_strategy, previous_night)[evaluation_mask]
    day_positions = position(day_strategy, previous_day)[evaluation_mask]
    positions = np.column_stack([night_positions, day_positions]).reshape(-1)
    realized_returns = np.column_stack(
        [
            sessions.loc[evaluation_mask, "night_return"].to_numpy(dtype=float),
            sessions.loc[evaluation_mask, "day_return"].to_numpy(dtype=float),
        ]
    ).reshape(-1)
    previous_positions = np.concatenate(([0.0], positions[:-1]))
    turnover = np.abs(positions - previous_positions)
    gross_session_returns = positions * realized_returns
    cost_rate = cost_bps / 10_000.0
    net_session_returns = (1.0 + gross_session_returns) * (
        1.0 - cost_rate * turnover
    ) - 1.0
    night_returns = net_session_returns.reshape(-1, 2)[:, 0]
    day_returns = net_session_returns.reshape(-1, 2)[:, 1]
    daily_returns = (1.0 + night_returns) * (1.0 + day_returns) - 1.0
    session_returns = np.column_stack([night_returns, day_returns]).reshape(-1)
    final_wealth = float(INITIAL_WEALTH * np.prod(1.0 + session_returns))
    daily_std = float(np.std(daily_returns, ddof=1))

    return {
        "final_wealth": final_wealth,
        "total_return": float(final_wealth / INITIAL_WEALTH - 1.0),
        "mdd": max_drawdown_session_level(night_returns, day_returns),
        "sharpe_ratio": float(
            np.mean(daily_returns) / daily_std * math.sqrt(ANNUALIZATION_DAYS)
        ),
        "annualized_volatility": float(daily_std * math.sqrt(ANNUALIZATION_DAYS)),
        "transaction_cost_bps": int(cost_bps),
        "trade_count": int(np.count_nonzero(turnover)),
        "total_turnover": float(turnover.sum()),
        "trading_days": int(len(daily_returns)),
        "sessions": int(len(session_returns)),
        "night_position_counts": {
            str(int(key)): int(value)
            for key, value in zip(*np.unique(night_positions, return_counts=True))
        },
        "day_position_counts": {
            str(int(key)): int(value)
            for key, value in zip(*np.unique(day_positions, return_counts=True))
        },
    }


def prepare_asset(asset: str, config: dict) -> dict:
    raw = pd.read_csv(config["source"])
    raw["datetime_utc"] = pd.to_datetime(raw["open_time"], utc=True, errors="coerce")
    raw["datetime_et"] = raw["datetime_utc"].dt.tz_convert("America/New_York")
    raw["date_et"] = raw["datetime_et"].dt.normalize().dt.tz_localize(None)
    raw["time_et"] = raw["datetime_et"].dt.strftime("%H:%M:%S")
    raw = raw.sort_values("datetime_utc").reset_index(drop=True)

    hourly_2025 = raw[raw["date_et"].dt.year.eq(2025)].copy()
    hourly_rows = []
    for row in hourly_2025.itertuples(index=False):
        hourly_rows.append(
            {
                "asset": asset,
                "datetime_utc": row.datetime_utc.isoformat(),
                "date_et": row.date_et.strftime("%Y-%m-%d"),
                "time_et": row.time_et,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": int(row.volume),
                "adjusted_close": float(row.adj_close),
            }
        )

    daily = (
        raw.groupby("date_et", sort=True)
        .agg(
            open=("open", "first"),
            close=("close", "last"),
            bar_count=("close", "size"),
        )
        .reset_index()
    )
    daily["previous_trading_date"] = daily["date_et"].shift(1)
    daily["previous_close"] = daily["close"].shift(1)
    sessions = daily.dropna(subset=["previous_close"]).copy()
    sessions["night_return"] = sessions["open"] / sessions["previous_close"] - 1.0
    sessions["day_return"] = sessions["close"] / sessions["open"] - 1.0
    sessions["close_to_close_return"] = (
        sessions["close"] / sessions["previous_close"] - 1.0
    )
    sessions["decomposition_error"] = (
        (1.0 + sessions["night_return"]) * (1.0 + sessions["day_return"])
        - 1.0
        - sessions["close_to_close_return"]
    )
    sessions = (
        sessions[
            sessions["date_et"].between(
                pd.Timestamp("2024-12-31"), pd.Timestamp("2025-12-31")
            )
        ]
        .rename(columns={"date_et": "trading_date"})
        .reset_index(drop=True)
    )
    if len(sessions) != 251 or sessions["trading_date"].dt.year.eq(2025).sum() != 250:
        raise ValueError(f"Unexpected session coverage for {asset}")

    session_rows = []
    for index, row in sessions.iterrows():
        session_rows.append(
            {
                "trading_date": row["trading_date"].strftime("%Y-%m-%d"),
                "previous_trading_date": row["previous_trading_date"].strftime(
                    "%Y-%m-%d"
                ),
                "previous_close": float(row["previous_close"]),
                "open": float(row["open"]),
                "close": float(row["close"]),
                "bar_count": int(row["bar_count"]),
                "night_return": float(row["night_return"]),
                "day_return": float(row["day_return"]),
                "close_to_close_return": float(row["close_to_close_return"]),
                "decomposition_error": float(row["decomposition_error"]),
                "warmup_only": bool(index == 0),
            }
        )

    metrics = {
        str(cost_bps): {
            "primary": compute_metrics(
                sessions,
                config["night_strategy"],
                config["day_strategy"],
                cost_bps,
            ),
            "buy_and_hold": compute_metrics(sessions, "long", "long", cost_bps),
        }
        for cost_bps in (0, 1, 2)
    }

    date_counts = hourly_2025.groupby("date_et").size()
    return {
        "source_file": str(config["source"]),
        "primary_label": config["primary_label"],
        "hourly_rows": hourly_rows,
        "session_rows": session_rows,
        "metrics": metrics,
        "quality": {
            "hourly_rows": int(len(hourly_2025)),
            "trading_dates": int(hourly_2025["date_et"].nunique()),
            "first_datetime_utc": hourly_2025["datetime_utc"].min().isoformat(),
            "last_datetime_utc": hourly_2025["datetime_utc"].max().isoformat(),
            "duplicate_timestamps": int(hourly_2025["datetime_utc"].duplicated().sum()),
            "missing_ohlc_cells": int(
                hourly_2025[["open", "high", "low", "close"]].isna().sum().sum()
            ),
            "nonpositive_price_rows": int(
                (hourly_2025[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
            ),
            "zero_volume_rows": int(hourly_2025["volume"].eq(0).sum()),
            "early_close_dates": [
                date.strftime("%Y-%m-%d")
                for date, count in date_counts.items()
                if int(count) != 7
            ],
            "max_abs_decomposition_error": float(
                sessions["decomposition_error"].abs().max()
            ),
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "evaluation_start": "2025-01-02",
            "evaluation_end": "2025-12-31",
            "timezone": "America/New_York",
            "session_definition": "Night = previous trading-day close to current open; Day = current open to current close",
            "signal_timing": "Reversal uses the immediately preceding return from the same session; 2024-12-31 is warm-up only",
            "initial_wealth": INITIAL_WEALTH,
            "annualization_days": ANNUALIZATION_DAYS,
            "transaction_cost_bps": 0,
            "mdd_definition": "Minimum session-level wealth divided by running session-level peak minus one",
            "sharpe_definition": "Mean daily return divided by sample standard deviation of daily returns, annualized by sqrt(252), zero risk-free rate",
            "volatility_definition": "Sample standard deviation of daily returns annualized by sqrt(252)",
        },
        "assets": {
            asset: prepare_asset(asset, config) for asset, config in ASSETS.items()
        },
    }
    output_path = OUTPUT_DIR / "analysis.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    metric_rows = []
    for asset, asset_payload in payload["assets"].items():
        for cost_bps, strategies in asset_payload["metrics"].items():
            for strategy_name, metrics in strategies.items():
                metric_rows.append(
                    {
                        "asset": asset,
                        "transaction_cost_bps": int(cost_bps),
                        "strategy": strategy_name,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if not isinstance(value, dict)
                        },
                    }
                )
    pd.DataFrame(metric_rows).to_csv(
        OUTPUT_DIR / "etf_2025_metrics.csv",
        index=False,
    )
    print(
        json.dumps(
            {
                asset: {
                    "quality": payload["assets"][asset]["quality"],
                    "metrics": payload["assets"][asset]["metrics"],
                }
                for asset in ASSETS
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
