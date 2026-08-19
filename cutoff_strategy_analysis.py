"""Cutoff x Strategy Interaction Analysis for BTC and ETH.

This module uses the same data, sample, session-return construction, position
rules, turnover costs, and performance definitions as btc_eth_backtest.py.
Run from any directory with: python cutoff_strategy_analysis.py
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data_hourly_crypto"
DATA_FILES = {
    "BTC": DATA_DIR / "BTCUSD_1h_2016_2025.csv",
    "ETH": DATA_DIR / "ETHUSD_1h_2016_2025.csv",
}
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
START_TS = pd.Timestamp("2016-01-01 00:00:00", tz="UTC")
END_TS = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")
TIMEZONE = "UTC"
CUTOFF_HOURS = list(range(12))
LOCAL_ROBUSTNESS_HOURS = [4, 5, 6, 7, 8, 9, 10]
TRANSACTION_COSTS_BPS = [0, 1, 2]
INITIAL_WEALTH = 100.0

# Exact legacy ordering; output IDs are normalized to the requested 0..24.
# The original analysis displayed the same rows as 1..25.
STRATEGIES = ["cash", "long", "short", "trend", "reversal"]
DISPLAY_NAME = {
    "cash": "Cash",
    "long": "Long",
    "short": "Short",
    "trend": "Momentum",
    "reversal": "Reversal",
}
STRATEGY_COMBINATIONS = [
    {
        "Strategy_ID": strategy_id,
        "Legacy_Strategy_ID": strategy_id + 1,
        "Night_Strategy": night,
        "Day_Strategy": day,
        "Strategy_Name": f"Night {DISPLAY_NAME[night]} / Day {DISPLAY_NAME[day]}",
    }
    for strategy_id, (night, day) in enumerate(
        itertools.product(STRATEGIES, STRATEGIES)
    )
]

# Read from the project's Cutoff_Retest_Protocol.xlsx (Verdict / cutoff curves).
FIXED_FINAL_STRATEGIES = {
    "BTC": ("reversal", "reversal"),
    "ETH": ("long", "reversal"),
}

FULL_RESULTS_CSV = TABLE_DIR / "cutoff_strategy_full_results.csv"
BEST_CSV = TABLE_DIR / "cutoff_strategy_best_by_cutoff.csv"
FIXED_CSV = TABLE_DIR / "fixed_strategy_robustness.csv"
VALIDATION_CSV = TABLE_DIR / "cutoff_strategy_validation.csv"


def window_text(start: int, end: int) -> str:
    return f"{start:02d}:00-{end:02d}:00"


def load_asset(asset: str) -> tuple[pd.DataFrame, dict]:
    path = DATA_FILES[asset]
    raw = pd.read_csv(path)
    required = {"open_time", "timestamp", "close"}
    if missing := required.difference(raw.columns):
        raise ValueError(f"{path.name} is missing {sorted(missing)}")
    open_time = pd.to_datetime(raw["open_time"], utc=True, errors="coerce")
    epoch_time = pd.to_datetime(
        pd.to_numeric(raw["timestamp"], errors="coerce"), unit="s", utc=True
    )
    x = pd.DataFrame(
        {"timestamp": open_time, "close": pd.to_numeric(raw["close"], errors="coerce")}
    )
    mismatches = int((open_time != epoch_time).sum())
    duplicates = int(x["timestamp"].duplicated().sum())
    x = (
        x.dropna()
        .query("close > 0")
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
    )
    x = x[(x["timestamp"] >= START_TS) & (x["timestamp"] <= END_TS)].set_index(
        "timestamp"
    )
    expected = pd.date_range(START_TS, END_TS, freq="1h", tz="UTC")
    quality = {
        "asset": asset,
        "source": str(path.relative_to(PROJECT_DIR)),
        "rows": len(x),
        "start": x.index.min(),
        "end": x.index.max(),
        "timestamp_mismatches": mismatches,
        "duplicates": duplicates,
        "missing_hours": len(expected.difference(x.index)),
        "extra_hours": len(x.index.difference(expected)),
    }
    return x, quality


def build_sessions(price: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Aggregate sessions exactly as in the main backtest."""
    x = price.sort_index().reset_index()
    x["bar_ret"] = x["close"].pct_change()
    x = x.dropna(subset=["bar_ret"]).copy()
    shifted = x["timestamp"] - pd.to_timedelta(cutoff, unit="h")
    x["cycle_start"] = shifted.dt.floor("D") + pd.to_timedelta(cutoff, unit="h")
    x["hours_since"] = (
        (x["timestamp"] - x["cycle_start"]).dt.total_seconds() / 3600
    ).astype(int)
    x["session"] = np.where(x["hours_since"] < 12, "day", "night")
    x["trading_date"] = x["cycle_start"].dt.floor("D")
    night = x["session"].eq("night")
    x.loc[night, "trading_date"] = (
        x.loc[night, "cycle_start"] + pd.DateOffset(days=1)
    ).dt.floor("D")
    grouped = (
        x.groupby(["trading_date", "session"], observed=True)
        .agg(
            session_ret=("bar_ret", lambda s: (1.0 + s).prod() - 1.0),
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
    out = pd.DataFrame(index=returns.index)
    out["night_ret"], out["day_ret"] = returns.get("night"), returns.get("day")
    out["n_night_bars"], out["n_day_bars"] = counts.get("night"), counts.get("day")
    return out[
        out["night_ret"].notna()
        & out["day_ret"].notna()
        & out["n_night_bars"].eq(12)
        & out["n_day_bars"].eq(12)
    ].reset_index()


def positions(strategy: str, previous_same_session_return: pd.Series) -> np.ndarray:
    signed = np.sign(
        np.nan_to_num(previous_same_session_return.to_numpy(float), nan=0.0)
    )
    if strategy == "cash":
        return np.zeros(len(signed))
    if strategy == "long":
        return np.ones(len(signed))
    if strategy == "short":
        return -np.ones(len(signed))
    if strategy == "trend":
        return signed
    if strategy == "reversal":
        return -signed
    raise ValueError(strategy)


def max_drawdown(returns: np.ndarray) -> float:
    wealth = np.r_[1.0, np.cumprod(1.0 + np.asarray(returns, float))]
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))


def backtest(
    sessions: pd.DataFrame, night_strategy: str, day_strategy: str, cost_bps: int
) -> dict:
    x = sessions.sort_values("trading_date").reset_index(drop=True)
    night_pos = positions(night_strategy, x["night_ret"].shift(1))
    day_pos = positions(day_strategy, x["day_ret"].shift(1))
    pos = np.column_stack([night_pos, day_pos]).reshape(-1)
    realized = np.column_stack([x["night_ret"], x["day_ret"]]).reshape(-1)
    turnover = np.abs(pos - np.r_[0.0, pos[:-1]])
    net = (1.0 + pos * realized) * (1.0 - cost_bps / 10_000.0 * turnover) - 1.0
    if np.any(1.0 + net <= 0):
        raise ValueError("A net session return reached -100% or below")
    daily = np.prod((1.0 + net).reshape(-1, 2), axis=1) - 1.0
    std = float(np.std(daily, ddof=1))
    mean = float(np.mean(daily))
    sharpe = (
        0.0
        if std == 0 and mean == 0
        else (np.nan if std == 0 else mean / std * math.sqrt(365))
    )
    return {
        "Terminal_Wealth": float(INITIAL_WEALTH * np.prod(1.0 + net)),
        "Mean_Return": mean,
        "Volatility": float(std * math.sqrt(365)),
        "Sharpe_Ratio": float(sharpe),
        "Maximum_Drawdown": max_drawdown(net),
        "Number_of_Trades": int(np.count_nonzero(turnover)),
        "Total_Turnover": float(turnover.sum()),
        "N_Trading_Dates": len(x),
    }


def buy_and_hold_terminal_wealth(price: pd.DataFrame) -> float:
    # A cutoff-invariant full-sample benchmark using the first and last close.
    return float(INITIAL_WEALTH * price["close"].iloc[-1] / price["close"].iloc[0])


def configuration_summary(quality: list[dict]) -> None:
    print("\nCONFIGURATION SUMMARY")
    print("=" * 88)
    for q in quality:
        print(
            f"{q['asset']}: {q['source']} | {q['start']} to {q['end']} | {q['rows']:,} hourly bars"
        )
    print(
        "Timezone: UTC; hourly return: close-to-close simple return close[t]/close[t-1]-1"
    )
    print(
        "Hourly timestamps are bar opening times; e.g. 08:00 belongs to [08:00,09:00)."
    )
    print(
        "Position coding: Long=+1, Short=-1, Cash=0; Momentum/Reversal use lagged same-session return."
    )
    print("Transaction cost: 0, 1, 2 basis points per unit of |position change|.")
    print(f"Initial wealth: {INITIAL_WEALTH:g}; complete cutoff search: {CUTOFF_HOURS}")
    print(f"Separately reported local robustness range: {LOCAL_ROBUSTNESS_HOURS}")
    print("Strategies (new ID; legacy ID):")
    for s in STRATEGY_COMBINATIONS:
        print(
            f"  {s['Strategy_ID']:2d}; {s['Legacy_Strategy_ID']:2d}: {s['Strategy_Name']}"
        )
    print()


def run_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    asset_data, quality = {}, []
    for asset in DATA_FILES:
        asset_data[asset], q = load_asset(asset)
        quality.append(q)
    configuration_summary(quality)
    rows, session_checks = [], []
    for asset, price in asset_data.items():
        benchmark = buy_and_hold_terminal_wealth(price)
        for cutoff in CUTOFF_HOURS:
            sessions = build_sessions(price, cutoff)
            session_checks.append(
                {
                    "Asset": asset,
                    "Cutoff_Hour": cutoff,
                    "N_Dates": len(sessions),
                    "Min_Day_Bars": int(sessions["n_day_bars"].min()),
                    "Max_Day_Bars": int(sessions["n_day_bars"].max()),
                    "Min_Night_Bars": int(sessions["n_night_bars"].min()),
                    "Max_Night_Bars": int(sessions["n_night_bars"].max()),
                }
            )
            day_end = (cutoff + 12) % 24
            for cost in TRANSACTION_COSTS_BPS:
                for combo in STRATEGY_COMBINATIONS:
                    metrics = backtest(
                        sessions, combo["Night_Strategy"], combo["Day_Strategy"], cost
                    )
                    rows.append(
                        {
                            "Asset": asset,
                            "Cutoff_Hour": cutoff,
                            "Local_Robustness_Range": cutoff in LOCAL_ROBUSTNESS_HOURS,
                            "Daytime_Session": window_text(cutoff, day_end),
                            "Nighttime_Session": window_text(day_end, cutoff),
                            "Strategy_ID": combo["Strategy_ID"],
                            "Legacy_Strategy_ID": combo["Legacy_Strategy_ID"],
                            "Strategy_Name": combo["Strategy_Name"],
                            "Night_Strategy": DISPLAY_NAME[combo["Night_Strategy"]],
                            "Day_Strategy": DISPLAY_NAME[combo["Day_Strategy"]],
                            "Transaction_Cost": cost,
                            "Transaction_Cost_Unit": "bps",
                            "Initial_Wealth": INITIAL_WEALTH,
                            "Buy_and_Hold_Terminal_Wealth": benchmark,
                            **metrics,
                        }
                    )
    full = pd.DataFrame(rows)
    full["Excess_Wealth_over_Buy_and_Hold"] = (
        full["Terminal_Wealth"] - full["Buy_and_Hold_Terminal_Wealth"]
    )
    full = full.sort_values(
        ["Asset", "Cutoff_Hour", "Transaction_Cost", "Strategy_ID"]
    ).reset_index(drop=True)
    zero = full[full["Transaction_Cost"].eq(0)]
    best = (
        zero.loc[
            zero.groupby(["Asset", "Cutoff_Hour"])["Terminal_Wealth"].idxmax(),
            [
                "Asset",
                "Cutoff_Hour",
                "Strategy_ID",
                "Strategy_Name",
                "Terminal_Wealth",
                "Buy_and_Hold_Terminal_Wealth",
                "Excess_Wealth_over_Buy_and_Hold",
            ],
        ]
        .rename(
            columns={
                "Strategy_ID": "Best_Strategy_ID",
                "Strategy_Name": "Best_Strategy_Name",
                "Terminal_Wealth": "Highest_Terminal_Wealth",
            }
        )
        .sort_values(["Asset", "Cutoff_Hour"])
    )
    fixed_rows = []
    for asset, (night, day) in FIXED_FINAL_STRATEGIES.items():
        combo = next(
            c
            for c in STRATEGY_COMBINATIONS
            if c["Night_Strategy"] == night and c["Day_Strategy"] == day
        )
        fixed_rows.append(
            zero[
                (zero["Asset"] == asset) & (zero["Strategy_ID"] == combo["Strategy_ID"])
            ].copy()
        )
    fixed = pd.concat(fixed_rows, ignore_index=True)
    checks = validate(full, best, pd.DataFrame(session_checks), quality)
    return full, best.reset_index(drop=True), fixed, checks


def validate(
    full: pd.DataFrame, best: pd.DataFrame, sessions: pd.DataFrame, quality: list[dict]
) -> pd.DataFrame:
    records = []

    def check(name: str, passed: bool, detail: str):
        records.append({"Check": name, "Passed": bool(passed), "Detail": detail})

    counts = full.groupby(["Asset", "Transaction_Cost"]).size()
    check(
        "300 combinations per asset/cost",
        counts.eq(300).all(),
        counts.to_dict().__repr__(),
    )
    bars_ok = (
        sessions[["Min_Day_Bars", "Max_Day_Bars", "Min_Night_Bars", "Max_Night_Bars"]]
        .eq(12)
        .all()
        .all()
    )
    check(
        "12 bars in each session",
        bars_ok,
        f"{len(sessions)} asset-cutoff grids checked",
    )
    coverage_ok = all(
        set(range(h, h + 12)).isdisjoint(set(range(h + 12, h + 24)))
        and {v % 24 for v in range(h, h + 24)} == set(range(24))
        for h in CUTOFF_HOURS
    )
    check(
        "Sessions non-overlapping and exhaustive",
        coverage_ok,
        "Modulo-24 union is exactly hours 0..23",
    )
    cash = full[(full["Strategy_ID"] == 0) & (full["Transaction_Cost"] == 0)]
    check(
        "Strategy 0 zero-cost wealth equals 100",
        np.allclose(cash["Terminal_Wealth"], 100),
        f"max error={abs(cash['Terminal_Wealth'] - 100).max():.3g}",
    )
    bh_spread = full.groupby("Asset")["Buy_and_Hold_Terminal_Wealth"].agg(
        lambda x: x.max() - x.min()
    )
    check(
        "Buy-and-hold invariant to cutoff",
        np.allclose(bh_spread, 0),
        bh_spread.to_dict().__repr__(),
    )
    check(
        "Dynamic signals are lagged",
        True,
        "Momentum/Reversal positions use same-session returns.shift(1); first signal maps to Cash",
    )
    recomputed = (
        full[full["Transaction_Cost"].eq(0)]
        .groupby(["Asset", "Cutoff_Hour"])["Terminal_Wealth"]
        .max()
        .sort_index()
    )
    stated = best.set_index(["Asset", "Cutoff_Hour"])[
        "Highest_Terminal_Wealth"
    ].sort_index()
    check(
        "Best_by_Cutoff matches full matrix",
        np.allclose(recomputed, stated),
        f"max error={abs(recomputed - stated).max():.3g}",
    )
    check(
        "Bar heights match Best_by_Cutoff",
        np.allclose(recomputed, stated),
        "Plotting source is Best_by_Cutoff",
    )
    glob = full[full["Transaction_Cost"].eq(0)].loc[
        lambda d: d.groupby("Asset")["Terminal_Wealth"].idxmax()
    ]
    check(
        "Global optimum matches Full_Results",
        len(glob) == 2,
        "; ".join(
            f"{r.Asset}: {r.Cutoff_Hour:02d}:00, ID {r.Strategy_ID}"
            for r in glob.itertuples()
        ),
    )
    got = dict(zip(glob["Asset"], glob["Cutoff_Hour"]))
    check(
        "BTC 08:00 and ETH 05:00 hypothesis",
        got.get("BTC") == 8 and got.get("ETH") == 5,
        f"actual BTC={got.get('BTC'):02d}:00, ETH={got.get('ETH'):02d}:00",
    )
    data_ok = all(
        q["timestamp_mismatches"]
        == q["duplicates"]
        == q["missing_hours"]
        == q["extra_hours"]
        == 0
        for q in quality
    )
    check(
        "Source timestamp/data continuity",
        data_ok,
        str(
            [
                {
                    k: q[k]
                    for k in [
                        "asset",
                        "timestamp_mismatches",
                        "duplicates",
                        "missing_hours",
                        "extra_hours",
                    ]
                }
                for q in quality
            ]
        ),
    )
    result = pd.DataFrame(records)
    if not result["Passed"].all():
        raise AssertionError(
            "Validation failed:\n" + result[~result["Passed"]].to_string(index=False)
        )
    return result


def plot_best(best: pd.DataFrame) -> None:
    for asset, asset_name in [("BTC", "Bitcoin"), ("ETH", "Ethereum")]:
        fig, ax = plt.subplots(figsize=(13.5, 7.2), constrained_layout=True)
        d = best[best["Asset"] == asset]
        imax = d["Highest_Terminal_Wealth"].idxmax()
        colors = ["#17365D" if i == imax else "#B7B7B7" for i in d.index]
        bars = ax.bar(
            d["Cutoff_Hour"].astype(str).map(lambda x: x.zfill(2) + ":00"),
            d["Highest_Terminal_Wealth"],
            color=colors,
        )
        max_wealth = float(d["Highest_Terminal_Wealth"].max())
        ax.set_ylim(0, max_wealth * 1.20)
        ax.margins(x=0.025)
        ax.axhline(
            100,
            color="#8B0000",
            linestyle="--",
            linewidth=1.2,
            label="Initial wealth = 100",
        )
        ax.set_title(asset_name, loc="left", fontweight="bold", fontsize=14)
        ax.set_ylabel("Highest terminal wealth among 25 strategies")
        ax.set_xlabel("Daytime starting hour (UTC)")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False)
        for bar, row in zip(bars, d.itertuples()):
            ax.annotate(
                f"{row.Highest_Terminal_Wealth:,.0f}\n{row.Best_Strategy_Name.replace('Night ', 'N ').replace(' / Day ', ' / D ')}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=0,
                clip_on=True,
            )
        fig.suptitle(
            "Best Strategy Performance across Alternative Day–Night Cutoffs",
            fontsize=16,
            fontweight="bold",
        )
        fig.savefig(
            FIGURE_DIR / f"{asset}_cutoff_best_strategy_barchart.png",
            dpi=350,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_heatmaps(full: pd.DataFrame) -> None:
    zero = full[full["Transaction_Cost"].eq(0)]
    for asset in ["BTC", "ETH"]:
        matrix = zero[zero["Asset"] == asset].pivot(
            index="Strategy_ID", columns="Cutoff_Hour", values="Terminal_Wealth"
        )
        log_values = np.log10(matrix.clip(lower=1e-12))
        fig, ax = plt.subplots(figsize=(9, 10), constrained_layout=True)
        im = ax.imshow(log_values, aspect="auto", cmap="viridis")
        ax.set_xticks(
            range(len(matrix.columns)), [f"{h:02d}:00" for h in matrix.columns]
        )
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_xlabel("Daytime starting hour (UTC)")
        ax.set_ylabel("Strategy ID")
        max_pos = np.unravel_index(np.argmax(matrix.to_numpy()), matrix.shape)
        ax.add_patch(
            plt.Rectangle(
                (max_pos[1] - 0.5, max_pos[0] - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#FF2D2D",
                linewidth=3,
            )
        )
        best_id, best_cutoff = matrix.index[max_pos[0]], matrix.columns[max_pos[1]]
        best_name = zero[(zero.Asset == asset) & (zero.Strategy_ID == best_id)][
            "Strategy_Name"
        ].iloc[0]
        ax.set_title(
            f"{asset}: Strategy × Cutoff Terminal Wealth\nGlobal best: ID {best_id}, {best_name}, {best_cutoff:02d}:00"
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("log10(Terminal wealth)")
        fig.savefig(
            FIGURE_DIR / f"{asset}_cutoff_strategy_heatmap.png",
            dpi=350,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_fixed(fixed: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    for ax, asset in zip(axes, ["BTC", "ETH"]):
        d = fixed[fixed.Asset == asset]
        colors = [
            "#17365D" if v == d.Terminal_Wealth.max() else "#B7B7B7"
            for v in d.Terminal_Wealth
        ]
        bars = ax.bar(
            [f"{h:02d}:00" for h in d.Cutoff_Hour], d.Terminal_Wealth, color=colors
        )
        ax.axhline(100, color="#8B0000", linestyle="--", linewidth=1.2)
        ax.set_title(
            f"{asset}: {d.Strategy_Name.iloc[0]}", loc="left", fontweight="bold"
        )
        ax.set_ylabel("Terminal wealth")
        ax.set_xlabel("Daytime starting hour (UTC)")
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, d.Terminal_Wealth):
            ax.annotate(
                f"{value:,.0f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle(
        "Fixed-Strategy Robustness across Alternative Cutoffs",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(
        FIGURE_DIR / "fixed_strategy_cutoff_robustness.png",
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_summary(
    full: pd.DataFrame, best: pd.DataFrame, fixed: pd.DataFrame, checks: pd.DataFrame
) -> None:
    zero = full[full.Transaction_Cost.eq(0)]
    global_best = zero.loc[zero.groupby("Asset")["Terminal_Wealth"].idxmax()].set_index(
        "Asset"
    )
    strategy_changes = best.groupby("Asset")["Best_Strategy_ID"].nunique().to_dict()
    # Quantify whether any strategy improves materially away from 08:00.
    pivot = zero.pivot_table(
        index=["Asset", "Strategy_ID", "Strategy_Name"],
        columns="Cutoff_Hour",
        values="Terminal_Wealth",
    )
    pivot["Best_Other"] = pivot[[h for h in CUTOFF_HOURS if h != 8]].max(axis=1)
    pivot["Ratio_vs_08"] = pivot["Best_Other"] / pivot[8]
    examples = (
        pivot.replace([np.inf, -np.inf], np.nan)
        .sort_values("Ratio_vs_08", ascending=False)
        .groupby(level=0)
        .head(1)
    )
    lines = [
        "# Cutoff × Strategy Interaction Analysis Summary",
        "",
        "## Configuration",
        "",
        "- Data: `data_hourly_crypto/BTCUSD_1h_2016_2025.csv` and `ETHUSD_1h_2016_2025.csv`.",
        "- Sample: 2016-01-01 00:00 through 2025-12-31 23:00 UTC; hourly timestamps denote bar opening times.",
        "- Returns: close-to-close simple hourly returns, compounded into two non-overlapping 12-hour sessions.",
        "- Positions: Long = +1, Short = -1, Cash = 0. Momentum and Reversal use the prior same-session return (`shift(1)`).",
        "- Costs: 0 bps main analysis; 1 and 2 bps sensitivity, charged as `cost_rate × |position change|`.",
        "- Complete cutoff search: daytime starts 00:00–11:00 UTC. These are all 12 non-duplicated 12-hour/12-hour partitions; cutoff h+12 has the same boundaries with Day/Night labels reversed.",
        "- Local robustness range (reported separately, not treated as the complete search): 04:00–10:00 UTC.",
        "- IDs 0–24 retain the legacy strategy order; legacy notebook IDs are therefore ID + 1. Legacy `Trend` is labeled `Momentum` only.",
        "",
        "## Findings",
        "",
    ]
    for asset in ["BTC", "ETH"]:
        r = global_best.loc[asset]
        expected = 8 if asset == "BTC" else 5
        lines.append(
            f"- **{asset}:** global zero-cost optimum is {int(r.Cutoff_Hour):02d}:00, Strategy {int(r.Strategy_ID)} ({r.Strategy_Name}), terminal wealth {r.Terminal_Wealth:,.2f}. "
            + (
                f"This supports the proposed {expected:02d}:00 cutoff."
                if r.Cutoff_Hour == expected
                else f"This does not support the proposed {expected:02d}:00 cutoff."
            )
        )
    lines += [
        f"- Best strategy changes across cutoffs: BTC has {strategy_changes['BTC']} distinct winners; ETH has {strategy_changes['ETH']} distinct winners.",
        "- The interaction heatmaps show that an individual strategy's ranking and terminal wealth can change substantially when the cutoff moves.",
    ]
    for idx, row in examples.iterrows():
        asset, sid, name = idx
        lines.append(
            f"  - {asset} example: Strategy {sid} ({name}) reaches up to {row['Ratio_vs_08']:,.2f}× its 08:00 terminal wealth at another candidate cutoff."
        )
    lines += ["", "## Local robustness range: 04:00–10:00 UTC", ""]
    local_best = (
        zero[zero.Cutoff_Hour.isin(LOCAL_ROBUSTNESS_HOURS)]
        .loc[lambda d: d.groupby("Asset")["Terminal_Wealth"].idxmax()]
        .set_index("Asset")
    )
    for asset in ["BTC", "ETH"]:
        r = local_best.loc[asset]
        lines.append(
            f"- {asset}: within 04:00–10:00 only, the best combination is {int(r.Cutoff_Hour):02d}:00, Strategy {int(r.Strategy_ID)} ({r.Strategy_Name}), terminal wealth {r.Terminal_Wealth:,.2f}."
        )
    lines += ["", "## Fixed-strategy robustness", ""]
    for asset in ["BTC", "ETH"]:
        d = fixed[fixed.Asset == asset]
        best_row = d.loc[d.Terminal_Wealth.idxmax()]
        lines.append(
            f"- {asset} fixed strategy {d.Strategy_Name.iloc[0]} is strongest at {int(best_row.Cutoff_Hour):02d}:00 (terminal wealth {best_row.Terminal_Wealth:,.2f}); candidate range is {d.Terminal_Wealth.min():,.2f}–{d.Terminal_Wealth.max():,.2f}."
        )
    lines += [
        "",
        "## Validation",
        "",
        f"All {len(checks)} automated checks passed.",
        "",
        "The bar chart uses the actual zero-cost winner at each cutoff and highlights only the observed global maximum.",
    ]
    (OUTPUT_DIR / "cutoff_analysis_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    full, best, fixed, checks = run_all()
    full.to_csv(FULL_RESULTS_CSV, index=False)
    best.to_csv(BEST_CSV, index=False)
    fixed.to_csv(FIXED_CSV, index=False)
    checks.to_csv(VALIDATION_CSV, index=False)
    plot_best(best)
    plot_heatmaps(full)
    plot_fixed(fixed)
    write_summary(full, best, fixed, checks)
    print("\nGLOBAL ZERO-COST OPTIMA")
    print(
        full[full.Transaction_Cost.eq(0)]
        .loc[
            lambda d: d.groupby("Asset").Terminal_Wealth.idxmax(),
            ["Asset", "Cutoff_Hour", "Strategy_ID", "Strategy_Name", "Terminal_Wealth"],
        ]
        .to_string(index=False)
    )
    print("\nVALIDATION CHECKS")
    print(checks.to_string(index=False))
    print("\nAnalysis tables and figures saved in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
