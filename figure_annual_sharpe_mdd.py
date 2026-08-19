"""YEARLY SHARPE RATIO AND MAXIMUM DRAWDOWN FIGURES.

This script replaces the former combined full-sample bar chart with two
publication-ready yearly line figures:

1. Annualized Sharpe Ratio by Year.
2. Maximum Drawdown by Year.

Each figure compares the same six 0-bps return series: Buy-and-Hold,
Long/Reversal, and Reversal/Reversal for both BTC and ETH. BTC is red, ETH is
blue, and line style identifies the strategy consistently across both assets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from statistical_tests import (
    EXISTING_RESULTS,
    INITIAL_WEALTH,
    backtest_daily_returns,
    build_day_night_sessions,
    load_coin_data,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHARPE_PNG_PATH = OUTPUT_DIR / "annualized_sharpe_ratio_by_year.png"
MDD_PNG_PATH = OUTPUT_DIR / "maximum_drawdown_by_year.png"
YEARLY_CSV_PATH = OUTPUT_DIR / "annual_sharpe_mdd_data.csv"
SUMMARY_CSV_PATH = OUTPUT_DIR / "annual_sharpe_mdd_summary.csv"
LATEX_PATH = OUTPUT_DIR / "annual_sharpe_mdd_figures.tex"
VALIDATION_PATH = OUTPUT_DIR / "annual_sharpe_mdd_validation.md"

TRANSACTION_COST_BPS = 0
START_YEAR = 2016
END_YEAR = 2025

SERIES_SPECS = (
    {
        "asset": "BTC",
        "label": "BTC Buy-and-Hold",
        "column": "BTC_Buy_and_Hold",
        "night": "long",
        "day": "long",
        "cutoff": 8,
        "color": "#C44E52",
        "linestyle": "-",
    },
    {
        "asset": "BTC",
        "label": "BTC Long/Reversal",
        "column": "BTC_Long_Reversal",
        "night": "long",
        "day": "reversal",
        "cutoff": 8,
        "color": "#C44E52",
        "linestyle": "--",
    },
    {
        "asset": "BTC",
        "label": "BTC Reversal/Reversal",
        "column": "BTC_Reversal_Reversal",
        "night": "reversal",
        "day": "reversal",
        "cutoff": 8,
        "color": "#C44E52",
        "linestyle": ":",
    },
    {
        "asset": "ETH",
        "label": "ETH Buy-and-Hold",
        "column": "ETH_Buy_and_Hold",
        "night": "long",
        "day": "long",
        "cutoff": 5,
        "color": "#4C72B0",
        "linestyle": "-",
    },
    {
        "asset": "ETH",
        "label": "ETH Long/Reversal",
        "column": "ETH_Long_Reversal",
        "night": "long",
        "day": "reversal",
        "cutoff": 5,
        "color": "#4C72B0",
        "linestyle": "--",
    },
    {
        "asset": "ETH",
        "label": "ETH Reversal/Reversal",
        "column": "ETH_Reversal_Reversal",
        "night": "reversal",
        "day": "reversal",
        "cutoff": 5,
        "color": "#4C72B0",
        "linestyle": ":",
    },
)


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.unicode_minus": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def maximum_drawdown_from_session_returns(session_returns: np.ndarray) -> float:
    """Return MDD from an interleaved night/day session-return path."""
    wealth = np.concatenate(
        ([INITIAL_WEALTH], INITIAL_WEALTH * np.cumprod(1.0 + session_returns))
    )
    running_peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / running_peak - 1.0))


def load_six_series() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Reproduce the six series and validate full-sample metrics."""
    existing = pd.read_csv(EXISTING_RESULTS)
    session_cache: dict[tuple[str, int], pd.DataFrame] = {}
    daily_by_column: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []

    for spec in SERIES_SPECS:
        cache_key = (str(spec["asset"]), int(spec["cutoff"]))
        if cache_key not in session_cache:
            prices, quality = load_coin_data(str(spec["asset"]))
            if quality["missing_close_values"] != 0:
                raise ValueError(f"Missing closes detected for {spec['asset']}.")
            session_cache[cache_key] = build_day_night_sessions(
                prices, int(spec["cutoff"])
            )

        daily, diagnostics = backtest_daily_returns(
            session_cache[cache_key],
            str(spec["night"]),
            str(spec["day"]),
            TRANSACTION_COST_BPS,
        )
        daily_by_column[str(spec["column"])] = daily.copy()

        returns = daily["daily_net_return"].to_numpy(dtype=float)
        daily_std = float(np.std(returns, ddof=1))
        full_sharpe = float(np.mean(returns) / daily_std * np.sqrt(365.0))
        full_volatility = daily_std * np.sqrt(365.0)
        full_session_returns = (
            daily[["night_net_return", "day_net_return"]]
            .to_numpy(dtype=float)
            .reshape(-1)
        )
        full_mdd = maximum_drawdown_from_session_returns(full_session_returns)

        table_row = existing[
            existing["coin"].eq(spec["asset"])
            & existing["cutoff"].eq(spec["cutoff"])
            & existing["transaction_cost_bps"].eq(TRANSACTION_COST_BPS)
            & existing["night_strategy"].eq(spec["night"])
            & existing["day_strategy"].eq(spec["day"])
        ]
        if len(table_row) != 1:
            raise ValueError(
                f"Expected one Main Results row for {spec['label']}; found {len(table_row)}."
            )
        table_row = table_row.iloc[0]
        comparisons = {
            "final_wealth": (
                float(diagnostics["final_wealth"]),
                float(table_row["final_wealth"]),
            ),
            "annualized_volatility": (
                full_volatility,
                float(table_row["annualized_daily_volatility"]),
            ),
            "sharpe_ratio": (full_sharpe, float(table_row["sharpe_ratio"])),
            "maximum_drawdown": (full_mdd, float(table_row["full_period_mdd"])),
        }
        if not all(
            np.isclose(reproduced, reported, rtol=0, atol=1e-7)
            for reproduced, reported in comparisons.values()
        ):
            raise AssertionError(f"Main Results validation failed for {spec['label']}.")

        summary_rows.append(
            {
                "strategy": spec["label"],
                "asset": spec["asset"],
                "cutoff": f"{spec['cutoff']}-{spec['cutoff']}-{spec['cutoff']}",
                "transaction_cost_bps": TRANSACTION_COST_BPS,
                "sample_start": daily["trading_date"].min().date().isoformat(),
                "sample_end": daily["trading_date"].max().date().isoformat(),
                "n_daily_returns": len(returns),
                "final_wealth": comparisons["final_wealth"][0],
                "annualized_volatility": full_volatility,
                "sharpe_ratio": full_sharpe,
                "maximum_drawdown": full_mdd,
                "validation": "PASS",
            }
        )

    return daily_by_column, pd.DataFrame(summary_rows)


def calculate_yearly_metrics(daily_by_column: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate within-year Sharpe ratios and session-path MDDs."""
    years = pd.Index(range(START_YEAR, END_YEAR + 1), name="year")
    output = pd.DataFrame(index=years)

    for spec in SERIES_SPECS:
        column = str(spec["column"])
        daily = daily_by_column[column].copy()
        daily["year"] = daily["trading_date"].dt.year

        yearly_sharpe: dict[int, float] = {}
        yearly_mdd: dict[int, float] = {}
        for year, group in daily.groupby("year", sort=True):
            returns = group["daily_net_return"].to_numpy(dtype=float)
            if len(returns) < 2:
                raise AssertionError(
                    f"Too few observations for {spec['label']} in {year}."
                )
            daily_std = float(np.std(returns, ddof=1))
            yearly_sharpe[int(year)] = float(
                np.mean(returns) / daily_std * np.sqrt(365.0)
            )
            session_returns = (
                group[["night_net_return", "day_net_return"]]
                .to_numpy(dtype=float)
                .reshape(-1)
            )
            yearly_mdd[int(year)] = maximum_drawdown_from_session_returns(
                session_returns
            )

        output[f"{column}_annualized_sharpe_ratio"] = pd.Series(yearly_sharpe).reindex(
            years
        )
        output[f"{column}_maximum_drawdown"] = pd.Series(yearly_mdd).reindex(years)

    if output.isna().any().any():
        raise AssertionError("Yearly Sharpe/MDD output contains missing values.")
    return output


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Year")
    ax.set_xlim(START_YEAR - 0.2, END_YEAR + 0.2)
    ax.set_xticks(range(START_YEAR, END_YEAR + 1))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, ncol=2)


def draw_metric_figure(
    data: pd.DataFrame,
    metric_suffix: str,
    title: str,
    ylabel: str,
    output_path: Path,
    percentage_axis: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    for spec in SERIES_SPECS:
        ax.plot(
            data.index,
            data[f"{spec['column']}_{metric_suffix}"],
            label=str(spec["label"]),
            color=str(spec["color"]),
            linestyle=str(spec["linestyle"]),
            linewidth=1.45,
            marker="o",
            markersize=3.2,
        )

    ax.set_title(title, pad=12)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", linewidth=0.8)
    if percentage_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(top=0.02)
    style_axis(ax)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_latex_snippet() -> None:
    snippet = r"""% Annualized Sharpe Ratio by Year
\begin{figure}[!htbp]
    \centering
    \includegraphics[width=\textwidth]{outputs/figures/annualized_sharpe_ratio_by_year.png}
    \caption{Annualized Sharpe Ratio by Year. Each point is calculated from daily net returns within the corresponding calendar year and annualized by $\sqrt{365}$. Results compare Buy-and-Hold, Long/Reversal, and Reversal/Reversal for BTC and Ethereum at 0 bps.}
    \label{fig:annualized-sharpe-ratio-by-year}
\end{figure}

% Maximum Drawdown by Year
\begin{figure}[!htbp]
    \centering
    \includegraphics[width=\textwidth]{outputs/figures/maximum_drawdown_by_year.png}
    \caption{Maximum Drawdown by Year. Each point is the most negative peak-to-trough drawdown from the complete interleaved night--day wealth path within the corresponding calendar year. Results compare Buy-and-Hold, Long/Reversal, and Reversal/Reversal for BTC and Ethereum at 0 bps.}
    \label{fig:maximum-drawdown-by-year}
\end{figure}
"""
    LATEX_PATH.write_text(snippet, encoding="utf-8")


def write_validation_summary(summary: pd.DataFrame) -> None:
    lines = [
        "# Figure execution and validation summary",
        "",
        "- Six curves in every figure: Buy-and-Hold, Long/Reversal, and Reversal/Reversal for BTC and ETH.",
        "- Colors: BTC red; ETH blue.",
        "- Line styles: Buy-and-Hold solid; Long/Reversal dashed; Reversal/Reversal dotted.",
        "- Transaction cost: 0 bps.",
        "- BTC cutoff: 8--8--8; ETH cutoff: 5--5--5.",
        "- Annualized Sharpe ratio: within-year mean daily net return divided by its sample standard deviation, multiplied by sqrt(365).",
        "- Annual MDD: minimum wealth/running-peak minus one over each year's complete interleaved night--day session path.",
        "- Full-sample final wealth, volatility, Sharpe ratio, and MDD match the Main Results file within 1e-7 for all six series.",
        "",
        summary[
            ["strategy", "sharpe_ratio", "maximum_drawdown", "validation"]
        ].to_string(index=False),
        "",
    ]
    VALIDATION_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_publication_style()
    daily_by_column, summary = load_six_series()
    yearly = calculate_yearly_metrics(daily_by_column)

    yearly.to_csv(YEARLY_CSV_PATH, float_format="%.12g")
    summary.to_csv(SUMMARY_CSV_PATH, index=False, float_format="%.12g")
    draw_metric_figure(
        yearly,
        "annualized_sharpe_ratio",
        "Annualized Sharpe Ratio by Year",
        "Annualized Sharpe Ratio",
        SHARPE_PNG_PATH,
    )
    draw_metric_figure(
        yearly,
        "maximum_drawdown",
        "Maximum Drawdown by Year",
        "Maximum Drawdown",
        MDD_PNG_PATH,
        percentage_axis=True,
    )
    write_latex_snippet()
    write_validation_summary(summary)

    print("YEARLY SHARPE AND MDD FIGURES: six series each")
    print(f"Plotted years: {yearly.index.min()} to {yearly.index.max()}")
    print(summary[["strategy", "validation"]].to_string(index=False))
    for path in (
        SHARPE_PNG_PATH,
        MDD_PNG_PATH,
        YEARLY_CSV_PATH,
        SUMMARY_CSV_PATH,
        LATEX_PATH,
        VALIDATION_PATH,
    ):
        print(path)


if __name__ == "__main__":
    main()
