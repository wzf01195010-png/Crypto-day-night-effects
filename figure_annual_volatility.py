"""FIGURE 2 — Annualized Volatility by Year.

This script corresponds only to:
    "Annualized Volatility by Year"

For each calendar year from 2016 through 2025, volatility is the sample
standard deviation of that year's daily net returns, annualized by sqrt(365)
exactly once.  This is the same volatility definition used in Table 9, applied
separately to each year instead of to the full 2016--2025 sample.
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
    backtest_daily_returns,
    build_day_night_sessions,
    load_coin_data,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUTPUT_DIR / "Annualized_volatility_year.png"
DATA_CSV_PATH = OUTPUT_DIR / "annualized_volatility_by_year.csv"
LATEX_PATH = OUTPUT_DIR / "annualized_volatility_by_year.tex"

TITLE = "Annualized Volatility by Year"
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
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.unicode_minus": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_daily_returns() -> pd.DataFrame:
    """Rebuild the six daily return series from the existing backtest logic."""
    session_cache: dict[tuple[str, int], pd.DataFrame] = {}
    daily_returns: dict[str, pd.Series] = {}
    for spec in SERIES_SPECS:
        cache_key = (str(spec["asset"]), int(spec["cutoff"]))
        if cache_key not in session_cache:
            prices, quality = load_coin_data(str(spec["asset"]))
            if quality["missing_close_values"] != 0:
                raise ValueError(f"Missing closes detected for {spec['asset']}.")
            session_cache[cache_key] = build_day_night_sessions(
                prices, int(spec["cutoff"])
            )
        daily, _ = backtest_daily_returns(
            session_cache[cache_key],
            str(spec["night"]),
            str(spec["day"]),
            TRANSACTION_COST_BPS,
        )
        daily_returns[str(spec["column"])] = daily.set_index("trading_date")[
            "daily_net_return"
        ].astype(float)

    data = pd.DataFrame(daily_returns)
    if data.isna().any().any():
        raise AssertionError("The common daily return sample contains missing values.")
    data.index.name = "date"
    return data


def calculate_yearly_volatility(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute calendar-year daily-return volatility, annualized once."""
    expected_years = pd.Index(range(START_YEAR, END_YEAR + 1), name="year")
    grouped = daily_returns.groupby(daily_returns.index.year)
    counts = grouped.count().reindex(expected_years)
    if counts.isna().any().any() or (counts < 2).any().any():
        raise AssertionError(
            "Every plotted year must contain at least two daily returns."
        )

    output = grouped.std(ddof=1).reindex(expected_years) * np.sqrt(365.0)
    output.columns = [f"{column}_annualized_volatility" for column in output.columns]
    if output.isna().any().any():
        raise AssertionError("Yearly annualized volatility contains missing values.")
    return output


def draw_figure(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    for spec in SERIES_SPECS:
        ax.plot(
            data.index,
            data[f"{spec['column']}_annualized_volatility"],
            label=str(spec["label"]),
            color=str(spec["color"]),
            linestyle=str(spec["linestyle"]),
            linewidth=1.35,
            marker="o",
            markersize=3.2,
        )

    ax.set_title(TITLE, pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annualized Volatility")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(START_YEAR - 0.2, END_YEAR + 0.2)
    ax.set_xticks(range(START_YEAR, END_YEAR + 1))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.set_ylim(bottom=0)
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_latex_snippet() -> None:
    snippet = r"""% Figure 2: Annualized Volatility by Year
\begin{figure}[!htbp]
    \centering
    \includegraphics[width=\textwidth]{outputs/figures/Annualized_volatility_year.png}
    \caption{Annualized Volatility by Year. Each point is the sample standard deviation of daily net returns within the corresponding calendar year, annualized by $\sqrt{365}$. The figure compares Buy-and-Hold, Long/Reversal, and Reversal/Reversal for BTC and Ethereum over the 2016--2025 common sample at 0 bps.}
    \label{fig:annualized-volatility-by-year}
\end{figure}
"""
    LATEX_PATH.write_text(snippet, encoding="utf-8")


def main() -> None:
    set_publication_style()
    daily_returns = load_daily_returns()
    figure_data = calculate_yearly_volatility(daily_returns)
    figure_data.to_csv(DATA_CSV_PATH, float_format="%.12g")
    draw_figure(figure_data)
    write_latex_snippet()

    print("FIGURE 2 script: Annualized Volatility by Year")
    print(
        f"Daily sample: {daily_returns.index.min().date()} to {daily_returns.index.max().date()}"
    )
    print(f"Plotted years: {figure_data.index.min()} to {figure_data.index.max()}")
    print(
        "Each point: within-year daily standard deviation; annualization: sqrt(365) exactly once"
    )
    for path in (PNG_PATH, DATA_CSV_PATH, LATEX_PATH):
        print(path)


if __name__ == "__main__":
    main()
