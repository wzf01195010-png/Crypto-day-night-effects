# Versioned paper outputs

These files are the version-controlled outputs used to audit the manuscript.
They were generated from the Python scripts in the repository on August 19,
2026.

## Figures

| File | Content |
| --- | --- |
| `BTC_cutoff_best_strategy_barchart.png` | Best BTC strategy at each UTC cutoff |
| `ETH_cutoff_best_strategy_barchart.png` | Best ETH strategy at each UTC cutoff |
| `Annualized_volatility_year.png` | Calendar-year annualized volatility |
| `annualized_sharpe_ratio_by_year.png` | Calendar-year annualized Sharpe ratio |
| `maximum_drawdown_by_year.png` | Calendar-year maximum drawdown |

## Tables and figure data

| File | Content |
| --- | --- |
| `btc_eth_all_metrics.csv` | Complete core-cutoff strategy metrics |
| `cutoff_strategy_full_results.csv` | Complete 12-cutoff strategy search |
| `cutoff_strategy_best_by_cutoff.csv` | Best candidate at every cutoff |
| `annualized_volatility_by_year.csv` | Data underlying the volatility figure |
| `annual_sharpe_mdd_data.csv` | Data underlying the Sharpe and MDD figures |
| `conditional_predictability_tests.csv` | Conditional-predictability inference |
| `preferred_vs_buyhold_tests.csv` | Preferred-strategy versus buy-and-hold inference |
| `etf_2025_metrics.csv` | IBIT and ETHA 2025 metrics at 0, 1, and 2 bps |

The `source_file` column in `btc_eth_all_metrics.csv` records only the expected
repository-relative input path. Raw market data are not distributed here; see
the data-availability note in the main README.
