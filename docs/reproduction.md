# Reproducing the research results

The public repository contains the source, tests, exact selected XGBoost
parameters by date, portfolio returns, and a populated results notebook.
The large historical datasets and fitted pickles remain in local/S3 storage.
No cloud write or model training is needed to read `model_results.ipynb`.

## Required private/local inputs for full replay

Restore these paths from the research workspace or its artifact storage:

| Path | Contents |
|---|---|
| `SP500_Historical_Data.csv` | Historical daily OHLCV / adjusted-close data |
| `local_cache/athena_panel_cc5c45cb64a96a00.parquet` | Validated normalized panel used in the completed run |
| `local_cache/historical_features_731a66adc016.parquet` | Raw weekly features and returns used for portfolio evaluation |
| `artifacts/ml/continuous_260w_v1/regression/as_of=YYYY-MM-DD/` | Dated XGBoost pickles and exact metadata |
| `artifacts/ml/continuous_260w_v2/lightgbm/as_of=YYYY-MM-DD/` | Dated LightGBM pickles and exact metadata for the companion audit |
| Both source run roots' `backtest/stock_scores_and_quintiles.parquet` | Archived scores used to verify model replay |

The existing names identify the original stored artifacts; they are kept so
hash checks and saved pickle imports continue to work. Do not manufacture
replacement manifests or substitute today's data for a historical replay.
The source snapshot ends on 2026-02-18 and contains 216 model dates from
2022-01-05, of which 215 have realized forward returns.

## Rebuild historical inputs

`build_sp500_features.py` validates NYSE daily prices and produces weekly
features. `sp500_s3_backfill.ipynb` contains the established weekly Parquet
upload workflow. `historical_panel_s3.py` constructs the sector-neutral panel
from weekly S3 features. `training_data.py` loads that panel via Athena.

Review your bucket/profile and the notebook's dry-run plan before writing.
Current processed S3 paths use:

```text
features_parquet/feature_version=close_features_v1/data_version=kaggle_2026_02/week_date=YYYY-MM-DD/
panels_parquet/feature_version=close_features_v1/data_version=kaggle_2026_02/panel_version=sector_neutral_active_returns_v2/week_date=YYYY-MM-DD/
```

The scheduled raw ingestion layer and prediction partitions described in the
README are planned work, not an already deployed service.

## Replay, backtest and publish results

Run from the repository root with the restored inputs:

```bash
# Independently refit and replay all dated source models before tuning.
python v3_incumbent_replay.py

# Reuse verified replay; evaluate the protected candidates chronologically.
python v3_incumbent_backtest.py --workers 4

# Re-score fitted selections without retraining.
python v3_incumbent_backtest.py --score-only

# Export the best XGBoost tables/configuration/chart and populate the notebook.
python build_results_notebook.py
```

Detailed outputs remain under
`artifacts/ml/continuous_260w_v3_incumbent_protected/`: candidate and fold
validation tables, dated selection reasons, selected pickles, weekly scores,
portfolio returns, yearly results, and the replay audit. They are ignored by
Git. The export script copies only small aggregate/portfolio outputs and
dated XGBoost configurations into `reports/best_model/` and `configs/`.

The test suite uses synthetic data and small committed configuration fixtures;
it does not require these private inputs, AWS access, or a Kafka broker.

See [the detailed selection methodology](research_methodology.md) for the
incumbent-preservation rules and companion model audit. The README focuses
on the best XGBoost result rather than the experiment history.
