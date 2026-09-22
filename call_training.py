#!/usr/bin/env python3
"""Load the catalog panel and run weekly rolling regression training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from training import FEATURES, RollingPanelTrainer, TrainingConfig
from training_data import DEFAULT_DATABASE, DEFAULT_OUTPUT, DEFAULT_TABLE, load_athena_panel
from scoring import DEFAULT_RAW


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--catalog", default="AwsDataCatalog")
    parser.add_argument("--profile", default="admin")
    parser.add_argument("--region", default="ap-southeast-2")
    parser.add_argument("--workgroup", default="primary")
    parser.add_argument("--athena-output", default=DEFAULT_OUTPUT)
    parser.add_argument("--data-start", default="2017-01-01")
    parser.add_argument("--data-end")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--input-parquet", type=Path, help="Explicit offline source; skips Athena")
    parser.add_argument("--raw-features", type=Path, default=DEFAULT_RAW,
                        help="Unchanged raw weekly return source for portfolio validation")
    windows = parser.add_mutually_exclusive_group()
    windows.add_argument("--window-years", type=int, help="Explicit legacy calendar-year window")
    windows.add_argument("--window-weeks", type=int, help="Weekly snapshots (default: 260)")
    parser.add_argument("--model-start", default="2022-01-01", help="First as-of date (default: 2022-01-01)")
    parser.add_argument("--model-end", help="Last as-of date (default: latest snapshot)")
    parser.add_argument("--max-models", type=int, help="Limit dates for an explicitly partial smoke run")
    parser.add_argument("--grid", choices=["compact", "reference"], default="compact")
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--cv-gap-weeks", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent model dates; CPU budget is workers x threads")
    parser.add_argument("--features", nargs="+", default=list(FEATURES), choices=FEATURES)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ml/continuous_260w_v2/regression"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "training.log")])
    config = TrainingConfig(data_start=args.data_start, window_years=(args.window_years if args.window_years is not None else 5),
                            window_weeks=(None if args.window_years is not None else
                                          (args.window_weeks if args.window_weeks is not None else 260)),
                            cv_splits=args.cv_splits, cv_gap_weeks=args.cv_gap_weeks,
                            grid=args.grid, threads=args.threads, features=tuple(args.features))
    if args.input_parquet:
        panel = pd.read_parquet(args.input_parquet)
        if args.data_end:
            date_column = next(c for c in panel if c.lower() == "week_date")
            panel = panel.loc[pd.to_datetime(panel[date_column]).le(args.data_end)]
        source = {"mode": "explicit_local_parquet", "path": str(args.input_parquet.resolve())}
    else:
        panel, source = load_athena_panel(
            database=args.database, table=args.table, catalog=args.catalog,
            start=args.data_start, end=args.data_end, profile=args.profile,
            region=args.region, workgroup=args.workgroup, output_location=args.athena_output,
            refresh=args.refresh_data,
        )
    raw = pd.read_parquet(args.raw_features)
    summary = RollingPanelTrainer(config, args.output_dir, workers=args.workers).run(
        panel, model_start=args.model_start, model_end=args.model_end,
        max_models=args.max_models, source_metadata=source, raw_returns=raw,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
