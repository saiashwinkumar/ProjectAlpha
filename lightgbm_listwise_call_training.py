#!/usr/bin/env python3
"""Load ProjectAlpha's panel and run the separate 260-week LightGBM ranker."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from lightgbm_listwise_training import RankingConfig, RollingListwiseTrainer
from training import FEATURES
from training_data import DEFAULT_DATABASE, DEFAULT_OUTPUT, DEFAULT_TABLE, load_athena_panel
from scoring import DEFAULT_RAW


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, help="Explicit offline panel; skips Athena")
    parser.add_argument("--raw-features", type=Path, default=DEFAULT_RAW,
                        help="Unchanged raw weekly return source for portfolio validation")
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--catalog", default="AwsDataCatalog")
    parser.add_argument("--profile", default="admin")
    parser.add_argument("--region", default="ap-southeast-2")
    parser.add_argument("--workgroup", default="primary")
    parser.add_argument("--athena-output", default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--data-start", default="2017-01-01")
    parser.add_argument("--data-end")
    parser.add_argument("--model-start", default="2022-01-01")
    parser.add_argument("--model-end")
    parser.add_argument("--max-models", type=int, help="Explicitly limit dates for a partial run")
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--cv-gap-weeks", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent dates; CPU budget is workers times threads")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ml/continuous_260w_v2/lightgbm"))
    args = parser.parse_args()
    cfg = RankingConfig(data_start=args.data_start, features=tuple(args.features), threads=args.threads,
                        cv_splits=args.cv_splits, cv_gap_weeks=args.cv_gap_weeks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "training.log")])
    if args.input_parquet:
        panel = pd.read_parquet(args.input_parquet)
        if args.data_end:
            column = next(c for c in panel if c.lower() == "week_date")
            panel = panel.loc[pd.to_datetime(panel[column]).le(args.data_end)]
        source = {"mode": "explicit_local_parquet", "path": str(args.input_parquet.resolve())}
    else:
        panel, source = load_athena_panel(database=args.database, table=args.table, catalog=args.catalog,
                                        start=args.data_start, end=args.data_end, profile=args.profile,
                                        region=args.region, workgroup=args.workgroup,
                                        output_location=args.athena_output, refresh=args.refresh_data)
    summary = RollingListwiseTrainer(cfg, args.output_dir, workers=args.workers).run(
        panel, model_start=args.model_start, model_end=args.model_end,
        max_models=args.max_models, source_metadata=source, raw_returns=pd.read_parquet(args.raw_features))
    print(json.dumps({"output_dir": str(args.output_dir), "model_count": summary["model_count"],
                      "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
