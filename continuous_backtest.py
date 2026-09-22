"""Train 260-week models and backtest independently reloaded weekly pickles.

Each as-of model is fitted through that week's close. Inference uses the same
week's current features, and evaluation uses next-week ground truth. No older
binary ensemble, missing-model fallback, or random time split is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lightgbm_listwise_training import RankingConfig, RollingListwiseTrainer, evaluate_scores
from scoring import DEFAULT_RAW, DEFAULT_EVENTS, assign_quintiles, quintile_performance, raw_forward_returns
from training import (TrainingConfig, RollingPanelTrainer, prepare_panel, weekly_training_window,
                      frame_fingerprint, regression_metrics, write_json)
from training_data import load_athena_panel
from portfolio_training import raw_training_outcomes

LOGGER = logging.getLogger(__name__)
DEFAULT_ROOT = Path("artifacts/ml/continuous_260w_v2")
MODEL_FILES = {"lightgbm": ("lightgbm", "lightgbm.pkl"),
               "xgboost": ("regression", "xgboost.pkl"),
               "linear_regression": ("regression", "linear_regression.pkl")}


def score_snapshot(snapshot, model_path, feature_cols, model_name, threads=1, raw_prediction=False):
    """Pure inference: load a local pickle and use only current-week features."""
    model = joblib.load(model_path)
    names = model.feature_name() if model_name == "lightgbm" else list(model.feature_names_in_)
    if list(names) != list(feature_cols):
        raise ValueError(f"Pickle feature ordering differs from metadata: {model_path}")
    current = snapshot[list(feature_cols)]
    valid = current.notna().all(axis=1)
    values = pd.Series(np.nan, index=snapshot.index)
    if valid.any():
        if model_name == "lightgbm":
            values.loc[valid] = model.predict(current.loc[valid].astype(np.float32), num_threads=threads)
        else:
            if model_name == "xgboost":
                model.set_params(n_jobs=threads)
            values.loc[valid] = (getattr(model, "predict_raw", model.predict)(current.loc[valid])
                                 if raw_prediction and model_name == "xgboost"
                                 else model.predict(current.loc[valid]))
    return values


def score_saved_models(panel, root, *, start="2022-01-01", end=None, threads=1, raw_returns=None):
    """Fail on missing/mismatched fits; never train inside this inference step."""
    root = Path(root)
    # A scoring job needs the per-date model sidecars, not a training run's
    # aggregate completion file. This also supports independently scoring a
    # completed date range while later model dates are still being trained.
    date_column = next(c for c in panel if c.lower() == "week_date")
    requested = pd.to_datetime(panel[date_column]).dt.normalize()
    requested = requested.loc[requested.ge(start) & requested.le(end if end else requested.max())]
    if requested.empty:
        raise ValueError("No dates in the requested scoring range")
    first_folder = f"as_of={requested.min().date()}"
    regression_run = json.loads((root / "regression" / first_folder / "metadata.json").read_text())
    ranking_run = json.loads((root / "lightgbm" / first_folder / "metadata.json").read_text())
    values = dict(regression_run["config"])
    values["features"] = tuple(values["features"])
    config = TrainingConfig(**values)
    if config.window_weeks != 260 or ranking_run["window_weeks"] != 260:
        raise ValueError("All models must use exactly 260 weekly snapshots")
    if list(config.features) != ranking_run["config"]["features"]:
        raise ValueError("The three models must use the same feature set")
    panel = prepare_panel(panel, config)
    raw_outcomes = raw_training_outcomes(panel, raw_returns) if raw_returns is not None else None
    weeks = pd.DatetimeIndex(panel.week_date.unique())
    dates = weeks[(weeks >= pd.Timestamp(start)) & (weeks <= (pd.Timestamp(end) if end else weeks[-1]))]
    if len(dates) == 0 or not dates.equals(pd.date_range(dates[0], dates[-1], freq="W-WED")):
        raise ValueError("Requested backtest has no dates or is not continuous")
    records, audit = [], []
    lag = [f"{f}_train" for f in config.features]
    with threadpool_limits(limits=threads):
        for number, date in enumerate(dates, 1):
            train = weekly_training_window(panel, date, config, 260, min_rows=2)
            if raw_outcomes is not None:
                raw_window = train[["ticker", "week_date"]].merge(
                    raw_outcomes, on=["ticker", "week_date"], how="left", validate="one_to_one")
                raw_fingerprint = frame_fingerprint(raw_window[["ticker", "week_date", "raw_active_return"]])
            snapshot = panel.loc[panel.week_date.eq(date)]
            for name, (directory, filename) in MODEL_FILES.items():
                folder = root / directory / f"as_of={date.date()}"
                metadata = json.loads((folder / "metadata.json").read_text())
                if (metadata["as_of"] != str(date.date()) or metadata["training_weeks"] != 260
                        or metadata["last_training_week"] != str(date.date())
                        or metadata["first_training_week"] != str(train.week_date.min().date())):
                    raise ValueError(f"Incorrect model date/window at {folder}")
                columns = (["ticker", "week_date", "prev_week_date", "active_return_train", *lag]
                           if name == "lightgbm" else ["ticker", "week_date", "prev_week_date", *lag, "active_return_train"])
                if frame_fingerprint(train[columns]) != metadata["training_fingerprint"]:
                    raise ValueError(f"Training input fingerprint differs at {folder}")
                if raw_outcomes is not None and metadata.get("raw_return_fingerprint") != raw_fingerprint:
                    raise ValueError(f"Training raw-return fingerprint differs at {folder}")
                if list(metadata["config"]["features"]) != list(config.features):
                    raise ValueError(f"Training feature list differs at {folder}")
                model_path = folder / filename
                actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
                expected_hash = (metadata["pickle_sha256"] if name == "lightgbm"
                                 else metadata["pickle_sha256"][filename])
                if actual_hash != expected_hash:
                    raise ValueError(f"Saved pickle hash mismatch: {model_path}")
                scored = snapshot[["ticker", "sector", "week_date"]].copy()
                scored["model"], scored["model_as_of"] = name, date
                scored["prediction"] = score_snapshot(snapshot, model_path, config.features, name, threads)
                scored["raw_prediction"] = (score_snapshot(snapshot, model_path, config.features, name,
                                                            threads, raw_prediction=True)
                                             if name == "xgboost" else scored["prediction"])
                records.append(scored)
                audit.append(dict(model=name, model_as_of=date, training_weeks=260,
                                  first_training_week=train.week_date.min(), last_training_week=date,
                                  training_rows=len(train), training_signature=metadata["signature"],
                                  training_fingerprint=metadata["training_fingerprint"],
                                  pickle=str(model_path), pickle_sha256=actual_hash,
                                  training_features=lag, inference_features=list(config.features),
                                  training_target=("quintiles of active_return_train" if name == "lightgbm"
                                                   else "active_return_train"),
                                  evaluation_target="active_return_fwd"))
            if number % 25 == 0 or number == len(dates):
                LOGGER.info("Independently scored and audited %s/%s weeks for all three models", number, len(dates))
    scores = assign_quintiles(pd.concat(records, ignore_index=True), highest_first=True)
    # Rank 1 is best; ties use the alphabetical ticker ordering from assign_quintiles.
    groups = scores.groupby(["model", "week_date"])
    scores["predicted_rank"] = groups.prediction.rank(method="first", ascending=False).astype("Int64")
    scores["predicted_percentile"] = groups.prediction.rank(method="average", pct=True)
    return panel, scores, pd.DataFrame(audit)


def plot_results(weekly, ranking, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter
    colors = ["#2166ac", "#67a9cf", "#888888", "#ef8a62", "#b2182b"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    for row, model in enumerate(MODEL_FILES):
        for quintile in range(1, 6):
            group = weekly.loc[weekly.model.eq(model) & weekly.quintile.eq(quintile)]
            group = group.loc[group.next_week_date.notna()]
            for col, metric in enumerate(["cumulative_active_return", "cumulative_excess_return"]):
                axes[row, col].plot([weekly.week_date.min(), *group.next_week_date],
                                    [0.0, *group[metric]], label=f"Q{quintile}", color=colors[quintile-1])
                axes[row, col].yaxis.set_major_formatter(PercentFormatter(1))
                axes[row, col].grid(alpha=0.2)
                axes[row, col].axhline(0, color="black", linewidth=0.5)
        axes[row, 0].set_title(f"{model}: sum of weekly active returns")
        axes[row, 1].set_title(f"{model}: compounded portfolio minus benchmark")
        axes[row, 0].set_ylabel("Percentage points")
        axes[row, 1].legend(ncol=5, fontsize=8)
    for ax in axes[-1]:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle("260-week rolling models | Same-week pickle inference | Q1 highest, Q5 lowest\n"
                 "Equal stock weights; matched sector benchmark; before costs", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "continuous_quintile_backtest.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for model, group in ranking.groupby("model"):
        indexed = group.set_index("week_date")
        for ax, column in zip(axes, ["spearman", "ls_ndcg"]):
            ax.plot(indexed.index, indexed[column].rolling(13, min_periods=13).mean(), label=model)
            ax.grid(alpha=0.2)
    axes[0].set(title="13-week mean Spearman rank IC", ylabel="Rank IC")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[1].set(title="13-week mean symmetric NDCG@20%", ylabel="NDCG")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "continuous_rank_diagnostics.png", dpi=150)
    plt.close(fig)


def run_backtest(panel, raw, root=DEFAULT_ROOT, *, start="2022-01-01", end=None, threads=1,
                 provenance=None, events=None, output_dir=None):
    panel, scores, audit = score_saved_models(panel, root, start=start, end=end,
                                             threads=threads, raw_returns=raw)
    if events is None:
        events = json.loads(DEFAULT_EVENTS.read_text())
    # Outcomes are joined only after predictions and portfolio membership are fixed.
    outcomes = raw_forward_returns(panel, raw, events)
    members = scores.merge(panel[["ticker", "week_date", "active_return_fwd"]],
                           on=["ticker", "week_date"], how="left", validate="many_to_one")
    members = members.merge(outcomes, on=["ticker", "week_date"], how="left", validate="many_to_one")
    members["ground_truth_rank"] = members.groupby(["model", "week_date"]).active_return_fwd.rank(
        method="average", ascending=False)
    members["ground_truth_complete_week"] = members.groupby(["model", "week_date"]).active_return_fwd.transform(
        lambda values: values.notna().all())
    members.loc[~members.ground_truth_complete_week, "ground_truth_rank"] = np.nan
    weekly = quintile_performance(members)
    ranking_frames, aggregates = [], {}
    for model, group in members.groupby("model"):
        ranking, metrics = evaluate_scores(group, target_col="active_return_fwd", score_col="prediction")
        ranking.insert(0, "model", model)
        ranking_frames.append(ranking)
        aggregates[model] = {"ranking": metrics}
        if model != "lightgbm":
            aggregates[model]["regression"] = regression_metrics(group.active_return_fwd,
                                                                  group.raw_prediction)
    ranking = pd.concat(ranking_frames, ignore_index=True)
    endpoint = weekly.dropna(subset=["cumulative_active_return"]).groupby(["model", "quintile"]).tail(1)
    latest = members.week_date.max()
    output = Path(output_dir) if output_dir is not None else Path(root) / "backtest"
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in {"stock_scores_and_quintiles": members, "model_audit": audit,
                        "quintile_returns": weekly, "quintile_summary": endpoint,
                        "weekly_ranking_metrics": ranking}.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    summary = dict(first_score_date=str(members.week_date.min().date()), last_score_date=str(latest.date()),
                   model_dates_per_model=members.groupby("model").week_date.nunique().to_dict(),
                   pickle_count=len(audit), prediction_rows=len(members), window_weeks=260,
                   latest_available_panel_date=str(panel.week_date.max().date()),
                   last_realized_outcome_date=str(endpoint.next_week_date.max().date()),
                   completed_return_weeks_per_model=weekly.loc[weekly.complete].groupby("model").week_date.nunique().to_dict(),
                   complete_normalized_target_weeks_per_model={m: v["ranking"]["n_groups"] for m, v in aggregates.items()},
                   metrics=aggregates, provenance=provenance or {},
                   missing_forward_labels_per_model=members.groupby("model").active_return_fwd.apply(lambda s: int(s.isna().sum())).to_dict(),
                   missing_predictions_per_model=members.groupby("model").prediction.apply(lambda s: int(s.isna().sum())).to_dict(),
                   corporate_return_events=events,
                   endpoints=endpoint[["model", "quintile", "next_week_date", "cumulative_active_return",
                                       "cumulative_excess_return"]].to_dict("records"),
                   timing="Fit 260 outcome weeks ending t: *_train -> active_return_train (quintiles for LightGBM); reload t.pkl; current features -> t-to-next-week rank; evaluate active_return_fwd",
                   quintile_convention="Q1 highest predicted score, Q5 lowest, for every model",
                   cumulative_active_return_definition="Sum of weekly portfolio minus matched sector benchmark returns; decimal percentage-point units",
                   cumulative_excess_return_definition="Compounded portfolio return minus compounded benchmark return",
                   missing_policy="Retain latest predictions; no fabricated forward labels. Withhold incomplete outcome weeks. Interior gaps stop cumulative performance.",
                   costs="Gross and 0/5/10/25 bps per traded dollar in cost_sensitivity.parquet; after-close information assumption")
    from comparative_evaluation import save_comparative_metrics
    save_comparative_metrics(members, output)
    costs = pd.read_parquet(output / "cost_sensitivity.parquet")
    summary["cost_sensitivity"] = costs.astype(object).where(costs.notna(), None).to_dict("records")
    write_json(output / "backtest_summary.json", summary)
    plot_results(weekly, ranking, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--input-parquet", type=Path)
    parser.add_argument("--raw-features", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--score-only", action="store_true", help="Require existing fits; independently reload and audit every pickle")
    parser.add_argument("--refresh-data", action="store_true", help="Query Athena again instead of using its recorded cache")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.root / "continuous_backtest.log")])
    if args.input_parquet:
        panel = pd.read_parquet(args.input_parquet)
        source = {"mode": "explicit_local_parquet", "path": str(args.input_parquet.resolve())}
    else:
        panel, source = load_athena_panel(start="2017-01-01", refresh=args.refresh_data)
    if not args.score_only:
        raw = pd.read_parquet(args.raw_features)
        RollingPanelTrainer(TrainingConfig(window_weeks=260, threads=args.threads),
                            args.root / "regression", workers=args.workers).run(
                                panel, model_start=args.start, model_end=args.end, source_metadata=source,
                                raw_returns=raw)
        RollingListwiseTrainer(RankingConfig(threads=args.threads), args.root / "lightgbm", workers=args.workers).run(
            panel, model_start=args.start, model_end=args.end, source_metadata=source,
            raw_returns=raw)
    else:
        raw = pd.read_parquet(args.raw_features)
    summary = run_backtest(panel, raw, args.root, start=args.start, end=args.end, threads=args.threads,
                           provenance={"panel": source, "raw_features": str(args.raw_features.resolve()),
                                       "raw_features_sha256": hashlib.sha256(args.raw_features.read_bytes()).hexdigest()})
    print(json.dumps({k: summary[k] for k in ["model_dates_per_model", "pickle_count", "first_score_date",
                                            "last_score_date", "metrics"]}, indent=2))


if __name__ == "__main__":
    main()
