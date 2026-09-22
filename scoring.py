"""Same-date pickle scoring and weekly quintile active-return evaluation."""

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

from cross_sectional_panel import build_cross_sectional_panel
from training import RollingPanelTrainer, TrainingConfig, prepare_panel, regression_metrics, write_json
from training_data import load_athena_panel

LOGGER = logging.getLogger(__name__)
MODELS = {"xgboost": "xgboost.pkl", "linear_regression": "linear_regression.pkl"}
DEFAULT_RAW = Path("local_cache/historical_features_731a66adc016.parquet")
DEFAULT_EVENTS = Path(__file__).with_name("scoring_return_events.json")


def assign_quintiles(scores: pd.DataFrame, *, scope: str = "universe", highest_first=False) -> pd.DataFrame:
    """Assign Q1=lowest to Q5=highest, using only score-date information.

    Alphabetical ticker order breaks exact prediction ties deterministically.
    Group sizes differ by at most one; outcomes never affect membership.
    """
    if scope not in {"universe", "sector"}:
        raise ValueError("scope must be universe or sector")
    result = scores.sort_values(["model", "week_date", "prediction", "ticker"],
                                ascending=[True, True, not highest_first, True]).copy()
    if result.duplicated(["model", "week_date", "ticker"]).any():
        raise ValueError("Duplicate model/date/ticker scores")
    result["quintile"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    valid = result.loc[np.isfinite(result["prediction"])].copy()
    keys = ["model", "week_date"] + (["sector"] if scope == "sector" else [])
    groups = valid.groupby(keys, sort=False)
    count = groups["prediction"].transform("size")
    if count.lt(5).any():
        raise ValueError("At least five valid predictions are required per quintile group")
    rank = groups.cumcount()
    valid["quintile"] = (rank * 5 // count + 1).astype("Int64")
    result.loc[valid.index, "quintile"] = valid["quintile"]
    return result.sort_values(["model", "week_date", "ticker"]).reset_index(drop=True)


def raw_forward_returns(panel: pd.DataFrame, raw: pd.DataFrame,
                        events: list[dict] | None = None) -> pd.DataFrame:
    """Recover realized forward returns from the source BEFORE normalization.

    Sector means use the full panel universe in the outcome week, matching
    construction of active_return_train. The same-ticker exact next-week join
    matches active_return_fwd; it never skips a missing ticker/week.
    """
    source = raw.rename(columns={c: c.lower() for c in raw.columns}).copy()
    source["ticker"] = source["ticker"].astype("string").str.strip().str.upper()
    source["week_date"] = pd.to_datetime(source["week_date"]).dt.normalize()
    if source.duplicated(["ticker", "week_date"]).any():
        raise ValueError("Duplicate raw ticker/week returns")
    base = panel[["ticker", "sector", "week_date", "active_return_train", "active_return_fwd"]].merge(
        source[["ticker", "week_date", "return_1w"]],
        on=["ticker", "week_date"], how="left", validate="one_to_one",
    )
    base["return_1w"] = pd.to_numeric(base["return_1w"], errors="raise")
    if not np.isfinite(base["return_1w"]).all() or base["return_1w"].lt(-1).any():
        raise ValueError("Missing or invalid raw stock returns; cannot compute performance")
    # Establish that the raw-return file belongs to this normalized panel.
    reconstructed = build_cross_sectional_panel(
        base.rename(columns={"ticker": "Ticker"}), feature_directions={"return_1w": 1}
    )
    if not np.allclose(base["active_return_train"], reconstructed["active_return_train"],
                       rtol=1e-8, atol=1e-10, equal_nan=True):
        raise ValueError("Raw returns do not reconstruct the panel's normalized active-return target")
    base["sector_return"] = base.groupby(["week_date", "sector"])["return_1w"].transform("mean")
    base["raw_active_return"] = base["return_1w"] - base["sector_return"]
    weeks = pd.DatetimeIndex(panel["week_date"].drop_duplicates().sort_values())
    next_map = pd.Series(weeks[1:].to_numpy(), index=weeks[:-1])
    result = base[["ticker", "week_date", "active_return_fwd"]].copy()
    result["next_week_date"] = result["week_date"].map(next_map)
    future = base[["ticker", "week_date", "return_1w", "sector_return", "raw_active_return", "active_return_train"]].rename(
        columns={"week_date": "next_week_date", "return_1w": "stock_return_fwd",
                 "sector_return": "sector_return_fwd", "raw_active_return": "raw_active_return_fwd",
                 "active_return_train": "expected_forward_zscore"}
    )
    result = result.merge(future, on=["ticker", "next_week_date"], how="left", validate="many_to_one")
    observed = result["next_week_date"].notna()
    if not np.allclose(result.loc[observed, "active_return_fwd"],
                       result.loc[observed, "expected_forward_zscore"],
                       rtol=1e-8, atol=1e-10, equal_nan=True):
        raise ValueError("Panel forward targets are not aligned to the next week's outcomes")
    result["return_source"] = np.where(result.stock_return_fwd.notna(), "raw_feature_snapshot", "unobserved")
    result["return_source_url"] = pd.NA
    for event in events or []:
        match = (result.ticker.eq(event["ticker"])
                 & result.week_date.eq(pd.Timestamp(event["week_date"]))
                 & result.next_week_date.eq(pd.Timestamp(event["next_week_date"])))
        if not match.any():
            continue
        if result.loc[match, "stock_return_fwd"].notna().any():
            raise ValueError("Corporate event would overwrite an observed raw return")
        event_date = pd.Timestamp(event["event_date"])
        if not pd.Timestamp(event["week_date"]) < event_date <= pd.Timestamp(event["next_week_date"]):
            raise ValueError("Corporate event falls outside the holding period")
        start_row = source.loc[source.ticker.eq(event["ticker"]) & source.week_date.eq(event["week_date"])]
        if len(start_row) != 1 or not np.isclose(float(start_row.adj_close.iloc[0]), event["start_close"]):
            raise ValueError("Event starting close does not match the raw adjusted-close basis")
        if event["event_type"] != "cash_acquisition" or event["start_close"] <= 0 or event["cash_per_share"] < 0:
            raise ValueError("Unsupported or invalid corporate return event")
        sector = panel.loc[panel.ticker.eq(event["ticker"]) & panel.week_date.eq(event["week_date"]), "sector"].iloc[0]
        benchmark = base.loc[base.sector.eq(sector) & base.week_date.eq(event["next_week_date"]), "sector_return"]
        if benchmark.empty:
            raise ValueError("Missing outcome-week sector benchmark for corporate event")
        stock_return = event["cash_per_share"] / event["start_close"] - 1
        result.loc[match, "stock_return_fwd"] = stock_return
        result.loc[match, "sector_return_fwd"] = benchmark.iloc[0]
        result.loc[match, "raw_active_return_fwd"] = stock_return - benchmark.iloc[0]
        result.loc[match, "return_source"] = event["event_type"]
        result.loc[match, "return_source_url"] = event["source_url"]
    return result.drop(columns=["active_return_fwd", "expected_forward_zscore"])


def score_pickles(panel: pd.DataFrame, config: TrainingConfig, models_dir: Path,
                  start: str = "2022-01-01", raw_returns: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Verify/refit each rolling model, then load the .pkl and score current X."""
    trainer = RollingPanelTrainer(config, models_dir, raw_returns=raw_returns)
    first = max(pd.Timestamp(start), pd.Timestamp(config.data_start) + pd.DateOffset(years=config.window_years))
    dates = sorted(panel.loc[panel.week_date.ge(first), "week_date"].unique())
    if not dates:
        raise ValueError("No eligible OOS scoring dates")
    records, audit = [], []
    with threadpool_limits(limits=config.threads):
        for i, date in enumerate(dates, 1):
            date = pd.Timestamp(date)
            # Reuses a completed fit only after checking its training-data hash;
            # new dates run the identical rolling training procedure.
            _, _, metadata = trainer.fit_week(panel, date)
            snapshot = panel.loc[panel.week_date.eq(date)].copy()
            inputs = snapshot[list(config.features)]
            valid = inputs.notna().all(axis=1)
            directory = models_dir / f"as_of={date.date()}"
            for name, filename in MODELS.items():
                model_path = directory / filename
                estimator = joblib.load(model_path)  # only locally produced artifacts
                if list(estimator.feature_names_in_) != list(config.features):
                    raise ValueError(f"Pickle feature schema mismatch: {model_path}")
                if name == "xgboost":
                    estimator.set_params(n_jobs=config.threads)
                scored = snapshot[["ticker", "sector", "week_date", "active_return_fwd"]].copy()
                scored["model"] = name
                scored["model_as_of"] = date
                scored["prediction"] = np.nan
                if valid.any():
                    scored.loc[valid, "prediction"] = estimator.predict(inputs.loc[valid])
                scored["raw_prediction"] = scored["prediction"]
                if name == "xgboost" and valid.any():
                    scored.loc[valid, "raw_prediction"] = getattr(
                        estimator, "predict_raw", estimator.predict)(inputs.loc[valid])
                records.append(scored)
                audit.append({"model": name, "model_as_of": str(date.date()),
                              "training_signature": metadata["signature"], "pickle": str(model_path),
                              "pickle_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()})
            if i % 25 == 0 or i == len(dates):
                LOGGER.info("Scored %s/%s dates through %s using same-date pickles", i, len(dates), date.date())
    return pd.concat(records, ignore_index=True), audit


def quintile_performance(members: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight returns; withhold a whole week if any holding lacks a return.

    This avoids hindsight-based reweighting when a selected stock disappears.
    A missing interior week also breaks the cumulative series thereafter.
    """
    data = members.dropna(subset=["quintile"]).copy()
    if data.empty:
        raise ValueError("No quintile assignments")
    data["has_return"] = data[["stock_return_fwd", "sector_return_fwd", "raw_active_return_fwd"]].notna().all(axis=1)
    weekly = data.groupby(["model", "week_date", "quintile"], observed=True).agg(
        next_week_date=("next_week_date", "first"), holdings=("ticker", "size"),
        observed_holdings=("has_return", "sum"),
        mean_prediction=("prediction", "mean"),
        mean_forward_zscore=("active_return_fwd", "mean"),
        portfolio_return=("stock_return_fwd", "mean"),
        benchmark_return=("sector_return_fwd", "mean"),
        active_return=("raw_active_return_fwd", "mean"),
    ).reset_index().sort_values(["model", "quintile", "week_date"])
    weekly["complete"] = weekly["holdings"].eq(weekly["observed_holdings"])
    complete_week = weekly.groupby(["model", "week_date"])["complete"].transform("all")
    weekly.loc[~complete_week, ["portfolio_return", "benchmark_return", "active_return", "mean_forward_zscore"]] = np.nan
    weekly["complete"] = complete_week
    for _, index in weekly.groupby(["model", "quintile"]).groups.items():
        group = weekly.loc[index]
        portfolio_wealth = (1 + group["portfolio_return"]).cumprod(skipna=False)
        benchmark_wealth = (1 + group["benchmark_return"]).cumprod(skipna=False)
        weekly.loc[index, "cumulative_active_return"] = group["active_return"].cumsum(skipna=False)
        weekly.loc[index, "compounded_active_spread"] = (1 + group["active_return"]).cumprod(skipna=False) - 1
        weekly.loc[index, "cumulative_portfolio_return"] = portfolio_wealth - 1
        weekly.loc[index, "cumulative_benchmark_return"] = benchmark_wealth - 1
        weekly.loc[index, "cumulative_excess_return"] = portfolio_wealth - benchmark_wealth
        weekly.loc[index, "relative_wealth_return"] = portfolio_wealth / benchmark_wealth - 1
    return weekly.sort_values(["model", "week_date", "quintile"]).reset_index(drop=True)


def plot_quintiles(weekly: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    colors = ["#b2182b", "#ef8a62", "#888888", "#67a9cf", "#2166ac"]
    for row, model in enumerate(MODELS):
        for quintile in range(1, 6):
            frame = weekly.loc[weekly.model.eq(model) & weekly.quintile.eq(quintile)].dropna(subset=["cumulative_active_return"])
            for col, metric in enumerate(["cumulative_active_return", "cumulative_excess_return"]):
                dates = [weekly.week_date.min(), *frame.next_week_date]
                values = [0.0, *frame[metric]]
                axes[row, col].plot(dates, values, label=f"Q{quintile}", color=colors[quintile-1], linewidth=1.5)
                axes[row, col].yaxis.set_major_formatter(PercentFormatter(1))
                axes[row, col].grid(alpha=0.2)
                axes[row, col].axhline(0, color="black", linewidth=0.5)
                axes[row, col].xaxis.set_major_locator(mdates.YearLocator())
                axes[row, col].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[row, 0].set_title(f"{model}: sum of weekly active returns")
        axes[row, 0].set_ylabel("Cumulative active return (percentage points)")
        axes[row, 1].set_title(f"{model}: compounded portfolio minus benchmark")
        axes[row, 1].set_ylabel("Difference in cumulative return (percentage points)")
        axes[row, 1].legend(ncol=5, fontsize=8)
    fig.suptitle("Weekly equal-weight quintiles | Q1 lowest predictions, Q5 highest\nSector-average return benchmark; before costs", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_scoring(panel: pd.DataFrame, raw: pd.DataFrame, config: TrainingConfig,
                models_dir: Path, output_dir: Path, *, start="2022-01-01",
                scope="universe", provenance=None, events: list[dict] | None = None) -> dict:
    panel = prepare_panel(panel, config)
    scores, audit = score_pickles(panel, config, Path(models_dir), start, raw_returns=raw)
    # Membership is fixed before realized returns are joined.
    members = assign_quintiles(scores, scope=scope)
    if events is None:
        events = json.loads(DEFAULT_EVENTS.read_text())
    returns = raw_forward_returns(panel, raw, events)
    members = members.merge(returns, on=["ticker", "week_date"], how="left", validate="many_to_one")
    weekly = quintile_performance(members)
    metrics = []
    for (model, date), group in members.groupby(["model", "week_date"]):
        metrics.append({"model": model, "week_date": date,
                        **regression_metrics(group.active_return_fwd, group.raw_prediction)})
    metric_frame = pd.DataFrame(metrics)
    endpoint = weekly.dropna(subset=["cumulative_active_return"]).groupby(["model", "quintile"]).tail(1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    members.to_parquet(output_dir / "stock_scores_and_quintiles.parquet", index=False)
    weekly.to_parquet(output_dir / "quintile_returns.parquet", index=False)
    endpoint.to_parquet(output_dir / "quintile_summary.parquet", index=False)
    metric_frame.to_parquet(output_dir / "oos_metrics.parquet", index=False)
    pd.DataFrame(audit).to_parquet(output_dir / "model_audit.parquet", index=False)
    plot_quintiles(weekly, output_dir / "quintile_cumulative_active_returns.png")
    summary = {
        "score_dates": int(scores.week_date.nunique()),
        "first_score_date": str(scores.week_date.min().date()),
        "last_score_date": str(scores.week_date.max().date()),
        "quintile_scope": scope, "weighting": "Equal stock weights, weekly rebalancing",
        "ties": "Prediction then alphabetical ticker; no outcome-based filtering",
        "cumulative_active_return_definition": "Sum of weekly (portfolio return minus matched sector-average benchmark return)",
        "cumulative_excess_return_definition": "Product(1+portfolio return) minus product(1+benchmark return)",
        "compounded_active_spread_definition": "Product(1+weekly active return)-1; not the difference of compounded wealth",
        "missing_return_policy": "Withhold all quintile returns in that model/week; cumulative series stops at an interior gap",
        "costs": "Before transaction costs; after-close research labels",
        "provenance": provenance or {},
        "corporate_return_events": events,
        "corporate_event_rows_per_model": members.loc[members.return_source.eq("cash_acquisition")].groupby("model").size().to_dict(),
        "evaluated_weeks_per_model": weekly.loc[weekly.complete].groupby("model").week_date.nunique().to_dict(),
        "endpoints": endpoint[["model", "quintile", "next_week_date", "cumulative_active_return",
                               "cumulative_excess_return", "compounded_active_spread"]].to_dict("records"),
        "forecast_metrics": {m: regression_metrics(g.active_return_fwd, g.raw_prediction) for m, g in members.groupby("model")},
    }
    write_json(output_dir / "scoring_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/ml/rolling_5y_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ml/oos_scoring_v1"))
    parser.add_argument("--raw-features", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--scope", choices=["universe", "sector"], default="universe")
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    manifest = json.loads((args.models_dir / "run_summary.json").read_text())
    config_values = manifest["config"]
    config_values["features"] = tuple(config_values["features"])
    config = TrainingConfig(**config_values)
    panel, source = load_athena_panel(start=config.data_start, refresh=args.refresh_data)
    if not args.raw_features.exists():
        raise FileNotFoundError(f"Raw source cache missing: {args.raw_features}. Supply --raw-features with the pre-normalization weekly panel.")
    raw = pd.read_parquet(args.raw_features)
    summary = run_scoring(panel, raw, config, args.models_dir, args.output_dir,
                          start=args.start, scope=args.scope,
                          provenance={"panel": source, "raw_features_path": str(args.raw_features.resolve()),
                                      "raw_features_sha256": hashlib.sha256(args.raw_features.read_bytes()).hexdigest()})
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
