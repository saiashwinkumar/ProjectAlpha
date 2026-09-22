"""Rolling XGBoost and ordinary least-squares stock return models.

At the close of week t, fit previous-week features to returns observed by t.
Use current-week features to forecast t -> next week. Forward outcomes are
used exclusively for evaluation, never fitting, filtering training, or tuning.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from cross_sectional_panel import DEFAULT_FEATURE_DIRECTIONS
from portfolio_training import (SeedRankEnsemble, attach_training_returns,
                                better_portfolio, portfolio_summary, portfolio_weekly,
                                raw_training_outcomes)

LOGGER = logging.getLogger(__name__)
FEATURES = tuple(DEFAULT_FEATURE_DIRECTIONS)
TARGET = "active_return_train"
FORWARD_TARGET = "active_return_fwd"

GRIDS = {
    "compact": [
        {"learning_rate": [lr], "max_depth": [depth], "min_child_weight": [child],
         "reg_lambda": [l2], "subsample": [0.8], "colsample_bytree": [0.8],
         "n_estimators": [300]}
        for lr, depth, child, l2 in [(.025, 1, 50, 10), (.025, 2, 50, 10),
                                     (.025, 3, 100, 10), (.01, 1, 50, 10),
                                     (.01, 2, 100, 10), (.025, 2, 10, 1)]
    ],
    "reference": [
        {"learning_rate": [lr], "max_depth": [depth], "min_child_weight": [child],
         "reg_lambda": [l2], "subsample": [0.8], "colsample_bytree": [0.8],
         "n_estimators": [400]}
        for lr, depth, child, l2 in [(.01, 1, 100, 20), (.01, 2, 100, 20),
                                     (.025, 1, 100, 20), (.025, 2, 50, 20)]
    ],
}


@dataclass(frozen=True)
class TrainingConfig:
    data_start: str = "2017-01-01"
    window_years: int = 5
    cv_splits: int = 4
    cv_gap_weeks: int = 1
    grid: str = "compact"
    threads: int = 4
    seed: int = 42
    features: tuple[str, ...] = FEATURES
    window_weeks: int | None = None

    def __post_init__(self):
        if self.window_weeks is not None and self.window_weeks < 1:
            raise ValueError("window_weeks must be positive")
        if self.window_years < 1 or self.cv_splits < 2 or self.cv_gap_weeks < 0 or self.threads < 1:
            raise ValueError("Require positive window/threads, at least 2 folds, and nonnegative gap")
        if self.grid not in GRIDS:
            raise ValueError(f"Unknown grid {self.grid}")
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("Features must be nonempty and unique")
        if not set(self.features).issubset(FEATURES):
            raise ValueError("Features must come from the panel feature allowlist")
        pd.Timestamp(self.data_start)


def prepare_panel(panel: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    panel = panel.copy()
    normalized = [str(c).lower() for c in panel.columns]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate case-insensitive column names in panel")
    panel.columns = normalized
    lag_features = [f"{f}_train" for f in config.features]
    required = ["ticker", "sector", "week_date", "prev_week_date", *config.features,
                *lag_features, TARGET, FORWARD_TARGET]
    missing = set(required) - set(panel.columns)
    if missing:
        raise ValueError(f"Missing model input columns: {sorted(missing)}")
    panel = panel[required].copy()
    for column in ["week_date", "prev_week_date"]:
        values = panel[column]
        # v2 Parquet stores datetime64[ns]; Glue exposes prev_week_date as
        # bigint. Athena-managed results may return those integers as strings.
        nonnull = values.dropna().astype(str)
        if len(nonnull) and nonnull.str.fullmatch(r"-?\d{16,19}").all():
            values = pd.to_numeric(values, errors="raise").astype("Int64")
            panel[column] = pd.to_datetime(values, unit="ns", errors="raise").dt.normalize()
        else:
            panel[column] = pd.to_datetime(values, errors="raise").dt.normalize()
    if panel["week_date"].isna().any():
        raise ValueError("Missing week_date")
    panel = panel.loc[panel["week_date"].ge(config.data_start)].copy()
    if panel.empty:
        raise ValueError("No panel data on/after the requested start")
    for column in ["ticker", "sector"]:
        panel[column] = panel[column].astype("string").str.strip()
        if panel[column].isna().any() or panel[column].eq("").any():
            raise ValueError(f"Missing {column}")
    panel["ticker"] = panel["ticker"].str.upper()
    if panel.duplicated(["ticker", "week_date"]).any():
        raise ValueError("Duplicate ticker/week observations")
    numeric = [*config.features, *lag_features, TARGET, FORWARD_TARGET]
    panel[numeric] = panel[numeric].apply(pd.to_numeric, errors="raise")
    if np.isinf(panel[numeric].to_numpy(dtype=float)).any():
        raise ValueError("Panel contains infinite model inputs or targets")
    panel = panel.sort_values(["week_date", "ticker"]).reset_index(drop=True)
    weeks = pd.DatetimeIndex(panel["week_date"].unique())
    expected = pd.date_range(weeks.min(), weeks.max(), freq="W-WED")
    if not weeks.equals(expected):
        raise ValueError("Expected complete Wednesday snapshots; missing/off-calendar weeks found")
    previous = pd.Series(weeks[:-1].to_numpy(), index=weeks[1:])
    check = panel["week_date"].gt(weeks[0])
    if not panel.loc[check, "prev_week_date"].equals(panel.loc[check, "week_date"].map(previous)):
        raise ValueError("prev_week_date disagrees with preceding panel snapshot")
    if (panel["prev_week_date"].notna() & panel["prev_week_date"].ge(panel["week_date"])).any():
        raise ValueError("Previous-week features must precede outcome dates")
    # Check the supplied panel's lag contract before using any train columns.
    lag_lookup = panel[["ticker", "week_date", *config.features]].rename(
        columns={"week_date": "prev_week_date", **{f: f"_expected_{f}" for f in config.features}}
    )
    joined = panel.merge(lag_lookup, on=["ticker", "prev_week_date"], how="left", validate="many_to_one")
    comparable = joined["prev_week_date"].isin(weeks)
    if not np.allclose(
        joined.loc[comparable, lag_features].to_numpy(dtype=float),
        joined.loc[comparable, [f"_expected_{f}" for f in config.features]].to_numpy(dtype=float),
        rtol=1e-8, atol=1e-10, equal_nan=True,
    ):
        raise ValueError("Lagged features do not match the exact previous-week ticker snapshot")
    return panel


def date_cv_splits(dates: pd.Series, n_splits: int = 4, gap: int = 1):
    """Return original row POSITIONS, keeping each weekly cross-section intact."""
    dates = pd.Series(pd.to_datetime(dates).to_numpy())
    weeks = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    for train_dates, valid_dates in splitter.split(weeks):
        yield (
            np.flatnonzero(dates.isin(weeks[train_dates]).to_numpy()),
            np.flatnonzero(dates.isin(weeks[valid_dates]).to_numpy()),
        )


def training_window(panel: pd.DataFrame, as_of: pd.Timestamp, config: TrainingConfig) -> pd.DataFrame:
    if config.window_weeks is not None:
        return weekly_training_window(panel, as_of, config, config.window_weeks)
    cutoff = as_of - pd.DateOffset(years=config.window_years)
    if as_of < pd.Timestamp(config.data_start) + pd.DateOffset(years=config.window_years):
        raise ValueError("A full five-year history is not available at this as-of date")
    # Allow at most one calendar week between cutoff and first available
    # Wednesday. This prevents treating a truncated source as a full window.
    if panel["week_date"].min() > cutoff + pd.Timedelta(days=7):
        raise ValueError("Source does not cover the full requested training window")
    selected = panel.loc[
        panel["week_date"].gt(cutoff) & panel["week_date"].le(as_of)
        & panel["prev_week_date"].ge(pd.Timestamp(config.data_start))
    ]
    selected = selected.dropna(subset=[TARGET, *[f"{f}_train" for f in config.features]])
    if selected.empty or selected["week_date"].max() != as_of:
        raise ValueError(f"No complete training observations ending at {as_of.date()}")
    return selected.reset_index(drop=True)


def weekly_training_window(panel, as_of, config, window_weeks=260, min_rows=1):
    """Exactly N outcome weeks, with only lagged predictors and observed labels."""
    as_of = pd.Timestamp(as_of)
    expected = pd.date_range(end=as_of, periods=window_weeks, freq="W-WED")
    selected = panel.loc[panel.week_date.between(expected[0], as_of)].copy()
    if as_of != expected[-1] or not pd.DatetimeIndex(sorted(selected.week_date.unique())).equals(expected):
        raise ValueError(f"Source does not cover the full {window_weeks}-week training window")
    selected = selected.loc[selected.prev_week_date.ge(pd.Timestamp(config.data_start))]
    selected = selected.dropna(subset=[TARGET, *[f"{f}_train" for f in config.features]])
    counts = selected.groupby("week_date").size()
    if not counts.index.equals(expected) or counts.min() < min_rows:
        raise ValueError(f"Every one of the {window_weeks} training weeks needs at least {min_rows} complete rows")
    return selected.sort_values(["week_date", "ticker"]).reset_index(drop=True)


def frame_fingerprint(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    return hashlib.sha256(values + "|".join(frame.columns).encode()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def export_scoring_pickles(path: Path, xgb, baseline, *, overwrite=False) -> None:
    """Expose fitted estimators as .pkl files for the separate scoring stage."""
    for name, estimator in [("xgboost.pkl", xgb), ("linear_regression.pkl", baseline)]:
        destination = path / name
        if overwrite or not destination.exists():
            temporary = path / (name + ".tmp")
            joblib.dump(estimator, temporary)
            temporary.replace(destination)


def regression_metrics(actual, predicted) -> dict:
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    result = {"n": int(len(actual)), "mse": None, "rmse": None, "mae": None,
              "r2": None, "pearson_ic": None, "spearman_ic": None, "zero_baseline_mse": None}
    if not len(actual):
        return result
    mse = float(mean_squared_error(actual, predicted))
    result.update(mse=mse, rmse=float(np.sqrt(mse)), mae=float(mean_absolute_error(actual, predicted)),
                  zero_baseline_mse=float(np.mean(actual ** 2)))
    if len(actual) > 1 and np.std(actual) > 0:
        result["r2"] = float(r2_score(actual, predicted))
        if np.std(predicted) > 0:
            result["pearson_ic"] = float(np.corrcoef(actual, predicted)[0, 1])
            result["spearman_ic"] = float(pd.Series(actual).rank().corr(pd.Series(predicted).rank()))
    return result


class RollingPanelTrainer:
    def __init__(self, config: TrainingConfig, output_dir: Path, workers: int = 1,
                 raw_returns: pd.DataFrame | None = None):
        self.config = config
        self.output_dir = Path(output_dir)
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.raw_returns = raw_returns
        self.raw_outcomes = None

    def fitted_weeks(self, panel, dates):
        # Bound the number of in-flight models and return results in date order.
        # Each fit reads the same panel and writes only its own as-of directory.
        with threadpool_limits(limits=self.config.threads):
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                remaining = iter(dates)
                pending = deque()
                for date in list(dates[:self.workers]):
                    next(remaining)
                    pending.append((date, executor.submit(self.fit_week, panel, date)))
                while pending:
                    date, future = pending.popleft()
                    result = future.result()
                    following = next(remaining, None)
                    if following is not None:
                        pending.append((following, executor.submit(self.fit_week, panel, following)))
                    yield date, result

    def fit_week(self, panel: pd.DataFrame, as_of: pd.Timestamp):
        cfg = self.config
        train = training_window(panel, as_of, cfg)
        if self.raw_outcomes is None:
            if self.raw_returns is None:
                raise ValueError("Validated raw returns are required for portfolio-based tuning")
            self.raw_outcomes = raw_training_outcomes(panel, self.raw_returns)
        train = attach_training_returns(train, self.raw_outcomes)
        lag_features = [f"{f}_train" for f in cfg.features]
        X = train[lag_features].copy()
        X.columns = list(cfg.features)
        y = train[TARGET]
        path = self.output_dir / f"as_of={as_of.date()}"
        path.mkdir(parents=True, exist_ok=True)
        identity = {
            "implementation_version": 4, "config": asdict(cfg),
            "as_of": str(as_of.date()),
            "training_fingerprint": frame_fingerprint(train[["ticker", "week_date", "prev_week_date", *lag_features, TARGET]]),
            "raw_return_fingerprint": frame_fingerprint(train[["ticker", "week_date", "raw_active_return"]]),
        }
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        manifest_path = path / "metadata.json"
        if manifest_path.exists():
            metadata = json.loads(manifest_path.read_text())
            if metadata["signature"] != signature:
                raise ValueError(f"Existing model/config/data differ at {path}; use a new --output-dir")
            # Load our own artifacts only; never load pickle files from documents.
            for name, expected in metadata["pickle_sha256"].items():
                if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Saved pickle hash mismatch at {path / name}")
            xgb = joblib.load(path / "xgboost.pkl")
            xgb.set_params(n_jobs=cfg.threads)
            baseline = joblib.load(path / "linear_regression.joblib")
            export_scoring_pickles(path, xgb, baseline)
            return xgb, baseline, metadata

        splits = list(date_cv_splits(train["week_date"], cfg.cv_splits, cfg.cv_gap_weeks))
        fold_metadata, baseline_mse = [], []
        for fold, (tr, va) in enumerate(splits):
            if train.iloc[tr]["week_date"].max() > train.iloc[va]["prev_week_date"].min():
                raise ValueError("CV training label is unavailable at the validation feature date")
            baseline_cv = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
            baseline_mse.append(float(mean_squared_error(y.iloc[va], baseline_cv.predict(X.iloc[va]))))
            fold_metadata.append({
                "fold": fold + 1, "training_rows": len(tr), "validation_rows": len(va),
                "train_start": str(train.iloc[tr]["week_date"].min().date()),
                "train_end": str(train.iloc[tr]["week_date"].max().date()),
                "validation_start": str(train.iloc[va]["week_date"].min().date()),
                "validation_end": str(train.iloc[va]["week_date"].max().date()),
                "linear_regression_mse": baseline_mse[-1],
            })
        started = time.monotonic()
        candidates = []
        for params in ParameterGrid(GRIDS[cfg.grid]):
            fold_scores, fold_predictions, best_iterations, fold_mse = [], [], [], []
            for tr, va in splits:
                model = XGBRegressor(objective="reg:squarederror", tree_method="hist", device="cpu",
                                     random_state=cfg.seed, n_jobs=cfg.threads,
                                     early_stopping_rounds=min(30, max(2, params["n_estimators"] // 3)),
                                     **params)
                model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
                predicted = model.predict(X.iloc[va])
                fold_scores.append(regression_metrics(y.iloc[va], predicted))
                fold_mse.append(float(mean_squared_error(y.iloc[va], predicted)))
                best_iterations.append(int(model.best_iteration + 1))
                fold_predictions.append(pd.DataFrame({"index": va, "score": predicted}))
            stitched = pd.concat(fold_predictions, ignore_index=True).sort_values("index")
            validated = train.iloc[stitched["index"].to_numpy()].copy()
            portfolio = portfolio_summary(portfolio_weekly(validated, stitched.score.to_numpy()))
            complexity = (params.get("max_depth", 6), -params.get("min_child_weight", 1),
                          -params.get("reg_lambda", 1), int(np.median(best_iterations)))
            candidates.append(dict(params=params, portfolio=portfolio, complexity=complexity,
                                   cv_mse=float(np.mean(fold_mse)), best_iterations=best_iterations,
                                   fold_regression=fold_scores))
        winner = None
        for candidate in candidates:
            if better_portfolio(candidate, winner, candidate["complexity"]):
                winner = candidate
        baseline = LinearRegression().fit(X, y)
        best_params = dict(winner["params"], n_estimators=max(1, int(np.median(winner["best_iterations"]))))
        seeds = tuple(cfg.seed + offset for offset in (0, 31, 65))
        seed_models = [XGBRegressor(objective="reg:squarederror", tree_method="hist", device="cpu",
                                    random_state=seed, n_jobs=cfg.threads, **best_params).fit(X, y)
                       for seed in seeds]
        xgb = SeedRankEnsemble(seed_models, cfg.features, "xgboost")
        seed_models[0].save_model(path / "xgboost.json")
        joblib.dump(baseline, path / "linear_regression.joblib")
        export_scoring_pickles(path, xgb, baseline, overwrite=True)
        write_json(path / "search_results.json", {"candidates": candidates,
                   "selection": "10bps net Q1-Q5 validation IR, then cumulative return, then simplicity"})
        pd.DataFrame([{**c["params"], "cv_mse": c["cv_mse"],
                       "rank_ic": c["portfolio"]["rank_ic"],
                       "ndcg_at_50": c["portfolio"]["ndcg_at_50"],
                       "weekly_turnover": c["portfolio"]["weekly_turnover"],
                       **{f"net_10bps_{k}": v for k, v in c["portfolio"]["cost_10bps"].items()},
                       "best_iterations": c["best_iterations"]} for c in candidates]).to_parquet(
                           path / "cv_results.parquet", index=False)
        pd.DataFrame({"feature": cfg.features, "xgboost_importance": np.mean(
                          [m.feature_importances_ for m in seed_models], axis=0),
                      "linear_coefficient": baseline.coef_}).to_parquet(path / "feature_effects.parquet", index=False)
        metadata = {
            **identity, "signature": signature, "training_rows": len(train),
            "training_weeks": int(train["week_date"].nunique()),
            "training_start_exclusive": str((as_of - (pd.Timedelta(days=7 * cfg.window_weeks)
                                                      if cfg.window_weeks else pd.DateOffset(years=cfg.window_years))).date()),
            "first_training_week": str(train["week_date"].min().date()),
            "last_training_week": str(train["week_date"].max().date()),
            "best_params": best_params, "xgboost_cv_mse": winner["cv_mse"],
            "selection_metric": "10bps net Q1-Q5 validation IR, then cumulative return; simpler model within 0.05 IR and 0.01 cumulative return",
            "best_cv_portfolio": winner["portfolio"], "ensemble_seeds": seeds,
            "linear_cv_mse": float(np.mean(baseline_mse)),
            "grid_candidates": len(ParameterGrid(GRIDS[cfg.grid])), "cv_folds": fold_metadata,
            "fit_seconds": time.monotonic() - started,
            "versions": {p: importlib.metadata.version(p) for p in ["xgboost", "scikit-learn", "pandas", "numpy"]},
            "linear_intercept": float(baseline.intercept_),
            "pickle_sha256": {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                              for name in ["xgboost.pkl", "linear_regression.pkl"]},
        }
        write_json(manifest_path, metadata)  # completion marker written last
        return xgb, baseline, metadata

    def run(self, raw_panel: pd.DataFrame, *, model_start: str | None = None,
            model_end: str | None = None, max_models: int | None = None,
            source_metadata: dict | None = None, raw_returns: pd.DataFrame | None = None) -> dict:
        panel = prepare_panel(raw_panel, self.config)
        raw = raw_returns if raw_returns is not None else self.raw_returns
        if raw is None:
            raise ValueError("Pass raw_returns for portfolio-based tuning")
        self.raw_outcomes = raw_training_outcomes(panel, raw)
        cfg = self.config
        weeks = pd.DatetimeIndex(panel["week_date"].unique())
        first = (weeks[0] + pd.Timedelta(days=7 * cfg.window_weeks) if cfg.window_weeks
                 else pd.Timestamp(cfg.data_start) + pd.DateOffset(years=cfg.window_years))
        if model_start:
            first = max(first, pd.Timestamp(model_start))
        last = pd.Timestamp(model_end) if model_end else weeks[-1]
        dates = weeks[(weeks >= first) & (weeks <= last)]
        if max_models is not None:
            if max_models < 1:
                raise ValueError("max_models must be positive")
            dates = dates[:max_models]
        if not len(dates):
            raise ValueError("No eligible model dates after the five-year warm-up")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        next_map = pd.Series(weeks[1:].to_numpy(), index=weeks[:-1])
        all_predictions, weekly_metrics, model_records = [], [], []
        LOGGER.info("Rolling run: %s rows, %s weeks, %s model dates (%s to %s), %s candidates x %s folds",
                    f"{len(panel):,}", len(weeks), len(dates), dates[0].date(), dates[-1].date(),
                    len(ParameterGrid(GRIDS[cfg.grid])), cfg.cv_splits)
        for number, (as_of, fitted) in enumerate(self.fitted_weeks(panel, dates), 1):
            xgb, baseline, metadata = fitted
            # Inference always crosses the serialized-model boundary, including fresh fits.
            binary_folder = self.output_dir / f"as_of={as_of.date()}"
            for name in ["xgboost.pkl", "linear_regression.pkl"]:
                expected_hash = metadata.get("pickle_sha256", {}).get(name)
                if expected_hash and hashlib.sha256((binary_folder / name).read_bytes()).hexdigest() != expected_hash:
                    raise ValueError(f"Saved pickle hash mismatch at {binary_folder / name}")
            xgb = joblib.load(binary_folder / "xgboost.pkl")
            baseline = joblib.load(binary_folder / "linear_regression.pkl")
            snapshot = panel.loc[panel["week_date"].eq(as_of)].copy()
            valid = snapshot[list(cfg.features)].notna().all(axis=1)
            predictions = snapshot[["ticker", "sector", "week_date", FORWARD_TARGET]].copy()
            predictions["next_week_date"] = predictions["week_date"].map(next_map)
            predictions["model_as_of"] = as_of
            predictions["xgboost_prediction"] = np.nan
            predictions["xgboost_raw_prediction"] = np.nan
            predictions["linear_prediction"] = np.nan
            if valid.any():
                current = snapshot.loc[valid, list(cfg.features)]
                predictions.loc[valid, "xgboost_prediction"] = xgb.predict(current)
                predictions.loc[valid, "xgboost_raw_prediction"] = xgb.predict_raw(current)
                predictions.loc[valid, "linear_prediction"] = baseline.predict(current)
            predictions.to_parquet(self.output_dir / f"as_of={as_of.date()}" / "predictions.parquet", index=False)
            all_predictions.append(predictions)
            for model, column in [("xgboost", "xgboost_raw_prediction"), ("linear_regression", "linear_prediction")]:
                weekly_metrics.append({"week_date": as_of, "model": model,
                                       **regression_metrics(predictions[FORWARD_TARGET], predictions[column])})
            model_records.append({"as_of": str(as_of.date()), "best_params": metadata["best_params"],
                                  "training_rows": metadata["training_rows"],
                                  "xgboost_cv_mse": metadata["xgboost_cv_mse"],
                                  "linear_cv_mse": metadata["linear_cv_mse"]})
            LOGGER.info("Completed model %s/%s: %s; train rows=%s; selected net CV IR=%s; diagnostic MSE xgb=%.6f, linear=%.6f",
                        number, len(dates), as_of.date(), metadata["training_rows"],
                        metadata["best_cv_portfolio"]["cost_10bps"]["information_ratio"],
                        metadata["xgboost_cv_mse"], metadata["linear_cv_mse"])
        combined = pd.concat(all_predictions, ignore_index=True)
        metric_frame = pd.DataFrame(weekly_metrics)
        combined.to_parquet(self.output_dir / "predictions.parquet", index=False)
        metric_frame.to_parquet(self.output_dir / "weekly_metrics.parquet", index=False)
        aggregate = {}
        for model, column in [("xgboost", "xgboost_raw_prediction"), ("linear_regression", "linear_prediction")]:
            metrics = regression_metrics(combined[FORWARD_TARGET], combined[column])
            ic = metric_frame.loc[metric_frame["model"].eq(model), "spearman_ic"].dropna().astype(float)
            metrics["mean_weekly_spearman_ic"] = float(ic.mean()) if len(ic) else None
            metrics["weekly_ic_std"] = float(ic.std(ddof=1)) if len(ic) > 1 else None
            metrics["positive_weekly_ic_fraction"] = float(ic.gt(0).mean()) if len(ic) else None
            aggregate[model] = metrics
        summary = {
            "config": asdict(cfg), "source": source_metadata or {}, "panel_rows": len(panel),
            "panel_weeks": len(weeks), "panel_start": str(weeks[0].date()), "panel_end": str(weeks[-1].date()),
            "model_count": len(dates), "first_model_date": str(dates[0].date()),
            "last_model_date": str(dates[-1].date()), "prediction_rows": len(combined),
            "missing_forward_labels": int(combined[FORWARD_TARGET].isna().sum()),
            "missing_predictions": int(combined["xgboost_prediction"].isna().sum()),
            "metrics": aggregate, "models": model_records,
            "timing": "Fit *_train -> active_return_train through as-of close; score current features -> active_return_fwd",
            "target_units": "Sector-ranked weekly z-scores, not percentage returns",
        }
        write_json(self.output_dir / "run_summary.json", summary)
        return summary
