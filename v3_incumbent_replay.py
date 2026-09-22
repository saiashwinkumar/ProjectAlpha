"""Replay the exact saved V1 XGBoost and V2 LightGBM weekly incumbents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from lightgbm_listwise_training import ranking_inputs
from portfolio_training import SeedRankEnsemble
from training import TrainingConfig, prepare_panel, weekly_training_window, write_json
from v3_portfolio import evaluate_portfolio, portfolio_metrics

PANEL_PATH = Path("local_cache/athena_panel_cc5c45cb64a96a00.parquet")
SOURCE = {
    "xgboost": (Path("artifacts/ml/continuous_260w_v1/regression"),
                 Path("artifacts/ml/continuous_260w_v1/backtest/stock_scores_and_quintiles.parquet"),
                 "xgboost.pkl"),
    "lightgbm": (Path("artifacts/ml/continuous_260w_v2/lightgbm"),
                  Path("artifacts/ml/continuous_260w_v2/backtest/stock_scores_and_quintiles.parquet"),
                  "lightgbm.pkl"),
}
EXPECTED_V1 = {"active_ir": 0.6083829085371613,
               "net_active_return": 0.03776052324664401}
OUTPUT = Path("artifacts/ml/continuous_260w_v3_incumbent_protected")


def _same(a, b, *, atol=1e-11, rtol=1e-10):
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                       atol=atol, rtol=rtol, equal_nan=True)


def replay(panel_path=PANEL_PATH, output=OUTPUT):
    """Re-infer all saved pickles and fail before tuning on any discrepancy."""
    panel = prepare_panel(pd.read_parquet(panel_path), TrainingConfig(window_weeks=260, threads=1))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit, reproduced, references = [], [], {}
    all_dates = pd.DatetimeIndex(panel.loc[panel.week_date.ge("2022-01-01"),
                                         "week_date"].unique()).sort_values()
    if len(all_dates) != 216:
        raise ValueError(f"Expected 216 incumbent dates, found {len(all_dates)}")
    for kind, (root, saved_path, filename) in SOURCE.items():
        archived = pd.read_parquet(saved_path)
        archived = archived.loc[archived.model.eq(kind)].copy()
        if not pd.DatetimeIndex(archived.week_date.unique()).sort_values().equals(all_dates):
            raise ValueError(f"Archived {kind} scoring dates differ from panel")
        seen = []
        for date in all_dates:
            directory = root/f"as_of={date.date()}"
            metadata = json.loads((directory/"metadata.json").read_text())
            if metadata["as_of"] != str(date.date()) or metadata["training_weeks"] != 260:
                raise ValueError(f"Wrong incumbent date/window: {directory}")
            if metadata["last_training_week"] != str(date.date()):
                raise ValueError(f"Incumbent training ends after/before score date: {directory}")
            saved = directory/filename
            digest = hashlib.sha256(saved.read_bytes()).hexdigest()
            expected = (metadata["pickle_sha256"] if kind == "lightgbm"
                        else metadata["pickle_sha256"][filename])
            if digest != expected:
                raise ValueError(f"Incumbent pickle hash mismatch: {saved}")
            features = list(metadata["config"]["features"])
            snapshot = panel.loc[panel.week_date.eq(date)].sort_values("ticker")
            observed = archived.loc[archived.week_date.eq(date)].sort_values("ticker")
            if not snapshot.ticker.reset_index(drop=True).equals(observed.ticker.reset_index(drop=True)):
                raise ValueError(f"Incumbent scoring universe differs at {date}")
            valid = snapshot[features].notna().all(axis=1)
            scores = np.full(len(snapshot), np.nan)
            model = joblib.load(saved)
            names = (model.feature_name() if kind == "lightgbm"
                     else list(model.feature_names_in_))
            if names != features:
                raise ValueError(f"Incumbent feature order differs: {saved}")
            if valid.any():
                X = snapshot.loc[valid, features]
                if kind == "lightgbm":
                    scores[valid.to_numpy()] = model.predict(X.astype(np.float32), num_threads=1)
                else:
                    scores[valid.to_numpy()] = model.predict(X)
            if not _same(scores, observed.prediction):
                maximum = np.nanmax(np.abs(scores-observed.prediction.to_numpy(dtype=float)))
                raise ValueError(f"Saved {kind} prediction replay differs at {date}: {maximum}")
            # Also rebuild the saved winner from its exact historical manifest.
            # This verifies training logic and seeds, not merely pickle loading.
            config = TrainingConfig(window_weeks=260, features=tuple(features), threads=1)
            train = weekly_training_window(panel, date, config, 260)
            params = metadata["best_params"]
            with threadpool_limits(limits=1):
                if kind == "xgboost":
                    X = train[[f"{f}_train" for f in features]].copy()
                    X.columns = features
                    refit = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                         device="cpu", random_state=metadata["config"]["seed"],
                                         n_jobs=1, **params).fit(X, train.active_return_train)
                    rebuilt = refit.predict(snapshot.loc[valid, features])
                else:
                    X, labels, groups, _ = ranking_inputs(train, features)
                    seeds = metadata["ensemble_seeds"]
                    boosters = [lgb.LGBMRanker(**dict(params, random_state=seed)).fit(
                        X, labels, group=groups).booster_ for seed in seeds]
                    refit = SeedRankEnsemble(boosters, features, "lightgbm")
                    rebuilt = refit.predict(snapshot.loc[valid, features].astype(np.float32),
                                             num_threads=1)
            if not _same(rebuilt, scores[valid.to_numpy()]):
                raise ValueError(f"Full refit of exact {kind} incumbent differs at {date}")
            part = observed[["ticker", "week_date", "sector", "active_return_fwd",
                             "raw_active_return_fwd", "stock_return_fwd"]].copy()
            part["score"] = scores
            seen.append(part)
            audit.append(dict(model=kind, week_date=date, model_path=str(saved),
                              pickle_sha256=digest, training_weeks=260,
                              parameters=metadata["best_params"],
                              ensemble_seeds=metadata.get("ensemble_seeds", [metadata["config"]["seed"]])))
        replayed = pd.concat(seen, ignore_index=True)
        reference = archived.rename(columns={"prediction":"score"})
        original_metrics = portfolio_metrics(evaluate_portfolio(
            reference, active_col="raw_active_return_fwd", stock_col="stock_return_fwd",
            target_col="active_return_fwd", buffer=.20, rank_weighted=False, cost_bps=5))
        replay_metrics = portfolio_metrics(evaluate_portfolio(
            replayed, active_col="raw_active_return_fwd", stock_col="stock_return_fwd",
            target_col="active_return_fwd", buffer=.20, rank_weighted=False, cost_bps=5))
        for metric in ("annualized_stock_return", "annualized_active_return",
                       "net_active_return", "active_ir", "q1_capture", "turnover",
                       "max_drawdown", "annualized_transaction_cost"):
            if not np.isclose(original_metrics[metric], replay_metrics[metric],
                              rtol=1e-9, atol=1e-11):
                raise ValueError(f"{kind} reproduced {metric} differs from archived result")
        if kind == "xgboost":
            for metric, expected in EXPECTED_V1.items():
                if not np.isclose(replay_metrics[metric], expected, rtol=1e-9, atol=1e-11):
                    raise ValueError(f"V1 XGBoost {metric} differs from fixed saved hurdle")
        references[kind] = {k:v for k,v in replay_metrics.items() if k != "yearly_performance"}
        reproduced.append(replayed.assign(model=kind))
    pd.concat(reproduced, ignore_index=True).to_parquet(output/"incumbent_scores.parquet", index=False)
    pd.DataFrame(audit).to_parquet(output/"incumbent_model_audit.parquet", index=False)
    result = dict(status="verified", dates=216, model_date_pickles=len(audit),
                  full_refit_models=len(audit),
                  score_tolerance=dict(atol=1e-11, rtol=1e-10),
                  metric_tolerance=dict(atol=1e-11, rtol=1e-9),
                  reference_metrics=references,
                  xgboost_configuration="Exact V1 per-date best_params (identical across dates), seed 42",
                  lightgbm_configuration="Exact V2 per-date best_params and three seeds 42/73/107")
    write_json(output/"incumbent_replay_audit.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
