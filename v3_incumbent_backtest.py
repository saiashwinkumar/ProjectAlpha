"""Run the focused, incumbent-protected V3 XGBoost/LightGBM backtest."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from training import TrainingConfig, prepare_panel, write_json
from v3_incumbent_replay import OUTPUT, PANEL_PATH, SOURCE, replay
from v3_incumbent_search import RAW_PATH, _source_returns, fit_date
from v3_portfolio import evaluate_portfolio, portfolio_metrics

LOGGER = logging.getLogger(__name__)
_PANEL = _OUTCOMES = _ROOT = None


def _init_worker(panel, outcomes, root):
    global _PANEL, _OUTCOMES, _ROOT
    _PANEL, _OUTCOMES, _ROOT = panel, outcomes, root


def _worker(task):
    kind, date = task
    with threadpool_limits(limits=1):
        chosen = fit_date(kind, _PANEL, _OUTCOMES, date, _ROOT)
    return kind, str(pd.Timestamp(date).date()), chosen["selected"]


def _score_selected(panel, archived, root):
    scores, selections, candidates, folds = [], [], [], []
    for (kind, date), group in archived.groupby(["model", "week_date"], sort=True):
        path = root/"models"/kind/f"as_of={date.date()}"
        record = json.loads((path/"selection.json").read_text())
        group = group.sort_values("ticker").copy()
        if record["selected"] != "incumbent":
            saved = path/"challenger.pkl"
            if hashlib.sha256(saved.read_bytes()).hexdigest() != record["challenger_sha256"]:
                raise ValueError(f"Challenger model hash mismatch: {saved}")
            model = joblib.load(saved)
            snapshot = panel.loc[panel.week_date.eq(date)].sort_values("ticker")
            if not snapshot.ticker.reset_index(drop=True).equals(group.ticker.reset_index(drop=True)):
                raise ValueError(f"Selected model universe changed at {date}")
            features = record["features"]
            valid = snapshot[features].notna().all(axis=1)
            prediction = np.full(len(snapshot), np.nan)
            if kind == "lightgbm":
                prediction[valid.to_numpy()] = model.predict(
                    snapshot.loc[valid, features].astype(np.float32), num_threads=1)
            else:
                prediction[valid.to_numpy()] = model.predict(snapshot.loc[valid, features])
            group["score"] = prediction
        group["selected_candidate"] = record["selected"]
        scores.append(group)
        selections.append(dict(model=kind, week_date=date,
                               selected=record["selected"],
                               reason=record["reason"],
                               incumbent_model_path=record["incumbent_model_path"],
                               challenger_model_path=(str(path/"challenger.pkl")
                                                      if record["selected"] != "incumbent" else None)))
        for candidate in record["candidates"]:
            base = dict(model=kind, week_date=date, candidate=candidate["name"],
                        params=json.dumps(candidate["params"], sort_keys=True),
                        changed=json.dumps(candidate["changed"], sort_keys=True),
                        selected=bool(candidate["name"] == record["selected"]),
                        incumbent=bool(candidate["name"] == "incumbent"),
                        gate_eligible=(candidate["gate"]["eligible"] if candidate["gate"] else None),
                        gate_checks=json.dumps(candidate["gate"]["checks"])
                        if candidate["gate"] else None,
                        fold_ir_wins=(sum(x>0 for x in candidate["gate"]["fold_ir_gains"])
                        if candidate["gate"] else None),
                        bootstrap_10pct_lower_annual_gain=(candidate["gate"][
                            "bootstrap_10pct_lower_annual_gain"] if candidate["gate"] else None),
                        positive_years=(candidate["gate"]["positive_years"]
                                        if candidate["gate"] else None),
                        yearly_performance=json.dumps(candidate["yearly_performance"], sort_keys=True))
            base.update({k:v for k,v in candidate["validation_metrics"].items()
                         if k != "yearly_performance"})
            candidates.append(base)
            for fold in candidate["fold_results"]:
                folds.append(dict(model=kind, week_date=date,
                                  candidate=candidate["name"], **fold))
    return (pd.concat(scores, ignore_index=True), pd.DataFrame(selections),
            pd.DataFrame(candidates), pd.DataFrame(folds))


def _report(archived, selected_scores, root, selections, candidates, folds):
    comparison, weekly_rows, yearly = [], [], []
    for kind in ("xgboost", "lightgbm"):
        for status, source in (("incumbent", archived), ("protected_selection", selected_scores)):
            group = source.loc[source.model.eq(kind)]
            weekly = evaluate_portfolio(group, score_col="score",
                                        active_col="raw_active_return_fwd",
                                        stock_col="stock_return_fwd",
                                        target_col="active_return_fwd",
                                        buffer=.20, rank_weighted=False, cost_bps=5)
            metrics = portfolio_metrics(weekly)
            comparison.append(dict(model=kind, status=status,
                                   **{k:v for k,v in metrics.items() if k != "yearly_performance"}))
            weekly_rows.append(weekly.assign(model=kind, status=status))
            for year, values in metrics["yearly_performance"].items():
                yearly.append(dict(model=kind, status=status, year=int(year), **values))
    table = pd.DataFrame(comparison)
    for kind in ("xgboost", "lightgbm"):
        mask = table.model.eq(kind)
        baseline = table.loc[mask & table.status.eq("incumbent")].iloc[0]
        table.loc[mask, "ir_change_vs_incumbent"] = table.loc[mask, "active_ir"]-baseline.active_ir
        table.loc[mask, "annual_net_active_change_vs_incumbent"] = (
            table.loc[mask, "net_active_return"]-baseline.net_active_return)
    table.to_csv(root/"comparison.csv", index=False)
    pd.DataFrame(yearly).to_csv(root/"yearly_performance.csv", index=False)
    pd.concat(weekly_rows, ignore_index=True).to_parquet(root/"weekly_portfolios.parquet", index=False)
    selected_scores.to_parquet(root/"selected_scores.parquet", index=False)
    selections.to_csv(root/"selections.csv", index=False)
    candidates.to_csv(root/"candidate_validation.csv", index=False)
    folds.to_csv(root/"fold_validation.csv", index=False)
    summary = dict(dates=216, realized_weeks=215,
                   models=["xgboost", "lightgbm"],
                   selected_challenger_dates={kind:int((selections.loc[
                       selections.model.eq(kind), "selected"] != "incumbent").sum())
                       for kind in ("xgboost", "lightgbm")},
                   incumbent_replay=json.loads((root/"incumbent_replay_audit.json").read_text()),
                   comparison=table.to_dict("records"),
                   policy="Exact saved incumbent per date; one-parameter conservative challengers; 4 chronological folds with 1-week gaps; 5 bps net equal-weight exact Q1; retain incumbent unless all five protection checks pass. No OOS outcomes enter selection.")
    write_json(root/"summary.json", summary)
    return summary


def _verified_replay(root):
    """Reuse a finished full-refit audit only while every source is unchanged."""
    marker = root/"incumbent_replay_audit.json"
    scores = root/"incumbent_scores.parquet"
    model_audit = root/"incumbent_model_audit.parquet"
    if not all(path.exists() for path in (marker, scores, model_audit)):
        return replay(PANEL_PATH, root)
    record = json.loads(marker.read_text())
    if (record.get("status") != "verified" or record.get("full_refit_models") != 432
            or record.get("dates") != 216):
        return replay(PANEL_PATH, root)
    inputs = [PANEL_PATH, *[saved for _, saved, _ in SOURCE.values()]]
    if any(path.stat().st_mtime > marker.stat().st_mtime for path in inputs):
        return replay(PANEL_PATH, root)
    audit = pd.read_parquet(model_audit)
    if len(audit) != 432:
        return replay(PANEL_PATH, root)
    for row in audit.itertuples():
        if hashlib.sha256(Path(row.model_path).read_bytes()).hexdigest() != row.pickle_sha256:
            return replay(PANEL_PATH, root)
    return record


def run(root=OUTPUT, workers=4, score_only=False, start="2022-01-01", end=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # This call must succeed before any tuning starts; it also rechecks hashes.
    verification = _verified_replay(root)
    if verification["status"] != "verified":
        raise ValueError("Incumbent reproduction failed")
    panel = prepare_panel(pd.read_parquet(PANEL_PATH), TrainingConfig(window_weeks=260, threads=1))
    archived = pd.read_parquet(root/"incumbent_scores.parquet")
    dates = pd.DatetimeIndex(panel.loc[panel.week_date.ge(start), "week_date"].unique()).sort_values()
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    if not len(dates):
        raise ValueError("No focused V3 dates")
    if not score_only:
        raw = pd.read_parquet(RAW_PATH)
        outcomes = _source_returns(panel, raw)
        tasks = [(kind, date) for date in dates for kind in ("xgboost", "lightgbm")]
        if workers == 1:
            _init_worker(panel, outcomes, root)
            for i, result in enumerate(map(_worker, tasks), 1):
                LOGGER.info("Focused V3 trained %d/%d: %s %s -> %s",
                            i, len(tasks), *result)
        else:
            with ProcessPoolExecutor(max_workers=workers,
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_init_worker,
                                     initargs=(panel, outcomes, root)) as pool:
                for i, result in enumerate(pool.map(_worker, tasks), 1):
                    LOGGER.info("Focused V3 trained %d/%d: %s %s -> %s",
                                i, len(tasks), *result)
    if len(dates) != 216:
        return dict(status="partial_training_only", trained_dates=len(dates))
    selected_scores, selections, candidates, folds = _score_selected(panel, archived, root)
    return _report(archived, selected_scores, root, selections, candidates, folds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.root/"focused_backfill.log")])
    result = run(args.root, args.workers, args.score_only, args.start, args.end)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
