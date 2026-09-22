"""Incumbent-protected V3 search for V1 XGBoost and V2 LightGBM only.

The original incumbent metadata is authoritative for every as-of date. The
incumbent is always candidate zero. Each challenger changes one parameter,
retains the exact incumbent tree count and seeds, and is tested on four
expanding weekly folds with the original one-week gap. A candidate can replace
the incumbent only after a conservative, explicit paired validation gate.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from lightgbm_listwise_training import ranking_inputs
from portfolio_training import SeedRankEnsemble, attach_training_returns, raw_training_outcomes
from training import TrainingConfig, date_cv_splits, prepare_panel, weekly_training_window, write_json
from v3_incumbent_replay import OUTPUT, PANEL_PATH, SOURCE
from v3_portfolio import evaluate_portfolio, percentile_ensemble, portfolio_metrics

LOGGER = logging.getLogger(__name__)
RAW_PATH = Path("local_cache/historical_features_731a66adc016.parquet")
FOLDS = 4
GAP_WEEKS = 1
COST_BPS = 5
MIN_IR_GAIN = .10
MIN_ANNUAL_NET_GAIN = .005
MIN_FOLD_WINS = 3
MAX_FOLD_IR_LOSS = -.25
BOOTSTRAP_QUANTILE = .10
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_BLOCK_WEEKS = 4


def _source_returns(panel, raw):
    outcomes = raw_training_outcomes(panel, raw)
    source = raw.rename(columns={c:c.lower() for c in raw.columns})
    source = source[["ticker", "week_date", "return_1w"]].copy()
    source.ticker = source.ticker.astype("string").str.strip().str.upper()
    source.week_date = pd.to_datetime(source.week_date).dt.normalize()
    source = source.rename(columns={"return_1w":"raw_stock_return"})
    outcomes = outcomes.merge(source, on=["ticker", "week_date"], validate="one_to_one")
    if not np.isfinite(outcomes.raw_stock_return.to_numpy(dtype=float)).all():
        raise ValueError("Raw training stock returns incomplete")
    return outcomes


def _manifest(kind, date):
    root, _, filename = SOURCE[kind]
    path = root/f"as_of={pd.Timestamp(date).date()}"/"metadata.json"
    metadata = json.loads(path.read_text())
    if metadata["training_weeks"] != 260 or metadata["as_of"] != str(pd.Timestamp(date).date()):
        raise ValueError(f"Incumbent manifest does not match requested date: {path}")
    saved = path.parent/filename
    expected = (metadata["pickle_sha256"] if kind == "lightgbm"
                else metadata["pickle_sha256"][filename])
    if hashlib.sha256(saved.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Incumbent pickle hash mismatch: {saved}")
    return metadata, saved


def candidate_specs(kind, metadata):
    """An unchanged incumbent plus one-parameter, conservative neighbours."""
    base = dict(metadata["best_params"])
    specs = [dict(name="incumbent", params=base, changed={})]
    if kind == "xgboost":
        edits = [
            ("shallower_depth", {"max_depth":max(1, base["max_depth"]-1)}),
            ("larger_child", {"min_child_weight":base["min_child_weight"]*2}),
            ("stronger_l2", {"reg_lambda":2.0}),
        ]
    elif kind == "lightgbm":
        edits = [
            ("fewer_leaves", {"num_leaves":max(5, base["num_leaves"]//2)}),
            ("larger_leaf_min", {"min_child_samples":int(base["min_child_samples"]*1.5)}),
            ("stronger_l2", {"reg_lambda":float(base["reg_lambda"])+2.0}),
        ]
    else:
        raise ValueError(kind)
    for name, changed in edits:
        if all(base.get(key) == value for key, value in changed.items()):
            continue
        params = dict(base, **changed)
        if params["n_estimators"] != base["n_estimators"]:
            raise ValueError("Challengers must retain incumbent tree count")
        specs.append(dict(name=name, params=params, changed=changed))
    if specs[0]["params"] != metadata["best_params"] or len(specs) < 2:
        raise ValueError("Exact incumbent was lost from candidate set")
    return specs


def _fit_predict(kind, params, seeds, fit, evaluation, features):
    """Match each incumbent's final fit mechanics on an earlier fold."""
    predictions = []
    if kind == "xgboost":
        columns = [f"{f}_train" for f in features]
        X = fit[columns].copy()
        X.columns = features
        y = fit.active_return_train
        VX = evaluation[columns].copy()
        VX.columns = features
        for seed in seeds:
            model = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                 device="cpu", random_state=seed, n_jobs=1, **params)
            model.fit(X, y)
            predictions.append(model.predict(VX))
        if len(predictions) != 1:
            raise ValueError("V1 XGBoost incumbent must remain single-seed")
        return np.asarray(predictions[0], dtype=float)
    X, labels, groups, _ = ranking_inputs(fit, features)
    VX, _, _, _ = ranking_inputs(evaluation, features)
    for seed in seeds:
        model = lgb.LGBMRanker(**dict(params, random_state=seed)).fit(X, labels, group=groups)
        predictions.append(model.booster_.predict(VX, num_threads=1))
    if len(predictions) != 3:
        raise ValueError("V2 LightGBM incumbent must remain a three-seed ensemble")
    return percentile_ensemble(np.column_stack(predictions), evaluation.week_date)


def _folds(train):
    splits = list(date_cv_splits(train.week_date, FOLDS, GAP_WEEKS))
    if len(splits) != FOLDS:
        raise ValueError("Expected four chronological folds")
    for fit, validation in splits:
        a, b = train.iloc[fit], train.iloc[validation]
        if (b.week_date.min()-a.week_date.max()).days != 14:
            raise ValueError("Expected an unused calendar week before validation")
        yield a, b


def _candidate_validation(kind, spec, seeds, train, features):
    fold_rows, scored, weeklies = [], [], []
    for index, (fit, evaluation) in enumerate(_folds(train), 1):
        scores = _fit_predict(kind, spec["params"], seeds, fit, evaluation, features)
        frame = evaluation[["ticker", "week_date", "active_return_train",
                            "raw_active_return", "raw_stock_return"]].copy()
        frame["score"] = scores
        weekly = evaluate_portfolio(frame, active_col="raw_active_return",
                                    stock_col="raw_stock_return",
                                    target_col="active_return_train",
                                    buffer=.20, rank_weighted=False, cost_bps=COST_BPS)
        metrics = portfolio_metrics(weekly)
        fold_rows.append(dict(fold=index, fit_start=str(fit.week_date.min().date()),
                              fit_end=str(fit.week_date.max().date()),
                              validation_start=str(evaluation.week_date.min().date()),
                              validation_end=str(evaluation.week_date.max().date()),
                              **{k:v for k,v in metrics.items() if k != "yearly_performance"}))
        scored.append(frame)
        weeklies.append(weekly.assign(fold=index))
    all_scores = pd.concat(scored, ignore_index=True).sort_values(["week_date", "ticker"])
    all_weekly = pd.concat(weeklies, ignore_index=True)
    # The four validation blocks are separated by training blocks. Costs for
    # their first weeks are entry-from-cash, exactly as in each fold above.
    all_metrics = portfolio_metrics(all_weekly)
    return dict(folds=fold_rows, metrics=all_metrics,
                weekly=all_weekly, scores=all_scores)


def _block_lower_bound(incumbent, challenger, date):
    """90% one-sided lower bound on annualized paired net-active improvement."""
    paired = incumbent[["week_date", "fold", "net_active_return"]].merge(
        challenger[["week_date", "fold", "net_active_return"]],
        on=["week_date", "fold"], validate="one_to_one", suffixes=("_inc", "_new"))
    if len(paired) != len(incumbent) or len(paired) != len(challenger):
        raise ValueError("Paired validation weeks differ")
    rng = np.random.default_rng(int(pd.Timestamp(date).strftime("%Y%m%d")))
    samples = np.zeros(BOOTSTRAP_REPLICATES)
    count = 0
    for _, group in paired.groupby("fold", sort=True):
        delta = (group.net_active_return_new - group.net_active_return_inc).to_numpy()
        n = len(delta)
        blocks = int(np.ceil(n/BOOTSTRAP_BLOCK_WEEKS))
        starts = rng.integers(0, n, size=(BOOTSTRAP_REPLICATES, blocks))
        offsets = np.arange(BOOTSTRAP_BLOCK_WEEKS)
        selected = delta[(starts[:, :, None]+offsets) % n].reshape(BOOTSTRAP_REPLICATES, -1)[:, :n]
        samples += selected.sum(axis=1)
        count += n
    return float(np.quantile(52*samples/count, BOOTSTRAP_QUANTILE))


def _gate(incumbent, challenger, date):
    im, cm = incumbent["metrics"], challenger["metrics"]
    fold_gains = [c["active_ir"]-i["active_ir"]
                  for i,c in zip(incumbent["folds"], challenger["folds"])]
    ir_gain = cm["active_ir"]-im["active_ir"]
    net_gain = cm["net_active_return"]-im["net_active_return"]
    lower = _block_lower_bound(incumbent["weekly"], challenger["weekly"], date)
    years = sorted(set(im["yearly_performance"]) & set(cm["yearly_performance"]))
    year_gains = {year:cm["yearly_performance"][year]["net_active_return"]-
                 im["yearly_performance"][year]["net_active_return"] for year in years}
    checks = dict(ir_gain_at_least_0_10=bool(ir_gain>=MIN_IR_GAIN),
                  annual_net_gain_at_least_0_5pct=bool(net_gain>=MIN_ANNUAL_NET_GAIN),
                  fold_ir_wins_at_least_3=bool(sum(x>0 for x in fold_gains)>=MIN_FOLD_WINS),
                  no_fold_ir_loss_below_0_25=bool(min(fold_gains)>=MAX_FOLD_IR_LOSS),
                  bootstrap_lower_gain_positive=bool(lower>0))
    return dict(eligible=all(checks.values()), checks=checks,
                ir_gain=float(ir_gain), annualized_net_active_gain=float(net_gain),
                fold_ir_gains=[float(x) for x in fold_gains],
                positive_years=int(sum(x>0 for x in year_gains.values())),
                year_net_active_gains=year_gains,
                bootstrap_10pct_lower_annual_gain=lower)


def _refit_challenger(kind, params, seeds, train, features):
    if kind == "xgboost":
        columns = [f"{f}_train" for f in features]
        X = train[columns].copy()
        X.columns = features
        model = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                             device="cpu", random_state=seeds[0], n_jobs=1,
                             **params).fit(X, train.active_return_train)
        return model
    X, labels, groups, _ = ranking_inputs(train, features)
    boosters = [lgb.LGBMRanker(**dict(params, random_state=seed)).fit(
        X, labels, group=groups).booster_ for seed in seeds]
    return SeedRankEnsemble(boosters, features, "lightgbm")


def fit_date(kind, panel, outcomes, date, output=OUTPUT):
    """Select an unchanged incumbent or a gate-passing challenger for one date."""
    date = pd.Timestamp(date)
    metadata, saved_incumbent = _manifest(kind, date)
    features = list(metadata["config"]["features"])
    config = TrainingConfig(window_weeks=260, features=tuple(features), threads=1)
    train = attach_training_returns(weekly_training_window(panel, date, config, 260), outcomes)
    seeds = tuple(metadata.get("ensemble_seeds", [metadata["config"]["seed"]]))
    if kind == "xgboost" and seeds != (42,):
        raise ValueError("V1 XGBoost seed changed")
    if kind == "lightgbm" and seeds != (42, 73, 107):
        raise ValueError("V2 LightGBM seeds changed")
    specs = candidate_specs(kind, metadata)
    path = Path(output)/"models"/kind/f"as_of={date.date()}"
    marker = path/"selection.json"
    identity = dict(model=kind, date=str(date.date()), incumbent_sha256=hashlib.sha256(
        saved_incumbent.read_bytes()).hexdigest(), candidates=specs,
        gate=dict(min_ir_gain=MIN_IR_GAIN, min_annual_net_gain=MIN_ANNUAL_NET_GAIN,
                  min_fold_wins=MIN_FOLD_WINS, max_fold_ir_loss=MAX_FOLD_IR_LOSS,
                  bootstrap_quantile=BOOTSTRAP_QUANTILE,
                  bootstrap_replicates=BOOTSTRAP_REPLICATES,
                  bootstrap_block_weeks=BOOTSTRAP_BLOCK_WEEKS))
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior["signature"] != signature:
            raise ValueError(f"Saved focused V3 search differs at {path}")
        if prior["selected"] != "incumbent":
            saved = path/"challenger.pkl"
            if hashlib.sha256(saved.read_bytes()).hexdigest() != prior["challenger_sha256"]:
                raise ValueError(f"Saved challenger hash differs at {saved}")
        return prior
    path.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        evaluated = [_candidate_validation(kind, spec, seeds, train, features)
                     for spec in specs]
    incumbent = evaluated[0]
    gates = [None]+[_gate(incumbent, item, date) for item in evaluated[1:]]
    eligible = [i for i, gate in enumerate(gates) if gate and gate["eligible"]]
    selected_index = (max(eligible, key=lambda i:(evaluated[i]["metrics"]["active_ir"],
                                                  evaluated[i]["metrics"]["net_active_return"],
                                                  -i)) if eligible else 0)
    winner = specs[selected_index]
    challenger_sha256 = None
    if selected_index:
        model = _refit_challenger(kind, winner["params"], seeds, train, features)
        temporary = path/"challenger.pkl.tmp"
        joblib.dump(model, temporary)
        temporary.replace(path/"challenger.pkl")
        challenger_sha256 = hashlib.sha256((path/"challenger.pkl").read_bytes()).hexdigest()
    reason = (f"{winner['name']} passed all five incumbent-protection checks"
              if selected_index else "No challenger passed all five incumbent-protection checks")
    result = dict(signature=signature, identity=identity, selected=winner["name"],
                  selected_index=selected_index, reason=reason,
                  incumbent_model_path=str(saved_incumbent),
                  challenger_sha256=challenger_sha256,
                  seeds=seeds, features=features,
                  candidates=[dict(name=spec["name"], params=spec["params"],
                                   changed=spec["changed"],
                                   fold_results=ev["folds"],
                                   validation_metrics={k:v for k,v in ev["metrics"].items()
                                                       if k != "yearly_performance"},
                                   yearly_performance=ev["metrics"]["yearly_performance"],
                                   gate=gate)
                              for spec,ev,gate in zip(specs,evaluated,gates)])
    write_json(marker, result)
    return result
