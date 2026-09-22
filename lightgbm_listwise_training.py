"""Weekly 260-observation-window LambdaRank pipeline for ProjectAlpha.

Adapted from the supplied LightGBM listwise scripts. Fit lagged features to
observed targets, tune within the window, then score current features. Ranking
scores and target spreads are not percentage returns.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import multiprocessing
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from portfolio_training import (SeedRankEnsemble, attach_training_returns, better_portfolio,
                                portfolio_summary, portfolio_weekly, raw_training_outcomes)

from training import (
    FEATURES, FORWARD_TARGET, TARGET, TrainingConfig, date_cv_splits,
    frame_fingerprint, prepare_panel, write_json, weekly_training_window, RollingPanelTrainer,
)

LOGGER = logging.getLogger(__name__)
WINDOW_WEEKS = 260
METRICS = ("mean_ls_ndcg", "mean_long_ndcg", "mean_short_ndcg",
           "mean_q1_q5_spread", "q1_q5_information_ratio", "mean_turnover", "mean_spearman")


@dataclass(frozen=True)
class RankingConfig:
    data_start: str = "2017-01-01"
    features: tuple[str, ...] = FEATURES
    cv_splits: int = 4
    cv_gap_weeks: int = 1
    threads: int = 4
    seed: int = 42
    n_estimators: int = 600
    learning_rate: float = 0.05
    early_stopping_rounds: int = 30
    stage1_truncation: int = 30
    stage2_offset: int = 3
    stage3_threshold: float = 0.90
    stage3_learning_rate: float = 0.03
    stage3_n_estimators: int = 1000
    stage3_early_stopping_rounds: int = 50

    def __post_init__(self):
        self.panel_config()  # Reuse the feature allowlist and calendar settings.
        for field in ("n_estimators", "early_stopping_rounds", "stage1_truncation",
                      "stage3_n_estimators", "stage3_early_stopping_rounds"):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if self.stage2_offset < 0 or not 0 < self.stage3_threshold <= 1:
            raise ValueError("Invalid Stage 2 offset or Stage 3 threshold")
        if not 0 < self.learning_rate <= 1 or not 0 < self.stage3_learning_rate <= 1:
            raise ValueError("Learning rates must be in (0, 1]")

    def panel_config(self):
        return TrainingConfig(data_start=self.data_start, features=self.features,
                              cv_splits=self.cv_splits, cv_gap_weeks=self.cv_gap_weeks,
                              threads=self.threads, seed=self.seed)


def ranking_window(panel, as_of, config):
    """Select exactly 260 weekly outcome snapshots, inclusive of as_of."""
    return weekly_training_window(panel, as_of, config, WINDOW_WEEKS, min_rows=2)


def relevance_labels(target):
    """Reference quintiles: higher target -> label 4; input ties use ticker order."""
    ranks = pd.Series(np.asarray(target)).rank(method="first").to_numpy()
    if len(ranks) < 2:
        return np.zeros(len(ranks), dtype=np.int32)
    return np.clip(np.ceil(ranks / len(ranks) * 5) - 1, 0, 4).astype(np.int32)


def ranking_inputs(frame, features):
    # Keep ticker order, rather than sorting on labels (which biases score ties).
    work = frame.sort_values(["week_date", "ticker"]).reset_index(drop=True)
    sizes = work.groupby("week_date", sort=False).size().to_numpy(dtype=int)
    y = np.concatenate([relevance_labels(g[TARGET]) for _, g in work.groupby("week_date", sort=False)])
    X = work[[f"{f}_train" for f in features]].astype(np.float32)
    X.columns = list(features)  # Stored names also work for current-week scoring.
    if sizes.sum() != len(X) or len(np.unique(y)) < 2:
        raise ValueError("Invalid ranking groups or constant relevance labels")
    return X, y, sizes, work


def stage1_candidates(n_rows, n_features, config):
    child = [100, 300] if n_rows < 50_000 else ([300, 1000] if n_rows < 300_000 else [1000, 3000])
    fixed = dict(objective="lambdarank", metric="None", boosting_type="gbdt",
                 n_estimators=config.n_estimators, learning_rate=config.learning_rate,
                 subsample=0.8, subsample_freq=1, colsample_bytree=1.0 if n_features < 10 else 0.8,
                 reg_alpha=0.0, label_gain=[0, 1, 2, 3, 4], lambdarank_norm=True,
                 lambdarank_truncation_level=config.stage1_truncation,
                 n_jobs=config.threads, random_state=config.seed, verbosity=-1,
                 deterministic=True, force_col_wise=True)
    candidates = [dict(fixed, num_leaves=leaves, max_depth=depth,
                       min_child_samples=c, reg_lambda=l2, learning_rate=lr)
                  for leaves, depth, c, l2, lr in [
                      (7, 3, child[1], 10.0, .02), (7, 3, child[1], 10.0, .05),
                      (15, 4, child[1], 10.0, .02), (15, 4, child[1], 1.0, .05),
                      (15, 4, child[0], 10.0, .05), (31, 5, child[1], 10.0, .02),
                      (31, 5, child[0], 1.0, .05)]]
    return candidates


def ndcg(labels, scores, k):
    labels, scores = np.asarray(labels), np.asarray(scores)
    discounts = np.log2(np.arange(2, k + 2))
    ideal = np.sort(labels)[::-1][:k]
    predicted = labels[np.argsort(-scores, kind="stable")[:k]]
    denominator = float(np.sum(ideal / discounts))
    return float(np.sum(predicted / discounts) / denominator) if denominator > 0 else 0.0


def finite_mean(values):
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(valid)) if valid else None


def summarize_metrics(weekly):
    spreads = weekly["q1_q5_spread"].dropna()
    std = spreads.std(ddof=1)
    result = {f"mean_{name}": finite_mean(weekly[name])
              for name in ("ls_ndcg", "long_ndcg", "short_ndcg", "q1_q5_spread", "turnover", "spearman")}
    result["q1_q5_information_ratio"] = float(spreads.mean() / std * np.sqrt(52)) if std > 0 else None
    result["n_groups"] = int(weekly.ls_ndcg.notna().sum())
    return result


def evaluate_scores(frame, target_col=TARGET, score_col="ranking_score"):
    """Evaluate complete weekly score universes; never filter holdings on outcomes.

    Q1 is the high-score side. Both long and short selections use ceil(20% * n).
    Missing outcomes withhold the whole week's outcome metrics. Turnover uses
    signed equal weights and is not bridged across missing calendar weeks.
    """
    records, previous, previous_date = [], None, None
    for date, group in frame.groupby("week_date", sort=True):
        g = group.dropna(subset=[score_col]).sort_values("ticker").reset_index(drop=True)
        row = dict(week_date=date, n_scored=len(g), n_observed=int(g[target_col].notna().sum()),
                   long_ndcg=None, short_ndcg=None, ls_ndcg=None, q1_q5_spread=None,
                   spearman=None, turnover=None)
        if len(g) < 2:
            records.append(row)
            previous, previous_date = None, date
            continue
        scores = g[score_col].to_numpy(dtype=float)
        k = math.ceil(0.20 * len(g))
        # A total order gives deterministic, nonoverlapping holdings even if all scores tie.
        order = np.argsort(-scores, kind="stable")
        long, short = order[:k], order[-k:]
        weights = dict.fromkeys(g.ticker, 0.0)
        for i in long:
            weights[g.ticker.iloc[i]] += 1 / k
        for i in short:
            weights[g.ticker.iloc[i]] -= 1 / k
        if previous is not None and date - previous_date == pd.Timedelta(weeks=1):
            row["turnover"] = float(0.5 * sum(abs(weights.get(t, 0) - previous.get(t, 0))
                                             for t in weights.keys() | previous.keys()))
        previous, previous_date = weights, date
        if g[target_col].notna().all():
            labels = relevance_labels(g[target_col])
            row["long_ndcg"] = ndcg(labels, scores, k)
            # Reverse the same total score order for the short side, including ties.
            reverse_order_scores = np.empty(len(g))
            reverse_order_scores[order] = np.arange(len(g))
            row["short_ndcg"] = ndcg(4 - labels, reverse_order_scores, k)
            row["ls_ndcg"] = 0.5 * (row["long_ndcg"] + row["short_ndcg"])
            row["q1_q5_spread"] = float(g[target_col].iloc[long].mean() - g[target_col].iloc[short].mean())
            if g[target_col].nunique() > 1 and g[score_col].nunique() > 1:
                row["spearman"] = float(g[target_col].rank().corr(g[score_col].rank()))
        records.append(row)
    weekly = pd.DataFrame(records)
    return weekly, summarize_metrics(weekly)


def candidate_key(result):
    if "portfolio" in result:
        net = result["portfolio"]["cost_10bps"]
        ir = net["information_ratio"]
        complexity = result["complexity"]
        return (round((ir if ir is not None else -1e6) / .05),
                round(net["cumulative_return"] / .01),
                -complexity[0], -complexity[1], -complexity[2], -complexity[3])
    metrics = result["aggregate_metrics"]
    def value(name, lower=False):
        v = metrics.get(name)
        return (-v if lower else v) if v is not None and np.isfinite(v) else -np.inf
    return (value("mean_ls_ndcg"), value("mean_q1_q5_spread"),
            value("q1_q5_information_ratio"), value("mean_turnover", lower=True))


class ValidationEvaluator:
    """Cache invariant query data across candidates; preserve reference metrics."""

    def __init__(self, ordered_frame):
        self.groups = []
        ticker_ids, tickers = pd.factorize(ordered_frame.ticker, sort=True)
        self.n_assets = len(tickers)
        offset = 0
        for date, group in ordered_frame.groupby("week_date", sort=False):
            n = len(group)
            target = group[TARGET].to_numpy(dtype=float)
            raw = (group["raw_active_return"].to_numpy(dtype=float)
                   if "raw_active_return" in group else None)
            self.groups.append((date, offset, n, target, relevance_labels(target),
                                rankdata(target), ticker_ids[offset:offset+n], raw))
            offset += n

    def portfolio_ir(self, predictions):
        """Exactly the final selector's 10 bps net, equal-weight Q1-Q5 IR."""
        spreads, previous, previous_date = [], None, None
        for date, offset, n, _, _, _, asset_ids, raw in self.groups:
            if raw is None or not np.isfinite(raw).all():
                raise ValueError("Raw validation returns are required for portfolio early stopping")
            order = np.argsort(-predictions[offset:offset+n], kind="stable")
            quintile = np.arange(n) * 5 // n + 1
            long, short = order[quintile == 1], order[quintile == 5]
            weights = np.zeros(self.n_assets)
            weights[asset_ids[long]] = 1 / len(long)
            weights[asset_ids[short]] = -1 / len(short)
            trade = (float(0.5 * np.abs(weights - previous).sum())
                     if previous is not None and (date - previous_date).days == 7 else 1.0)
            spreads.append(float(raw[long].mean() - raw[short].mean() - .002 * trade))
            previous, previous_date = weights, date
        values = np.asarray(spreads)
        std = values.std(ddof=1) if len(values) > 1 else np.nan
        return float(np.sqrt(52) * values.mean() / std) if std > 0 else -1e6

    def __call__(self, predictions):
        rows, previous, previous_date = [], None, None
        for date, offset, n, target, labels, target_rank, asset_ids, _ in self.groups:
            if n < 2:
                continue
            scores = predictions[offset:offset+n]
            k = math.ceil(0.2 * n)
            order = np.argsort(-scores, kind="stable")
            long, short = order[:k], order[-k:]
            long_ndcg = ndcg(labels, scores, k)
            reverse = np.empty(n)
            reverse[order] = np.arange(n)
            short_ndcg = ndcg(4-labels, reverse, k)
            weights = np.zeros(self.n_assets)
            weights[asset_ids[long]] += 1/k
            weights[asset_ids[short]] -= 1/k
            turnover = None
            if previous is not None and (date - previous_date).days == 7:
                turnover = float(0.5 * np.abs(weights-previous).sum())
            previous, previous_date = weights, date
            ic = None
            if np.ptp(target) > 0 and np.ptp(scores) > 0:
                ic = float(np.corrcoef(target_rank, rankdata(scores))[0, 1])
            rows.append(dict(long_ndcg=long_ndcg, short_ndcg=short_ndcg,
                             ls_ndcg=0.5*(long_ndcg+short_ndcg),
                             q1_q5_spread=float(target[long].mean()-target[short].mean()),
                             turnover=turnover, spearman=ic))
        return summarize_metrics(pd.DataFrame(rows))


def staged_search(evaluate, candidates, market_k, config):
    """Bounded search; evaluator injection also allows testing all stage branches."""
    def choose(items):
        winner = None
        for item in items:
            if winner is None or (better_portfolio(item, winner, item["complexity"])
                                  if "portfolio" in item else candidate_key(item) > candidate_key(winner)):
                winner = item
        return winner

    results = [evaluate(params, f"stage1_{i:02d}", config.early_stopping_rounds)
               for i, params in enumerate(candidates, 1)]
    first = choose(results)
    finalists = [first, choose([item for item in results if item is not first])]
    for i, result in enumerate(finalists, 1):
        params = dict(result["params"])
        if params["lambdarank_truncation_level"] == market_k + config.stage2_offset:
            continue
        params["lambdarank_truncation_level"] = market_k + config.stage2_offset
        results.append(evaluate(params, f"stage2_finalist_{i}", config.early_stopping_rounds))
    winner = choose(results)
    ceiling = winner["params"]["n_estimators"]
    triggered = (winner["median_best_iteration"] >= math.ceil(config.stage3_threshold * ceiling)
                 or winner["max_best_iteration"] >= ceiling)
    if triggered:
        params = dict(winner["params"], learning_rate=config.stage3_learning_rate,
                      n_estimators=config.stage3_n_estimators)
        results.append(evaluate(params, "stage3_conditional_refinement", config.stage3_early_stopping_rounds))
        winner = choose(results)
    return winner, results, triggered


_PROCESS_PANEL = None
_PROCESS_RAW_OUTCOMES = None


def _initialize_ranking_worker(panel, raw_outcomes):
    global _PROCESS_PANEL, _PROCESS_RAW_OUTCOMES
    _PROCESS_PANEL = panel
    _PROCESS_RAW_OUTCOMES = raw_outcomes


def _fit_ranking_worker(config, output_dir, date):
    with threadpool_limits(limits=config.threads):
        return RollingListwiseTrainer(config, output_dir, raw_outcomes=_PROCESS_RAW_OUTCOMES).fit_week(_PROCESS_PANEL, date)


class RollingListwiseTrainer:

    def __init__(self, config: RankingConfig, output_dir: Path, workers=1,
                 raw_returns=None, raw_outcomes=None):
        self.config, self.output_dir = config, Path(output_dir)
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.raw_returns, self.raw_outcomes = raw_returns, raw_outcomes

    def fitted_weeks(self, panel, dates):
        if self.workers == 1:
            with threadpool_limits(limits=self.config.threads):
                for date in dates:
                    yield date, self.fit_week(panel, date)
            return
        # Processes keep candidate evaluation from competing for the Python GIL.
        # Each process receives the panel once; only dates and fitted models travel per task.
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize_ranking_worker,
                                 initargs=(panel, self.raw_outcomes)) as executor:
            remaining, pending = iter(dates), deque()
            for date in dates[:self.workers]:
                next(remaining)
                pending.append((date, executor.submit(_fit_ranking_worker, self.config, self.output_dir, date)))
            while pending:
                date, future = pending.popleft()
                fitted = future.result()
                following = next(remaining, None)
                if following is not None:
                    pending.append((following, executor.submit(_fit_ranking_worker, self.config, self.output_dir, following)))
                yield date, fitted

    def fit_week(self, panel, as_of):
        cfg = self.config
        train = ranking_window(panel, as_of, cfg)
        if self.raw_outcomes is None:
            if self.raw_returns is None:
                raise ValueError("Validated raw returns are required for portfolio-based tuning")
            self.raw_outcomes = raw_training_outcomes(panel, self.raw_returns)
        train = attach_training_returns(train, self.raw_outcomes)
        path = self.output_dir / f"as_of={as_of.date()}"
        path.mkdir(parents=True, exist_ok=True)
        identity = dict(implementation_version=5, config=asdict(cfg), window_weeks=WINDOW_WEEKS,
                        as_of=str(as_of.date()), training_fingerprint=frame_fingerprint(
                            train[["ticker", "week_date", "prev_week_date", TARGET,
                                   *[f"{f}_train" for f in cfg.features]]]),
                        raw_return_fingerprint=frame_fingerprint(
                            train[["ticker", "week_date", "raw_active_return"]]))
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        manifest = path / "metadata.json"
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            if metadata["signature"] != signature:
                raise ValueError(f"Existing model/config/data differ at {path}; use a new --output-dir")
            model_path = path / "lightgbm.txt"
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
                raise ValueError(f"Saved model hash mismatch at {path}")
            if hashlib.sha256((path / "lightgbm.pkl").read_bytes()).hexdigest() != metadata["pickle_sha256"]:
                raise ValueError(f"Saved pickle hash mismatch at {path}")
            return joblib.load(path / "lightgbm.pkl"), metadata

        started = time.monotonic()
        splits = list(date_cv_splits(train.week_date, cfg.cv_splits, cfg.cv_gap_weeks))
        validation_indices = np.unique(np.concatenate([va for _, va in splits]))
        market_k = max(1, math.ceil(0.20 * train.iloc[validation_indices].groupby("week_date").size().median()))
        folds, fold_metadata, evaluators = [], [], []
        for tr, va in splits:
            t, v = train.iloc[tr], train.iloc[va]
            if t.week_date.max() > v.prev_week_date.min():
                raise ValueError("CV training label is unavailable at validation feature time")
            folds.append((ranking_inputs(t, cfg.features), ranking_inputs(v, cfg.features)))
            evaluators.append(ValidationEvaluator(folds[-1][1][3]))
            fold_metadata.append(dict(train_start=str(t.week_date.min().date()),
                                      train_end=str(t.week_date.max().date()),
                                      validation_start=str(v.week_date.min().date()),
                                      validation_end=str(v.week_date.max().date()),
                                      training_rows=len(t), validation_rows=len(v)))

        def evaluate(params, stage, patience):
            metrics, iterations, stitched = [], [], []
            for ((X, y, groups, _), (VX, vy, vg, valid)), evaluator in zip(folds, evaluators):
                model = lgb.LGBMRanker(**params)
                # LightGBM 4.7 replaces eval_set; retain support for earlier 4.x.
                validation = ({"eval_X": VX, "eval_y": vy}
                              if "eval_X" in inspect.signature(model.fit).parameters
                              else {"eval_set": [(VX, vy)]})
                def aligned_metric(y_true, y_pred, weight=None, group=None):
                    return "net_q1_q5_ir_10bps", evaluator.portfolio_ir(y_pred), True
                model.fit(X, y, group=groups, **validation, eval_group=[vg],
                          eval_metric=aligned_metric, callbacks=[lgb.early_stopping(
                              patience, first_metric_only=True, verbose=False)])
                evaluated = [name for metrics_for_set in model.evals_result_.values()
                             for name in metrics_for_set]
                if evaluated != ["net_q1_q5_ir_10bps"]:
                    raise ValueError(f"Early stopping metric changed unexpectedly: {evaluated}")
                predicted = model.predict(VX)
                fold_metrics = evaluator(predicted)
                metrics.append(fold_metrics)
                iterations.append(int(model.best_iteration_ or model.n_estimators_))
                stitched.append(valid.assign(ranking_score=predicted))
            aggregate = {name: finite_mean([m[name] for m in metrics]) for name in METRICS}
            aggregate["n_groups"] = sum(m["n_groups"] for m in metrics)
            validated = pd.concat(stitched, ignore_index=True).sort_values(["week_date", "ticker"])
            portfolio = portfolio_summary(portfolio_weekly(validated, validated.ranking_score.to_numpy()))
            complexity = (params["num_leaves"], params["max_depth"],
                          -params["min_child_samples"], -params["reg_lambda"])
            result = dict(stage=stage, params=params, aggregate_metrics=aggregate,
                          fold_metrics=metrics, best_iterations=iterations,
                          portfolio=portfolio, complexity=complexity,
                          median_best_iteration=int(round(float(np.median(iterations)))),
                          max_best_iteration=max(iterations), early_stopping_rounds=patience)
            LOGGER.info("%s %s LS-NDCG=%.6f median trees=%s", as_of.date(), stage,
                        aggregate["mean_ls_ndcg"], result["median_best_iteration"])
            return result

        winner, results, triggered = staged_search(
            evaluate, stage1_candidates(len(train), len(cfg.features), cfg), market_k, cfg)
        final_params = dict(winner["params"], n_estimators=max(1, winner["median_best_iteration"]))
        X, y, groups, _ = ranking_inputs(train, cfg.features)
        seeds = tuple(cfg.seed + offset for offset in (0, 31, 65))
        seed_models = [lgb.LGBMRanker(**dict(final_params, random_state=seed)).fit(X, y, group=groups).booster_
                       for seed in seeds]
        final = SeedRankEnsemble(seed_models, cfg.features, "lightgbm")
        temporary = path / "lightgbm.txt.tmp"
        seed_models[0].save_model(str(temporary))
        temporary.replace(path / "lightgbm.txt")
        joblib.dump(final, path / "lightgbm.pkl.tmp")
        (path / "lightgbm.pkl.tmp").replace(path / "lightgbm.pkl")
        write_json(path / "search_results.json", {"candidates": results})
        pd.DataFrame({"feature": cfg.features,
                      "gain": np.mean([m.feature_importance("gain") for m in seed_models], axis=0),
                      "split": np.mean([m.feature_importance("split") for m in seed_models], axis=0)}).to_parquet(
                          path / "feature_importance.parquet", index=False)
        metadata = dict(identity, signature=signature, training_rows=len(train),
                        first_training_week=str(train.week_date.min().date()),
                        last_training_week=str(train.week_date.max().date()),
                        training_weeks=int(train.week_date.nunique()), market_eval_at=market_k,
                        best_params=final_params, best_search_params=winner["params"],
                        best_search_stage=winner["stage"], best_cv_metrics=winner["aggregate_metrics"],
                        best_cv_portfolio=winner["portfolio"], ensemble_seeds=seeds,
                        selection_metric="10bps net Q1-Q5 IR then cumulative return; simpler model within 0.05 IR and 0.01 cumulative return",
                        early_stopping_metric="net_q1_q5_ir_10bps_only",
                        final_n_estimators=final_params["n_estimators"], actual_trees=seed_models[0].num_trees(),
                        candidate_count=len(results), stage3_triggered=triggered, cv_folds=fold_metadata,
                        model_sha256=hashlib.sha256((path / "lightgbm.txt").read_bytes()).hexdigest(),
                        pickle_sha256=hashlib.sha256((path / "lightgbm.pkl").read_bytes()).hexdigest(),
                        fit_seconds=time.monotonic() - started,
                        versions={p: importlib.metadata.version(p) for p in ["lightgbm", "scikit-learn", "pandas", "numpy"]})
        write_json(manifest, metadata)  # Completion marker is written last.
        return joblib.load(path / "lightgbm.pkl"), metadata

    def run(self, raw_panel, *, model_start=None, model_end=None, max_models=None,
            source_metadata=None, raw_returns=None):
        cfg = self.config
        panel = prepare_panel(raw_panel, cfg.panel_config())
        raw = raw_returns if raw_returns is not None else self.raw_returns
        if raw is None:
            raise ValueError("Pass raw_returns for portfolio-based tuning")
        self.raw_outcomes = raw_training_outcomes(panel, raw)
        weeks = pd.DatetimeIndex(panel.week_date.unique())
        # A full window also needs the preceding feature snapshot in the source.
        dates = weeks[WINDOW_WEEKS:]
        if model_start:
            dates = dates[dates >= pd.Timestamp(model_start)]
        if model_end:
            dates = dates[dates <= pd.Timestamp(model_end)]
        if max_models is not None:
            if max_models < 1:
                raise ValueError("max_models must be positive")
            dates = dates[:max_models]
        if not len(dates):
            raise ValueError("No eligible model dates after the 260-week warm-up")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        next_map = pd.Series(weeks[1:].to_numpy(), index=weeks[:-1])
        predictions, models = [], []
        for number, (as_of, fitted) in enumerate(self.fitted_weeks(panel, dates), 1):
            LOGGER.info("Fitting 260-week model %s/%s: %s", number, len(dates), as_of.date())
            model, metadata = fitted
            snapshot = panel.loc[panel.week_date.eq(as_of)].copy()
            scored = snapshot[["ticker", "sector", "week_date", FORWARD_TARGET]].copy()
            scored["next_week_date"] = scored.week_date.map(next_map)
            scored["model_as_of"] = as_of
            scored["ranking_score"] = np.nan
            valid = snapshot[list(cfg.features)].notna().all(axis=1)
            if valid.any():
                scored.loc[valid, "ranking_score"] = model.predict(
                    snapshot.loc[valid, list(cfg.features)].astype(np.float32), num_threads=cfg.threads)
            scored["quintile"] = pd.Series(pd.NA, index=scored.index, dtype="Int64")
            scored["long_selected"], scored["short_selected"] = False, False
            ordered = scored.loc[valid].sort_values(["ranking_score", "ticker"], ascending=[False, True]).index
            if len(ordered):
                scored.loc[ordered, "quintile"] = np.arange(len(ordered)) * 5 // len(ordered) + 1
                if len(ordered) >= 2:
                    k = math.ceil(0.20 * len(ordered))
                    scored.loc[ordered[:k], "long_selected"] = True
                    scored.loc[ordered[-k:], "short_selected"] = True
            scored.to_parquet(self.output_dir / f"as_of={as_of.date()}" / "predictions.parquet", index=False)
            predictions.append(scored)
            models.append({k: metadata[k] for k in ["as_of", "signature", "best_search_stage", "candidate_count",
                                                    "best_cv_metrics", "final_n_estimators", "training_rows"]})
        combined = pd.concat(predictions, ignore_index=True)
        weekly, metrics = evaluate_scores(combined, target_col=FORWARD_TARGET)
        combined.to_parquet(self.output_dir / "predictions.parquet", index=False)
        weekly.to_parquet(self.output_dir / "weekly_metrics.parquet", index=False)
        summary = dict(config=asdict(cfg), window_weeks=WINDOW_WEEKS, model_count=len(dates),
                       first_model_date=str(dates[0].date()), last_model_date=str(dates[-1].date()),
                       source=source_metadata or {}, prediction_rows=len(combined),
                       missing_predictions=int(combined.ranking_score.isna().sum()),
                       missing_forward_labels=int(combined[FORWARD_TARGET].isna().sum()),
                       metrics=metrics, models=models,
                       timing="Fit *_train -> active_return_train through as-of close; score current features -> active_return_fwd",
                       target_units="Sector-ranked weekly z-scores, not percentage returns",
                       quintile_convention="Q1 highest score; Q5 lowest. Long/short selections each use ceil(20% of scored universe).")
        write_json(self.output_dir / "run_summary.json", summary)
        return summary
