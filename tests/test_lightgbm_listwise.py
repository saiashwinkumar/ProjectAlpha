import json
from dataclasses import replace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from cross_sectional_panel import build_cross_sectional_panel
from lightgbm_listwise_training import (
    RankingConfig, RollingListwiseTrainer, candidate_key, evaluate_scores,
    ranking_inputs, ranking_window, relevance_labels, stage1_candidates, staged_search, ValidationEvaluator,
)
from training import prepare_panel
from portfolio_training import portfolio_summary, portfolio_weekly


@pytest.fixture
def panel():
    rng = np.random.default_rng(12)
    weeks = pd.date_range("2017-01-04", periods=262, freq="W-WED")
    n = 20
    raw = pd.DataFrame({"Ticker": np.tile([f"T{i:02d}" for i in range(n)], len(weeks)),
                        "sector": "Tech", "week_date": weeks.repeat(n),
                        "return_1w": rng.normal(size=n * len(weeks)),
                        "volatility_20d": rng.uniform(size=n * len(weeks))})
    return build_cross_sectional_panel(raw, feature_directions={"return_1w": 1, "volatility_20d": -1})


@pytest.fixture
def cfg():
    return RankingConfig(features=("return_1w", "volatility_20d"), threads=1,
                         n_estimators=3, early_stopping_rounds=2,
                         stage3_n_estimators=5, stage3_early_stopping_rounds=2)


@pytest.fixture
def raw_returns():
    rng = np.random.default_rng(12)
    weeks = pd.date_range("2017-01-04", periods=262, freq="W-WED")
    return pd.DataFrame({"Ticker": np.tile([f"T{i:02d}" for i in range(20)], len(weeks)),
                         "week_date": weeks.repeat(20),
                         "return_1w": rng.normal(size=20 * len(weeks)) / 100})


def test_exact_window_and_group_alignment(panel, cfg):
    prepared = prepare_panel(panel, cfg.panel_config())
    as_of = prepared.week_date.max()
    window = ranking_window(prepared, as_of, cfg)
    assert window.week_date.nunique() == 260
    assert window.week_date.min() == as_of - pd.Timedelta(weeks=259)
    changed = prepared.copy()
    changed["active_return_fwd"] = 10000
    pd.testing.assert_frame_equal(window.drop(columns="active_return_fwd"),
                                  ranking_window(changed, as_of, cfg).drop(columns="active_return_fwd"))
    X, labels, groups, work = ranking_inputs(window.sample(frac=1, random_state=3), cfg.features)
    assert groups.sum() == len(X) == len(labels)
    for start in np.cumsum(np.r_[0, groups[:-1]]):
        assert work.iloc[start:start + 20].week_date.nunique() == 1
    assert list(X.columns) == list(cfg.features)
    assert set(labels) == set(range(5))
    with pytest.raises(ValueError, match="260-week"):
        ranking_window(prepared, prepared.week_date.unique()[100], cfg)


def test_metrics_direction_ties_turnover_and_missing_outcomes():
    dates = pd.date_range("2020-01-01", periods=3, freq="W-WED")
    frame = pd.DataFrame({"ticker": list("ABCDE") * 3, "week_date": dates.repeat(5),
                          "active_return_train": list(range(5)) * 3, "ranking_score": list(range(5)) * 3})
    weekly, metrics = evaluate_scores(frame)
    assert metrics["mean_ls_ndcg"] == 1
    assert metrics["mean_q1_q5_spread"] == 4
    assert weekly.turnover.iloc[1] == 0
    assert relevance_labels([1, 2, 3, 4, 5]).tolist() == [0, 1, 2, 3, 4]
    frame["ranking_score"] *= -1
    assert evaluate_scores(frame)[1]["mean_ls_ndcg"] == 0
    frame["ranking_score"] = 0.0
    tied = evaluate_scores(frame)[0]
    shuffled = evaluate_scores(frame.sample(frac=1, random_state=4))[0]
    pd.testing.assert_frame_equal(tied, shuffled)
    frame.loc[5, "active_return_train"] = np.nan
    weekly, _ = evaluate_scores(frame)
    assert pd.isna(weekly.ls_ndcg.iloc[1])
    assert pd.isna(weekly.q1_q5_spread.iloc[1])
    assert weekly.n_scored.iloc[1] == 5


@pytest.mark.parametrize("trigger,market_k,expected", [(False, 10, 9), (True, 10, 10), (False, 27, 7)])
def test_stages_and_conditional_budget(trigger, market_k, expected):
    cfg = RankingConfig()
    candidates = stage1_candidates(100_000, 12, cfg)
    assert len(candidates) == 7
    assert {c["min_child_samples"] for c in candidates} == {300, 1000}
    assert candidates[-1]["num_leaves"] == 31
    def evaluate(params, stage, patience):
        return dict(params=params, stage=stage, median_best_iteration=550 if trigger else 10,
                    max_best_iteration=550 if trigger else 10,
                    aggregate_metrics=dict(mean_ls_ndcg=0.5, mean_q1_q5_spread=0,
                                           q1_q5_information_ratio=None, mean_turnover=None))
    winner, results, triggered = staged_search(evaluate, candidates, market_k, cfg)
    assert len(results) == expected
    assert triggered == trigger
    assert winner["stage"] == "stage1_01"
    if trigger:
        assert results[-1]["params"]["n_estimators"] == 1000
    assert np.isfinite(candidate_key(winner)[0])


def test_fit_reload_resume_conflicts_and_forward_independence(panel, cfg, raw_returns, tmp_path):
    trainer = RollingListwiseTrainer(cfg, tmp_path, raw_returns=raw_returns)
    result = trainer.run(panel)
    assert result["model_count"] == 2
    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions.ranking_score.notna().all()
    last = predictions.week_date.max()
    assert predictions.loc[predictions.week_date.eq(last), "active_return_fwd"].isna().all()
    assert predictions.loc[predictions.week_date.eq(last), "quintile"].notna().all()
    folder = tmp_path / f"as_of={last.date()}"
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["training_weeks"] == 260
    assert metadata["candidate_count"] in [9, 10]
    assert metadata["final_n_estimators"] <= cfg.stage3_n_estimators
    import joblib
    model = joblib.load(folder / "lightgbm.pkl")
    prepared = prepare_panel(panel, cfg.panel_config())
    current = prepared.loc[prepared.week_date.eq(last), list(cfg.features)].astype(np.float32)
    np.testing.assert_allclose(model.predict(current), predictions.loc[predictions.week_date.eq(last), "ranking_score"])
    before = (folder / "metadata.json").stat().st_mtime_ns
    changed = panel.copy()
    changed["active_return_fwd"] = 100
    trainer.run(changed)
    assert (folder / "metadata.json").stat().st_mtime_ns == before
    np.testing.assert_allclose(pd.read_parquet(tmp_path / "predictions.parquet").ranking_score,
                               predictions.ranking_score)
    with pytest.raises(ValueError, match="differ"):
        RollingListwiseTrainer(replace(cfg, seed=99), tmp_path, raw_returns=raw_returns).run(panel)
    (folder / "lightgbm.txt").write_text("corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        trainer.fit_week(prepared, last)


def test_cached_validation_equivalent_for_changing_universes_and_ties():
    rng = np.random.default_rng(72)
    groups = []
    for i, date in enumerate(pd.date_range("2022-01-05", periods=12, freq="W-WED")):
        n = 43 + i
        groups.append(pd.DataFrame({"ticker": [f"T{x:03d}" for x in range(n)], "week_date": date,
                                    "active_return_train": rng.normal(size=n),
                                    "ranking_score": rng.integers(-3, 4, size=n)}))
    frame = pd.concat(groups, ignore_index=True)
    frame["raw_active_return"] = frame.active_return_train / 100
    cached = ValidationEvaluator(frame)(frame.ranking_score.to_numpy())
    reference = evaluate_scores(frame)[1]
    for name in cached:
        assert cached[name] == pytest.approx(reference[name], abs=1e-12)
    net_ir = portfolio_summary(portfolio_weekly(frame, frame.ranking_score.to_numpy()))[
        "cost_10bps"]["information_ratio"]
    assert ValidationEvaluator(frame).portfolio_ir(frame.ranking_score.to_numpy()) == pytest.approx(net_ir)


def test_all_three_use_exact_windows_and_independent_pickles(panel, cfg, raw_returns, tmp_path, monkeypatch):
    from continuous_backtest import score_saved_models, score_snapshot, run_backtest
    from training import GRIDS, RollingPanelTrainer, TrainingConfig, training_window
    monkeypatch.setitem(GRIDS, "compact", {"n_estimators": [3], "max_depth": [2]})
    regression_cfg = TrainingConfig(window_weeks=260, features=cfg.features, threads=1)
    RollingPanelTrainer(regression_cfg, tmp_path / "regression", raw_returns=raw_returns).run(panel)
    RollingListwiseTrainer(cfg, tmp_path / "lightgbm", workers=2, raw_returns=raw_returns).run(panel)
    first = pd.to_datetime(panel.week_date).drop_duplicates().iloc[260]
    prepared, scores, audit = score_saved_models(panel, tmp_path, start=str(first.date()))
    assert len(audit) == 6
    assert audit.training_weeks.eq(260).all()
    assert scores.groupby("model").week_date.nunique().eq(2).all()
    assert scores.predicted_rank.notna().all()
    assert "active_return_fwd" not in scores
    for date in scores.week_date.unique():
        pd.testing.assert_frame_equal(training_window(prepared, pd.Timestamp(date), regression_cfg),
                                      ranking_window(prepared, pd.Timestamp(date), cfg))
    snapshot = prepared.loc[prepared.week_date.eq(first)].copy()
    changed = snapshot.copy()
    changed[[f"{f}_train" for f in cfg.features]] = 999
    changed["active_return_fwd"] = -999
    for model, filename, folder in [("lightgbm", "lightgbm.pkl", "lightgbm"),
                                     ("xgboost", "xgboost.pkl", "regression"),
                                     ("linear_regression", "linear_regression.pkl", "regression")]:
        path = tmp_path / folder / f"as_of={first.date()}" / filename
        np.testing.assert_array_equal(score_snapshot(snapshot, path, cfg.features, model),
                                      score_snapshot(changed, path, cfg.features, model))
    # Reconstruct the fixture's original raw source for complete P&L integration.
    rng = np.random.default_rng(12)
    weeks = pd.date_range("2017-01-04", periods=262, freq="W-WED")
    raw = pd.DataFrame({"Ticker": np.tile([f"T{i:02d}" for i in range(20)], len(weeks)),
                        "sector": "Tech", "week_date": weeks.repeat(20),
                        "return_1w": rng.normal(size=20 * len(weeks)) / 100})
    summary = run_backtest(panel, raw, tmp_path, start=str(first.date()), events=[])
    assert summary["pickle_count"] == 6
    assert set(summary["completed_return_weeks_per_model"].values()) == {1}
    assert set(summary["missing_forward_labels_per_model"].values()) == {20}
    membership = pd.read_parquet(tmp_path / "backtest/stock_scores_and_quintiles.parquet")
    assert membership.loc[membership.week_date.eq(weeks[-1]), "prediction"].notna().all()
    path.write_bytes(b"invalid model")
    with pytest.raises(ValueError, match="pickle hash mismatch"):
        score_saved_models(panel, tmp_path, start=str(first.date()))
