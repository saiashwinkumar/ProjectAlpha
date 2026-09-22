import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import ndcg_score

from comparative_evaluation import comparative_metrics, cost_sensitivity
from scoring import assign_quintiles


def sample_members(n=10, weeks=2):
    rows = []
    for model in ["linear_regression", "xgboost", "lightgbm"]:
        for step, date in enumerate(pd.date_range("2022-01-05", periods=weeks, freq="W-WED")):
            for i in range(n):
                rows.append(dict(model=model, week_date=date, ticker=f"T{i:03d}", prediction=float(i),
                                 active_return_fwd=float(i), stock_return_fwd=(step+1)*i/1000+0.01,
                                 sector_return_fwd=0.01, raw_active_return_fwd=(step+1)*i/1000))
    return assign_quintiles(pd.DataFrame(rows), highest_first=True)


def test_rank_return_ir_and_unchanged_holdings():
    comparison, weekly, details = comparative_metrics(sample_members())
    assert np.allclose(comparison.rank_ic, 1)
    assert comparison.ndcg_at_50.eq(1).all()
    assert np.allclose(comparison.q1_q5_annualized_return, 52 * np.mean([0.008, 0.016]))
    expected_ir = np.sqrt(52) * np.mean([0.008, 0.016]) / np.std([0.008, 0.016], ddof=1)
    assert np.allclose(comparison.q1_q5_annualized_ir, expected_ir)
    assert comparison.mean_weekly_turnover.eq(0).all()
    assert comparison.turnover_observations.eq(1).all()
    assert details["ndcg_k_min"] == 10
    costs = cost_sensitivity(weekly)
    gross = costs.loc[costs.cost_bps_per_dollar_traded.eq(0)]
    net = costs.loc[costs.cost_bps_per_dollar_traded.eq(10)]
    assert np.allclose(gross.annualized_return, 52 * .012)
    # One initial 100% long + 100% short entry costs 2 * 10bps; no rebalance trade.
    assert np.allclose(net.annualized_return, 52 * (.012 - .001))
    assert np.all(net.cumulative_return.to_numpy() < gross.cumulative_return.to_numpy())
    assert np.all(gross.max_drawdown <= 0)


def test_fixed_50_matches_independent_ndcg_implementation():
    frame = sample_members(n=100, weeks=1)
    frame.prediction *= -1
    frame = assign_quintiles(frame, highest_first=True)
    comparison, _, details = comparative_metrics(frame)
    # Independent closed-form labels for exactly 100 monotonically ordered targets.
    labels = np.repeat(np.arange(5), 20)
    expected = ndcg_score(labels.reshape(1, -1), -np.arange(100).reshape(1, -1), k=50, ignore_ties=True)
    at_20 = ndcg_score(labels.reshape(1, -1), -np.arange(100).reshape(1, -1), k=20, ignore_ties=True)
    assert np.allclose(comparison.ndcg_at_50, expected)
    assert expected != at_20
    assert details["ndcg_k_min"] == details["ndcg_k_max"] == 50


def test_switched_sides_and_common_coverage_without_return_filtering():
    frame = sample_members(weeks=3)
    second = frame.week_date.eq("2022-01-12")
    frame.loc[second, "prediction"] *= -1
    frame = assign_quintiles(frame, highest_first=True)
    frame.loc[frame.model.eq("lightgbm") & second & frame.ticker.eq("T000"), "active_return_fwd"] = np.nan
    comparison, weekly, _ = comparative_metrics(frame)
    assert comparison.ranking_weeks.eq(2).all()
    assert comparison.return_weeks.eq(3).all()
    assert comparison.mean_weekly_turnover.eq(2).all()
    assert weekly.loc[weekly.week_date.eq("2022-01-12"), "common_ranking_week"].eq(False).all()
    assert weekly.loc[weekly.week_date.eq("2022-01-12"), "q1_q5_active_return"].notna().all()
    wrong = frame.copy()
    wrong.quintile = 6 - wrong.quintile
    with pytest.raises(ValueError, match="Q1-high"):
        comparative_metrics(wrong)


def test_no_turnover_across_calendar_gap_and_missing_raw_returns():
    frame = sample_members(weeks=3)
    frame = frame.loc[~frame.week_date.eq("2022-01-12")].copy()
    mask = frame.model.eq("lightgbm") & frame.week_date.eq("2022-01-19") & frame.ticker.eq("T005")
    frame.loc[mask, "raw_active_return_fwd"] = np.nan
    comparison, _, _ = comparative_metrics(frame)
    assert comparison.turnover_observations.eq(0).all()
    assert comparison.mean_weekly_turnover.isna().all()
    assert comparison.return_weeks.eq(1).all()
    assert comparison.q1_q5_annualized_ir.isna().all()
