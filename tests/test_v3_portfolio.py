import numpy as np
import pandas as pd
import pytest

from v3_portfolio import (Q1NetIRStopMetric, evaluate_portfolio,
                          percentile_ensemble, portfolio_metrics, target_weights)
from v3_incumbent_search import _folds


def test_chronological_folds_have_disjoint_fit_and_evaluation_weeks():
    weeks = pd.date_range("2017-01-04", periods=260, freq="W-WED")
    frame = pd.DataFrame({"week_date": weeks.repeat(2)})
    folds = list(_folds(frame))
    assert len(folds) == 4
    for fit, evaluate in folds:
        assert (evaluate.week_date.min()-fit.week_date.max()).days == 14
        assert len(evaluate.week_date.unique()) == 52
        assert fit.week_date.max() < evaluate.week_date.min()


def test_percentile_ensemble_uses_week_positions_even_with_nonconsecutive_index():
    scores = np.array([[3, 1, 2], [2, 2, 1], [1, 3, 3],
                       [1, 3, 3], [2, 2, 1], [3, 1, 2]])
    dates = pd.Series(pd.to_datetime(["2022-01-05"]*3+["2022-01-12"]*3),
                      index=[50, 51, 52, 90, 91, 92])
    result = percentile_ensemble(scores, dates)
    np.testing.assert_allclose(result[:3], [2/3, 5/9, 7/9])
    np.testing.assert_allclose(result[3:], [7/9, 5/9, 2/3])


def test_buffer_keeps_incumbent_and_charges_actual_dollar_trading():
    tickers = np.array(list("ABCDEFGHIJ"))
    first = np.arange(10, 0, -1, dtype=float)
    second = np.array([10, 8, 9, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    previous, _ = target_weights(tickers, first)
    exact, _ = target_weights(tickers, second, previous, buffer=.2)
    buffered, _ = target_weights(tickers, second, previous, buffer=.3)
    assert set(previous) == {"A", "B"}
    assert set(exact) == {"A", "C"}
    assert set(buffered) == {"A", "B"}
    dates = pd.date_range("2022-01-05", periods=2, freq="W-WED")
    frame = pd.DataFrame({"ticker": list(tickers)*2,
                          "week_date": dates.repeat(10),
                          "score": np.r_[first, second],
                          "raw_active_return": np.r_[np.zeros(10), np.ones(10)*.01],
                          "stock_return": np.r_[np.zeros(10), np.ones(10)*.02],
                          "active_return_train": np.r_[first, second]})
    exact_weekly = evaluate_portfolio(frame, stock_col="stock_return", buffer=.2)
    buffered_weekly = evaluate_portfolio(frame, stock_col="stock_return", buffer=.3)
    assert exact_weekly.transaction_cost.tolist() == pytest.approx([.0005, .0005])
    assert buffered_weekly.transaction_cost.tolist() == pytest.approx([.0005, 0])
    assert buffered_weekly.turnover.tolist() == pytest.approx([.5, 0])
    metrics = portfolio_metrics(buffered_weekly)
    assert metrics["annualized_stock_return"] == pytest.approx(.52)
    assert metrics["annualized_transaction_cost"] == pytest.approx(.013)


def test_early_stop_metric_matches_portfolio_net_ir():
    weeks = pd.date_range("2022-01-05", periods=4, freq="W-WED")
    returns = np.array([[.02, 0, 0, 0, -.02], [.01, 0, 0, 0, -.01],
                        [-.005, 0, 0, 0, .005], [.03, 0, 0, 0, -.03]])
    frame = pd.DataFrame({"ticker": list("ABCDE")*4,
                          "week_date": weeks.repeat(5),
                          "raw_active_return": returns.ravel(),
                          "active_return_train": returns.ravel(),
                          "score": np.tile([5, 4, 3, 2, 1], 4)})
    stop = Q1NetIRStopMetric(frame)
    actual = portfolio_metrics(evaluate_portfolio(frame))["active_ir"]
    assert stop.ir(frame.score.to_numpy()) == pytest.approx(actual)
