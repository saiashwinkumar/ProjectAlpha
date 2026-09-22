import numpy as np
import pandas as pd
import pytest

from portfolio_training import (SeedRankEnsemble, better_portfolio, portfolio_summary,
                                portfolio_weekly, symmetric_ndcg)


class DummyModel:
    def __init__(self, values):
        self.values = np.asarray(values)

    def predict(self, X):
        return self.values


def test_seed_ensemble_averages_within_week_percentiles():
    X = pd.DataFrame({"factor": [1., 2., 3., 4., 5.]})
    ensemble = SeedRankEnsemble([DummyModel([5, 4, 3, 2, 1]),
                                 DummyModel([1, 2, 3, 4, 5]),
                                 DummyModel([5, 4, 3, 2, 1])], ["factor"], "xgboost")
    np.testing.assert_allclose(ensemble.predict(X), [11/15, 10/15, 9/15, 8/15, 7/15])
    with pytest.raises(ValueError, match="feature order"):
        ensemble.predict(X.rename(columns={"factor": "other"}))


def test_validation_portfolio_uses_raw_returns_and_stable_quintiles():
    weeks = pd.date_range("2022-01-05", periods=3, freq="W-WED")
    frame = pd.DataFrame({"ticker": list("ABCDE") * 3, "week_date": weeks.repeat(5),
                          "active_return_train": list(range(5)) * 3,
                          "raw_active_return": [0, 0, 0, 0, .02] * 3})
    weekly = portfolio_weekly(frame, np.tile(np.arange(5), 3))
    assert weekly.gross_return.tolist() == pytest.approx([.02] * 3)
    assert np.isnan(weekly.turnover.iloc[0])
    assert weekly.turnover.iloc[1:].eq(0).all()
    gross = portfolio_summary(weekly)["cost_0bps"]
    net = portfolio_summary(weekly)["cost_10bps"]
    assert gross["cumulative_return"] == pytest.approx(1.02**3 - 1)
    assert net["cumulative_return"] == pytest.approx(1.018 * 1.02**2 - 1)
    assert symmetric_ndcg(np.tile(np.arange(5), 3), np.tile(np.arange(5), 3), [5, 5, 5]) == 1
    frame["active_return_train"] = 0
    assert portfolio_weekly(frame, np.tile(np.arange(5), 3)).gross_return.tolist() == pytest.approx([.02] * 3)


def test_portfolio_selection_prefers_ir_then_simple_ties():
    def candidate(ir, cumulative, complexity):
        return {"portfolio": {"cost_10bps": {"information_ratio": ir,
                                               "cumulative_return": cumulative}},
                "complexity": complexity}
    simple = candidate(.50, .10, (2, -50, -10, 20))
    complex_model = candidate(.52, .105, (3, -10, -1, 80))
    assert not better_portfolio(complex_model, simple, complex_model["complexity"])
    stronger = candidate(.60, .09, (3, -10, -1, 80))
    assert better_portfolio(stronger, simple, stronger["complexity"])
