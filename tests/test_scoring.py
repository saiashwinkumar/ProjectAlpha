import numpy as np
import pandas as pd
import pytest

from cross_sectional_panel import build_cross_sectional_panel
from scoring import assign_quintiles, quintile_performance, raw_forward_returns


def test_quintiles_use_predictions_only_with_stable_tie_breaking():
    scores = pd.DataFrame({"model": "xgboost", "week_date": pd.Timestamp("2022-01-05"),
                           "ticker": [f"T{i:02}" for i in range(11)], "sector": "Tech",
                           "prediction": [1.0] * 11, "active_return_fwd": np.arange(11)})
    first = assign_quintiles(scores)
    changed = scores.sample(frac=1, random_state=7)
    changed["active_return_fwd"] = np.nan
    second = assign_quintiles(changed)
    pd.testing.assert_frame_equal(first[["ticker", "quintile"]], second[["ticker", "quintile"]])
    counts = first.quintile.value_counts()
    assert counts.max() - counts.min() <= 1
    assert first.iloc[0].quintile == 1 and first.iloc[-1].quintile == 5


def test_raw_forward_returns_exact_join_and_sector_benchmark():
    dates = pd.date_range("2022-01-05", periods=3, freq="W-WED")
    raw = pd.DataFrame({"Ticker": list("ABCDE") * 3, "sector": ["Tech"] * 15,
                        "week_date": dates.repeat(5), "return_1w": np.arange(15) / 100})
    panel = build_cross_sectional_panel(raw, feature_directions={"return_1w": 1})
    panel = panel.rename(columns={"Ticker": "ticker"})
    result = raw_forward_returns(panel, raw)
    first_a = result.loc[result.ticker.eq("A") & result.week_date.eq(dates[0])].iloc[0]
    assert first_a.stock_return_fwd == pytest.approx(0.05)
    assert first_a.sector_return_fwd == pytest.approx(0.07)
    assert first_a.raw_active_return_fwd == pytest.approx(-0.02)
    assert result.loc[result.week_date.eq(dates[-1]), "raw_active_return_fwd"].isna().all()
    bad = raw.copy()
    bad.loc[bad.Ticker.eq("A"), "return_1w"] = 5
    with pytest.raises(ValueError, match="reconstruct"):
        raw_forward_returns(panel, bad)


def test_cumulative_formulas_and_missing_holdings_are_not_reweighted():
    rows = []
    for week, portfolio, benchmark in [("2022-01-05", 0.10, 0.02), ("2022-01-12", -0.05, 0.01)]:
        for i in range(10):
            rows.append({"model": "xgboost", "week_date": pd.Timestamp(week),
                         "next_week_date": pd.Timestamp(week) + pd.Timedelta(days=7),
                         "ticker": f"T{i}", "quintile": i // 2 + 1, "prediction": i,
                         "active_return_fwd": i / 10, "stock_return_fwd": portfolio,
                         "sector_return_fwd": benchmark, "raw_active_return_fwd": portfolio-benchmark})
    members = pd.DataFrame(rows)
    weekly = quintile_performance(members)
    end = weekly.loc[weekly.week_date.eq("2022-01-12")]
    assert np.allclose(end.cumulative_active_return, 0.02)
    assert np.allclose(end.cumulative_excess_return, 1.10 * 0.95 - 1.02 * 1.01)
    assert np.allclose(end.compounded_active_spread, 1.08 * 0.94 - 1)
    members.loc[10, ["stock_return_fwd", "raw_active_return_fwd"]] = np.nan
    incomplete = quintile_performance(members)
    end = incomplete.loc[incomplete.week_date.eq("2022-01-12")]
    assert end.active_return.isna().all()
    assert end.cumulative_active_return.isna().all()
    assert end.holdings.sum() == 10


def test_documented_cash_exit_fills_only_raw_performance():
    dates = pd.date_range("2022-01-05", periods=2, freq="W-WED")
    raw = pd.DataFrame({"Ticker": list("ABCDE") * 2, "sector": "Tech",
                        "week_date": dates.repeat(5), "return_1w": np.arange(10) / 100,
                        "adj_close": 100.0})
    raw = raw.loc[~(raw.Ticker.eq("A") & raw.week_date.eq(dates[1]))]
    panel = build_cross_sectional_panel(raw, feature_directions={"return_1w": 1}).rename(columns={"Ticker": "ticker"})
    event = {"ticker": "A", "week_date": "2022-01-05", "next_week_date": "2022-01-12",
             "event_date": "2022-01-12", "event_type": "cash_acquisition", "cash_per_share": 110.0,
             "start_close": 100.0, "source_url": "https://example.com/test-event"}
    result = raw_forward_returns(panel, raw, [event])
    a = result.loc[result.ticker.eq("A")].iloc[0]
    assert a.stock_return_fwd == pytest.approx(0.10)
    assert a.sector_return_fwd == pytest.approx(0.075)
    assert a.raw_active_return_fwd == pytest.approx(0.025)
    assert a.return_source == "cash_acquisition"
    assert panel.loc[panel.ticker.eq("A"), "active_return_fwd"].isna().all()
