import json
import joblib

import numpy as np
import pandas as pd
import pytest

from cross_sectional_panel import build_cross_sectional_panel
from training import (
    GRIDS, RollingPanelTrainer, TrainingConfig, date_cv_splits,
    frame_fingerprint, prepare_panel, regression_metrics, training_window,
)
from training_data import panel_query


@pytest.fixture
def panel():
    rng = np.random.default_rng(17)
    dates = pd.date_range("2017-01-04", "2022-02-02", freq="W-WED")
    raw = pd.DataFrame({
        "Ticker": np.tile(list("ABCDEFGH"), len(dates)),
        "sector": np.tile(["Tech"] * 4 + ["Health"] * 4, len(dates)),
        "week_date": dates.repeat(8),
        "return_1w": rng.normal(size=len(dates) * 8),
        "volatility_20d": rng.uniform(size=len(dates) * 8),
    })
    return build_cross_sectional_panel(
        raw, feature_directions={"return_1w": 1, "volatility_20d": -1}
    )


@pytest.fixture
def config():
    return TrainingConfig(features=("return_1w", "volatility_20d"), threads=1)


@pytest.fixture
def raw_returns():
    rng = np.random.default_rng(17)
    dates = pd.date_range("2017-01-04", "2022-02-02", freq="W-WED")
    return pd.DataFrame({"Ticker": np.tile(list("ABCDEFGH"), len(dates)),
                         "week_date": dates.repeat(8),
                         "return_1w": rng.normal(size=len(dates) * 8) / 100})


def test_cv_groups_whole_weeks_and_returns_positions():
    dates = pd.Series(pd.date_range("2020-01-01", periods=20, freq="W-WED").repeat(3))
    dates = dates.sample(frac=1, random_state=19)
    dates.index = dates.index * 11 + 1000
    weeks = pd.DatetimeIndex(sorted(dates.unique()))
    for tr, va in date_cv_splits(dates, n_splits=4, gap=1):
        td, vd = dates.iloc[tr], dates.iloc[va]
        assert td.max() < vd.min()
        assert weeks.get_loc(vd.min()) - weeks.get_loc(td.max()) == 2
        assert len(td) % 3 == len(vd) % 3 == 0
        assert set(tr).isdisjoint(va)
        assert set(dates[dates.isin(vd)].index) == set(vd.index)


def test_window_dates_and_future_targets_do_not_affect_training(panel, config):
    prepared = prepare_panel(panel, config)
    as_of = pd.Timestamp("2022-01-05")
    train = training_window(prepared, as_of, config)
    assert train.week_date.min() > as_of - pd.DateOffset(years=5)
    assert train.week_date.max() == as_of
    assert train.prev_week_date.min() >= pd.Timestamp(config.data_start)
    changed = prepared.copy()
    changed["active_return_fwd"] = np.nan
    changed.loc[changed.week_date.gt(as_of), "active_return_train"] = 10000
    columns = ["return_1w_train", "volatility_20d_train", "active_return_train"]
    pd.testing.assert_frame_equal(train[columns], training_window(changed, as_of, config)[columns])
    with pytest.raises(ValueError, match="full five-year"):
        training_window(prepared, pd.Timestamp("2021-12-29"), config)


def test_loader_timestamp_and_lag_validation(panel, config):
    original = prepare_panel(panel, config)
    panel["prev_week_date"] = panel["prev_week_date"].map(
        lambda d: str(d.value) if pd.notna(d) else None
    )
    parsed = prepare_panel(panel, config)
    pd.testing.assert_series_equal(original.prev_week_date, parsed.prev_week_date)
    panel.loc[100, "return_1w_train"] = 999
    with pytest.raises(ValueError, match="Lagged features"):
        prepare_panel(panel, config)


def test_fit_predict_save_reload_and_resume(panel, config, raw_returns, tmp_path, monkeypatch):
    monkeypatch.setitem(GRIDS, "compact", {"n_estimators": [3], "max_depth": [2]})
    trainer = RollingPanelTrainer(config, tmp_path, workers=2, raw_returns=raw_returns)
    result = trainer.run(panel, model_start="2022-01-26")
    assert result["model_count"] == 2
    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert predictions.xgboost_prediction.notna().all()
    assert predictions.loc[predictions.week_date.eq("2022-02-02"), "active_return_fwd"].isna().all()
    first = tmp_path / "as_of=2022-01-26"
    before = (first / "metadata.json").stat().st_mtime_ns
    rerun = trainer.run(panel, model_start="2022-01-26")
    assert result["metrics"] == rerun["metrics"]
    assert (first / "metadata.json").stat().st_mtime_ns == before
    pd.testing.assert_frame_equal(predictions, pd.read_parquet(tmp_path / "predictions.parquet"))
    metadata = json.loads((first / "metadata.json").read_text())
    assert (first / "xgboost.pkl").exists()
    assert (first / "linear_regression.pkl").exists()
    current = prepare_panel(panel, config).query('week_date == "2022-01-26"')
    for filename, column in [("xgboost.pkl", "xgboost_prediction"), ("linear_regression.pkl", "linear_prediction")]:
        model = joblib.load(first / filename)
        predicted = model.predict(current[list(config.features)])
        expected = predictions.loc[predictions.week_date.eq("2022-01-26"), column]
        assert np.allclose(predicted, expected)
    assert metadata["training_weeks"] >= 260
    assert "active_return_fwd" not in metadata["config"]["features"]


def test_invalid_data_and_sql(panel, config):
    with pytest.raises(ValueError, match="Duplicate ticker"):
        prepare_panel(pd.concat([panel, panel.iloc[:1]]), config)
    with pytest.raises(ValueError, match="allowlist"):
        TrainingConfig(features=("active_return_fwd",))
    sql = panel_query("stock_market_test", "panel_table", "2017-01-01", None)
    assert '\"week_date\" >= \'2017-01-01\'' in sql
    assert '"active_return_train"' in sql and '"active_return_fwd"' in sql
    with pytest.raises(ValueError, match="identifier"):
        panel_query("bad;drop", "panel", "2017-01-01", None)
    assert regression_metrics([np.nan], [1])["n"] == 0
    assert regression_metrics([1, 2], [1, 2])["mse"] == 0
