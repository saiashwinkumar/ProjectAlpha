import numpy as np
import pandas as pd
import pytest

from cross_sectional_panel import build_cross_sectional_panel


FEATURE_DIRECTIONS = {
    "momentum": 1,
    "volatility": -1,
    "rsi": -1,
}


def sample_weekly_panel() -> pd.DataFrame:
    rows = []
    values = {
        pd.Timestamp("2024-01-03"): {
            "A": (1.0, 1.0, 20.0, 0.01),
            "B": (2.0, 2.0, 40.0, 0.03),
            "C": (3.0, 3.0, 60.0, 0.02),
            "D": (4.0, 4.0, 80.0, 0.04),
        },
        # Deliberately skip 2024-01-10: prev_week_date means the last observed
        # panel week, not a hard-coded seven-day subtraction.
        pd.Timestamp("2024-01-17"): {
            "A": (4.0, 4.0, 80.0, 0.04),
            "B": (3.0, 3.0, 60.0, 0.02),
            "C": (2.0, 2.0, 40.0, 0.05),
            "D": (1.0, 1.0, 20.0, 0.01),
        },
    }
    sectors = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Health"}
    for week, stocks in values.items():
        for ticker, (momentum, volatility, rsi, return_1w) in stocks.items():
            rows.append(
                {
                    "Ticker": ticker,
                    "sector": sectors[ticker],
                    "week_date": week,
                    "momentum": momentum,
                    "volatility": volatility,
                    "rsi": rsi,
                    "return_1w": return_1w,
                }
            )
    return pd.DataFrame(rows)


def test_direction_zscores_sector_neutrality_and_lag() -> None:
    result = build_cross_sectional_panel(
        sample_weekly_panel(), feature_directions=FEATURE_DIRECTIONS
    )

    for _, week in result.groupby("week_date"):
        for feature in FEATURE_DIRECTIONS:
            assert week[feature].mean() == pytest.approx(0.0)
            assert week[feature].std(ddof=0) == pytest.approx(1.0)
            sector_means = week.groupby("sector")[feature].mean()
            assert np.allclose(sector_means, 0.0)

    first = result[result["week_date"].eq("2024-01-03")].set_index("Ticker")
    # Higher raw momentum is better; lower volatility and RSI are better.
    assert first.loc["B", "momentum"] > first.loc["A", "momentum"]
    assert first.loc["A", "volatility"] > first.loc["B", "volatility"]
    assert first.loc["A", "rsi"] > first.loc["B", "rsi"]

    second = result[result["week_date"].eq("2024-01-17")].set_index("Ticker")
    assert second["prev_week_date"].eq(pd.Timestamp("2024-01-03")).all()
    for feature in FEATURE_DIRECTIONS:
        expected = first[feature].sort_index()
        actual = second[f"{feature}_train"].sort_index()
        pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_lag_does_not_bridge_a_missing_ticker_week() -> None:
    source = sample_weekly_panel()
    source = source.loc[
        ~(
            source["Ticker"].eq("A")
            & source["week_date"].eq(pd.Timestamp("2024-01-03"))
        )
    ]
    result = build_cross_sectional_panel(
        source, feature_directions=FEATURE_DIRECTIONS
    )
    current_a = result.loc[
        result["Ticker"].eq("A")
        & result["week_date"].eq(pd.Timestamp("2024-01-17"))
    ].iloc[0]
    assert current_a[[f"{name}_train" for name in FEATURE_DIRECTIONS]].isna().all()


def test_active_return_targets_are_sector_neutral_and_exactly_aligned() -> None:
    result = build_cross_sectional_panel(
        sample_weekly_panel(), feature_directions=FEATURE_DIRECTIONS
    )
    first = result[result["week_date"].eq("2024-01-03")].set_index("Ticker")
    second = result[result["week_date"].eq("2024-01-17")].set_index("Ticker")

    # A and C underperform their respective sector averages in the first week;
    # B and D outperform them.
    assert first.loc["A", "active_return_train"] < 0
    assert first.loc["C", "active_return_train"] < 0
    assert first.loc["B", "active_return_train"] > 0
    assert first.loc["D", "active_return_train"] > 0

    pd.testing.assert_series_equal(
        first["active_return_fwd"].sort_index(),
        second["active_return_train"].sort_index(),
        check_names=False,
    )
    assert second["active_return_fwd"].isna().all()
    for _, week in result.groupby("week_date"):
        assert week["active_return_train"].mean() == pytest.approx(0.0)
        assert week["active_return_train"].std(ddof=0) == pytest.approx(1.0)
        assert np.allclose(
            week.groupby("sector")["active_return_train"].mean(), 0.0
        )


def test_external_sector_map_handles_share_class_punctuation() -> None:
    source = sample_weekly_panel().drop(columns="sector")
    source.loc[source["Ticker"].eq("A"), "Ticker"] = "BRK.B"
    mapping = pd.DataFrame(
        {
            "Ticker": ["BRK-B", "B", "C", "D"],
            "sector": ["Tech", "Tech", "Health", "Health"],
        }
    )
    result = build_cross_sectional_panel(
        source, mapping, feature_directions=FEATURE_DIRECTIONS
    )
    assert result.loc[result["Ticker"].eq("BRK.B"), "sector"].eq("Tech").all()
