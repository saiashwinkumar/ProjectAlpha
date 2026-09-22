#!/usr/bin/env python3
"""Build a sector-neutral, point-in-time weekly feature panel.

The final feature and target columns are weekly cross-sectional z-scores.  Every
score is oriented so that a larger value is more favorable for the stock.
Feature columns with the ``_train`` suffix contain the already-transformed score
from the preceding observed week, joined by ticker and date (never by a row-wise
backward fill).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


# +1 means that a larger raw value is favorable; -1 means that a smaller raw
# value is favorable.  RSI is intentionally treated as a mean-reversion signal
# here, as requested.
DEFAULT_FEATURE_DIRECTIONS: dict[str, int] = {
    "return_1w": 1,
    "return_4w": 1,
    "return_13w": 1,
    "return_26w": 1,
    "momentum_12_1": 1,
    "volatility_20d": -1,
    "volatility_60d": -1,
    "price_to_sma20": 1,
    "price_to_sma60": 1,
    "bollinger_z20": 1,
    "rsi14_simple": -1,
    "close_to_high252": 1,
}


def _ticker_key(values: pd.Series) -> pd.Series:
    """Normalize share-class punctuation only for ticker matching."""

    return (
        values.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )


def _prepare_sector_mapping(sectors: pd.DataFrame) -> pd.DataFrame:
    required = {"Ticker", "sector"}
    missing = required - set(sectors.columns)
    if missing:
        raise ValueError(f"Sector mapping is missing columns: {sorted(missing)}")

    mapping = sectors.loc[:, ["Ticker", "sector"]].copy()
    mapping["_ticker_key"] = _ticker_key(mapping["Ticker"])
    mapping["sector"] = mapping["sector"].astype("string").str.strip()
    mapping = mapping.loc[
        mapping["_ticker_key"].notna()
        & mapping["_ticker_key"].ne("")
        & mapping["sector"].notna()
        & mapping["sector"].ne("")
    ]

    conflicting = mapping.groupby("_ticker_key")["sector"].nunique().gt(1)
    if conflicting.any():
        tickers = conflicting.index[conflicting].tolist()
        raise ValueError(f"Conflicting sector assignments for: {tickers}")

    return mapping.drop_duplicates("_ticker_key").loc[:, ["_ticker_key", "sector"]]


def _centered_percentile_rank(
    values: pd.DataFrame,
    groups: list[pd.Series],
) -> pd.DataFrame:
    """Return (average_rank - 0.5) / group_count for each feature.

    Unlike pandas ``rank(pct=True)``, this convention has a mean of exactly 0.5
    in every non-empty group, including groups with ties.  That property keeps
    sector means neutral after the subsequent weekly z-score.
    """

    grouped = values.groupby(groups, observed=True, sort=False)
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    return (ranks - 0.5) / counts


def _weekly_zscore(values: pd.DataFrame, week_date: pd.Series) -> pd.DataFrame:
    grouped = values.groupby(week_date, observed=True, sort=False)
    means = grouped.transform("mean")
    mean_squares = values.pow(2).groupby(
        week_date, observed=True, sort=False
    ).transform("mean")
    variances = (mean_squares - means.pow(2)).clip(lower=0.0)
    std = np.sqrt(variances)

    scores = (values - means) / std.mask(std.eq(0.0))
    # A non-null constant cross-section has no dispersion and therefore no
    # active signal; represent it with the neutral score zero.
    return scores.mask(std.eq(0.0) & values.notna(), 0.0)


def build_cross_sectional_panel(
    weekly_panel: pd.DataFrame,
    sectors: pd.DataFrame | None = None,
    *,
    feature_directions: Mapping[str, int] | None = None,
    active_return_column: str = "return_1w",
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> pd.DataFrame:
    """Construct the requested current/previous-week cross-sectional panel.

    Processing is strictly within a snapshot:

    1. Reverse lower-is-better features.
    2. Winsorize at the supplied quantiles within ``week_date, sector``.
    3. Percentile-rank within ``week_date, sector``.
    4. Z-score the ranked values across the full ``week_date`` cross-section.
    5. Construct active return as raw stock return minus its week/sector mean,
       then apply steps 2-4 to that target.
    6. Join the preceding observed week's feature scores as ``*_train`` and the
       next observed week's active-return score as ``active_return_fwd``.

    ``sectors`` is optional when ``weekly_panel`` already has a complete
    ``sector`` column.  Missing raw features remain missing throughout.
    """

    if weekly_panel.empty:
        raise ValueError("weekly_panel is empty")
    if not 0.0 <= winsor_lower < winsor_upper <= 1.0:
        raise ValueError("Require 0 <= winsor_lower < winsor_upper <= 1")

    directions = dict(feature_directions or DEFAULT_FEATURE_DIRECTIONS)
    if not directions:
        raise ValueError("feature_directions must contain at least one feature")
    invalid_directions = {
        name: direction
        for name, direction in directions.items()
        if direction not in (-1, 1)
    }
    if invalid_directions:
        raise ValueError(
            "Feature directions must be +1 or -1; got "
            f"{invalid_directions}"
        )

    feature_columns = list(directions)
    required = {"Ticker", "week_date", active_return_column, *feature_columns}
    missing = required - set(weekly_panel.columns)
    if missing:
        raise ValueError(f"weekly_panel is missing columns: {sorted(missing)}")

    work = weekly_panel.copy()
    work["Ticker"] = work["Ticker"].astype("string").str.strip().str.upper()
    if work["Ticker"].isna().any() or work["Ticker"].eq("").any():
        raise ValueError("Ticker contains missing or blank values")
    work["week_date"] = pd.to_datetime(
        work["week_date"], errors="raise"
    ).dt.normalize()
    if work["week_date"].isna().any():
        raise ValueError("week_date contains missing values")

    work["_ticker_key"] = _ticker_key(work["Ticker"])
    duplicate_keys = work.duplicated(["_ticker_key", "week_date"], keep=False)
    if duplicate_keys.any():
        examples = work.loc[
            duplicate_keys, ["Ticker", "week_date"]
        ].head(10).to_dict("records")
        raise ValueError(f"Duplicate ticker/week rows; examples: {examples}")

    if sectors is not None:
        mapping = _prepare_sector_mapping(sectors)
        work = work.merge(
            mapping,
            on="_ticker_key",
            how="left",
            validate="many_to_one",
            suffixes=("", "_mapped"),
        )
        if "sector_mapped" in work.columns:
            existing = work["sector"].astype("string").str.strip()
            mapped = work["sector_mapped"].astype("string").str.strip()
            mismatch = existing.notna() & mapped.notna() & existing.ne(mapped)
            if mismatch.any():
                examples = work.loc[
                    mismatch, ["Ticker", "sector", "sector_mapped"]
                ].drop_duplicates().head(10).to_dict("records")
                raise ValueError(
                    f"Panel and mapping contain conflicting sectors: {examples}"
                )
            work["sector"] = existing.fillna(mapped)
            work = work.drop(columns="sector_mapped")
    elif "sector" not in work.columns:
        raise ValueError("Supply sectors or include a sector column in weekly_panel")

    work["sector"] = work["sector"].astype("string").str.strip()
    missing_sector = work["sector"].isna() | work["sector"].eq("")
    if missing_sector.any():
        tickers = sorted(work.loc[missing_sector, "Ticker"].unique().tolist())
        raise ValueError(f"Missing sector assignments for: {tickers}")

    numeric_columns = list(dict.fromkeys([*feature_columns, active_return_column]))
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="raise")
    work[numeric_columns] = work[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    work = work.sort_values(["week_date", "Ticker"]).reset_index(drop=True)

    oriented = work[feature_columns].mul(pd.Series(directions), axis="columns")
    sector_groups = [work["week_date"], work["sector"]]
    grouped = oriented.groupby(sector_groups, observed=True, sort=False)
    lower = grouped.transform("quantile", q=winsor_lower)
    upper = grouped.transform("quantile", q=winsor_upper)
    winsorized = oriented.clip(lower=lower, upper=upper, axis=None)

    percentiles = _centered_percentile_rank(winsorized, sector_groups)
    scores = _weekly_zscore(percentiles, work["week_date"])

    observed_return = work[active_return_column]
    sector_average_return = observed_return.groupby(
        sector_groups, observed=True, sort=False
    ).transform("mean")
    active_return = (observed_return - sector_average_return).to_frame(
        "active_return_train"
    )
    active_grouped = active_return.groupby(
        sector_groups, observed=True, sort=False
    )
    active_lower = active_grouped.transform("quantile", q=winsor_lower)
    active_upper = active_grouped.transform("quantile", q=winsor_upper)
    active_winsorized = active_return.clip(
        lower=active_lower, upper=active_upper, axis=None
    )
    active_percentile = _centered_percentile_rank(
        active_winsorized, sector_groups
    )
    active_score = _weekly_zscore(active_percentile, work["week_date"])

    result = work.loc[:, ["Ticker", "sector", "week_date"]].copy()
    result[feature_columns] = scores
    result["active_return_train"] = active_score["active_return_train"]

    # The previous date is the greatest observed panel date strictly below the
    # current date.  The explicit self-join prevents a missing ticker/week from
    # silently borrowing an older observation.
    observed_weeks = pd.Index(result["week_date"].drop_duplicates().sort_values())
    previous_by_week = pd.Series(
        observed_weeks[:-1].to_numpy(),
        index=observed_weeks[1:],
    )
    result["prev_week_date"] = result["week_date"].map(previous_by_week)

    lagged = result.loc[:, ["Ticker", "week_date", *feature_columns]].rename(
        columns={
            "week_date": "prev_week_date",
            **{name: f"{name}_train" for name in feature_columns},
        }
    )
    result = result.merge(
        lagged,
        on=["Ticker", "prev_week_date"],
        how="left",
        validate="many_to_one",
    )

    next_by_week = pd.Series(
        observed_weeks[1:].to_numpy(),
        index=observed_weeks[:-1],
    )
    result["_next_week_date"] = result["week_date"].map(next_by_week)
    forward = result.loc[
        :, ["Ticker", "week_date", "active_return_train"]
    ].rename(
        columns={
            "week_date": "_next_week_date",
            "active_return_train": "active_return_fwd",
        }
    )
    result = result.merge(
        forward,
        on=["Ticker", "_next_week_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns="_next_week_date")

    ordered = [
        "Ticker",
        "sector",
        "week_date",
        "prev_week_date",
        *feature_columns,
        *[f"{name}_train" for name in feature_columns],
        "active_return_train",
        "active_return_fwd",
    ]
    result = result.loc[:, ordered].sort_values(
        ["week_date", "Ticker"]
    ).reset_index(drop=True)
    result.attrs["feature_directions"] = directions
    result.attrs["winsor_limits"] = [winsor_lower, winsor_upper]
    result.attrs["target_columns"] = [
        "active_return_train", "active_return_fwd"
    ]
    result.attrs["active_return_definition"] = (
        f"{active_return_column} - mean({active_return_column}) by "
        "week_date/sector"
    )
    result.attrs["transform"] = (
        "direction -> week/sector winsorization -> week/sector centered "
        "percentile rank -> weekly population z-score -> exact feature lag "
        "and forward active-return join"
    )
    return result


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _write_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


def _load_sector_maps(paths: list[str]) -> pd.DataFrame:
    frames = [_read_table(path).loc[:, ["Ticker", "sector"]] for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    # _prepare_sector_mapping performs conflict validation and normalization.
    prepared = _prepare_sector_mapping(combined)
    return prepared.rename(columns={"_ticker_key": "Ticker"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Weekly CSV or Parquet")
    parser.add_argument(
        "--sector-map",
        action="append",
        help=(
            "Ticker/sector CSV or Parquet; repeat to combine maps. If omitted, "
            "the input must already contain sector."
        ),
    )
    parser.add_argument("--output", required=True, help="Output CSV or Parquet")
    parser.add_argument("--winsor-lower", type=float, default=0.01)
    parser.add_argument("--winsor-upper", type=float, default=0.99)
    args = parser.parse_args()

    weekly = _read_table(args.input)
    sectors = _load_sector_maps(args.sector_map) if args.sector_map else None
    panel = build_cross_sectional_panel(
        weekly,
        sectors,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
    )
    _write_table(panel, args.output)
    summary = {
        "output": str(args.output),
        "rows": len(panel),
        "tickers": int(panel["Ticker"].nunique()),
        "weeks": int(panel["week_date"].nunique()),
        "feature_columns": list(DEFAULT_FEATURE_DIRECTIONS),
        "target_columns": ["active_return_train", "active_return_fwd"],
        "active_return_definition": (
            "raw return_1w - week/sector average raw return_1w"
        ),
        "lower_is_better": [
            name for name, direction in DEFAULT_FEATURE_DIRECTIONS.items()
            if direction == -1
        ],
        "winsor_limits": [args.winsor_lower, args.winsor_upper],
        "zscore_ddof": 0,
        "transform_order": [
            "orient higher-is-better",
            "winsorize within week_date/sector",
            "centered percentile rank within week_date/sector",
            "z-score across week_date",
            "exact previous-observed-week self-join by Ticker",
            "exact next-observed-week active-return self-join by Ticker",
        ],
    }
    metadata_path = Path(args.output).with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    summary["metadata"] = str(metadata_path)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
