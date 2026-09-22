"""Market-data download helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf


def download_data_daily(
    ticker: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker
        Yahoo Finance ticker symbol.
    start_date
        Inclusive start date.
    end_date
        Inclusive end date.

    Returns
    -------
    pd.DataFrame
        Daily market data sorted chronologically.
    """

    symbol = ticker.strip().upper()

    if not symbol:
        raise ValueError("ticker must not be empty")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if pd.isna(start) or pd.isna(end):
        raise ValueError(
            "start_date and end_date must be valid dates"
        )

    if end < start:
        raise ValueError(
            "end_date must be on or after start_date"
        )

    # yfinance treats end as exclusive
    data = yf.download(
        symbol,
        start=start.date().isoformat(),
        end=(end.date() + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=30,
    )

    if data.empty:
        raise ValueError(
            f"No daily price data returned for {symbol} "
            f"from {start.date()} through {end.date()}"
        )

    # yfinance can return MultiIndex columns even for one ticker
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns.name = None

    data = (
        data.rename_axis("date")
        .reset_index()
    )

    # Normalize names
    data.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in data.columns
    ]

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Unexpected Yahoo Finance response for {symbol}; "
            f"missing columns: {missing}"
        )

    # Keep adjusted close if Yahoo supplies it
    columns_to_keep = required_columns.copy()

    if "adj_close" in data.columns:
        columns_to_keep.append("adj_close")

    result = data.loc[:, columns_to_keep].copy()

    result["date"] = (
        pd.to_datetime(result["date"])
        .dt.tz_localize(None)
    )

    result["ticker"] = symbol
    result["asset_id"] = f"US_{symbol}"
    result["data_source"] = "yahoo_finance"

    result = result.dropna(
        subset=["open", "high", "low", "close"]
    )

    result = (
        result
        .sort_values("date")
        .drop_duplicates(["asset_id", "date"])
        .reset_index(drop=True)
    )

    return result