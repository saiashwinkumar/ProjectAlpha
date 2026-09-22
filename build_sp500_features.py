#!/usr/bin/env python3
"""Close-based weekly features for SP500_Historical_Data.csv.

Install: pip install pandas numpy pandas_market_calendars
Run: python build_sp500_features.py --input SP500_Historical_Data.csv
Optional: --as-of 2026-02-18 --output features_20260218

Assumes complete end-of-day data through --as-of (default: max CSV date).
Wednesday anchors use the last NYSE session on/before Wednesday. A missing
stock quote on that session stays missing; it is NOT a holiday fallback.
Training is X(t) -> adjusted-close return from t to t+1; inference is X(latest).
These are research labels, NOT executable returns after observing close t.
For an executable backtest, separately align entry/exit prices after signals.
No survivorship correction or sector neutralization is performed.
Outputs are CSV plus metadata.json. Pass ONLY metadata['feature_columns'] to ML.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


def build(input_path, output_path, as_of=None):
    df = pd.read_csv(input_path, usecols=['Ticker', 'Date', 'Adj Close'])
    if df.empty or df[['Ticker', 'Date']].isna().any().any():
        raise ValueError('Empty data or missing ticker/date.')
    df['Ticker'] = df['Ticker'].astype(str).str.strip()
    df['Date'] = pd.to_datetime(df['Date'], errors='raise').dt.normalize()
    if (df['Ticker'] == '').any() or df.duplicated(['Ticker', 'Date']).any():
        raise ValueError('Blank ticker or duplicate ticker/date: fix input first.')
    df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='raise')
    invalid = df['Adj Close'].isna() | ~np.isfinite(df['Adj Close']) | (df['Adj Close'] <= 0)
    if invalid.any():
        raise ValueError(f'{invalid.sum()} invalid adjusted prices: investigate first.')
    source_max = df['Date'].max()
    cutoff = pd.Timestamp(as_of).normalize() if as_of else source_max
    if cutoff < df['Date'].min():
        raise ValueError('As-of precedes the data.')
    df = df.loc[df['Date'] <= cutoff].copy()
    # Include a buffer so an early holiday anchor can find its preceding session.
    sessions = mcal.get_calendar('NYSE').valid_days(
        start_date=df['Date'].min() - pd.Timedelta(days=10), end_date=cutoff
    ).tz_localize(None)
    if not df['Date'].isin(sessions).all():
        raise ValueError('Input contains dates outside the NYSE session calendar.')
    anchors = pd.DataFrame({'week_date': pd.date_range(df['Date'].min(), cutoff, freq='W-WED')})
    if anchors.empty:
        raise ValueError('No completed Wednesday anchors in this range.')
    calendar = pd.merge_asof(
        anchors, pd.DataFrame({'price_date': sessions}),
        left_on='week_date', right_on='price_date', direction='backward'
    )
    if calendar['price_date'].iloc[-1] > df['Date'].max():
        raise ValueError('Latest required snapshot is after available data; update CSV or reduce --as-of.')
    calendar['label_end_week'] = calendar['week_date'] + pd.Timedelta(days=7)
    calendar['label_end_date'] = calendar['price_date'].shift(-1)
    feature_columns = [
        'return_1w', 'return_4w', 'return_13w', 'return_26w', 'momentum_12_1',
        'volatility_20d', 'volatility_60d', 'price_to_sma20', 'price_to_sma60',
        'bollinger_z20', 'rsi14_simple', 'close_to_high252',
    ]
    parts = []
    missing_sessions = 0
    for ticker, group in df.groupby('Ticker', sort=True):
        # Reindex to sessions: rolling windows cannot silently skip missing days.
        grid = sessions[(sessions >= group['Date'].min()) & (sessions <= cutoff)]
        p = group.set_index('Date')['Adj Close'].reindex(grid)
        missing_sessions += int(p.isna().sum())
        r = p.pct_change(fill_method=None)
        daily = pd.DataFrame({'price_date': grid, 'adj_close': p.to_numpy()})
        mean20, std20 = p.rolling(20).mean(), p.rolling(20).std()
        daily['volatility_20d'] = (r.rolling(20).std() * np.sqrt(252)).to_numpy()
        daily['volatility_60d'] = (r.rolling(60).std() * np.sqrt(252)).to_numpy()
        daily['price_to_sma20'] = (p / mean20 - 1).to_numpy()
        daily['price_to_sma60'] = (p / p.rolling(60).mean() - 1).to_numpy()
        z = (p - mean20) / std20.replace(0, np.nan)
        daily['bollinger_z20'] = z.mask(std20.eq(0), 0).to_numpy()
        delta = p.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        # Simple rolling RSI, deliberately not Wilder-smoothed RSI.
        rsi = 100 * gain / (gain + loss)
        daily['rsi14_simple'] = rsi.mask((gain + loss).eq(0), 50).to_numpy()
        daily['close_to_high252'] = (p / p.rolling(252).max() - 1).to_numpy()
        weekly = calendar.merge(daily, on='price_date', how='left', validate='one_to_one')
        weekly.insert(0, 'Ticker', ticker)
        for weeks in (1, 4, 13, 26):
            weekly[f'return_{weeks}w'] = weekly['adj_close'] / weekly['adj_close'].shift(weeks) - 1
        # Approximate 12-minus-1-month momentum using exact 52/4 weekly anchors.
        weekly['momentum_12_1'] = weekly['adj_close'].shift(4) / weekly['adj_close'].shift(52) - 1
        weekly['target_return_1w'] = weekly['adj_close'].shift(-1) / weekly['adj_close'] - 1
        parts.append(weekly)
    panel = pd.concat(parts, ignore_index=True).sort_values(['week_date', 'Ticker'])
    panel[feature_columns] = panel[feature_columns].replace([np.inf, -np.inf], np.nan)
    eligible = panel[feature_columns].notna().all(axis=1) & panel['adj_close'].notna()
    latest = calendar['week_date'].iloc[-1]
    training = panel.loc[eligible & panel['target_return_1w'].notna() &
                         panel['label_end_week'].le(latest)].copy()
    # Equal outcomes get equal relevance. Categories can be empty in small groups.
    pct = training.groupby('week_date')['target_return_1w'].rank(method='average', pct=True)
    training['target_relevance'] = np.minimum(np.ceil(pct * 10) - 1, 9).astype(int)
    inference_columns = ['Ticker', 'week_date', 'price_date', 'adj_close'] + feature_columns
    inference = panel.loc[eligible & panel['week_date'].eq(latest), inference_columns].copy()
    if inference.empty:
        raise ValueError('No eligible inference rows. Need about 52 weeks of history and complete windows.')
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / 'weekly_panel.csv', index=False)
    training.to_csv(out / 'training.csv', index=False)
    inference.to_csv(out / 'inference.csv', index=False)
    calendar.to_csv(out / 'snapshot_calendar.csv', index=False)
    metadata = {
        'input': str(input_path), 'source_max_date': str(source_max.date()),
        'as_of': str(cutoff.date()), 'inference_week': str(latest.date()),
        'inference_price_date': str(calendar['price_date'].iloc[-1].date()),
        'feature_columns': feature_columns, 'regression_target': 'target_return_1w',
        'ranking_target': 'target_relevance', 'ranking_group': 'week_date',
        'training_rows': len(training), 'inference_rows': len(inference),
        'missing_ticker_sessions_including_after_last_quote': missing_sessions,
        'latest_ineligible_tickers': panel.loc[panel['week_date'].eq(latest) & ~eligible, 'Ticker'].tolist(),
        'target_definition': 'X(t) -> AdjClose(next weekly snapshot)/AdjClose(t)-1; no second feature shift',
        'warning': 'Research close-to-close labels; not post-signal execution returns. Survivor universe remains biased.',
        'versions': {'pandas': pd.__version__, 'numpy': np.__version__, 'pandas_market_calendars': mcal.__version__},
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    return panel, training, inference, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='SP500_Historical_Data.csv')
    parser.add_argument('--output', default='sp500_features')
    parser.add_argument('--as-of', help='YYYY-MM-DD; assumes that session is complete. Default max CSV date.')
    args = parser.parse_args()
    build(args.input, args.output, args.as_of)
