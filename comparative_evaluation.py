"""Comparable rank and Q1-Q5 metrics from saved, independently audited scores.

No fitting or inference occurs here. Q1 is highest score for every model.
Annualization is arithmetic (52 weeks); turnover excludes portfolio entry and
uses target weights rather than drifted holdings, matching the reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm_listwise_training import ndcg, relevance_labels
from portfolio_training import COST_BPS
from training import write_json

DEFAULT_OUTPUT = Path("artifacts/ml/continuous_260w_v2/backtest")
MODEL_ORDER = ["linear_regression", "xgboost", "lightgbm"]


def comparative_metrics(members: pd.DataFrame):
    required = ["model", "week_date", "ticker", "prediction", "quintile", "active_return_fwd",
                "stock_return_fwd", "sector_return_fwd", "raw_active_return_fwd"]
    missing = set(required) - set(members)
    if missing:
        raise ValueError(f"Missing comparison inputs: {sorted(missing)}")
    frame = members[required].copy()
    frame["week_date"] = pd.to_datetime(frame.week_date)
    if frame.duplicated(["model", "week_date", "ticker"]).any():
        raise ValueError("Duplicate model/week/ticker observations")
    models = [m for m in MODEL_ORDER if m in set(frame.model)]
    models += sorted(set(frame.model) - set(models))
    if not models:
        raise ValueError("No model observations")
    rows = []
    for model, model_frame in frame.groupby("model", sort=False):
        previous_weights, previous_date = None, None
        for date, group in model_frame.groupby("week_date", sort=True):
            # Membership and score coverage are fixed independently of ground truth.
            group = group.loc[np.isfinite(group.prediction)].sort_values("ticker")
            row = dict(model=model, week_date=date, scored_stocks=len(group), rank_ic=np.nan,
                       ndcg_at_50=np.nan, effective_ndcg_k=min(50, len(group)),
                       q1_q5_active_return=np.nan, q1_q5_stock_return=np.nan, turnover=np.nan)
            if len(group) < 5:
                rows.append(row)
                previous_weights, previous_date = None, date
                continue
            # Verify that stored membership is the requested Q1-high convention.
            ordered = group.sort_values(["prediction", "ticker"], ascending=[False, True])
            expected = np.arange(len(ordered)) * 5 // len(ordered) + 1
            if not np.array_equal(ordered.quintile.to_numpy(), expected):
                raise ValueError(f"Quintiles are not Q1-high with ticker tie breaks: {model} {date}")
            long = group.loc[group.quintile.eq(1)]
            short = group.loc[group.quintile.eq(5)]
            weights = pd.Series(0.0, index=group.ticker)
            weights.loc[long.ticker] = 1.0 / len(long)
            weights.loc[short.ticker] = -1.0 / len(short)
            if previous_weights is not None and (date - previous_date).days == 7:
                row["turnover"] = float(0.5 * weights.subtract(previous_weights, fill_value=0).abs().sum())
            previous_weights, previous_date = weights, date
            targets = group.active_return_fwd
            if np.isfinite(targets).all():
                row["ndcg_at_50"] = ndcg(relevance_labels(targets), group.prediction.to_numpy(),
                                         row["effective_ndcg_k"])
                if targets.nunique() > 1 and group.prediction.nunique() > 1:
                    row["rank_ic"] = float(group.prediction.rank().corr(targets.rank()))
            return_columns = ["stock_return_fwd", "sector_return_fwd", "raw_active_return_fwd"]
            # Match the continuous backtest's whole-week missing-return policy.
            if np.isfinite(group[return_columns].to_numpy(dtype=float)).all():
                row["q1_q5_active_return"] = float(long.raw_active_return_fwd.mean() - short.raw_active_return_fwd.mean())
                row["q1_q5_stock_return"] = float(long.stock_return_fwd.mean() - short.stock_return_fwd.mean())
            rows.append(row)
    weekly = pd.DataFrame(rows)
    if weekly.empty:
        raise ValueError("No weekly observations")
    # Every metric uses the same dates across models. Different metrics can have
    # different coverage because returns, normalized targets, and turnover differ.
    common_dates = {}
    for name, columns in {"ranking": ["rank_ic", "ndcg_at_50"],
                          "returns": ["q1_q5_active_return", "q1_q5_stock_return"],
                          "turnover": ["turnover"]}.items():
        eligible = weekly.loc[weekly[columns].notna().all(axis=1)]
        counts = eligible.groupby("week_date").model.nunique()
        common_dates[name] = counts.index[counts.eq(len(models))]
        weekly[f"common_{name}_week"] = weekly.week_date.isin(common_dates[name])
    def annual_ir(series):
        std = series.std(ddof=1)
        return float(np.sqrt(52) * series.mean() / std) if std > 0 else None
    def mean(series):
        return float(series.mean()) if len(series) else None
    comparison = []
    for model in models:
        g = weekly.loc[weekly.model.eq(model)]
        ranks = g.loc[g.common_ranking_week]
        returns = g.loc[g.common_returns_week]
        turnover = g.loc[g.common_turnover_week]
        comparison.append(dict(model=model, rank_ic=mean(ranks.rank_ic), ndcg_at_50=mean(ranks.ndcg_at_50),
                               q1_q5_annualized_return=(52 * mean(returns.q1_q5_active_return) if len(returns) else None),
                               q1_q5_annualized_ir=annual_ir(returns.q1_q5_active_return),
                               mean_weekly_turnover=mean(turnover.turnover),
                               # Retain the unadjusted stock-return variant to make
                               # the primary sector-active return convention explicit.
                               q1_q5_annualized_stock_return=(52 * mean(returns.q1_q5_stock_return) if len(returns) else None),
                               q1_q5_annualized_stock_ir=annual_ir(returns.q1_q5_stock_return),
                               ranking_weeks=len(ranks), return_weeks=len(returns),
                               turnover_observations=len(turnover)))
    comparison = pd.DataFrame(comparison)
    details = dict(first_score_date=str(weekly.week_date.min().date()),
                   last_score_date=str(weekly.week_date.max().date()),
                   definitions={
                       "rank_ic": "Mean weekly cross-sectional Spearman correlation of prediction with active_return_fwd",
                       "ndcg_at_50": "Mean weekly long-side NDCG at min(50, scored stocks); forward target ranked within week into labels 0-4; linear gains 0-4; prediction ties use ticker order",
                       "q1_q5_annualized_return": "52 * mean(weekly Q1 raw sector-active return minus Q5 raw sector-active return); arithmetic, not CAGR",
                       "q1_q5_annualized_ir": "sqrt(52) * mean(weekly Q1-Q5 raw sector-active spread) / sample standard deviation(ddof=1)",
                       "mean_weekly_turnover": "Mean 0.5 * sum(abs(current target weights - previous target weights)); Q1 +100%, Q5 -100%; no drift adjustment, initial entry excluded, no bridging absent calendar weeks",
                       "q1_q5_annualized_stock_return": "52 * mean(weekly Q1 stock return minus Q5 stock return), without sector benchmark subtraction",
                       "q1_q5_annualized_stock_ir": "sqrt(52) * mean(weekly Q1-Q5 stock-return spread) / sample standard deviation(ddof=1)",
                   }, common_dates={name: [str(d.date()) for d in dates] for name, dates in common_dates.items()},
                   return_convention="Q1 highest score, Q5 lowest; actual saved quintile memberships; equal weights; before costs",
                   missing_policy="Whole scored universe needs observed targets for ranking and raw returns for return metrics; common eligible weeks across every model, by metric; missing values are not imputed",
                   ndcg_k_min=int(weekly.effective_ndcg_k.min()), ndcg_k_max=int(weekly.effective_ndcg_k.max()))
    return comparison, weekly, details


def cost_sensitivity(weekly: pd.DataFrame) -> pd.DataFrame:
    """Realized Q1-Q5 active returns; bps charged per dollar of gross trading."""
    records = []
    for model, group in weekly.groupby("model", sort=True):
        group = group.loc[group.common_returns_week].sort_values("week_date")
        if group.empty:
            continue
        # Entry from cash trades both 100% legs. A later score week with no
        # realized outcome still changes holdings, so use its stored turnover.
        trading = group.turnover.fillna(1.0).to_numpy(dtype=float)
        for bps in COST_BPS:
            returns = group.q1_q5_active_return.to_numpy(dtype=float) - 2 * bps / 10000 * trading
            wealth = np.cumprod(1 + returns)
            drawdown = wealth / np.maximum.accumulate(np.r_[1.0, wealth])[1:] - 1
            std = np.std(returns, ddof=1) if len(returns) > 1 else np.nan
            records.append(dict(model=model, cost_bps_per_dollar_traded=bps,
                                annualized_return=float(52 * np.mean(returns)),
                                annualized_ir=float(np.sqrt(52) * np.mean(returns) / std) if std > 0 else np.nan,
                                cumulative_return=float(wealth[-1] - 1),
                                max_drawdown=float(drawdown.min()),
                                mean_weekly_turnover=float(np.mean(trading)),
                                return_weeks=len(group)))
    return pd.DataFrame(records)


def save_comparative_metrics(members, output_dir=DEFAULT_OUTPUT):
    comparison, weekly, details = comparative_metrics(members)
    costs = cost_sensitivity(weekly)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "comparative_evaluation.csv", index=False)
    comparison.to_parquet(output / "comparative_evaluation.parquet", index=False)
    weekly.to_parquet(output / "comparative_weekly_metrics.parquet", index=False)
    costs.to_csv(output / "cost_sensitivity.csv", index=False)
    costs.to_parquet(output / "cost_sensitivity.parquet", index=False)
    # Missing metrics are represented as null in the machine-readable summary.
    records = comparison.astype(object).where(comparison.notna(), None).to_dict("records")
    write_json(output / "comparative_evaluation.json", {**details, "models": records,
               "cost_sensitivity": costs.astype(object).where(costs.notna(), None).to_dict("records"),
               "cost_definition": "Net weekly Q1-Q5 active return = gross - 2 * turnover * (cost_bps / 10000); turnover is half gross target-weight trade, with first entry from cash charged as 1.0. Cost scenarios are 0, 5, 10, 25 bps per dollar traded."})
    return comparison, weekly, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    members = pd.read_parquet(args.backtest_dir / "stock_scores_and_quintiles.parquet")
    comparison, _, _ = save_comparative_metrics(members, args.backtest_dir)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
