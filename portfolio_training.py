"""Point-in-time portfolio validation and three-seed rank ensembles.

The input return for outcome week t is the raw stock return observed at t.
It is joined to the lagged-feature training row at t; no t+1 outcome is used.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cross_sectional_panel import build_cross_sectional_panel

COST_BPS = (0, 5, 10, 25)
SELECTION_COST_BPS = 10
ENSEMBLE_SEEDS = (42, 73, 107)


def raw_training_outcomes(panel: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw source against the unchanged normalized training target."""
    source = raw.rename(columns={c: c.lower() for c in raw.columns}).copy()
    source["ticker"] = source["ticker"].astype("string").str.strip().str.upper()
    source["week_date"] = pd.to_datetime(source["week_date"]).dt.normalize()
    if source.duplicated(["ticker", "week_date"]).any():
        raise ValueError("Duplicate raw ticker/week returns")
    joined = panel[["ticker", "sector", "week_date", "active_return_train"]].merge(
        source[["ticker", "week_date", "return_1w"]],
        on=["ticker", "week_date"], how="left", validate="one_to_one")
    joined["return_1w"] = pd.to_numeric(joined["return_1w"], errors="raise")
    if not np.isfinite(joined.return_1w.to_numpy(dtype=float)).all() or joined.return_1w.lt(-1).any():
        raise ValueError("Missing or invalid raw stock returns for training portfolio validation")
    reconstructed = build_cross_sectional_panel(
        joined.rename(columns={"ticker": "Ticker"}), feature_directions={"return_1w": 1})
    if not np.allclose(joined.active_return_train, reconstructed.active_return_train,
                       rtol=1e-8, atol=1e-10, equal_nan=True):
        raise ValueError("Raw returns do not reconstruct the panel training target")
    joined["raw_active_return"] = joined.return_1w - joined.groupby(
        ["week_date", "sector"], observed=True).return_1w.transform("mean")
    return joined[["ticker", "week_date", "raw_active_return"]]


def attach_training_returns(train: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    result = train.merge(outcomes, on=["ticker", "week_date"], how="left", validate="one_to_one")
    if result.raw_active_return.isna().any():
        raise ValueError("Training window lacks validated raw realized returns")
    return result


def within_week_percentile(predictions) -> np.ndarray:
    """Higher score means better rank; input rows are already ticker sorted."""
    values = np.asarray(predictions, dtype=float)
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


class SeedRankEnsemble:
    """Pickle-compatible scorer; predict is called for one weekly universe."""

    def __init__(self, models, features, kind):
        self.models = tuple(models)
        self.features = tuple(features)
        self.kind = kind
        self.feature_names_in_ = np.asarray(features, dtype=object)

    def feature_name(self):
        return list(self.features)

    def set_params(self, **params):
        if self.kind == "xgboost":
            for model in self.models:
                model.set_params(**params)
        return self

    def predict(self, X, num_threads=None):
        if list(X.columns) != list(self.features):
            raise ValueError("Ensemble feature order differs from training")
        if len(X) == 0:
            return np.empty(0)
        ranks = []
        for model in self.models:
            if self.kind == "lightgbm":
                kw = {"num_threads": num_threads} if num_threads is not None else {}
                scores = model.predict(X, **kw)
            else:
                scores = model.predict(X)
            ranks.append(within_week_percentile(scores))
        return np.mean(ranks, axis=0)

    def predict_raw(self, X):
        """Continuous mean prediction for regression diagnostics only."""
        if self.kind != "xgboost" or list(X.columns) != list(self.features):
            raise ValueError("Raw predictions require the XGBoost training feature order")
        return np.mean([model.predict(X) for model in self.models], axis=0)


def _ndcg(labels, order, k):
    discount = np.log2(np.arange(2, k + 2))
    ideal = np.sort(labels)[::-1][:k]
    denominator = float(np.sum(ideal / discount))
    return float(np.sum(labels[order[:k]] / discount) / denominator) if denominator > 0 else 0.0


def symmetric_ndcg(labels, scores, groups):
    """Mean long/short NDCG at 20% using stable ticker order within each group."""
    values, offset = [], 0
    for n in groups:
        n = int(n)
        y, pred = np.asarray(labels[offset:offset+n]), np.asarray(scores[offset:offset+n])
        k = max(1, math.ceil(n / 5))
        order = np.argsort(-pred, kind="stable")
        values.append(0.5 * (_ndcg(y, order, k) + _ndcg(4-y, order[::-1], k)))
        offset += n
    if offset != len(scores):
        raise ValueError("Ranking group sizes do not match predictions")
    return float(np.mean(values))


def portfolio_weekly(frame: pd.DataFrame, scores) -> pd.DataFrame:
    """Q1/Q5 match assign_quintiles: floor-sized groups, stable ticker ties."""
    work = frame[["ticker", "week_date", "active_return_train", "raw_active_return"]].copy()
    work["score"] = np.asarray(scores, dtype=float)
    rows, prior, prior_date = [], None, None
    for date, group in work.groupby("week_date", sort=True):
        group = group.sort_values("ticker").reset_index(drop=True)
        if len(group) < 5 or not np.isfinite(group[["score", "raw_active_return"]]).all().all():
            raise ValueError("Incomplete validation portfolio universe or returns")
        ordered = group.sort_values(["score", "ticker"], ascending=[False, True]).reset_index(drop=True)
        quintile = np.arange(len(ordered)) * 5 // len(ordered) + 1
        long, short = ordered.loc[quintile == 1], ordered.loc[quintile == 5]
        weights = pd.Series(0.0, index=group.ticker.to_numpy())
        weights.loc[long.ticker] = 1 / len(long)
        weights.loc[short.ticker] = -1 / len(short)
        consecutive = prior is not None and (date - prior_date).days == 7
        turnover = (float(0.5 * weights.subtract(prior, fill_value=0).abs().sum())
                    if consecutive else np.nan)
        # A new validation block enters both legs from cash (half gross trade = 1).
        cost_turnover = turnover if consecutive else 1.0
        y, s = group.active_return_train.to_numpy(), group.score.to_numpy()
        ic = float(pd.Series(y).rank().corr(pd.Series(s).rank())) if np.ptp(y) and np.ptp(s) else np.nan
        relevance = np.clip(np.ceil(pd.Series(y).rank(method="first").to_numpy() / len(y) * 5)-1, 0, 4)
        order = np.argsort(-s, kind="stable")
        rows.append(dict(week_date=date, gross_return=float(long.raw_active_return.mean()-short.raw_active_return.mean()),
                         turnover=turnover, cost_turnover=cost_turnover, rank_ic=ic,
                         ndcg_at_50=_ndcg(relevance, order, min(50, len(y)))))
        prior, prior_date = weights, date
    return pd.DataFrame(rows)


def portfolio_summary(weekly: pd.DataFrame, cost_bps=COST_BPS) -> dict:
    rank_ic = weekly.rank_ic.mean()
    ndcg50 = weekly.ndcg_at_50.mean()
    result = {"weeks": len(weekly), "rank_ic": float(rank_ic) if np.isfinite(rank_ic) else None,
              "ndcg_at_50": float(ndcg50) if np.isfinite(ndcg50) else None,
              "weekly_turnover": float(weekly.turnover.mean()) if weekly.turnover.notna().any() else None}
    for bps in cost_bps:
        returns = weekly.gross_return.to_numpy(dtype=float) - (2 * bps / 10000) * weekly.cost_turnover.to_numpy(dtype=float)
        std = np.std(returns, ddof=1) if len(returns) > 1 else np.nan
        wealth = np.cumprod(1 + returns)
        drawdown = wealth / np.maximum.accumulate(np.r_[1.0, wealth])[1:] - 1
        result[f"cost_{bps}bps"] = dict(
            annualized_return=float(52 * np.mean(returns)),
            information_ratio=float(np.sqrt(52) * np.mean(returns) / std) if std > 0 else None,
            cumulative_return=float(wealth[-1]-1), max_drawdown=float(drawdown.min()))
    return result


def better_portfolio(candidate: dict, incumbent: dict | None, complexity: tuple) -> bool:
    """10bp net IR, then cumulative return, then simpler model within tolerance."""
    if incumbent is None:
        return True
    a = candidate["portfolio"]["cost_10bps"]
    b = incumbent["portfolio"]["cost_10bps"]
    ai, bi = a["information_ratio"], b["information_ratio"]
    ai = -np.inf if ai is None else ai
    bi = -np.inf if bi is None else bi
    if abs(ai-bi) > 0.05:
        return ai > bi
    if abs(a["cumulative_return"]-b["cumulative_return"]) > 0.01:
        return a["cumulative_return"] > b["cumulative_return"]
    return complexity < incumbent["complexity"]
