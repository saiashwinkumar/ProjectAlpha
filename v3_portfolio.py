"""Equal-capital Q1 portfolios and diagnostics for the V3 walk-forward study."""

from __future__ import annotations

import numpy as np
import pandas as pd

COST_BPS = 5
VARIANTS = {
    "exact_q1": (.20, False),
    "buffer_20_25": (.25, False),
    "buffer_20_30": (.30, False),
    "exact_q1_rank_weighted": (.20, True),
    "selected_buffer_rank_weighted": (None, True),
}


class Q1NetIRStopMetric:
    """Fast, deterministic 5 bps long-only IR for boosting early stopping."""

    def __init__(self, frame):
        ordered = frame.sort_values(["week_date", "ticker"])
        ids, tickers = pd.factorize(ordered.ticker, sort=True)
        self.n_assets = len(tickers)
        self.groups = []
        offset = 0
        for date, group in ordered.groupby("week_date", sort=False):
            n = len(group)
            self.groups.append((date, slice(offset, offset+n), ids[offset:offset+n],
                                group.raw_active_return.to_numpy(dtype=float)))
            offset += n
        self.n_rows = offset

    def ir(self, predictions):
        if len(predictions) != self.n_rows:
            raise ValueError("Early-stopping predictions do not match the stop window")
        returns, previous = [], None
        for _, rows, ids, realized in self.groups:
            scores = np.asarray(predictions[rows], dtype=float)
            top = np.argsort(-scores, kind="stable")[:int(np.ceil(.2*len(scores)))]
            weights = np.zeros(self.n_assets)
            weights[ids[top]] = 1 / len(top)
            traded = (np.abs(weights-previous).sum() if previous is not None
                      else np.abs(weights).sum())
            returns.append(float(realized[top].mean() - .0005*traded))
            previous = weights
        values = np.asarray(returns)
        std = values.std(ddof=1) if len(values) > 1 else np.nan
        return float(np.sqrt(52)*values.mean()/std) if std > 0 else -1e6

    def xgb(self, y_true, predictions):
        return -self.ir(predictions)

    def lightgbm(self, predictions, dataset):
        return "q1_net_ir_5bps", self.ir(predictions), True

    def catboost(self):
        parent = self

        class Metric:
            def is_max_optimal(self):
                return True

            def evaluate(self, approxes, target, weight):
                # CatBoost also evaluates the metric on its fitting Pool.
                # Only the separate stop Pool determines best_iteration_.
                if len(approxes[0]) != parent.n_rows:
                    return 0.0, 1.0
                return parent.ir(approxes[0]), 1.0

            def get_final_error(self, error, weight):
                return error / weight

        return Metric()


def percentile_ensemble(raw_predictions: np.ndarray, dates) -> np.ndarray:
    """Average each seed's within-week percentile rank, highest score best."""
    values = np.asarray(raw_predictions, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    result = np.empty(len(values), dtype=float)
    for _, indices in pd.Series(np.arange(len(values))).groupby(
            np.asarray(pd.to_datetime(dates)), sort=False):
        idx = indices.to_numpy()
        ranks = np.column_stack([
            pd.Series(values[idx, j]).rank(method="average", pct=True).to_numpy()
            for j in range(values.shape[1])
        ])
        result[idx] = ranks.mean(axis=1)
    return result


def target_weights(tickers, scores, previous=None, *, buffer=.20, rank_weighted=False):
    """Keep at most the Q1 count; fill vacancies only from the current top 20%."""
    tickers = np.asarray(tickers, dtype=str)
    scores = np.asarray(scores, dtype=float)
    if len(tickers) < 5 or not np.isfinite(scores).all() or len(set(tickers)) != len(tickers):
        raise ValueError("Need at least five unique tickers and finite scores")
    if not .20 <= buffer <= .30:
        raise ValueError("Retention threshold must be between 20% and 30%")
    order = np.lexsort((tickers, -scores))
    n_hold = int(np.ceil(.20 * len(tickers)))
    n_retain = int(np.ceil(buffer * len(tickers)))
    prior = set(previous or {})
    retained = [int(i) for i in order[:n_retain] if tickers[i] in prior][:n_hold]
    selected = retained + [int(i) for i in order[:n_hold] if int(i) not in retained][:n_hold-len(retained)]
    if len(selected) != n_hold:
        raise ValueError("Portfolio could not fill Q1 target size")
    selected.sort(key=lambda i: (-scores[i], tickers[i]))
    if rank_weighted and len(selected) > 1:
        # Linear 1.25 -> 0.75 multiplier across held ranks; sums to 1.
        multipliers = np.linspace(1.25, .75, len(selected))
        weights = multipliers / multipliers.sum()
    else:
        weights = np.full(len(selected), 1 / len(selected))
    return dict(zip(tickers[selected], weights)), set(tickers[order[:n_hold]])


def _ndcg_at_50(target, scores, tickers):
    n = len(target)
    order = np.lexsort((tickers, -scores))
    ranks = pd.Series(target).rank(method="first").to_numpy()
    labels = np.clip(np.ceil(ranks / n * 5) - 1, 0, 4)
    k = min(50, n)
    discounts = np.log2(np.arange(2, k + 2))
    ideal = np.sort(labels)[::-1][:k]
    denominator = float(np.sum(ideal / discounts))
    return float(np.sum(labels[order[:k]] / discounts) / denominator) if denominator > 0 else None


def evaluate_portfolio(frame, *, score_col="score", active_col="raw_active_return",
                       stock_col=None, target_col="active_return_train",
                       buffer=.20, rank_weighted=False, cost_bps=COST_BPS,
                       buffer_by_date=None):
    """Score a chronological series without filtering holdings on outcomes.

    Week t's target weights trade at t-1's feature snapshot. The joined raw
    outcome return at t measures that selection. For final OOS scoring, the
    caller supplies next-week realized returns on score week t instead.
    """
    rows, previous, previous_date = [], None, None
    for date, group in frame.groupby("week_date", sort=True):
        g = group.sort_values("ticker").reset_index(drop=True)
        tickers, scores = g.ticker.to_numpy(dtype=str), g[score_col].to_numpy(dtype=float)
        threshold = (buffer_by_date[date] if buffer_by_date is not None else buffer)
        weights, predicted_q1 = target_weights(tickers, scores, previous, buffer=threshold,
                                               rank_weighted=rank_weighted)
        trades = set(weights) | set(previous or {})
        consecutive = previous is not None and (date - previous_date).days == 7
        trade_dollars = sum(abs(weights.get(t, 0) - (previous or {}).get(t, 0)) for t in trades)
        if not consecutive:
            trade_dollars = sum(weights.values())
        cost = cost_bps / 10000 * trade_dollars
        active = g.set_index("ticker")[active_col]
        observed = active.reindex(list(weights))
        complete = bool(np.isfinite(observed.to_numpy(dtype=float)).all())
        gross_active = float(sum(weights[t] * active[t] for t in weights)) if complete else np.nan
        gross_stock = None
        if stock_col:
            stock = g.set_index("ticker")[stock_col].reindex(list(weights))
            gross_stock = (float(sum(weights[t] * stock[t] for t in weights))
                           if np.isfinite(stock.to_numpy(dtype=float)).all() else np.nan)
        target = g[target_col].to_numpy(dtype=float) if target_col in g else None
        capture, rank_ic, ndcg50 = None, None, None
        if target is not None and np.isfinite(target).all():
            order = np.lexsort((tickers, -target))
            actual_q1 = set(tickers[order[:int(np.ceil(.20 * len(g)))]] )
            capture = len(actual_q1 & set(weights)) / len(actual_q1)
            rank_ic = float(pd.Series(target).rank().corr(pd.Series(scores).rank())) if np.ptp(scores) and np.ptp(target) else None
            ndcg50 = _ndcg_at_50(target, scores, tickers)
        rows.append(dict(week_date=date, holdings=len(weights), gross_stock_return=gross_stock,
                         gross_active_return=gross_active,
                         net_stock_return=(gross_stock-cost if gross_stock is not None and np.isfinite(gross_stock) else None),
                         net_active_return=(gross_active-cost if complete else None),
                         transaction_cost=cost, turnover=trade_dollars/2,
                         q1_capture=capture, rank_ic=rank_ic, ndcg_at_50=ndcg50,
                         predicted_q1_count=len(predicted_q1), complete=complete))
        previous, previous_date = weights, date
    return pd.DataFrame(rows)


def portfolio_metrics(weekly: pd.DataFrame) -> dict:
    """Arithmetic annualization, net active IR, and compounded stock drawdown."""
    observed = weekly.loc[weekly.complete].copy()
    if observed.empty:
        raise ValueError("No complete realized portfolio weeks")
    active = observed.net_active_return.to_numpy(dtype=float)
    std = np.std(active, ddof=1) if len(active) > 1 else np.nan
    result = dict(return_weeks=len(observed),
                  annualized_stock_return=float(52 * observed.gross_stock_return.mean()) if observed.gross_stock_return.notna().all() else None,
                  annualized_net_stock_return=float(52 * observed.net_stock_return.mean()) if observed.net_stock_return.notna().all() else None,
                  annualized_active_return=float(52 * observed.gross_active_return.mean()),
                  net_active_return=float(52 * active.mean()),
                  active_ir=float(np.sqrt(52) * active.mean() / std) if std > 0 else None,
                  net_cumulative_active_return=float(active.sum()),
                  q1_capture=float(observed.q1_capture.mean()) if observed.q1_capture.notna().any() else None,
                  rank_ic=float(observed.rank_ic.mean()) if observed.rank_ic.notna().any() else None,
                  ndcg_at_50=float(observed.ndcg_at_50.mean()) if observed.ndcg_at_50.notna().any() else None,
                  turnover=float(weekly.turnover.iloc[1:].mean()) if len(weekly) > 1 else None,
                  annualized_transaction_cost=float(52 * observed.transaction_cost.mean()),
                  total_transaction_cost=float(observed.transaction_cost.sum()))
    if observed.net_stock_return.notna().all():
        wealth = np.cumprod(1 + observed.net_stock_return.to_numpy(dtype=float))
        result["max_drawdown"] = float(np.min(wealth / np.maximum.accumulate(np.r_[1., wealth])[1:] - 1))
        result["net_stock_cagr"] = float(wealth[-1] ** (52 / len(wealth)) - 1)
    else:
        result["max_drawdown"] = result["net_stock_cagr"] = None
    result["yearly_performance"] = {
        str(year): dict(weeks=len(group), annualized_stock_return=float(52*group.gross_stock_return.mean())
                        if group.gross_stock_return.notna().all() else None,
                        annualized_net_stock_return=float(52*group.net_stock_return.mean())
                        if group.net_stock_return.notna().all() else None,
                        annualized_active_return=float(52*group.gross_active_return.mean()),
                        net_active_return=float(52*group.net_active_return.mean()),
                        active_ir=(float(np.sqrt(52)*group.net_active_return.mean()/group.net_active_return.std(ddof=1))
                                   if len(group)>1 and group.net_active_return.std(ddof=1)>0 else None))
        for year, group in observed.groupby(observed.week_date.dt.year)
    }
    return result


def q1_subbuckets(frame, *, score_col="score", active_col="raw_active_return", stock_col=None):
    """Five equal count rank slices inside the predicted top quintile."""
    rows = []
    for date, group in frame.groupby("week_date", sort=True):
        g = group.sort_values([score_col, "ticker"], ascending=[False, True])
        q1 = g.iloc[:int(np.ceil(.20 * len(g)))].copy()
        bucket = np.arange(len(q1)) * 5 // len(q1) + 1
        for value in range(1, 6):
            chunk = q1.iloc[np.flatnonzero(bucket == value)]
            rows.append(dict(week_date=date, subbucket=value, holdings=len(chunk),
                             active_return=float(chunk[active_col].mean()) if chunk[active_col].notna().all() else None,
                             stock_return=(float(chunk[stock_col].mean()) if stock_col and chunk[stock_col].notna().all() else None)))
    return pd.DataFrame(rows)
