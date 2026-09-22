# Focused V3: incumbent-protected XGBoost and LightGBM

This experiment uses only the existing panel, twelve features, normalized
training target, 260-week rolling window, current-week inference timing,
equal-weight exact Q1, and 5 bps per actual dollar traded. It tests no new
model type, retention buffer, rank weighting, or recency weighting.

## Protected incumbents

- **XGBoost:** the completed V1 model at each as-of date, read from
  `artifacts/ml/continuous_260w_v1/regression/as_of=YYYY-MM-DD/metadata.json`.
  Its selected configuration is identical on all 216 dates: rate 0.025,
  depth 3, child weight 10, 50 trees, 0.8 row and column fractions, CPU
  histogram algorithm, squared-error objective, and seed 42. Its saved
  manifest does not explicitly specify L2, so the exact original library
  default is retained for the incumbent.
- **LightGBM:** the completed V2 model at each as-of date, read from
  `artifacts/ml/continuous_260w_v2/lightgbm/as_of=YYYY-MM-DD/metadata.json`.
  There is **no single static LightGBM configuration** to freeze: the saved
  rolling backtest chose different parameters and tree counts on different
  dates. This experiment protects that exact per-date `best_params` record,
  including learning rate, leaves, leaf minimum, L2, truncation, and final
  tree count. It retains the original three seeds 42, 73, and 107 and their
  within-week percentile-rank averaging.

`v3_incumbent_replay.py` independently loads each saved pickle, scores the
current-week snapshot, and rebuilds each model from the recorded parameters,
training window, and seeds. It compares rebuilt scores with saved pickle
scores and archived backtest scores at 1e-11 absolute / 1e-10 relative
tolerance. It then recomputes the exact equal-weight Q1 portfolio at 5 bps
and checks every reported metric. The search aborts unless replay passes.

## Additive search and protection gate

Every date's candidate list starts with its **unchanged incumbent**. All
challengers keep that date's incumbent tree count, learning rate, sampling,
feature order, target, estimator type, and seeds. They each change exactly
one parameter:

| Model | Conservative challenger changes |
|---|---|
| XGBoost | Depth 3→2; child weight 10→20; or explicitly set L2 to 2 instead of the incumbent's library default |
| LightGBM | Halve the incumbent's leaf count (minimum 5); multiply its minimum leaf size by 1.5; or add 2 to its L2 penalty |

These are generated around the actual per-date incumbent configuration, so
the exact LightGBM challenger values also vary by date. The complete values
are written to `candidate_validation.csv` and each date's `selection.json`.

Four expanding chronological validation folds preserve the original one-week
gap. All four candidates for a model/date use identical fit and evaluation
weeks. Weekly model scores form an equal-weight top-20% portfolio, charged
5 bps times actual traded dollars including initial entry. The primary
metric is net active IR; annualized net active return, calendar-year
performance, turnover, maximum net stock drawdown, Q1 capture, Rank IC, and
NDCG@50 are retained as diagnostics.

A challenger can replace the incumbent only if **all five** checks pass on
the chronological validation weeks:

1. Aggregate net active IR improves by at least 0.10.
2. Annualized net active return improves by at least 0.5 percentage points.
3. Net active IR improves in at least three of the four folds.
4. No fold loses more than 0.25 IR relative to the incumbent.
5. The 10th percentile of a deterministic, four-week-block bootstrap of the
   paired annualized net active-return improvement is above zero.

If multiple challengers pass, choose the one with highest validation net IR,
then higher net active return. Otherwise the exact saved incumbent pickle is
used at inference. No OOS return is used to choose a candidate. Since the
LightGBM incumbent configuration itself came from the earlier V2 search on
the same 260-week window, these internal fold diagnostics are a conservative
replacement gate, **not an independent estimate of future performance**.
The continuous OOS comparison is the final performance test.

## Run and inspect

```bash
.venv/bin/python v3_incumbent_replay.py
.venv/bin/python v3_incumbent_backtest.py --workers 4
# Re-score and rebuild reports from fitted selections:
.venv/bin/python v3_incumbent_backtest.py --score-only
.venv/bin/python build_results_notebook.py
```

Outputs are isolated in
`artifacts/ml/continuous_260w_v3_incumbent_protected/`. The replay audit,
all candidate parameter dictionaries, fold-level validation rows, reasons
for selection, selected weekly scores, continuous portfolio returns, yearly
results, and OOS incumbent-versus-selected comparison are saved there.
The portable best-model results walkthrough is `model_results.ipynb`.

## Completed run (2022-01-05 through 2026-02-18)

All **432 dated incumbent pickles** were replayed and independently refit
before challenger tuning. Scores matched the archived backtests at absolute
tolerance `1e-11` and relative tolerance `1e-10`; portfolio metrics matched
at absolute tolerance `1e-11` and relative tolerance `1e-9`. The audit is in
`incumbent_replay_audit.json` under the focused output directory.

The XGBoost incumbent's exact saved `best_params` are:

```json
{"colsample_bytree": 0.8, "learning_rate": 0.025, "max_depth": 3, "min_child_weight": 10, "n_estimators": 50, "subsample": 0.8}
```

It uses the V1 `reg:squarederror` CPU histogram fit with seed 42. The V2
LightGBM incumbent has 136 distinct **full** dated `best_params` records
(including tree counts); all are preserved verbatim in `candidate_validation.csv`
where `candidate=incumbent`, with their original manifest paths and seeds in
each `selection.json`. For example, the 2022-01-05 incumbent has 22 trees,
learning rate 0.05, 31 leaves, depth 5, minimum leaf size 300, L2 penalty 1,
`lambdarank` with truncation 30, and seeds 42/73/107. This example is not
substituted for other dates' configurations.

| Model | Selection | Challenger dates / 216 | Annualized gross active | Annualized net active | Net active IR | Q1 capture | Weekly turnover | Net stock max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | Exact V1 incumbent | 0 | 6.46% | 3.78% | 0.608 | 24.50% | 51.64% | −19.25% |
| XGBoost | Protected search | 14 | 6.63% | 3.94% | 0.632 | 24.49% | 51.65% | −19.25% |
| LightGBM | Exact V2 incumbent | 0 | 4.57% | 2.32% | 0.336 | 26.27% | 43.35% | −22.62% |
| LightGBM | Protected search | 21 | 3.88% | 1.62% | 0.235 | 26.20% | 43.57% | −22.60% |

The XGBoost changes added **0.023 IR** and **0.16 percentage points** of
annualized net active return OOS, a small improvement. The LightGBM changes
lost **0.102 IR** and **0.71 percentage points** of annualized net active
return OOS. Its 21 accepted challengers passed the historical validation
gate, but the gate did not predict OOS improvement; the original V2 LightGBM
incumbent remains the stronger observed result. This OOS finding is reported
without retrospectively changing any historical selection.

`candidate_validation.csv` contains **all 1,728 exact candidate parameter
records** and aggregate validation diagnostics. `fold_validation.csv`
contains **all 6,912 fold-level rows** (four folds per candidate).
`selections.csv` records each dated choice and reason;
`yearly_performance.csv` gives all model/strategy calendar-year comparisons.
`comparison.csv`, `weekly_portfolios.parquet`, and the results notebook give
the full OOS metrics and cumulative-return chart. Transaction costs are
5 bps on actual dollars traded for every row.
