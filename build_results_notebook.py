"""Export the best XGBoost research result and a self-contained results notebook.

Reads completed local artifacts; never trains models or contacts AWS/Kafka.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


def build(source: Path, destination=Path("reports/best_model")):
    destination.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(source / "comparison.csv")
    best = comparison.loc[comparison.model.eq("xgboost") &
                          comparison.status.eq("protected_selection")].iloc[0].to_dict()
    weekly = pd.read_parquet(source / "weekly_portfolios.parquet")
    weekly = weekly.loc[weekly.model.eq("xgboost") &
                        weekly.status.eq("protected_selection") & weekly.complete].copy()
    weekly = weekly.drop(columns=["model", "status"]).sort_values("week_date")
    weekly.to_csv(destination / "weekly_returns.csv", index=False)
    yearly = pd.read_csv(source / "yearly_performance.csv")
    yearly = yearly.loc[yearly.model.eq("xgboost") &
                        yearly.status.eq("protected_selection")].drop(columns=["model", "status"])
    yearly.to_csv(destination / "yearly_performance.csv", index=False)
    candidates = pd.read_csv(source / "candidate_validation.csv")
    selected = candidates.loc[candidates.model.eq("xgboost") & candidates.selected]
    selected[["week_date", "candidate", "params", "changed", "gate_checks"]].to_csv(
        destination / "weekly_configurations.csv", index=False)
    summary = dict(model="XGBoost regression with protected chronological selection",
                   score_start=str(weekly.week_date.min().date()),
                   last_realized_score_date=str(weekly.week_date.max().date()),
                   data_available_through="2026-02-18", window_weeks=260,
                   cost_bps_per_dollar_traded=5, seed=42,
                   selected_challenger_dates=int(selected.candidate.ne("incumbent").sum()),
                   configuration_counts=selected.candidate.value_counts().to_dict(),
                   metrics={k:v for k,v in best.items() if k not in
                            ("model", "status", "ir_change_vs_incumbent",
                             "annual_net_active_change_vs_incumbent")},
                   definitions={"annualized_returns":"52 times mean weekly return",
                                "active_return":"Stock return minus contemporaneous sector mean return",
                                "active_ir":"sqrt(52) times mean net active return / sample std",
                                "turnover":"Half the sum of absolute weekly target-weight changes",
                                "cost":"0.0005 times sum of absolute target-weight changes; initial entry charged",
                                "drawdown":"Maximum drawdown of compounded net stock returns"})
    (destination / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    fixtures = json.loads(Path("tests/fixtures/incumbents.json").read_text())
    config = dict(estimator="XGBRegressor", objective="reg:squarederror", tree_method="hist",
                  device="cpu", random_state=42, n_jobs=1,
                  **fixtures["xgboost"]["best_params"])
    config = dict(base_parameters=config, reg_lambda="Unspecified: original XGBoost default 1",
                  rolling_weeks=260, ensemble_seeds=[42],
                  challengers=[{"max_depth":2}, {"min_child_weight":20}, {"reg_lambda":2.0}],
                  selection={"folds":4,"gap_weeks":1,"cost_bps":5,"min_ir_gain":0.10,
                             "min_annual_net_active_gain":0.005,"min_fold_ir_wins":3,
                             "max_fold_ir_loss":0.25,"block_bootstrap_weeks":4,
                             "bootstrap_replicates":500,"required_gain_10th_percentile":"> 0"},
                  note="Each challenger changes one parameter. The reported result uses dated choices, not one globally tuned parameter set.")
    Path("configs").mkdir(exist_ok=True)
    Path("configs/best_xgboost.json").write_text(json.dumps(config, indent=2)+"\n")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(weekly.week_date, weekly.net_active_return.cumsum(), color="#176B87")
    ax.axhline(0,color="black",lw=.6)
    ax.set_title("XGBoost Q1: cumulative net active return (5 bps per dollar traded)")
    ax.set_ylabel("Sum of weekly net active returns")
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.2)
    fig.tight_layout(); image = destination / "cumulative_net_active.png"
    fig.savefig(image,dpi=150); plt.close(fig)
    context = {}
    cells = [dict(cell_type="markdown",metadata={},source=[
        "# ProjectAlpha — current best model results\n\n",
        "Work in progress. These saved research results use an equal-weight top-20% XGBoost portfolio, a 260-week rolling window, seed 42, and 5 bps per dollar traded. The configuration varies only when a conservative chronological validation gate accepts a challenger.\n\n",
        "All inputs to this notebook are included under `reports/best_model/`. Running it does not retrain models or contact cloud services. See `README.md` for architecture, definitions and limitations.\n"])]
    code_cells = [
        "from pathlib import Path\nimport json\nimport pandas as pd\nROOT = Path('reports/best_model')\nsummary = json.loads((ROOT/'summary.json').read_text())\nprint(json.dumps(summary, indent=2))\n",
        "config = json.loads(Path('configs/best_xgboost.json').read_text())\nprint(json.dumps(config, indent=2))\n",
        "yearly = pd.read_csv(ROOT/'yearly_performance.csv')\nprint(yearly.round(4).to_string(index=False))\nprint('2026 is a partial year; annualized values are not full-year realized returns.')\n",
        "choices = pd.read_csv(ROOT/'weekly_configurations.csv')\nprint(choices.candidate.value_counts().to_string())\nprint('All dated parameters and gate checks: reports/best_model/weekly_configurations.csv')\n"
    ]
    for i,code in enumerate(code_cells,1):
        captured=io.StringIO()
        with contextlib.redirect_stdout(captured): exec(code,context)
        cells.append(dict(cell_type="code",metadata={},execution_count=i,
                          source=code.splitlines(keepends=True),outputs=[dict(
                              output_type="stream",name="stdout",text=captured.getvalue())]))
    cells.append(dict(cell_type="code",metadata={},execution_count=5,
                      source=["from IPython.display import Image, display\n",
                              "display(Image(filename=str(ROOT/'cumulative_net_active.png')))\n"],
                      outputs=[dict(output_type="display_data",metadata={},data={
                          "image/png":base64.b64encode(image.read_bytes()).decode()})]))
    notebook=dict(cells=cells,nbformat=4,nbformat_minor=4,metadata={
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"}})
    Path("model_results.ipynb").write_text(json.dumps(notebook,indent=1)+"\n")
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("artifacts/ml/continuous_260w_v3_incumbent_protected"))
    args=parser.parse_args()
    print(json.dumps(build(args.source),indent=2))
