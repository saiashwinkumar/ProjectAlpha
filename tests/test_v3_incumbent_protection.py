import json
from pathlib import Path

import numpy as np
import pandas as pd

from v3_incumbent_search import _gate, candidate_specs


def test_real_saved_incumbents_are_unchanged_candidate_zero():
    fixtures = json.loads((Path(__file__).parent / "fixtures/incumbents.json").read_text())
    for model, manifest in fixtures.items():
        specs = candidate_specs(model, manifest)
        base = manifest["best_params"]
        assert specs[0] == dict(name="incumbent", params=base, changed={})
        assert len(specs) == 4
        for challenger in specs[1:]:
            changed = {key for key in set(base) | set(challenger["params"])
                       if base.get(key) != challenger["params"].get(key)}
            assert changed == set(challenger["changed"])
            assert len(changed) == 1
            assert challenger["params"]["n_estimators"] == base["n_estimators"]


def test_protection_gate_rejects_weak_gain_and_accepts_consistent_gain():
    dates = pd.date_range("2018-01-03", periods=208, freq="W-WED")
    fold = np.repeat(np.arange(1, 5), 52)
    base_weekly = pd.DataFrame({"week_date":dates, "fold":fold,
                                "net_active_return":np.sin(np.arange(208))*.01})
    good_weekly = base_weekly.copy()
    good_weekly.net_active_return += .002
    inc = dict(metrics=dict(active_ir=.2, net_active_return=.01,
                            yearly_performance={"2018":dict(net_active_return=.01)}),
               folds=[dict(active_ir=.2)]*4, weekly=base_weekly)
    weak = dict(metrics=dict(active_ir=.23, net_active_return=.012,
                             yearly_performance={"2018":dict(net_active_return=.012)}),
                folds=[dict(active_ir=x) for x in (.25, .24, .20, .19)],
                weekly=good_weekly)
    strong = dict(metrics=dict(active_ir=.4, net_active_return=.03,
                               yearly_performance={"2018":dict(net_active_return=.03)}),
                  folds=[dict(active_ir=.35)]*4, weekly=good_weekly)
    assert not _gate(inc, weak, dates[-1])["eligible"]
    assert _gate(inc, strong, dates[-1])["eligible"]
