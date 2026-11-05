"""The released tables must reproduce the numbers stated in the manuscript."""
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

D = os.path.join(os.path.dirname(__file__), "..", "data", "derived")


def test_family_c_counts():
    c = pd.read_csv(os.path.join(D, "stats_C_india_label_budget.csv"))
    assert len(c) == 36
    assert int(c.sig_raw.sum()) == 28 and int(c.sig_bh.sum()) == 27
    assert (c.loc[c.sig_raw, "delta"] < 0).all()
    anyf = c[c.cell.str.contains("abnormal")]
    hem = c[c.cell.str.contains("hemorrhag")]
    assert (anyf.delta < 0).all() and int(anyf.sig_bh.sum()) == 17
    assert int(hem.sig_bh.sum()) == 10
    assert int((hem[~hem.sig_bh].delta > 0).sum()) == 3


def test_sign_test():
    c = pd.read_csv(os.path.join(D, "stats_C_india_label_budget.csv"))
    s = pd.read_csv(os.path.join(D, "stats_C_sign_test.csv")).iloc[0]
    k = int((c.delta < 0).sum())
    assert k == int(s.n_favour_classical) == 33
    assert abs(binomtest(k, 36, 0.5).pvalue - s.binomial_p_two_sided) < 1e-12


def test_bh_q_values_recompute():
    from qct3d.models import benjamini_hochberg
    for f in ("stats_C_india_label_budget.csv", "stats_A_quantum_vs_classical.csv", "stats_B_placement.csv"):
        t = pd.read_csv(os.path.join(D, f))
        assert np.allclose(benjamini_hochberg(t.p.values)[1], t.q.values, rtol=1e-9), f


def test_equivalence_family_a():
    e = pd.read_csv(os.path.join(D, "stats_A_equivalence.csv"))
    assert len(e) == 12 and int(e.equivalent.sum()) == 9
    assert set(e.loc[~e.equivalent, "cell"]) == {"B2 Q2", "B2 Q4", "B3 Q2"}


def test_table1_matches_analysis_sets():
    t = pd.read_csv(os.path.join(D, "table1_cohorts.csv")).set_index("Characteristic")
    assert t.loc["Series analysed, n"].tolist() == ["3,889", "1,200"]
    assert t.loc["Patients, n", "External cohort (Nellore)"] == "1,178"
    assert t.loc["Hemorrhage, n (%)"].tolist() == ["1,598 (41.1)", "94 (7.8)"]
    assert t.loc["Any reported finding, n (%)", "External cohort (Nellore)"] == "859 (71.6)"
