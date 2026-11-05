import numpy as np
import pytest

pytest.importorskip("pennylane")
torch = pytest.importorskip("torch")
from sklearn.metrics import roc_auc_score  # noqa: E402

from qct3d.models import (QUANTUM_FAMILIES, AdapterModel, Reducer, benjamini_hochberg,  # noqa: E402
                          fast_auroc, match_hidden_width, predict_proba, train_adapter)
from qct3d.protocol import make_holdout_split, run_arm  # noqa: E402


def _data(n=240, d=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype("float32")
    y = (X[:, :3].sum(1) + rng.standard_normal(n) > 1.0).astype(int)
    return X, y


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.random(500), 2)
    assert abs(fast_auroc(y, p) - roc_auc_score(y, p)) < 1e-12


def test_bh_matches_definition():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rej, q = benjamini_hochberg(p)
    m = len(p)
    ref = np.minimum.accumulate((p * m / np.arange(1, m + 1))[::-1])[::-1]
    assert np.allclose(q, np.minimum(ref, 1))
    assert rej.sum() == 2


def test_classical_control_is_parameter_matched():
    h, err, p = match_hidden_width(4, 41)
    assert err <= 0.10 and abs(p - 41) / 41 <= 0.10


def test_encoder_norm_adds_no_parameters_and_bounds_preactivation():
    X, y = _data()
    red = Reducer("trainable", X.shape[1], 4, seed=0).fit(X, y, seed=0)
    n0 = sum(p.numel() for p in red.parameters() if p.requires_grad)
    red.set_encoder_norm(True)
    assert sum(p.numel() for p in red.parameters() if p.requires_grad) == n0


@pytest.mark.parametrize("family", ["Q1_sel", "Q2_reupload", "Q4_ttn"])
def test_quantum_arm_trains(family):
    X, y = _data()
    red = Reducer("trainable", X.shape[1], 4, seed=0).fit(X[:180], y[:180], seed=0)
    m = AdapterModel("quantum", red, 4, 3, family=family, diff_method="backprop", init_std=0.01, seed=0)
    before = m.q.weights.detach().clone()
    Z = red.transform_np(X[:180])
    train_adapter(m, Z, y[:180], epochs=5, lr=0.05, batch=64, beta=1e-4, weight_decay=1e-3,
                  alpha=0.25, gamma=2.0, seed=0)
    assert float((m.q.weights.detach() - before).abs().max()) > 1e-6
    p = predict_proba(m, red.transform_np(X[180:]))
    assert np.isfinite(p).all() and p.shape == (60,)


def test_run_arm_is_deterministic():
    X, y = _data()
    cfg = dict(n_qubits=4, depth=3, diff_method="backprop", init_std=0.01, epochs=3, epochs_spatial=1,
               batch=64, batch_spatial=32, lr=0.05, angle_l2_beta=1e-4, projection_weight_decay=1e-3,
               focal_alpha=0.25, focal_gamma=2.0, predict_batch=256)
    a = run_arm("quantum", X[:180], y[:180], X[180:], "Q1_sel", "trainable", cfg, seed=3)[0]
    b = run_arm("quantum", X[:180], y[:180], X[180:], "Q1_sel", "trainable", cfg, seed=3)[0]
    assert np.array_equal(a, b)


def test_holdout_split_groups_patients():
    y = np.array([0, 1] * 50)
    groups = np.arange(100) // 2
    tr, te = make_holdout_split(y, groups, seed=7)
    assert not set(groups[tr]) & set(groups[te])


def test_families_registered():
    assert {"Q1_sel", "Q2_reupload", "Q3_quanv", "Q4_ttn"} <= set(QUANTUM_FAMILIES)
