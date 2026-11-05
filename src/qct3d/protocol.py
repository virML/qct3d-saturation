"""Patient-grouped splits, the three arms, the paired comparison, resumable result logs and the
depolarising-noise evaluation.

Moved verbatim from cell 6 of notebooks 02, 04 and 06 (identical in all three). In the notebooks
this cell ran in the same namespace as the model cell; here those names are imported explicitly.
"""
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .models import (QUANTUM_FAMILIES, AdapterModel, Reducer, SpatialAdapter, benjamini_hochberg,  # noqa: F401
                     calibration_metrics, cluster_bootstrap_ci, fast_auprc, fast_auroc, match_hidden_width,
                     paired_diff_test, pick_thresholds, predict_proba, threshold_metrics, train_adapter,
                     ttn_blocks)

# 6. The protocol: patient-grouped splits, the three arms, and the paired comparison
#
#   1. hold out HOLDOUT_FRAC of PATIENTS (not rows), label-stratified
#   2. StratifiedGroupKFold on the rest; out-of-fold predictions for every arm
#   3. thresholds chosen from OUT-OF-FOLD predictions only -- choosing them on the
#      holdout makes every sensitivity below optimistically biased and nothing
#      complains
#   4. each arm refitted on the pool, scored once on the holdout, with
#      patient-clustered bootstrap intervals
#   5. quantum vs its matched control: PAIRED bootstrap on the same patients
import json
import time

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

ARMS = ("quantum", "classical", "linear_probe")


# ==================================================================================== #
# splits
# ==================================================================================== #
def make_holdout_split(y, groups, holdout_frac=0.2, seed=12345):
    """Patient-grouped AND label-stratified.

    GroupShuffleSplit alone ignores y, so at these prevalences it regularly returns a
    holdout with zero positives -- at which point the target silently disappears from the
    results with a one-line skip. StratifiedGroupKFold stratifies on y while keeping every
    patient wholly on one side.
    """
    y = np.asarray(y).astype(int)
    n_splits = max(2, int(round(1.0 / max(holdout_frac, 1e-6))))
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pool_idx, hold_idx = next(iter(sgkf.split(np.zeros(len(y)), y, groups)))
        if y[hold_idx].sum() > 0 and y[pool_idx].sum() > 0:
            return pool_idx, hold_idx
    except Exception:
        pass
    gss = GroupShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
    return next(iter(gss.split(np.zeros(len(y)), y, groups)))


# ==================================================================================== #
# one arm
# ==================================================================================== #
def run_arm(arm, Xtr, ytr, Xte, family, reducer_mode, cfg, seed,
            Mtr=None, Mte=None, epochs=None, return_model=False):
    """arm in {'quantum','classical','linear_probe'}.

    Mtr/Mte are the spatial feature MAPS, required only when the family is a
    deep-placement one (Q3). Returns (probs, n_params, n_tail_params, extra).
    """
    ytr = np.asarray(ytr).astype(int)
    classes = np.unique(ytr)
    if len(classes) < 2:
        # A single-class training fold teaches nothing, and the three arms fail differently
        # on it: LogisticRegression raises, the supervised reducer correlates against a
        # constant, the adapter trains to a constant. Predict the seen class, mark it
        # degenerate, and let the NaN propagate honestly instead of inventing an AUROC.
        c = float(classes[0])
        p = np.full(len(Xte), min(max(c, 1e-6), 1 - 1e-6), dtype=float)
        return p, 0, 0, {"degenerate": f"training fold had only class {int(classes[0])}"}

    if arm == "linear_probe":
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
        lr.fit(sc.transform(Xtr), ytr)
        return lr.predict_proba(sc.transform(Xte))[:, 1], int(Xtr.shape[1] + 1), 0, {}

    nq, depth = cfg["n_qubits"], cfg["depth"]
    spatial = QUANTUM_FAMILIES[family]["spatial"]
    extra = {}

    if spatial:
        if Mtr is None or Mte is None:
            raise ValueError(f"family {family} is a deep-placement arm and needs spatial "
                             f"feature maps, but none were supplied. Re-run Phase A with "
                             f"SAVE_SPATIAL_MAP=True.")
        c_in = Mtr.shape[1]
        tail = SpatialAdapter("quantum", c_in, nq, depth, family=family,
                              diff_method=cfg["diff_method"], seed=seed).tail_params()
        hidden, _err, _cp = match_hidden_width(nq, tail)
        model = SpatialAdapter(arm, c_in, nq, depth, hidden=hidden, family=family,
                               diff_method=cfg["diff_method"],
                               init_std=cfg["init_std"], seed=seed)
        # STANDARDISE THE MAPS, ON THE TRAINING FOLD ONLY.
        #
        # This is not cosmetic; without it the whole arm is dead. The cached maps are
        # post-ReLU backbone activations: non-negative, with a large per-channel mean that
        # every series shares. SpatialAdapter feeds them straight into a 1x1x1 convolution
        # and then (pi/2)*tanh(.), and that shared mean dominates the projection -- pre-tanh
        # magnitudes reach ~33 where signed inputs give ~7, tanh saturates, and every series
        # lands on the same +/-pi/2 angle pattern. The output is then constant, which scores
        # AUROC exactly 0.5000 and AUPRC exactly the prevalence, for the quantum arm AND for
        # its classical control, because both sit downstream of the same saturated angles.
        # That is precisely what the first full run produced: 0.500 in all 15 Q3 folds.
        #
        # Q1/Q2/Q4 never hit this because they go through Reducer, which centres and scales
        # (pca/pca_supervised) or can learn a bias that absorbs the mean (trainable). Q3's
        # path had neither: proj.bias is zero-initialised and there was no scaler.
        #
        # Statistics come from Mtr alone -- computing them over Mtr and Mte together would
        # leak the test fold into the transform.
        _mu = Mtr.mean(axis=(0, 2, 3, 4), keepdims=True)
        _sd = Mtr.std(axis=(0, 2, 3, 4), keepdims=True)
        # A deep post-ReLU map has many channels that essentially never fire. Dividing
        # those by (sd + 1e-6) turns numerical dust into values of order 1e7 -- measured --
        # which is a second, independent route to a saturated encoder. Near-constant
        # channels are left alone rather than amplified, and the result is clipped so no
        # single outlier can do it either.
        _sd = np.where(_sd < 1e-4, 1.0, _sd)
        Atr = np.clip((Mtr - _mu) / _sd, -8.0, 8.0).astype(np.float32)
        Ate = np.clip((Mte - _mu) / _sd, -8.0, 8.0).astype(np.float32)
        extra["map_norm"] = {"pre_mean": float(Mtr.mean()), "pre_std": float(Mtr.std()),
                             "n_flat_channels": int((_sd == 1.0).sum()),
                             "post_absmax": float(np.abs(Atr).max())}
        prefixes = ("proj.",)
        ep = epochs if epochs is not None else cfg["epochs_spatial"]
        batch = cfg["batch_spatial"]
    else:
        red = Reducer(reducer_mode, Xtr.shape[1], nq, seed=seed).fit(Xtr, ytr, seed=seed)
        probe = AdapterModel("quantum", Reducer(reducer_mode, Xtr.shape[1], nq), nq, depth,
                             family=family, diff_method=cfg["diff_method"])
        tail = probe.tail_params()
        hidden, _err, _cp = match_hidden_width(nq, tail)
        model = AdapterModel(arm, red, nq, depth, family=family, hidden=hidden,
                             diff_method=cfg["diff_method"], init_std=cfg["init_std"],
                             seed=seed)
        Atr, Ate = red.transform_np(Xtr), red.transform_np(Xte)
        prefixes = ("reducer.",)
        ep = epochs if epochs is not None else cfg["epochs"]
        batch = cfg["batch"]
        extra["pca_explained_variance"] = red.explained
        extra["reducer_obj"] = red

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tail = model.tail_params()
    train_adapter(model, Atr, ytr, epochs=ep, lr=cfg["lr"], batch=batch,
                  beta=cfg["angle_l2_beta"], weight_decay=cfg["projection_weight_decay"],
                  alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"], seed=seed,
                  reducer_prefixes=prefixes)
    probs = predict_proba(model, Ate, batch=cfg.get("predict_batch", 256))
    extra["matched_hidden"] = hidden
    extra["quantum_tail_params"] = tail
    if arm == "quantum":
        extra["angle_max_abs"] = float(model.q.weights.detach().abs().max())
        extra["state_dict"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if return_model:
        extra["model"] = model
    return probs, n_par, n_tail, extra


# ==================================================================================== #
# one grid cell
# ==================================================================================== #
def run_cell(backbone_id, family, reducer_mode, target_name, X, y, groups, cfg,
             M=None, verbose=True, keep_model=False):
    """One (backbone, family, reducer, target) cell: CV + holdout + paired comparison."""
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    t_start = time.time()
    pool_idx, hold_idx = make_holdout_split(y, groups, cfg["holdout_frac"], cfg["seed"])
    Xp, yp, gp = X[pool_idx], y[pool_idx], groups[pool_idx]
    Xh, yh, gh = X[hold_idx], y[hold_idx], groups[hold_idx]
    Mp = M[pool_idx] if M is not None else None
    Mh = M[hold_idx] if M is not None else None

    if yh.sum() == 0 or yp.sum() == 0:
        if verbose:
            print(f"  SKIP: holdout has {int(yh.sum())} positives, pool {int(yp.sum())}.")
        return None
    if verbose:
        print(f"  pool {len(yp):,} ({int(yp.sum())} pos) / {len(set(gp)):,} patients   "
              f"holdout {len(yh):,} ({int(yh.sum())} pos) / {len(set(gh)):,} patients")

    pool_pos, pool_neg = int(yp.sum()), int(len(yp) - yp.sum())
    folds_used = min(int(cfg["cv_folds"]), pool_pos, pool_neg)
    oof = {a: np.full(len(yp), np.nan) for a in ARMS}
    fold_auroc = {a: [] for a in ARMS}

    if folds_used < 2:
        folds = []
        if verbose:
            print(f"  NOTE: no CV ({pool_pos} pos / {pool_neg} neg in pool). cv_* will be NaN.")
    else:
        sgkf = StratifiedGroupKFold(n_splits=folds_used, shuffle=True,
                                    random_state=cfg["seed"])
        folds = list(sgkf.split(Xp, yp, gp))

    for k, (tr, va) in enumerate(folds):
        bits = [f"    fold {k+1}/{folds_used}"]
        for a in ARMS:
            t0 = time.time()
            p, _, _, ex = run_arm(a, Xp[tr], yp[tr], Xp[va], family, reducer_mode, cfg,
                                  cfg["seed"] + k,
                                  Mtr=(Mp[tr] if Mp is not None else None),
                                  Mte=(Mp[va] if Mp is not None else None))
            oof[a][va] = p
            auc = fast_auroc(yp[va], p)
            fold_auroc[a].append(auc)
            bits.append(f"{a[:4]} {auc:.3f} ({time.time()-t0:.0f}s)")
        if verbose:
            print("  ".join(bits))

    rows, hold_p, fitted = [], {}, None
    for a in ARMS:
        t0 = time.time()
        p_hold, n_par, n_tail, extra = run_arm(a, Xp, yp, Xh, family, reducer_mode, cfg,
                                               cfg["seed"], Mtr=Mp, Mte=Mh,
                                               return_model=keep_model)
        hold_p[a] = p_hold
        if keep_model and a == "quantum":
            fitted = {"model": extra.get("model"), "reducer": extra.get("reducer_obj")}
        thrs = pick_thresholds(yp, oof[a], cfg["target_sensitivity"])
        au = cluster_bootstrap_ci(yh, p_hold, gh, fast_auroc, cfg["n_bootstrap"], cfg["seed"])
        ap = cluster_bootstrap_ci(yh, p_hold, gh, fast_auprc, cfg["n_bootstrap"], cfg["seed"])
        cal = calibration_metrics(yh, p_hold)
        vals = [v for v in fold_auroc[a] if v == v]
        row = {
            "config_sig": cfg["signature"], "backbone": backbone_id, "family": family,
            "family_label": QUANTUM_FAMILIES[family]["label"],
            "placement": QUANTUM_FAMILIES[family]["placement"],
            "dataset": cfg.get("dataset", "rsna-ich"), "target": target_name,
            "reducer": reducer_mode, "arm": a,
            "n_qubits": cfg["n_qubits"] if a != "linear_probe" else 0,
            "depth": cfg["depth"] if a == "quantum" else 0,
            "trainable_params": int(n_par), "adapter_tail_params": int(n_tail),
            "matched_hidden": extra.get("matched_hidden", np.nan),
            "feature_dim": int(X.shape[1]),
            "n_total": int(len(y)), "n_positive": int(y.sum()), "prevalence": float(y.mean()),
            "n_pool": int(len(yp)), "n_pool_pos": int(yp.sum()),
            "n_holdout": int(len(yh)), "n_holdout_pos": int(yh.sum()),
            "n_patients": int(len(set(groups))), "cv_folds": int(folds_used),
            "cv_auroc": fast_auroc(yp, oof[a]), "cv_auprc": fast_auprc(yp, oof[a]),
            "cv_auroc_fold_mean": float(np.mean(vals)) if vals else float("nan"),
            "cv_auroc_fold_std": float(np.std(vals)) if vals else float("nan"),
            "holdout_auroc": au[0], "holdout_auroc_lo": au[1], "holdout_auroc_hi": au[2],
            "holdout_auprc": ap[0], "holdout_auprc_lo": ap[1], "holdout_auprc_hi": ap[2],
            "holdout_auprc_baseline": float(yh.mean()),
            "holdout_auprc_lift": (ap[0] / yh.mean()) if yh.mean() > 0 else float("nan"),
            "brier": cal["brier"], "calibration_slope": cal["calibration_slope"],
            "calibration_intercept": cal["calibration_intercept"], "ece": cal["ece"],
            "pca_explained_variance": extra.get("pca_explained_variance", float("nan")),
            "angle_max_abs": extra.get("angle_max_abs", float("nan")),
            "fit_seconds": float(time.time() - t0),
            "cell_seconds": float(time.time() - t_start),
        }
        for nm, t in thrs.items():
            for mk, mv in threshold_metrics(yh, p_hold, t).items():
                row[f"{nm}__{mk}"] = mv
        rows.append(row)

    cmp_c = paired_diff_test(yh, hold_p["quantum"], hold_p["classical"], gh,
                             n_boot=cfg["n_bootstrap"], seed=cfg["seed"])
    cmp_p = paired_diff_test(yh, hold_p["quantum"], hold_p["linear_probe"], gh,
                             n_boot=cfg["n_bootstrap"], seed=cfg["seed"])
    for r in rows:
        if r["arm"] == "quantum":
            r.update({f"vs_classical_{k}": v for k, v in cmp_c.items()})
            r.update({f"vs_probe_{k}": v for k, v in cmp_p.items()})
    if verbose:
        print(f"    quantum vs matched classical : dAUROC {cmp_c['diff']:+.4f} "
              f"[{cmp_c['lo']:+.4f},{cmp_c['hi']:+.4f}] p={cmp_c['p_two_sided']:.4f}")
        print(f"    quantum vs linear probe      : dAUROC {cmp_p['diff']:+.4f} "
              f"[{cmp_p['lo']:+.4f},{cmp_p['hi']:+.4f}] p={cmp_p['p_two_sided']:.4f}")
    return rows, {"hold_p": hold_p, "yh": yh, "gh": gh, "pool_idx": pool_idx,
                  "hold_idx": hold_idx, "oof": oof,
                  "fitted": (fitted if keep_model else None)}


# ==================================================================================== #
# logging
# ==================================================================================== #
def primary_grid(df, reducer, target):
    """The CONFIRMATORY grid: one reducer, one target, one row per (backbone, family).

    Everything the finalist sweep adds -- the other reducers, the exploratory subtypes --
    is a different question asked of one cell, and mixing it into the grid double-counts
    that cell. Table 1, the RQ1 decomposition and the primary BH family all take this.

    `reducer` and `target` are REQUIRED rather than defaulting to the config globals: this
    cell is carried verbatim into nb3, which has a different configuration cell and does
    not define them. A default that silently resolves in one notebook and raises NameError
    in another is a trap with a long fuse.
    """
    d = df
    if "reducer" in d.columns:
        d = d[d["reducer"] == reducer]
    if "target" in d.columns:
        d = d[d["target"] == target]
    return d.copy()


def append_rows(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                for k, v in r.items()}) + "\n")


def load_results(path, config_sig=None):
    import pandas as pd
    rows = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if config_sig is not None and "config_sig" in df:
        df = df[df["config_sig"] == config_sig]
    ident = {"config_sig", "backbone", "family", "family_label", "placement", "dataset",
             "target", "reducer", "arm"}
    for c in df.columns:
        if c not in ident:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keys = [c for c in ["backbone", "family", "target", "reducer", "arm"] if c in df]
    return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def done_cells(path, config_sig):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("config_sig") == config_sig:
                done.add((r.get("backbone"), r.get("family"), r.get("target"),
                          r.get("reducer")))
    return done


# ==================================================================================== #
# RQ1 -- family-level variance decomposition
# ==================================================================================== #
def family_variance_decomposition(df, metric="cv_auroc_fold_mean", verbose=True):
    """Two-way decomposition of the grid: how much of the variation in generalisation is
    explained by the BACKBONE, by the QUANTUM FAMILY, and by their interaction.

    This is the RQ1 answer. Naming whichever single cell topped the leaderboard is a much
    weaker and more overfitting-prone claim than a family-level effect that holds across
    backbones -- which is the whole reason the grid is a grid and not a shortlist.

    Uses type-II-style sums of squares on a balanced design via direct means, which is
    exact when every (backbone, family) cell is present exactly once.
    """
    q = df[(df["arm"] == "quantum") & df[metric].notna()].copy()
    if q.empty or q["backbone"].nunique() < 2 or q["family"].nunique() < 2:
        if verbose:
            print("  not enough cells for a two-way decomposition "
                  f"({q['backbone'].nunique()} backbone(s), {q['family'].nunique()} famil(ies))")
        return None
    # THE DESIGN HAS TO BE BALANCED, and after the finalist sweep it is not: the sweep
    # re-runs ONE cell at other reducers and on subtype targets, so LB holds several rows
    # for that (backbone, family) and none for the others. Decomposing that attributes the
    # winning cell's repeated measurements to its FAMILY, which is exactly the quantity
    # RQ1 is trying to estimate. Caught here rather than trusted to the caller, because
    # the arithmetic below produces a plausible number either way.
    dup = q.groupby(["backbone", "family"]).size()
    if int(dup.max()) > 1:
        raise ValueError(
            f"the decomposition needs ONE row per (backbone, family) and got up to "
            f"{int(dup.max())} -- e.g. {dup.idxmax()}. Pass the confirmatory grid only "
            f"(one reducer, one target); the finalist sweep is a separate question. "
            f"`primary_grid(LB)` does this.")
    grand = q[metric].mean()
    bmeans = q.groupby("backbone")[metric].mean()
    fmeans = q.groupby("family")[metric].mean()
    nb, nf, n = len(bmeans), len(fmeans), len(q)
    ss_total = float(((q[metric] - grand) ** 2).sum())
    ss_b = float(sum(((q["backbone"] == b).sum()) * (m - grand) ** 2 for b, m in bmeans.items()))
    ss_f = float(sum(((q["family"] == f).sum()) * (m - grand) ** 2 for f, m in fmeans.items()))
    pred = q.apply(lambda r: bmeans[r["backbone"]] + fmeans[r["family"]] - grand, axis=1)
    ss_resid = float(((q[metric] - pred) ** 2).sum())
    ss_inter = max(ss_total - ss_b - ss_f - ss_resid, 0.0)
    out = {
        "metric": metric, "n_cells": int(n), "n_backbones": int(nb), "n_families": int(nf),
        "grand_mean": float(grand), "ss_total": ss_total,
        "frac_backbone": ss_b / ss_total if ss_total > 0 else float("nan"),
        "frac_family": ss_f / ss_total if ss_total > 0 else float("nan"),
        "frac_interaction": ss_inter / ss_total if ss_total > 0 else float("nan"),
        "frac_residual": ss_resid / ss_total if ss_total > 0 else float("nan"),
        "family_means": {k: float(v) for k, v in fmeans.sort_values(ascending=False).items()},
        "backbone_means": {k: float(v) for k, v in bmeans.sort_values(ascending=False).items()},
    }
    if verbose:
        print(f"  cells={n}  backbones={nb}  families={nf}   grand mean {metric}={grand:.4f}")
        print(f"  variance explained -- backbone {out['frac_backbone']:.1%} | "
              f"quantum family {out['frac_family']:.1%} | "
              f"interaction {out['frac_interaction']:.1%}")
        print("\n  family means (ranked):")
        for k, v in out["family_means"].items():
            print(f"    {k:<16}{v:.4f}   {QUANTUM_FAMILIES.get(k, {}).get('placement','')}")
        print("\n  backbone means (ranked):")
        for k, v in out["backbone_means"].items():
            print(f"    {k:<26}{v:.4f}")
    return out


def grid_multiplicity_correction(df, alpha=0.05, verbose=True):
    """Benjamini-Hochberg across every quantum-vs-matched-control comparison in the grid.

    This is the step that keeps a multi-cell grid honest: without it, the best-looking cell
    of N is a selection effect, not a finding.
    """
    q = df[(df["arm"] == "quantum") & df["vs_classical_p_two_sided"].notna()].copy()
    if q.empty:
        if verbose:
            print("  no comparisons to correct")
        return q
    rej, adj = benjamini_hochberg(q["vs_classical_p_two_sided"].to_numpy(), alpha=alpha)
    q["p_bh_adjusted"] = adj
    q["significant_after_bh"] = rej
    if verbose:
        print(f"  {len(q)} comparison(s), BH-FDR at alpha={alpha}: "
              f"{int(rej.sum())} survive correction")
        cols = ["backbone", "family", "target", "reducer", "vs_classical_diff",
                "vs_classical_p_two_sided", "p_bh_adjusted", "significant_after_bh"]
        print(q[cols].to_string(index=False))
    return q


# ==================================================================================== #
# NISQ noise ablation
# ==================================================================================== #
def nisq_noisy_expvals(family, weights, angles, p_depol, n_qubits, depth):
    """Re-evaluate a TRAINED circuit on a density-matrix simulator with a depolarizing
    channel after every entangling gate. The angles are NOT retrained: noise-aware
    training would do better and would not answer the question, which is whether the
    circuit you actually trained transfers.
    """
    import pennylane as qml
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def circ(inputs, w):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        if family in ("Q1_sel", "Q3_quanv"):
            for layer in range(w.shape[0]):
                for q in range(n_qubits):
                    qml.Rot(w[layer, q, 0], w[layer, q, 1], w[layer, q, 2], wires=q)
                if n_qubits > 1:
                    r = layer % (n_qubits - 1) + 1 if n_qubits > 1 else 1
                    for q in range(n_qubits):
                        qml.CNOT(wires=[q, (q + r) % n_qubits])
                        if p_depol > 0:
                            qml.DepolarizingChannel(p_depol, wires=(q + r) % n_qubits)
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
        if family == "Q2_reupload":
            for r_ in range(w.shape[0]):
                if r_ > 0:
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                for q in range(n_qubits):
                    qml.Rot(w[r_, q, 0], w[r_, q, 1], w[r_, q, 2], wires=q)
                if n_qubits > 1:
                    for q in range(n_qubits):
                        qml.CNOT(wires=[q, (q + 1) % n_qubits])
                        if p_depol > 0:
                            qml.DepolarizingChannel(p_depol, wires=(q + 1) % n_qubits)
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
        if family == "Q4_ttn":
            blocks, root = ttn_blocks(n_qubits)
            for i, (a, b) in enumerate(blocks):
                qml.Rot(w[i, 0], w[i, 1], w[i, 2], wires=a)
                qml.Rot(w[i, 3], w[i, 4], w[i, 5], wires=b)
                qml.CNOT(wires=[a, b])
                if p_depol > 0:
                    qml.DepolarizingChannel(p_depol, wires=b)
            return [qml.expval(qml.PauliZ(root))]
        raise ValueError(family)

    # float64 throughout. The readout layer is cast to double by the caller (the trained
    # weights are float32, and mixing the two raises "expected m1 and m2 to have the same
    # dtype" at the matmul rather than silently downcasting), and the density-matrix
    # simulation is double anyway.
    outs = []
    for i in range(len(angles)):
        z = circ(angles[i], weights)
        z = torch.stack(z) if isinstance(z, (list, tuple)) else z
        outs.append(z.detach().double().reshape(-1))
    return torch.stack(outs)