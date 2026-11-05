"""Quantum-circuit families, adapters, parameter-matched classical controls, training loop,
metrics and multiplicity correction.

Moved verbatim from cell 5 of notebooks 04 and 06 (originally nb6 and nb8). Notebook 02
(originally nb2) ran the same code without `Reducer.set_encoder_norm`, which changes nothing
unless it is called. Nothing in this module reads notebook globals; every setting is an argument.
"""
# 5. The four quantum-circuit families, the adapters, and the training loop
#
# This is the tested engine. Five choices below are NOT stylistic -- getting any
# one wrong yields an untrainable circuit that looks exactly like 'the quantum
# adapter just does not work':
#   1. diff_method (benchmarked two cells down, not inherited from folklore)
#   2. every circuit wrapped in qml.qnn.TorchLayer, so the classical control is a
#      true swap-in rather than a separately-engineered pipeline
#   3. near-identity init -- barren-plateau mitigation
#   4. angles bounded by (pi/2)*tanh(z): the circuit is 2pi-periodic, so an
#      unbounded projection lets two different feature vectors land on the same
#      rotation and the loss surface grows spurious minima
#   5. the control matches the quantum TAIL, because the reducer is shared
import math
from statistics import NormalDist

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_HAVE_PENNYLANE = True

# ==================================================================================== #
# 1.  The four quantum-circuit families
# ==================================================================================== #
# Each returns (TorchLayer, weight_shapes, n_out) so the caller never hard-codes a width.


def make_q1(n_qubits, n_layers, diff_method="backprop", init_std=0.01, seed=0):
    """Q1 -- Strongly Entangling Layers. The standard reference ansatz.

    AngleEmbedding(RY) -> StronglyEntanglingLayers -> <Z> on every wire.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)}
    torch.manual_seed(seed)
    init = {"weights": lambda t: nn.init.normal_(t, mean=0.0, std=init_std)}
    return qml.qnn.TorchLayer(circuit, shapes, init_method=init), shapes, n_qubits


def make_q2(n_qubits, n_reuploads, diff_method="backprop", init_std=0.01, seed=0):
    """Q2 -- Data re-uploading (Perez-Salinas et al., Quantum 2020).

    The data is re-encoded before EVERY variational block, which is what buys a few-qubit
    circuit its universality. This is the parameter-efficiency extreme of the zoo.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        for r in range(n_reuploads):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for w in range(n_qubits):
                qml.Rot(weights[r, w, 0], weights[r, w, 1], weights[r, w, 2], wires=w)
            if n_qubits > 1:
                for w in range(n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    shapes = {"weights": (n_reuploads, n_qubits, 3)}
    torch.manual_seed(seed)
    init = {"weights": lambda t: nn.init.normal_(t, mean=0.0, std=init_std)}
    return qml.qnn.TorchLayer(circuit, shapes, init_method=init), shapes, n_qubits


def make_q3(n_qubits, n_layers, diff_method="backprop", init_std=0.01, seed=0):
    """Q3 -- the quanvolutional filter (Henderson et al. 2020).

    Structurally an SEL block, but its PLACEMENT is what makes it a different arm: it is
    applied to every spatial position of an intermediate feature map (see SpatialAdapter),
    not once to a pooled vector. That placement is the whole point of this arm.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)}
    torch.manual_seed(seed)
    init = {"weights": lambda t: nn.init.normal_(t, mean=0.0, std=init_std)}
    return qml.qnn.TorchLayer(circuit, shapes, init_method=init), shapes, n_qubits


def ttn_blocks(n_qubits):
    """Binary tree of two-qubit blocks; the second wire of each pair carries the branch up."""
    if n_qubits & (n_qubits - 1):
        raise ValueError(f"the TTN ansatz needs a power-of-two qubit count, got {n_qubits}")
    active, blocks = list(range(n_qubits)), []
    while len(active) > 1:
        nxt = []
        for i in range(0, len(active), 2):
            a, b = active[i], active[i + 1]
            blocks.append((a, b))
            nxt.append(b)
        active = nxt
    return blocks, active[0]


def make_q4(n_qubits, diff_method="backprop", init_std=0.01, seed=0):
    """Q4 -- Tree Tensor Network classifier (Grant et al. 2018).

    A hierarchical tree of two-qubit unitaries, read out on the single root wire. Its role
    in the paper is the CLASSICALLY-EXACT ANCHOR: a TTN of this depth is efficiently
    contractible on a classical computer by construction, so if it matches the deeper
    circuits, that is evidence about how much the quantum substrate is really contributing.
    """
    blocks, root = ttn_blocks(n_qubits)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for i, (a, b) in enumerate(blocks):
            qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=a)
            qml.Rot(weights[i, 3], weights[i, 4], weights[i, 5], wires=b)
            qml.CNOT(wires=[a, b])
        return [qml.expval(qml.PauliZ(root))]

    shapes = {"weights": (len(blocks), 6)}
    torch.manual_seed(seed)
    init = {"weights": lambda t: nn.init.normal_(t, mean=0.0, std=init_std)}
    return qml.qnn.TorchLayer(circuit, shapes, init_method=init), shapes, 1


QUANTUM_FAMILIES = {
    "Q1_sel": {
        "label": "Strongly Entangling Layers",
        "placement": "pooled bottleneck vector",
        "role": "standard reference ansatz",
        "citation": "Schuld et al. 2020; qml.StronglyEntanglingLayers",
        "spatial": False,
    },
    "Q2_reupload": {
        "label": "Data re-uploading classifier",
        "placement": "pooled bottleneck vector, data re-encoded each block",
        "role": "parameter-efficiency extreme; strongest direct medical-imaging precedent",
        "citation": "Perez-Salinas et al., Quantum 4, 226 (2020)",
        "spatial": False,
    },
    "Q3_quanv": {
        "label": "Quanvolutional filter",
        "placement": "INTERMEDIATE feature map, one circuit evaluation per spatial position",
        "role": "the deep-placement arm; without it the placement question is untested",
        "citation": "Henderson et al., Quantum Machine Intelligence 2, 2 (2020)",
        "spatial": True,
    },
    "Q4_ttn": {
        "label": "Tree Tensor Network",
        "placement": "pooled bottleneck vector",
        "role": "classically-exact anchor for the 'is this really quantum' ablation",
        "citation": "Grant et al., npj Quantum Information 4, 65 (2018)",
        "spatial": False,
    },
}


def build_quantum_layer(family, n_qubits, depth, diff_method="backprop", init_std=0.01, seed=0):
    """One entry point so nothing downstream branches on the family name."""
    if not _HAVE_PENNYLANE:
        raise RuntimeError("PennyLane is not installed")
    if family == "Q1_sel":
        return make_q1(n_qubits, depth, diff_method, init_std, seed)
    if family == "Q2_reupload":
        return make_q2(n_qubits, depth, diff_method, init_std, seed)
    if family == "Q3_quanv":
        return make_q3(n_qubits, depth, diff_method, init_std, seed)
    if family == "Q4_ttn":
        return make_q4(n_qubits, diff_method, init_std, seed)
    raise ValueError(f"unknown quantum family {family!r}; known: {list(QUANTUM_FAMILIES)}")


# ==================================================================================== #
# 2.  The reducer
# ==================================================================================== #
class Reducer(nn.Module):
    """FEATURE_DIM -> n_qubits pre-angle values. Fitted on the TRAINING FOLD ONLY.

    How 1536 dimensions become 4 rotation angles dominates the result far more than the
    choice of ansatz does, so all three plausible reducers are run rather than one being
    picked and presented as neutral.
    """

    def __init__(self, mode, d_in, n_qubits, seed=0):
        super().__init__()
        if mode not in ("trainable", "pca", "pca_supervised"):
            raise ValueError(f"unknown reducer {mode!r}")
        self.mode, self.d_in, self.n_qubits = mode, d_in, n_qubits
        self.explained = float("nan")
        self.norm = None          # see set_encoder_norm below
        if mode == "trainable":
            torch.manual_seed(seed)
            self.proj = nn.Linear(d_in, n_qubits)
            nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(d_in))
            nn.init.zeros_(self.proj.bias)
        else:
            self.proj = None

    def fit(self, Xtr, ytr=None, seed=0):
        self._sc = StandardScaler().fit(Xtr)
        Ztr = self._sc.transform(Xtr).astype("float32")
        if self.mode == "trainable":
            return self
        k = (self.n_qubits if self.mode == "pca"
             else int(min(max(4 * self.n_qubits, 16), Ztr.shape[0] - 1, Ztr.shape[1])))
        k = max(int(k), int(self.n_qubits))
        self._pca = PCA(n_components=k, random_state=seed).fit(Ztr)
        P = self._pca.transform(Ztr)
        if self.mode == "pca_supervised":
            y = np.asarray(ytr).astype(float)
            # Guard BOTH sides. corrcoef against a constant returns NaN and argsort puts NaN
            # last rather than raising, so a "supervised" selection would quietly become an
            # arbitrary one on a degenerate fold.
            y_varies = float(np.std(y)) > 0
            r = np.abs([np.corrcoef(P[:, j], y)[0, 1]
                        if (y_varies and np.std(P[:, j]) > 0) else 0.0
                        for j in range(P.shape[1])])
            r = np.nan_to_num(r, nan=0.0)
            self._keep = np.argsort(-r)[:self.n_qubits]
        else:
            self._keep = np.arange(self.n_qubits)
        S = P[:, self._keep]
        self._scale = np.maximum(S.std(axis=0, keepdims=True), 1e-6)
        self.explained = float(self._pca.explained_variance_ratio_[self._keep].sum())
        return self

    def transform_np(self, X):
        Z = self._sc.transform(X).astype("float32")
        if self.mode == "trainable":
            return Z
        return (self._pca.transform(Z)[:, self._keep] / self._scale).astype("float32")

    def set_encoder_norm(self, on=True):
        """Put a non-affine BatchNorm between the learned projection and the tanh.

        This is the SAME device SpatialAdapter already carries, and it exists for the same
        reason: under AdamW the projection weights grow without bound, and once the pre-tanh
        magnitude passes roughly 5 the tanh is flat, every series receives the same angles,
        and the arm emits a constant. affine=False is deliberate -- a learnable scale would
        grow the way the projection did and put the saturation straight back -- and it also
        means ZERO new parameters, so a parameter-matched comparison is unaffected.

        Only the trainable reducer can saturate. The PCA reducers have no trainable
        projection and divide their components by their own standard deviation at fit time,
        so their pre-tanh values are O(1) by construction. Calling this on a PCA reducer is
        therefore a no-op, on purpose.
        """
        if self.proj is None:
            self.norm = None
            return self
        self.norm = nn.BatchNorm1d(self.n_qubits, affine=False) if on else None
        return self

    def forward(self, x):
        if self.proj is None:
            return x
        z = self.proj(x)
        if self.norm is not None:
            # BatchNorm1d needs >1 row in train mode; a size-1 batch would raise.
            if self.training and z.shape[0] < 2:
                self.norm.eval()
                z = self.norm(z)
                self.norm.train()
            else:
                z = self.norm(z)
        return z


# ==================================================================================== #
# 3.  Adapters
# ==================================================================================== #
class AdapterModel(nn.Module):
    """reducer -> bounded angles -> (quantum circuit | parameter-matched MLP) -> logit.

    The classical control passes through the IDENTICAL tanh bounding, so the two arms
    differ only in what sits after it.
    """

    def __init__(self, kind, reducer, n_qubits, depth, family="Q1_sel", hidden=None,
                 diff_method="backprop", init_std=0.01, seed=0):
        super().__init__()
        self.kind, self.reducer, self.n_qubits, self.family = kind, reducer, n_qubits, family
        if kind == "quantum":
            self.q, self.shapes, n_out = build_quantum_layer(
                family, n_qubits, depth, diff_method, init_std, seed)
            self.n_out = n_out
            self.out = nn.Linear(n_out, 1)
            nn.init.zeros_(self.out.bias)
            nn.init.normal_(self.out.weight, std=0.1)
            self.mlp = None
        elif kind == "classical":
            self.q = None
            self.shapes = None
            torch.manual_seed(seed)
            self.mlp = nn.Sequential(nn.Linear(n_qubits, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        else:
            raise ValueError(kind)

    def forward(self, x):
        a = (math.pi / 2.0) * torch.tanh(self.reducer(x))
        if self.kind == "quantum":
            z = self.q(a)
            if isinstance(z, (list, tuple)):
                z = torch.stack(z, dim=-1)
            z = z.float()
            if z.dim() == 1:                       # a 1-output circuit on a batch of 1
                z = z.unsqueeze(-1)
            return self.out(z).squeeze(-1)
        return self.mlp(a).squeeze(-1)

    def angle_params(self):
        return [self.q.weights] if self.q is not None else []

    def tail_params(self):
        """Trainable parameters AFTER the reducer -- the number the control is matched on."""
        return sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and not n.startswith("reducer."))


class SpatialAdapter(nn.Module):
    """Q3's deep-placement arm: a quantum filter applied at EVERY spatial position.

    Input is a cached intermediate feature map (B, C, d, h, w), not a pooled vector. A
    1x1x1 projection takes C -> n_qubits per position, the angles are bounded, the circuit
    is evaluated once per position (batched as B*P rows in a single tensor op), and the
    per-position outputs are mean-pooled before the readout.

    That is what "the quantum component sits deeper in the pipeline" has to mean
    operationally. Pooling first and calling it quanvolution would be Q1 with a new name.
    """

    def __init__(self, kind, c_in, n_qubits, depth, hidden=None, family="Q3_quanv",
                 diff_method="backprop", init_std=0.01, seed=0):
        super().__init__()
        self.kind, self.n_qubits = kind, n_qubits
        torch.manual_seed(seed)
        # THE ANGLE ENCODER HAS TO STAY OUT OF SATURATION, AND A BARE CONV DOES NOT.
        #
        # Q3 sat at AUROC exactly 0.500 for every backbone, quantum and classical alike.
        # Standardising the input maps rescued B1 and B2 but NOT B3, and the reason is not
        # the input scale at all: under AdamW the projection weights grow by roughly lr per
        # step, so after 700 steps the pre-tanh magnitude reaches 30-45 (measured). tanh is
        # flat to machine precision there, so every series lands on the same +/-pi/2 angle
        # pattern, the gradient through the encoder vanishes, and the output is a constant.
        # A constant scores AUROC exactly 0.500 and AUPRC exactly the prevalence -- which is
        # what the grid printed.
        #
        # A non-affine BatchNorm between the convolution and the encoder pins the
        # pre-activation to unit scale no matter how large the weights get. Measured on
        # post-ReLU maps with 30% / 60% dead channels: 0.500 -> 1.000 and 0.500 -> 1.000,
        # with pre-tanh magnitude 1.6 and 2.9 instead of 42 and 45.
        #
        # affine=False is deliberate: a learnable scale would grow the same way the
        # convolution did and put the saturation straight back. It also means NO new
        # parameters, and living inside `proj` means it inherits the projection's parameter
        # group and stays out of tail_params() -- so the parameter-matched comparison
        # against the classical control is exactly what it was.
        _conv = nn.Conv3d(c_in, n_qubits, kernel_size=1, bias=True)
        nn.init.normal_(_conv.weight, std=1.0 / math.sqrt(c_in))
        nn.init.zeros_(_conv.bias)
        self.proj = nn.Sequential(_conv, nn.BatchNorm3d(n_qubits, affine=False))
        if kind == "quantum":
            self.q, self.shapes, n_out = build_quantum_layer(
                family, n_qubits, depth, diff_method, init_std, seed)
            self.n_out = n_out
            self.out = nn.Linear(n_out, 1)
            nn.init.zeros_(self.out.bias)
            nn.init.normal_(self.out.weight, std=0.1)
            self.mlp = None
        elif kind == "classical":
            self.q, self.shapes = None, None
            self.mlp = nn.Sequential(nn.Linear(n_qubits, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        else:
            raise ValueError(kind)

    def forward(self, x):                       # x: (B, C, d, h, w)
        B = x.shape[0]
        a = self.proj(x)                        # (B, nq, d, h, w)
        a = a.flatten(2).transpose(1, 2)        # (B, P, nq)
        P = a.shape[1]
        a = (math.pi / 2.0) * torch.tanh(a.reshape(B * P, self.n_qubits))
        if self.kind == "quantum":
            z = self.q(a)
            if isinstance(z, (list, tuple)):
                z = torch.stack(z, dim=-1)
            z = z.float().reshape(B, P, -1).mean(dim=1)
            return self.out(z).squeeze(-1)
        z = self.mlp(a).reshape(B, P).mean(dim=1)
        return z

    def angle_params(self):
        return [self.q.weights] if self.q is not None else []

    def tail_params(self):
        return sum(p.numel() for n, p in self.named_parameters()
                   if p.requires_grad and not n.startswith("proj."))


def match_hidden_width(n_in, target_params, tol=0.10):
    """Hidden width whose 2-layer MLP parameter count is closest to `target_params`."""
    best, best_err, best_p = None, float("inf"), None
    for h in range(1, 8192):
        p = (n_in * h + h) + (h + 1)
        err = abs(p - target_params) / max(target_params, 1)
        if err < best_err:
            best, best_err, best_p = h, err, p
        if p > target_params * (1 + tol) and best is not None:
            break
    return best, best_err, best_p


# ==================================================================================== #
# 4.  Training
# ==================================================================================== #
def focal_loss(logits, y, alpha=0.25, gamma=2.0):
    """Plain cross-entropy at these prevalences spends most of its gradient on easy
    negatives; focal loss down-weights them and materially changes reported AUPRC."""
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = torch.where(y > 0.5, p, 1 - p)
    at = torch.where(y > 0.5, torch.full_like(y, alpha), torch.full_like(y, 1 - alpha))
    return (at * (1 - pt).pow(gamma) * ce).mean()


def train_adapter(model, Xtr, ytr, epochs, lr, batch, beta, weight_decay,
                  alpha=0.25, gamma=2.0, seed=0, reducer_prefixes=("reducer.", "proj."),
                  verbose=False):
    """Two parameter groups: the projection (which has enough capacity to overfit) gets
    weight decay; the adapter tail does not, and the rotation angles instead get an
    explicit beta*||theta||^2 penalty -- on the ANGLES specifically, because decaying the
    readout layer alongside them would confound the two."""
    torch.manual_seed(seed)
    red_p = [p for n, p in model.named_parameters()
             if p.requires_grad and n.startswith(reducer_prefixes)]
    tail_p = [p for n, p in model.named_parameters()
              if p.requires_grad and not n.startswith(reducer_prefixes)]
    groups = [{"params": tail_p, "weight_decay": 0.0}]
    if red_p:
        groups.append({"params": red_p, "weight_decay": weight_decay})
    opt = torch.optim.AdamW(groups, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    Xtr = torch.as_tensor(Xtr, dtype=torch.float32)
    ytr = torch.as_tensor(np.asarray(ytr), dtype=torch.float32)
    n = len(ytr)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = focal_loss(model(Xtr[idx]), ytr[idx], alpha, gamma)
            for w in model.angle_params():
                loss = loss + beta * (w ** 2).sum()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"      ep{ep:3d} loss {tot / n:.5f}")
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model, X, batch=512):
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    return torch.cat([torch.sigmoid(model(X[i:i + batch]))
                      for i in range(0, len(X), batch)]).numpy()


# ==================================================================================== #
# 5.  Metrics
# ==================================================================================== #
def fast_auroc(y, p):
    """Rank-based AUROC, ties by mid-rank. NaN in, NaN out -- deliberately, so an
    uncomputed out-of-fold vector cannot come back looking like an ordinary number."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0 or not np.isfinite(p).all():
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fast_auprc(y, p):
    from sklearn.metrics import average_precision_score
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    if y.sum() == 0 or y.sum() == len(y) or not np.isfinite(p).all():
        return float("nan")
    return float(average_precision_score(y, p))


def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    z = NormalDist().inv_cdf(1 - (1 - conf) / 2)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (ph, max(0.0, c - h), min(1.0, c + h))


def cluster_bootstrap_ci(y, p, groups, stat_fn, n_boot=2000, seed=0, conf=0.95):
    """Resample PATIENTS, not rows. One patient can contribute several series, so a row
    bootstrap reports an interval that is too narrow."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    point = stat_fn(y, p)
    uniq = np.unique(groups)
    if len(uniq) < 3:
        return (point, float("nan"), float("nan"))
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        v = stat_fn(y[idx], p[idx])
        if v == v:
            vals.append(v)
    if len(vals) < 20:
        return (point, float("nan"), float("nan"))
    lo = float(np.percentile(vals, 100 * (1 - conf) / 2))
    hi = float(np.percentile(vals, 100 * (1 - (1 - conf) / 2)))
    return (point, lo, hi)


def paired_diff_test(y, p_a, p_b, groups, stat_fn=fast_auroc, n_boot=2000, seed=0, conf=0.95):
    """Paired cluster bootstrap: both models scored on the SAME resampled patients, which
    is far more powerful than asking whether two independent intervals overlap."""
    y = np.asarray(y)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    groups = np.asarray(groups)
    d0 = stat_fn(y, p_a) - stat_fn(y, p_b)
    uniq = np.unique(groups)
    if len(uniq) < 3:
        return {"diff": d0, "lo": float("nan"), "hi": float("nan"), "p_two_sided": float("nan")}
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        da = stat_fn(y[idx], p_a[idx])
        db = stat_fn(y[idx], p_b[idx])
        if da == da and db == db:
            diffs.append(da - db)
    if len(diffs) < 20:
        return {"diff": d0, "lo": float("nan"), "hi": float("nan"), "p_two_sided": float("nan")}
    diffs = np.asarray(diffs)
    lo = float(np.percentile(diffs, 100 * (1 - conf) / 2))
    hi = float(np.percentile(diffs, 100 * (1 - (1 - conf) / 2)))
    # Two-sided bootstrap p: how often the resampled difference crosses zero.
    frac = float(np.mean(diffs <= 0)) if d0 > 0 else float(np.mean(diffs >= 0))
    p = min(1.0, 2 * max(frac, 1.0 / (len(diffs) + 1)))
    return {"diff": float(d0), "lo": lo, "hi": hi, "p_two_sided": p}


def calibration_metrics(y, p, n_bins=10):
    y = np.asarray(y).astype(float)
    p = np.asarray(p, dtype=float)
    out = {"brier": float("nan"), "calibration_slope": float("nan"),
           "calibration_intercept": float("nan"), "ece": float("nan")}
    if not np.isfinite(p).all() or len(y) == 0:
        return out
    out["brier"] = float(np.mean((p - y) ** 2))
    eps = 1e-6
    lo = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    if len(np.unique(y)) == 2 and np.std(lo) > 0:
        try:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(lo.reshape(-1, 1), y.astype(int))
            out["calibration_slope"] = float(lr.coef_[0][0])
            out["calibration_intercept"] = float(lr.intercept_[0])
        except Exception:
            pass
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= 1.0)
        if m.sum():
            ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    out["ece"] = float(ece)
    return out


def pick_thresholds(y, p, target_sensitivity=0.85):
    """Chosen on OUT-OF-FOLD predictions only. Choosing them on the holdout would make
    every sensitivity below optimistically biased and nothing would complain."""
    from sklearn.metrics import roc_curve
    out = {"fixed_0.5": 0.5}
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() < 5 or len(np.unique(y[ok])) < 2:
        out["youden"] = 0.5
        out["high_sensitivity"] = 0.5
        return out
    fpr, tpr, thr = roc_curve(y[ok], p[ok])
    out["youden"] = float(thr[int(np.argmax(tpr - fpr))])
    idx = np.where(tpr >= target_sensitivity)[0]
    out["high_sensitivity"] = float(thr[idx[0]]) if len(idx) else float(thr[-1])
    for k, v in out.items():
        if not np.isfinite(v):
            out[k] = 0.5
    return out


def threshold_metrics(y, p, thr):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        return {}
    yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum())
    tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum())
    fn = int(((yh == 0) & (y == 1)).sum())
    sens, sens_lo, sens_hi = wilson_ci(tp, tp + fn)
    spec, spec_lo, spec_hi = wilson_ci(tn, tn + fp)
    ppv, ppv_lo, ppv_hi = wilson_ci(tp, tp + fp) if (tp + fp) else (float("nan"),) * 3
    acc = (tp + tn) / max(len(y), 1)
    maj = max(y.mean(), 1 - y.mean())
    bal = np.nanmean([sens, spec])
    return {"threshold": float(thr), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": float(acc), "majority_class_rate": float(maj),
            "accuracy_minus_majority": float(acc - maj), "balanced_accuracy": float(bal),
            "sensitivity": sens, "sensitivity_lo": sens_lo, "sensitivity_hi": sens_hi,
            "specificity": spec, "specificity_lo": spec_lo, "specificity_hi": spec_hi,
            "ppv": ppv, "ppv_lo": ppv_lo, "ppv_hi": ppv_hi}


def benjamini_hochberg(pvals, alpha=0.05):
    """Returns (rejected, adjusted_p) in the input order."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    adj = np.full(len(p), np.nan)
    rej = np.zeros(len(p), dtype=bool)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return rej, adj
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(prev, p[i] * m / (rank + 1))
        adj[i] = prev = val
    rej[ok] = adj[ok] <= alpha
    return rej, adj