"""Resample -> centre crop/pad -> HU windowing, shared by both cohorts.

Moved verbatim from cell 12 of notebook 01 (RSNA) and the identically worded cell in notebook 03
(external cohort); the two cells were byte-identical. Having one copy is what makes "identical
preprocessing for both cohorts" checkable.

The functions read five module-level settings. They default to the values used in the paper and
are set from each notebook's configuration cell with `configure(...)`, so the notebook stays the
single place where a setting is chosen and the preprocessing signature is computed.
"""
# 12. Preprocessing for the 3D backbone -- resample -> whole-brain crop/pad -> HU windowing
#
# Every choice below is sourced or reasoned, not picked by hand.
#
#   RESAMPLE to TARGET_SPACING_XYZ using each scan's OWN measured spacing (assemble_volume
#     measures z from the ImagePositionPatient gaps rather than trusting SliceThickness, which
#     disagrees with reality whenever there is slice overlap or a gap). SimpleITK, sitkLinear by
#     default: in plane this is a DOWNsample on this dataset (~0.45 mm native to the configured
#     TARGET_SPACING_XYZ, so about 2.2x at 1.0 mm), where B-spline buys nothing and its overshoot
#     at the skull/air boundary is a real artefact. (This note used to quote a 4x factor and a
#     2.0 mm target, left over from an earlier default -- the grid-fit report in cell 10 prints
#     the real factor measured on your own cohort.) Set RESAMPLE_INTERPOLATOR = "bspline" if you disagree; the preprocessing
#     signature tracks the choice, so cached features are re-extracted rather than mixed.
#
#   CROP/PAD to TARGET_SHAPE_DHW on a fixed whole-head grid, NOT a lesion-centred crop: negatives
#     and deployment-time cases, where no lesion location is known in advance, must be framed
#     identically to training positives, or the framing itself leaks the label. True centre
#     crop/pad on each axis: start = (current - target) // 2. Padding uses the volume's own
#     minimum, which for a head CT is air.
#
#   DUAL HU WINDOWING, brain + subdural, each clipped to [C - W/2, C + W/2] and
#     rescaled to [0, 1]. Then each window is z-scored over the whole volume, which is MedicalNet's
#     own intensity normalisation (Med3D, Chen et al. 2019) -- the pretrained weights expect
#     zero-mean unit-variance input, and feeding them [0, 1] instead shifts every activation in
#     the frozen stem before a single learned parameter is involved.
#
#   ONE FORWARD PASS PER WINDOW. MONAI refuses `pretrained=True` for any n_input_channels != 1
#     (see monai/networks/nets/resnet.py::_resnet), so a 2-channel input would mean inflating
#     conv1's weights -- which changes the pretrained function. Running the frozen backbone twice
#     and concatenating keeps the published weights bit-exact. Cost: 2x the forward passes, which
#     on a T4 is ~0.1 s per volume against ~30 s of zip reading. Not the bottleneck.
import numpy as np
import SimpleITK as sitk
import torch

# ---- settings (paper values; overwritten by configure()) -------------------------------------
TARGET_SPACING_XYZ = (1.0, 1.0, 3.0)                     # (x, y, z) mm
TARGET_SHAPE_DHW = (64, 256, 256)                        # (D, H, W) voxels
HU_WINDOWS = [("brain", 40.0, 80.0), ("subdural", 80.0, 200.0), ("soft", 40.0, 380.0)]
RESAMPLE_INTERPOLATOR = "linear"                         # "linear" | "bspline"
SKULL_STRIP = False

_SETTINGS = ("TARGET_SPACING_XYZ", "TARGET_SHAPE_DHW", "HU_WINDOWS", "RESAMPLE_INTERPOLATOR",
             "SKULL_STRIP")


def configure(**settings):
    """Set the preprocessing constants from a notebook's configuration cell.

    Unknown names raise, so a typo cannot leave a default silently in force.
    Returns the settings now in force.
    """
    g = globals()
    for k, v in settings.items():
        if k not in _SETTINGS:
            raise KeyError(f"unknown preprocessing setting {k!r}; expected one of {_SETTINGS}")
        g[k] = v
    if g["RESAMPLE_INTERPOLATOR"] not in ("linear", "bspline"):
        raise ValueError(f"RESAMPLE_INTERPOLATOR must be 'linear' or 'bspline', "
                         f"got {g['RESAMPLE_INTERPOLATOR']!r}")
    return current_settings()


def current_settings():
    return {k: globals()[k] for k in _SETTINGS}


_SITK_INTERP = {"linear": sitk.sitkLinear, "bspline": sitk.sitkBSpline}


def resample_to_spacing(volume, spacing, target_spacing_xyz=None):
    """volume: (D, H, W) HU. spacing: {"sx","sy","sz"} in mm. Returns a resampled (D, H, W)."""
    target = tuple(target_spacing_xyz or TARGET_SPACING_XYZ)
    native_xyz = (float(spacing["sx"]), float(spacing["sy"]), float(spacing["sz"]))
    if all(abs(native_xyz[i] - target[i]) < 1e-4 for i in range(3)):
        return volume

    img = sitk.GetImageFromArray(volume)      # sitk index order is (x, y, z); array is (z, y, x)
    img.SetSpacing(native_xyz)
    native_size = img.GetSize()
    new_size = [max(1, int(round(native_size[i] * native_xyz[i] / target[i]))) for i in range(3)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(target)
    r.SetSize(new_size)
    r.SetOutputOrigin(img.GetOrigin())
    r.SetOutputDirection(img.GetDirection())
    r.SetTransform(sitk.Transform())
    r.SetInterpolator(_SITK_INTERP[RESAMPLE_INTERPOLATOR])
    r.SetDefaultPixelValue(float(np.min(volume)))   # pad the resample grid with air, not with 0 HU
    return sitk.GetArrayFromImage(r.Execute(img))


def center_crop_or_pad(volume, target_shape, pad_value):
    """True centre crop/pad on each axis: start = (current - target) // 2. Not stride sampling,
    which quietly changes what the model sees."""
    out = volume
    for axis, target in enumerate(target_shape):
        current = out.shape[axis]
        if current > target:
            start = (current - target) // 2
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(start, start + target)
            out = out[tuple(sl)]
        elif current < target:
            total = target - current
            before = total // 2
            pad = [(0, 0)] * out.ndim
            pad[axis] = (before, total - before)
            out = np.pad(out, pad, mode="constant", constant_values=pad_value)
    return out


def hu_window(volume, center, width):
    """Clip to [C - W/2, C + W/2] and rescale to [0, 1]."""
    lo, hi = center - width / 2.0, center + width / 2.0
    return np.clip((volume - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def preprocess_for_backbone(volume, spacing):
    """(D, H, W) HU volume -> (n_windows, 1, D, H, W) float32 tensor, z-scored per window.

    Returned with the window axis FIRST and an explicit singleton channel axis, so the caller can
    treat it as a batch of n_windows single-channel volumes and hand it straight to the backbone.
    """
    pad_value = float(np.min(volume))
    resampled = resample_to_spacing(volume, spacing)
    cropped = center_crop_or_pad(resampled, tuple(TARGET_SHAPE_DHW), pad_value=pad_value)

    chans = []
    for _name, center, width in HU_WINDOWS:
        w = hu_window(cropped, center, width)
        # Med3D / MedicalNet intensity normalisation: zero mean, unit variance over the volume.
        # A window with no contrast at all (a constant volume) would divide by ~0; guarded.
        w = (w - w.mean()) / max(float(w.std()), 1e-5)
        chans.append(w)
    x = torch.from_numpy(np.ascontiguousarray(np.stack(chans, axis=0))).float().unsqueeze(1)

    expected = (len(HU_WINDOWS), 1, *TARGET_SHAPE_DHW)
    if tuple(x.shape) != expected:
        raise RuntimeError(f"preprocessing produced {tuple(x.shape)}, expected {expected}")
    return x


# --- self-test on synthetic data: cheap, and it catches an axis swap immediately ----------------
def _preprocessing_self_test():
    # Deliberately anisotropic and non-square: 40 slices of 320x384 at 0.7 x 0.7 mm in plane and
    # 3.0 mm through plane. If sx/sy/sz were ever transposed the resampled shape comes out wrong
    # and this fails loudly instead of quietly rescaling every scan in the cohort.
    rng = np.random.default_rng(0)
    vol = (rng.standard_normal((40, 320, 384)).astype(np.float32) * 200.0) - 400.0
    spacing = {"sx": 0.7, "sy": 0.7, "sz": 3.0}

    resampled = resample_to_spacing(vol, spacing)
    exp_d = int(round(40 * 3.0 / TARGET_SPACING_XYZ[2]))
    exp_h = int(round(320 * 0.7 / TARGET_SPACING_XYZ[1]))
    exp_w = int(round(384 * 0.7 / TARGET_SPACING_XYZ[0]))
    assert resampled.shape == (exp_d, exp_h, exp_w), (
        f"resample gave {resampled.shape}, expected {(exp_d, exp_h, exp_w)} -- axis order is wrong")

    out = preprocess_for_backbone(vol, spacing)
    assert tuple(out.shape) == (len(HU_WINDOWS), 1, *TARGET_SHAPE_DHW), out.shape
    assert torch.isfinite(out).all(), "non-finite values after windowing"
    for i in range(out.shape[0]):
        m, s = float(out[i].mean()), float(out[i].std())
        assert abs(m) < 1e-3 and abs(s - 1.0) < 1e-2, f"window {i} not standardised: mean {m}, std {s}"

    # A volume SHORTER than the target depth must come out PADDED, not short -- the backbone is
    # fully convolutional so a short volume would not error, it would just be a different framing.
    thin = (rng.standard_normal((12, 256, 256)).astype(np.float32) * 100.0) - 500.0
    thin_out = preprocess_for_backbone(thin, {"sx": 2.0, "sy": 2.0, "sz": 2.0})
    assert tuple(thin_out.shape)[2:] == tuple(TARGET_SHAPE_DHW), thin_out.shape

    # The two windows must actually differ. A copy-paste that windowed twice with the same
    # centre/width would double the feature vector with no new information and nothing would
    # complain -- the dimensions would still be 1024.
    if len(HU_WINDOWS) > 1:
        d = float((out[0] - out[1]).abs().mean())
        assert d > 1e-3, f"the {len(HU_WINDOWS)} windows produced near-identical volumes (mean |diff| {d:.2e})"
    return resampled.shape, tuple(out.shape)
