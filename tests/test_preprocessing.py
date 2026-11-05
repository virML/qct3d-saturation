import numpy as np
import pytest

pytest.importorskip("SimpleITK")
import qct3d.preprocessing as prep  # noqa: E402


def test_paper_settings_are_the_defaults():
    s = prep.current_settings()
    assert s["TARGET_SPACING_XYZ"] == (1.0, 1.0, 3.0)
    assert s["TARGET_SHAPE_DHW"] == (64, 256, 256)
    assert [w[1:] for w in s["HU_WINDOWS"]] == [(40.0, 80.0), (80.0, 200.0), (40.0, 380.0)]
    assert s["RESAMPLE_INTERPOLATOR"] == "linear" and s["SKULL_STRIP"] is False


def test_self_test_passes():
    resampled, out = prep._preprocessing_self_test()
    assert out == (3, 1, 64, 256, 256)
    assert resampled == (40, 224, 269)


def test_configure_rejects_unknown_and_bad_values():
    with pytest.raises(KeyError):
        prep.configure(TARGET_SPACING=(1, 1, 1))
    before = prep.current_settings()
    with pytest.raises(ValueError):
        prep.configure(RESAMPLE_INTERPOLATOR="cubic")
    prep.configure(**before)
    assert prep.current_settings() == before


def test_window_and_crop():
    v = np.linspace(-1000, 1000, 5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    w = prep.hu_window(v, 40.0, 80.0)
    assert w.min() == 0.0 and w.max() == 1.0
    c = prep.center_crop_or_pad(v, (3, 8, 7), pad_value=-1024.0)
    assert c.shape == (3, 8, 7) and (c[:, 0, :] == -1024.0).all()
