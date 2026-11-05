import os

import pytest

from qct3d.config import REQUIRED_KEYS, load_config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_example_config_loads_and_expands():
    c = load_config(os.path.join(ROOT, "config.example.yaml"), verbose=False)
    for k in REQUIRED_KEYS:
        assert k in c and "{" not in str(c[k]), k
    assert c["rsna_output_dir"] == c["drive_root"] + "/rsna_q3d_output"
    assert c["external_drive_ids"] == {}


def test_missing_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("drive_mount: /x\ndrive_root: /x\n")
    with pytest.raises(KeyError):
        load_config(str(p), verbose=False)


def test_relative_paths_resolve_against_drive_root(tmp_path):
    lines = [f"{k}: rel_{k}" for k in REQUIRED_KEYS if k not in ("drive_mount", "drive_root")]
    p = tmp_path / "c.yaml"
    p.write_text("drive_mount: /m\ndrive_root: /data\n" + "\n".join(lines) + "\n")
    c = load_config(str(p), verbose=False)
    assert c["rsna_output_dir"] == "/data/rel_rsna_output_dir"
