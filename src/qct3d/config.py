"""Paths for every notebook, read from one YAML file.

The notebooks never contain a path. Each one calls `load_config()` in its first cell and reads
locations from the returned dict, so moving the data means editing `config.yaml` once.

Lookup order for the file:
  1. the path in the environment variable QCT3D_CONFIG, if set;
  2. `config.yaml` in the repository root;
  3. `config.example.yaml` in the repository root (the defaults used for the paper, which assume
     Google Colab with Google Drive mounted at /content/drive).

Relative paths in the file are resolved against `drive_root`. Placeholders of the form
`{key}` are expanded from other keys, so `rsna_output_dir: "{drive_root}/rsna_q3d_output"` works.
"""
import os
import re

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REQUIRED_KEYS = (
    "drive_mount", "drive_root",
    "rsna_archive_dir", "rsna_output_dir",
    "external_input_dir", "external_matched_csv", "external_ct_root", "external_output_dir",
    "external_part1_map_csv",
    "pillar0_external_output_dir", "pillar0_external_output_dir_v1",
    "pillar0_rsna_archive_dir", "pillar0_rsna_output_dir",
)

_PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_]+)\}")


def config_path():
    env = os.environ.get("QCT3D_CONFIG")
    if env:
        return env
    for name in ("config.yaml", "config.example.yaml"):
        p = os.path.join(REPO_ROOT, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no config.yaml or config.example.yaml in {REPO_ROOT}")


def _expand(cfg):
    out = dict(cfg)
    for _ in range(10):                      # placeholders may refer to keys that hold placeholders
        changed = False
        for k, v in out.items():
            if isinstance(v, str) and _PLACEHOLDER.search(v):
                def sub(m):
                    key = m.group(1)
                    if key not in out:
                        raise KeyError(f"config key {k!r} refers to undefined key {key!r}")
                    return str(out[key])
                nv = _PLACEHOLDER.sub(sub, v)
                if nv != v:
                    out[k], changed = nv, True
        if not changed:
            return out
    raise ValueError("config placeholders do not resolve (circular reference?)")


def load_config(path=None, verbose=True):
    """Return the configuration as a plain dict of strings (and the occasional dict/list)."""
    path = path or config_path()
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")
    cfg = _expand(cfg)
    root = cfg["drive_root"]
    for k, v in list(cfg.items()):
        if (k.endswith(("_dir", "_csv", "_root", "_path")) and isinstance(v, str) and v
                and not os.path.isabs(v)):
            cfg[k] = os.path.join(root, v)
    cfg.setdefault("external_drive_ids", {})
    if cfg["external_drive_ids"] is None:
        cfg["external_drive_ids"] = {}
    cfg["_config_file"] = os.path.abspath(path)
    if verbose:
        print(f"config: {cfg['_config_file']}")
    return cfg


def mount_drive_if_colab(cfg):
    """Mount Google Drive when running on Colab. Returns True on Colab, False elsewhere."""
    try:
        from google.colab import drive   # noqa: F401
    except ImportError:
        return False
    drive.mount(cfg["drive_mount"])
    return True
