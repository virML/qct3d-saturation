import glob
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_release_checker_passes():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_release.py"), ROOT],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_every_notebook_reads_paths_from_config():
    nbs = sorted(glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb")))
    assert len(nbs) == 7
    for p in nbs:
        nb = json.load(open(p))
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        assert code[0].startswith("# 0. Repository and paths"), p
        assert "P = load_config()" in code[0], p
        assert not any("/content/drive" in c for c in code), p
        compile("\n".join(l for c in code for l in c.splitlines()
                          if not l.lstrip().startswith(("!", "%"))), p, "exec")
