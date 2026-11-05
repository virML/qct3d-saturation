"""Refuse to release anything that could carry protected health information or private details.

usage:  python tools/check_release.py [repo_root]

Exits non-zero, listing every problem, if any of these holds:
  * a notebook has outputs, execution counts, attachments, or any cell/notebook metadata beyond
    the kernel name (Colab stores the user's display name and account id in cell metadata);
  * a file type that holds patient-level data is present (.parquet, .npy, .npz, .jsonl, .dcm,
    .nii, .zip, .pt, .pth, .pkl, .h5);
  * a CSV under data/ has a column that looks like an identifier;
  * any text file contains a DICOM UID other than a standard SOP class, a hospital study id,
    a 7+ digit number not on the allow-list, a Google Drive file id, a hard-coded Drive path
    outside config.example.yaml, an access token, an e-mail address, an embedded base64 blob,
    or one of the personal names that the original notebooks quoted.
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
BINARY_DATA = (".parquet", ".npy", ".npz", ".jsonl", ".dcm", ".nii", ".gz", ".zip", ".pt", ".pth",
               ".pkl", ".pickle", ".h5", ".hdf5", ".feather")
TEXT = (".py", ".ipynb", ".md", ".txt", ".yaml", ".yml", ".csv", ".cff", ".json", ".toml", ".cfg",
        ".gitignore", "")
ID_COLUMN = re.compile(r"(?i)(patient|person|name|birth|mrn|accession|series_?uid|study_?uid|"
                       r"sop|instance|scan_id|study_id|report_date|study_date|institution)")
ALLOWED_NUMBERS = {"12345", "20250825"}          # random seeds used in the notebooks

PATTERNS = {
    "DICOM UID": re.compile(r"(?<![\d.])(?!1\.2\.840\.10008\.)\d+(?:\.\d+){5,}\b"),
    "hospital study id": re.compile(r"\bNLR\d{2,}"),
    "long number": re.compile(r"(?<![\w.\-])\d{7,}(?![\w.\-])"),
    "Drive file id": re.compile(r"[\"'/=]1[A-Za-z0-9_-]{27,40}[\"'/&]"),
    "Drive path": re.compile(r"/content/drive/|MyDrive/"),
    "token": re.compile(r"\b(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9]{20,})"),
    "e-mail": re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}"),
    "base64 blob": re.compile(r"[A-Za-z0-9+/]{200,}={0,2}"),
    "personal name": re.compile(r"(?i)\b(ANANTHA|LAKSHUMMA|RAMESH)\b"),
    "private folder": re.compile(r"(?i)MyBook|My Mac|july 11th|july 3rd"),
    "Colab user": re.compile(r"displayName|userId|user_tz"),
}
# where a pattern is expected and harmless
ALLOW = {
    "Drive path": {"config.example.yaml", "src/qct3d/config.py", "tools/check_release.py",
                   "README.md", "tests/test_release.py"},
    "e-mail": {"CITATION.cff", ".zenodo.json"},
    "personal name": {"tools/check_release.py"},
    "private folder": {"tools/check_release.py"},
    "Colab user": {"tools/check_release.py", "tests/test_release.py"},
    "hospital study id": {"tools/check_release.py"},
}
ALLOWED_EMAILS = {"[EMAIL]", "noreply@anthropic.com", "virinchi227@gmail.com"}  # corresponding author, as printed in the article


def scan_text(rel, text, problems):
    for name, pat in PATTERNS.items():
        if rel in ALLOW.get(name, set()):
            continue
        for m in pat.finditer(text):
            s = m.group(0)
            if name == "long number" and s in ALLOWED_NUMBERS:
                continue
            if name == "e-mail" and (s in ALLOWED_EMAILS or s.endswith("@example.org")):
                continue
            line = text.count("\n", 0, m.start()) + 1
            problems.append(f"{rel}:{line}: {name}: {s[:60]!r}")


def check_notebook(rel, path, problems):
    nb = json.load(open(path))
    extra = set(nb.get("metadata", {})) - {"kernelspec", "language_info"}
    if extra:
        problems.append(f"{rel}: notebook metadata carries {sorted(extra)}")
    for i, c in enumerate(nb["cells"]):
        if c.get("metadata"):
            problems.append(f"{rel}: cell {i} has metadata {list(c['metadata'])}")
        if c.get("attachments"):
            problems.append(f"{rel}: cell {i} has attachments")
        if c["cell_type"] == "code":
            if c.get("outputs"):
                problems.append(f"{rel}: cell {i} has {len(c['outputs'])} output(s)")
            if c.get("execution_count") is not None:
                problems.append(f"{rel}: cell {i} has an execution count")
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def check_csv(rel, path, problems):
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    bad = [h for h in header if ID_COLUMN.search(h)]
    if bad:
        problems.append(f"{rel}: identifier-like column(s) {bad}")


def main():
    problems, n = [], 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
            ext = os.path.splitext(fn)[1].lower()
            if ext in BINARY_DATA:
                problems.append(f"{rel}: data file type {ext} must not be released")
                continue
            if ext not in TEXT and fn not in ("LICENSE",):
                problems.append(f"{rel}: unexpected file type; add it to the checker deliberately")
                continue
            n += 1
            if ext == ".ipynb":
                text = check_notebook(rel, path, problems)
            else:
                text = open(path, encoding="utf-8", errors="replace").read()
                if ext == ".csv" and rel.startswith("data/derived/"):
                    check_csv(rel, path, problems)
            scan_text(rel, text, problems)
    if problems:
        print(f"RELEASE CHECK FAILED -- {len(problems)} problem(s) in {n} file(s):")
        print("\n".join("  " + p for p in problems))
        sys.exit(1)
    print(f"release check passed: {n} files scanned, no outputs, metadata, identifiers or private paths.")


if __name__ == "__main__":
    main()
