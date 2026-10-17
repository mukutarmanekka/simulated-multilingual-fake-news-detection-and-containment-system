"""Phase 4: mechanical enforcement of the leakage and black-box rules.

- `src.*` imports (the real detector) are allowed only in
  detector/real_adapter.py.
- the string `true_label` (ground truth) is allowed only in the modules
  that may legitimately touch it: the corpus, the stub's calibration,
  metrics, and the CLI driver. Tests may use labels freely.
"""

from verifacto import config

VERIFACTO_DIR = config.REPO_ROOT / "verifacto"

SRC_IMPORT_ALLOWED = {"detector/real_adapter.py"}
TRUE_LABEL_ALLOWED = {
    "data/posts.py",
    "detector/stub.py",
    "evaluation/metrics.py",
    "main.py",
}


def _source_files():
    files = []
    for py_file in sorted(VERIFACTO_DIR.rglob("*.py")):
        if "tests" in py_file.parts:
            continue
        files.append(py_file)
    return files


def test_src_imports_only_in_real_adapter():
    for py_file in _source_files():
        text = py_file.read_text(encoding="utf-8")
        imports_src = "import src." in text or "from src." in text
        rel = py_file.relative_to(VERIFACTO_DIR).as_posix()
        if imports_src:
            assert rel in SRC_IMPORT_ALLOWED, f"{rel} must not import the detector"


def test_true_label_only_in_whitelisted_modules():
    for py_file in _source_files():
        text = py_file.read_text(encoding="utf-8")
        rel = py_file.relative_to(VERIFACTO_DIR).as_posix()
        if "true_label" in text:
            assert rel in TRUE_LABEL_ALLOWED, f"{rel} must not touch true_label"
