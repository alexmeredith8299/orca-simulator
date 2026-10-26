"""The spec forbids detection logic in the simulator: only the CLI may import the detectors, and
the detectors may use nothing from the simulator but the params loader."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "expand3"
IMPORT = re.compile(r"^\s*(?:from|import)\s+expand3(?:\.(\w+(?:\.\w+)*))?", re.MULTILINE)


def test_simulator_does_not_import_detectors() -> None:
    offenders = [p.name for p in SRC.rglob("*.py")
                 if "detect" not in p.relative_to(SRC).parts and p.name != "cli.py"
                 and re.search(r"expand3\.detect|from expand3 import detect", p.read_text())]
    assert not offenders


def test_detectors_import_only_params_from_the_simulator() -> None:
    used = set()
    for p in (SRC / "detect").glob("*.py"):
        for m in IMPORT.finditer(p.read_text()):
            used.add((m.group(1) or "").split(" ")[0])
    outside = {u for u in used if not u.startswith("detect") and u != "params"}
    assert not outside, outside
