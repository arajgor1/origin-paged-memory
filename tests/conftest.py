"""Safety net: no test may write to the real ledger, run index or runs/ folder.

The redirect uses its own MonkeyPatch, so a test calling monkeypatch.undo() cannot undo it."""
from pathlib import Path

import pytest

from opa import PROJECT_DIR, budget, launch

# (module, attribute) pairs that hold a ledger / index / runs path
GUARDED = [(budget, "LEDGER"), (budget, "RECON_DIR"), (launch, "RUNS"), (launch, "INDEX")]


def _real_paths() -> list[Path]:
    return [Path(PROJECT_DIR) / "runs", Path(PROJECT_DIR) / "budget"]


@pytest.fixture(autouse=True)
def _no_real_ledger_or_runs(tmp_path):
    safe = tmp_path / "_guard"
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(budget, "LEDGER", safe / "budget" / "ledger.csv")
        mp.setattr(budget, "RECON_DIR", safe / "budget" / "reconciliation")
        mp.setattr(launch, "RUNS", safe / "runs")
        mp.setattr(launch, "INDEX", safe / "runs" / "index.csv")
        (safe / "budget").mkdir(parents=True, exist_ok=True)
        yield
        bad = []
        for mod, name in GUARDED:
            p = Path(getattr(mod, name))
            if any(p == r or r in p.parents for r in _real_paths()):
                bad.append(f"{mod.__name__}.{name} -> {p}")
        assert not bad, "test left a path pointing at the real project data: " + "; ".join(bad)
    finally:
        mp.undo()
