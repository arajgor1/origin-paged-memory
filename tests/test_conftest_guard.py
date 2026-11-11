"""The conftest redirect must survive a test's own monkeypatch.undo()."""
from pathlib import Path

from opa import PROJECT_DIR, budget, launch


def test_undo_does_not_revert_guard(monkeypatch):
    monkeypatch.undo()
    real = [Path(PROJECT_DIR) / "runs", Path(PROJECT_DIR) / "budget"]
    for p in (launch.RUNS, launch.INDEX, budget.LEDGER, budget.RECON_DIR):
        p = Path(p)
        assert not any(p == r or r in p.parents for r in real), p
