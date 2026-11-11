"""Spawn-failure release and orphan-release behaviour of the launcher. No Modal, no money: ledger and RUNS are temp."""
import json
import sys
import types

import pytest
import yaml

from opa import budget, launch

ORPHAN = "A-thrash-pin-thrash-r3+r4+r6-test-s100900-20261108001955-4a37974-0db4"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "LEDGER", tmp_path / "ledger.csv")
    monkeypatch.setattr(launch, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(launch, "INDEX", tmp_path / "runs" / "index.csv")
    monkeypatch.setattr(budget, "check", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(launch, "git", lambda *a: "abc1234")
    monkeypatch.setattr(launch, "cfg", lambda: {"gcp": {"bucket": "b"}, "modal": {"app_name": "opa-experiments"}})
    return tmp_path


def job_file(tmp_path):
    job = {"name": "x", "envelope": "E0", "track": "A", "hypotheses": ["H1"], "split": "dev", "seed": 1, "entrypoint": "m:f",
           "compute": {"kind": "gpu", "gpu": "L4", "cpu": 2, "memory_gib": 16}, "estimate_usd": 0.5, "max_usd": 1.0, "max_api_usd": 0}
    p = tmp_path / "job.yaml"
    p.write_text(yaml.safe_dump(job))
    return str(p)


class FakeNotFound(Exception):
    pass


class FakeInvalid(Exception):
    pass


class Unknown(Exception):
    pass


def fake_modal(monkeypatch, spawn_exc=None, hydrate_exc=None, from_name_exc=None, after=None):
    class Call:
        object_id = "fc-1"

    class Fn:
        def with_options(self, **kw):
            return self

        def hydrate(self):
            if hydrate_exc:
                raise hydrate_exc

        def spawn(self, spec):
            if spawn_exc:
                raise spawn_exc
            return Call()

    def from_name(app, name):
        if from_name_exc:
            raise from_name_exc
        return Fn()

    mod = types.ModuleType("modal")
    mod.Function = types.SimpleNamespace(from_name=from_name)
    mod.exception = types.SimpleNamespace(NotFoundError=FakeNotFound, InvalidError=FakeInvalid)
    monkeypatch.setitem(sys.modules, "modal", mod)


def net_reserved(run_id=None):
    rows = [r for r in budget.read_ledger() if run_id is None or r["run_id"] == run_id]
    return sum(float(r["amount_usd"]) * (1 if r["kind"] == "reserve" else -1 if r["kind"] == "release" else 0) for r in rows)


def test_spawn_failure_releases(env, monkeypatch):
    fake_modal(monkeypatch, spawn_exc=AttributeError("boom\nline two " + "x" * 400))
    assert launch.cmd_run(job_file(env)) == 5
    rows = budget.read_ledger()
    assert [r["kind"] for r in rows] == ["reserve", "release"]
    a, b = rows
    assert (a["run_id"], a["envelope"], a["job"], a["provider"]) == (b["run_id"], b["envelope"], b["job"], b["provider"])
    assert float(a["amount_usd"]) == float(b["amount_usd"]) == 1.0
    assert b["note"].startswith("spawn failed: AttributeError: boom line two")
    assert "\n" not in b["note"] and b["note"].endswith("; nothing ran")
    assert net_reserved() == 0
    assert not (launch.RUNS / a["run_id"] / "call.json").exists()


def seed_orphan(env, envelope="E2", amount=1.43):
    budget.append_ledger({"envelope": envelope, "job": "j", "kind": "reserve", "amount_usd": amount, "provider": "modal", "run_id": ORPHAN})


def test_release_orphan_once(env):
    seed_orphan(env)
    assert launch.cmd_release_orphan(ORPHAN) == 0
    rows = budget.read_ledger()
    assert [r["kind"] for r in rows] == ["reserve", "release"]
    assert rows[1]["envelope"] == "E2" and float(rows[1]["amount_usd"]) == 1.43
    assert "orphan reservation released" in rows[1]["note"]
    assert net_reserved(ORPHAN) == pytest.approx(0)
    assert launch.cmd_release_orphan(ORPHAN) == 2
    assert len(budget.read_ledger()) == 2


def test_release_orphan_refuses_with_call_json(env):
    seed_orphan(env)
    d = launch.RUNS / ORPHAN
    d.mkdir(parents=True)
    (d / "call.json").write_text("{}")
    assert launch.cmd_release_orphan(ORPHAN) == 2
    assert len(budget.read_ledger()) == 1


def test_release_orphan_refuses_without_reserve(env):
    budget.append_ledger({"envelope": "E2", "job": "j", "kind": "reserve", "amount_usd": 1.0, "provider": "modal", "run_id": "other"})
    assert launch.cmd_release_orphan(ORPHAN) == 2
    assert len(budget.read_ledger()) == 1


def kinds():
    return [r["kind"] for r in budget.read_ledger()]


def test_failed_import_writes_nothing(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)   # makes `import modal` raise ImportError
    with pytest.raises(ImportError):
        launch.cmd_run(job_file(env))
    assert kinds() == [] and not launch.RUNS.exists()


@pytest.mark.parametrize("kw", [{"from_name_exc": FakeNotFound("x")}, {"hydrate_exc": RuntimeError("y")}])
def test_lookup_or_hydrate_failure_reserves_nothing(env, monkeypatch, kw):
    fake_modal(monkeypatch, **kw)
    assert launch.cmd_run(job_file(env)) == 5
    assert kinds() == [] and not launch.RUNS.exists()


@pytest.mark.parametrize("exc", [FakeNotFound("a"), FakeInvalid("b"), TypeError("c")])
def test_pre_submission_errors_release(env, monkeypatch, exc):
    fake_modal(monkeypatch, spawn_exc=exc)
    assert launch.cmd_run(job_file(env)) == 5
    assert kinds() == ["reserve", "release"]


def test_failure_after_successful_spawn_keeps_reserve(env, monkeypatch):
    fake_modal(monkeypatch)
    monkeypatch.setattr(launch, "read_index", lambda: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        launch.cmd_run(job_file(env))
    assert kinds() == ["reserve"]
    run_id = budget.read_ledger()[0]["run_id"]
    assert (launch.RUNS / run_id / "call.json").exists()
    assert launch.cmd_release_orphan(run_id) == 2
    assert kinds() == ["reserve"]


def test_refuses_when_run_folder_exists_without_call_json(env):
    seed_orphan(env)
    (launch.RUNS / ORPHAN).mkdir(parents=True)
    assert launch.cmd_release_orphan(ORPHAN) == 2
    assert kinds() == ["reserve"]


def test_ambiguous_spawn_error_keeps_reserve_then_window(env, monkeypatch):
    fake_modal(monkeypatch, spawn_exc=Unknown("network"))
    assert launch.cmd_run(job_file(env)) == 5
    assert kinds() == ["reserve"]
    run_id = budget.read_ledger()[0]["run_id"]
    marker = launch.RUNS / run_id / "spawn_error.json"
    m = json.loads(marker.read_text())
    assert not (launch.RUNS / run_id / "call.json").exists()
    due = m["ts"] + m["timeout"] + 3600
    monkeypatch.setattr(launch.time, "time", lambda: due - 1)
    assert launch.cmd_release_orphan(run_id) == 2
    assert kinds() == ["reserve"]
    monkeypatch.setattr(launch.time, "time", lambda: due + 1)
    assert launch.cmd_release_orphan(run_id) == 0
    assert kinds() == ["reserve", "release"]


def test_refuses_with_two_reserve_lines(env):
    seed_orphan(env)
    seed_orphan(env)
    assert launch.cmd_release_orphan(ORPHAN) == 2
    assert kinds() == ["reserve", "reserve"]
