"""The launcher's money path with Modal mocked: reserve -> spawn -> collect -> actual + release; crash; still running."""
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import modal  # noqa: E402
from modal.exception import TimeoutError as ModalNotReady  # noqa: E402

from opa import budget, launch  # noqa: E402


class FakeCall:
    store = {}

    def __init__(self, cid, outcome):
        self.object_id, self.outcome = cid, outcome
        FakeCall.store[cid] = self

    def get(self, timeout=None):
        if self.outcome == "running":
            raise ModalNotReady()
        if self.outcome == "running_builtin":
            raise TimeoutError()
        if self.outcome == "crash":
            raise RuntimeError("container died")
        return self.outcome


class FakeFn:
    def __init__(self, outcome):
        self.outcome, self.opts = outcome, None

    def with_options(self, **opts):
        self.opts = opts
        return self

    def spawn(self, spec):
        return FakeCall(f"fc-{len(FakeCall.store)}", self.outcome if not callable(self.outcome) else self.outcome(spec))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    (tmp_path / "budget").mkdir()
    ap = {"total_cap_usd": 100, "envelopes": [{"id": "E0", "approved_usd": 50, "approved_by": "AR", "approved_on": "2026-11-06"}]}
    (tmp_path / "budget/approvals.yaml").write_text(yaml.safe_dump(ap))
    monkeypatch.setattr(budget, "APPROVALS", tmp_path / "budget/approvals.yaml")
    monkeypatch.setattr(budget, "LEDGER", tmp_path / "budget/ledger.csv")
    monkeypatch.setattr(launch, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(launch, "INDEX", tmp_path / "runs/index.csv")
    monkeypatch.setattr(launch, "cfg", lambda: {"gcp": {"bucket": "b"}, "modal": {"app_name": "opa-experiments"}})
    monkeypatch.setattr(launch, "git", lambda *a: "" if a[0] in ("tag", "status") else "abcdef1234567")
    job = {"name": "j", "envelope": "E0", "track": "A", "hypotheses": ["H1"], "system": "S11", "dataset": "lbmhqa", "cell": "c",
           "split": "dev", "seed": 1, "entrypoint": "opa.tracks.smoke:main", "args": {},
           "compute": {"kind": "gpu", "gpu": "L4", "cpu": 2, "memory_gib": 16}, "estimate_usd": 2, "max_usd": 10, "max_api_usd": 0}
    p = tmp_path / "job.yaml"
    p.write_text(yaml.safe_dump(job))
    return tmp_path, p


def ledger_by_kind():
    out = {}
    for r in budget.read_ledger():
        out.setdefault((r["kind"], r["provider"]), 0.0)
        out[(r["kind"], r["provider"])] += float(r["amount_usd"])
    return out


def test_ok_flow(env, monkeypatch):
    tmp, job = env
    fn = FakeFn(lambda spec: {"status": "ok", "compute_usd": 1.5, "api_usd": 0, "metrics": {"em": 0.6},
                              "manifest": {"git_sha": spec["git_sha"]}, "events": {"uri": "gs://b/x", "sha256": "a" * 64}})
    monkeypatch.setattr(modal.Function, "from_name", staticmethod(lambda app, name: fn))
    monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(lambda cid: FakeCall.store[cid]))
    assert launch.cmd_run(str(job)) == 0
    # reservation is held while running: remaining = 50 - 10
    assert abs(budget.envelope_states()["E0"].remaining - 40) < 1e-9
    timeout_expected = int(10 / launch.rate_per_second({"kind": "gpu", "gpu": "L4", "cpu": 2, "memory_gib": 16}, budget.load_prices()))
    assert fn.opts["gpu"] == "L4" and fn.opts["timeout"] == timeout_expected and fn.opts["memory"] == (16384, 32768) and fn.opts["cpu"] == (2.0, 4.0) and fn.opts["scaledown_window"] == 2
    launch.main(["collect", "--all"])
    k = ledger_by_kind()
    assert k[("actual", "modal")] == 1.5 and ("actual", "anthropic") not in k and k[("reserve", "modal")] == k[("release", "modal")] == 10
    s = budget.envelope_states()["E0"]
    assert abs(s.spent - 1.5) < 1e-9 and s.reserved == 0
    row = launch.read_index()[0]
    assert row["status"] == "ok" and row["cost_usd"] == "1.5000"
    rid = row["run_id"]
    assert json.loads((tmp / "runs" / rid / "manifest.json").read_text())["git_sha"] == "abcdef1234567"
    assert (tmp / "runs" / rid / "events_uri.json").exists()


def test_crash_and_running(env, monkeypatch):
    tmp, job = env
    monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(lambda cid: FakeCall.store[cid]))
    monkeypatch.setattr(modal.Function, "from_name", staticmethod(lambda app, name: FakeFn("running")))
    launch.cmd_run(str(job))
    launch.main(["collect", "--all"])
    assert launch.read_index()[0]["status"] == "running"          # still running: nothing finalised
    monkeypatch.setattr(modal.Function, "from_name", staticmethod(lambda app, name: FakeFn("crash")))
    launch.cmd_run(str(job))
    launch.main(["collect", "--all"])
    rows = {r["status"] for r in launch.read_index()}
    assert rows == {"running", "failed"}
    s = budget.envelope_states()["E0"]
    assert s.reserved == 10                                        # only the running job is still reserved
    assert 0 <= s.spent <= 10


def test_refusals(env, monkeypatch):
    tmp, job = env
    j = yaml.safe_load(job.read_text())
    job.write_text(yaml.safe_dump({**j, "max_usd": 60}))
    assert launch.cmd_run(str(job)) == 3                           # over the envelope
    job.write_text(yaml.safe_dump({**j, "split": "test"}))
    assert launch.cmd_run(str(job)) == 4                           # no prereg-v2 tag
    job.write_text(yaml.safe_dump({**j, "envelope": "E5"}))
    assert launch.cmd_run(str(job)) == 3                           # unapproved envelope
    job.write_text(yaml.safe_dump({**j, "max_api_usd": 1}))
    assert launch.cmd_run(str(job)) == 2                           # v3: no API budget
    job.write_text(yaml.safe_dump({**j, "compute": {**j["compute"], "gpu": "A100-80GB"}}))
    assert launch.cmd_run(str(job)) == 2                           # v3: L4/T4 only
    assert budget.read_ledger() == []                              # nothing reserved by refused jobs


def _spawn(env, monkeypatch, outcome):
    tmp, job = env
    monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(lambda cid: FakeCall.store[cid]))
    monkeypatch.setattr(modal.Function, "from_name", staticmethod(lambda app, name: FakeFn(outcome)))
    launch.cmd_run(str(job))
    return launch.read_index()[0]["run_id"], tmp


def test_builtin_timeout_means_still_running(env, monkeypatch):
    rid, tmp = _spawn(env, monkeypatch, "running_builtin")
    launch.main(["collect", "--all"])
    assert launch.read_index()[0]["status"] == "running"
    assert not (tmp / "runs" / rid / "result.json").exists()
    assert [r["kind"] for r in budget.read_ledger()] == ["reserve"]


def _premature(rid, tmp):
    """Reproduce the old bug's footprint: failed TimeoutError(), crash-cost actual, release."""
    call = json.loads((tmp / "runs" / rid / "call.json").read_text())
    launch._finalize(rid, {"status": "failed", "error": "TimeoutError()", "metrics": {}, "manifest": {}}, "failed", 0.02, 0.0, launch.PREMATURE_NOTE)
    return call


def _recheck_case(env, monkeypatch, outcome):
    """A run finalized prematurely by the old bug; at recheck time the call turns out to be `outcome`."""
    rid, tmp = _spawn(env, monkeypatch, "running_builtin")
    _premature(rid, tmp)
    assert budget.envelope_states()["E0"].reserved == 0
    call = FakeCall.store[launch.read_index()[0]["call_id"]]
    if isinstance(outcome, type) and issubclass(outcome, BaseException):
        def get(timeout=None):
            raise outcome()
        call.get = get
    else:
        call.outcome = outcome
    launch.main(["collect", "--recheck", rid])
    return rid, tmp


def test_recheck_still_running(env, monkeypatch):
    rid, tmp = _recheck_case(env, monkeypatch, "running_builtin")
    s = budget.envelope_states()["E0"]
    assert abs(s.spent) < 1e-9 and s.reserved == 10
    assert launch.read_index()[0]["status"] == "running"
    assert not (tmp / "runs" / rid / "result.json").exists()
    kinds = [(r["kind"], r["note"]) for r in budget.read_ledger()]
    assert kinds[-2][0] == "reserve" and kinds[-1] == ("actual", launch.CORRECTION_NOTE)   # re-reserve first, then correction
    launch.main(["collect", "--recheck", rid])                      # repeat: nothing premature any more
    assert budget.envelope_states()["E0"].reserved == 10 and len(budget.read_ledger()) == len(kinds)


def test_recheck_finished(env, monkeypatch):
    rid, tmp = _recheck_case(env, monkeypatch, {"status": "ok", "compute_usd": 1.5, "api_usd": 0, "metrics": {"em": 1}, "manifest": {"x": 1}})
    s = budget.envelope_states()["E0"]
    assert abs(s.spent - 1.5) < 1e-9 and s.reserved == 0
    row = launch.read_index()[0]
    assert row["status"] == "ok" and row["cost_usd"] == "1.5000"
    assert json.loads((tmp / "runs" / rid / "metrics.json").read_text()) == {"em": 1}
    n = len(budget.read_ledger())
    launch.main(["collect", "--recheck", rid])                      # repeat changes nothing
    assert len(budget.read_ledger()) == n and abs(budget.envelope_states()["E0"].spent - 1.5) < 1e-9


def test_recheck_crash(env, monkeypatch):
    rid, tmp = _recheck_case(env, monkeypatch, "crash")
    assert launch.read_index()[0]["status"] == "failed"
    s = budget.envelope_states()["E0"]
    assert s.reserved == 0 and 0 < s.spent <= 10                    # honest crash estimate, premature cost not double counted
    n = len(budget.read_ledger())
    launch.main(["collect", "--recheck", rid])                      # now a real crash error: not premature
    assert len(budget.read_ledger()) == n


def test_recheck_function_timeout(env, monkeypatch):
    from modal.exception import FunctionTimeoutError
    rid, tmp = _recheck_case(env, monkeypatch, FunctionTimeoutError)
    assert launch.read_index()[0]["status"] == "timeout"
    s = budget.envelope_states()["E0"]
    assert s.reserved == 0 and 0 < s.spent <= 10


def test_recheck_output_expired(env, monkeypatch):
    from modal.exception import OutputExpiredError
    from opa import gcs

    def missing(uri):
        raise FileNotFoundError(uri)
    monkeypatch.setattr(gcs, "get_bytes", missing)
    rid, tmp = _recheck_case(env, monkeypatch, OutputExpiredError)
    assert launch.read_index()[0]["status"] == "lost"
    s = budget.envelope_states()["E0"]
    assert s.reserved == 0 and 0 < s.spent <= 10


def test_recheck_usage(env, capsys):
    assert launch.main(["collect", "--recheck"]) == 2
    assert "usage" in capsys.readouterr().out
