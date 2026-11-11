"""collect must not finalize a queued/running call as lost, and recheck must undo a premature lost finalize.
No Modal, no money: ledger, RUNS and INDEX are temp; modal and gcs are fakes."""
import json
import sys
import types

import pytest

from opa import budget, gcs, launch

RID = "A-embed-x-y-test-s1-20261108002153-a23d447-d624"
MAX_USD = 1.5
RATE = 0.0005
TIMEOUT = 900


class OutputExpired(Exception):
    pass


class ModalConn(Exception):
    pass


class FnTimeout(Exception):
    pass


class ModalNotReady(Exception):
    pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "LEDGER", tmp_path / "ledger.csv")
    monkeypatch.setattr(launch, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(launch, "INDEX", tmp_path / "runs" / "index.csv")
    monkeypatch.setattr(launch, "cfg", lambda: {"gcp": {"bucket": "b"}, "modal": {"app_name": "opa-experiments"}})
    state = {"get": TimeoutError(), "gcs": None, "gcs_calls": []}

    class FC:
        def get(self, timeout=None):
            g = state["get"]
            if isinstance(g, BaseException):
                raise g
            return g

    mod = types.ModuleType("modal")
    mod.FunctionCall = types.SimpleNamespace(from_id=lambda cid: FC())
    exc = types.ModuleType("modal.exception")
    exc.OutputExpiredError, exc.FunctionTimeoutError, exc.TimeoutError = OutputExpired, FnTimeout, ModalNotReady
    exc.ConnectionError = ModalConn
    mod.exception = exc
    monkeypatch.setitem(sys.modules, "modal", mod)
    monkeypatch.setitem(sys.modules, "modal.exception", exc)

    def get_bytes(uri):
        state["gcs_calls"].append(uri)
        if state["gcs"] is None:
            raise FileNotFoundError(uri)
        return json.dumps(state["gcs"]).encode()

    monkeypatch.setattr(gcs, "get_bytes", get_bytes)
    monkeypatch.setattr(gcs, "ls", lambda p: [])
    return state


def spawn(spawned_ago):
    """A reserved + spawned run, as cmd_run leaves it."""
    d = launch.RUNS / RID
    d.mkdir(parents=True)
    import time
    (d / "call.json").write_text(json.dumps({"call_id": "fc-1", "function": "run_job_gpu", "rate_per_second": RATE, "max_usd": MAX_USD,
                                             "envelope": "E2", "job": "j", "spawned_at": time.time() - spawned_ago, "timeout": TIMEOUT}))
    budget.append_ledger({"envelope": "E2", "job": "j", "kind": "reserve", "amount_usd": MAX_USD, "provider": "modal", "run_id": RID})
    launch.write_index([{"run_id": RID, "status": "running", "envelope": "E2"}])


def row():
    return next(r for r in launch.read_index() if r["run_id"] == RID)


def st():
    approvals = {"envelopes": [{"id": "E2", "approved_usd": 10, "approved_by": "x", "approved_on": "2026-01-01"}]}
    s = budget.envelope_states(approvals, budget.read_ledger())["E2"]
    return round(s.spent, 6), round(s.reserved, 6)


def ledger_kinds():
    return [r["kind"] for r in budget.read_ledger()]


def test_pending_beyond_call_timeout_stays_running(env):
    spawn(TIMEOUT + launch.LOST_GRACE_SECONDS + 3000)   # the old rule would have called this lost
    assert launch.collect(RID) == 0
    assert row()["status"] == "running"
    assert ledger_kinds() == ["reserve"]
    assert st() == (0, MAX_USD)


def test_gcs_result_finalizes_ok_with_its_cost(env):
    spawn(TIMEOUT + launch.LOST_GRACE_SECONDS + 3000)
    env["gcs"] = {"run_id": RID, "status": "ok", "compute_usd": 0.42, "api_usd": 0.0, "metrics": {"n": 3}, "manifest": {}}
    launch.collect(RID)
    assert row()["status"] == "ok" and float(row()["cost_usd"]) == pytest.approx(0.42)
    assert env["gcs_calls"] == [f"gs://b/opa/runs/{RID}/result.json"]
    assert st() == (pytest.approx(0.42), 0)
    assert (launch.RUNS / RID / "result.json").exists()


def test_gcs_failed_result_is_failed_not_lost(env):
    spawn(TIMEOUT + 5000)
    env["gcs"] = {"run_id": RID, "status": "failed", "compute_usd": 0.1, "metrics": {}, "manifest": {}}
    launch.collect(RID)
    assert row()["status"] == "failed" and st() == (pytest.approx(0.1), 0)


def test_beyond_24h_ceiling_is_lost(env):
    spawn(launch.MAX_TIMEOUT + launch.LOST_GRACE_SECONDS + 10)
    launch.collect(RID)
    assert row()["status"] == "lost"
    assert st() == (MAX_USD, 0)
    assert [r["note"] for r in budget.read_ledger() if r["kind"] == "actual"] == [launch.LOST_NOTE]
    assert not (launch.RUNS / RID / "result.json").exists()


def test_just_inside_ceiling_still_running(env):
    spawn(launch.MAX_TIMEOUT + launch.LOST_GRACE_SECONDS - 60)
    launch.collect(RID)
    assert row()["status"] == "running"


def make_premature_lost(env):
    """Reproduce what the old collect did: lost finalize at ~48 min."""
    spawn(2900)
    launch._finalize(RID, None, "lost", min(MAX_USD, 2900 * RATE), 0.0, launch.LOST_NOTE)
    assert row()["status"] == "lost"
    assert st() == (pytest.approx(min(MAX_USD, 2900 * RATE)), 0)


def test_recheck_lost_with_gcs_result(env):
    make_premature_lost(env)
    env["gcs"] = {"run_id": RID, "status": "ok", "compute_usd": 0.3, "api_usd": 0.0, "metrics": {}, "manifest": {}}
    assert launch.recheck(RID) == 0
    assert row()["status"] == "ok" and float(row()["cost_usd"]) == pytest.approx(0.3)
    assert st() == (pytest.approx(0.3), 0)    # premature actual cancelled, real cost charged, reserve released
    kinds = ledger_kinds()
    assert kinds.count("reserve") == 2 and kinds.count("release") == 2
    # second call: finalized normally now, refused, no new lines
    n = len(kinds)
    launch.recheck(RID)
    assert len(budget.read_ledger()) == n


def test_recheck_lost_still_pending(env):
    make_premature_lost(env)
    n0 = len(budget.read_ledger())
    launch.recheck(RID)
    assert row()["status"] == "running" and row()["cost_usd"] == ""
    assert st() == (0, MAX_USD)
    new = budget.read_ledger()[n0:]
    assert [r["kind"] for r in new] == ["reserve", "actual"]
    assert new[1]["note"] == launch.CORRECTION_NOTE and float(new[1]["amount_usd"]) < 0
    assert not (launch.RUNS / RID / "result.json").exists()
    # second recheck while still running: refused, ledger unchanged
    launch.recheck(RID)
    assert len(budget.read_ledger()) == n0 + 2
    # later the job really finishes
    env["get"] = {"status": "ok", "compute_usd": 0.6, "api_usd": 0.0, "metrics": {}, "manifest": {}}
    launch.collect(RID)
    assert row()["status"] == "ok" and st() == (pytest.approx(0.6), 0)


def test_recheck_removes_local_files_the_lost_finalize_wrote(env):
    make_premature_lost(env)
    d = launch.RUNS / RID
    for n in ("metrics.json", "manifest.json", "events_uri.json"):   # stray files; the lost finalize itself writes no result.json
        (d / n).write_text("{}")
    launch.recheck(RID)
    assert not any((d / n).exists() for n in ("metrics.json", "manifest.json", "events_uri.json"))


def test_recheck_refuses_normally_finalized(env):
    spawn(100)
    env["get"] = {"status": "ok", "compute_usd": 0.2, "api_usd": 0.0, "metrics": {}, "manifest": {}}
    launch.collect(RID)
    n = len(budget.read_ledger())
    launch.recheck(RID)
    assert len(budget.read_ledger()) == n and row()["status"] == "ok"


def test_recheck_refuses_real_timeout(env):
    spawn(100)
    env["get"] = FnTimeout("t")
    launch.collect(RID)
    assert row()["status"] == "timeout"
    n = len(budget.read_ledger())
    launch.recheck(RID)
    assert len(budget.read_ledger()) == n


def test_recheck_refuses_without_call_json(env):
    launch.RUNS.mkdir(parents=True)
    budget.append_ledger({"envelope": "E2", "job": "j", "kind": "actual", "amount_usd": 1.0, "provider": "modal", "run_id": RID,
                          "note": launch.LOST_NOTE})
    launch.write_index([{"run_id": RID, "status": "lost", "envelope": "E2"}])
    launch.recheck(RID)
    assert len(budget.read_ledger()) == 1


def test_recheck_refuses_lost_without_premature_note(env):
    """A lost finalize from an expired output with no GCS result is not undone."""
    spawn(100)
    env["get"] = OutputExpired("x")
    launch.collect(RID)
    assert row()["status"] == "lost"
    n = len(budget.read_ledger())
    launch.recheck(RID)
    assert len(budget.read_ledger()) == n


def test_premature_failed_behaviour_intact(env):
    spawn(100)
    launch._finalize(RID, {"status": "failed", "error": "TimeoutError()", "metrics": {}, "manifest": {}}, "failed", 0.4, 0.0,
                     launch.PREMATURE_NOTE)
    launch.recheck(RID)
    assert row()["status"] == "running" and st() == (0, MAX_USD)


# ---- hardening ----

OK = {"status": "ok", "compute_usd": 0.3, "api_usd": 0.0, "metrics": {}, "manifest": {}}


def test_transport_errors_leave_running(env):
    spawn(100)
    excs = [OSError("reset"), ModalConn("down")]
    try:
        import grpclib.exceptions as gx
        excs.append(gx.GRPCError(gx.Status.UNAVAILABLE))
    except ImportError:
        pass
    for e in excs:
        env["get"] = e
        launch.collect(RID)
        assert row()["status"] == "running" and ledger_kinds() == ["reserve"]
    assert not (launch.RUNS / RID / "finalizing.json").exists()


def test_genuine_function_failure_still_fails(env):
    spawn(100)
    env["get"] = RuntimeError("boom")
    launch.collect(RID)
    assert row()["status"] == "failed"


def test_gcs_result_must_match_run_id(env):
    spawn(TIMEOUT + 5000)
    env["gcs"] = {**OK, "run_id": "someone-else"}
    launch.collect(RID)
    assert row()["status"] == "running"
    env["gcs"] = {**OK, "run_id": RID}
    launch.collect(RID)
    assert row()["status"] == "ok"


def test_gcs_lookup_error_type_printed(env, monkeypatch, capsys):
    spawn(100)
    monkeypatch.setattr(gcs, "get_bytes", lambda u: (_ for _ in ()).throw(PermissionError("denied")))
    launch.collect(RID)
    out = capsys.readouterr().out
    assert "PermissionError" in out and row()["status"] == "running"


def test_output_expired_uses_gcs_result(env):
    spawn(100)
    env["get"] = OutputExpired("x")
    env["gcs"] = {**OK, "run_id": RID}
    launch.collect(RID)
    assert row()["status"] == "ok"


def test_recheck_refuses_genuine_loss_unless_gcs_result(env):
    spawn(launch.MAX_TIMEOUT + launch.LOST_GRACE_SECONDS + 10)
    launch.collect(RID)
    assert row()["status"] == "lost"
    n = len(budget.read_ledger())
    launch.recheck(RID)
    assert len(budget.read_ledger()) == n and row()["status"] == "lost"
    env["gcs"] = {**OK, "run_id": RID}
    launch.recheck(RID)
    assert row()["status"] == "ok" and st() == (pytest.approx(0.3), 0)


def test_finalize_crash_before_release_resumes_without_duplicates(env, monkeypatch):
    spawn(100)
    env["get"] = {**OK, "run_id": RID}
    real = budget.append_ledger

    def flaky(row_, path=None):
        if row_["kind"] == "release":
            raise OSError("disk")
        return real(row_, path)

    monkeypatch.setattr(budget, "append_ledger", flaky)
    with pytest.raises(OSError):
        launch.collect(RID)
    monkeypatch.setattr(budget, "append_ledger", real)
    assert ledger_kinds() == ["reserve", "actual"] and (launch.RUNS / RID / "finalizing.json").exists()
    assert not (launch.RUNS / RID / "result.json").exists()
    launch.collect(RID)
    assert ledger_kinds() == ["reserve", "actual", "release"]
    assert row()["status"] == "ok" and (launch.RUNS / RID / "result.json").exists()
    assert not (launch.RUNS / RID / "finalizing.json").exists()
    assert st() == (pytest.approx(0.3), 0)
    launch.collect(RID)
    assert len(budget.read_ledger()) == 3


def test_finalize_crash_before_index_resumes(env, monkeypatch):
    spawn(100)
    env["get"] = {**OK, "run_id": RID}
    real = launch.write_index
    monkeypatch.setattr(launch, "write_index", lambda rows: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        launch.collect(RID)
    monkeypatch.setattr(launch, "write_index", real)
    launch.collect(RID)
    assert ledger_kinds() == ["reserve", "actual", "release"] and row()["status"] == "ok"


def test_lost_finalize_cannot_apply_twice(env):
    spawn(launch.MAX_TIMEOUT + launch.LOST_GRACE_SECONDS + 10)
    launch.collect(RID)
    launch.collect(RID)
    launch.collect(RID)
    assert ledger_kinds() == ["reserve", "actual", "release"]


def test_recheck_crash_after_reserve_is_idempotent(env, monkeypatch):
    make_premature_lost(env)
    real = budget.append_ledger

    def flaky(row_, path=None):
        if row_["note"] == launch.CORRECTION_NOTE:
            raise OSError("disk")
        return real(row_, path)

    monkeypatch.setattr(budget, "append_ledger", flaky)
    with pytest.raises(OSError):
        launch.recheck(RID)
    monkeypatch.setattr(budget, "append_ledger", real)
    assert ledger_kinds().count("reserve") == 2
    launch.recheck(RID)
    assert ledger_kinds().count("reserve") == 2 and ledger_kinds().count("actual") == 2
    assert row()["status"] == "running" and st() == (0, MAX_USD)


def test_recheck_crash_before_index_reset_resumes(env, monkeypatch):
    make_premature_lost(env)
    real = launch.write_index
    monkeypatch.setattr(launch, "write_index", lambda rows: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        launch.recheck(RID)
    monkeypatch.setattr(launch, "write_index", real)
    assert row()["status"] == "lost" and st() == (0, MAX_USD)    # ledger fully corrected, index still lost
    n = len(budget.read_ledger())
    launch.recheck(RID)
    assert len(budget.read_ledger()) == n and row()["status"] == "running"
    launch.recheck(RID)    # now plain running: refused
    assert len(budget.read_ledger()) == n


# ---- round 3 ----

def test_transport_error_with_gcs_result_finalizes(env):
    spawn(100)
    env["get"] = ModalConn("down")
    env["gcs"] = {**OK, "run_id": RID}
    launch.collect(RID)
    assert row()["status"] == "ok" and st() == (pytest.approx(0.3), 0)


def test_transport_error_past_ceiling_warns_but_does_not_finalize(env, capsys):
    spawn(launch.MAX_TIMEOUT + launch.LOST_GRACE_SECONDS + 10)
    env["get"] = OSError("net")
    launch.collect(RID)
    out = capsys.readouterr().out
    assert f"needs a human: {RID}" in out and "poll error OSError; left running" in out
    assert row()["status"] == "running" and ledger_kinds() == ["reserve"]
    # a young run: no warning
    launch.RUNS.joinpath(RID, "call.json").write_text(
        json.dumps({**json.loads((launch.RUNS / RID / "call.json").read_text()), "spawned_at": __import__("time").time() - 60}))
    launch.collect(RID)
    assert "needs a human" not in capsys.readouterr().out


def test_recheck_resets_index_before_deleting_files(env, monkeypatch):
    spawn(100)
    launch._finalize(RID, {"status": "failed", "error": "TimeoutError()", "metrics": {}, "manifest": {}}, "failed", 0.4, 0.0,
                     launch.PREMATURE_NOTE)
    d = launch.RUNS / RID
    seen = {}
    real = launch.write_index

    def spy(rows):
        seen["result_present"] = (d / "result.json").exists()
        seen["status"] = [r["status"] for r in rows if r["run_id"] == RID]
        return real(rows)

    monkeypatch.setattr(launch, "write_index", spy)
    launch.recheck(RID)
    assert seen == {"result_present": True, "status": ["running"]}
    assert not (d / "result.json").exists() and row()["status"] == "running"


def test_recheck_failed_branch_resumes_after_crash_between_index_and_delete(env, monkeypatch):
    spawn(100)
    launch._finalize(RID, {"status": "failed", "error": "TimeoutError()", "metrics": {}, "manifest": {}}, "failed", 0.4, 0.0,
                     launch.PREMATURE_NOTE)
    real_collect = launch.collect
    monkeypatch.setattr(launch, "collect", lambda *a, **k: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        launch.recheck(RID)       # everything done up to collect
    monkeypatch.setattr(launch, "collect", real_collect)
    n = len(budget.read_ledger())
    (launch.RUNS / RID / "result.json").write_text(json.dumps({"status": "failed", "error": "TimeoutError()"}))   # crash left it behind
    launch.recheck(RID)
    assert len(budget.read_ledger()) == n and not (launch.RUNS / RID / "result.json").exists()


def test_marker_written_atomically(env, monkeypatch):
    spawn(100)
    seen = []
    real = launch.os.replace
    monkeypatch.setattr(launch.os, "replace", lambda a, b: (seen.append((str(a), str(b))), real(a, b))[1])
    env["get"] = {**OK, "run_id": RID}
    launch.collect(RID)
    assert len(seen) == 1 and seen[0][0].endswith("finalizing.json.tmp") and seen[0][1].endswith("finalizing.json")
    assert not (launch.RUNS / RID / "finalizing.json.tmp").exists()


def test_collect_all_continues_after_error(env, monkeypatch, capsys):
    spawn(100)
    launch.write_index([{"run_id": "bad", "status": "running"}, {"run_id": RID, "status": "running", "envelope": "E2"}])
    calls = []

    def fake(rid, block=False):
        calls.append(rid)
        if rid == "bad":
            raise ValueError("kaput")
        return 0

    monkeypatch.setattr(launch, "collect", fake)
    assert launch.main(["collect", "--all"]) == 0
    assert calls == ["bad", RID]
    assert "bad: collect error ValueError: kaput" in capsys.readouterr().out


def test_fresh_lock_skips_and_leaves_everything(env, capsys):
    spawn(100)
    env["get"] = {**OK, "run_id": RID}
    lock = launch.RUNS / RID / ".finalize.lock"
    lock.write_text("")
    assert launch.collect(RID) == 0
    assert f"{RID}: another collect is finalizing; skipped" in capsys.readouterr().out
    assert lock.exists() and ledger_kinds() == ["reserve"] and row()["status"] == "running"
    assert not (launch.RUNS / RID / "finalizing.json").exists()


def test_stale_lock_is_replaced_and_removed_after(env):
    import os
    import time
    spawn(100)
    env["get"] = {**OK, "run_id": RID}
    lock = launch.RUNS / RID / ".finalize.lock"
    lock.write_text("")
    old = time.time() - 2 * launch.LOCK_STALE_SECONDS
    os.utime(lock, (old, old))
    launch.collect(RID)
    assert row()["status"] == "ok" and not lock.exists()


def test_lock_removed_even_when_finalize_crashes(env, monkeypatch):
    spawn(100)
    env["get"] = {**OK, "run_id": RID}
    monkeypatch.setattr(launch, "write_index", lambda rows: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        launch.collect(RID)
    assert not (launch.RUNS / RID / ".finalize.lock").exists()


# ---- overlapping collects, cancellation, stale-lock rename ----

def _overlap(env, spawned_ago, first_get):
    """B polls (first_get); while B waits, A runs a complete collect; then B returns first_get's outcome."""
    n = {"i": 0}
    real_from_id = sys.modules["modal"].FunctionCall.from_id

    class Plain:
        def get(self, timeout=None):
            return first_get_result() if callable(first_get_result) else first_get_result

    first_get_result = None

    class B:
        def get(self, timeout=None):
            n["i"] += 1
            if n["i"] == 1:
                sys.modules["modal"].FunctionCall.from_id = real_from_id
                launch.collect(RID)       # A, complete
                sys.modules["modal"].FunctionCall.from_id = lambda cid: B()
            return first_get()

    return B


def test_overlap_ok_finalizes_once(env):
    spawn(100)
    res = {**OK, "run_id": RID}
    env["get"] = res                       # A gets the result straight away
    B = _overlap(env, 100, lambda: res)
    sys.modules["modal"].FunctionCall.from_id = lambda cid: B()
    launch.collect(RID)
    assert ledger_kinds() == ["reserve", "actual", "release"]
    assert st() == (pytest.approx(0.3), 0)


def test_overlap_lost_past_ceiling_finalizes_once(env):
    spawn(launch.MAX_TIMEOUT + launch.LOST_GRACE_SECONDS + 10)
    env["get"] = TimeoutError()            # A: pending -> lost

    def b_result():
        raise TimeoutError()

    B = _overlap(env, 0, b_result)
    sys.modules["modal"].FunctionCall.from_id = lambda cid: B()
    launch.collect(RID)
    assert ledger_kinds() == ["reserve", "actual", "release"]
    assert row()["status"] == "lost" and st() == (MAX_USD, 0)


def test_finalize_never_releases_more_than_reserved(env):
    spawn(100)
    launch._finalize(RID, {**OK, "run_id": RID}, "ok", 0.3, 0.0)
    launch._finalize(RID, {**OK, "run_id": RID}, "ok", 0.3, 0.0)
    assert ledger_kinds() == ["reserve", "actual", "release"]


def test_recheck_still_works_with_skip_guard(env):
    make_premature_lost(env)
    env["gcs"] = {**OK, "run_id": RID}
    launch.recheck(RID)
    assert row()["status"] == "ok" and st() == (pytest.approx(0.3), 0)


def test_input_cancellation_finalizes_failed(env):
    class Cancel(BaseException):
        pass

    sys.modules["modal.exception"].InputCancellation = Cancel
    spawn(100)
    env["get"] = Cancel()
    launch.collect(RID)
    assert row()["status"] == "failed" and ledger_kinds() == ["reserve", "actual", "release"]


def test_collect_all_catches_cancellation(env, monkeypatch, capsys):
    class Cancel(BaseException):
        pass

    sys.modules["modal.exception"].InputCancellation = Cancel
    launch.write_index([{"run_id": "a", "status": "running"}, {"run_id": "b", "status": "running"}])
    calls = []

    def fake(rid, block=False):
        calls.append(rid)
        if rid == "a":
            raise Cancel()
        return 0

    monkeypatch.setattr(launch, "collect", fake)
    launch.main(["collect", "--all"])
    assert calls == ["a", "b"] and "a: collect error Cancel" in capsys.readouterr().out


def test_stale_lock_renamed_not_deleted(env):
    import os
    import time
    spawn(100)
    env["get"] = {**OK, "run_id": RID}
    lock = launch.RUNS / RID / ".finalize.lock"
    lock.write_text("")
    old = time.time() - 2 * launch.LOCK_STALE_SECONDS
    os.utime(lock, (old, old))
    launch.collect(RID)
    assert row()["status"] == "ok" and not lock.exists()
    assert list((launch.RUNS / RID).glob(".finalize.lock.stale-*"))


def test_skipped_finalize_does_not_print_success(env, capsys):
    spawn(100)
    env["get"] = {**OK, "run_id": RID}
    launch.collect(RID)
    capsys.readouterr()
    assert launch._finalize(RID, {**OK, "run_id": RID}, "ok", 0.3, 0.0) is False
    out = capsys.readouterr().out
    assert "already finalized by another collect; skipped" in out and ": ok" not in out


def test_overlap_prints_no_success_for_loser(env, capsys):
    spawn(100)
    res = {**OK, "run_id": RID}
    env["get"] = res
    B = _overlap(env, 100, lambda: res)
    sys.modules["modal"].FunctionCall.from_id = lambda cid: B()
    launch.collect(RID)
    out = capsys.readouterr().out
    assert out.count(f"{RID}: ok") == 1 and "already finalized by another collect" in out


def test_lock_holds_pid(env):
    import os
    spawn(100)
    seen = []
    env["get"] = {**OK, "run_id": RID}
    real = launch._apply_finalize
    launch._apply_finalize = lambda rid: (seen.append((launch.RUNS / rid / ".finalize.lock").read_text()), real(rid))[1]
    try:
        launch.collect(RID)
    finally:
        launch._apply_finalize = real
    assert seen == [str(os.getpid())]
