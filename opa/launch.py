"""The only way to spend money in this project.

  python -m opa.launch deploy                    deploy the Modal app (needs at least one approved envelope)
  python -m opa.launch run jobs/<job>.yaml       check budget, reserve max_usd, spawn the job on Modal, return at once
  python -m opa.launch run jobs/<job>.yaml --wait   same, then block until it finishes and collect it
  python -m opa.launch collect <run_id>|--all    fetch finished jobs, record actual cost, release reservations
  python -m opa.launch collect --recheck <run_id>  undo a premature finalize (see recheck())
  python -m opa.launch release-orphan <run_id>   release a reservation whose spawn never happened (no call.json)
  python -m opa.launch list                      show running jobs

Job YAML fields (see jobs/TEMPLATE.yaml):
  name, envelope, track, hypotheses, system, dataset, cell, split (dev|test), seed,
  entrypoint ("module:function"), args, models,
  compute: {kind: cpu|gpu, gpu: L4|T4, image: default|vllm, cpu: 1.0, memory_gib: 4},
  estimate_usd, max_usd, max_api_usd (must be present and 0: no Anthropic API in v3, PLAN_V3.md)
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import contextlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import PROJECT_DIR, budget

RUNS = PROJECT_DIR / "runs"
INDEX = RUNS / "index.csv"
INDEX_FIELDS = ["run_id", "track", "hypotheses", "system", "dataset", "cell", "split", "seed", "started_at", "finished_at",
                "status", "cost_usd", "envelope", "gcs_uri", "git_sha", "call_id"]
REQUIRED = ["name", "envelope", "track", "hypotheses", "split", "seed", "entrypoint", "compute", "estimate_usd", "max_usd"]
ALLOWED_GPUS = ("L4", "T4")  # v3: no A100/H100/B200/L40S/A10 (CLAUDE.md Section 4 rule 9, PLAN_V3.md)
MAX_TIMEOUT = 86400         # Modal's per-call limit; MAX_TIMEOUT + LOST_GRACE_SECONDS after the spawn is a heuristic bound
                            # on queue time + execution (Modal's own timeout counts from execution start)
LOST_GRACE_SECONDS = 1800    # slack added to MAX_TIMEOUT before an unfinished call is recorded as lost


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True).stdout.strip()


def prereg_tagged() -> bool:
    return bool(git("tag", "-l", "prereg-v2"))


def lock_sha() -> str:
    for name in ("requirements.lock", "uv.lock", "requirements.txt"):
        p = PROJECT_DIR / name
        if p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()
    return ""


def cfg() -> dict:
    c = yaml.safe_load((PROJECT_DIR / "config" / "project.yaml").read_text())
    if (c.get("gcp") or {}).get("bucket"):
        from .gcs import normalize_bucket
        c["gcp"]["bucket"] = normalize_bucket(c["gcp"]["bucket"])   # "gs://name/" and "name" both work
    return c


def read_index() -> list[dict]:
    if not INDEX.exists():
        return []
    with INDEX.open(newline="") as f:
        return list(csv.DictReader(f))


def write_index(rows: list[dict]) -> None:
    RUNS.mkdir(exist_ok=True)
    with INDEX.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in INDEX_FIELDS})


LIMIT_FACTOR = 2.0   # Modal bills max(request, usage) up to the limit, so the limit is the worst case and prices the timeout


def resources(compute: dict) -> dict:
    """CPU cores and memory GiB as (request, limit). Limits default to LIMIT_FACTOR x request (compute.cpu_limit / memory_gib_limit override)."""
    cpu = max(float(compute.get("cpu", 1.0)), 0.125)
    mem = float(compute.get("memory_gib", 4))
    return {"cpu": cpu, "cpu_limit": max(cpu, float(compute.get("cpu_limit", cpu * LIMIT_FACTOR))),
            "memory_gib": mem, "memory_gib_limit": max(mem, float(compute.get("memory_gib_limit", mem * LIMIT_FACTOR)))}


def rate_per_second(compute: dict, prices: dict) -> float:
    """Worst-case $/s: GPU + CPU and memory priced at their limits."""
    p = prices["modal"]
    r = resources(compute)
    rate = p["cpu_core_per_second"] * r["cpu_limit"] + p["memory_gib_per_second"] * r["memory_gib_limit"]
    if compute.get("kind") == "gpu":
        rate += p["gpu_per_second"][compute.get("gpu") or "L4"]
    return rate


def make_run_id(job: dict, sha: str) -> str:
    """<track>-<system>-<dataset>-<cell>-<split>-s<seed>-<UTC yyyymmddHHMMSS>-<git sha7>-<4 hex>; unique even for repeats."""
    import secrets
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    while True:
        parts = [job["track"], job.get("system", "x"), job.get("dataset", "x"), job.get("cell", "x"), job["split"], f"s{job['seed']}",
                 ts, sha[:7], secrets.token_hex(2)]
        rid = "-".join(str(p).replace("/", "_").replace(" ", "") for p in parts)
        if not (RUNS / rid).exists():
            return rid


PLACEHOLDER = re.compile(r"\b[A-Z0-9_]+_RUN_ID\b")


TEST_PATH = re.compile(r"(?:^|/)test/")        # a /test/ path segment in a URI or path (not "tests/")


def _strings(x):
    """Every string inside nested dicts / lists (args values, URIs, embedder specs, ...)."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def validate(job: dict) -> list[str]:
    errs = [f"missing field: {k}" for k in REQUIRED if k not in job]
    ph = sorted(set(PLACEHOLDER.findall(yaml.safe_dump(job, default_flow_style=False))))
    if re.search(r"gs://BUCKET(/|)", yaml.safe_dump(job)):
        errs.append("spec still contains the template placeholder gs://BUCKET/ (use the real bucket from config/project.yaml)")
    if ph:
        errs.append(f"spec still contains unfilled placeholder(s): {', '.join(ph)}")
    if job.get("split") not in ("dev", "test"):
        errs.append("split must be dev or test")
    if job.get("split") != "test":
        bad = sorted({s for s in _strings(job.get("args")) if TEST_PATH.search(s)})
        if bad:
            errs.append("a non-test spec references test-split data (a /test/ path segment) in its args: " + ", ".join(bad[:3])
                        + "; only split: test specs may read test data (and then only after the prereg-v2 tag)")
    if float(job.get("max_usd", 0)) < float(job.get("estimate_usd", 0)):
        errs.append("max_usd must be >= estimate_usd")
    if "max_api_usd" not in job:
        errs.append("max_api_usd is required and must be 0 (no Anthropic API in v3 experiments, PLAN_V3.md)")
    else:
        try:
            api = float(job["max_api_usd"])
        except (TypeError, ValueError):
            api = None
        if api != 0:
            errs.append("max_api_usd must be 0 (no Anthropic API in v3 experiments, PLAN_V3.md)")
    comp = job.get("compute") or {}
    if comp.get("kind") not in ("cpu", "gpu"):
        errs.append("compute.kind must be cpu or gpu")
    if comp.get("image", "default") not in ("default", "vllm"):
        errs.append("compute.image must be default or vllm")
    if comp.get("image") == "vllm" and comp.get("kind") != "gpu":
        errs.append("compute.image vllm needs compute.kind gpu")
    if comp.get("kind") == "gpu":
        gpu = comp.get("gpu") or "L4"
        if gpu not in ALLOWED_GPUS:
            errs.append(f"compute.gpu must be one of {list(ALLOWED_GPUS)}, got {gpu!r}; "
                        "bigger GPUs are not available in v3 (CLAUDE.md Section 4 rule 9)")
        if comp.get("image") == "vllm" and gpu != "L4":
            errs.append("compute.image vllm runs on L4 only")
    return errs


def function_name(compute: dict) -> str:
    """Modal function for a validated compute block."""
    if compute.get("kind") != "gpu":
        return "run_job_cpu"
    if compute.get("image") == "vllm":
        return "run_job_vllm"
    return "run_job_t4" if (compute.get("gpu") or "L4") == "T4" else "run_job_gpu"


def deploy_command() -> list[str]:
    """`python -m modal` rather than the modal.exe launcher, whose uv trampoline can be broken."""
    return [sys.executable, "-m", "modal", "deploy", "-m", "opa.modal_app"]


def cmd_deploy() -> int:
    if not budget.approved_envelopes(budget.load_approvals()):
        print("No approved envelope yet: the human must approve E0 in budget/approvals.yaml.")
        return 3
    r = subprocess.run(deploy_command(), cwd=PROJECT_DIR)
    return r.returncode


def _flat(e: BaseException) -> str:
    return " ".join(f"{type(e).__name__}: {e}".split())[:200]


def cmd_run(job_path: str, wait: bool = False) -> int:
    job = yaml.safe_load(Path(job_path).read_text())
    errs = validate(job)
    if errs:
        print("Invalid job:\n  " + "\n  ".join(errs))
        return 2
    if job["split"] == "test" and not prereg_tagged():
        print("Refused: test-split jobs need the prereg-v2 tag first (CLAUDE.md Section 6).")
        return 4
    if job["split"] == "test" and git("status", "--porcelain"):
        print("Refused: commit your changes before a test-split job, so the run's git SHA is reproducible.")
        return 4
    max_usd = float(job["max_usd"])
    ok, msg = budget.check(job["envelope"], max_usd)
    print(msg)
    if not ok:
        return 3
    prices = budget.load_prices()
    compute = job["compute"]
    rate = rate_per_second(compute, prices)
    max_compute_usd = max_usd - float(job.get("max_api_usd", 0))
    timeout = int(min(MAX_TIMEOUT, max_compute_usd / rate))
    if timeout < 60:
        print(f"Refused: max_usd leaves only {timeout}s of compute at ${rate*3600:.2f}/h.")
        return 2
    sha = git("rev-parse", "HEAD") or "nogit"
    run_id = make_run_id(job, sha)
    c = cfg()
    spec = {**job, "run_id": run_id, "bucket": c["gcp"]["bucket"], "git_sha": sha, "lock_sha256": lock_sha(),
            "_prices": prices, "_rate_per_second": rate, "_gpu": (compute.get("gpu") or "L4") if compute.get("kind") == "gpu" else None, "_started_at": budget.now()}
    import modal   # a failed import happens before any reservation
    fn_name = function_name(compute)
    r = resources(compute)
    opts = {"timeout": timeout, "cpu": (r["cpu"], r["cpu_limit"]),
            "memory": (int(1024 * r["memory_gib"]), int(1024 * r["memory_gib_limit"])), "scaledown_window": 2}
    if compute.get("kind") == "gpu":
        opts["gpu"] = compute.get("gpu") or "L4"
    try:   # free steps first: nothing is reserved if the lookup or hydration fails
        fo = modal.Function.from_name(c["modal"]["app_name"], fn_name).with_options(**opts)
        fo.hydrate()
    except Exception as e:
        print(f"Function lookup failed, nothing reserved, nothing ran: {_flat(e)}")
        return 5
    budget.append_ledger({"envelope": job["envelope"], "job": job["name"], "kind": "reserve", "amount_usd": max_usd,
                          "provider": "modal", "run_id": run_id, "note": f"estimate {job['estimate_usd']}"})
    try:
        call = fo.spawn(spec)
    except Exception as e:
        msg = _flat(e)
        mx = modal.exception
        if isinstance(e, (mx.NotFoundError, mx.InvalidError, TypeError, AttributeError, pickle.PicklingError)):
            budget.append_ledger({"envelope": job["envelope"], "job": job["name"], "kind": "release", "amount_usd": max_usd,
                                  "provider": "modal", "run_id": run_id, "note": f"spawn failed: {msg}; nothing ran"})
            print(f"Spawn failed before submission, reservation released, nothing ran: {msg}")
        else:
            d = RUNS / run_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "spawn_error.json").write_text(json.dumps({"error": msg, "ts": time.time(), "timeout": timeout}, indent=2))
            print(f"Spawn failed, outcome unknown ({msg}); reservation kept. Check Modal; "
                  f"release-orphan allowed after ts+timeout+3600 ({timeout + 3600}s from now).")
        return 5
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "call.json").write_text(json.dumps({"call_id": call.object_id, "function": fn_name, "options": opts, "rate_per_second": rate,
                                             "max_usd": max_usd, "envelope": job["envelope"], "job": job["name"],
                                             "spawned_at": time.time(), "timeout": timeout}, indent=2))
    (d / "job.yaml").write_text(yaml.safe_dump(job, sort_keys=False))
    rows = read_index()
    rows.append({"run_id": run_id, "track": job["track"], "hypotheses": ",".join(job["hypotheses"]), "system": job.get("system", ""),
                 "dataset": job.get("dataset", ""), "cell": job.get("cell", ""), "split": job["split"], "seed": job["seed"],
                 "started_at": spec["_started_at"], "status": "running", "envelope": job["envelope"],
                 "gcs_uri": f"gs://{c['gcp']['bucket']}/opa/runs/{run_id}", "git_sha": sha, "call_id": call.object_id})
    write_index(rows)
    print(f"spawned {run_id} (call {call.object_id}, timeout {timeout}s)")
    return collect(run_id, block=True) if wait else 0


def _run_ledger(run_id: str) -> list[dict]:
    return [r for r in budget.read_ledger() if r["run_id"] == run_id]


def _finalize(run_id: str, result: dict | None, status: str, cost_compute: float, cost_api: float, note: str = "") -> bool:
    """Returns False when the run was already finalized and nothing was written.
    Crash-safe: a finalizing.json marker (with the arguments and the run's ledger line count) is written first,
    then the ledger lines, the index row and the local result files (result.json last), and the marker is removed.
    A crash at any point is resumed by collect() from the marker, never applied twice."""
    d = RUNS / run_id
    marker = d / "finalizing.json"
    with _finalize_lock(run_id):
        if not marker.exists():
            net = sum(float(r["amount_usd"]) * (1 if r["kind"] == "reserve" else -1)
                      for r in _run_ledger(run_id) if r["kind"] in ("reserve", "release"))
            row = next((r for r in read_index() if r["run_id"] == run_id), None)
            if net <= 1e-9 or (d / "result.json").exists() or (row is not None and row.get("status") != "running"):
                print(f"{run_id}: already finalized by another collect; skipped")   # never release more than was reserved
                return False
            tmp = d / "finalizing.json.tmp"
            tmp.write_text(json.dumps({"result": result, "status": status, "cost_compute": cost_compute, "cost_api": cost_api,
                                       "note": note, "n_before": len(_run_ledger(run_id))}, indent=2, default=str))
            os.replace(tmp, marker)      # atomic: a reader never sees a half-written marker
        _apply_finalize(run_id)
    return True


LOCK_STALE_SECONDS = 3600


class _FinalizeBusy(Exception):
    pass


@contextlib.contextmanager
def _finalize_lock(run_id: str):
    """Per-run lock so two collects never finalize the same run at once. A lock older than an hour is stale (a killed process)."""
    lock = RUNS / run_id / ".finalize.lock"
    pid = str(os.getpid())

    def take() -> bool:
        with open(lock, "x") as f:
            f.write(pid)
        return lock.read_text() == pid      # re-read: another process taking over a stale lock may have won

    try:
        if not take():
            raise FileExistsError
    except FileExistsError:
        try:
            age = time.time() - lock.stat().st_mtime
        except FileNotFoundError:
            age = LOCK_STALE_SECONDS + 1
        if age < LOCK_STALE_SECONDS:
            print(f"{run_id}: another collect is finalizing; skipped")
            raise _FinalizeBusy()
        try:   # take a stale lock over by renaming it away (atomic), not by delete + create
            os.replace(lock, lock.with_name(f".finalize.lock.stale-{os.getpid()}-{int(time.time())}"))
        except FileNotFoundError:
            pass
        try:
            if not take():
                raise FileExistsError
        except FileExistsError:
            print(f"{run_id}: another collect is finalizing; skipped")
            raise _FinalizeBusy()
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _apply_finalize(run_id: str) -> None:
    d = RUNS / run_id
    marker = d / "finalizing.json"
    m = json.loads(marker.read_text())
    call = json.loads((d / "call.json").read_text())
    env, job = call["envelope"], call["job"]
    done = _run_ledger(run_id)[m["n_before"]:]      # lines a crashed earlier attempt already wrote
    if not any(r["kind"] == "actual" and r["provider"] == "modal" for r in done):
        budget.append_ledger({"envelope": env, "job": job, "kind": "actual", "amount_usd": m["cost_compute"], "provider": "modal", "run_id": run_id, "note": m["note"]})
    if m["cost_api"] and not any(r["kind"] == "actual" and r["provider"] == "anthropic" for r in done):
        budget.append_ledger({"envelope": env, "job": job, "kind": "actual", "amount_usd": m["cost_api"], "provider": "anthropic", "run_id": run_id, "note": m["note"]})
    if not any(r["kind"] == "release" for r in done):
        budget.append_ledger({"envelope": env, "job": job, "kind": "release", "amount_usd": call["max_usd"], "provider": "modal", "run_id": run_id})
    rows = read_index()
    for r in rows:
        if r["run_id"] == run_id:
            r.update(finished_at=budget.now(), status=m["status"], cost_usd=f"{m['cost_compute'] + m['cost_api']:.4f}")
    write_index(rows)
    result = m["result"]
    if result is not None:
        (d / "metrics.json").write_text(json.dumps(result.get("metrics", {}), indent=2))
        (d / "manifest.json").write_text(json.dumps(result.get("manifest", {}), indent=2))
        if result.get("events"):
            (d / "events_uri.json").write_text(json.dumps(result["events"], indent=2))
        (d / "result.json").write_text(json.dumps(result, indent=2, default=str))   # last: its presence means "finalized"
    marker.unlink()


LOST_NOTE = "no result long after timeout; cost estimated from elapsed time"


def _gcs_result(run_id: str) -> dict | None:
    """The result.json every job writes to GCS for this run, or None when it is not (yet) there."""
    try:
        from . import gcs
        r = json.loads(gcs.get_bytes(f"gs://{cfg()['gcp']['bucket']}/opa/runs/{run_id}/result.json"))
    except Exception as e:
        if not (isinstance(e, FileNotFoundError) or type(e).__name__ in ("NotFound", "NoSuchKey")):
            print(f"{run_id}: GCS result lookup error {type(e).__name__}; treated as no result yet")
        return None
    return r if isinstance(r, dict) and r.get("run_id") == run_id else None


def _cancel_excs() -> tuple:
    """Defensive catch only: InputCancellation is a BaseException (so `except Exception` would miss it), but in modal
    1.6.3 a cancellation reaches the client as RemoteError and lands in the crash branch."""
    try:
        import modal.exception as mx
        return (mx.InputCancellation,) if hasattr(mx, "InputCancellation") else ()
    except ImportError:
        return ()


def _transport_excs() -> tuple:
    """Errors that mean "could not reach Modal", not "the function failed"."""
    out: list = [OSError]
    try:
        import modal.exception as mx
        if hasattr(mx, "ConnectionError"):
            out.append(mx.ConnectionError)
    except ImportError:
        pass
    try:
        import grpclib.exceptions as gx
        out.append(gx.GRPCError)
    except ImportError:
        pass
    return tuple(out)


def collect(run_id: str, block: bool = False) -> int:
    try:
        return _collect(run_id, block)
    except _FinalizeBusy:
        return 0


def _collect(run_id: str, block: bool = False) -> int:
    import modal
    d = RUNS / run_id
    if not (d / "call.json").exists():
        return 0
    if (d / "finalizing.json").exists():          # a crashed finalize: resume it, never redo it
        with _finalize_lock(run_id):
            if (d / "finalizing.json").exists():
                _apply_finalize(run_id)
        print(f"{run_id}: resumed an interrupted finalize")
        return 0
    if (d / "result.json").exists():
        return 0
    row = next((r for r in read_index() if r["run_id"] == run_id), None)
    if row is not None and row.get("status") != "running":    # already finalized (e.g. lost): only recheck may reopen it
        return 0
    from modal.exception import FunctionTimeoutError, OutputExpiredError
    from modal.exception import TimeoutError as ModalNotReady   # Modal's own class, not Python's TimeoutError

    call = json.loads((d / "call.json").read_text())
    fc = modal.FunctionCall.from_id(call["call_id"])
    elapsed = time.time() - call["spawned_at"]
    try:
        result = fc.get(timeout=None if block else 0)
    except OutputExpiredError:
        # Modal no longer holds the output; every job also wrote result.json to GCS.
        result = _gcs_result(run_id)
        if result is None:
            cost = min(call["max_usd"], min(elapsed, call["timeout"]) * call["rate_per_second"])
            if not _finalize(run_id, None, "lost", cost, 0.0, "output expired and no result.json in GCS; cost estimated"):
                return 0
            print(f"{run_id}: lost")
            return 0
    except FunctionTimeoutError as e:   # Modal killed the container at the budget-derived timeout
        cost = min(call["max_usd"], call["timeout"] * call["rate_per_second"])
        if not _finalize(run_id, {"status": "timeout", "error": repr(e), "metrics": {}, "manifest": {}}, "timeout", cost, 0.0, "hit budget timeout; cost estimated"):
            return 0
        print(f"{run_id}: timeout")
        return 0
    except (ModalNotReady, TimeoutError):   # Modal 1.6 raises the builtin TimeoutError for an unfinished call
        # Modal's timeout counts from execution start, not from spawn, so a queued call can legitimately be
        # "not ready" far past spawn + timeout. Trust a result.json in GCS first; otherwise a call Modal still
        # reports as pending/running stays running until MAX_TIMEOUT + LOST_GRACE_SECONDS after the spawn.
        result = _gcs_result(run_id)
        if result is None:
            if elapsed > MAX_TIMEOUT + LOST_GRACE_SECONDS:
                cost = min(call["max_usd"], elapsed * call["rate_per_second"])
                if not _finalize(run_id, None, "lost", cost, 0.0, LOST_NOTE):
                    return 0
                print(f"{run_id}: lost")
            else:
                print(f"{run_id}: still running or queued ({elapsed/60:.0f} min)")
            return 0
    except _cancel_excs() as e:   # the call was cancelled: nothing is running any more
        cost = min(call["max_usd"], min(elapsed, call["timeout"]) * call["rate_per_second"])
        if not _finalize(run_id, {"status": "failed", "error": repr(e), "metrics": {}, "manifest": {}}, "failed", cost, 0.0, "cancelled; nothing running; cost estimated"):
            return 0
        print(f"{run_id}: failed (cancelled)")
        return 0
    except _transport_excs() as e:   # could not reach Modal: say nothing about the job unless GCS has its result
        result = _gcs_result(run_id)
        if result is None:
            print(f"{run_id}: poll error {type(e).__name__}; left running")
            if elapsed > MAX_TIMEOUT + LOST_GRACE_SECONDS:
                print(f"!!! needs a human: {run_id} (cannot reach Modal and no result in GCS, {elapsed/3600:.0f} h after spawn)")
            return 0
    except Exception as e:   # the container crashed
        cost = min(call["max_usd"], min(elapsed, call["timeout"]) * call["rate_per_second"])
        if not _finalize(run_id, {"status": "failed", "error": repr(e), "metrics": {}, "manifest": {}}, "failed", cost, 0.0, "crash; cost estimated"):
            return 0
        print(f"{run_id}: failed ({e!r})")
        return 0
    status = "ok" if result.get("status") == "ok" else result.get("status", "failed")
    if not _finalize(run_id, result, status, float(result.get("compute_usd", 0)), float(result.get("api_usd", 0))):
        return 0
    print(f"{run_id}: {status}  compute ${result.get('compute_usd', 0):.3f}  api ${result.get('api_usd', 0):.3f}")
    return 0


PREMATURE_NOTE = "crash; cost estimated"
CORRECTION_NOTE = "correction: premature finalize by collect bug"
RERESERVE_NOTE = "re-reserve after premature finalize by collect bug"
NOTES = (PREMATURE_NOTE, LOST_NOTE, CORRECTION_NOTE)


def _ts(s: str) -> float:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def recheck(run_id: str) -> int:
    """Undo a premature finalize (collect once treated a still-running call as a crash, or as lost while it was only
    queued), then collect again for real. Safe to repeat after a crash: a re-reserve or correction already written
    is not written twice. Local runs/ files and append-only ledger lines only; nothing in GCS is touched."""
    d = RUNS / run_id
    res_path = d / "result.json"
    if not (d / "call.json").exists():
        print(f"{run_id}: nothing to recheck")
        return 0
    if (d / "finalizing.json").exists():
        return collect(run_id)       # resumes the interrupted finalize
    ledger = _run_ledger(run_id)
    prem = round(sum(float(r["amount_usd"]) for r in ledger if r["kind"] == "actual" and r["note"] in NOTES), 6)
    net_res = round(sum(float(r["amount_usd"]) * (1 if r["kind"] == "reserve" else -1) for r in ledger if r["kind"] in ("reserve", "release")), 6)
    if res_path.exists():
        res = json.loads(res_path.read_text())
        undoable = res.get("status") == "failed" and res.get("error") == "TimeoutError()"
    else:   # a "lost" finalize writes no result.json; it leaves an index row with status lost and a LOST_NOTE actual
        undoable = any(r["run_id"] == run_id and r.get("status") == "lost" for r in read_index())
    if not undoable or (prem <= 1e-9 and net_res <= 1e-9):
        print(f"{run_id}: not (or no longer) in the prematurely finalized state; left alone")
        return 0
    call = json.loads((d / "call.json").read_text())
    if not res_path.exists() and prem > 1e-9:
        lost = [r for r in ledger if r["kind"] == "actual" and r["note"] == LOST_NOTE]
        if lost and _ts(lost[-1]["ts_utc"]) > call["spawned_at"] + MAX_TIMEOUT + LOST_GRACE_SECONDS and _gcs_result(run_id) is None:
            print(f"{run_id}: lost after the 24 h ceiling and no result in GCS: a genuine loss; left alone")
            return 0
    env, job = call["envelope"], call["job"]
    if net_res <= 1e-9:      # else an earlier, interrupted recheck already re-reserved
        budget.append_ledger({"envelope": env, "job": job, "kind": "reserve", "amount_usd": call["max_usd"], "provider": "modal",
                              "run_id": run_id, "note": RERESERVE_NOTE})
    if prem > 1e-9:
        budget.append_ledger({"envelope": env, "job": job, "kind": "actual", "amount_usd": -prem, "provider": "modal",
                              "run_id": run_id, "note": CORRECTION_NOTE})
    rows = read_index()      # index first: every crash point is then resumable by running recheck again
    for r in rows:
        if r["run_id"] == run_id:
            r.update(status="running", finished_at="", cost_usd="")
    write_index(rows)
    for name in ("result.json", "metrics.json", "manifest.json", "events_uri.json"):
        (d / name).unlink(missing_ok=True)
    print(f"{run_id}: premature finalize undone (cost ${prem:.4f} taken back, reservation restored); collecting again")
    return collect(run_id)


def cmd_release_orphan(run_id: str) -> int:
    """Release a reservation whose spawn never happened (no call.json). Refuses anything else."""
    d = RUNS / run_id
    if d.exists():
        marker = d / "spawn_error.json"
        extra = [p.name for p in d.iterdir() if p.name != "spawn_error.json"]
        if extra or not marker.exists():
            print(f"Refused: runs/{run_id}/ exists: spawn may have succeeded; check the Modal dashboard.")
            return 2
        m = json.loads(marker.read_text())
        due = float(m["ts"]) + float(m["timeout"]) + 3600
        if time.time() <= due:
            print(f"Refused: spawn outcome unknown; wait until {due - time.time():.0f}s more, and check Modal.")
            return 2
    rows = [r for r in budget.read_ledger() if r["run_id"] == run_id]
    reserves = [r for r in rows if r["kind"] == "reserve"]
    if len(reserves) != 1:
        print(f"Refused: need exactly one reserve line for {run_id} in the ledger, found {len(reserves)}.")
        return 2
    if any(r["kind"] in ("release", "actual") for r in rows):
        print(f"Refused: {run_id} already has a release or actual line.")
        return 2
    res = reserves[0]
    budget.append_ledger({"envelope": res["envelope"], "job": res["job"], "kind": "release", "amount_usd": float(res["amount_usd"]),
                          "provider": res["provider"], "run_id": run_id,
                          "note": "orphan reservation released: spawn never happened (no call.json); nothing ran"})
    print(f"released orphan reservation {run_id} (${float(res['amount_usd']):.2f}, {res['envelope']})")
    return 0


def cmd_list() -> int:
    for r in read_index():
        if r["status"] == "running":
            print(r["run_id"], r["call_id"], r["started_at"])
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "deploy":
        return cmd_deploy()
    if argv[0] == "run":
        return cmd_run(argv[1], wait="--wait" in argv)
    if argv[0] == "collect":
        if "--recheck" in argv:
            ids = [a for a in argv[1:] if not a.startswith("--")]
            if not ids:
                print("usage: python -m opa.launch collect --recheck <run_id>")
                return 2
            return recheck(ids[0])
        if "--all" in argv:
            for r in read_index():
                if r["status"] == "running":
                    try:
                        collect(r["run_id"])
                    except (Exception, *_cancel_excs()) as e:
                        print(f"{r['run_id']}: collect error {type(e).__name__}: {e}")
            return 0
        return collect(argv[1], block="--wait" in argv)
    if argv[0] == "release-orphan":
        if len(argv) < 2:
            print("usage: python -m opa.launch release-orphan <run_id>")
            return 2
        return cmd_release_orphan(argv[1])
    if argv[0] == "list":
        return cmd_list()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
