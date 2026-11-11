"""Build a small, complete, valid proof bundle inside a throwaway git repo, for testing verify_bundle.py."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TAG_TIME = "2026-12-01T10:00:00+00:00"


def sh(cwd, *args, env=None):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, env={**os.environ, **(env or {})})


def write_csv(p: Path, rows: list[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest(b: Path):
    files = {p.relative_to(b).as_posix(): sha(p) for p in sorted(b.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (b / "manifest.json").write_text(json.dumps({"created_at": "2027-01-01T00:00:00Z", "files": files}, indent=2))


def build(root: Path) -> Path:
    """Returns the bundle path; root becomes a git repo with the prereg-v2 tag."""
    root.mkdir(parents=True, exist_ok=True)
    sh(root, "git", "init", "-q")
    sh(root, "git", "config", "user.email", "t@example.com")
    sh(root, "git", "config", "user.name", "t")
    (root / "prereg").mkdir()
    shutil.copy(REPO_ROOT / "prereg" / "rules.yaml", root / "prereg" / "rules.yaml")
    sizes = {"pilot_discordant_rates": {}, "min_n": {
        "S11_vs_S2_lbmhqa_late_32k_primary": 300, "S11_vs_S4_lbmhqa_late_32k_primary": 300, "D4_vs_D2_retention_t50": 150,
        "S11_vs_S8_late_bound": 300, "V3_vs_B1_source_attribution": 200, "A2_vs_A1_agent_mtlbmhqa_32k_primary_cross": 900,
        "A2_vs_A1_model_thread_confusion_task": 400, "R5_vs_R1_accuracy": 200, "descent_beam4_vs_flat_recall_at_M": 500,
        "pinned_vs_unpinned_repeat_pagein_reduction": 100},
        "equivalence": {"A2_vs_A3_margin_points": 3.0, "A2_vs_A3_agent_min_n": 1428}}
    import yaml
    (root / "prereg" / "sample_sizes.yaml").write_text(yaml.safe_dump(sizes, sort_keys=False))
    (root / "prereg" / "prereg-v2.md").write_text("# Pre-registration (fixture)\n")
    sh(root, "git", "add", "-A")
    sh(root, "git", "commit", "-q", "-m", "prereg", env={"GIT_COMMITTER_DATE": TAG_TIME, "GIT_AUTHOR_DATE": TAG_TIME})
    sh(root, "git", "tag", "prereg-v2")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()

    b = root / "bundle"
    (b / "prereg").mkdir(parents=True)
    for n in ("rules.yaml", "sample_sizes.yaml", "prereg-v2.md"):
        shutil.copy(root / "prereg" / n, b / "prereg" / n)
    (b / "prereg" / "PREREG_TAG.json").write_text(json.dumps({"tag": "prereg-v2", "commit": commit, "committed_at": TAG_TIME}))

    # gates: G0 passed, G1 failed, later gates never reached
    (b / "gates").mkdir()
    for g, passed in [("G0", True), ("G1", False), ("G2", None), ("G3", None), ("G4", True)]:
        (b / "gates" / f"{g}.json").write_text(json.dumps({"id": g, "passed": passed, "decided_at": "2026-12-20T00:00:00Z",
                                                          "criteria": [{"name": "x", "threshold": 1, "observed": 1, "met": bool(passed)}],
                                                          "evidence": ["runs/index.csv"], "notes": ""}))

    # tables
    rows = [
        {"key": "S11_vs_S2_lbmhqa_late_32k_primary", "value": 7.2, "ci_lo": 4.1, "ci_hi": 10.3, "n": 320, "p_raw": 0.001, "p_holm": 0.004},
        {"key": "S11_vs_S4_lbmhqa_late_32k_primary", "value": 4.0, "ci_lo": 1.2, "ci_hi": 6.9, "n": 320, "p_raw": "", "p_holm": ""},
        {"key": "D4_vs_D2_retention_t50", "value": 6.0, "ci_lo": 1.0, "ci_hi": 11.0, "n": 200, "p_raw": 0.3, "p_holm": 0.3},
        {"key": "R2_predicted_vs_observed_success", "value": 0.62, "ci_lo": "", "ci_hi": "", "n": 0, "p_raw": "", "p_holm": ""},
        {"key": "S11_vs_S8_late_bound", "value": 3.1, "ci_lo": 0.4, "ci_hi": 5.8, "n": 100, "p_raw": 0.04, "p_holm": 0.08},
        {"key": "pinned_vs_unpinned_repeat_pagein_reduction", "value": 1.8, "ci_lo": 0.9, "ci_hi": 2.7, "n": 120, "p_raw": 0.01, "p_holm": 0.03},
    ]
    write_csv(b / "tables" / "main.csv", rows)
    T = lambda k: f"tables/main.csv#row={k}"

    def res(key, value, ci, n, run, p_raw=None, p_holm=None, primary=False, level="agent", split="test"):
        return {"key": key, "level": level, "value": value, "ci95": ci, "n": n, "p_raw": p_raw, "p_holm": p_holm,
                "split": split, "primary": primary, "run_ids": [run], "table_ref": T(key)}

    runs = {"r1": "A-S11-lbmhqa-late-test-s100000-202612051200-abc1234", "r2": "A2-D4-drift-t50-test-s100001-202612061200-abc1234",
            "r3": "A-S11-mt-late-test-s100002-202612071200-abc1234", "r4": "A-pin-thrash-r4-test-s100003-202612081200-abc1234"}
    claims = {
        "H1": ("supported", [res("S11_vs_S2_lbmhqa_late_32k_primary", 7.2, [4.1, 10.3], 320, runs["r1"], 0.001, 0.004, True),
                             res("S11_vs_S4_lbmhqa_late_32k_primary", 4.0, [1.2, 6.9], 320, runs["r1"])]),
        "H2": ("refuted", [res("D4_vs_D2_retention_t50", 6.0, [1.0, 11.0], 200, runs["r2"], 0.3, 0.3, True)]),
        "H3": ("inconclusive", [res("R2_predicted_vs_observed_success", 0.62, None, 0, runs["r3"], level="measurement", split="n/a"),
                                res("S11_vs_S8_late_bound", 3.1, [0.4, 5.8], 100, runs["r3"], 0.04, 0.08, True)]),
        "H10": ("supported", [res("pinned_vs_unpinned_repeat_pagein_reduction", 1.8, [0.9, 2.7], 120, runs["r4"], 0.01, 0.03, True)]),
    }
    rules = yaml.safe_load((root / "prereg" / "rules.yaml").read_text())["hypotheses"]
    (b / "claims").mkdir()
    for h in sorted(rules, key=lambda x: int(x[1:])):
        if h in claims:
            verdict, results = claims[h]
            obj = {"id": h, "statement": rules[h]["statement"], "verdict": verdict, "results": results,
                   "figures": ["figures/h1.png"] if h == "H1" else [], "deviations": [], "notes": ""}
        else:
            obj = {"id": h, "statement": rules[h]["statement"], "verdict": "not_run_gate_failed", "failed_gate": "G1",
                   "results": [], "figures": [], "deviations": [], "notes": "Track B skipped after G1 failed"}
        (b / "claims" / f"{h}.json").write_text(json.dumps(obj, indent=2))

    # runs
    idx = []
    for i, rid in enumerate(runs.values()):
        idx.append({"run_id": rid, "track": "A", "hypotheses": "H1", "system": "S11", "dataset": "lbmhqa", "cell": "c", "split": "test",
                    "seed": 100000 + i, "started_at": f"2026-12-0{5 + i}T12:00:00Z", "finished_at": f"2026-12-0{5 + i}T13:00:00Z",
                    "status": "ok", "cost_usd": "1.0", "envelope": "E2", "gcs_uri": f"gs://b/opa/runs/{rid}", "git_sha": commit, "call_id": "fc-1"})
        d = b / "runs" / rid
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"run_id": rid, "git_sha": commit, "lock_sha256": "a" * 64,
                                                    "models": {"api_model": "claude-sonnet-5-5"}, "seeds": {"seed": 100000 + i},
                                                    "image": {"python": "3.11"}}))
        (d / "metrics.json").write_text(json.dumps({"em": 0.5}))
        (d / "events_uri.json").write_text(json.dumps({"uri": f"gs://b/opa/runs/{rid}/events.jsonl.gz", "sha256": "b" * 64}))
    write_csv(b / "runs" / "index.csv", idx)

    # figure with sidecar
    (b / "figures").mkdir()
    (b / "code").mkdir()
    (b / "code" / "plot.py").write_text("print('plot')\n")
    (b / "figures" / "h1.png").write_bytes(b"\x89PNG fake")
    (b / "figures" / "h1.json").write_text(json.dumps({"script": "code/plot.py", "inputs": [{"path": "tables/main.csv", "sha256": sha(b / "tables" / "main.csv")}],
                                                       "created_at": "2027-01-01T00:00:00Z"}))

    # budget
    (b / "budget" / "reconciliation").mkdir(parents=True)
    approvals = {"total_cap_usd": 1000, "envelopes": [
        {"id": "ORCH", "approved_usd": 300, "approved_by": "AR", "approved_on": "2026-11-10"},
        {"id": "E0", "approved_usd": 150, "approved_by": "AR", "approved_on": "2026-11-10"},
        {"id": "E2", "approved_usd": 400, "approved_by": "AR", "approved_on": "2026-12-01"}]}
    (b / "budget" / "approvals.yaml").write_text(yaml.safe_dump(approvals))
    shutil.copy(REPO_ROOT / "budget" / "prices.yaml", b / "budget" / "prices.yaml")
    ledger = [
        {"ts_utc": "2026-12-05T12:00:00Z", "envelope": "E2", "job": "j1", "kind": "reserve", "amount_usd": "50", "provider": "modal+anthropic", "run_id": runs["r1"], "note": ""},
        {"ts_utc": "2026-12-05T13:00:00Z", "envelope": "E2", "job": "j1", "kind": "actual", "amount_usd": "12.5", "provider": "modal", "run_id": runs["r1"], "note": ""},
        {"ts_utc": "2026-12-05T13:00:00Z", "envelope": "E2", "job": "j1", "kind": "actual", "amount_usd": "30", "provider": "anthropic", "run_id": runs["r1"], "note": ""},
        {"ts_utc": "2026-12-05T13:00:00Z", "envelope": "E2", "job": "j1", "kind": "release", "amount_usd": "50", "provider": "modal+anthropic", "run_id": runs["r1"], "note": ""},
        {"ts_utc": "2026-12-06T00:00:00Z", "envelope": "RECON", "job": "reconcile", "kind": "reconcile", "amount_usd": "12.8", "provider": "modal", "run_id": "", "note": ""},
    ]
    write_csv(b / "budget" / "ledger.csv", ledger)
    (b / "budget" / "reconciliation" / "modal_20261130_20261207.json").write_text(json.dumps(
        {"provider": "modal", "period_start": "2026-11-30T00:00:00+00:00", "period_end": "2026-12-07T00:00:00+00:00",
         "provider_total_usd": 12.8, "ledger_total_usd": 12.5, "explanation": "", "rows": []}))

    (b / "deviations.md").write_text("None.\n")
    (b / "PAPER_INPUT.md").write_text(
        "# PAPER_INPUT\n\n## 0. Run summary\n\n- Spend: $42.50 of $1,000 (not checked here)\n\n## Results\n\n"
        "### H1 — fold-left beats retrieval — VERDICT: supported\n\n"
        "- S11 vs S2: 7.2 points (95% CI 4.1 to 10.3) [[ref:tables/main.csv#row=S11_vs_S2_lbmhqa_late_32k_primary&cols=value,ci_lo,ci_hi]], "
        "Holm p = 0.004 [[ref:claims/H1.json#results.0.p_holm]] on `Qwen2.5-1.5B` and 1.5B models.\n\n"
        "### H2 — drift — VERDICT: refuted\n\n- Gain 6.0 points [[ref:tables/main.csv#row=D4_vs_D2_retention_t50&cols=value]], short of the 10-point threshold.\n\n"
        "### H3 — coverage — VERDICT: inconclusive\n\n- R² 0.62 [[ref:claims/H3.json#results.0.value]]; n = 100 below the pre-registered 300.\n\n"
        + "".join(f"### H{i} — not run — VERDICT: not_run_gate_failed\n\n- G1 failed.\n\n" for i in range(4, 10))
        + "### H10 — thrashing — VERDICT: supported\n\n- Reduction 1.8 [[ref:tables/main.csv#row=pinned_vs_unpinned_repeat_pagein_reduction&cols=value]] page-ins per item, 62% of baseline `0.0` [[ref:claims/H3.json#results.0.value]].\n\n"
        + "".join(f"### H{i} — not run — VERDICT: not_run_gate_failed\n\n- G1 failed.\n\n" for i in range(11, 14))
        + "## Measured numbers for the paper\n\n| Location | Old | New |\n|---|---|---|\n| §7.2 | `0.913` | 0.62 [[ref:claims/H3.json#results.0.value]] |\n\n"
        "## Surprises\n\nNone, 3.14 is not checked here.\n")
    manifest(b)
    return b


if __name__ == "__main__":
    import sys
    print(build(Path(sys.argv[1])))
