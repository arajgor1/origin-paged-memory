"""Assemble, hash and upload the proof bundle.

  python -m opa.bundle assemble        copy prereg, budget, cited runs and env files into bundle/ (after claims exist)
  python -m opa.bundle code            copy the analysis scripts into bundle/code/ (before figures are saved)
  python -m opa.bundle manifest        write bundle/manifest.json (SHA-256 of every file); run this last
  python -m opa.bundle check           run the verifier in-process without the git checks and list failures
  python -m opa.bundle prereg-tag      write prereg/PREREG_TAG.json from the prereg-v2 git tag (after `git tag`)
  python -m opa.bundle upload --snapshot | --final

Every function takes optional ``project`` (repo root) and ``bundle`` directories so tests can run on a copy.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import PROJECT_DIR

B = PROJECT_DIR / "bundle"
RUN_INDEX_FIELDS = ["run_id", "track", "hypotheses", "system", "dataset", "cell", "split", "seed", "started_at",
                    "finished_at", "status", "cost_usd", "envelope", "gcs_uri", "git_sha", "call_id"]
RUN_FILES = ("manifest.json", "metrics.json", "events_uri.json", "events.jsonl.gz", "result.json", "job.yaml")
PREREG_FILES = ("prereg-v2.md", "rules.yaml", "sample_sizes.yaml", "PREREG_TAG.json")
CODE_FILES = ("opa/stats.py", "opa/claims.py", "opa/scoring.py", "opa/analysis/__init__.py",
              "opa/analysis/tables.py", "opa/analysis/figures.py", "opa/analysis/distortion.py")


def _dirs(project: Path | str | None, bundle: Path | str | None) -> tuple[Path, Path]:
    p = Path(project) if project is not None else PROJECT_DIR
    return p, (Path(bundle) if bundle is not None else (B if project is None else p / "bundle"))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cited_runs(bundle: Path | str | None = None) -> set[str]:
    """Run ids cited by bundle/claims/H*.json results and by bundle/tests/U*.json records."""
    b = Path(bundle) if bundle is not None else B
    ids = set()
    for f in sorted((b / "claims").glob("H*.json")):
        c = json.loads(f.read_text(encoding="utf-8"))
        for r in c.get("results", []):
            ids.update(r.get("run_ids", []))
    for f in sorted((b / "tests").glob("U*.json")):
        rid = json.loads(f.read_text(encoding="utf-8")).get("run_id")
        if rid:
            ids.add(rid)
    return ids


def copy_code(files=CODE_FILES, project: Path | str | None = None, bundle: Path | str | None = None) -> list[str]:
    """Copy analysis/plotting scripts into bundle/code/<same relative path> (figure sidecars name them, and the
    verifier requires them inside the bundle). Call before saving figures. Returns the bundle-relative paths."""
    project, bundle = _dirs(project, bundle)
    out = []
    for rel in files:
        src = project / rel
        if not src.is_file():
            raise FileNotFoundError(f"{rel} not found under {project}")
        dst = bundle / "code" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        out.append(f"code/{Path(rel).as_posix()}")
    return out


def assemble(project: Path | str | None = None, bundle: Path | str | None = None, strict: bool = False) -> list[str]:
    """Copy prereg, budget (approvals, prices, ledger, reconciliations), env lock files and every run cited by
    bundle/claims/*.json and bundle/tests/U*.json (index rows + run folders) into the bundle. Claims must be written first.

    Returns the list of problems found (missing prereg files, cited runs absent from runs/index.csv, without a
    folder, without status ok, or missing manifest/metrics/events); with strict=True a non-empty list raises."""
    project, bundle = _dirs(project, bundle)
    problems: list[str] = []
    for sub in ("prereg", "budget", "budget/reconciliation", "runs", "env"):
        (bundle / sub).mkdir(parents=True, exist_ok=True)
    for name in PREREG_FILES:
        src = project / "prereg" / name
        if src.exists():
            shutil.copy2(src, bundle / "prereg" / name)
        else:
            problems.append(f"prereg/{name} missing")
    for name in ("approvals.yaml", "prices.yaml", "ledger.csv"):
        src = project / "budget" / name
        if src.exists():
            shutil.copy2(src, bundle / "budget" / name)
        else:
            problems.append(f"budget/{name} missing")
    recon = sorted((project / "budget" / "reconciliation").glob("*.json"))
    for f in recon:
        shutil.copy2(f, bundle / "budget" / "reconciliation" / f.name)
    for name in ("requirements.txt", "requirements.lock", "uv.lock"):
        if (project / name).exists():
            shutil.copy2(project / name, bundle / "env" / name)

    ids = cited_runs(bundle)
    src_index = project / "runs" / "index.csv"
    rows, fields = [], list(RUN_INDEX_FIELDS)
    if src_index.exists():
        with src_index.open(newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            rows = list(rd)
            fields = list(rd.fieldnames or RUN_INDEX_FIELDS)
    elif ids:
        problems.append("runs/index.csv missing in the project")
    by_id = {r["run_id"]: r for r in rows}
    keep = [r for r in rows if r["run_id"] in ids]
    with (bundle / "runs" / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)
    for rid in sorted(ids):
        row = by_id.get(rid)
        if row is None:
            problems.append(f"cited run {rid} is not in runs/index.csv")
        elif row.get("status") != "ok":
            problems.append(f"cited run {rid} has status {row.get('status')!r}")
        src = project / "runs" / rid
        if not src.is_dir():
            problems.append(f"cited run {rid} has no folder runs/{rid}")
            continue
        dst = bundle / "runs" / rid
        dst.mkdir(parents=True, exist_ok=True)
        for name in RUN_FILES:
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        for name in ("manifest.json", "metrics.json"):
            if not (src / name).exists():
                problems.append(f"runs/{rid}/{name} missing")
        if not (src / "events_uri.json").exists() and not (src / "events.jsonl.gz").exists():
            problems.append(f"runs/{rid}: neither events_uri.json nor events.jsonl.gz")
    stale = sorted(d.name for d in (bundle / "runs").iterdir() if d.is_dir() and d.name not in ids)
    if stale:
        print(f"note: {len(stale)} run folder(s) in the bundle are no longer cited (harmless, still hashed): {stale[:5]}")
    print(f"assembled bundle with {len(keep)} cited runs and {len(recon)} reconciliation file(s)")
    for p in problems:
        print("problem:", p)
    if strict and problems:
        raise SystemExit("assemble found problems:\n  " + "\n  ".join(problems))
    return problems


def manifest(bundle: Path | str | None = None) -> Path:
    """bundle/manifest.json: {"created_at", "files": {rel: sha256}}. Like the verifier, "files" leaves out every file
    named manifest.json (so per-run manifests are not in it); their hashes are recorded separately under
    "nested_manifests" (ignored by the verifier, kept for integrity)."""
    b = Path(bundle) if bundle is not None else B
    files, nested = {}, {}
    for p in sorted(b.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(b).as_posix()
        if p.name != "manifest.json":
            files[rel] = sha(p)
        elif rel != "manifest.json":
            nested[rel] = sha(p)
    out = b / "manifest.json"
    out.write_text(json.dumps({"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "files": files,
                               "nested_manifests": nested}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"manifest: {len(files)} files (+{len(nested)} run manifests)")
    return out


def write_prereg_tag(project: Path | str | None = None, tag: str = "prereg-v2") -> Path:
    """prereg/PREREG_TAG.json {"tag", "commit", "committed_at"} from the git tag (commit time = %cI, which the
    verifier compares to within 1 s). Use this instead of the Write tool: after the tag, protect_files blocks
    edits under prereg/."""
    project, _ = _dirs(project, None)

    def git(*a):
        r = subprocess.run(["git", *a], cwd=project, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"git {' '.join(a)} failed: {r.stderr.strip()}")
        return r.stdout.strip()

    commit = git("rev-list", "-n", "1", tag)
    rec = {"tag": tag, "commit": commit, "committed_at": git("show", "-s", "--format=%cI", commit)}
    p = project / "prereg" / "PREREG_TAG.json"
    p.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rec))
    return p


def check(bundle: Path | str | None = None) -> int:
    from .claims import preflight
    fails = preflight(bundle if bundle is not None else B)
    for f in fails:
        print(" ", f)
    print("preflight (no git checks):", "OK" if not fails else f"{len(fails)} failure(s)")
    return 0 if not fails else 1


def upload(kind: str, project: Path | str | None = None, bundle: Path | str | None = None) -> str:
    from . import gcs
    project, b = _dirs(project, bundle)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = gcs.uri(f"opa/bundle/{'final-' + ts if kind == 'final' else 'snapshots/' + ts}")
    if kind == "final" and not (b / "manifest.json").exists():
        raise SystemExit("bundle/manifest.json missing: run `python -m opa.bundle manifest` first")
    n = 0
    for p in sorted(b.rglob("*")):
        if p.is_file():
            gcs.put_file(p, f"{prefix}/{p.relative_to(b).as_posix()}")
            n += 1
    if kind == "final":
        local = sha(b / "manifest.json")
        remote = hashlib.sha256(gcs.get_bytes(f"{prefix}/manifest.json")).hexdigest()
        rec = {"gcs_prefix": prefix, "files": n, "manifest_sha256_local": local, "manifest_sha256_remote": remote,
               "match": local == remote, "uploaded_at": ts}
        (project / "state" / "FINAL_UPLOAD.json").write_text(json.dumps(rec, indent=2))
        print(json.dumps(rec, indent=2))
        if local != remote:
            raise SystemExit("manifest hash mismatch after upload")
    print(f"uploaded {n} files to {prefix}")
    return prefix


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "assemble":
        return 1 if assemble() else 0
    if argv[0] == "code":
        print("\n".join(copy_code()))
    elif argv[0] == "manifest":
        manifest()
    elif argv[0] == "check":
        return check()
    elif argv[0] == "prereg-tag":
        write_prereg_tag()
    elif argv[0] == "upload":
        upload("final" if "--final" in argv else "snapshot")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
