#!/usr/bin/env python3
"""verify_bundle.py — the machine-checkable definition of done (GOAL.md, requirements D1-D13).

Usage:  python tools/verify_bundle.py bundle/ [--repo .] [--no-git]
Exit 0 and print "BUNDLE VERIFIED" only if every check passes. Otherwise print every failure and exit 1.
This file is owned by the human; the agent cannot edit it.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

try:
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None

DEFAULT_H_IDS = [f"H{i}" for i in range(1, 14)]   # v3 (PLAN_V3.md): H1-H13; the bundle's rules.yaml is authoritative
G_IDS = [f"G{i}" for i in range(5)]
U_IDS = [f"U{i}" for i in range(1, 9)]
GATE_ORDER = {g: i for i, g in enumerate(G_IDS)}
TRACK_B_HYPOTHESES = {"H4", "H6", "H7", "H9", "H11", "H12", "H13"}
REF_SECTIONS = ("## Results", "## Measured numbers for the paper")


def hypothesis_ids(rules: dict | None) -> list[str]:
    """Hypothesis ids from the frozen rules file (sorted H1, H2, ... H13); defaults if the file is missing."""
    ids = list(((rules or {}).get("hypotheses") or {}).keys())
    if not ids:
        return list(DEFAULT_H_IDS)
    return sorted(ids, key=lambda h: (int(h[1:]) if h[1:].isdigit() else 10**6, h))


class Report:
    def __init__(self):
        self.fails: list[str] = []
        self.notes: list[str] = []

    def fail(self, code: str, msg: str):
        self.fails.append(f"[{code}] {msg}")

    def note(self, msg: str):
        self.notes.append(msg)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_time(s: str) -> dt.datetime:
    s = s.strip().replace("Z", "+00:00")
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def load_json(p: Path, rep: Report, code: str):
    try:
        return json.loads(p.read_text())
    except Exception as e:
        rep.fail(code, f"{p}: unreadable JSON ({e})")
        return None


# ------------------------------------------------------------------ rule evaluation (D3)
def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def eval_rule(rule: dict, results: dict, min_n: dict, rep: Report, hid: str) -> str:
    if "all_of" in rule:
        vs = [eval_rule(r, results, min_n, rep, hid) for r in rule["all_of"]]
        if "refuted" in vs:
            return "refuted"
        return "inconclusive" if "inconclusive" in vs else "supported"
    if "any_of" in rule:
        vs = [eval_rule(r, results, min_n, rep, hid) for r in rule["any_of"]]
        if "supported" in vs:
            return "supported"
        return "inconclusive" if "inconclusive" in vs else "refuted"
    key = rule["key"]
    res = results.get(key)
    if res is None:
        return "inconclusive"
    if not rule.get("measurement"):
        need = (min_n or {}).get(key)
        if need is None:
            rep.fail("D3", f"{hid}: rule key {key} has no pre-registered minimum n in prereg/sample_sizes.yaml")
            return "inconclusive"
        if int(res.get("n", 0)) < int(need):
            return "inconclusive"
        if res.get("split") != "test":
            rep.fail("D3", f"{hid}: result {key} feeds a verdict but split is {res.get('split')!r}, not 'test'")
            return "inconclusive"
    v = float(res["value"])
    t = rule["type"]
    if t in ("min_delta", "min_value"):
        ok = v > rule["threshold"] if rule.get("strict") else v >= rule["threshold"]
    elif t in ("max_value", "max_delta"):
        ok = v < rule["threshold"] if rule.get("strict") else v <= rule["threshold"]
    elif t == "within":
        ok = rule["lo"] <= v <= rule["hi"]
    else:
        rep.fail("D3", f"{hid}: unknown rule type {t}")
        return "inconclusive"
    ci = res.get("ci95")
    if rule.get("require_ci_excludes_zero"):
        ok = ok and ci is not None and ci[0] > 0
    if "require_ci_lower_above" in rule:
        ok = ok and ci is not None and ci[0] > rule["require_ci_lower_above"]
    if "require_p_holm_below" in rule:
        ph = res.get("p_holm")
        ok = ok and ph is not None and ph < rule["require_p_holm_below"]
    return "supported" if ok else "refuted"


# ------------------------------------------------------------------ PAPER_INPUT references (D7)
NUM_RE = re.compile(r"(?<![\w.$§#])(\$?)(-|−)?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%?)(?![\w.])")   # skips 1.5B, v2.1.3, H1
REF_RE = re.compile(r"\[\[ref:([^\]]+)\]\]")


def resolve_ref(target: str, bundle: Path) -> list[float]:
    path, _, frag = target.partition("#")
    p = bundle / path
    if not p.exists():
        raise ValueError(f"reference target {path} does not exist")
    if path.endswith(".csv"):
        params = dict(kv.split("=", 1) for kv in frag.split("&") if "=" in kv)
        row_key, cols = params.get("row"), params.get("cols") or params.get("col")
        if not row_key or not cols:
            raise ValueError(f"CSV reference needs row= and cols=: {target}")
        rows = read_csv(p)
        keycol = "key" if rows and "key" in rows[0] else (list(rows[0].keys())[0] if rows else None)
        match = [r for r in rows if r.get(keycol) == row_key]
        if not match:
            raise ValueError(f"row {row_key} not found in {path}")
        out = []
        for c in cols.split(","):
            if c not in match[0]:
                raise ValueError(f"column {c} not found in {path}")
            out.append(float(match[0][c]))
        return out
    if path.endswith(".json"):
        obj = json.loads(p.read_text())
        out = []
        for dotted in frag.split(","):
            cur = obj
            for part in dotted.split("."):
                cur = cur[int(part)] if isinstance(cur, list) else cur[part]
            out.append(float(cur))
        return out
    raise ValueError(f"unsupported reference type: {path}")


def number_matches(disp: str, decimals: int, percent: bool, src: float) -> bool:
    shown = float(disp)
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    cands = [src, src * 100] if percent else [src]
    return any(abs(c - shown) <= tol for c in cands)


def check_refs(text: str, bundle: Path, rep: Report) -> None:
    lines = text.splitlines()
    in_sec = False
    for ln, line in enumerate(lines, 1):
        if line.startswith("## "):
            in_sec = any(line.startswith(s) for s in REF_SECTIONS)
            continue
        if not in_sec:
            continue
        clean = re.sub(r"`[^`]*`", lambda m: " " * len(m.group(0)), line)
        refs = [(m.start(), m.end(), m.group(1)) for m in REF_RE.finditer(clean)]
        masked = REF_RE.sub(lambda m: " " * len(m.group(0)), clean)
        nums = []
        for m in NUM_RE.finditer(masked):
            dollar, sign, intpart, frac, pct = m.groups()
            if not (frac or pct or dollar):
                continue
            if pct and masked[m.end():m.end() + 3].strip().startswith("CI"):
                continue                                            # "95% CI" is not a measurement
            value = (("-" if sign else "") + intpart.replace(",", "") + (frac or ""))
            nums.append({"pos": m.start(), "text": m.group(0), "value": value, "dec": len(frac) - 1 if frac else 0,
                         "pct": bool(pct), "bound": False})
        for start, _, target in refs:
            try:
                vals = resolve_ref(target, bundle)
            except Exception as e:
                rep.fail("D7", f"PAPER_INPUT.md line {ln}: {e}")
                continue
            before = [n for n in nums if n["pos"] < start and not n["bound"]]
            if len(before) < len(vals):
                rep.fail("D7", f"PAPER_INPUT.md line {ln}: reference {target} names {len(vals)} value(s) but only {len(before)} unreferenced number(s) precede it")
                for n in before:
                    n["bound"] = True
                continue
            for n, src in zip(before[-len(vals):], vals):
                n["bound"] = True
                if not number_matches(n["value"], n["dec"], n["pct"], src):
                    rep.fail("D7", f"PAPER_INPUT.md line {ln}: '{n['text']}' does not match {target} (source value {src})")
        for n in nums:
            if not n["bound"]:
                rep.fail("D7", f"PAPER_INPUT.md line {ln}: number '{n['text']}' has no [[ref:...]]")


def results_section(text: str) -> str:
    m = re.search(r"^## Results\s*$(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1) if m else ""


# ------------------------------------------------------------------ main checks
def verify(bundle: Path, repo: Path | None) -> Report:
    rep = Report()
    if not bundle.is_dir():
        rep.fail("D0", f"{bundle} is not a directory")
        return rep
    spec_dir = Path(__file__).resolve().parent.parent / "bundle_spec"
    schemas = {}
    for n in ("claim", "gate", "test"):
        p = spec_dir / f"{n}.schema.json"
        schemas[n] = json.loads(p.read_text()) if p.exists() else None

    def validate(obj, kind, where):
        if jsonschema is None or schemas.get(kind) is None:
            rep.fail("D0", "jsonschema package or bundle_spec schemas missing; cannot validate")
            return
        errs = sorted(jsonschema.Draft202012Validator(schemas[kind]).iter_errors(obj), key=lambda e: list(e.path))
        for e in errs[:5]:
            rep.fail("D2" if kind == "claim" else ("D4" if kind == "gate" else "D5"), f"{where}: {'/'.join(map(str, e.path)) or '(root)'}: {e.message}")

    # ---- D12 integrity first: everything else reads files the manifest vouches for
    man_p = bundle / "manifest.json"
    if not man_p.exists():
        rep.fail("D12", "manifest.json missing")
    else:
        man = load_json(man_p, rep, "D12") or {}
        listed = man.get("files", {})
        actual = {p.relative_to(bundle).as_posix(): p for p in bundle.rglob("*") if p.is_file() and p.name != "manifest.json"}
        for rel in sorted(set(actual) - set(listed)):
            rep.fail("D12", f"{rel} is in the bundle but not in manifest.json")
        for rel in sorted(set(listed) - set(actual)):
            rep.fail("D12", f"{rel} is listed in manifest.json but missing")
        for rel in sorted(set(listed) & set(actual)):
            if sha256(actual[rel]) != listed[rel]:
                rep.fail("D12", f"{rel}: SHA-256 differs from manifest.json")

    # ---- pre-registration inputs
    rules_p, sizes_p, tag_p = bundle / "prereg/rules.yaml", bundle / "prereg/sample_sizes.yaml", bundle / "prereg/PREREG_TAG.json"
    rules = yaml.safe_load(rules_p.read_text()) if rules_p.exists() else None
    sizes = yaml.safe_load(sizes_p.read_text()) if sizes_p.exists() else None
    tag = load_json(tag_p, rep, "D1") if tag_p.exists() else None
    if rules is None:
        rep.fail("D1", "prereg/rules.yaml missing from the bundle")
    if sizes is None:
        rep.fail("D1", "prereg/sample_sizes.yaml missing from the bundle")
    if tag is None:
        rep.fail("D1", "prereg/PREREG_TAG.json missing")
    if not (bundle / "prereg/prereg-v2.md").exists():
        rep.fail("D1", "prereg/prereg-v2.md missing")
    tag_time = None
    if tag:
        try:
            tag_time = parse_time(tag["committed_at"])
        except Exception:
            rep.fail("D1", "PREREG_TAG.json committed_at is not an ISO timestamp")
        if repo is not None:
            def git(*a):
                return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)
            r = git("rev-list", "-n", "1", tag.get("tag", "prereg-v2"))
            if r.returncode != 0 or not r.stdout.strip():
                rep.fail("D1", f"git tag {tag.get('tag')} not found in {repo}")
            else:
                if r.stdout.strip() != tag.get("commit"):
                    rep.fail("D1", "git tag points to a different commit than PREREG_TAG.json says")
                ct = git("show", "-s", "--format=%cI", r.stdout.strip()).stdout.strip()
                if ct and tag_time and abs((parse_time(ct) - tag_time).total_seconds()) > 1:
                    rep.fail("D1", f"PREREG_TAG.json committed_at {tag['committed_at']} differs from the commit time {ct}")
                for name in ("rules.yaml", "sample_sizes.yaml"):
                    shown = git("show", f"{r.stdout.strip()}:prereg/{name}")
                    if shown.returncode != 0:
                        rep.fail("D1", f"prereg/{name} is not in the tagged commit")
                    elif (bundle / "prereg" / name).exists() and shown.stdout != (bundle / "prereg" / name).read_text():
                        rep.fail("D1", f"bundle prereg/{name} differs from the version in the tagged commit")
    min_n = (sizes or {}).get("min_n", {}) if sizes else {}

    # ---- D4 gates
    gates = {}
    for g in G_IDS:
        p = bundle / "gates" / f"{g}.json"
        if not p.exists():
            rep.fail("D4", f"gates/{g}.json missing")
            continue
        obj = load_json(p, rep, "D4")
        if obj is not None:
            validate(obj, "gate", f"gates/{g}.json")
            gates[g] = obj
    if gates.get("G0", {}).get("passed") is not True:
        rep.fail("D4", "G0 (harness valid) did not pass")

    # ---- runs index (D6, D1 timing)
    idx_p = bundle / "runs" / "index.csv"
    runs = {r["run_id"]: r for r in read_csv(idx_p)} if idx_p.exists() else {}
    if not idx_p.exists():
        rep.fail("D6", "runs/index.csv missing")
    if tag_time:
        for rid, r in runs.items():
            if r.get("split") == "test":
                try:
                    if parse_time(r["started_at"]) <= tag_time:
                        rep.fail("D1", f"test-split run {rid} started before the prereg-v2 commit")
                except Exception:
                    rep.fail("D6", f"run {rid}: started_at unreadable")

    # ---- D2/D3 claims
    claims = {}
    for h in hypothesis_ids(rules):
        p = bundle / "claims" / f"{h}.json"
        if not p.exists():
            rep.fail("D2", f"claims/{h}.json missing")
            continue
        obj = load_json(p, rep, "D2")
        if obj is None:
            continue
        validate(obj, "claim", f"claims/{h}.json")
        if obj.get("id") != h:
            rep.fail("D2", f"claims/{h}.json has id {obj.get('id')!r}")
        claims[h] = obj

    primary_p = {}
    for h, c in claims.items():
        for r in c.get("results", []):
            if r.get("primary") and r.get("p_raw") is not None:
                primary_p[f"{h}:{r['key']}"] = float(r["p_raw"])
        if sum(1 for r in c.get("results", []) if r.get("primary")) > 1:
            rep.fail("D3", f"{h}: more than one result marked primary")
    holm_adj = holm(primary_p)
    for h, c in claims.items():
        for r in c.get("results", []):
            k = f"{h}:{r['key']}"
            if k in holm_adj:
                if r.get("p_holm") is None or abs(float(r["p_holm"]) - holm_adj[k]) > 1e-6:
                    rep.fail("D3", f"{h}: p_holm for {r['key']} is {r.get('p_holm')}, recomputed {holm_adj[k]:.6g}")

    hyp_rules = (rules or {}).get("hypotheses", {})
    for h, c in claims.items():
        verdict = c.get("verdict")
        hr = hyp_rules.get(h)
        if hr is None:
            rep.fail("D3", f"{h}: no rule in prereg/rules.yaml")
            continue
        if verdict == "not_run_gate_failed":
            fg = c.get("failed_gate")
            dep = hr.get("gate_dependency", "G0")
            if not fg or fg not in gates:
                rep.fail("D4", f"{h}: not_run_gate_failed must name a recorded gate (failed_gate)")
            elif gates[fg].get("passed") is not False:
                rep.fail("D4", f"{h}: claims gate {fg} failed, but gates/{fg}.json does not show passed: false")
            elif GATE_ORDER[fg] > GATE_ORDER[dep]:
                rep.fail("D4", f"{h}: depends on {dep}, but cites the later gate {fg}")
            continue
        by_key = {}
        for r in c.get("results", []):
            if r["key"] in by_key:
                rep.fail("D3", f"{h}: duplicate result key {r['key']}")
            by_key[r["key"]] = r
        computed = eval_rule(hr["rule"], by_key, min_n, rep, h)
        if computed != verdict:
            rep.fail("D3", f"{h}: declared verdict '{verdict}' but the frozen rule gives '{computed}'")

    # ---- D5 unit tests
    needs_b = any(h in TRACK_B_HYPOTHESES and c.get("verdict") != "not_run_gate_failed" and c.get("results") for h, c in claims.items()) or \
        any(r.get("level") == "model" for c in claims.values() for r in c.get("results", []))
    if needs_b:
        for u in U_IDS:
            p = bundle / "tests" / f"{u}.json"
            if not p.exists():
                rep.fail("D5", f"tests/{u}.json missing (model-level results are reported)")
                continue
            obj = load_json(p, rep, "D5")
            if obj is not None:
                validate(obj, "test", f"tests/{u}.json")
                if obj.get("passed") is not True:
                    rep.fail("D5", f"unit test {u} did not pass")

    # ---- D6 runs and D13 reproducibility metadata
    cited = {}
    for h, c in claims.items():
        for r in c.get("results", []):
            for rid in r.get("run_ids", []):
                cited.setdefault(rid, []).append((h, r))
            ref = r.get("table_ref", "")
            path, _, frag = ref.partition("#")
            if not (bundle / path).exists():
                rep.fail("D6", f"{h}: table_ref {path} does not exist")
            else:
                row = dict(kv.split("=", 1) for kv in frag.split("&") if "=" in kv).get("row")
                rows = read_csv(bundle / path)
                keycol = "key" if rows and "key" in rows[0] else (list(rows[0].keys())[0] if rows else None)
                hit = [x for x in rows if x.get(keycol) == row]
                if not hit:
                    rep.fail("D6", f"{h}: row {row} not found in {path}")
                elif "value" in hit[0] and abs(float(hit[0]["value"]) - float(r["value"])) > 1e-9 * max(1, abs(float(r["value"]))):
                    rep.fail("D6", f"{h}: {r['key']} value {r['value']} differs from {path} row value {hit[0]['value']}")
    for rid, uses in cited.items():
        row = runs.get(rid)
        if row is None:
            rep.fail("D6", f"run {rid} cited by {uses[0][0]} is not in runs/index.csv")
            continue
        if row.get("status") != "ok":
            rep.fail("D6", f"run {rid} has status {row.get('status')!r}")
        for h, r in uses:
            if r.get("split") == "test" and row.get("split") != "test":
                rep.fail("D6", f"{h}: result {r['key']} is marked test but run {rid} is split {row.get('split')!r}")
        d = bundle / "runs" / rid
        for name in ("manifest.json", "metrics.json"):
            if not (d / name).exists():
                rep.fail("D6", f"runs/{rid}/{name} missing")
        ev = d / "events_uri.json"
        if not ev.exists() and not (d / "events.jsonl.gz").exists():
            rep.fail("D6", f"runs/{rid}: no events.jsonl.gz or events_uri.json")
        elif ev.exists():
            e = load_json(ev, rep, "D6") or {}
            if not str(e.get("uri", "")).startswith("gs://") or not re.fullmatch(r"[0-9a-f]{64}", str(e.get("sha256", ""))):
                rep.fail("D6", f"runs/{rid}/events_uri.json needs a gs:// uri and a sha256")
        mp = d / "manifest.json"
        if mp.exists():
            m = load_json(mp, rep, "D13") or {}
            for k in ("git_sha", "lock_sha256", "models", "seeds", "image"):
                if not m.get(k):
                    rep.fail("D13", f"runs/{rid}/manifest.json lacks {k}")

    # ---- D7 traceability and D10 negative results
    pi = bundle / "PAPER_INPUT.md"
    if not pi.exists():
        rep.fail("D7", "PAPER_INPUT.md missing")
    else:
        text = pi.read_text()
        check_refs(text, bundle, rep)
        res = results_section(text)
        if not res:
            rep.fail("D10", "PAPER_INPUT.md has no '## Results' section")
        for h, c in claims.items():
            pat = re.compile(rf"^###\s+{h}\b.*VERDICT:\s*{re.escape(c.get('verdict', ''))}\b", re.M | re.I)
            if not pat.search(res):
                rep.fail("D10", f"Results section lacks a '### {h} ... VERDICT: {c.get('verdict')}' heading")

    # ---- D8 figures
    fig_dir = bundle / "figures"
    for png in sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []:
        side = png.with_suffix(".json")
        if not side.exists():
            rep.fail("D8", f"figures/{png.name} has no sidecar .json")
            continue
        s = load_json(side, rep, "D8") or {}
        script = s.get("script")
        if not script or not (bundle / script).exists():
            rep.fail("D8", f"figures/{side.name}: script {script!r} not in the bundle")
        if not s.get("inputs"):
            rep.fail("D8", f"figures/{side.name}: no inputs listed")
        for inp in s.get("inputs", []):
            ip = bundle / inp.get("path", "")
            if not ip.is_file():
                rep.fail("D8", f"figures/{side.name}: input {inp.get('path')} missing")
            elif sha256(ip) != inp.get("sha256"):
                rep.fail("D8", f"figures/{side.name}: input {inp.get('path')} hash differs")
    for c in claims.values():
        for f in c.get("figures", []):
            if not (bundle / f).exists():
                rep.fail("D8", f"{c['id']}: figure {f} missing")

    # ---- D9 budget
    ap, lp = bundle / "budget/approvals.yaml", bundle / "budget/ledger.csv"
    if not (bundle / "budget").exists():
        pass   # public release: the budget records are not published, so D9 is skipped (a partial budget/ still fails)
    elif not ap.exists() or not lp.exists():
        rep.fail("D9", "budget/approvals.yaml or budget/ledger.csv missing")
    else:
        appr = yaml.safe_load(ap.read_text()) or {}
        ok_env = {}
        for e in appr.get("envelopes", []):
            if float(e.get("approved_usd") or 0) > 0 and str(e.get("approved_by") or "").strip() and str(e.get("approved_on") or "").strip():
                ok_env[str(e["id"])] = float(e["approved_usd"])
        cap = float(appr.get("total_cap_usd") or 0)
        spent, modal_weeks = {}, set()
        for r in read_csv(lp):
            if r["kind"] == "reconcile":
                continue
            if r["envelope"] not in ok_env:
                rep.fail("D9", f"ledger line for job {r['job']!r} uses unapproved envelope {r['envelope']!r}")
            if r["kind"] == "actual":
                spent[r["envelope"]] = spent.get(r["envelope"], 0.0) + float(r["amount_usd"])
                if r["provider"] == "modal" and float(r["amount_usd"]) > 0:
                    modal_weeks.add(parse_time(r["ts_utc"]).isocalendar()[:2])
        for env, s in spent.items():
            if env in ok_env and s > 1.05 * ok_env[env] + 1e-6:
                rep.fail("D9", f"envelope {env}: spent ${s:.2f} > 1.05 x approved ${ok_env[env]:.2f}")
        total = sum(spent.values())
        if cap <= 0 or total > cap + 1e-6:
            rep.fail("D9", f"total spend ${total:.2f} exceeds total_cap_usd ${cap:.2f}")
        covered = set()
        for f in sorted((bundle / "budget/reconciliation").glob("*.json")) if (bundle / "budget/reconciliation").exists() else []:
            rec = load_json(f, rep, "D9") or {}
            try:
                a, b = parse_time(rec["period_start"]), parse_time(rec["period_end"])
            except Exception:
                rep.fail("D9", f"{f.name}: period_start/period_end missing")
                continue
            d = a
            while d <= b:
                covered.add(d.isocalendar()[:2])
                d += dt.timedelta(days=1)
            pt, lt = float(rec.get("provider_total_usd", 0)), float(rec.get("ledger_total_usd", 0))
            if max(pt, lt) > 0 and abs(pt - lt) > 0.10 * max(pt, lt) and not str(rec.get("explanation", "")).strip():
                rep.fail("D9", f"{f.name}: provider ${pt:.2f} vs ledger ${lt:.2f} differ by >10% with no explanation")
        for wk in sorted(modal_weeks - covered):
            rep.fail("D9", f"no Modal reconciliation covers ISO week {wk[0]}-W{wk[1]:02d}")
        rep.note(f"total actual spend ${total:.2f} of cap ${cap:.2f}")

    # ---- D11 deviations
    dv = bundle / "deviations.md"
    if not dv.exists():
        rep.fail("D11", "deviations.md missing")
    else:
        t = dv.read_text().strip()
        if t != "None.":
            entries = re.split(r"^###\s+", t, flags=re.M)[1:]
            if not entries:
                rep.fail("D11", "deviations.md must be 'None.' or contain '### DEV-n' entries")
            for e in entries:
                name = e.splitlines()[0].strip()
                for field in ("Date:", "Change:", "Reason:", "Impact:"):
                    if field not in e:
                        rep.fail("D11", f"deviation {name} lacks '{field}'")
        for c in claims.values():
            for dref in c.get("deviations", []):
                if dref not in t:
                    rep.fail("D11", f"{c['id']} cites deviation {dref} not found in deviations.md")
    return rep


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    bundle = Path(argv[0]).resolve()
    repo = None
    if "--no-git" not in argv:
        repo = Path(argv[argv.index("--repo") + 1]).resolve() if "--repo" in argv else bundle.parent
    rep = verify(bundle, repo)
    for n in rep.notes:
        print("note:", n)
    if rep.fails:
        print(f"BUNDLE NOT VERIFIED: {len(rep.fails)} failure(s)")
        for f in rep.fails:
            print("  " + f)
        return 1
    print("BUNDLE VERIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
