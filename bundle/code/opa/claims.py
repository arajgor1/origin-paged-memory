"""Claims, gates, unit-test records and the PAPER_INPUT.md skeleton for the proof bundle (phase P6).

Verdicts are computed with the verifier's own rule evaluator: ``tools/verify_bundle.py`` (human-owned) is imported
read-only and its ``eval_rule`` / ``Report`` / ``check_refs`` / ``resolve_ref`` are called as they are, so a claim
written here cannot disagree with the verifier. Holm adjustment uses ``opa.stats.holm`` (the same step-down
algorithm as the verifier's ``holm``; tests/test_bundle_tools.py checks that they agree).

P6 sequence (bundle_spec/BUNDLE_SPEC.md; every step is a function here or in opa.bundle):

    from opa import bundle as bd, claims as cl
    B = bd.B
    # 1. tables first (opa.analysis.tables.write_contrast_table / write_table); they are frozen after this step
    # 2. code into the bundle, then figures (their sidecars hash the tables and name the script)
    bd.copy_code()                                     # opa/stats.py, opa/claims.py, opa/analysis/*.py -> bundle/code/
    # 3. gates
    cl.write_gates(B, cl.complete_gates({"G0": cl.gate("G0", True, [...], [...]), ...}))
    # 3b. deviations (known from prereg/deviations/ and state/decisions.md) and unit-test records, BEFORE the claims
    cl.write_deviations(B, entries)                    # or "None."
    cl.write_unit_tests(B, records)                    # only if any Track B / model-level result is reported;
                                                       # run ids checked against the project's runs/index.csv
    # 4. claims: one list of results per hypothesis; Holm across primaries; verdicts from the frozen rules
    res = {"H1": [cl.result_from_table(B, "tables/H1_main.csv", "S11_vs_S2_lbmhqa_late_32k_primary", level="agent",
                                       primary=True), ...], ...}
    claims = cl.build_claims(res, cl.read_gates(B), deviations=devs, deviation_ids=cl.read_deviation_ids(B))
    cl.write_claims(B, claims)                         # rules + min_n from prereg/ (identical to the tagged copies)
    # 5. copy prereg, budget, runs cited by claims/*.json and tests/U*.json, env (reads steps 3b and 4)
    bd.assemble()
    cl.write_budget_table(B)                           # tables/budget.csv from the bundle's ledger and approvals
    cl.write_paper_input(B, claims, cl.read_gates(B))  # skeleton; fill the TODO prose, then re-check:
    assert not cl.paper_input_problems(B)
    bd.manifest()                                      # last
    # python tools/verify_bundle.py bundle/  ->  BUNDLE VERIFIED

Nothing here spends money or touches GCS.
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from . import PROJECT_DIR
from .stats import holm

H_IDS = [f"H{i}" for i in range(1, 14)]   # default (v3); the live list comes from rules.yaml via hypothesis_ids()
G_IDS = [f"G{i}" for i in range(5)]
U_IDS = [f"U{i}" for i in range(1, 9)]
GATE_ORDER = {g: i for i, g in enumerate(G_IDS)}
LEVELS = ("agent", "model", "systems", "measurement")
SPLITS = ("test", "dev", "n/a")
VERIFIER_PATH = PROJECT_DIR / "tools" / "verify_bundle.py"
SCHEMA_DIR = PROJECT_DIR / "bundle_spec"


class ClaimError(ValueError):
    """A claim, gate or record that the verifier would reject (message lists every problem)."""


# ====================================================================== read-only access to the verifier
_VERIFIER = None


def verifier():
    """The human-owned verifier module, loaded read-only from tools/verify_bundle.py (cached)."""
    global _VERIFIER
    if _VERIFIER is None:
        spec = importlib.util.spec_from_file_location("opa_verify_bundle_readonly", VERIFIER_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VERIFIER = mod
    return _VERIFIER


def _schema(kind: str) -> dict:
    return json.loads((SCHEMA_DIR / f"{kind}.schema.json").read_text(encoding="utf-8"))


def schema_errors(obj: dict, kind: str) -> list[str]:
    """jsonschema (Draft 2020-12) errors of ``obj`` against bundle_spec/<kind>.schema.json, as the verifier sees them."""
    import jsonschema
    errs = sorted(jsonschema.Draft202012Validator(_schema(kind)).iter_errors(obj), key=lambda e: list(e.path))
    return [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}" for e in errs]


# ====================================================================== pre-registration inputs
def load_prereg(prereg_dir: Path | str | None = None) -> tuple[dict, dict]:
    """(rules, min_n) from <prereg_dir>/rules.yaml and sample_sizes.yaml (default: the repo's prereg/).
    Use the bundle's copies (bundle/prereg) once assembled: that is what the verifier evaluates."""
    d = Path(prereg_dir) if prereg_dir is not None else PROJECT_DIR / "prereg"
    rules = yaml.safe_load((d / "rules.yaml").read_text(encoding="utf-8"))
    sizes_p = d / "sample_sizes.yaml"
    sizes = yaml.safe_load(sizes_p.read_text(encoding="utf-8")) if sizes_p.exists() else {}
    return rules, (sizes or {}).get("min_n", {}) or {}


def hypothesis_ids(rules: dict | None) -> list[str]:
    """Hypothesis ids from the frozen rules (sorted H1, H2, ... H13), as the verifier does; H_IDS if there are none."""
    return list(verifier().hypothesis_ids(rules))


def rule_leaves(rule: dict) -> list[dict]:
    """Leaf conditions of a (possibly nested) all_of / any_of rule, in order."""
    if "all_of" in rule:
        return [x for r in rule["all_of"] for x in rule_leaves(r)]
    if "any_of" in rule:
        return [x for r in rule["any_of"] for x in rule_leaves(r)]
    return [rule]


def rule_keys(rules: dict, hid: str) -> list[str]:
    return [leaf["key"] for leaf in rule_leaves(rules["hypotheses"][hid]["rule"])]


def evaluate(hid: str, results: Iterable[dict] | Mapping[str, dict], rules: dict, min_n: dict) -> tuple[str, list[str]]:
    """Verdict of the frozen rule for ``hid`` on ``results`` (a list of result dicts or {key: result}), computed by
    the verifier's own eval_rule. Returns (verdict, verifier failure messages); a non-empty list means the verifier
    would fail the bundle (e.g. a rule key without a pre-registered minimum n, or a non-test split)."""
    v = verifier()
    by_key = dict(results) if isinstance(results, Mapping) else {r["key"]: r for r in results}
    rep = v.Report()
    verdict = v.eval_rule(rules["hypotheses"][hid]["rule"], by_key, min_n, rep, hid)
    return verdict, list(rep.fails)


def evaluate_pilot(hid: str, results: Iterable[dict] | Mapping[str, dict], rules: dict,
                   point_estimate_only: bool = True) -> str:
    """Rule evaluation for a gate decision on pilot (dev) data, e.g. G1 = "H1 and H3 pass on the pilot".

    Every leaf is treated as a measurement (no minimum n, no test-split requirement) so the verifier's evaluator
    can run on dev results. With ``point_estimate_only`` (default) the CI and Holm conditions are dropped too: a
    50-items-per-cell pilot is not powered for them, so "pass" means the point estimates meet the kill-criterion
    thresholds. The comparison logic is still the verifier's. Never use this for a claim verdict."""
    rule = copy.deepcopy(rules["hypotheses"][hid]["rule"])
    for leaf in rule_leaves(rule):
        leaf["measurement"] = True
        if point_estimate_only:
            for k in ("require_ci_excludes_zero", "require_ci_lower_above", "require_p_holm_below"):
                leaf.pop(k, None)
    v = verifier()
    by_key = dict(results) if isinstance(results, Mapping) else {r["key"]: r for r in results}
    rep = v.Report()
    verdict = v.eval_rule(rule, by_key, {}, rep, hid)
    if rep.fails:
        raise ClaimError("; ".join(rep.fails))
    return verdict


def rule_text(rule: dict) -> str:
    """Plain-text rendering of a rule (for PAPER_INPUT.md; put it in backticks there)."""
    if "all_of" in rule:
        return "all of (" + "; ".join(rule_text(r) for r in rule["all_of"]) + ")"
    if "any_of" in rule:
        return "any of (" + "; ".join(rule_text(r) for r in rule["any_of"]) + ")"
    t, k = rule["type"], rule["key"]
    if t in ("min_delta", "min_value"):
        s = f"{k} {'>' if rule.get('strict') else '>='} {rule['threshold']}"
    elif t in ("max_value", "max_delta"):
        s = f"{k} {'<' if rule.get('strict') else '<='} {rule['threshold']}"
    elif t == "within":
        s = f"{rule['lo']} <= {k} <= {rule['hi']}"
    else:
        s = f"{k} {t}"
    extra = []
    if rule.get("require_ci_excludes_zero"):
        extra.append("95% CI lower bound > 0")
    if "require_ci_lower_above" in rule:
        extra.append(f"95% CI lower bound > {rule['require_ci_lower_above']}")
    if "require_p_holm_below" in rule:
        extra.append(f"Holm p < {rule['require_p_holm_below']}")
    if rule.get("measurement"):
        extra.append("measurement")
    return s + (" [" + ", ".join(extra) + "]" if extra else "")


# ====================================================================== results
def _num(x):
    if x is None or (isinstance(x, str) and x.strip() == ""):
        return None
    x = float(x)
    if math.isnan(x):
        return None
    return x


def result(key: str, value: float, *, level: str, n: int, split: str, run_ids: Sequence[str], table_ref: str,
           ci95: Sequence[float] | None = None, p_raw: float | None = None, primary: bool = False,
           test: str | None = None, description: str | None = None) -> dict:
    """One schema-valid claim result (bundle_spec/claim.schema.json). p_holm is filled by build_claims for
    primary results that report p_raw; it is None otherwise."""
    if level not in LEVELS:
        raise ClaimError(f"{key}: level {level!r} not in {LEVELS}")
    if split not in SPLITS:
        raise ClaimError(f"{key}: split {split!r} not in {SPLITS}")
    v = _num(value)
    if v is None or math.isinf(v):
        raise ClaimError(f"{key}: value {value!r} is not a finite number")
    out = {"key": key, "level": level, "value": v,
           "ci95": None if ci95 is None or any(_num(c) is None for c in ci95) else [float(ci95[0]), float(ci95[1])],
           "n": int(n), "p_raw": _num(p_raw), "p_holm": None, "split": split, "primary": bool(primary),
           "run_ids": [str(r) for r in run_ids if str(r)], "table_ref": table_ref}
    if test:
        out["test"] = test
    if description:
        out["description"] = description
    if not out["run_ids"]:
        raise ClaimError(f"{key}: a result needs at least one run id")
    return out


def table_ref(table_rel: str, key: str) -> str:
    return f"{Path(table_rel).as_posix()}#row={key}"


def read_table_row(bundle: Path | str, table_rel: str, key: str) -> dict:
    p = Path(bundle) / table_rel
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keycol = "key" if rows and "key" in rows[0] else (list(rows[0].keys())[0] if rows else None)
    hit = [r for r in rows if r.get(keycol) == key]
    if not hit:
        raise ClaimError(f"row {key} not found in {table_rel}")
    return hit[0]


_FROM_TABLE = object()
from .analysis.tables import NI_TEST_LABEL  # noqa: E402  (single definition)


def holm_required(rules: dict, hid: str, key: str | None = None) -> bool:
    """True if the primary result of ``hid`` (default: its primary_key) is decided with a Holm condition, i.e. its
    rule leaf has require_p_holm_below. Only those results enter the Holm family and carry p_raw; non-inferiority
    and measurement primaries (H8, H11, H13, ...) are decided from the CI lower bound and report p_raw null."""
    hr = rules["hypotheses"][hid]
    key = key or hr.get("primary_key")
    return any(leaf["key"] == key and "require_p_holm_below" in leaf for leaf in rule_leaves(hr["rule"]))


def non_inferiority_keys(rules: dict) -> list[str]:
    """Primary keys that are contrasts decided from the CI lower bound alone (no Holm condition, not a measurement:
    H8, H11, H13 in v3). Pass to opa.analysis.tables.write_contrast_table(ci_only_keys=...) so the table carries
    p_raw blank and the label NI_TEST_LABEL, in agreement with the claim."""
    out = []
    for hid, hr in rules["hypotheses"].items():
        pk = hr.get("primary_key")
        leaf = next((x for x in rule_leaves(hr["rule"]) if x["key"] == pk), None)
        if leaf and leaf.get("type") == "min_delta" and not holm_required(rules, hid, pk):
            out.append(pk)
    return out


def result_from_table(bundle: Path | str, table_rel: str, key: str, *, level: str, split: str | None = None,
                      primary: bool = False, run_ids: Sequence[str] | None = None, n: int | None = None,
                      description: str | None = None, p_raw=_FROM_TABLE) -> dict:
    """Result built from a bundle table row (columns key, value, ci_lo, ci_hi, n, p_raw, test, run_ids[, split]), so
    the claim and the table cannot disagree. ``split`` defaults to the row's split column, else "test" for
    contrasts and "n/a" for measurements. ``p_raw`` overrides the table's value (pass None for a primary whose rule
    has no Holm condition: see holm_required)."""
    row = read_table_row(bundle, table_rel, key)
    ci = (row.get("ci_lo"), row.get("ci_hi"))
    ci = None if any(_num(c) is None for c in ci) else ci
    rids = list(run_ids) if run_ids is not None else [r for r in (row.get("run_ids") or "").split(";") if r]
    sp = split or row.get("split") or ("n/a" if level == "measurement" else "test")
    nn = n if n is not None else int(float(row["n"])) if _num(row.get("n")) is not None else 0
    return result(key, float(row["value"]), level=level, n=nn, split=sp, run_ids=rids,
                  table_ref=table_ref(table_rel, key), ci95=ci, p_raw=row.get("p_raw") if p_raw is _FROM_TABLE else p_raw, primary=primary,
                  test=row.get("test") or None, description=description)


def result_from_contrast(row: dict, table_rel: str, *, level: str, primary: bool = False,
                         split: str | None = None, description: str | None = None, p_raw=_FROM_TABLE) -> dict:
    """Result from an opa.stats.contrast() dict (the same dict that was written to ``table_rel``); ``p_raw``
    overrides the dict's value (None for a non-Holm primary: see holm_required)."""
    return result(row["key"], row["value"], level=level, n=row["n"], split=split or row.get("split") or "test",
                  run_ids=row.get("run_ids") or [], table_ref=table_ref(table_rel, row["key"]),
                  ci95=row.get("ci95") or ((row.get("ci_lo"), row.get("ci_hi")) if row.get("ci_lo") is not None else None),
                  p_raw=row.get("p_raw") if p_raw is _FROM_TABLE else p_raw, primary=primary, test=row.get("test"),
                  description=description)


# ====================================================================== gates
def criterion(name: str, threshold: Any, observed: Any, met: bool) -> dict:
    return {"name": name, "threshold": threshold, "observed": observed, "met": bool(met)}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def gate(gid: str, passed: bool | None, criteria: Sequence[dict], evidence: Sequence[str], notes: str = "",
         decided_at: str | None = None) -> dict:
    """One gate record (bundle_spec/gate.schema.json). passed=None only for a gate never reached because an earlier
    gate failed (use unreached_gate)."""
    g = {"id": gid, "passed": passed, "decided_at": decided_at or _now(), "criteria": list(criteria),
         "evidence": list(evidence), "notes": notes}
    errs = schema_errors(g, "gate")
    if passed is not None and criteria and bool(passed) != all(c["met"] for c in criteria):
        errs.append(f"passed={passed} but criteria met = {[c['met'] for c in criteria]}")
    if errs:
        raise ClaimError(f"gate {gid}: " + "; ".join(errs))
    return g


def unreached_gate(gid: str, failed_gate: str, decided_at: str | None = None) -> dict:
    return gate(gid, None, [], [f"gates/{failed_gate}.json"],
                f"Not reached: {failed_gate} failed, so the phases this gate closes were skipped (CLAUDE.md s3).",
                decided_at)


def complete_gates(gates: Mapping[str, dict], decided_at: str | None = None) -> dict[str, dict]:
    """Add unreached (passed: null) records for G1-G3 after the first failed gate; G4 (paper-ready) is always
    decided in P6 and must be given."""
    out = dict(gates)
    failed = next((g for g in G_IDS if out.get(g, {}).get("passed") is False), None)
    for g in G_IDS:
        if g in out:
            continue
        if failed is not None and GATE_ORDER[g] > GATE_ORDER[failed] and g != "G4":
            out[g] = unreached_gate(g, failed, decided_at)
    missing = [g for g in G_IDS if g not in out]
    if missing:
        raise ClaimError(f"gates missing: {missing}")
    return out


def write_gates(bundle: Path | str, gates: Mapping[str, dict]) -> list[Path]:
    d = Path(bundle) / "gates"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for gid, g in gates.items():
        if g.get("id") != gid:
            raise ClaimError(f"gate {gid} has id {g.get('id')!r}")
        errs = schema_errors(g, "gate")
        if errs:
            raise ClaimError(f"gate {gid}: " + "; ".join(errs))
        p = d / f"{gid}.json"
        _write_json(p, g)
        out.append(p)
    return out


def read_gates(bundle: Path | str) -> dict[str, dict]:
    d = Path(bundle) / "gates"
    return {g: json.loads((d / f"{g}.json").read_text(encoding="utf-8")) for g in G_IDS if (d / f"{g}.json").exists()}


def failed_gate_for(hid: str, gates: Mapping[str, dict], rules: dict) -> str | None:
    """Earliest gate at or before the hypothesis's gate_dependency that shows passed: false, else None."""
    dep = rules["hypotheses"][hid].get("gate_dependency", "G0")
    for g in G_IDS:
        if GATE_ORDER[g] > GATE_ORDER[dep]:
            break
        if gates.get(g, {}).get("passed") is False:
            return g
    return None


# ====================================================================== claims
def build_claims(results: Mapping[str, Sequence[dict]], gates: Mapping[str, dict], *, rules: dict | None = None,
                 min_n: dict | None = None, prereg: Path | str | None = None,
                 figures: Mapping[str, Sequence[str]] | None = None,
                 deviations: Mapping[str, Sequence[str]] | None = None, notes: Mapping[str, str] | None = None,
                 not_run: Mapping[str, str] | None = None,
                 deviation_ids: Iterable[str] | None = None) -> dict[str, dict]:
    """One claim per hypothesis in rules.yaml (H1-H13 in v3), schema-valid, with Holm-adjusted p over the primary family and verdicts from the frozen rules.

    results:   {hid: [result dicts]}; a hypothesis with no results and a failed gate at or before its
               gate_dependency becomes not_run_gate_failed (failed_gate = the earliest such gate). A hypothesis with
               no results and no failed gate is evaluated anyway (-> inconclusive, as the rule says).
    not_run:   {hid: gate} to force not_run_gate_failed (the gate must show passed: false and be at or before the
               dependency).
    Holm:      over the primaries whose rule leaf has require_p_holm_below (H1-H6, H10, H12 in v3; not the CI-based
               H8, H11, H13), with opa.stats.holm; they must report p_raw and every other primary must not.
               Non-primary results keep p_holm = None.
    G2:        once G2 failed, results for H11, H12, H13 are rejected unless ``deviations[hid]`` is non-empty.
    Raises ClaimError listing every problem the verifier would flag (D2, D3, D4) and every scientific misuse it
    would not (a dev-split result under a rule key, a primary that is not the rule's primary_key)."""
    if rules is None or min_n is None:
        r2, m2 = load_prereg(prereg)
        rules = rules if rules is not None else r2
        min_n = min_n if min_n is not None else m2
    hyp = rules["hypotheses"]
    hids = hypothesis_ids(rules)
    figures, deviations, notes, not_run = figures or {}, deviations or {}, notes or {}, dict(not_run or {})
    problems: list[str] = []
    if deviation_ids is None and any(deviations.values()):
        problems.append("deviations are cited but deviation_ids is None; write deviations.md first and pass "
                        "deviation_ids=read_deviation_ids(bundle)")
    if deviation_ids is not None:
        known = set(deviation_ids)
        for h, refs in deviations.items():
            problems.extend(f"{h}: deviation {d} is not a '### DEV-n' entry in deviations.md" for d in refs if d not in known)
    unknown = sorted((set(results) | set(not_run)) - set(hids))
    if unknown:
        problems.append(f"unknown hypotheses {unknown}")

    res = {h: [dict(r) for r in results.get(h, [])] for h in hids}
    for h in hids:
        if not res[h] and h not in not_run:
            fg = failed_gate_for(h, gates, rules)
            if fg:
                not_run[h] = fg

    # ---- per-hypothesis checks
    for h in hids:
        if h in not_run:
            fg, dep = not_run[h], hyp[h].get("gate_dependency", "G0")
            if gates.get(fg, {}).get("passed") is not False:
                problems.append(f"{h}: not_run_gate_failed cites {fg}, which does not show passed: false")
            elif GATE_ORDER[fg] > GATE_ORDER[dep]:
                problems.append(f"{h}: depends on {dep} but cites the later gate {fg}")
            if res[h]:
                problems.append(f"{h}: not_run_gate_failed but results were given")
            continue
        keys = [r["key"] for r in res[h]]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            problems.append(f"{h}: duplicate result keys {dup}")
        prim = [r for r in res[h] if r.get("primary")]
        if len(prim) > 1:
            problems.append(f"{h}: {len(prim)} results marked primary (at most one)")
        pk = hyp[h].get("primary_key")
        fg = failed_gate_for(h, gates, rules)
        if fg and res[h] and not deviations.get(h) and not (h == "H9" and fg == "G2"):
            # H9 may run after a G2 failure when its envelope is funded (CLAUDE.md s3); everything else needs a deviation
            problems.append(f"{h}: results reported although gate {fg} failed; pass a deviation reference for {h} or "
                            f"drop the results")
        for r in prim:
            if r["key"] != pk:
                problems.append(f"{h}: primary result {r['key']} is not the rule's primary_key {pk}")
            elif holm_required(rules, h, pk):
                if r.get("p_raw") is None:
                    problems.append(f"{h}: primary {pk} is decided with a Holm condition but p_raw is None")
            elif r.get("p_raw") is not None:
                problems.append(f"{h}: primary {pk} has no Holm condition in rules.yaml (CI-based or measurement) "
                                f"but p_raw is set; pass p_raw=None")
            if r.get("split") != "test":
                problems.append(f"{h}: primary result {r['key']} has split {r.get('split')!r}, not 'test'")
        leaves = {leaf["key"]: leaf for leaf in rule_leaves(hyp[h]["rule"])}
        for r in res[h]:
            leaf = leaves.get(r["key"])
            if leaf is None:
                continue
            if leaf.get("measurement"):
                if r.get("split") == "dev":
                    problems.append(f"{h}: {r['key']} feeds the verdict but is a dev-split measurement")
            else:
                if r.get("split") != "test":
                    problems.append(f"{h}: {r['key']} feeds the verdict but split is {r.get('split')!r}, not 'test'")
                if r["key"] not in min_n or min_n.get(r["key"]) is None:
                    problems.append(f"{h}: {r['key']} has no pre-registered minimum n in sample_sizes.yaml")
            if r.get("level") == "measurement" and not leaf.get("measurement"):
                problems.append(f"{h}: {r['key']} is a contrast in rules.yaml but level is 'measurement'")
        for f in figures.get(h, []):
            if not str(f).startswith("figures/") or not str(f).endswith(".png"):
                problems.append(f"{h}: figure {f} must be figures/<name>.png")

    # ---- Holm across the primary family (same family the verifier recomputes)
    family = {f"{h}:{r['key']}": float(r["p_raw"]) for h in hids if h not in not_run for r in res[h]
              if r.get("primary") and r.get("p_raw") is not None and holm_required(rules, h, r["key"])}
    adj = holm(family)
    for h in hids:
        for r in res[h]:
            r["p_holm"] = adj.get(f"{h}:{r['key']}")

    # ---- claims and verdicts
    claims = {}
    for h in hids:
        c = {"id": h, "statement": hyp[h]["statement"], "verdict": None, "results": res[h],
             "figures": list(figures.get(h, [])), "deviations": list(deviations.get(h, [])), "notes": notes.get(h, "")}
        if h in not_run:
            c["verdict"] = "not_run_gate_failed"
            c["failed_gate"] = not_run[h]
            if not c["notes"]:
                c["notes"] = (f"Not run: gate {not_run[h]} failed (gates/{not_run[h]}.json); this hypothesis depends on "
                              f"{hyp[h].get('gate_dependency', 'G0')}.")
        else:
            verdict, fails = evaluate(h, res[h], rules, min_n)
            problems.extend(fails)
            c["verdict"] = verdict
        problems.extend(f"claims/{h}.json: {e}" for e in schema_errors(c, "claim"))
        claims[h] = c
    if problems:
        raise ClaimError("claims rejected:\n  " + "\n  ".join(problems))
    return claims


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_claims(bundle: Path | str, claims: Mapping[str, dict]) -> list[Path]:
    out = []
    for h, c in claims.items():
        errs = schema_errors(c, "claim")
        if errs or c.get("id") != h:
            raise ClaimError(f"claims/{h}.json: " + "; ".join(errs or [f"id {c.get('id')!r}"]))
        p = Path(bundle) / "claims" / f"{h}.json"
        _write_json(p, c)
        out.append(p)
    return out


# ====================================================================== unit tests, deviations, budget table
def read_deviation_ids(bundle: Path | str) -> list[str]:
    """The '### DEV-n' ids in bundle/deviations.md (empty for "None.")."""
    p = Path(bundle) / "deviations.md"
    if not p.exists():
        return []
    return [ln[4:].strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.startswith("### ")]


BF16_UNIT_IDS = ("U2", "U3", "U4")     # the Track B model runs these in fp32 and bf16 (opa.trackb.units)


def unit_test_problems(rec: Mapping, *, require_bf16: bool = False) -> list[str]:
    """Why a unit-test record may not be written as tests/U<i>.json (empty list: fine). A consistent passed: false
    record is NOT a problem here: it is written and the verifier's D5 then fails, as intended."""
    out = []
    uid = rec.get("id")
    if rec.get("stage") == "fp32_only" or rec.get("action") == "unit_test_fp32_stage":
        out.append("fp32-only stage record (event-log intermediate), not a final unit-test record")
    if uid not in U_IDS:
        out.append(f"id {uid!r} not in {U_IDS}")
    bf = rec.get("bf16")
    if require_bf16 and uid in BF16_UNIT_IDS and not isinstance(bf, Mapping):
        out.append(f"{uid} from the Track B model needs its bf16 block")
    if isinstance(bf, Mapping):
        fp = rec.get("fp32")
        fp_ok = fp.get("passed") if isinstance(fp, Mapping) and "passed" in fp else None
        bf_bad = bf.get("passed") is not True or bool(bf.get("error")) or bool(bf.get("state_restored"))
        if rec.get("passed") is True and (bf_bad or fp_ok is False):
            out.append("passed: true but fp32/bf16 evidence says otherwise "
                       f"(fp32 passed={fp_ok}, bf16 passed={bf.get('passed')}, error={bf.get('error')!r}, "
                       f"state_restored={bf.get('state_restored')!r})")
        if rec.get("passed") is False and not bf_bad and fp_ok is True:
            out.append("passed: false although fp32 and bf16 both passed (passed must equal fp32 AND bf16)")
    if rec.get("error") and rec.get("passed") is True:
        out.append("passed: true although the record carries an error")
    return out


def _run_status(runs_index: Path) -> dict[str, str]:
    with runs_index.open(newline="", encoding="utf-8") as f:
        return {r["run_id"]: r.get("status", "") for r in csv.DictReader(f)}


def write_unit_tests(bundle: Path | str, records: Iterable[dict], *, runs_index: Path | str | None = None,
                     require_bf16: bool = True, same_run_deviation: str | None = None,
                     deviation_ids: Iterable[str] | None = None) -> list[Path]:
    """tests/U<i>.json from records as opa.trackb.units writes them ({id, name, passed, details, run_id, ...}).

    Rejects (ClaimError, nothing written): fp32-only stage records; ids outside U1-U8 or repeated in one call;
    passed: true contradicting a bf16 block (bf16.passed false, bf16.error, non-empty bf16.state_restored) or an
    fp32 block; U2-U4 without a bf16 block (require_bf16=False for the fallback model or a bf16-less run);
    a run_id missing from ``runs_index`` (default: the project's runs/index.csv; assemble then copies the run
    folders cited by tests/U*.json into the bundle) or without status ok; U1-U5 coming from different runs unless
    ``same_run_deviation`` names a deviation (it must be in ``deviation_ids``, default
    read_deviation_ids(bundle)) that explains it; it is stored in each record as ``deviation``. A consistent passed: false record is written."""
    records = list(records)
    problems = []
    ids = [r.get("id") for r in records]
    problems += [f"duplicate id {i} in one call" for i in sorted({i for i in ids if ids.count(i) > 1}, key=str)]
    for rec in records:
        problems += [f"tests/{rec.get('id')}.json: {m}" for m in unit_test_problems(rec, require_bf16=require_bf16)]
    ri = Path(runs_index) if runs_index is not None else PROJECT_DIR / "runs" / "index.csv"
    status = _run_status(ri) if ri.exists() else {}
    for rec in records:
        st = status.get(rec.get("run_id"))
        if st is None:
            problems.append(f"tests/{rec.get('id')}.json: run_id {rec.get('run_id')!r} not in {ri}")
        elif st != "ok":
            problems.append(f"tests/{rec.get('id')}.json: run {rec.get('run_id')} has status {st!r}")
    core = {r.get("run_id") for r in records if r.get("id") in ("U1", "U2", "U3", "U4", "U5")}
    if len(core) > 1 and not same_run_deviation:
        problems.append(f"U1-U5 come from {len(core)} different runs; pass same_run_deviation=<DEV-n> if a deviation explains it")
    if same_run_deviation:
        known = set(deviation_ids) if deviation_ids is not None else set(read_deviation_ids(bundle))
        if same_run_deviation not in known:
            problems.append(f"same_run_deviation {same_run_deviation} is not a '### DEV-n' entry in deviations.md")
    if problems:
        raise ClaimError("unit-test records rejected:\n  " + "\n  ".join(problems))
    out = []
    for rec in records:
        if same_run_deviation:
            rec = {**rec, "deviation": same_run_deviation}
        errs = schema_errors(rec, "test")
        if errs:
            raise ClaimError(f"tests/{rec.get('id')}.json: " + "; ".join(errs))
        p = Path(bundle) / "tests" / f"{rec['id']}.json"
        _write_json(p, rec)
        out.append(p)
    return out


def write_deviations(bundle: Path | str, entries: Sequence[dict] = ()) -> Path:
    """deviations.md: "None." or one '### DEV-n' block per entry {id, date, change, reason, impact}."""
    p = Path(bundle) / "deviations.md"
    if not entries:
        p.write_text("None.\n", encoding="utf-8", newline="\n")
        return p
    blocks = []
    for i, e in enumerate(entries, 1):
        missing = [k for k in ("date", "change", "reason", "impact") if not str(e.get(k, "")).strip()]
        if missing:
            raise ClaimError(f"deviation {e.get('id', i)} lacks {missing}")
        blocks.append(f"### {e.get('id', f'DEV-{i}')}\n- Date: {e['date']}\n- Change: {e['change']}\n"
                      f"- Reason: {e['reason']}\n- Impact: {e['impact']}\n")
    p.write_text("# Deviations from the pre-registration\n\n" + "\n".join(blocks), encoding="utf-8", newline="\n")
    return p


def budget_rows(bundle: Path | str) -> list[dict]:
    """Per-envelope approved and actual spend from the bundle's budget/approvals.yaml and ledger.csv (the same
    approval rule the verifier uses), plus a 'total' row whose approved_usd is total_cap_usd."""
    b = Path(bundle)
    appr = yaml.safe_load((b / "budget" / "approvals.yaml").read_text(encoding="utf-8")) or {}
    ok = {}
    for e in appr.get("envelopes", []):
        if float(e.get("approved_usd") or 0) > 0 and str(e.get("approved_by") or "").strip() and str(e.get("approved_on") or "").strip():
            ok[str(e["id"])] = float(e["approved_usd"])
    spent: dict[str, float] = {}
    with (b / "budget" / "ledger.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["kind"] == "actual":
                spent[r["envelope"]] = spent.get(r["envelope"], 0.0) + float(r["amount_usd"])
    rows = [{"key": e, "approved_usd": ok.get(e, 0.0), "actual_usd": round(spent.get(e, 0.0), 6)}
            for e in sorted(set(ok) | set(spent))]
    rows.append({"key": "total", "approved_usd": float(appr.get("total_cap_usd") or 0),
                 "actual_usd": round(sum(spent.values()), 6)})
    return rows


def write_budget_table(bundle: Path | str, rel: str = "tables/budget.csv") -> Path:
    from .analysis.tables import write_table
    return write_table(Path(bundle) / rel, budget_rows(bundle), columns=["key", "approved_usd", "actual_usd"])


# ====================================================================== PAPER_INPUT.md skeleton
def fmt_num(x: float, decimals: int) -> str:
    """ASCII number at ``decimals`` places (ASCII '-' so the verifier's number regex sees the sign)."""
    s = f"{float(x):.{decimals}f}"
    return "0." + "0" * decimals if s.lstrip("-") == "0." + "0" * decimals and decimals else s


P_MAX_DECIMALS = 6


def p_decimals(p: float) -> int:
    """Decimals that show a p-value with two significant digits in positional notation (never scientific, which
    the verifier's number regex cannot read): 0.040, 0.012, 0.00012; capped at 6 (p < 5e-7 shows as 0.000000,
    which still matches the source at the displayed precision; _p_text adds a backticked "< 1e-6" note)."""
    if p is None or p <= 0:
        return 3
    return min(P_MAX_DECIMALS, max(3, int(math.ceil(-math.log10(p))) + 1))


def _p_text(bundle: Path, target: str, p: float) -> str:
    s = ref_numbers(bundle, target, [p_decimals(p)])[0]
    return f"{s} [[ref:{target}]]" + (" `(< 1e-6)`" if p < 0.5 * 10 ** -P_MAX_DECIMALS else "")


def default_decimals(key: str, level: str) -> int:
    return 1 if ("_vs_" in key and level != "measurement") else 3


def ref_numbers(bundle: Path | str, target: str, decimals: Sequence[int]) -> list[str]:
    """The values a [[ref:target]] tag resolves to (via the verifier's own resolve_ref), formatted at ``decimals``,
    so displayed numbers always match their reference."""
    vals = verifier().resolve_ref(target, Path(bundle))
    if len(vals) != len(decimals):
        raise ClaimError(f"{target}: {len(vals)} values but {len(decimals)} decimals given")
    return [fmt_num(v, d) for v, d in zip(vals, decimals)]


def _result_line(bundle: Path, h: str, i: int, r: dict, decimals: int) -> str:
    path, _, frag = r["table_ref"].partition("#")
    key = frag.split("=", 1)[1]
    row = read_table_row(bundle, path, key)
    cols = ["value"] + (["ci_lo", "ci_hi"] if _num(row.get("ci_lo")) is not None and _num(row.get("ci_hi")) is not None else [])
    tgt = f"{path}#row={key}&cols={','.join(cols)}"
    nums = ref_numbers(bundle, tgt, [decimals] * len(cols))
    unit = " points" if ("_vs_" in key and r["level"] != "measurement") else ""
    tag = "primary" if r.get("primary") else "secondary"
    s = f"`{key}` ({tag}; {r['level']}; split {r['split']}; n = {r['n']}): {nums[0]}{unit}"
    if len(nums) == 3:
        s += f" (95% CI {nums[1]} to {nums[2]})"
    s += f" [[ref:{tgt}]]"
    if r.get("p_raw") is not None:
        s += ", raw p = " + _p_text(bundle, f"claims/{h}.json#results.{i}.p_raw", r["p_raw"])
    if r.get("p_holm") is not None:
        s += ", Holm-adjusted p = " + _p_text(bundle, f"claims/{h}.json#results.{i}.p_holm", r["p_holm"])
    if r.get("test"):
        s += f"; test: `{r['test']}`"
    return s


def write_paper_input(bundle: Path | str, claims: Mapping[str, dict], gates: Mapping[str, dict], *,
                      rules: dict | None = None, min_n: dict | None = None,
                      decimals: Mapping[str, int] | None = None, captions: Mapping[str, str] | None = None,
                      measured: Sequence[dict] = (), summary: Mapping[str, str] | None = None) -> Path:
    """Skeleton bundle/PAPER_INPUT.md from bundle_spec/PAPER_INPUT.template.md: run summary, one Results subsection
    per hypothesis with every number tagged [[ref:...]] (values resolved through the verifier's resolve_ref), the
    measured-numbers table, and TODO markers for the prose the agent must write. Claims must already be written
    to bundle/claims (p-values are referenced from there) and tables must be final.

    measured: rows for "Measured numbers for the paper": {location (shown in backticks), old (paper's value, shown
              in backticks), ref (a [[ref:]] target with one value), decimals, note (plain text: keep any decimal
              number in it inside backticks or the verifier asks for a reference)}.
    After editing the prose, run paper_input_problems(bundle)."""
    b = Path(bundle)
    if rules is None or min_n is None:
        r2, m2 = load_prereg(b / "prereg" if (b / "prereg" / "rules.yaml").exists() else None)
        rules, min_n = rules or r2, min_n if min_n is not None else m2
    decimals, captions, summary = dict(decimals or {}), dict(captions or {}), dict(summary or {})
    L: list[str] = []
    A = L.append
    A("# PAPER_INPUT - Origin-Paged Attention experiments")
    A("")
    A("This file is the hand-off to the paper-drafting assistant. Write it for a reader who has the v2 paper "
      "(docs/paper_v2.md) but has not seen any run. Every decimal number, percentage and dollar amount in "
      "\"Results\" and \"Measured numbers for the paper\" carries a `[[ref:...]]` tag (bundle_spec/BUNDLE_SPEC.md).")
    A("")
    A("## 0. Run summary")
    A("")
    runs_idx = b / "runs" / "index.csv"
    starts, ends = [], []
    if runs_idx.exists():
        with runs_idx.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("started_at"):
                    starts.append(r["started_at"])
                if r.get("finished_at"):
                    ends.append(r["finished_at"])
    A(f"- Run window (UTC): {summary.get('window') or (f'{min(starts)} to {max(ends or starts)}' if starts else 'TODO')}")
    A(f"- Models: {summary.get('models', 'TODO: subject Qwen/Qwen3-4B-Instruct-2507 [revision TODO]; replication google/gemma-4-E4B-it [revision TODO]; embedder microsoft/harrier-oss-v1-0.6b [revision TODO]; backup embedder Qwen/Qwen3-Embedding-4B [revision TODO]')}")
    tag_p = b / "prereg" / "PREREG_TAG.json"
    if tag_p.exists():
        t = json.loads(tag_p.read_text(encoding="utf-8"))
        A(f"- Pre-registration: tag {t.get('tag')} at commit {t.get('commit')}, {t.get('committed_at')}")
    else:
        A("- Pre-registration: TODO (prereg/PREREG_TAG.json missing)")
    gl = []
    for g in G_IDS:
        p = gates.get(g, {}).get("passed")
        gl.append(f"{g} {'pass' if p is True else 'fail' if p is False else 'not reached'}")
    A("- Gates: " + ", ".join(gl))
    if (b / "tables" / "budget.csv").exists():
        tot, cap = ref_numbers(b, "tables/budget.csv#row=total&cols=actual_usd,approved_usd", [2, 2])
        A(f"- Spend: total ${tot} of an approved ${cap} [[ref:tables/budget.csv#row=total&cols=actual_usd,approved_usd]]")
    else:
        A("- Spend: TODO (write tables/budget.csv with write_budget_table first)")
    dv = b / "deviations.md"
    n_dev = 0
    if dv.exists():
        n_dev = sum(1 for ln in dv.read_text(encoding="utf-8").splitlines() if ln.startswith("### "))
    A(f"- Deviations: {n_dev} (see deviations.md)")
    A("")
    A("## Results")
    A("")
    for h in hypothesis_ids(rules):
        c = claims[h]
        hr = rules["hypotheses"][h]
        A(f"### {h} - {c['statement']} - VERDICT: {c['verdict']}")
        A("")
        if c["verdict"] == "not_run_gate_failed":
            A(f"- **Not run.** Gate {c['failed_gate']} failed (gates/{c['failed_gate']}.json); this hypothesis depends on "
              f"{hr.get('gate_dependency', 'G0')}, so its phase was skipped as the protocol's gate rules require.")
            A(f"- **Rule that would have applied.** `{rule_text(hr['rule'])}`")
            A("")
            continue
        order = sorted(range(len(c["results"])), key=lambda i: (not c["results"][i].get("primary"), i))
        for j, i in enumerate(order):
            r = c["results"][i]
            dec = decimals.get(r["key"], default_decimals(r["key"], r["level"]))
            label = "Primary result" if r.get("primary") else ("Result" if j == 0 else "Further result")
            A(f"- **{label}.** " + _result_line(b, h, i, r, dec))
        if not c["results"]:
            A("- **Results.** None reported: every rule input is missing, so the frozen rule gives inconclusive.")
        needs = sorted({leaf['key'] for leaf in rule_leaves(hr['rule']) if not leaf.get('measurement')})
        mn = ", ".join(f"{k} >= {min_n.get(k)}" for k in needs)
        A(f"- **Rule applied.** `{rule_text(hr['rule'])}`" + (f"; pre-registered minimum n: `{mn}`" if mn else "")
          + f". The frozen rule gives **{c['verdict']}**. TODO: one sentence on which condition decided it.")
        for f in c["figures"]:
            A(f"- **Figure.** {f} - {captions.get(f, 'TODO: one-sentence caption')}")
        A("- **Failure analysis.** TODO (failure-code distribution with references, if relevant)")
        A(f"- **Caveats and deviations.** {', '.join(c['deviations']) if c['deviations'] else 'none'}. TODO: limitations specific to this result.")
        A("")
    A("## Measured numbers for the paper")
    A("")
    A("Each row replaces a synthetic or assumed number in docs/paper_v2.md with a measured one, or confirms it. "
      "The paper's old values are in backticks (quotes, not measurements); every measured value has a reference.")
    A("")
    A("| Paper location | Synthetic or assumed value in v2 | Measured value | Note |")
    A("|---|---|---|---|")
    for m in measured:
        v = ref_numbers(b, m["ref"], [int(m.get("decimals", 3))])[0]
        A(f"| `{m['location']}` | `{m.get('old', '')}` | {v} [[ref:{m['ref']}]] | {m.get('note', '')} |")
    if not measured:
        A("| TODO | | | |")
    A("")
    A("## Framing implied by the results")
    A("")
    A(f"- H5 outcome and the framing it implies (protocol Section 5.5 table): verdict {claims['H5']['verdict']}. TODO")
    A("- Which paper sections are now evidence-backed, which are refuted, which remain theory: TODO")
    A("- Claims to remove or soften: TODO")
    A("")
    A("## Surprises")
    A("")
    A("TODO")
    A("")
    A("## Limitations observed")
    A("")
    A("TODO (model scale, synthetic-data dependence, small open-weight models only, cost-driven reductions)")
    A("")
    A("## Reproduction")
    A("")
    A("- Re-run a primary result: `python -m opa.launch run jobs/<job>.yaml` with the recorded seed and git SHA.")
    A("- Data and run logs: gs://[bucket]/opa/ (append-only).")
    A("")
    p = b / "PAPER_INPUT.md"
    p.write_text("\n".join(L), encoding="utf-8", newline="\n")
    probs = paper_input_problems(b, claims)
    if probs:
        raise ClaimError("generated PAPER_INPUT.md fails the verifier's checks:\n  " + "\n  ".join(probs))
    return p


def paper_input_problems(bundle: Path | str, claims: Mapping[str, dict] | None = None) -> list[str]:
    """The verifier's D7 (reference) and D10 (verdict heading) checks on bundle/PAPER_INPUT.md, plus a portability
    check: the verifier reads the file with the platform's default encoding (cp1252 on this Windows host), so
    characters whose UTF-8 bytes are undefined there (e.g. U+201D, U+2010) crash it, and a Unicode minus (U+2212)
    before a number is lost. Run the final verification on Linux or with PYTHONUTF8=1 if non-ASCII prose is kept."""
    import re
    b = Path(bundle)
    v = verifier()
    p = b / "PAPER_INPUT.md"
    if not p.exists():
        return ["PAPER_INPUT.md missing"]
    raw = p.read_bytes()
    out = []
    try:
        raw.decode("cp1252")
    except UnicodeDecodeError as e:
        out.append(f"PAPER_INPUT.md is not decodable as cp1252 at byte {e.start} (the verifier would crash on a "
                   f"Windows host without PYTHONUTF8=1); replace the character")
    text = raw.decode("utf-8")
    if "−" in text:
        out.append("PAPER_INPUT.md contains U+2212 (Unicode minus); use ASCII '-' before numbers")
    rep = v.Report()
    v.check_refs(text, b, rep)
    out.extend(rep.fails)
    res = v.results_section(text)
    if not res:
        out.append("[D10] PAPER_INPUT.md has no '## Results' section")
    if claims is None:
        claims = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted((b / "claims").glob("H*.json"))}
    known_dev = set(read_deviation_ids(b))
    for h, c in claims.items():
        out.extend(f"[D11] claims/{h}.json cites deviation {d}, not a '### DEV-n' entry in deviations.md"
                   for d in c.get("deviations", []) if d not in known_dev)
    for h, c in claims.items():
        pat = re.compile(rf"^###\s+{h}\b.*VERDICT:\s*{re.escape(c.get('verdict', ''))}\b", re.M | re.I)
        if not pat.search(res):
            out.append(f"[D10] Results section lacks a '### {h} ... VERDICT: {c.get('verdict')}' heading")
    return out


def preflight(bundle: Path | str, repo: Path | str | None = None) -> list[str]:
    """Run the verifier itself (in-process) and return its failure list; [] means BUNDLE VERIFIED.
    repo=None skips the git checks (like --no-git)."""
    rep = verifier().verify(Path(bundle).resolve(), Path(repo).resolve() if repo else None)
    return list(rep.fails)


__all__ = ["ClaimError", "verifier", "schema_errors", "load_prereg", "rule_leaves", "rule_keys", "evaluate",
           "evaluate_pilot", "rule_text", "result", "table_ref", "read_table_row", "result_from_table",
           "result_from_contrast", "holm_required", "non_inferiority_keys", "NI_TEST_LABEL", "read_deviation_ids", "unit_test_problems", "BF16_UNIT_IDS", "criterion", "gate", "unreached_gate", "complete_gates", "write_gates",
           "read_gates", "failed_gate_for", "build_claims", "write_claims", "write_unit_tests", "write_deviations",
           "budget_rows", "write_budget_table", "fmt_num", "p_decimals", "ref_numbers", "write_paper_input",
           "paper_input_problems", "preflight"]
