"""bundle/PAPER_INPUT_interim[_dev].md: the interim Track A hand-off, in the section structure of
bundle_spec/PAPER_INPUT.template.md. Every decimal number, percentage and dollar amount (the verifier's definition:
bundle_spec/BUNDLE_SPEC.md "References in PAPER_INPUT.md") carries a ``[[ref:tables/<file>.csv#row=<key>&cols=<col>]]`` tag
(or a gates/*.json path) that resolves; integers (counts, steps, depths) are not required to carry one. Values printed in
backticks are quotes or code and are skipped by the verifier.

``check_refs(md_path, bundle_dir)`` re-checks the file with the verifier's own parsing (opa.claims.verifier().check_refs:
REF_RE, NUM_RE, resolve_ref, number_matches) over EVERY section, not only the two sections the verifier reads, and returns
the list of problems (empty = fine).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from .. import claims as cl
from .. import PROJECT_DIR
from . import interim as itm
from .tables import read_table

def fmt_away(x: float, dec: int) -> str:
    """Fixed-point text rounded half away from zero (never toward the favourable side for a negative effect: -65.625 at 2
    decimals is -65.63, not -65.62); no negative zero."""
    from decimal import Decimal, ROUND_HALF_UP
    v = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)
    out = format(v, "f")
    return out.lstrip("-") if v == 0 else out


def exact_decimals(x: float, lo: int = 2, hi: int = 6) -> int:
    """Decimals that show ``x`` exactly (as repr gives it), at least ``lo`` and at most ``hi``."""
    from decimal import Decimal
    e = Decimal(repr(float(x))).as_tuple().exponent
    return min(hi, max(lo, -e if isinstance(e, int) and e < 0 else 0))


REF_SECTION_PREFIXES = ("## Results", "## Measured numbers for the paper")


# ---------------------------------------------------------------------------------------------- reference checker
def check_refs(md_path: Path | str, bundle_dir: Path | str, strict: bool = True) -> list[str]:
    """Problems found by the verifier's reference check on ``md_path``. With ``strict`` every ``## `` section is checked
    (the verifier itself reads only "## Results" and "## Measured numbers for the paper"). Also reports any [[ref:...]] that
    does not resolve, wherever it stands."""
    v = cl.verifier()
    text = Path(md_path).read_text(encoding="utf-8")
    problems: list[str] = []
    for m in v.REF_RE.finditer(text):
        try:
            v.resolve_ref(m.group(1), Path(bundle_dir))
        except Exception as e:                                          # noqa: BLE001 - report, do not stop
            problems.append(f"unresolvable reference {m.group(1)}: {e}")
    if strict:
        text = re.sub(r"^## (?!Results)", "## Results — ", text, flags=re.M)
    rep = v.Report()
    v.check_refs(text, Path(bundle_dir), rep)
    return problems + list(rep.fails)


# ---------------------------------------------------------------------------------------------- text from tables
_DEFANG = re.compile(r"(?<![\w.])(-?\d+\.\d+|\$\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?%(?! CI))(?![\w.])")


def defang(text: str) -> str:
    """Put decimals, dollar amounts and percentages of free text taken from a table (notes, basis, criteria names) in
    backticks, so they count as quotes and not as unreferenced measurements; text already in backticks is untouched."""
    parts = str(text).split("`")
    for i in range(0, len(parts), 2):
        parts[i] = _DEFANG.sub(lambda m: f"`{m.group(1)}`", parts[i])
    return "`".join(parts)


# ---------------------------------------------------------------------------------------------- builder
class Doc:
    def __init__(self, bundle: Path, split: str):
        self.b, self.split, self.sfx = bundle, split, itm.suffix(split)
        self.final: Mapping | None = None
        self.lines: list[str] = []
        self._cache: dict[str, dict] = {}

    # tables --------------------------------------------------------------------------------
    def tab(self, name: str) -> dict:
        if name not in self._cache:
            p = self.b / "tables" / name
            self._cache[name] = {r["key"]: r for r in read_table(p)} if p.exists() else {}
        return self._cache[name]

    def cell(self, name: str, row: str, col: str):
        r = self.tab(name).get(row)
        return None if r is None else itm.f(r.get(col))

    def num(self, name: str, row: str, col: str, dec: int, pct: bool = False) -> str:
        """"<number> [[ref:...]]" or "n/a" when the cell is empty."""
        x = self.cell(name, row, col)
        if x is None:
            return "n/a"
        shown = fmt_away(x * 100 if pct else x, dec) + ("%" if pct else "")
        return f"{shown} [[ref:tables/{name}#row={row}&cols={col}]]"

    def pnum(self, name: str, row: str, col: str) -> str:
        x = self.cell(name, row, col)
        if x is None:
            return "n/a"
        if x < 5e-7:                                        # 0.0 (underflow) or below the display precision
            return self.num(name, row, col, 6) + " `(< 0.000001)`"
        return self.num(name, row, col, cl.p_decimals(x))

    def design(self, row: str, dec: int) -> str:
        return self.num(f"interim_design{self.sfx}.csv", row, "value", dec)

    def tiny(self, name: str, row: str, col: str) -> str:
        """A very small measured value: fixed-point (6 decimals, referenced) plus the scientific form in backticks."""
        x = self.cell(name, row, col)
        if x is None:
            return "n/a"
        return f"{self.num(name, row, col, 6)} (`{x:.3g}`)"

    def render_obs(self, path: str, idx: int, observed: Any) -> str:
        """The observed value of a gate criterion with every non-integer number referenced into the gate JSON."""
        base = f"criteria.{idx}.observed"

        def walk(o, dotted):
            if isinstance(o, bool):
                return str(o).lower()
            if isinstance(o, int):
                return str(o)
            if isinstance(o, float):
                return self.jref(path, dotted, exact_decimals(o))
            if isinstance(o, list):
                return "[" + ", ".join(walk(x, f"{dotted}.{i}") for i, x in enumerate(o)) + "]"
            if isinstance(o, dict):
                return "; ".join(f"{k} {walk(v, f'{dotted}.{k}')}" for k, v in o.items())
            return f"`{o}`"
        return walk(observed, base)

    def usd(self, name: str, row: str, col: str) -> str:
        x = self.cell(name, row, col)
        return "n/a" if x is None else f"${fmt_away(x, 2)} [[ref:tables/{name}#row={row}&cols={col}]]"

    def jref(self, path: str, dotted: str, dec: int, pct: bool = False) -> str:
        p = self.b / path
        if not p.exists():
            return "n/a"
        cur: Any = json.loads(p.read_text(encoding="utf-8"))
        for part in dotted.split("."):
            cur = cur[int(part)] if isinstance(cur, list) else cur[part]
        x = float(cur)
        shown = fmt_away(x * 100 if pct else x, dec) + ("%" if pct else "")
        return f"{shown} [[ref:{path}#{dotted}]]"

    # text ----------------------------------------------------------------------------------
    def add(self, *ls: str) -> None:
        self.lines.extend(ls)

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def _verdict(d: Doc, vname: str, key: str) -> str:
    r = d.tab(vname).get(key) or {}
    return r.get("verdict_interim", "n/a")


def _ci(d: Doc, name: str, row: str, dec: int) -> str:
    return f"{d.num(name, row, 'ci_lo', dec)} to {d.num(name, row, 'ci_hi', dec)}"


def _pm(d: Doc, vname: str, row: str) -> str:
    """raw p and Bonferroni p of a verdict-table row."""
    if d.cell(vname, row, "p_raw") is None:
        return "no p-value (decided from the CI or a measurement)"
    if d.final is not None:
        return (f"raw p = {d.pnum(vname, row, 'p_raw')}, Holm-adjusted p = {d.pnum(vname, row, 'p_holm')} "
                "(over every primary that reports a p)")
    prov = (f", provisional Holm p = {d.pnum(vname, row, 'p_holm_provisional')}" if d.cell(vname, row, "p_holm_provisional") is not None else "")
    return (f"raw p = {d.pnum(vname, row, 'p_raw')}, Bonferroni-adjusted p (m = {int(d.cell(vname, row, 'family_m') or 8)}) = "
            f"{d.pnum(vname, row, 'p_bonferroni')}{prov}; final Holm {itm.HOLM_PENDING}")


def floor_lines(d: Doc, vt: Mapping, P2: str, rule_key: str, hid: str) -> list[str]:
    """Floor disclosure for a non-inferiority leaf whose two arm rates are both below the floor limit (verdict-table note)."""
    leaf = vt.get(f"{hid}:{rule_key}") or {}
    if "at floor" not in str(leaf.get("notes", "")):
        return []
    n = int(d.cell(P2, rule_key, "n") or 0)
    ea, eb = d.cell(P2, rule_key, "em_a"), d.cell(P2, rule_key, "em_b")
    m = re.search(r"M=(\d+)", str((d.tab(P2).get(rule_key) or {}).get("note", "")))
    mtxt = f" = {m.group(1)}" if m else ""
    dev = ""
    devn = "p2_dev_check/p2_contrasts_dev.csv"
    if d.split == "test" and d.cell(devn, rule_key, "em_a") is not None:
        dev = (f"; dev pilot {d.num(devn, rule_key, 'em_a', 3)} vs {d.num(devn, rule_key, 'em_b', 3)}")
    return [f"- **Floor disclosure.** Arm rates: descent {d.num(P2, rule_key, 'em_a', 3)}, flat {d.num(P2, rule_key, 'em_b', 3)}. Both arms are at "
            f"floor (every gold origin within M{mtxt} slots for {int(round((ea or 0) * n))} and {int(round((eb or 0) * n))} of {n} items{dev}), so the "
            f"{leaf.get('verdict_interim', '')} verdict is non-inferiority at floor and says nothing about routing quality; the number of "
            "scores is not compared."]


def leaf_section(d: Doc, V: str, vt: Mapping, hid: str, rules: dict) -> bool:
    """Result lines for the leaf rows of ``hid`` in the verdict table (Track B test tables); False if there are none."""
    leaves = [r for k, r in vt.items() if k.startswith(hid + ":") and r.get("kind") == "leaf"]
    if not leaves:
        return False
    d.add(f"- **Rule applied.** `{(vt.get(hid) or {}).get('rule', '')}`. Hypothesis verdict {_verdict(d, V, hid)}; leaves:")
    for r in leaves:
        k = r["key"]
        meas = r.get("level") == "measurement"
        val = d.num(V, k, "value", 3 if meas else 1)
        ci = f" (95% CI {_ci(d, V, k, 3 if meas else 1)})" if d.cell(V, k, "ci_lo") is not None else ""
        n = f", n = {int(d.cell(V, k, 'n'))}" if d.cell(V, k, "n") else ""
        mn = f" (minimum {int(d.cell(V, k, 'min_n'))})" if d.cell(V, k, "min_n") else ""
        pm = f", {_pm(d, V, k)}" if d.cell(V, k, "p_raw") is not None else ""
        d.add(f"  - `{r['rule_key']}`: {val}{ci}{n}{mn}{pm}; leaf outcome {r['verdict_interim']}; reason: {r.get('verdict_reason', '')}.")
    if hid == "H4":
        if (d.b / "figures" / "fig_v3_learning_curve_dev.png").exists():
            d.add("- **Figure.** figures/fig_v3_learning_curve_dev.png \u2014 the V3 learning curve on the dev monitoring set (own-origin fraction, "
                  "attribution, passkey, loss) with the stopping step and the G2 state in the title.")
        d.add("- **Disclosure.** B1's attribution reflects the text origin headers that both V3 and B1 carry (registration Section 12 item 22), so "
              "V3 minus B1 measures provenance beyond what the headers give.")
    return True


def gate_criteria_lines(d: Doc, gid: str) -> list[str]:
    p = d.b / "gates" / f"{gid}.json"
    if not p.exists():
        return []
    out = []
    for i, c in enumerate(json.loads(p.read_text(encoding="utf-8")).get("criteria", [])):
        name = re.sub(r"[;,]?\s*[^;,()]*\d+(?:\.\d+)?%[^;,()]*", "", c.get("name", ""))
        out.append(f"- {defang(name)} (threshold {defang(str(c.get('threshold', '')))}): {d.render_obs(f'gates/{gid}.json', i, c.get('observed'))}; "
                   f"met {str(bool(c.get('met'))).lower()}.")
    return out


def write_paper_input_interim(bundle: Path | str, split: str, *, rules: dict | None = None, min_n: Mapping | None = None,
                              args_runs: Mapping | None = None, icfg: Mapping | None = None, root: Path | str | None = None,
                              final: Mapping | None = None) -> Path:
    """``final``: {"claims": {hid: claim}, "verdict_table": "final_verdicts.csv", "family_table": "final_family.csv"} writes the final
    bundle/PAPER_INPUT.md (verdict words from the claims, Holm-adjusted p-values, no interim wording) instead of the interim file."""
    bundle = Path(bundle)
    itm.check_split(split)
    rules = rules or cl.load_prereg(Path(root or PROJECT_DIR) / "prereg")[0]
    d = Doc(bundle, split)
    d.final = final
    s = d.sfx
    fs = s if split == "dev" else "_test_with_dev_pilot"          # figure / figure-table stem suffix (test mode holds both splits)
    V, P2 = (final["verdict_table"] if final else f"interim_verdicts{s}.csv"), f"p2_contrasts_{split}.csv"
    U, H7T, FAM = f"interim_units{s}.csv", f"interim_h7{s}.csv", f"interim_family{s}.csv"
    if final:
        FAM = final["family_table"]
    CE, CG, GT = f"interim_costs_envelope{s}.csv", f"interim_costs_group{s}.csv", f"interim_gates{s}.csv"
    GC, GH, DV, TC = f"interim_gate_criteria{s}.csv", f"interim_gate_history{s}.csv", f"interim_deviations{s}.csv", f"interim_trackc{s}.csv"
    LC, TBS = "v3_learning_curve_dev.csv", "trackb_status_dev.csv"
    vt = d.tab(V)
    g2rec = itm.load_g2(bundle)
    g2s = itm.g2_state(g2rec)
    devtag = split == "dev"
    banner = ("DEV DRY RUN: every value below comes from the dev pilot (the pipeline exercised on dev data). It is NOT a result "
              "and must not be quoted as one.") if devtag else (
        "INTERIM: Track A (agents) on the test split, plus the Track B status (learning curve, stopping rule, G2, and the H4 / H6 test "
        "tables when they exist). Verdicts are the frozen rules applied to the data that exist now; the final Holm correction is pending "
        "until every Track B result of the family is in.")

    if final:
        banner = ("FINAL: every hypothesis H1 to H13 with the verdict the frozen rule gives (the verifier recomputes it); Holm-adjusted "
                  "p-values over every primary that reports a p.")
    d.add("# PAPER_INPUT — Origin-Paged Attention experiments" + ("" if final else " (interim), Track A"), "",
          f"> {banner}", "",
          ("This file is the hand-off to the paper-drafting assistant" if final else "This file is the interim hand-off to the paper-drafting assistant")
          + ", in the section structure of "
          "bundle_spec/PAPER_INPUT.template.md. Every decimal number, percentage and dollar amount carries a reference tag that "
          "resolves to a bundle table (check: `python -m opa.analysis.paper_input_interim`). Integers (counts, steps, depths) are "
          "written plain. Numbers in backticks are quotes or code.", "")

    # ---------------------------------------------------------------- 0. run summary
    tag = Path(root or PROJECT_DIR) / "prereg" / "PREREG_TAG.json"
    tagj = json.loads(tag.read_text(encoding="utf-8")) if tag.exists() else {}
    as_of = (d.tab(CE).get("TOTAL") or {}).get("as_of", "")
    if split == "test" and args_runs and tag.exists():
        tb_ids = [i for r in d.tab("trackb/trackb_series_test.csv").values() for i in (r.get("run_ids") or "").split(";") if i]
        n_a = sum(len(v) for v in args_runs.values())
        n_chk, bad = itm.started_after_tag({**args_runs, "trackb": tb_ids}, tag)
        after_tag = (f"all {n_chk} test-split runs ({n_a} Track A + {len(tb_ids)} Track B) started after it (checked by the UTC timestamp in each run id)." if not bad
                     else "WARNING, runs that did not start after it: " + "; ".join(bad))
    else:
        after_tag = "(dev dry run: the started-after-tag check applies to the test split only.)"
    gst = {k: r.get("status", "") for k, r in d.tab(GT).items()}
    cfg = (yaml.safe_load((Path(root or PROJECT_DIR) / "config" / "project.yaml").read_text(encoding="utf-8")) or {}) \
        if (Path(root or PROJECT_DIR) / "config" / "project.yaml").exists() else {}
    revs = cfg.get("model_revisions") or {}
    models = {m: f"`{m}@{revs[m]}`" if m in revs else f"`{m}`" for m in ("Qwen/Qwen3-4B-Instruct-2507", "microsoft/harrier-oss-v1-0.6b")}
    window = None
    ip = (d.b / "runs" / "index.csv")
    if final and ip.exists():
        with ip.open(newline="", encoding="utf-8") as fh:
            ix = list(csv.DictReader(fh))
        st = sorted(r["started_at"] for r in ix if r.get("started_at"))
        fn = sorted(r["finished_at"] for r in ix if r.get("finished_at"))
        window = (st[0], fn[-1]) if st and fn else None
    miss_p = [r["hypothesis"] + " (" + r.get("status", "") + ")" for r in d.tab(FAM).values() if not r.get("p_raw")]
    prov_missing = ((", ".join(miss_p) + " contribute no p-value (registration Section 5); ") if miss_p else "") + \
        "the provisional Holm stays provisional until the bundle is assembled."
    d.add("## 0. Run summary", "",
          f"- Data as of (UTC, last ledger line): `{as_of}`. Split of the verdict table: `{split}`.",
          *([f"- Run window (UTC): `{window[0]}` to `{window[1]}` (first start and last finish among the cited runs, runs/index.csv)."] if window else []),
          f"- Models: subject model, all tracks {models['Qwen/Qwen3-4B-Instruct-2507']}; embedder {models['microsoft/harrier-oss-v1-0.6b']}; "
          "Track A replication on Gemma 4 E4B and the embedder robustness check are not run (not funded).",
          f"- Pre-registration: tag `prereg-v2` at commit `{str(tagj.get('commit', ''))[:12]}`, committed `{tagj.get('committed_at', '')}`; "
          f"{after_tag}",
          f"- Gates: G0 {gst.get('G0', 'n/a')}, G1 {gst.get('G1', 'n/a')} (owner override, disclosed below), G2 {gst.get('G2', 'n/a')}, "
          f"G3 {gst.get('G3', 'n/a')}, G4 {gst.get('G4', 'n/a')}.",
          f"- Spend: total {d.usd(CE, 'TOTAL', 'spent_usd')} actual plus {d.usd(CE, 'TOTAL', 'reserved_usd')} reserved, of an approved cap "
          f"{d.usd(CE, 'TOTAL', 'approved_usd')}.",
          f"- Deviations and disclosures: {len(d.tab(DV))} (listed in full below).",
          *([f"- Holm family: {', '.join(r['hypothesis'] for r in d.tab(FAM).values() if r.get('p_raw'))} report a p-value and are adjusted "
             f"together (Holm-Bonferroni, `opa.stats.holm`, as the verifier recomputes it); {', '.join(miss_p)} contribute none "
             "(registration Section 5)."] if final else
            [f"- Holm family: {', '.join(r['hypothesis'] for r in d.tab(FAM).values())} (m = {len(d.tab(FAM))}); p-values below are raw and "
             f"Bonferroni-adjusted over the whole family; the final Holm is {itm.HOLM_PENDING}. A provisional Holm over the primaries that "
             f"report a p now ({', '.join(r['hypothesis'] for r in d.tab(FAM).values() if r.get('p_raw'))}) is shown next to them; {prov_missing}"]), "")

    # ---------------------------------------------------------------- Results
    d.add("## Results", "", *(["Verdicts: the frozen rules (prereg/rules.yaml) applied by the verifier's own rule evaluator to the claim results. "
          "A leaf whose n is below the pre-registered minimum is inconclusive, whatever its point estimate. A hypothesis whose gate failed "
          "is `not_run_gate_failed` with the gate named; hypotheses without a funded test are inconclusive with \"not funded\" in their notes. "
          "Intention to treat applies throughout.", ""] if final else []),
          *([] if final else ["Interim verdicts: the frozen rules (prereg/rules.yaml) evaluated by the verifier's own rule evaluator on the results "
          "that exist now. A leaf whose n is below the pre-registered minimum is inconclusive, whatever its point estimate. "
          "Where a leaf needs a Holm-adjusted p, the Bonferroni p over the full family stands in (it is never smaller than the "
          "Holm p, so a \"supported\" stays supported); a refutation that holds only under the Bonferroni stand-in is marked pending the "
          "final Holm (`pending_final_holm`), with both outcomes in the verdict basis. The provisional Holm column uses opa.stats.holm over the "
          "primaries that report a p now. Hypotheses outside the interim scope are listed with "
          "their status.", ""]))

    def hsec(hid: str):
        h = rules["hypotheses"][hid]
        if final:
            cv = final["claims"][hid]["verdict"]
            note = " (not funded)" if hid == "H2" else (f" ({final['claims'][hid].get('failed_gate')})" if cv == "not_run_gate_failed" else "")
            d.add(f"### {hid} — {h['statement']} — VERDICT: {cv}{note}", "")
        else:
            d.add(f"### {hid} — {h['statement']} — VERDICT: {_verdict(d, V, hid)}", "")

    # H1
    hsec("H1")
    k1, k2 = "H1:S11_vs_S2_lbmhqa_late_32k_primary", "H1:S11_vs_S4_lbmhqa_late_32k_primary"
    c1, c2 = "S11_vs_S2_lbmhqa_late_32k_primary", "S11_vs_S4_lbmhqa_late_32k_primary"
    d.add(f"- **Primary result.** S11 vs S2 on LB-MHQA late h=2, 32K, n = {int(d.cell(V, k1, 'n') or 0)} (pre-registered minimum "
          f"{int(d.cell(V, k1, 'min_n') or 0)}): {d.num(V, k1, 'value', 1)} points (95% CI {_ci(d, V, k1, 1)}), {_pm(d, V, k1)}.",
          f"- **Second leaf.** S11 vs S4 (first {int(d.cell(V, k2, 'n') or 0)} H1 items): {d.num(V, k2, 'value', 1)} points "
          f"(95% CI {_ci(d, V, k2, 1)}); minimum n {int(d.cell(V, k2, 'min_n') or 0)}; leaf outcome {_verdict(d, V, k2)}.",
          f"- **Rule applied.** `{(vt.get('H1') or {}).get('rule', '')}`. Leaf outcome S11 vs S2: {_verdict(d, V, k1)}. Basis: "
          f"{(vt.get(k1) or {}).get('verdict_reason', '')}. The same estimates would give "
          f"{(vt.get('H1') or {}).get('verdict_ignoring_min_n', 'n/a')} if the minimum n were met (informational only).",
          f"- **Arm EMs.** S11 {d.num(P2, c1, 'em_a', 2)}, S2 {d.num(P2, c1, 'em_b', 2)}, S4 {d.num(P2, c2, 'em_b', 2)} on the shared items.",
          "- **Secondary results.** `S11_vs_S2_slope_per_h_late`, `S11_vs_S2_lbmhqa_late_16k`, `S11_vs_S2_lbmhqa_late_8k`: not run "
          "(only the 32K h=2 late cell is funded; prereg Section 2b).",
          f"- **Figure.** figures/fig_h1_depth_context{fs}.png — the H1 contrasts on the single funded cell"
          + (", dev pilot only." if devtag else "; the dev pilot is drawn as hollow diamonds for context, the test split as filled circles."),
          f"- **Failure analysis.** Intention-to-treat: pairs with an imputed (failed) record {int(d.cell(P2, c1, 'n_itt_imputed') or 0)}, "
          f"pairs excluded for a missing record {int(d.cell(P2, c1, 'n_excluded_missing') or 0)} (tables/{P2}).",
          "- **Caveats and deviations.** DEV-G0-noise: S1 repeats differ by near-tie flips, per-item flip rate "
          f"{_g0_flip(d)} at n = 50 on G0; the H1 test is sized for 7 points (design power {d.design('h1_design_power_at_5_points', 2)} at the 5-point rule threshold, "
          f"{d.design('h1_design_power_at_7_points', 2)} at 7 points; disclosed in the registration). S4 is the v2 baseline (gist cap 300, tuned within the protocol's tuning budget; G1 audit).", "")

    for hid in ("H2",):
        hsec(hid)
        d.add(f"- Status: {defang((vt.get(hid) or {}).get('verdict_basis', ''))}.", "")

    # H3
    hsec("H3")
    k3 = "H3:S11_vs_S8_late_bound"
    legv = _verdict(d, V, k3)
    d.add(f"- **Hypothesis verdict and leg outcome.** {(vt.get('H3') or {}).get('notes', '')}. Leg outcome on its own: {legv}.",
          f"- **Primary result (the testable leg).** S11 vs S8 (graph edges add recall beyond skeletons) on LB-MHQA late h=2, 32K, "
          f"n = {int(d.cell(V, k3, 'n') or 0)} (minimum {int(d.cell(V, k3, 'min_n') or 0)}): {d.num(V, k3, 'value', 1)} points "
          f"(95% CI {_ci(d, V, k3, 1)}), {_pm(d, V, k3)}.",
          f"- **Rule applied.** `{(vt.get('H3') or {}).get('rule', '')}`. Reason: {(vt.get(k3) or {}).get('verdict_reason', '')}.",
          "- **Secondary results.** none reported.", "- **Figure.** none (the leg shares the cell of the H1 figure).",
          "- **Failure analysis.** shares the H1 runs.", "- **Caveats and deviations.** R2 leaf not testable at 32K: the coverage predictor "
          "equals its maximum for almost every item, so no theory can fit it (pilot identity-line R2 below zero; disclosed in the "
          "registration and in the G1 history).", "")

    hsec("H4")
    if not leaf_section(d, V, vt, "H4", rules):
        d.add(f"- Status: {defang((vt.get('H4') or {}).get('verdict_basis', ''))}.")
    d.add("")

    # H5
    hsec("H5")
    k5 = "H5:A2_vs_A1_agent_mtlbmhqa_32k_primary_cross"
    c5 = "A2_vs_A1_agent_mtlbmhqa_32k_primary_cross"
    d.add(f"- **Hypothesis verdict and the agent arm are different things.** The rule is an any_of over the agent arm and the model "
          f"arm; the model arm is {(vt.get('H5:A2_vs_A1_model_thread_confusion_task') or {}).get('notes', 'pending Track B').replace('model arm of the any_of rule: ', '')}, "
          f"so a failed agent arm alone leaves the hypothesis {_verdict(d, V, 'H5')}.",
          f"- **Primary result (agent arm).** A2 (S12) vs A1 (S14) on MT-LB-MHQA T4 h=3 cross, 32K, n = {int(d.cell(V, k5, 'n') or 0)} "
          f"(minimum {int(d.cell(V, k5, 'min_n') or 0)}): {d.num(V, k5, 'value', 1)} points (95% CI {_ci(d, V, k5, 1)}), {_pm(d, V, k5)}. "
          f"Leaf outcome {_verdict(d, V, k5)}.",
          f"- **Rule applied.** `{(vt.get('H5') or {}).get('rule', '')}`. Reason: {(vt.get(k5) or {}).get('verdict_reason', '')}.")
    kt, kp = "H5:A2_vs_A3_agent_equivalence", "H5:A2_vs_A1_pages_read_before_gold_hit"
    if kt in vt:
        d.add(f"- **Secondary: A2 vs A3 equivalence (TOST, margin 3 points).** {d.num(V, kt, 'value', 1)} points (95% CI {_ci(d, V, kt, 1)}), "
              f"n = {int(d.cell(V, kt, 'n') or 0)}, TOST p = {d.pnum(V, kt, 'p_raw')}; reported, no verdict.")
    if kp in vt:
        d.add(f"- **Secondary: pages read before the first gold page, A2 minus A1.** {d.num(V, kp, 'value', 2)} pages (95% CI "
              f"{_ci(d, V, kp, 2)}), n = {int(d.cell(V, kp, 'n') or 0)}; "
              "reported, no verdict (lower is better).")
    d.add(f"- **Figure.** figures/fig_h5_thread_depth{fs}.png — left: the agent-level difference at T4 (the only funded thread depth); right: "
          "the model-free Euclidean-over-hyperbolic distortion ratio at T = 4, 8, 16 (dev only).",
          "- **Failure analysis.** see tables/" + P2 + " (ITT columns).",
          "- **Caveats and deviations.** `A2_vs_A1_slope_per_thread_depth`: not run (one thread depth funded). The question thread is the "
          f"largest thread at T8/T16 (structural confound, prereg Section 12 item 18). Track A lambda {d.design('track_a_lambda', 1)} was not tuned for S12/S14/S15.", "")
    dist = d.tab("fig_h5_thread_depth" + fs + ".csv")
    if dist:
        rows = [r for r in dist.values() if r.get("panel") == "distortion"]
        if rows:
            parts = []
            for r in sorted(rows, key=lambda r: int(float(r["threads_T"]))):
                parts.append(f"T{int(float(r['threads_T']))} {d.num('fig_h5_thread_depth' + fs + '.csv', r['key'], 'value', 2)}")
            d.add("- **Distortion check (dev, model-free).** Euclidean over hyperbolic tree distortion: " + "; ".join(parts) + ".", "")

    hsec("H6")
    if not leaf_section(d, V, vt, "H6", rules):
        d.add(f"- Status: {defang((vt.get('H6') or {}).get('verdict_basis', ''))}.")
    d.add("")

    # H7
    hsec("H7")
    h7 = {k: d.cell(H7T, k, "value") for k in itm.H7_KEYS}
    if h7.get("global_fail_depth_bf16") is not None:
        d.add(f"- **Primary result.** First failing depth of global coordinates: bf16 {int(h7['global_fail_depth_bf16'])}, fp16 "
              f"{int(h7['global_fail_depth_fp16'])}, fp32 {int(h7['global_fail_depth_fp32'])} (predicted 4 / 7 / 16; rule bands 3-5, 6-8, "
              f"15-17). Re-anchored bf16 error growth, depth 40 over depth 1: "
              f"{d.num(H7T, 'reanchored_error_growth_ratio_depth40_vs_depth1_bf16', 'value', 3)} (limit {d.design('h7_reanchored_growth_limit', 1)}).",
              f"- **Rule applied.** `{(vt.get('H7') or {}).get('rule', '')}`. All four measurement leaves met -> {_verdict(d, V, 'H7')}.",
              "- **Caveats.** A measurement on 512 synthetic chains (split n/a), run after the tag on dev-seeded chains; the re-anchored "
              "ratio composes two offsets in the LCA frame of one node, so it demonstrates same-branch precision and cannot show "
              "cross-branch error growth.", "")
    else:
        d.add("- No H7 measurement was given.", "")

    # H8
    hsec("H8")
    k8 = "H8:descent_beam4_vs_flat_recall_at_M"
    d.add(f"- **Primary result.** Descent (beam 4) minus flat scan, recall at M slots, n = {int(d.cell(V, k8, 'n') or 0)} (minimum "
          f"{int(d.cell(V, k8, 'min_n') or 0)}): {d.num(V, k8, 'value', 1)} points (95% CI {_ci(d, V, k8, 1)}); non-inferiority decided "
          "from the CI lower bound, so no p-value.",
          f"- **Rule applied.** `{(vt.get('H8') or {}).get('rule', '')}`. Reason: {(vt.get(k8) or {}).get('verdict_reason', '')}.",
          *floor_lines(d, vt, P2, "descent_beam4_vs_flat_recall_at_M", "H8"),
          "- **Caveats.** Computed on the H5 items with real skeletons and harrier embeddings; no model run (no intention-to-treat case).", "")

    hsec("H9")
    d.add(f"- Status: {defang((vt.get('H9') or {}).get('verdict_basis', ''))}.", "")

    # H10
    hsec("H10")
    k10 = "H10:pinned_vs_unpinned_repeat_pagein_reduction"
    d.add(f"- **Primary result.** Repeated page-ins per item, unpinned minus pinned, n = {int(d.cell(V, k10, 'n') or 0)} (minimum "
          f"{int(d.cell(V, k10, 'min_n') or 0)}): {d.num(V, k10, 'value', 2)} (95% CI {_ci(d, V, k10, 2)}), {_pm(d, V, k10)}.",
          f"- **Rule applied.** `{(vt.get('H10') or {}).get('rule', '')}`. Reason: {(vt.get(k10) or {}).get('verdict_reason', '')}.",
          "- **Caveats.** Failures are imputed with the worst value of the (r, M) cell (intention to treat).", "")

    for hid in ("H11", "H12", "H13"):
        hsec(hid)
        d.add(f"- Status: {defang((vt.get(hid) or {}).get('verdict_basis', ''))}.", "")

    # ---------------------------------------------------------------- Measured numbers
    d.add("## Measured numbers for the paper", "",
          "Each row replaces a synthetic or assumed number in docs/paper_v2.md with a measured one, or states that it is not measured. "
          "Old values are in backticks (quotes, skipped by the verifier).", "",
          "| Paper location | Synthetic or assumed value in v2 | Measured value | Note |", "|---|---|---|---|")
    lcs = d.tab(TBS)
    if final and "H4:V3_own_origin_mass_fraction" in vt:
        d.add("| §7.2 own-origin mass | `0.913` (positional head, simulation) | "
              f"{d.num(V, 'H4:V3_own_origin_mass_fraction', 'value', 3)} | trained V3 at the stopping step, mean over the test probe windows, all layers; "
              + (f"the dev monitoring set gave {d.num(TBS, 'V3_latest_own_origin_mass_fraction', 'value', 3)} at step "
                 f"{int(d.cell(TBS, 'V3_latest_own_origin_mass_fraction', 'step') or 0)} (dev, not a result); " if "V3_latest_own_origin_mass_fraction" in lcs else "")
              + "not at lambda 5 |")
    elif "V3_latest_own_origin_mass_fraction" in lcs:
        d.add("| §7.2 own-origin mass | `0.913` (positional head, simulation) | "
              f"{d.num(TBS, 'V3_latest_own_origin_mass_fraction', 'value', 3)} | trained V3, all-layer mean at step "
              f"{int(d.cell(TBS, 'V3_latest_own_origin_mass_fraction', 'step') or 0)}; dev monitoring set; not at lambda 5 |")
    if k8 in vt:
        fl = floor_lines(d, vt, P2, "descent_beam4_vs_flat_recall_at_M", "H8")
        flt = (f"; arm recalls descent {d.num(P2, 'descent_beam4_vs_flat_recall_at_M', 'em_a', 3)}, flat {d.num(P2, 'descent_beam4_vs_flat_recall_at_M', 'em_b', 3)}: "
               "both arms are at floor, so this is non-inferiority at floor, not evidence about routing quality" if fl else "")
        d.add(f"| §8.3 descent vs flat at matched M | simulation only | {d.num(V, k8, 'value', 1)} | real summaries, MT T4 h3 cross 32K, "
              f"recall points, n = {int(d.cell(V, k8, 'n') or 0)}{flt} |")
    d.add(f"| §11 page-in copy vs prefill | `2.7 ms` at `25 GB/s` (arithmetic only) | not measured | {defang((d.tab(TC).get('H9') or {}).get('status', 'Track C not started'))} |")
    if h7.get("global_fail_depth_bf16") is not None:
        d.add("| §(global-coordinate failure depth, bf16) | `4` (predicted) | "
              f"{int(h7['global_fail_depth_bf16'])} | table interim_h7 row global_fail_depth_bf16; fp16 {int(h7['global_fail_depth_fp16'])}, "
              f"fp32 {int(h7['global_fail_depth_fp32'])} |")
        d.add(f"| §(re-anchored bf16 error growth, depth 40 vs 1) | limit {d.design('h7_reanchored_growth_limit', 1)} | "
              f"{d.num(H7T, 'reanchored_error_growth_ratio_depth40_vs_depth1_bf16', 'value', 3)} | same-branch precision only |")
    d.add("")
    d.add("### Units U1-U8", "")
    for i in range(1, 9):
        r = d.tab(U).get(f"U{i}")
        if not r:
            continue
        obs = itm.f(r.get("observed"))
        if obs is not None and (abs(obs) < 1e-3 and obs != 0):
            shown = f"observed {d.tiny(U, f'U{i}', 'observed')}"
        elif obs is not None:
            shown = f"observed {d.num(U, f'U{i}', 'observed', 3)}"
        else:
            shown = "observed n/a"
        thr = itm.f(r.get("threshold"))
        if thr is not None:
            shown += f", threshold {d.tiny(U, f'U{i}', 'threshold') if thr < 1e-3 else d.num(U, f'U{i}', 'threshold', 1)}"
        aux = itm.f(r.get("aux_observed"))
        if aux is not None:
            shown += f"; {r.get('aux_name')} {d.tiny(U, f'U{i}', 'aux_observed') if abs(aux) < 1e-3 else d.num(U, f'U{i}', 'aux_observed', 3)}"
        if itm.f(r.get("ulps")) is not None:
            shown += f", fp32 error in ulps of the largest key {d.num(U, f'U{i}', 'ulps', 1)}"
        dev = f" ({r['deviation']})" if r.get("deviation") else ""
        d.add(f"- U{i}: {r['status']}{dev}; {shown}. {defang(r.get('notes', ''))}")
    d.add("")

    # ---------------------------------------------------------------- remaining sections
    d.add("## Gates", "")
    for gid in ("G0", "G1", "G2", "G3", "G4"):
        r = d.tab(GT).get(gid)
        if r:
            d.add(f"- **{gid}** {r['status']}" + (f" (decided `{r['decided_at']}`, criteria met {r['n_met']} of {r['n_criteria']})"
                                                   if r.get("decided_at") else "") + (f". {r.get('note', '')}" if r["status"] == "pending" else "."))
    d.add("", "G0 criteria, with the observed values (\"met false\" entries are attempts that were superseded by a pre-committed final "
          "attempt or a fallback, as the gate record explains; they are kept, not removed):", "")
    g0p = d.b / "gates" / "G0.json"
    if g0p.exists():
        for i, c in enumerate(json.loads(g0p.read_text(encoding="utf-8")).get("criteria", [])):
            name = re.sub(r"[;,]?\s*[^;,()]*\d+(?:\.\d+)?%[^;,()]*", "", c.get("name", ""))      # measured clauses live in the record
            d.add(f"- {defang(name)}: {d.render_obs('gates/G0.json', i, c.get('observed'))}; met {str(bool(c.get('met'))).lower()} "
                  f"(full criterion text in gates/G0.json).")
    g2lines = gate_criteria_lines(d, "G2")
    if g2lines:
        d.add("", f"G2 criteria, with the observed values (G2 {g2s}):", "", *g2lines)
    d.add("", "### G1 and the owner override (disclosure)", "",
          "G1 was first decided under the protocol's definition at `2026-11-07T18:30:16Z` and failed on H3's R2 criterion alone; H1 "
          "passed. The owner then replaced the definition (PLAN_V3 revision 3, same day) and G1 was re-decided and passed. The original "
          "decision is kept unchanged in bundle/gates/history/.", "")
    for k, r in d.tab(GC).items():
        if "original" in r["gate"] and "R2" in r["name"]:
            d.add(f"- Original G1, the failed criterion: pilot R2 of the H3 predictor {d.num(GC, k, 'observed_value', 2)} against a "
                  f"threshold of {d.design('h3_r2_threshold', 1)} (the predictor equals its maximum for almost every item at 32K).")
    d.add("- Revision 3 definition: S4 passes the baseline audit and H1 passes on the pilot's point estimates; H3 is not part of G1.",
          "- Results that had been seen when the override was made are listed in the registration (Section 4).", "")

    d.add("## Deviations and disclosures", "", f"Every pre-registered deviation file and every disclosure ({len(d.tab(DV))} entries):", "")
    for k, r in d.tab(DV).items():
        d.add(f"- **{k}** ({r['kind']}, `{r['date']}`): {defang(r['title'])} (source: {r['source']})")
    d.add("")

    fr = d.tab(f"interim_failed_runs{s}.csv")
    if fr:
        sens = d.tab(f"interim_rerun_sensitivity{s}.csv")
        d.add("## Failed and excluded runs", "", "Runs that did not finish ok. None is deleted; each is excluded from the analysis for the reason given.", "")
        for k, r in fr.items():
            d.add(f"- `{k}` (job {r['job']}, attempt {r['attempt']} of {r['max_attempts']}): {r['status']}; {r['reason']}. Handling: {r['handling']}.")
            for sk, sr in sens.items():
                if sr["first_run"] != k:
                    continue
                if not itm.f(sr.get("n_shared")):
                    d.add(f"  - {sr['system']}: no records shared with the retry.")
                else:
                    d.add(f"  - Sensitivity {sr['system']}: EM of the first attempt minus the retry {d.num(f'interim_rerun_sensitivity{s}.csv', sk, 'diff_points', 1)} "
                          f"points on {int(float(sr['n_shared']))} shared items (first {d.num(f'interim_rerun_sensitivity{s}.csv', sk, 'em_first', 3)}, retry "
                          f"{d.num(f'interim_rerun_sensitivity{s}.csv', sk, 'em_retry', 3)}).")
        d.add("")

    d.add("## Costs to date", "", f"As of `{as_of}` (ledger, read-only). Reservations are held for running jobs and released when they settle.", "",
          "| Envelope | Approved | Spent | Reserved | Remaining |", "|---|---|---|---|---|")
    for k, r in d.tab(CE).items():
        d.add(f"| {k} | {d.usd(CE, k, 'approved_usd')} | {d.usd(CE, k, 'spent_usd')} | {d.usd(CE, k, 'reserved_usd')} | "
              f"{d.usd(CE, k, 'remaining_usd')} |")
    d.add("", "| Work group | Hypotheses | Spent |", "|---|---|---|")
    for k, r in d.tab(CG).items():
        d.add(f"| {k} | {r['hypotheses']} | {d.usd(CG, k, 'spent_usd')} |")
    d.add("")

    d.add("## Track B status", "")
    lc = d.tab(LC)
    if lc:
        d.add("V3 learning curve on the dev monitoring set (arm V3):", "")
        for k, r in lc.items():
            d.add(f"- step {int(float(r['opt_step']))}: own-origin mass fraction {d.num(LC, k, 'own_origin_mass_fraction', 3)}, ratio to uniform "
                  f"{d.num(LC, k, 'own_origin_mass_ratio_to_uniform', 2)}, attribution EM {d.num(LC, k, 'attrib_em', 3)}, passkey EM "
                  f"{d.num(LC, k, 'passkey_em', 3)}, training loss {d.num(LC, k, 'train_loss', 2) if d.cell(LC, k, 'train_loss') is not None else 'n/a'}.")
        fired = "V3_stopping_rule_fired_step" in lcs
        rk = "V3_rise_last_two_evaluations"
        if rk in lcs:
            step = int(d.cell(TBS, rk, "step") or 0)
            if "V3_stopped_at_cap" in lcs:
                d.add(f"- Stopping rule: did not fire; training stopped at the 2,000-step cap (step {int(d.cell(TBS, 'V3_stopped_at_cap', 'step') or 0)}); "
                      f"the last rise over 200 steps is {d.num(TBS, rk, 'value', 4)} against the threshold {d.num(TBS, rk, 'threshold', 2)}.")
            elif fired:
                d.add(f"- Stopping rule: fired at step {step} (L = {step}): the own-origin mass fraction rose by {d.num(TBS, rk, 'value', 4)} between steps "
                      f"{step - 200} and {step}, below the threshold {d.num(TBS, rk, 'threshold', 2)}.")
            else:
                d.add(f"- Stopping rule: not fired at step {step}: the rise over the last 200 steps is {d.num(TBS, rk, 'value', 4)}, at or above the "
                      f"threshold {d.num(TBS, rk, 'threshold', 2)}.")
        pins = d.tab("trackb_pinned_checkpoints_dev.csv")
        for k, r in pins.items():
            d.add(f"- Pinned {r['arm']} checkpoint (DEV-TRAIN-PIN): final step {int(float(r['final_step']))}, run `{r['run_id']}`, SHA-256 `{r['sha256']}`.")
        if g2s:
            d.add(f"- G2: {g2s} (decided `{(g2rec or {}).get('decided_at', '')}`); the criteria and observed values are listed under Gates. "
                  f"The own-origin mass fraction threshold is {d.design('g2_own_origin_threshold', 2)}, judged on fresh dev seeds.")
        else:
            d.add(f"- G2 requires the own-origin mass fraction to reach {d.design('g2_own_origin_threshold', 2)} on fresh dev seeds after the "
                  "stopping rule fires; no G2 record exists yet.")
    else:
        d.add("No learning curve was given.")
    d.add("- DEV-TRAIN-PIN: the V3 and B1 checkpoints are pinned after the tag (the training to the stopping rule ran after it); the "
          "configuration, the stopping rule and the G2 design were fixed before it.", "")

    d.add("## Track C status", "")
    for k, r in d.tab(TC).items():
        d.add(f"- **{k}**: {r['status']} ({defang(r['note'])}).")
    d.add("")

    d.add("## Framing implied by the results", "",
          "- H5: the agent-level contrast is read together with the distortion ratio; only the T4 agent cell is funded, so no statement about "
          "thread depth beyond T4 follows.",
          ("- Evidence-backed: the Track A contrasts above, the paging identity and precision measurements" if final else
           "- Evidence-backed so far: the Track A contrasts above (interim verdicts), the paging identity and precision measurements")
          + (" and, where listed, the H4 / H6 test results." if any(k.startswith(("H4:", "H6:")) for k in vt) else ".")
          + (" G2 failed, so the model-level arm of H5, H11, H12, H13 and H9 were not run and stay theory." if g2s == "failed"
             else " Still theory: everything that needs trained Track B models beyond G2, or Track C."),
          "- Claims to soften: anything about hyperbolic distance at model level, memory at equal accuracy, and copy-vs-recompute"
          + (", for which no run exists." if g2s == "failed" else ", until G2 and the Track B runs report."), "")
    d.add("## Surprises", "",
          "- H3's R2 leaf cannot be tested at 32K; the S11 vs S8 leg is the only testable part of H3.",
          "- U8's auxiliary SDPA check exceeds its 1e-5 absolute tolerance while the pre-registered identity is met (DEV-U8).", "")
    d.add("## Limitations observed", "",
          "- Model scale (4B), context 32K, synthetic data, one funded cell for each of H1 and H5, cost-driven reductions listed in the "
          "registration (Section 2b). Dev-pilot numbers are never test results; they appear only in files and figure markers labelled dev.", "")
    d.add("## Reproduction", "",
          *(["- Final bundle: `python -m opa.analysis.final_bundle --runs jobs/interim/series_test.yaml --trackb jobs/interim/trackb_test_runs.yaml "
             "--src bundle --out <new folder>` (reads the interim tables, writes claims, tests, deviations, PAPER_INPUT and the manifest, then runs "
             "the verifier on a scratch copy); upload with `python -m opa.bundle upload --final`.",
             "- Interim tables behind it: `python -m opa.analysis.interim --split test --runs <series yaml> --out bundle`."] if final else
            [f"- Interim package: `python -m opa.analysis.interim --split {split} --runs <series yaml> --out bundle`; archive: "
             "`python -m opa.analysis.release`."]),
          "- Re-run a primary result: `python -m opa.launch run jobs/<job>.yaml` with the recorded seed and git SHA.",
          "- Data and run logs: the bucket in config/project.yaml under opa/ (append-only).")
    name = "PAPER_INPUT.md" if final else f"PAPER_INPUT_interim{s}.md"
    if not final:
        itm.check_name(split, name)
    out = bundle / name
    out.write_text(d.text(), encoding="utf-8")
    return out


def _g0_flip(d: Doc) -> str:
    p = d.b / "gates" / "G0.json"
    if not p.exists():
        return "n/a"
    g = json.loads(p.read_text(encoding="utf-8"))
    for i, c in enumerate(g.get("criteria", [])):
        if "attempt 2" in c.get("name", "") and isinstance(c.get("observed"), dict) and "per_item_flip_rate" in c["observed"]:
            return d.jref("gates/G0.json", f"criteria.{i}.observed.per_item_flip_rate", 0, pct=True)
    return "n/a"


def main(argv=None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="check the [[ref:]] tags of a PAPER_INPUT_interim file")
    ap.add_argument("md")
    ap.add_argument("--bundle", default="bundle")
    a = ap.parse_args(argv)
    probs = check_refs(a.md, a.bundle)
    for p in probs:
        print(p)
    print("refs OK" if not probs else f"{len(probs)} problem(s)")
    return 1 if probs else 0


if __name__ == "__main__":
    raise SystemExit(main())
