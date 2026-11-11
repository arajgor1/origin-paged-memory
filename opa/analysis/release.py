"""Release archive of the interim package: bundle/release/opa_v3_interim.zip.

  python -m opa.analysis.release [--repo .] [--out bundle/release] [--date YYYY-MM-DD] [--dry-run] [--include-dev-dryrun]

Contents: code (opa/, jobs/, tools/), configs (config/), the pre-registration folder (prereg/ incl. deviations and the tag
record), the plan of record (PLAN_V3.md), README.md, requirements.txt, and from the bundle the tables, figures (PDF + PNG +
sidecar + source CSV), the figure script (code/), the gate records and PAPER_INPUT_interim*.md, plus CITATION.cff. The file
list is built from ``git ls-files --cached --others --exclude-standard`` (so anything gitignored is never listed), then
files are dropped by name (credentials.md, .env*, *token*, key files, service-account JSON, *.pem) and dev dry-run
outputs (bundle/tables/interim_*_dev.csv, fig_h1_*_dev.*, fig_h5_*_dev.*, PAPER_INPUT_interim_dev.md; real dev-data tables such
as the V3 learning curve and the pilot tables stay) unless --include-dev-dryrun. The archive copy of config/project.yaml has the GCP
project id, the bucket name, the key-file path and any service-account email replaced by "<redacted>" (the repository file is
unchanged; the contents file lists the redaction). A real build refuses when no PAPER_INPUT_interim*.md is in the file list
unless --allow-missing-paper-input; a dry run prints the kept and dropped lists.

A secret scan runs over every file that would go into the zip. Hard patterns (private-key blocks and JSON key fields, AWS / GCP
/ Hugging Face / GitHub / Slack / Modal tokens, OAuth tokens, quoted key-or-token assignments) apply to text and binary files;
a Shannon-entropy heuristic (40+ characters of base64 alphabet, entropy >= 4.6) flags high-entropy strings in text files for
review. ANY finding blocks the zip: nothing is written except secret_scan.txt (masked excerpts only). Otherwise the zip is
written (deterministic: sorted, fixed timestamps), with opa_v3_interim_contents.txt (path, size, sha256 of every file) and
opa_v3_interim.zip.sha256.
Read-only on the repo; spends nothing; never touches GCS.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

from .. import PROJECT_DIR

ZIP_NAME = "opa_v3_interim.zip"
INCLUDE_PREFIXES = ("opa/", "jobs/", "tools/", "config/", "prereg/", "bundle/tables/", "bundle/figures/", "bundle/code/",
                    "bundle/gates/", "bundle/PAPER_INPUT_interim", "bundle/deviations")
INCLUDE_FILES = ("PLAN_V3.md", "README.md", "requirements.txt")
EXCLUDE_NAME_PATTERNS = [r"(^|/)credentials\.md$", r"(^|/)\.env($|\.)", r"token", r"\.key\.json$", r"(^|/)opa-runner[^/]*\.json$",
                         r"\.pem$", r"\.p12$", r"service[-_]?account[^/]*\.json$", r"(^|/)id_(rsa|ed25519)"]
DEV_DRYRUN = [r"^bundle/tables/interim_[a-z_0-9]*_dev\.csv$", r"^bundle/tables/p2_(contrasts|series|exclusions)_dev\.csv$",
              r"^bundle/tables/fig_(h1|h5)_[a-z_]*_dev\.csv$", r"^bundle/figures/fig_(h1|h5)_[a-z_]*_dev\.", r"^bundle/PAPER_INPUT_interim_dev\.md$"]
BINARY_SUFFIXES = {".png", ".pdf", ".zip", ".gz", ".npz", ".pt", ".safetensors", ".jpg", ".jpeg"}
ENTROPY_MIN_LEN, ENTROPY_MIN = 40, 4.6
_PK = "private" + "_key"                       # written in two pieces so this file does not match its own scan

HARD_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private-key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("private-key JSON field", re.compile(r'"' + _PK + r'"')),
    ("AWS access key", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google OAuth token", re.compile(r"ya29\.[0-9A-Za-z_-]{20,}")),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_-]{30,}")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("Modal token", re.compile(r"\b(ak|as)-[A-Za-z0-9]{20,}\b")),
    ("API-style secret key", re.compile(r"\bsk-[A-Za-z0-9_-]{32,}")),
]
TEXT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("quoted key/token/password assignment", re.compile(
        r"(?i)\b(api[_-]?key|secret|access[_-]?token|auth[_-]?token|password|passwd)\b\s*[:=]\s*['\"][A-Za-z0-9+/_=.-]{16,}['\"]")),
    ("bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{30,}")),
]
_TOKEN = re.compile(r"[A-Za-z0-9+/_=]{%d,}" % ENTROPY_MIN_LEN)


class ReleaseError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- file list
def git_files(repo: Path) -> list[str]:
    """Tracked plus untracked-but-not-ignored files (POSIX paths), existing on disk."""
    r = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=repo, capture_output=True)
    if r.returncode != 0:
        raise ReleaseError(f"git ls-files failed: {r.stderr.decode(errors='replace')[:300]}")
    out = [p for p in r.stdout.decode("utf-8", errors="replace").split("\0") if p]
    return sorted({p for p in out if (repo / p).is_file()})


def select(files: Iterable[str], include_dev_dryrun: bool = False) -> tuple[list[str], list[tuple[str, str]]]:
    """(kept, [(path, reason) for files that were in an included area but dropped])."""
    kept, dropped = [], []
    for p in files:
        if not (p.startswith(INCLUDE_PREFIXES) or p in INCLUDE_FILES):
            continue
        low = p.lower()
        why = next((f"name matches {pat}" for pat in EXCLUDE_NAME_PATTERNS if re.search(pat, low)), None)
        if why is None and not include_dev_dryrun and any(re.search(pat, p) for pat in DEV_DRYRUN):
            why = "dev dry-run output"
        if why:
            dropped.append((p, why))
        else:
            kept.append(p)
    return kept, dropped


# ---------------------------------------------------------------------------------------------- secret scan
def shannon(s: str) -> float:
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in (s.count(ch) for ch in set(s)))


def _mask(s: str) -> str:
    s = s.strip()
    return (s[:4] + "..." + f"({len(s)} chars)") if len(s) > 8 else "***"


def is_binary(path: str, data: bytes) -> bool:
    return Path(path).suffix.lower() in BINARY_SUFFIXES or b"\0" in data[:4096]


def scan_bytes(path: str, data: bytes) -> list[dict]:
    """Findings for one file: {"file", "line", "rule", "excerpt" (masked)}."""
    findings = []
    text = data.decode("latin-1" if is_binary(path, data) else "utf-8", errors="replace")
    binary = is_binary(path, data)
    pats = HARD_PATTERNS + ([] if binary else TEXT_PATTERNS)
    for ln, line in enumerate(text.splitlines() if not binary else [text], 1):
        for rule, pat in pats:
            for m in pat.finditer(line):
                findings.append({"file": path, "line": ln if not binary else 0, "rule": rule, "excerpt": _mask(m.group(0))})
        if not binary:
            for m in _TOKEN.finditer(line):
                tok = m.group(0)
                if tok.count("/") >= 2 or not (re.search(r"\d", tok) and re.search(r"[a-z]", tok) and re.search(r"[A-Z]", tok)):
                    continue
                if shannon(tok) >= ENTROPY_MIN:
                    findings.append({"file": path, "line": ln, "rule": "high-entropy string (review)", "excerpt": _mask(tok)})
    return findings


def scan_files(repo: Path, paths: Sequence[str]) -> list[dict]:
    out = []
    for p in paths:
        out += scan_bytes(p, (repo / p).read_bytes())
    return out


def scan_report(n_files: int, findings: Sequence[dict]) -> str:
    lines = [f"secret scan: {n_files} files scanned, {len(findings)} finding(s)",
             "rules: private-key blocks and JSON key fields, AWS / Google / Hugging Face / GitHub / Slack / Modal tokens, API-style "
             "keys, quoted key/token/password assignments, bearer tokens, high-entropy strings (40+ chars, entropy >= 4.6)",
             "excerpts are masked (first 4 characters and the length only)", ""]
    for f in findings:
        lines.append(f"{f['file']}:{f['line']}: {f['rule']}: {f['excerpt']}")
    lines.append("RESULT: " + ("CLEAN" if not findings else "BLOCKED: the zip was not written"))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------- citation + zip
def citation_cff(date: str, g2: str | None = None, trackb: bool = False) -> str:
    return "\n".join([
        "cff-version: 1.2.0",
        'message: "If you use this software, data or analysis, please cite it as below."',
        'title: "Origin-Paged Attention experiments, interim Track A package (plan version 3)"',
        "type: software",
        "authors:",
        '  - family-names: "Rajgor"',
        '    given-names: "Aaditya"',
        '    affiliation: "Independent Researcher"',
        "    # ORCID placeholder: replace before publishing",
        '    orcid: "https://orcid.org/0000-0000-0000-0000"',
        'version: "interim"',
        f'date-released: "{date}"',
        'abstract: "Code, job specifications, pre-registration, result tables, figures and the interim PAPER_INPUT for the '
        'Track A (agent-level) experiments of the Origin-Paged Attention study'
        + ((", with the Track B status" + ("; H4 / H6 test results included" if trackb else "") + (f"; G2 {g2}" if g2 else "")
            + ("; model-level arms H11 to H13 and H9 not run; Track C not run" if g2 == "failed" else ""))
           if (g2 or trackb) else ", with the Track B learning curve so far")
        + '. Verdicts are interim: the final Holm correction is pending."',
        "keywords:",
        '  - "origin-paged attention"',
        '  - "long-context agents"',
        '  - "pre-registered experiments"',
        "",
    ])


REDACT_FILES = ("config/project.yaml",)
_REDACT_KEYS = ("project_id", "bucket", "credentials_file")
_SA_EMAIL = re.compile(r"[A-Za-z0-9._-]+@[A-Za-z0-9.-]*iam\.gserviceaccount\.com")


def redact_config(text: str) -> tuple[str, list[str], list[str]]:
    """(redacted text, names of the redacted fields, the removed values): the GCP project id, the bucket name, the key-file path
    and any service-account email become "<redacted>". The repository file is never changed; only the copy in the archive."""
    fields, values = [], []
    out = []
    for line in text.splitlines():
        m = re.match(r"^(\s*)(project_id|bucket|credentials_file)(\s*:\s*)(\"[^\"]*\"|'[^']*'|[^#\s][^#]*?)(\s*(#.*)?)$", line)
        if m:
            raw = m.group(4).strip().strip("\"'")
            fields.append(m.group(2))
            values.append(raw)
            line = f'{m.group(1)}{m.group(2)}{m.group(3)}"<redacted>"{m.group(5)}'
        out.append(line)
    t = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    emails = _SA_EMAIL.findall(t)
    if emails:
        fields.append("service-account email")
        values += emails
        t = _SA_EMAIL.sub("<redacted>", t)
    return t, fields, values


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_zip(repo: Path, files: Sequence[str], extra: dict[str, bytes], dest: Path) -> str:
    """Deterministic zip (sorted names, fixed timestamps); returns its SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(set(files) | set(extra))
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            data = extra[n] if n in extra else (repo / n).read_bytes()
            zi = zipfile.ZipInfo(n, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data)
    return sha256_bytes(dest.read_bytes())


def build_release(repo: Path | str | None = None, out_dir: Path | str | None = None, date: str | None = None,
                  dry_run: bool = False, include_dev_dryrun: bool = False, allow_missing_paper_input: bool = False) -> dict:
    """Select, scan, write. Returns {"zip", "n_files", "findings", "dropped", "missing", "sha256"}; the zip is not written if
    the scan found anything (and never in a dry run)."""
    repo = Path(repo) if repo else Path(PROJECT_DIR)
    out_dir = Path(out_dir) if out_dir else repo / "bundle" / "release"
    date = date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    kept, dropped = select(git_files(repo), include_dev_dryrun)
    # release outputs never go into the archive
    kept = [p for p in kept if not p.startswith("bundle/release/")]
    g2rec = repo / "bundle" / "gates" / "G2.json"
    g2s = None
    if g2rec.exists():
        import json as _json
        g2s = "passed" if _json.loads(g2rec.read_text(encoding="utf-8")).get("passed") else "failed"
    cff = citation_cff(date, g2s, (repo / "bundle" / "tables" / "trackb" / "trackb_contrasts_test.csv").exists()).encode("utf-8")
    extra = {"CITATION.cff": cff}
    redactions, removed = [], []
    for rf in REDACT_FILES:
        if rf in kept:
            red, fields, values = redact_config((repo / rf).read_text(encoding="utf-8"))
            extra[rf] = red.encode("utf-8")
            redactions.append(f"{rf}: " + (", ".join(fields) if fields else "nothing to redact") + " replaced by <redacted> in the archive copy only")
            removed += values
    # the same identifiers are redacted in every other text file of the archive (archive copies; repository files unchanged)
    idents = sorted({v.replace("gs://", "").strip("/") for v in removed if v and len(v.replace("gs://", "")) >= 6}, key=len, reverse=True)
    for p in kept:
        if p in extra or Path(p).suffix.lower() in BINARY_SUFFIXES:
            continue
        raw = (repo / p).read_bytes()
        if b"\0" in raw[:4096]:
            continue
        txt = raw.decode("utf-8", errors="surrogateescape")
        if any(v in txt for v in idents):
            for v in idents:
                txt = txt.replace(v, "<redacted>")
            extra[p] = txt.encode("utf-8", errors="surrogateescape")
            redactions.append(f"{p}: project id / bucket name replaced by <redacted> in the archive copy only")
    findings = scan_files(repo, [p for p in kept if p not in extra]) + [f for n, b in extra.items() for f in scan_bytes(n, b)]
    residual = []                      # an identifier that is still present anywhere in the archive is BLOCKING
    for p in kept:
        b = extra[p] if p in extra else (repo / p).read_bytes()
        if any(v.encode() in b for v in idents):
            residual.append(p)
            findings.append({"file": p, "line": 0, "rule": "GCP project id or bucket name still present (cannot be redacted)",
                             "excerpt": "<identifier>"})
    missing = []
    if not any(p.startswith("bundle/PAPER_INPUT_interim") for p in kept):
        missing.append("bundle/PAPER_INPUT_interim*.md")
    if not any(p.startswith("bundle/tables/") for p in kept):
        missing.append("bundle/tables/")
    if not any(p.startswith("bundle/figures/") and p.endswith(".png") for p in kept):
        missing.append("bundle/figures/*.png")
    res = {"zip": str(out_dir / ZIP_NAME), "n_files": len(kept) + 1, "findings": findings, "dropped": dropped, "missing": missing,
           "sha256": None, "kept": sorted(kept), "redactions": redactions, "residual": residual}
    if dry_run:
        return res
    if "bundle/PAPER_INPUT_interim*.md" in missing and not allow_missing_paper_input:
        raise ReleaseError("bundle/PAPER_INPUT_interim*.md is missing from the file list (generate and commit it, or pass "
                           "--allow-missing-paper-input); nothing was written")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "secret_scan.txt").write_text(scan_report(len(kept) + 1, findings), encoding="utf-8")
    if findings:
        raise ReleaseError(f"secret scan found {len(findings)} item(s); see {out_dir / 'secret_scan.txt'}; the zip was not written")
    (out_dir / "CITATION.cff").write_bytes(cff)
    sha = write_zip(repo, kept, extra, out_dir / ZIP_NAME)
    rows = []
    for p in sorted(kept):
        b = extra[p] if p in extra else (repo / p).read_bytes()
        rows.append(f"{sha256_bytes(b)}  {len(b):>10}  {p}")
    rows.append(f"{sha256_bytes(cff)}  {len(cff):>10}  CITATION.cff")
    head = [f"{ZIP_NAME}: {len(rows)} files; columns: sha256, bytes, path",
            f"built from git ls-files (tracked + untracked, gitignored files excluded) in {repo.name}; date-released {date}",
            f"dropped by name or as dev dry-run output ({len(dropped)}):"] + [f"  {p}  [{why}]" for p, why in dropped]
    head += ["redacted in the archive copy (repository files unchanged); file, sha256 of the redacted copy:"]
    for r in redactions:
        path = r.split(":", 1)[0]
        head.append(f"  {r}  sha256 {sha256_bytes(extra[path]) if path in extra else 'n/a'}")
    if missing:
        head.append("MISSING from the package: " + ", ".join(missing))
    (out_dir / "opa_v3_interim_contents.txt").write_text("\n".join(head + [""] + sorted(rows, key=lambda r: r.split("  ")[-1])) + "\n",
                                                          encoding="utf-8")
    (out_dir / f"{ZIP_NAME}.sha256").write_text(f"{sha}  {ZIP_NAME}\n", encoding="utf-8")
    res["sha256"] = sha
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--date", default=None, help="date-released in CITATION.cff (default: today, UTC)")
    ap.add_argument("--dry-run", action="store_true", help="select and scan; write nothing")
    ap.add_argument("--include-dev-dryrun", action="store_true")
    ap.add_argument("--allow-missing-paper-input", action="store_true", help="write the zip even if no PAPER_INPUT_interim*.md is in the file list")
    a = ap.parse_args(argv)
    try:
        r = build_release(a.repo, a.out, a.date, a.dry_run, a.include_dev_dryrun, a.allow_missing_paper_input)
    except ReleaseError as e:
        print(f"release FAILED: {e}", file=sys.stderr)
        return 1
    print(f"files: {r['n_files']}  dropped: {len(r['dropped'])}  findings: {len(r['findings'])}  missing: {r['missing'] or 'none'}")
    if a.dry_run:
        sep = "\n  "
        print(f"kept ({len(r['kept'])}):", *r["kept"], sep=sep)
        print(f"dropped ({len(r['dropped'])}):", *[f"{p}  [{why}]" for p, why in r["dropped"]], sep=sep)
        print("redactions:", *r["redactions"], sep=sep)
    if r["findings"]:
        for f in r["findings"]:
            print(f"  {f['file']}:{f['line']}: {f['rule']}: {f['excerpt']}")
        return 1
    print("dry run: nothing written" if a.dry_run else f"wrote {r['zip']} sha256 {r['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
