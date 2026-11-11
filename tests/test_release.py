"""Release archive builder (opa.analysis.release): file selection from git, secret scan, deterministic zip."""
from __future__ import annotations

import hashlib
import subprocess
import zipfile
from pathlib import Path

import pytest

from opa.analysis import release as rel

SECRET_PK = "-----BEGIN " + "PRIVATE KEY-----"
SECRET_HF = "hf_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
SECRET_AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
SECRET_GOOGLE = "ya29" + ".a0AfH6SMBxxxxxxxxxxxxxxxxxxxxxxxx"
SECRET_JSON = '{"' + "private" + '_key": "x"}'


def sh(cwd, *a):
    subprocess.run(a, cwd=cwd, check=True, capture_output=True)


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    sh(repo, "git", "init", "-q")
    files = {
        "opa/a.py": "x = 1\n", "jobs/j.yaml": "name: j\n", "tools/t.py": "print(1)\n", "config/project.yaml": 'gcp:\n  project_id: "my-proj-123456"\n  bucket: "gs://my-secret-bucket"   # note\n  region: "us-central1"\n  sa: "runner@my-proj-123456.iam.gserviceaccount.com"\nmodal:\n  app_name: "x"\n',
        "prereg/rules.yaml": "version: 2\n", "prereg/deviations/DEV-X.md": "# DEV-X\n", "PLAN_V3.md": "plan\n", "README.md": "r\n",
        "requirements.txt": "numpy\n", "bundle/tables/t.csv": "key,value\na,1\n", "bundle/figures/f.json": "{}\n",
        "bundle/PAPER_INPUT_interim.md": "# p\n", "bundle/tables/interim_verdicts_dev.csv": "key\n", "bundle/tables/v3_learning_curve_dev.csv": "key\n",
        "state/decisions.md": "internal\n", "runs/r/result.json": "{}\n", "tests/test_x.py": "k = '" + SECRET_PK + "'\n",
        "credentials.md": "hello\n", "opa/.env": "A=1\n", "opa/my_token.txt": "nothing\n", "config/svc.key.json": "{}\n",
        "ignored.txt": "x\n", ".gitignore": "ignored.txt\nbundle/figures/*.tmp\n", "bundle/figures/skip.tmp": "x\n",
    }
    for p, t in files.items():
        f = repo / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(t, encoding="utf-8")
    (repo / "bundle/figures/f.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)))
    return repo


def test_select_uses_git_and_drops_secrets_and_dev_dryrun(tmp_path):
    repo = make_repo(tmp_path)
    files = rel.git_files(repo)
    assert "ignored.txt" not in files and "bundle/figures/skip.tmp" not in files        # gitignored: never listed
    kept, dropped = rel.select(files)
    for p in ("opa/a.py", "jobs/j.yaml", "tools/t.py", "config/project.yaml", "prereg/rules.yaml", "prereg/deviations/DEV-X.md", "PLAN_V3.md",
              "bundle/tables/t.csv", "bundle/figures/f.png", "bundle/PAPER_INPUT_interim.md", "bundle/tables/v3_learning_curve_dev.csv"):
        assert p in kept, p
    for p in ("credentials.md", "state/decisions.md", "runs/r/result.json", "tests/test_x.py", "ignored.txt"):
        assert p not in kept
    reasons = dict(dropped)
    assert "opa/.env" in reasons and "opa/my_token.txt" in reasons and "config/svc.key.json" in reasons
    assert reasons["bundle/tables/interim_verdicts_dev.csv"] == "dev dry-run output"
    kept2, _ = rel.select(files, include_dev_dryrun=True)
    assert "bundle/tables/interim_verdicts_dev.csv" in kept2


@pytest.mark.parametrize("secret,rule", [(SECRET_PK, "private-key block"), (SECRET_JSON, "private-key JSON field"), (SECRET_HF, "Hugging Face token"),
                                         (SECRET_AWS, "AWS access key"), (SECRET_GOOGLE, "Google OAuth token"),
                                         ("AIza" + "Sy" + "A" * 33, "Google API key"), ("ghp_" + "a" * 36, "GitHub token"),
                                         ('api_key = "' + "k" * 24 + '"', "quoted key/token/password assignment")])
def test_scan_detects_hard_patterns_and_masks_the_secret(secret, rule):
    f = rel.scan_bytes("opa/x.py", ("line\nvalue = '" + secret + "'\n").encode())
    assert any(x["rule"] == rule for x in f), f
    assert all(secret not in x["excerpt"] for x in f)
    report = rel.scan_report(1, f)
    assert secret not in report and "BLOCKED" in report


def test_scan_flags_high_entropy_but_not_hashes_run_ids_or_paths():
    rnd = "Zq3Xk9Lm2Vb7Nc4Rt8Yw1Ps6Gh5Jd0Fa3Ue9Ki2Ol7"                       # 44 chars, mixed case, digits, high entropy
    assert any("high-entropy" in x["rule"] for x in rel.scan_bytes("a.txt", f"token_ish {rnd}\n".encode()))
    sha = hashlib.sha256(b"x").hexdigest()
    run = "A-S1+S2+S4+S8+S11+S18-lbmhqa-late-h2-32k-a0-dev-s0-20261107154535-a0c95f6-4168"
    path = "opa/analysis/interim_figures_with_a_long_name/and/more_parts/here.py"
    assert rel.scan_bytes("a.csv", f"{sha},{run},{path}\n".encode()) == []


def test_scan_binary_files_only_hard_patterns():
    clean = b"\x89PNG" + bytes(range(256)) * 4
    assert rel.scan_bytes("f.png", clean) == []
    assert rel.scan_bytes("f.png", clean + SECRET_PK.encode())


def test_build_release_writes_deterministic_zip_with_contents_and_citation(tmp_path):
    repo = make_repo(tmp_path)
    out = tmp_path / "out"
    r1 = rel.build_release(repo, out, date="2026-11-08")
    assert r1["findings"] == [] and r1["sha256"] and (out / rel.ZIP_NAME).is_file()
    r2 = rel.build_release(repo, tmp_path / "out2", date="2026-11-08")
    assert r1["sha256"] == r2["sha256"]                                        # deterministic
    with zipfile.ZipFile(out / rel.ZIP_NAME) as z:
        names = set(z.namelist())
        cff = z.read("CITATION.cff").decode()
    assert {"opa/a.py", "prereg/rules.yaml", "bundle/tables/t.csv", "bundle/PAPER_INPUT_interim.md", "CITATION.cff"} <= names
    assert not names & {"credentials.md", "ignored.txt", "opa/.env", "tests/test_x.py", "state/decisions.md"}
    assert 'family-names: "Rajgor"' in cff and 'affiliation: "Independent Researcher"' in cff
    assert "https://orcid.org/0000-0000-0000-0000" in cff and "@" not in cff and "mail" not in cff.lower()
    contents = (out / "opa_v3_interim_contents.txt").read_text()
    assert hashlib.sha256((repo / "opa" / "a.py").read_bytes()).hexdigest() in contents and "opa/a.py" in contents and "credentials.md" not in contents.split("\n\n", 1)[1]
    assert "CLEAN" in (out / "secret_scan.txt").read_text()
    assert (out / f"{rel.ZIP_NAME}.sha256").read_text().startswith(r1["sha256"])


def test_build_release_refuses_when_the_scan_finds_something(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "opa" / "leak.py").write_text(f"T = '{SECRET_HF}'\n")
    out = tmp_path / "out"
    with pytest.raises(rel.ReleaseError):
        rel.build_release(repo, out, date="2026-11-08")
    assert not (out / rel.ZIP_NAME).exists() and not (out / "opa_v3_interim_contents.txt").exists()
    rep = (out / "secret_scan.txt").read_text()
    assert "opa/leak.py:1" in rep and SECRET_HF not in rep and "BLOCKED" in rep
    # a secret in a file that is dropped by name or gitignored never reaches the scan
    (repo / "opa" / "leak.py").unlink()
    (repo / "credentials.md").write_text(SECRET_HF)
    (repo / "ignored.txt").write_text(SECRET_PK)
    assert rel.build_release(repo, out, date="2026-11-08")["findings"] == []


def test_dry_run_writes_nothing_and_reports_missing(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "bundle" / "PAPER_INPUT_interim.md").unlink()
    r = rel.build_release(repo, tmp_path / "o", dry_run=True)
    assert not (tmp_path / "o").exists() and "bundle/PAPER_INPUT_interim*.md" in r["missing"]


def test_this_repository_scans_clean_in_a_dry_run():
    r = rel.build_release(dry_run=True)
    assert r["findings"] == [], r["findings"]


def test_config_is_redacted_in_the_archive_only_and_listed(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "jobs" / "j.yaml").write_text("uri: gs://my-secret-bucket/opa/x\n")           # an identifier elsewhere: reported, not blocking
    out = tmp_path / "out"
    r = rel.build_release(repo, out, date="2026-11-08")
    with zipfile.ZipFile(out / rel.ZIP_NAME) as z:
        cfg = z.read("config/project.yaml").decode()
    for secret in ("my-proj-123456", "my-secret-bucket", "iam.gserviceaccount.com"):
        assert secret not in cfg
    assert cfg.count("<redacted>") == 3 and "region" in cfg and "# note" in cfg
    assert "my-secret-bucket" in (repo / "config" / "project.yaml").read_text()                  # repository file unchanged
    contents = (out / "opa_v3_interim_contents.txt").read_text()
    assert "redacted in the archive copy" in contents and "config/project.yaml: project_id, bucket" in contents
    assert hashlib.sha256(cfg.encode()).hexdigest() in contents                                   # hash of the redacted copy
    with zipfile.ZipFile(out / rel.ZIP_NAME) as z:
        j = z.read("jobs/j.yaml").decode()
    assert "my-secret-bucket" not in j and "gs://<redacted>/opa/x" in j and r["residual"] == []
    assert "jobs/j.yaml: project id / bucket name replaced" in contents
    assert hashlib.sha256(j.encode()).hexdigest() in contents                                     # redacted sha listed
    assert "my-secret-bucket" in (repo / "jobs" / "j.yaml").read_text()                          # repository file unchanged
    # an identifier that cannot be redacted (inside a binary file) blocks the zip
    (repo / "bundle" / "figures" / "f.png").write_bytes(b"\x89PNG" + b"my-secret-bucket" + bytes(range(64)))
    with pytest.raises(rel.ReleaseError):
        rel.build_release(repo, tmp_path / "out3", date="2026-11-08")
    assert "still present" in (tmp_path / "out3" / "secret_scan.txt").read_text()
    assert not (tmp_path / "out3" / rel.ZIP_NAME).exists()
    t, fields, vals = rel.redact_config("a: 1\n")
    assert fields == [] and t == "a: 1\n"


def test_missing_paper_input_blocks_a_real_build_unless_allowed(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "bundle" / "PAPER_INPUT_interim.md").unlink()
    out = tmp_path / "out"
    with pytest.raises(rel.ReleaseError, match="PAPER_INPUT"):
        rel.build_release(repo, out, date="2026-11-08")
    assert not (out / rel.ZIP_NAME).exists()
    assert rel.build_release(repo, out, date="2026-11-08", allow_missing_paper_input=True)["sha256"]


def test_dry_run_cli_prints_kept_and_dropped(tmp_path, capsys):
    repo = make_repo(tmp_path)
    assert rel.main(["--repo", str(repo), "--dry-run"]) == 0
    o = capsys.readouterr().out
    assert "kept (" in o and "opa/a.py" in o and "dropped (" in o and "credentials" not in o.split("kept (")[1].split("dropped (")[0]
    assert "redactions:" in o and "config/project.yaml" in o


def test_citation_has_orcid_placeholder_comment():
    c = rel.citation_cff("2026-11-08")
    assert "# ORCID placeholder: replace before publishing" in c and "0000-0000-0000-0000" in c
