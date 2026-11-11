"""Upload built dataset cells to GCS (create-only; nothing is ever overwritten or deleted).

  python -m opa.data.upload --root ../opa_data --dataset all --split dev --version v2 [--dry-run]

Each cell's items.jsonl.gz and manifest.json go to gs://<bucket>/opa/data/<dataset>/<split>/<version>/<cell>/ with
opa.gcs.put_file (if_generation_match=0). If the object already exists with the same SHA-256 it is skipped; if it
exists with a different SHA-256 the run stops with UploadConflict (no later object is written). items.jsonl.gz is
written before manifest.json, so a manifest in GCS implies complete items.

Credentials: env GCP_SA_JSON (service-account key JSON) or env OPA_GCP_KEY_FILE (path to the key file; its contents
are read into GCP_SA_JSON of this process only and never printed). Otherwise Application Default Credentials.
Test-split data is refused before the prereg-v2 tag. --dry-run lists what would happen without contacting GCS.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from opa.data import SplitRefused
from opa.data.io import sha256_file

DATASETS = ("lbmhqa", "mtlbmhqa", "thrash", "drift")
FILES = ("items.jsonl.gz", "manifest.json")          # items first: a manifest in GCS implies complete items


class UploadConflict(RuntimeError):
    """An object already exists in GCS with different content."""


def load_credentials(env: dict | None = None) -> str:
    """Make GCP_SA_JSON available from OPA_GCP_KEY_FILE if needed. Returns "env", "key-file" or "adc"; never the key."""
    env = os.environ if env is None else env
    if env.get("GCP_SA_JSON"):
        return "env"
    path = env.get("OPA_GCP_KEY_FILE")
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"OPA_GCP_KEY_FILE points to a missing file: {path}")
        env["GCP_SA_JSON"] = p.read_text(encoding="utf-8")
        return "key-file"
    return "adc"


def find_cells(root: Path, datasets, split: str, version: str) -> list[tuple[str, str, Path]]:
    out = []
    for ds in datasets:
        base = root / ds / split / version
        if not base.is_dir():
            continue
        for d in sorted(p for p in base.iterdir() if (p / "items.jsonl.gz").is_file()):
            if not (d / "manifest.json").is_file():
                raise FileNotFoundError(f"{d} has items.jsonl.gz but no manifest.json")
            out.append((ds, d.name, d))
    return out


def remote_sha(gcs_mod, uri: str) -> str | None:
    """SHA-256 of the object at `uri`, or None if it does not exist."""
    if uri not in gcs_mod.ls(uri):
        return None
    import hashlib
    return hashlib.sha256(gcs_mod.get_bytes(uri)).hexdigest()


def upload(root, datasets=DATASETS, split: str = "dev", version: str = "v2", dry_run: bool = False, gcs_mod=None,
           bucket: str | None = None, prereg_tagged=None, log=print) -> dict:
    """Returns {"uploaded": [...], "skipped": [...], "planned": [...]} (gs:// URIs)."""
    if split == "test":
        if prereg_tagged is None:
            from opa import launch
            prereg_tagged = launch.prereg_tagged
        if not prereg_tagged():
            raise SplitRefused("test-split data may only be uploaded after the prereg-v2 tag (CLAUDE.md Section 6)")
    root = Path(root)
    cells = find_cells(root, datasets, split, version)
    if not cells:
        raise FileNotFoundError(f"no cells under {root} for {list(datasets)} / {split} / {version}")
    for ds, cell, d in cells:                              # local sanity: items file matches its manifest
        import json
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if m.get("items_sha256") != sha256_file(d / "items.jsonl.gz"):
            raise ValueError(f"{d}: items.jsonl.gz does not match manifest items_sha256")
        if m.get("split") != split:
            raise ValueError(f"{d}: manifest split {m.get('split')!r} != {split!r}")
    res = {"uploaded": [], "skipped": [], "planned": []}
    if dry_run:
        bucket = bucket or "<bucket>"
        for ds, cell, d in cells:
            for f in FILES:
                uri = f"gs://{bucket}/opa/data/{ds}/{split}/{version}/{cell}/{f}"
                res["planned"].append(uri)
                log(f"[dry-run] would upload {d / f} -> {uri}")
        return res
    if gcs_mod is None:
        load_credentials()
        from opa import gcs as gcs_mod
    if bucket is None:
        bucket = gcs_mod.bucket_name()
    if bucket.upper().startswith("YOUR-") or not bucket:
        raise RuntimeError(f"bucket {bucket!r} in config/project.yaml is still a placeholder")
    for ds, cell, d in cells:
        for f in FILES:
            local = d / f
            uri = f"gs://{bucket}/opa/data/{ds}/{split}/{version}/{cell}/{f}"
            want = sha256_file(local)
            have = remote_sha(gcs_mod, uri)
            if have is not None:
                if have != want:
                    raise UploadConflict(f"{uri} exists with sha256 {have[:12]}, local {local} is {want[:12]}; "
                                         "objects are never overwritten")
                res["skipped"].append(uri)
                log(f"skip (same sha256) {uri}")
                continue
            try:
                gcs_mod.put_file(local, uri)
            except Exception as e:                          # lost a create-only race: re-check content
                have = remote_sha(gcs_mod, uri)
                if have == want:
                    res["skipped"].append(uri)
                    continue
                if have is not None:
                    raise UploadConflict(f"{uri} appeared with different content during upload") from e
                raise
            res["uploaded"].append(uri)
            log(f"uploaded {uri}")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m opa.data.upload", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="default: paths.local_data from config/project.yaml")
    ap.add_argument("--dataset", default="all", choices=("all",) + DATASETS)
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--version", default="v2")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.root is None:
        from opa.data.filler import local_data_dir
        a.root = local_data_dir()
    ds = DATASETS if a.dataset == "all" else (a.dataset,)
    r = upload(a.root, ds, a.split, a.version, a.dry_run)
    print(f"uploaded {len(r['uploaded'])}, skipped {len(r['skipped'])}, planned {len(r['planned'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
