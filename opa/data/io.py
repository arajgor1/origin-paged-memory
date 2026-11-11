"""Item files: gzip JSON Lines (one item per line) and manifest.json (SCHEMA.md)."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable


def _dumps(item: dict) -> str:
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"))


def write_items(items: Iterable[dict], path: str | Path) -> dict:
    """Write items as gzip JSON Lines. Byte-deterministic (gzip mtime 0, no file name in the header).
    Returns {"path", "n_items", "bytes", "sha256"}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    n = 0
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=6) as gz:
        for it in items:
            gz.write((_dumps(it) + "\n").encode("utf-8"))
            n += 1
    data = buf.getvalue()
    path.write_bytes(data)
    return {"path": str(path), "n_items": n, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _read_bytes(path_or_uri: str | Path) -> bytes:
    s = str(path_or_uri)
    if s.startswith("gs://"):
        from opa import gcs  # needs google-cloud-storage and credentials
        return gcs.get_bytes(s)
    return Path(s).read_bytes()


def read_items(path_or_uri: str | Path) -> list[dict]:
    """Read a local items.jsonl.gz, or a gs:// URI through opa.gcs.get_bytes."""
    raw = gzip.decompress(_read_bytes(path_or_uri)).decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def write_manifest(directory: str | Path, manifest: dict) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return p


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
