"""Google Cloud Storage helpers. Writes are create-only (if_generation_match=0): an existing object is never overwritten.

Credentials: inside Modal, the `opa-gcp` secret provides GCP_SA_JSON (service-account key JSON).
Locally, Application Default Credentials are used (`gcloud auth application-default login`).
The bucket name comes from config/project.yaml or the OPA_BUCKET environment variable.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import yaml

from . import PROJECT_DIR


def normalize_bucket(name: str) -> str:
    """Accept "name", "gs://name" and "gs://name/": return the bare bucket name."""
    name = str(name).strip()
    if name.startswith("gs://"):
        name = name[len("gs://"):]
    return name.rstrip("/")


def bucket_name() -> str:
    if os.environ.get("OPA_BUCKET"):
        return normalize_bucket(os.environ["OPA_BUCKET"])
    cfg = yaml.safe_load((PROJECT_DIR / "config" / "project.yaml").read_text())
    return normalize_bucket(cfg["gcp"]["bucket"])


def client():
    from google.cloud import storage
    sa = os.environ.get("GCP_SA_JSON")
    if sa:
        from google.oauth2 import service_account
        info = json.loads(sa)
        creds = service_account.Credentials.from_service_account_info(info)
        return storage.Client(project=info.get("project_id"), credentials=creds)
    return storage.Client()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def uri(path: str) -> str:
    return f"gs://{bucket_name()}/{path.lstrip('/')}"


def _split(gs_uri: str) -> tuple[str, str]:
    assert gs_uri.startswith("gs://"), gs_uri
    b, _, p = gs_uri[5:].partition("/")
    return b, p


def _crc32c_b64(data: bytes) -> str:
    import base64
    import google_crc32c
    return base64.b64encode(google_crc32c.Checksum(data).digest()).decode()


def _md5_b64(data: bytes) -> str:
    import base64
    return base64.b64encode(hashlib.md5(data).digest()).decode()


def _file_bytes_digests(path: str | Path) -> tuple[str, str]:
    import base64
    import google_crc32c
    c = google_crc32c.Checksum()
    m = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            c.update(chunk)
            m.update(chunk)
    return base64.b64encode(c.digest()).decode(), base64.b64encode(m.digest()).decode()


def _confirm_identical(blob, gs_uri: str, crc: str, md5: str) -> None:
    """After a 412 on a create-only upload: succeed only if the existing object is byte-identical
    (crc32c and md5 both must be present and match). Never overwrites. Raises on conflict/unknown."""
    try:
        blob.reload()
    except Exception as e:  # metadata unavailable: cannot prove identity
        raise RuntimeError(f"GCS 412 on {gs_uri} and existing object metadata unavailable ({type(e).__name__}); not overwriting") from e
    rc, rm = getattr(blob, "crc32c", None), getattr(blob, "md5_hash", None)
    if not rc or not rm:
        raise RuntimeError(f"GCS 412 on {gs_uri}: existing object lacks crc32c or md5 (composite?); cannot prove identity; not overwriting")
    if rc != crc or rm != md5:
        raise RuntimeError(f"GCS conflict on {gs_uri}: object already exists with different content; not overwriting")


def put_bytes(gs_uri: str, data: bytes, content_type: str = "application/octet-stream") -> dict:
    from google.api_core.exceptions import PreconditionFailed
    b, p = _split(gs_uri)
    blob = client().bucket(b).blob(p)
    try:
        blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
    except PreconditionFailed:
        # A library-internal retry may re-send an upload whose first attempt succeeded.
        _confirm_identical(blob, gs_uri, _crc32c_b64(data), _md5_b64(data))
    # The returned sha256 is computed locally; identity with any pre-existing object was proven via crc32c + md5.
    return {"uri": gs_uri, "sha256": sha256_bytes(data), "bytes": len(data)}


def put_file(local: str | Path, gs_uri: str) -> dict:
    from google.api_core.exceptions import PreconditionFailed
    b, p = _split(gs_uri)
    blob = client().bucket(b).blob(p)
    try:
        blob.upload_from_filename(str(local), if_generation_match=0)
    except PreconditionFailed:
        crc, md5 = _file_bytes_digests(local)
        _confirm_identical(blob, gs_uri, crc, md5)
    return {"uri": gs_uri, "sha256": sha256_file(local), "bytes": Path(local).stat().st_size}


def put_json(gs_uri: str, obj) -> dict:
    return put_bytes(gs_uri, json.dumps(obj, indent=2, default=str).encode(), "application/json")


def put_jsonl_gz(gs_uri: str, rows) -> dict:
    raw = "\n".join(json.dumps(r, default=str) for r in rows).encode()
    return put_bytes(gs_uri, gzip.compress(raw), "application/gzip")


def get_bytes(gs_uri: str) -> bytes:
    b, p = _split(gs_uri)
    return client().bucket(b).blob(p).download_as_bytes()


def get_file(gs_uri: str, local: str | Path) -> Path:
    b, p = _split(gs_uri)
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    client().bucket(b).blob(p).download_to_filename(str(local))
    return Path(local)


def ls(prefix_uri: str) -> list[str]:
    b, p = _split(prefix_uri)
    return [f"gs://{b}/{x.name}" for x in client().list_blobs(b, prefix=p)]
