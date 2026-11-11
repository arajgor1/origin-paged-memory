"""opa.data.upload with a fake gcs module (no network, no credentials)."""
import hashlib
import json

import pytest

from opa.data import SplitRefused
from opa.data.io import write_items, write_manifest
from opa.data.upload import UploadConflict, find_cells, load_credentials, upload


class FakeGcs:
    def __init__(self, bucket="real-bucket"):
        self.store: dict[str, bytes] = {}
        self.puts: list[str] = []
        self._bucket = bucket

    def bucket_name(self):
        return self._bucket

    def ls(self, prefix):
        return [k for k in self.store if k.startswith(prefix)]

    def get_bytes(self, uri):
        return self.store[uri]

    def put_file(self, local, uri):
        if uri in self.store:
            raise RuntimeError("412 precondition failed")
        self.store[uri] = open(local, "rb").read()
        self.puts.append(uri)


def make_root(tmp_path, split="dev", version="v2", cells=(("thrash", "r2"), ("lbmhqa", "late-h3-8k-a0"))):
    for ds, cell in cells:
        d = tmp_path / ds / split / version / cell
        info = write_items([{"item_id": f"{ds}-{cell}", "n": 1}], d / "items.jsonl.gz")
        write_manifest(d, {"dataset": ds, "split": split, "version": version, "cell": cell,
                           "items_sha256": info["sha256"]})
    return tmp_path


def uri(ds, cell, f, bucket="real-bucket", split="dev", version="v2"):
    return f"gs://{bucket}/opa/data/{ds}/{split}/{version}/{cell}/{f}"


def test_uploads_items_then_manifest(tmp_path):
    root, g = make_root(tmp_path), FakeGcs()
    r = upload(root, split="dev", version="v2", gcs_mod=g, log=lambda *_: None)
    assert len(r["uploaded"]) == 4 and not r["skipped"]
    assert g.puts.index(uri("thrash", "r2", "items.jsonl.gz")) < g.puts.index(uri("thrash", "r2", "manifest.json"))
    local = (root / "thrash" / "dev" / "v2" / "r2" / "items.jsonl.gz").read_bytes()
    assert hashlib.sha256(g.store[uri("thrash", "r2", "items.jsonl.gz")]).hexdigest() == hashlib.sha256(local).hexdigest()


def test_rerun_skips_identical(tmp_path):
    root, g = make_root(tmp_path), FakeGcs()
    upload(root, gcs_mod=g, log=lambda *_: None)
    n = len(g.puts)
    r = upload(root, gcs_mod=g, log=lambda *_: None)
    assert len(r["skipped"]) == 4 and not r["uploaded"] and len(g.puts) == n


def test_different_content_fails_loudly_and_writes_nothing_after(tmp_path):
    root, g = make_root(tmp_path), FakeGcs()
    g.store[uri("lbmhqa", "late-h3-8k-a0", "items.jsonl.gz")] = b"other bytes"
    with pytest.raises(UploadConflict):
        upload(root, gcs_mod=g, log=lambda *_: None)
    assert g.store[uri("lbmhqa", "late-h3-8k-a0", "items.jsonl.gz")] == b"other bytes"       # never overwritten
    assert uri("lbmhqa", "late-h3-8k-a0", "manifest.json") not in g.store
    assert uri("thrash", "r2", "items.jsonl.gz") not in g.store                               # stops at the conflict


def test_dataset_filter_and_dry_run(tmp_path):
    root, g = make_root(tmp_path), FakeGcs()
    logs = []
    r = upload(root, ["thrash"], gcs_mod=g, dry_run=True, bucket="b", log=logs.append)
    assert len(r["planned"]) == 2 and not g.store and not g.puts and all("thrash" in x for x in logs)
    r = upload(root, ["thrash"], gcs_mod=g, log=lambda *_: None)
    assert len(r["uploaded"]) == 2 and not any("lbmhqa" in k for k in g.store)


def test_dry_run_needs_no_credentials_or_bucket(tmp_path, monkeypatch):
    for k in ("GCP_SA_JSON", "OPA_GCP_KEY_FILE", "OPA_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    r = upload(make_root(tmp_path), dry_run=True, log=lambda *_: None)
    assert len(r["planned"]) == 4


def test_test_split_refused_before_prereg_tag(tmp_path):
    root = make_root(tmp_path, split="test")
    with pytest.raises(SplitRefused):
        upload(root, split="test", gcs_mod=FakeGcs(), prereg_tagged=lambda: False, log=lambda *_: None)
    g = FakeGcs()
    upload(root, split="test", gcs_mod=g, prereg_tagged=lambda: True, log=lambda *_: None)
    assert len(g.puts) == 4


def test_local_consistency_and_placeholder_bucket(tmp_path):
    root = make_root(tmp_path)
    mf = root / "thrash" / "dev" / "v2" / "r2" / "manifest.json"
    m = json.loads(mf.read_text())
    m["items_sha256"] = "0" * 64
    mf.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="does not match"):
        upload(root, gcs_mod=FakeGcs(), log=lambda *_: None)
    root2 = make_root(tmp_path / "b")
    with pytest.raises(RuntimeError, match="placeholder"):
        upload(root2, gcs_mod=FakeGcs("YOUR-EXISTING-BUCKET"), log=lambda *_: None)
    with pytest.raises(FileNotFoundError):
        upload(tmp_path / "empty", gcs_mod=FakeGcs(), log=lambda *_: None)


def test_find_cells(tmp_path):
    root = make_root(tmp_path)
    assert [(d, c) for d, c, _ in find_cells(root, ["lbmhqa", "thrash", "drift"], "dev", "v2")] == \
        [("lbmhqa", "late-h3-8k-a0"), ("thrash", "r2")]


def test_credentials_from_env_or_key_file_without_printing(tmp_path, capsys):
    env = {"GCP_SA_JSON": "{}"}
    assert load_credentials(env) == "env"
    key = tmp_path / "k.json"
    key.write_text('{"private_key": "SECRET"}')
    env = {"OPA_GCP_KEY_FILE": str(key)}
    assert load_credentials(env) == "key-file" and env["GCP_SA_JSON"] == key.read_text()
    assert "SECRET" not in capsys.readouterr().out
    assert load_credentials({}) == "adc"
    with pytest.raises(FileNotFoundError) as e:
        load_credentials({"OPA_GCP_KEY_FILE": str(tmp_path / "missing.json")})
    assert "SECRET" not in str(e.value)
