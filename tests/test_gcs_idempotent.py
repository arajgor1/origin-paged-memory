import pytest
from google.api_core.exceptions import PreconditionFailed

from opa import gcs


class FakeBlob:
    def __init__(self, existing=None, fail_reload=False, drop=None):
        self.drop = drop
        self.existing = existing  # bytes already stored, or None
        self.fail_reload = fail_reload
        self.crc32c = self.md5_hash = None
        self.uploads = 0

    def _up(self, data):
        self.uploads += 1
        if self.existing is not None:
            raise PreconditionFailed("412")
        self.existing = data

    def upload_from_string(self, data, content_type=None, if_generation_match=None):
        assert if_generation_match == 0
        self._up(data)

    def upload_from_filename(self, path, if_generation_match=None):
        assert if_generation_match == 0
        self._up(open(path, "rb").read())

    def reload(self):
        if self.fail_reload:
            raise RuntimeError("no metadata")
        self.crc32c = None if self.drop == "crc" else gcs._crc32c_b64(self.existing)
        self.md5_hash = None if self.drop == "md5" else gcs._md5_b64(self.existing)


@pytest.fixture
def blob(monkeypatch):
    holder = {}

    class Bucket:
        def blob(self, p):
            return holder["blob"]

    class Client:
        def bucket(self, b):
            return Bucket()

    monkeypatch.setattr(gcs, "client", lambda: Client())

    def make(**kw):
        holder["blob"] = FakeBlob(**kw)
        return holder["blob"]
    return make


U = "gs://bkt/a/b.bin"


def test_normal_path(blob):
    b = blob()
    r = gcs.put_bytes(U, b"abc")
    assert r["uri"] == U and r["bytes"] == 3 and r["sha256"] == gcs.sha256_bytes(b"abc")
    assert b.existing == b"abc"


def test_412_identical_is_success(blob):
    b = blob(existing=b"abc")
    r = gcs.put_bytes(U, b"abc")
    assert r == {"uri": U, "sha256": gcs.sha256_bytes(b"abc"), "bytes": 3}
    assert b.existing == b"abc" and b.uploads == 1


def test_412_different_raises(blob):
    b = blob(existing=b"xyz")
    with pytest.raises(RuntimeError, match="different content"):
        gcs.put_bytes(U, b"abc")
    assert b.existing == b"xyz" and b.uploads == 1


def test_412_metadata_unavailable_raises(blob):
    blob(existing=b"abc", fail_reload=True)
    with pytest.raises(RuntimeError, match="metadata unavailable"):
        gcs.put_bytes(U, b"abc")


def test_put_file_normal_and_412(blob, tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello")
    b = blob()
    assert gcs.put_file(f, U)["bytes"] == 5
    b = blob(existing=b"hello")
    r = gcs.put_file(f, U)
    assert r["sha256"] == gcs.sha256_bytes(b"hello") and r["bytes"] == 5


def test_put_file_412_different_raises(blob, tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello")
    b = blob(existing=b"world")
    with pytest.raises(RuntimeError, match="different content"):
        gcs.put_file(f, U)
    assert b.existing == b"world"


@pytest.mark.parametrize("drop", ["md5", "crc"])
def test_412_missing_one_checksum_raises(blob, drop):
    b = blob(existing=b"abc", drop=drop)
    with pytest.raises(RuntimeError, match="lacks crc32c or md5"):
        gcs.put_bytes(U, b"abc")
    assert b.uploads == 1
