"""Test-split Track B specs (H4 evaluation, H6 remedies, training-free) and the guards behind them."""
import gzip
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from opa.trackb import metrics as M
from opa.trackb import remedies as R
from opa.trackb import trainfree
from opa.trackb import train as T
from tests.test_trackb_train import _cfg

ROOT = Path(__file__).resolve().parent.parent
_sp = importlib.util.spec_from_file_location("make_p4_test_jobs", ROOT / "jobs" / "make_p4_test_jobs.py")
G = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(G)
TAG = {"tag": "prereg-v2", "commit": "c8a0a5e910abf2da315bd00425f5c90d9e82600c"}
RATE = 0.00034544


def _test_cfg(**kw):
    base = dict(max_opt_steps=3, eval_at_steps=[3], save_checkpoint_at=[], eval_split="test", prereg_tag=TAG, resume_from="gs://x/step3/checkpoint.pt",
                monitor_seed0=100000, monitor={"attrib": 4, "passkey": 0, "probe": 1})
    base.update(kw)
    return _cfg("V3", **base)


@pytest.fixture
def no_tag_file(monkeypatch, tmp_path):
    monkeypatch.setattr(trainfree, "PROJECT_DIR", tmp_path)            # no prereg/PREREG_TAG.json: the container situation


# ---------------------------------------------------------------- guards
def test_test_split_evaluation_config_is_accepted():
    cfg = _test_cfg()
    assert cfg.eval_split == "test" and cfg.sizes(full=True).role == "test"
    assert _cfg("V3", eval_at_steps=[3], max_opt_steps=3).sizes(full=True).role == "eval"


def test_test_split_without_tag_is_refused(no_tag_file):
    with pytest.raises(trainfree.SplitRefused):
        _test_cfg(prereg_tag=None)


def test_test_split_with_tag_in_args_passes_without_file(no_tag_file):
    _test_cfg()


def test_test_split_needs_test_seeds():
    with pytest.raises(ValueError, match="100000"):
        _test_cfg(monitor_seed0=3000)


@pytest.mark.parametrize("kw", [{"save_checkpoint_at": [3]}, {"eval_at_steps": [2]}, {"resume_from": None}])
def test_test_split_is_evaluation_only(kw):
    with pytest.raises(ValueError, match="evaluation only"):
        _test_cfg(**kw)


def test_bad_split_name_refused():
    with pytest.raises(ValueError):
        _test_cfg(eval_split="holdout")


def test_dev_defaults_unchanged():
    cfg = _cfg("V3")
    assert cfg.eval_split == "dev" and cfg.prereg_tag is None
    assert M.EvalSizes().role == "eval"
    assert "eval_split" in T.RESUME_FREE_FIELDS and "prereg_tag" in T.RESUME_FREE_FIELDS


def test_dev_run_with_dev_eval_seeds_untouched():
    _cfg("V3", eval_at_steps=[3], monitor_seed0=3000, monitor={"attrib": 4, "passkey": 2, "probe": 1})


def test_remedies_refuses_test_without_items_uri_and_without_tag(no_tag_file):
    with pytest.raises(trainfree.SplitRefused):
        R.main({"split": "test", "seed0": 100000, "n_items": 1}, None)
    with pytest.raises(ValueError, match="items_uri"):
        R.main({"split": "test", "seed0": 100000, "n_items": 1, "prereg_tag": TAG}, None)
    with pytest.raises(ValueError, match="100000"):
        R.main({"split": "test", "seed0": 5, "n_items": 1, "prereg_tag": TAG, "items_uri": "x"}, None)


def test_load_items_uri_reads_a_local_gz(tmp_path):
    p = tmp_path / "items.jsonl.gz"
    p.write_bytes(gzip.compress(b'{"a": 1}\n\n{"a": 2}\n'))
    assert R.load_items_uri(str(p)) == [{"a": 1}, {"a": 2}]


# ---------------------------------------------------------------- specs
SPECS = G.all_specs()


def test_committed_specs_equal_the_generator_output():
    for name, (text, _, _) in SPECS.items():
        assert (ROOT / "jobs" / "p4" / f"{name}.yaml").read_text(encoding="utf8") == text, name


@pytest.mark.parametrize("name", sorted(SPECS))
def test_every_spec_is_test_split_free_of_api_and_within_cap(name):
    d = yaml.safe_load(SPECS[name][0])
    assert d["split"] == "test" and d["max_api_usd"] == 0 and d["compute"]["gpu"] == "L4"
    assert d["estimate_usd"] <= 2.50 and d["max_usd"] == round(2 * d["estimate_usd"], 2)
    assert d["seed"] >= 100000 and d["args"]["prereg_tag"] == TAG
    assert d["envelope"] == ("E4" if "h4" in name else "E3")
    assert int(d["max_usd"] / RATE) >= 1000
    assert "_RUN_ID" not in SPECS[name][0]


def test_h4_chunks_cover_2451_disjoint_seeds_identically_for_both_arms():
    cover = {}
    for arm in ("v3", "b1"):
        seeds = []
        for c in (1, 2):
            a = yaml.safe_load(SPECS[f"p4-h4-test-{arm}-c0{c}"][0])["args"]
            seeds += list(range(a["monitor_seed0"], a["monitor_seed0"] + a["monitor"]["attrib"]))
            assert a["monitor"]["passkey"] == 0 and a["max_opt_steps"] == 600 and a["eval_at_steps"] == [600] and a["save_checkpoint_at"] == []
            assert a["eval_split"] == "test" and a["skip_resume_eval"] is True
        cover[arm] = seeds
    assert cover["v3"] == cover["b1"] == list(range(100000, 100000 + 2451))
    v3 = yaml.safe_load(SPECS["p4-h4-test-v3-c01"][0])["args"]
    assert v3["monitor"]["probe"] == 16 and v3["resume_from"].startswith("gs://{bucket}/opa/models/" + G.V3_RUN)
    assert yaml.safe_load(SPECS["p4-h4-test-v3-c02"][0])["args"]["monitor"]["probe"] == 0
    assert G.B1_RUN in yaml.safe_load(SPECS["p4-h4-test-b1-c01"][0])["args"]["resume_from"]


def test_h4_args_are_accepted_by_trainconfig():
    for arm in ("v3", "b1"):
        for c in (1, 2):
            a = dict(yaml.safe_load(SPECS[f"p4-h4-test-{arm}-c0{c}"][0])["args"])
            a["resume_from"] = a["resume_from"].replace("{bucket}", "bkt")
            T.TrainConfig.from_args(a)


def test_h6_chunks_cover_each_cell_once():
    for kind, ds, cell in G.H6_CELLS:
        offs = []
        for c in (1, 2):
            d = yaml.safe_load(SPECS[f"p4-h6-{kind}-c0{c}"][0])
            assert d["dataset"] == ds and d["args"]["items_uri"] == f"gs://{{bucket}}/opa/data/{ds}/test/v2/{cell}/items.jsonl.gz"
            offs += list(range(d["args"]["offset"], d["args"]["offset"] + d["args"]["n_items"]))
        assert offs == list(range(200))
    assert "late-h2-16k-a0" in "\n".join(G.data_commands()) and "T4-h2-16k-nocross" in "\n".join(G.data_commands())


def test_trainfree_spec_matches_the_trainfree_entrypoint_args():
    a = yaml.safe_load(SPECS["p4-trainfree-test"][0])["args"]
    trainfree.check_split(a["split"], a["seed0"], a, a["n"])
    assert a["n"] == 80 and a["n_probe_windows"] == 16 and a["variants"] == ["V0", "V1", "B2", "B4"]


# ---------------------------------------------------------------- resume guards (tiny model)
def _checkpoint(tmp_path, steps=3):
    from tests.test_trackb_train import _run
    kw = dict(eval_every=0, log_every=1, monitor={"attrib": 2, "passkey": 2, "probe": 1})
    _, c, _ = _run(_cfg("V3", max_opt_steps=steps, eval_at_steps=[steps], save_checkpoint_at=[steps], **kw))
    ck = tmp_path / "c.pt"
    ck.write_bytes(c.bytes[f"models/{c.run_id}/step{steps}/checkpoint.pt"])
    return ck, kw


def test_test_split_refuses_a_checkpoint_that_is_not_at_max_opt_steps(tmp_path):
    from tests.test_trackb_train import _run
    ck, kw = _checkpoint(tmp_path, 3)
    kw["monitor"] = {"attrib": 2, "passkey": 0, "probe": 1}
    cfg = _cfg("V3", max_opt_steps=6, eval_at_steps=[6], resume_from=str(ck), skip_resume_eval=True, eval_split="test", prereg_tag=TAG, monitor_seed0=100000, **kw)
    with pytest.raises(ValueError, match="must not train"):
        _run(cfg)


def test_expected_resume_sha256_is_checked(tmp_path):
    import hashlib
    from tests.test_trackb_train import _run
    ck, kw = _checkpoint(tmp_path, 3)
    good = hashlib.sha256(ck.read_bytes()).hexdigest()
    base = dict(max_opt_steps=3, eval_at_steps=[3], resume_from=str(ck), skip_resume_eval=True, **kw)
    with pytest.raises(ValueError, match="resume refused: checkpoint .* sha256"):
        _run(_cfg("V3", expected_resume_sha256="0" * 64, **base))
    _, c, _ = _run(_cfg("V3", expected_resume_sha256=good, **base))
    assert any(e.get("resumed_from_sha256") == good for e in c.events) and "expected_resume_sha256" in T.RESUME_FREE_FIELDS


def test_v3_and_b1_specs_pin_the_checkpoint_sha_and_b1_sha_flag_overrides():
    assert all(yaml.safe_load(SPECS[f"p4-h4-test-v3-c0{c}"][0])["args"]["expected_resume_sha256"] == G.V3_SHA for c in (1, 2))
    pinned = "eeccb50e8f89f95dd82f008a376a6b6b245137c633a3017fd01b10b136ed8bff"      # DEV-TRAIN-PIN.md
    assert all(yaml.safe_load(SPECS[f"p4-h4-test-b1-c0{c}"][0])["args"]["expected_resume_sha256"] == pinned for c in (1, 2))
    s = "ab" * 32
    assert all(yaml.safe_load(G.all_specs(s)[f"p4-h4-test-b1-c0{c}"][0])["args"]["expected_resume_sha256"] == s for c in (1, 2))
    gg = importlib.util.spec_from_file_location("mk", ROOT / "jobs" / "make_p4_train_jobs.py")
    m = importlib.util.module_from_spec(gg)
    gg.loader.exec_module(m)
    assert yaml.safe_load(m.g2_eval("V3", 600)[0])["args"]["expected_resume_sha256"] == G.V3_SHA
    assert yaml.safe_load(m.g2_eval("B1", 600)[0])["args"]["expected_resume_sha256"] == pinned == m.B1_FINAL_SHA[600]
    assert yaml.safe_load(m.g2_eval("B1", 600, sha=s)[0])["args"]["expected_resume_sha256"] == s
