"""P4 stopping-rule training jobs: the generator (jobs/make_p4_train_jobs.py), the rule-4 helper, the chain of chunks on a tiny model."""
import hashlib
import importlib.util
import io
import json
import re
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml

from opa import launch
from opa.trackb import data as D
from opa.trackb import train as T
from tests.test_trackb_train import Ctx, _cfg, _run

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("make_p4", ROOT / "jobs" / "make_p4_train_jobs.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def _load(text):
    return yaml.safe_load(text)


def _cfg_of(args):
    return T.TrainConfig.from_args(dict(args))


STEP1_ARGS = yaml.safe_load((ROOT / "jobs" / "p3_learning_curve_v3.yaml").read_text())["args"]
B1_PILOT_ARGS = {**yaml.safe_load((ROOT / "jobs" / "p1_trackb_minilora_b1.yaml").read_text())["args"], "max_opt_steps": 500}


# ---------------------------------------------------------------- the committed V3 chunk specs


def test_committed_v3_chunks_equal_the_generator_output():
    for k in range(1, 9):
        text = (ROOT / "jobs" / "p4" / f"p4-v3-c{k:02d}.yaml").read_text()
        assert text == G.v3_chunk(k)[0], k
    assert not any((ROOT / "jobs" / "p4").glob("p4-b1-c*"))                              # B1 runs in parallel (p4-b1-par-*), not as chained chunks


@pytest.mark.parametrize("k", range(1, 9))
def test_v3_chunk_fields_estimates_and_placeholders(k):
    text, est, mx = G.v3_chunk(k)
    d = _load(text)
    end = 400 + 200 * k
    a = d["args"]
    assert d["envelope"] == "E4" and d["split"] == "dev" and d["max_api_usd"] == 0 and d["compute"]["gpu"] == "L4" and d["track"] == "B"
    assert d["entrypoint"] == "opa.trackb.train:main" and d["name"] == f"p4-v3-c{k:02d}"
    assert a["arm"] == "V3" and a["max_opt_steps"] == end and a["eval_at_steps"] == [end] and a["save_checkpoint_at"] == [end]
    assert a["skip_resume_eval"] is True and a["monitor"] == {"attrib": 64, "passkey": 32, "probe": 3} and "monitor_seed0" not in a   # seeds 2000-2063
    assert d["estimate_usd"] == est <= 2.50 and d["max_usd"] == mx == round(2 * est, 2) and est == pytest.approx(5150 * 0.00034544, abs=0.006)
    prev = G.STEP1_RUN_ID if k == 1 else f"P4_V3_C{k - 1:02d}_RUN_ID"
    assert a["resume_from"] == f"gs://{{bucket}}/opa/models/{prev}/step{end - 200}/checkpoint.pt"
    unfilled = set(launch.PLACEHOLDER.findall(yaml.safe_dump(d)))
    assert unfilled == (set() if k == 1 else {f"P4_V3_C{k - 1:02d}_RUN_ID"})              # the launcher refuses an unfilled chunk
    filled = _load(text.replace(f"P4_V3_C{k - 1:02d}_RUN_ID", "B-V3-fake-run") if k > 1 else text)
    assert not launch.PLACEHOLDER.findall(yaml.safe_dump(filled))


def test_every_chunk_passes_the_resume_config_check_against_the_learning_curve_run_and_its_predecessor():
    c0 = _cfg_of(STEP1_ARGS)
    prev = c0
    for k in range(1, 9):
        c = _cfg_of(_load(G.v3_chunk(k)[0])["args"])
        assert T.resume_config_diff(c, asdict(c0)) == {}, k            # identical training configuration to the Step-1 run
        assert T.resume_config_diff(c, asdict(prev)) == {}, k           # and to the previous chunk (what its checkpoint carries)
        prev = c


# ---------------------------------------------------------------- the rule-4 helper


def test_decide_continues_from_the_learning_curve_and_stops_on_a_small_rise():
    pts = {0: 0.137, 50: 0.183, 100: 0.254, 200: 0.249, 400: 0.3066}
    d = G.decide(pts)
    assert d["decision"] == "CONTINUE" and d["next_chunk_end"] == 600 and d["next_chunk_index"] == 1 and d["rise"] == pytest.approx(0.0576)
    d = G.decide({**pts, 600: 0.3265})
    assert d["decision"] == "STOP" and d["use_checkpoint_step"] == 600 and d["rise"] == pytest.approx(0.0199)     # 0.0199 < 0.02


def test_decide_boundary_is_exactly_0_02_continue():
    base = {200: 0.2866, 400: 0.3066}
    assert G.decide({**base, 600: 0.3266})["decision"] == "CONTINUE"             # rise 0.0200 (float noise 2e-17): not "less than"
    assert G.decide({**base, 600: 0.32659})["decision"] == "STOP"
    assert G.decide({**base, 600: 0.32661})["decision"] == "CONTINUE"
    assert G.decide({200: 0.30, 400: 0.32})["decision"] == "CONTINUE"            # step 400 itself obeys the rule at exactly 0.02
    assert G.decide({200: 0.30, 400: 0.3199})["decision"] == "STOP"
    assert G.decide({200: 0.30, 400: 0.30})["decision"] == "STOP"                # zero rise
    fall = G.decide({200: 0.2, 400: 0.5, 600: 0.4})
    assert fall["decision"] == "STOP" and fall["rise"] < 0                       # a fall also stops


def test_decide_never_before_400_and_always_at_2000_and_final_not_best():
    assert G.decide({0: 0.1, 50: 0.1, 100: 0.1, 200: 0.1})["decision"] == "CONTINUE"        # flat before 400 does not stop
    chain = {200: 0.2, 400: 0.4, 600: 0.5, 800: 0.6, 1000: 0.7, 1200: 0.8, 1400: 0.85, 1600: 0.88, 1800: 0.90, 2000: 0.95}
    d = G.decide(chain)
    assert d["decision"] == "STOP" and d["use_checkpoint_step"] == 2000 and "limit" in d["reason"]
    d = G.decide({200: 0.2, 400: 0.40, 600: 0.50, 800: 0.49})                              # peaked at 600, fell at 800
    assert d["decision"] == "STOP" and d["use_checkpoint_step"] == 800 and d["best_step_observed"] == 600     # final, never the best
    with pytest.raises(ValueError, match="missing"):
        G.decide({200: 0.2, 400: 0.3, 800: 0.4})                                             # the step-600 evaluation is missing
    with pytest.raises(ValueError):
        G.decide({})


def test_decide_raises_when_trained_past_the_stopping_point():
    pts = {200: 0.2, 400: 0.4, 600: 0.405, 800: 0.6}              # the rule stops at 600 (rise 0.005); step 800 should not exist
    with pytest.raises(ValueError, match="trained past the stopping point at step 600"):
        G.decide(pts)
    assert G.decide({200: 0.2, 400: 0.4, 600: 0.405})["use_checkpoint_step"] == 600
    pts = {200: 0.2, 400: 0.4, 600: 0.5, 800: 0.6, 1000: 0.601, 1200: 0.9}
    with pytest.raises(ValueError, match="step 1000"):
        G.decide(pts)
    full = {s: 0.1 * i for i, s in enumerate(range(200, 2001, 200))}
    with pytest.raises(ValueError, match="trained past"):
        G.decide({**full, 2200: 5.0})                                                          # the limit was reached at 2000


@pytest.mark.parametrize("n", [0, 200, 300, 500, 2200, 2400])
def test_b1_refuses_n_that_is_not_a_multiple_of_200_in_400_2000(n):
    with pytest.raises(ValueError, match="multiple of 200"):
        G.b1_chunk(1, n)
    with pytest.raises(ValueError):
        G.check_steps(n)


def test_b1_steps_from_a_decide_stop_json(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUT", tmp_path)
    stop = tmp_path / "stop.json"
    stop.write_text(json.dumps({"decision": "STOP", "use_checkpoint_step": 1200}))
    G.main(["b1", "--steps-from", str(stop)])
    assert sorted(p.name for p in tmp_path.glob("p4-b1-*.yaml")) == ["p4-b1-c01.yaml", "p4-b1-c02.yaml"]
    assert _load((tmp_path / "p4-b1-c02.yaml").read_text())["args"]["max_opt_steps"] == 1200
    cont = tmp_path / "cont.json"
    cont.write_text(json.dumps({"decision": "CONTINUE", "next_chunk_end": 800}))
    with pytest.raises(ValueError, match="not STOP"):
        G.main(["b1", "--steps-from", str(cont)])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"decision": "STOP", "use_checkpoint_step": 1300}))
    with pytest.raises(ValueError, match="multiple of 200"):
        G.main(["b1", "--steps-from", str(bad)])
    with pytest.raises(ValueError):
        G.main(["b1", "--steps", "500"])


def test_g2_time_budget_and_consistent_probe_weighting():
    text, est, mx = G.g2_eval("V3", 1200)
    d = _load(text)
    assert d["args"]["time_budget_s"] == 1200
    unit = 240 / (64 + 32 + 3 * G.PROBE_ITEM_EQUIV)
    expect = (300 + 60 + unit * (128 + 32 + 6 * G.PROBE_ITEM_EQUIV)) * 0.00034544
    assert est == pytest.approx(expect, abs=0.006)
    assert "weighted as 10 items" in d["estimate_formula"] and "220 units" in d["estimate_formula"]
    with pytest.raises(ValueError, match="multiple of 200"):
        G.main(["g2", "--v3-step", "1300", "--b1-step", "1300"])                              # not a multiple of 200


def test_decide_cli_reads_learning_curve_files(tmp_path, capsys):
    def entry(tag, step, frac):
        return {"tag": tag, "opt_step": step, "probe": {"own_origin_mass_fraction": frac}}

    f1 = tmp_path / "step1.json"
    f1.write_text(json.dumps({"status": "complete", "curve": [entry("step0", 0, .137), entry("step200", 200, .249), entry("step400", 400, .3066)]}))
    f2 = tmp_path / "c01.json"
    f2.write_text(json.dumps({"status": "complete", "curve": [{"tag": "resume_start", "opt_step": 400, "probe": {"own_origin_mass_fraction": 9.9}},
                                                               entry("step600", 600, .3265), {"tag": "step800", "opt_step": 800}]}))
    assert G.curve_points([f1, f2]) == {0: .137, 200: .249, 400: .3066, 600: .3265}            # resume_start / entries without a probe are ignored
    f3 = tmp_path / "bare.json"
    f3.write_text(json.dumps([entry("step600", 600, .3267)]))
    assert G.curve_points([f1, f3])[600] == .3267
    assert G.main(["--decide", str(f1), str(f2)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "STOP" and out["use_checkpoint_step"] == 600
    G.main(["--decide", str(f1), str(f3)])
    assert json.loads(capsys.readouterr().out)["decision"] == "CONTINUE"


# ---------------------------------------------------------------- B1 family and G2 evaluation


@pytest.mark.parametrize("n,ends", [(2000, [1000, 2000]), (1400, [1000, 1400]), (800, [800]), (1000, [1000]), (400, [400])])
def test_b1_family_trains_exactly_n_steps(n, ends):
    specs = [_load(G.b1_chunk(j, n)[0]) for j in range(1, len(ends) + 1)]
    assert [s["args"]["max_opt_steps"] for s in specs] == ends and ends[-1] == n
    for j, s in enumerate(specs, 1):
        a = s["args"]
        assert a["arm"] == "B1" and isinstance(a["max_opt_steps"], int) and a["eval_at_steps"] == [ends[j - 1]] and a["save_checkpoint_at"] == [ends[j - 1]]
        assert s["envelope"] == "E4" and s["split"] == "dev" and s["max_api_usd"] == 0 and s["estimate_usd"] <= 2.5
        if j == 1:
            assert "resume_from" not in a and "skip_resume_eval" not in a
        else:
            assert a["resume_from"] == f"gs://{{bucket}}/opa/models/P4_B1_C{j - 1:02d}_RUN_ID/step{ends[j - 2]}/checkpoint.pt" and a["skip_resume_eval"] is True
        c = _cfg_of(a)
        assert T.resume_config_diff(c, asdict(_cfg_of(B1_PILOT_ARGS))) == {}                 # the pilot's data and configuration except length
    with pytest.raises(ValueError):
        G.b1_chunk(len(ends) + 1, n)


def test_g2_eval_specs_use_fresh_seeds_and_evaluate_only():
    for arm in ("V3", "B1"):
        d = _load(G.g2_eval(arm, 1200)[0])
        a = d["args"]
        assert a["arm"] == arm and a["max_opt_steps"] == 1200 == a["eval_at_steps"][0] and a["save_checkpoint_at"] == [] and a["save_adapter"] is False
        assert a["monitor_seed0"] == D.G2_SEED0 == 3000 and a["resume_from"].endswith("/step1200/checkpoint.pt")
        assert f"P4_{arm}_FINAL_RUN_ID" in a["resume_from"] and d["envelope"] == "E4" and d["split"] == "dev" and d["estimate_usd"] <= 2.5
        lo, hi = a["monitor_seed0"], a["monitor_seed0"] + max(a["monitor"]["attrib"], a["monitor"]["passkey"], a["monitor"]["probe"])
        assert lo >= 3000 and not (lo < 2064 and hi > 2000) and not (lo < 1000 and hi > 900)       # disjoint from monitoring 2000-2063 and pilot 900-999
        c = _cfg_of(a)
        ref = _cfg_of(_load(G.v3_chunk(1)[0])["args"] if arm == "V3" else G_b1_args())
        assert T.resume_config_diff(c, asdict(ref)) == {}                                             # the resume check passes for evaluation too
    with pytest.raises(SystemExit):
        G.main(["g2", "--v3-step", "1200", "--b1-step", "1000"])                                       # B1 must have exactly V3's steps


def G_b1_args():
    return {**_load(G.b1_chunk(1, 1200)[0])["args"]}


def test_generator_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUT", tmp_path)
    G.main(["b1", "--steps", "1400"])
    G.main(["g2", "--v3-step", "1400", "--b1-step", "1400"])
    G.main(["v3", "--chunks", "2"])
    assert sorted(p.name for p in tmp_path.glob("*.yaml")) == ["p4-b1-c01.yaml", "p4-b1-c02.yaml", "p4-g2-eval-b1.yaml", "p4-g2-eval-v3.yaml",
                                                          "p4-v3-c01.yaml", "p4-v3-c02.yaml"]
    assert all(b"\r" not in p.read_bytes() for p in tmp_path.glob("*.yaml"))                     # LF only


# ---------------------------------------------------------------- the chain on a tiny model: resume without re-evaluation, hashes, evaluation only


def test_chain_chunk_resume_skip_eval_hashes_and_g2_style_evaluation(tmp_path):
    # chunk 1 (from scratch here): steps 0 -> 2, checkpoint and one monitoring evaluation
    c1 = dict(max_opt_steps=2, eval_at_steps=[2], save_checkpoint_at=[2], eval_every=0, monitor={"attrib": 2, "passkey": 2, "probe": 1})
    out1, ctx1, _ = _run(_cfg("V3", **c1))
    ck1 = ctx1.bytes[f"models/{ctx1.run_id}/step2/checkpoint.pt"]
    assert out1["metrics"]["checkpoint_sha256_step2"] == hashlib.sha256(ck1).hexdigest() == out1["metrics"]["final_checkpoint_sha256"]
    assert ctx1.put["config.json"]["checkpoint_sha256"] == {"step2": hashlib.sha256(ck1).hexdigest()}
    assert out1["metrics"]["adapter_sha256"] == hashlib.sha256(ctx1.bytes[f"models/{ctx1.run_id}/adapter.pt"]).hexdigest()
    saved = [e for e in ctx1.events if e["action"] == "checkpoint_saved"][0]
    assert saved["sha256"] == hashlib.sha256(ck1).hexdigest() and saved["bytes"] == len(ck1)
    p1 = tmp_path / "c1.pt"
    p1.write_bytes(ck1)
    # chunk 2: resume, skip the evaluation of the resumed state, steps 2 -> 4, one evaluation at 4
    c2 = dict(max_opt_steps=4, eval_at_steps=[4], save_checkpoint_at=[4], eval_every=0, monitor={"attrib": 2, "passkey": 2, "probe": 1},
              resume_from=str(p1), skip_resume_eval=True)
    out2, ctx2, _ = _run(_cfg("V3", **c2))
    assert [c["tag"] for c in ctx2.put["learning_curve.json"]["curve"]] == ["step4"]               # one evaluation only, no resume_start
    assert out2["metrics"]["opt_steps"] == 4 and any(e["action"] == "resume_eval_skipped" for e in ctx2.events)
    assert [e["ok"] for e in ctx2.events if e["action"] == "resume_config_check"] == [True]
    ck2 = ctx2.bytes[f"models/{ctx2.run_id}/step4/checkpoint.pt"]
    assert out2["metrics"]["final_checkpoint_sha256"] == hashlib.sha256(ck2).hexdigest() != hashlib.sha256(ck1).hexdigest()
    assert ctx2.put["learning_curve.json"]["curve"][0]["train_loss"] is not None
    # G2-style evaluation only: zero training steps, fresh seeds, no adapter, no checkpoint
    p2 = tmp_path / "c2.pt"
    p2.write_bytes(ck2)
    g2 = dict(max_opt_steps=4, eval_at_steps=[4], save_checkpoint_at=[], eval_every=0, resume_from=str(p2), skip_resume_eval=True, save_adapter=False,
              monitor_seed0=D.G2_SEED0, monitor={"attrib": 3, "passkey": 2, "probe": 1})
    out3, ctx3, _ = _run(_cfg("V3", **g2))
    assert out3["metrics"]["opt_steps"] == 4 and out3["metrics"]["micro_steps"] == 8 and out3["metrics"]["nan_steps"] == 0
    assert not any(e["action"] == "train_step" for e in ctx3.events) and f"models/{ctx3.run_id}/adapter.pt" not in ctx3.bytes
    fe = ctx3.put["final_eval.json"]
    assert fe["status"] == "complete" and fe["step0"] is None and fe["per_item"]["final"]["meta"]["attrib"][0]["seed"] == 3000
    assert [m["seed"] for m in fe["per_item"]["final"]["meta"]["attrib"]] == [3000, 3001, 3002] and "probe" in fe["final"]
    # the same weights as the checkpoint: evaluating the step-4 checkpoint twice gives identical per-item outcomes (determinism on CPU)
    out4, ctx4, _ = _run(_cfg("V3", **g2))
    assert ctx4.put["final_eval.json"]["per_item_final"] == fe["per_item_final"]


def test_chunk_with_a_changed_training_config_is_refused(tmp_path):
    _, ctx1, _ = _run(_cfg("V3", max_opt_steps=2, eval_at_steps=[2], save_checkpoint_at=[2], eval_every=0, monitor={"attrib": 2, "passkey": 2, "probe": 1}))
    p = tmp_path / "c.pt"
    p.write_bytes(ctx1.bytes[f"models/{ctx1.run_id}/step2/checkpoint.pt"])
    with pytest.raises(ValueError, match="resume refused"):
        _run(_cfg("V3", max_opt_steps=4, eval_at_steps=[4], eval_every=0, resume_from=str(p), skip_resume_eval=True, lr_new=2e-3,
                  monitor={"attrib": 2, "passkey": 2, "probe": 1}))


# ---------------------------------------------------------------- {bucket} resolution, resumed_from_sha256, bit-exact chunked resume


class SpecCtx(Ctx):
    def __init__(self, bucket="gs://bkt/"):
        super().__init__()
        self.spec = {"bucket": bucket}


def _saved_ckpt(tmp_path):
    _, ctx1, _ = _run(_cfg("V3", max_opt_steps=2, eval_at_steps=[2], save_checkpoint_at=[2], eval_every=0, monitor={"attrib": 2, "passkey": 2, "probe": 1}))
    return ctx1.bytes[f"models/{ctx1.run_id}/step2/checkpoint.pt"]


def test_resolve_bucket(monkeypatch):
    from opa import gcs
    assert T.resolve_bucket("gs://{bucket}/opa/models/x", SpecCtx("gs://bkt/")) == "gs://bkt/opa/models/x"
    assert T.resolve_bucket("gs://{bucket}/opa/models/x", SpecCtx("plain")) == "gs://plain/opa/models/x"
    assert T.resolve_bucket("/tmp/local.pt", SpecCtx()) == "/tmp/local.pt" and T.resolve_bucket(None) is None
    monkeypatch.setattr(gcs, "bucket_name", lambda: "fallback-bucket")
    assert T.resolve_bucket("gs://{bucket}/a", Ctx()) == "gs://fallback-bucket/a"                  # no spec on the context: project config
    assert T.resolve_bucket("gs://{bucket}/a", None) == "gs://fallback-bucket/a"


def test_resumed_run_resolves_bucket_logs_it_and_records_the_checkpoint_hash(tmp_path, monkeypatch):
    from opa import gcs
    from opa.trackb import units
    data = _saved_ckpt(tmp_path)
    asked = []
    monkeypatch.setattr(gcs, "get_bytes", lambda uri: (asked.append(uri), data)[1])
    ftok, enc = D.make_tokenizers("hash", 512)
    filler = D.make_filler("synthetic", n_books=24, words_per_book=6000)
    ctx = SpecCtx("gs://bkt/")
    cfg = _cfg("V3", max_opt_steps=4, eval_at_steps=[4], eval_every=0, skip_resume_eval=True, monitor={"attrib": 2, "passkey": 2, "probe": 1},
               resume_from="gs://{bucket}/opa/models/RUN-1/step2/checkpoint.pt")
    out = T.run(cfg, units.tiny_model(seed=0, vocab=512), ftok, enc, filler, ctx)
    sha = hashlib.sha256(data).hexdigest()
    assert asked == ["gs://bkt/opa/models/RUN-1/step2/checkpoint.pt"]
    ev = [e for e in ctx.events if e["action"] == "resumed"][0]
    assert ev["from"] == asked[0] and ev["resumed_from_sha256"] == sha
    assert ctx.put["config.json"]["resumed_from_sha256"] == sha and ctx.put["config.json"]["resume_from_resolved"] == asked[0]
    assert out["metrics"]["resumed_from_sha256"] == sha and ctx.put["config.json"]["config"]["resume_from"].startswith("gs://{bucket}")


def test_partial_config_carries_the_resume_hash_and_every_p4_spec_resolves(tmp_path, monkeypatch):
    from opa import gcs
    from opa.trackb import arms, units
    data = _saved_ckpt(tmp_path)
    monkeypatch.setattr(gcs, "get_bytes", lambda uri: data)
    ftok, enc = D.make_tokenizers("hash", 512)
    filler = D.make_filler("synthetic", n_books=24, words_per_book=6000)
    calls = {"n": 0}
    orig = arms.forward_example

    def boom(*a, **k):
        if k.get("backward"):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated")
        return orig(*a, **k)

    monkeypatch.setattr(arms, "forward_example", boom)
    ctx = SpecCtx()
    cfg = _cfg("V3", max_opt_steps=6, eval_at_steps=[6], eval_every=0, skip_resume_eval=True, monitor={"attrib": 2, "passkey": 2, "probe": 1},
               resume_from="gs://{bucket}/opa/models/R/step2/checkpoint.pt")
    with pytest.raises(RuntimeError):
        T.run(cfg, units.tiny_model(seed=0, vocab=512), ftok, enc, filler, ctx)
    assert ctx.put["config.json"]["status"] == "partial" and ctx.put["config.json"]["resumed_from_sha256"] == hashlib.sha256(data).hexdigest()
    # every {bucket} in the P4 specs is in resume_from, which resolve_bucket handles
    for f in sorted((ROOT / "jobs" / "p4").glob("*.yaml")):
        d = _load(f.read_text())
        if f.name.startswith(("p4-h6-", "p4-trainfree-")):                                        # test-split specs: no checkpoint (tests/test_p4_test_jobs.py)
            continue
        if f.name == "p4-b1-par-c01.yaml":                                                          # from scratch: no checkpoint to resolve
            assert "{bucket}" not in json.dumps(d)
            continue
        assert json.dumps(d).count("{bucket}") == 1 and "{bucket}" in d["args"]["resume_from"]
        assert T.resolve_bucket(d["args"]["resume_from"], SpecCtx("bkt")).startswith("gs://bkt/opa/models/")
    for arm_ in ("V3", "B1"):
        assert G.g2_eval(arm_, 1200)[0].count("{bucket}") == 1


@pytest.mark.parametrize("arm", ["B1", "V3"])
def test_chunked_resume_is_bit_exact(arm, tmp_path):
    """0 -> 3 -> 6 with skip_resume_eval equals 0 -> 6 exactly: step losses, LoRA and OAA parameters, optimizer state and the final evaluation."""
    mon = {"attrib": 2, "passkey": 2, "probe": 1}
    kw = dict(eval_every=0, log_every=1, monitor=mon)
    full, cf, _ = _run(_cfg(arm, max_opt_steps=6, eval_at_steps=[6], save_checkpoint_at=[6], **kw))
    p1, c1, _ = _run(_cfg(arm, max_opt_steps=3, eval_at_steps=[3], save_checkpoint_at=[3], **kw))
    ck = tmp_path / "c.pt"
    ck.write_bytes(c1.bytes[f"models/{c1.run_id}/step3/checkpoint.pt"])
    p2, c2, _ = _run(_cfg(arm, max_opt_steps=6, eval_at_steps=[6], save_checkpoint_at=[6], resume_from=str(ck), skip_resume_eval=True, **kw))
    loss = lambda c: {e["opt_step"]: e["loss"] for e in c.events if e["action"] == "train_step"}
    lf, l1, l2 = loss(cf), loss(c1), loss(c2)
    assert sorted(l2) == [4, 5, 6] and {**l1, **l2} == lf                                              # identical floats, step by step
    a = torch.load(io.BytesIO(cf.bytes[f"models/{cf.run_id}/step6/checkpoint.pt"]), weights_only=False)
    b = torch.load(io.BytesIO(c2.bytes[f"models/{c2.run_id}/step6/checkpoint.pt"]), weights_only=False)
    assert a["micro"] == b["micro"] == 12 and a["opt_step"] == b["opt_step"] == 6
    assert all(torch.equal(a["lora"][k], b["lora"][k]) for k in a["lora"])
    assert all(torch.equal(a["oaa"][k], b["oaa"][k]) for k in a["oaa"]) and (len(a["oaa"]) > 0) == (arm == "V3")
    for sa, sb in zip(a["opt"]["state"].values(), b["opt"]["state"].values()):
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert cf.put["final_eval.json"]["per_item_final"] == c2.put["final_eval.json"]["per_item_final"]
    assert cf.put["learning_curve.json"]["curve"][-1]["probe"] == c2.put["learning_curve.json"]["curve"][-1]["probe"]


# ---------------------------------------------------------------- B1 in parallel with V3 (owner instruction 2026-11-08)


def test_b1_parallel_saves_every_multiple_of_200_from_400_and_each_chunk_end():
    saves = [G.b1par_saves(j) for j in (1, 2)]
    assert saves == [[400, 600, 800, 1000], [1200, 1400, 1600, 1800, 2000]]
    assert sorted(s for ss in saves for s in ss) == list(range(400, 2001, 200))                      # each step exactly once, none missing
    for chunk in (400, 600, 700, 1000, 2000):
        n = -(-2000 // chunk)
        got = [G.b1par_saves(j, chunk) for j in range(1, n + 1)]
        flat = sorted(s for ss in got for s in ss)
        assert set(range(400, 2001, 200)) <= set(flat) and len(flat) == len(set(flat))               # every V3 stopping step is checkpointed once
        assert all(min(j * chunk, 2000) in got[j - 1] for j in range(1, n + 1))                       # and each chunk end


def test_b1_parallel_specs():
    for j in (1, 2):
        text, est, mx = G.b1par_chunk(j)
        d = _load(text)
        a = d["args"]
        assert d["name"] == f"p4-b1-par-c{j:02d}" and d["envelope"] == "E4" and d["split"] == "dev" and d["max_api_usd"] == 0 and d["compute"]["gpu"] == "L4"
        assert a["arm"] == "B1" and a["no_eval"] is True and "eval_at_steps" not in a and a["max_opt_steps"] == 1000 * j
        assert a["save_checkpoint_at"] == G.b1par_saves(j) and a["headers"] is True and a["seed"] == 0
        assert d["estimate_usd"] == est <= 2.5 and d["max_usd"] == mx == round(2 * est, 2) and 1.7 < est < 2.0
        assert est == pytest.approx((300 + 60 + 1000 * 5 + 15 * len(G.b1par_saves(j))) * 0.00034544, abs=0.006)
        c = _cfg_of(a)
        assert T.resume_config_diff(c, asdict(_cfg_of(B1_PILOT_ARGS))) == {}                          # the pilot's data and configuration except length
        if j == 1:
            assert "resume_from" not in a and not launch.PLACEHOLDER.findall(yaml.safe_dump(d))
        else:
            assert a["resume_from"] == "gs://{bucket}/opa/models/P4_B1PAR_C01_RUN_ID/step1000/checkpoint.pt"
            assert set(launch.PLACEHOLDER.findall(yaml.safe_dump(d))) == {"P4_B1PAR_C01_RUN_ID"}
    assert [(ROOT / "jobs" / "p4" / f"p4-b1-par-c{j:02d}.yaml").read_text() for j in (1, 2)] == [G.b1par_chunk(j)[0] for j in (1, 2)]
    with pytest.raises(ValueError):
        G.b1par_chunk(3)


@pytest.mark.parametrize("L", list(range(400, 2001, 200)))
def test_g2_b1_parallel_picks_the_chunk_that_saved_step_l(L):
    specs = {j: _load(G.b1par_chunk(j)[0]) for j in (1, 2)}
    saved_by = [j for j in specs if L in specs[j]["args"]["save_checkpoint_at"]]
    assert len(saved_by) == 1                                                                        # exactly one chunk wrote the step-L checkpoint
    d = _load(G.g2_eval("B1", L, b1_parallel=True)[0])
    a = d["args"]
    assert a["resume_from"] == f"gs://{{bucket}}/opa/models/P4_B1PAR_C{saved_by[0]:02d}_RUN_ID/step{L}/checkpoint.pt" == f"gs://{{bucket}}/opa/models/P4_B1PAR_C{G.b1par_chunk_of(L):02d}_RUN_ID/step{L}/checkpoint.pt"
    assert a["max_opt_steps"] == L == a["eval_at_steps"][0] and a["save_checkpoint_at"] == [] and a["save_adapter"] is False and a["monitor_seed0"] == 3000
    assert set(launch.PLACEHOLDER.findall(yaml.safe_dump(d))) == {f"P4_B1PAR_C{saved_by[0]:02d}_RUN_ID"}
    assert T.resume_config_diff(_cfg_of(a), asdict(_cfg_of(specs[saved_by[0]]["args"]))) == {}      # the resume check passes for the evaluation
    plain = _load(G.g2_eval("B1", L)[0])
    assert "P4_B1_FINAL_RUN_ID" in plain["args"]["resume_from"]                                      # the non-parallel variant is unchanged


def test_g2_b1_parallel_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUT", tmp_path)
    G.main(["g2", "--v3-step", "1200", "--b1-parallel"])
    b1 = _load((tmp_path / "p4-g2-eval-b1.yaml").read_text())["args"]
    v3 = _load((tmp_path / "p4-g2-eval-v3.yaml").read_text())["args"]
    assert "P4_B1PAR_C02_RUN_ID/step1200" in b1["resume_from"] and "P4_V3_FINAL_RUN_ID/step1200" in v3["resume_from"]
    assert b1["max_opt_steps"] == v3["max_opt_steps"] == 1200
    with pytest.raises(ValueError, match="multiple of 200"):
        G.main(["g2", "--v3-step", "1300", "--b1-parallel"])
    G.main(["b1-parallel", "--chunk", "1000"])
    assert sorted(p.name for p in tmp_path.glob("p4-b1-par-*.yaml")) == ["p4-b1-par-c01.yaml", "p4-b1-par-c02.yaml"]


def test_no_eval_trains_and_saves_every_requested_checkpoint_without_evaluating(tmp_path):
    out, ctx, _ = _run(_cfg("B1", max_opt_steps=6, save_checkpoint_at=[2, 4, 6], no_eval=True, eval_every=0, log_every=1))
    assert out["metrics"]["opt_steps"] == 6 and out["metrics"]["status"] == "complete"
    assert sorted(k for k in ctx.bytes if k.endswith("checkpoint.pt")) == [f"models/{ctx.run_id}/step{s}/checkpoint.pt" for s in (2, 4, 6)]
    for e in ctx.events:
        if e["action"] == "eval":
            assert e["eval_seconds"] < 1.0 and "attrib_em" not in e and "passkey_em" not in e and "lm_ppl" not in e
    final = ctx.put["final_eval.json"]["final"]
    assert final.get("probe") is None and "attrib_em" not in final and out["metrics"]["B1_source_attribution"] is None
    assert set(out["metrics"]["checkpoint_sha256_step2"]) and out["metrics"]["final_checkpoint_sha256"] == out["metrics"]["checkpoint_sha256_step6"]


def _ck_of(ctx, step):
    return torch.load(io.BytesIO(ctx.bytes[f"models/{ctx.run_id}/step{step}/checkpoint.pt"]), weights_only=False)


def test_b1_parallel_chunk_boundary_resume_is_bit_exact_at_every_checkpoint(tmp_path):
    """The scaled-down pattern of the chunks: 0 -> 6 saving every 2 steps equals chunk 1 (0 -> 4, saving 2, 4) followed by chunk 2 (resume from
    step 4 -> 6, saving 6): every checkpoint, the step losses and the parameters are identical."""
    kw = dict(no_eval=True, eval_every=0, log_every=1)
    full, cf, _ = _run(_cfg("B1", max_opt_steps=6, save_checkpoint_at=[2, 4, 6], **kw))
    p1, c1, _ = _run(_cfg("B1", max_opt_steps=4, save_checkpoint_at=[2, 4], **kw))
    ck = tmp_path / "c4.pt"
    ck.write_bytes(c1.bytes[f"models/{c1.run_id}/step4/checkpoint.pt"])
    p2, c2, _ = _run(_cfg("B1", max_opt_steps=6, save_checkpoint_at=[6], resume_from=str(ck), **kw))
    loss = lambda c: {e["opt_step"]: e["loss"] for e in c.events if e["action"] == "train_step"}
    assert {**loss(c1), **loss(c2)} == loss(cf) and sorted(loss(c2)) == [5, 6]
    for step, other in ((2, c1), (4, c1), (6, c2)):
        a, b = _ck_of(cf, step), _ck_of(other, step)
        assert a["opt_step"] == b["opt_step"] == step and a["micro"] == b["micro"]
        assert all(torch.equal(a["lora"][k], b["lora"][k]) for k in a["lora"])
        for sa, sb in zip(a["opt"]["state"].values(), b["opt"]["state"].values()):
            assert all(torch.equal(sa[k], sb[k]) for k in sa)
