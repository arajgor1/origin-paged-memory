"""v3 Modal app / launcher checks. The app file is inspected as text (never imported), so nothing here needs Modal."""
import re
from pathlib import Path

import pytest
import yaml

from opa import budget, launch
from opa.llm import LLMClient

ROOT = Path(__file__).resolve().parents[1]
APP_TEXT = (ROOT / "opa" / "modal_app.py").read_text()


def job(**compute):
    c = {"kind": "gpu", "gpu": "L4", "cpu": 2, "memory_gib": 16}
    c.update(compute)
    return {"name": "x", "envelope": "E0", "track": "A", "hypotheses": ["H1"], "split": "dev", "seed": 1, "entrypoint": "m:f",
            "compute": c, "estimate_usd": 0.5, "max_usd": 1.0, "max_api_usd": 0}


def test_valid_specs():
    assert launch.validate(job()) == []
    assert launch.validate(job(gpu="T4")) == []
    assert launch.validate(job(image="vllm")) == []
    assert launch.validate(job(gpu=None)) == []                       # default L4
    assert launch.validate(job(kind="cpu", gpu=None)) == []


@pytest.mark.parametrize("gpu", ["A100-80GB", "A100-40GB", "H100", "L40S", "A10", "B200"])
def test_big_gpus_refused(gpu):
    errs = launch.validate(job(gpu=gpu))
    assert errs and any("compute.gpu" in e for e in errs)


def test_vllm_needs_l4_gpu():
    assert launch.validate(job(image="vllm", gpu="T4"))
    assert launch.validate(job(kind="cpu", gpu=None, image="vllm"))
    assert launch.validate(job(image="other"))


def test_api_budget_refused():
    j = job()
    assert launch.validate({**j, "max_api_usd": 0.5})
    assert launch.validate({**j, "max_api_usd": 4})
    assert launch.validate({**j, "max_api_usd": "abc"})
    missing = dict(j)
    del missing["max_api_usd"]
    assert any("max_api_usd" in e for e in launch.validate(missing))


@pytest.mark.parametrize("compute,expected", [
    ({"kind": "cpu"}, "run_job_cpu"),
    ({"kind": "gpu", "gpu": "L4"}, "run_job_gpu"),
    ({"kind": "gpu"}, "run_job_gpu"),
    ({"kind": "gpu", "gpu": "T4"}, "run_job_t4"),
    ({"kind": "gpu", "gpu": "L4", "image": "vllm"}, "run_job_vllm"),
    ({"kind": "gpu", "gpu": "L4", "image": "default"}, "run_job_gpu"),
])
def test_routing(compute, expected):
    assert launch.function_name(compute) == expected


def test_rates_use_cheap_gpus():
    prices = budget.load_prices()
    l4 = launch.rate_per_second({"kind": "gpu", "gpu": "L4", "cpu": 2, "memory_gib": 16}, prices)
    t4 = launch.rate_per_second({"kind": "gpu", "gpu": "T4", "cpu": 2, "memory_gib": 16}, prices)
    default = launch.rate_per_second({"kind": "gpu", "cpu": 2, "memory_gib": 16}, prices)
    assert t4 < l4 and default == l4


def test_modal_app_text_v3():
    assert "opa-anthropic" not in APP_TEXT
    assert not re.search(r"anthropic", APP_TEXT.replace("Anthropic", ""), re.I)      # no pip package or secret of that name
    assert "ANTHROPIC" not in APP_TEXT
    pip_args = " ".join(re.findall(r"pip_install\((.*?)\)\n", APP_TEXT, re.S))
    assert "anthropic" not in pip_args.lower()
    for big in ("A100", "H100", "L40S", "B200", "A10G"):
        assert big not in APP_TEXT
    assert APP_TEXT.count('gpu="L4"') == 2 and APP_TEXT.count('gpu="T4"') == 1
    for fn in ("run_job_cpu", "run_job_gpu", "run_job_t4", "run_job_vllm"):
        assert f"def {fn}(" in APP_TEXT
    assert 'tags={"project": "opa"}' in APP_TEXT and 'APP_NAME = "opa-experiments"' in APP_TEXT
    assert re.search(r'VLLM_VERSION = "\d+\.\d+\.\d+"', APP_TEXT)
    assert '"opa-gcp"' in APP_TEXT and '"opa-hf"' not in APP_TEXT


def test_llm_anthropic_client_unreachable():
    c = LLMClient("m", 1.0, {})
    with pytest.raises(RuntimeError, match="no Anthropic API in v3"):
        c.messages(messages=[])
    with pytest.raises(RuntimeError, match="no Anthropic API in v3"):
        c._api()


@pytest.mark.parametrize("name", ["TEMPLATE", "smoke_cpu", "smoke_gpu"])
def test_shipped_specs_validate(name):
    text = (ROOT / "jobs" / f"{name}.yaml").read_text()
    j = yaml.safe_load(text)
    if "gs://BUCKET" in text:                                         # the template must be refused until filled
        assert launch.validate(j)
        j = yaml.safe_load(text.replace("gs://BUCKET", "gs://real-bucket"))
    assert launch.validate(j) == []
    assert j["max_api_usd"] == 0 and j["envelope"] in ("E0", "E1")
    assert j["compute"].get("gpu") in (None, "L4", "T4")
    assert j["max_usd"] >= j["estimate_usd"]


def test_smoke_gpu_is_t4():
    j = yaml.safe_load((ROOT / "jobs" / "smoke_gpu.yaml").read_text())
    assert j["compute"]["gpu"] == "T4" and j["envelope"] == "E0"


def test_timeout_priced_at_limits():
    prices = budget.load_prices()["modal"]
    comp = {"kind": "gpu", "gpu": "L4", "cpu": 2, "memory_gib": 16}
    r = launch.resources(comp)
    assert (r["cpu"], r["cpu_limit"], r["memory_gib"], r["memory_gib_limit"]) == (2.0, 4.0, 16.0, 32.0)
    expect = prices["gpu_per_second"]["L4"] + 4.0 * prices["cpu_core_per_second"] + 32.0 * prices["memory_gib_per_second"]
    assert abs(launch.rate_per_second(comp, budget.load_prices()) - expect) < 1e-12
    over = launch.resources({**comp, "cpu_limit": 3, "memory_gib_limit": 20})
    assert over["cpu_limit"] == 3.0 and over["memory_gib_limit"] == 20.0
    assert launch.resources({**comp, "cpu_limit": 1})["cpu_limit"] == 2.0     # never below the request


def test_functions_one_job_per_container():
    for fn in ("run_job_cpu", "run_job_vllm", "run_job_gpu", "run_job_t4"):
        i = APP_TEXT.index(f"def {fn}(")
        decorator = APP_TEXT[APP_TEXT.rindex("@app.function", 0, i):i]
        assert "single_use_containers=True" in decorator, fn
        assert "max_inputs" not in decorator, fn
    assert APP_TEXT.count("single_use_containers=True)") + APP_TEXT.count("single_use_containers=True,") == 4


def test_bucket_placeholder_refused():
    j = job()
    assert any("BUCKET" in e for e in launch.validate({**j, "args": {"items_uri": "gs://BUCKET/opa/data/x.gz"}}))
    assert launch.validate({**j, "args": {"items_uri": "gs://YOUR_BUCKET/opa/data/x.gz"}}) == []


def test_placeholder_refused():
    j = job()
    assert launch.validate({**j, "args": {"embed_uri": "gs://b/opa/runs/EMBED_RUN_ID/x.npy"}})
    assert any("SKELETON_RUN_ID" in e for e in launch.validate({**j, "args": {"a": ["SKELETON_RUN_ID"]}}))
    assert launch.validate({**j, "args": {"ok": "B-S2-lbmhqa-run_id-ok", "n": 3}}) == []


def test_bucket_forms():
    from opa import gcs
    for raw in ("YOUR_BUCKET", "gs://YOUR_BUCKET", "gs://YOUR_BUCKET/", " YOUR_BUCKET/ "):
        assert gcs.normalize_bucket(raw) == "YOUR_BUCKET"


def test_launch_cfg_strips_bucket(monkeypatch, tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "project.yaml").write_text("gcp: {bucket: 'gs://YOUR_BUCKET/'}")
    monkeypatch.setattr(launch, "PROJECT_DIR", tmp_path)
    assert launch.cfg()["gcp"]["bucket"] == "YOUR_BUCKET"


def test_deploy_uses_python_dash_m_modal(monkeypatch):
    import sys
    cmd = launch.deploy_command()
    assert cmd[:3] == [sys.executable, "-m", "modal"] and cmd[3] == "deploy"
    seen = {}
    monkeypatch.setattr(budget, "approved_envelopes", lambda a: ["E0"])
    monkeypatch.setattr(budget, "load_approvals", lambda: {})
    monkeypatch.setattr(launch.subprocess, "run", lambda c, **kw: seen.setdefault("cmd", c) and type("R", (), {"returncode": 0})())
    assert launch.cmd_deploy() == 0 and seen["cmd"][:3] == [sys.executable, "-m", "modal"]
    assert 'subprocess.run(["modal"' not in (ROOT / "opa" / "launch.py").read_text()


def _spec(split, args):
    return {"name": "x", "envelope": "E2", "track": "A", "hypotheses": ["H1"], "split": split, "seed": 0, "entrypoint": "opa.tracks.track_a:run_system",
            "args": args, "compute": {"kind": "gpu", "gpu": "L4", "image": "vllm", "cpu": 2.0, "memory_gib": 16}, "estimate_usd": 1.0, "max_usd": 2.0, "max_api_usd": 0}


def test_dev_spec_with_a_test_uri_is_refused_everywhere_in_args():
    from opa import launch
    uri = "gs://{bucket}/opa/data/lbmhqa/test/v2/late-h2-32k-a0/items.jsonl.gz"
    for args in ({"items_uri": uri}, {"items_uri": [{"uri": uri, "n_items": 5}]}, {"embedder": "precomputed:" + uri.replace("items.jsonl.gz", "e.npz")},
                 {"skeleton_llm_uri": "gs://{bucket}/opa/data/x/test/v2/sk"}, {"model": {"extra": ["--x", uri]}}):
        errs = launch.validate(_spec("dev", args))
        assert any("references test-split data" in e for e in errs), (args, errs)
    # dev paths, "tests/" and a test-split spec are fine
    ok = "gs://{bucket}/opa/data/lbmhqa/dev/v2/late-h2-32k-a0/items.jsonl.gz"
    assert launch.validate(_spec("dev", {"items_uri": ok, "x": "tests/foo/test_a.py", "y": "latest/ok"})) == []
    assert launch.validate(_spec("test", {"items_uri": [{"uri": uri, "n_items": 5}]})) == []                # unchanged: the tag check is in cmd_run


def test_cmd_run_still_applies_the_tag_check_to_test_specs():
    from opa import launch
    import inspect
    src = inspect.getsource(launch.cmd_run)
    assert 'job["split"] == "test" and not prereg_tagged()' in src


def test_max_containers_vllm_8_others_4():
    """Owner instruction 2026-11-08: the vLLM function (Track A) may run 8 containers; the Track B GPU functions stay at 4."""
    blocks = {re.search(r"def (\w+)\(", b).group(1): int(re.search(r"max_containers=(\d+)", b).group(1))
              for b in APP_TEXT.split("@app.function")[1:]}
    mc = blocks.__getitem__
    assert mc("run_job_vllm") == 8 and mc("run_job_gpu") == 4 and mc("run_job_t4") == 4 and mc("run_job_cpu") == 32
