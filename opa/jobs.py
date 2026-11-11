"""Job dispatch inside a Modal container.

A job spec (built by opa.launch from jobs/*.yaml) names an entrypoint "module:function". The function is called as
    fn(args: dict, ctx: JobContext) -> dict | None
and may return {"metrics": {...}}. Outputs go to gs://<bucket>/opa/runs/<run_id>/ through ctx.put*().
Compute time is capped by the Modal timeout the launcher sets from max_usd; API spend is capped by ctx.llm().
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import platform
import time
import traceback

from . import gcs
from .llm import BudgetExceeded, LLMClient


class JobContext:
    def __init__(self, spec: dict):
        self.spec = spec
        self.run_id = spec["run_id"]
        self.prefix = f"gs://{spec['bucket']}/opa/runs/{self.run_id}"
        self.events: list[dict] = []
        self.outputs: list[dict] = []
        self.clients: list[LLMClient] = []
        self.seq = 0

    # ---- event log (protocol event schema v2; one dict per controller action)
    def log(self, event: dict) -> None:
        self.seq += 1
        self.events.append({"run_id": self.run_id, "seq": self.seq, "t_ms": int(time.time() * 1000), **event})

    # ---- outputs
    def put_json(self, rel: str, obj) -> dict:
        info = gcs.put_json(f"{self.prefix}/{rel}", obj)
        self.outputs.append(info)
        return info

    def put_bytes(self, rel: str, data: bytes, content_type: str = "application/octet-stream") -> dict:
        info = gcs.put_bytes(f"{self.prefix}/{rel}", data, content_type)
        self.outputs.append(info)
        return info

    def put_file(self, rel: str, local: str) -> dict:
        info = gcs.put_file(local, f"{self.prefix}/{rel}")
        self.outputs.append(info)
        return info

    # ---- model access with a spend cap shared across all clients of this job
    def llm(self, model: str | None = None, batch: bool = False) -> LLMClient:
        remaining = float(self.spec.get("max_api_usd", 0)) - self.api_usd
        c = LLMClient(model or self.spec["models"]["api_model"], remaining, self.spec["_prices"]["anthropic"], batch=batch, on_call=self.log)
        self.clients.append(c)
        return c

    @property
    def api_usd(self) -> float:
        return sum(c.spent for c in self.clients)

    def flush_events(self) -> dict | None:
        if not self.events:
            return None
        info = gcs.put_jsonl_gz(f"{self.prefix}/events.jsonl.gz", self.events)
        self.outputs.append(info)
        return info

    def manifest(self) -> dict:
        pkgs = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions() if d.metadata.get("Name"))
        return {
            "run_id": self.run_id, "git_sha": self.spec["git_sha"], "lock_sha256": self.spec["lock_sha256"],
            "models": self.spec.get("models", {}), "seeds": {"seed": self.spec.get("seed")},
            "split": self.spec.get("split"), "dataset": self.spec.get("dataset"), "entrypoint": self.spec["entrypoint"],
            "args": self.spec.get("args", {}), "image": {"modal_image_id": os.environ.get("MODAL_IMAGE_ID", ""), "python": platform.python_version(),
                                                       "packages_sha256": gcs.sha256_bytes("\n".join(pkgs).encode()), "packages": pkgs},
            "gpu": self.spec.get("_gpu"), "started_at": self.spec.get("_started_at"),
        }


def dispatch(spec: dict) -> dict:
    t0 = time.time()
    ctx = JobContext(spec)
    status, err, result = "ok", None, {}
    try:
        mod, fn = spec["entrypoint"].split(":")
        result = getattr(importlib.import_module(mod), fn)(spec.get("args", {}), ctx) or {}
    except BudgetExceeded as e:
        status, err = "stopped_budget", str(e)
    except Exception:
        status, err = "failed", traceback.format_exc()
    wall = time.time() - t0
    events = None
    try:
        events = ctx.flush_events()
    except Exception:
        err = (err or "") + "\nevent flush failed:\n" + traceback.format_exc()
    out = {
        "run_id": ctx.run_id, "status": status, "error": err, "wall_seconds": wall,
        "compute_usd": wall * float(spec["_rate_per_second"]), "api_usd": ctx.api_usd,
        "api_usage": [c.usage_totals | {"model": c.model, "calls": c.calls} for c in ctx.clients],
        "metrics": result.get("metrics", {}), "outputs": ctx.outputs, "events": events, "manifest": ctx.manifest(),
    }
    try:
        gcs.put_json(f"{ctx.prefix}/result.json", out)
    except Exception:
        out["error"] = (out["error"] or "") + "\nresult upload failed:\n" + traceback.format_exc()
    return out
