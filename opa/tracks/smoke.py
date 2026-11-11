"""G0 smoke test: proves Modal, GCS, the event log, the cost accounting and (optionally) the API path work end to end."""
import platform
import time


def main(args: dict, ctx) -> dict:
    t0 = time.time()
    ctx.log({"action": "smoke_start", "python": platform.python_version()})
    info = ctx.put_json("smoke/hello.json", {"hello": "opa", "run_id": ctx.run_id})
    metrics = {"gcs_write_ok": True, "gcs_sha256": info["sha256"]}
    if args.get("gpu_check"):
        import torch
        metrics["cuda_available"] = bool(torch.cuda.is_available())
        metrics["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if args.get("api_check"):
        client = ctx.llm()
        r = client.messages(max_tokens=8, messages=[{"role": "user", "content": "Reply with the single word: ok"}])
        metrics["api_reply"] = r.content[0].text.strip()[:20]
        metrics["api_cost_usd"] = round(client.spent, 6)
    ctx.log({"action": "smoke_end", "seconds": round(time.time() - t0, 3)})
    return {"metrics": metrics}
