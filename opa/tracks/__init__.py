"""Experiment code lives here, one module per track (track_a.py, track_a2.py, track_b.py, track_c.py, generators.py, ...).
Each job entrypoint has the signature  fn(args: dict, ctx: opa.jobs.JobContext) -> dict | None
and returns {"metrics": {...}}. Write outputs with ctx.put_json / ctx.put_bytes / ctx.put_file, log every controller
action with ctx.log({...}) using the protocol's event schema v2, and call models only through ctx.llm()."""
