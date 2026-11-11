"""Modal app for the OPA experiments (version 3). Deploy with `python -m opa.launch deploy` (never `modal deploy` directly).

Generic job runners. The launcher picks GPU type, CPU, memory and a hard timeout per job with
`Function.with_options(...)`; the timeout is derived from the job's max_usd, so a runaway job is killed by
Modal before it can spend more than its ceiling.

v3 (PLAN_V3.md Sections 3-4): open models only, no external model API anywhere, GPUs limited to L4 (default) and T4 (small).
  run_job_cpu    CPU container
  run_job_gpu    L4, training/attention image (torch + transformers + peft): Track B, LoRA, embeddings
  run_job_t4     T4, same image: smoke tests and small jobs
  run_job_vllm   L4, inference image (vLLM): Track A with Qwen3-4B-Instruct-2507 and Gemma 4 E4B

Every function has single_use_containers=True (also on CPU): each job gets a fresh container (no warm vLLM server shared between jobs).

Secrets (create once, see README):
  opa-gcp   GCP_SA_JSON   service-account key JSON with object create + view on the bucket (required)
There is no Hugging Face secret: Modal's Secret.from_name has no "optional" mode (a missing secret fails the container),
and every v3 model is ungated, so no Hugging Face token is needed.
"""
import modal

APP_NAME = "opa-experiments"
app = modal.App(APP_NAME, tags={"project": "opa"})

hf_cache = modal.Volume.from_name("opa-hf-cache", create_if_missing=True)

SECRETS = [
    modal.Secret.from_name("opa-gcp", required_keys=["GCP_SA_JSON"]),
]

# Pin versions in requirements-modal-*.txt once P0 settles them; the job manifest records the installed set.
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "scipy", "pyyaml", "google-cloud-storage", "rank-bm25", "tiktoken", "httpx")
    # Track A ingest embeds origins with a small encoder (bge-small) on CPU
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("opa")
)

# Inference image for Track A: the job starts a local vLLM server inside its own L4 container, so the whole cost is
# this job's GPU time, capped by the launcher's timeout. vLLM 0.32.0 (released 2026-10-18) is the newest release that is
# at least two weeks old (0.33.0 came out 2026-11-02); it requires transformers<5.20.0, so transformers is pinned to
# 5.19.0 (released 2026-10-06), the newest allowed. Both Qwen3-4B-Instruct-2507 and Gemma 4 E4B are served by this pair.
VLLM_VERSION = "0.32.0"
VLLM_TRANSFORMERS = "5.19.0"
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(f"vllm=={VLLM_VERSION}", f"transformers=={VLLM_TRANSFORMERS}", "numpy", "pandas", "scipy", "pyyaml",
                 "google-cloud-storage", "httpx")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("opa")
)

# Training / attention image for Track B and C (separate from vLLM, so transformers can be newer).
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.22.2", "accelerate", "peft", "datasets", "numpy", "pandas", "scipy",
                 "pyyaml", "google-cloud-storage", "sentencepiece", "safetensors")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("opa")
)


@app.function(image=cpu_image, secrets=SECRETS, cpu=1.0, memory=4096, timeout=3600, max_containers=32,
              single_use_containers=True, volumes={"/cache/hf": hf_cache})
def run_job_cpu(spec: dict) -> dict:
    from opa.jobs import dispatch
    return dispatch(spec)


@app.function(image=vllm_image, gpu="L4", secrets=SECRETS, volumes={"/cache/hf": hf_cache}, timeout=3600, max_containers=8, single_use_containers=True)
def run_job_vllm(spec: dict) -> dict:
    from opa.jobs import dispatch
    try:
        return dispatch(spec)
    finally:
        hf_cache.commit()


@app.function(image=gpu_image, gpu="L4", secrets=SECRETS, volumes={"/cache/hf": hf_cache}, timeout=3600, max_containers=4, single_use_containers=True)
def run_job_gpu(spec: dict) -> dict:
    from opa.jobs import dispatch
    try:
        return dispatch(spec)
    finally:
        hf_cache.commit()


@app.function(image=gpu_image, gpu="T4", secrets=SECRETS, volumes={"/cache/hf": hf_cache}, timeout=3600, max_containers=4, single_use_containers=True)
def run_job_t4(spec: dict) -> dict:
    from opa.jobs import dispatch
    try:
        return dispatch(spec)
    finally:
        hf_cache.commit()
