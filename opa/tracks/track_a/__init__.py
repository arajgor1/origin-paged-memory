"""Track A: agent-level OPA (protocol Section 3). Job entrypoints: opa.tracks.track_a:run_system, :build_skeletons and :embed_items, :embedder_check."""
from .run import build_skeletons, embed_items, embedder_check, run_system  # noqa: F401
from .thrash import run_thrash  # noqa: F401,E402
