"""Anthropic API wrapper with a hard spend cap (legacy; the client is UNREACHABLE in v3).

v3 experiments use open models on Modal only (PLAN_V3.md): building the Anthropic client raises. The class and the
cap accounting stay so price helpers and old imports keep working (tests inject a fake `_client`).

The cap is enforced before each call (refuse if already at the cap) and after it (raise once crossed),
so a job overshoots its max_api_usd by at most one call.
"""
from __future__ import annotations

import time

from .budget import anthropic_cost


class BudgetExceeded(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str, max_usd: float, prices: dict, batch: bool = False, on_call=None):
        self.model, self.max_usd, self.prices, self.batch = model, float(max_usd), prices, batch
        self.spent = 0.0
        self.calls = 0
        self.usage_totals = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        self.on_call = on_call          # callback(event_dict) for the event log
        self._client = None
        import threading
        self._lock = threading.Lock()   # Track A runs items in parallel threads that share one client and one cap

    def _api(self):
        if self._client is None:
            raise RuntimeError("no Anthropic API in v3 experiments, PLAN_V3.md: the Anthropic client is disabled")
        return self._client

    def account(self, usage) -> float:
        u = {k: int(getattr(usage, k, 0) or (usage.get(k, 0) if isinstance(usage, dict) else 0) or 0) for k in self.usage_totals}
        cost = anthropic_cost(self.model, u, batch=self.batch, prices=self.prices)
        with self._lock:
            for k, v in u.items():
                self.usage_totals[k] += v
            self.spent += cost
            self.calls += 1
        return cost

    def messages(self, **kwargs):
        if self.spent >= self.max_usd:
            raise BudgetExceeded(f"API cap ${self.max_usd:.2f} reached after {self.calls} calls")
        kwargs.setdefault("temperature", 0)
        t0 = time.time()
        resp = self._api().messages.create(model=self.model, **kwargs)
        cost = self.account(resp.usage)
        if self.on_call:
            self.on_call({"action": "llm_call", "model": self.model, "latency_ms": int(1000 * (time.time() - t0)),
                          "usage": {k: getattr(resp.usage, k, None) for k in self.usage_totals}, "cost_usd": round(cost, 6)})
        if self.spent > self.max_usd:
            raise BudgetExceeded(f"API cap ${self.max_usd:.2f} crossed (spent ${self.spent:.4f})")
        return resp
