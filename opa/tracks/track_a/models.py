"""Chat-model backends for Track A. Every system talks to the model through `Chat.ask(system, user, max_tokens)`,
which returns the reply text and accounts tokens. Actions are exchanged as one JSON object per reply (a text
protocol, identical for the API model and the open model, so the controller is model-agnostic)."""
from __future__ import annotations

import json
import re
import threading
import time


class Usage:
    def __init__(self):
        self.tokens_in = self.tokens_out = self.calls = self.cached_tokens = 0
        self.n_transport_retries = 0
        self.lock = threading.Lock()

    def add_retry(self):
        with self.lock:
            self.n_transport_retries += 1

    def add(self, tin: int, tout: int, cached: int = 0):
        with self.lock:
            self.tokens_in += int(tin)
            self.tokens_out += int(tout)
            self.cached_tokens += int(cached or 0)
            self.calls += 1


# Transport-level failures that are retried (by class name anywhere in the exception's MRO, so httpx need not be imported)
RETRY_EXCEPTIONS = {"RemoteProtocolError", "ConnectError", "ReadError", "ReadTimeout", "WriteError", "PoolTimeout"}
RETRY_STATUS = {500, 502, 503, 504}
RETRY_DELAYS = (1.0, 3.0, 9.0)                       # seconds before retry 1, 2, 3 (so at most 4 attempts)


def is_transport_error(e: BaseException) -> bool:
    return any(c.__name__ in RETRY_EXCEPTIONS for c in type(e).__mro__)


class ContextOverflow(RuntimeError):
    """The prompt does not fit the served model's window (vLLM 400). Items that hit it are `does_not_fit`."""


_OVERFLOW_MARKERS = ("maximum context length", "context_length_exceeded", "context length", "max_model_len",
                     "maximum input length", "too long")


def is_context_overflow(status: int, body_text: str) -> bool:
    t = (body_text or "").lower()
    return status == 400 and any(m in t for m in _OVERFLOW_MARKERS)


class Chat:
    name = "base"

    def ask(self, system: str, user: str, max_tokens: int = 512, usage: Usage | None = None, log=None) -> str:
        raise NotImplementedError


class ApiChat(Chat):
    """Refused in v3 (PLAN_V3 Section 3, project rule 8): every experiment call goes to an open model.
    Kept only so old imports fail with a clear message instead of an ImportError."""

    def __init__(self, client=None):
        raise RuntimeError("ApiChat is disabled in v3: experiment calls go to an open model "
                           "(model: qwen | gemma | {kind: open_local|open}); the Anthropic API is not used")


class OpenAIChat(Chat):
    """Open model (Qwen3-4B-Instruct-2507, Gemma 4 E4B) behind an OpenAI-compatible endpoint (vLLM). Temperature 0.
    `fold_system`: for chat templates without a system role (Gemma) the system prompt goes in front of the user turn."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: float = 600.0, fold_system: bool = False,
                 seed: int = 0, provenance: dict | None = None):
        self._conn = (base_url.rstrip("/"), timeout, {"Authorization": f"Bearer {api_key}"})
        self._http = None
        self.name, self.fold_system, self.seed = model, fold_system, int(seed)
        self.provenance = provenance or {}

    @property
    def http(self):                                      # created on first call, so building the object needs no httpx
        if self._http is None:
            import httpx
            self._http = httpx.Client(base_url=self._conn[0], timeout=self._conn[1], headers=self._conn[2])
        return self._http

    def messages(self, system: str, user: str) -> list[dict]:
        if self.fold_system:
            return [{"role": "user", "content": system + "\n\n" + user if system else user}]
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # A fixed probe, well over 64 tokens (vLLM caches full 16-token blocks), that starts with a marker no real prompt shares.
    PROBE_SYSTEM = "PROBE-7f3a9c cache probe, ignore."
    PROBE_USER = " ".join(f"probe{i:03d}-alpha{(i * 7) % 97:02d}" for i in range(120))

    _sleep = staticmethod(time.sleep)

    def _post(self, body: dict, usage=None, log=None):
        """POST the chat request. Transport failures (RETRY_EXCEPTIONS) and 502/503/504 are retried up to 3 times after
        1 s, 3 s, 9 s; 4xx (including 400 / 429) and other statuses are returned at once for the caller to judge. Each retry is
        logged as `llm_transport_retry` and counted in usage.n_transport_retries. After the last attempt the exception
        is raised (or the 5xx response returned), which the caller records as an error."""
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                r = self.http.post("/v1/chat/completions", json=body)
                err, status = None, r.status_code
            except Exception as e:
                if not is_transport_error(e):
                    raise
                r, err, status = None, e, None
            if err is None and status not in RETRY_STATUS:
                return r
            if attempt == len(RETRY_DELAYS):
                if err is not None:
                    raise err
                return r
            delay = RETRY_DELAYS[attempt]
            if usage is not None:
                usage.add_retry()
            if log:
                log({"action": "llm_transport_retry", "model": self.name, "attempt": attempt + 1, "sleep_s": delay,
                     "error": type(err).__name__ if err is not None else None, "status": status})
            self._sleep(delay)

    def probe_cache(self) -> dict:
        """Send the fixed probe (max_tokens 1, temperature 0, not counted in any Usage) and report the server's
        prompt_tokens and prompt_tokens_details.cached_tokens for it. Needs --enable-prompt-tokens-details."""
        body = {"model": self.name, "temperature": 0, "seed": self.seed, "max_tokens": 1,
                "messages": self.messages(self.PROBE_SYSTEM, self.PROBE_USER)}
        r = self.http.post("/v1/chat/completions", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"cache probe failed with {r.status_code}: {str(r.text)[:300]}")
        u = r.json().get("usage", {})
        return {"prompt_tokens": int(u.get("prompt_tokens", 0)),
                "cached_tokens": int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)}

    def reset_prefix_cache(self, attempts: int = 5, sleep_s: float = 1.0, _sleep=time.sleep) -> dict:
        """POST /reset_prefix_cache (vLLM 0.32.0: served only with VLLM_SERVER_DEV_MODE=1, route in
        vllm/entrypoints/serve/dev/cache/api_router.py). The route answers 200 with {"success": bool}; success is false
        while KV blocks are still held, so retry up to `attempts` times, `sleep_s` apart, and raise if it never
        succeeds. Returns {"success": True, "attempts": n}."""
        last = None
        for n in range(1, attempts + 1):
            r = self.http.post("/reset_prefix_cache", json=None)
            if r.status_code != 200:
                raise RuntimeError(f"vLLM /reset_prefix_cache failed with {r.status_code}: {str(r.text)[:300]}; the server must "
                                   "run with VLLM_SERVER_DEV_MODE=1 (otherwise restart it between systems)")
            try:
                last = json.loads(r.text).get("success")
            except Exception:
                last = None
            if last is True:
                return {"success": True, "attempts": n}
            if n < attempts:
                _sleep(sleep_s)
        raise RuntimeError(f"vLLM /reset_prefix_cache did not succeed after {attempts} attempts (last success={last!r}); "
                           "blocks are still held")

    def ask(self, system, user, max_tokens=512, usage=None, log=None):
        msgs = self.messages(system, user)
        body = {"model": self.name, "temperature": 0, "seed": self.seed, "max_tokens": max_tokens, "messages": msgs}
        r = self._post(body, usage, log)
        if r.status_code == 400 and is_context_overflow(400, r.text):
            raise ContextOverflow(r.text[:500])
        r.raise_for_status()
        j = r.json()
        u = j.get("usage", {})
        cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)   # needs --enable-prompt-tokens-details
        if usage is not None:
            usage.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), cached)
        ch = j["choices"][0]
        text = ch["message"]["content"]
        if log:                                            # raw model I/O (project rule: store raw inputs and outputs)
            log({"action": "llm_call", "model": self.name, "messages": msgs, "max_tokens": max_tokens, "reply": text,
                 "usage": {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                       "cached_tokens": cached},
                 "finish_reason": ch.get("finish_reason"), "seed": self.seed})
        return text


class FakeChat(Chat):
    """Scripted model for offline tests: `policy(system, user) -> reply text`."""

    name = "fake"

    def __init__(self, policy):
        self.policy = policy

    def ask(self, system, user, max_tokens=512, usage=None, log=None):
        out = self.policy(system, user)
        if usage is not None:
            usage.add(len(system + user) // 4, len(out) // 4)
        return out


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_json(text: str) -> dict | None:
    """First JSON object in a reply (fenced or bare). None if there is none."""
    m = _FENCE.search(text)
    cands = [m.group(1)] if m else []
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                cands.append(text[start:i + 1])
                break
    for c in cands:
        try:
            v = json.loads(c)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            continue
    return None


MISSING = object()


def salvage_answer(text: str):
    """The value of the top-level "answer" member of a reply that is not valid JSON as a whole (unescaped quotes in a
    later field, a runaway list cut off by max_tokens, ...). The object is read member by member from its first "{"
    with strict JSON value parsing (json.JSONDecoder.raw_decode); the answer counts only when it is a complete JSON
    value followed by a delimiter (`,` or `}`): `57 bar`, a number cut at the end of the text, `12abc` and `nullable`
    do not. Parsing stops at the first malformed member (a bad member after the answer does not matter, one before
    it means refusal). More than one top-level "answer" member, nested "answer" keys and keys such as "answer_found"
    never match. Returns MISSING when nothing can be salvaged (the caller retries)."""
    s = text or ""
    i = s.find("{")
    if i < 0:
        return MISSING
    dec = json.JSONDecoder()
    found, n = MISSING, 0

    def ws(j):
        while j < len(s) and s[j] in " \t\r\n":
            j += 1
        return j

    i += 1
    while True:
        i = ws(i)
        if i >= len(s) or s[i] != '"':
            break
        try:
            key, i = dec.raw_decode(s, i)
        except ValueError:
            break
        i = ws(i)
        if i >= len(s) or s[i] != ":":
            break
        i = ws(i + 1)
        try:
            val, j = dec.raw_decode(s, i)
        except ValueError:
            break
        k = ws(j)
        if k >= len(s) or s[k] not in ",}":
            break
        if key == "answer":
            n += 1
            if n > 1:
                return MISSING
            found = val
        if s[k] == "}":
            break
        i = k + 1
    return found


def salvage_members(text: str, outer_key: str) -> dict:
    """The complete members of the top-level object member `outer_key` of a (possibly truncated) JSON reply, in order:
    {"entities": {"0": [...], "1": [...], "2": ["Ste` -> the members "0" and "1". Member by member with strict JSON value
    parsing (raw_decode); a list member that is complete counts even if the text ends right after it; reading stops at
    the first malformed or cut-off member. Returns {} when nothing can be read."""
    s = text or ""
    dec = json.JSONDecoder()

    def ws(j):
        while j < len(s) and s[j] in " \t\r\n":
            j += 1
        return j

    def walk(i, want):
        """Yield (key, value) for the members of the object whose '{' precedes index i; for key == want yield the
        position of its '{' instead of parsing it."""
        while True:
            i = ws(i)
            if i >= len(s) or s[i] != '"':
                return
            try:
                key, i = dec.raw_decode(s, i)
            except ValueError:
                return
            i = ws(i)
            if i >= len(s) or s[i] != ":":
                return
            i = ws(i + 1)
            if want is not None and key == want and i < len(s) and s[i] == "{":
                yield key, i + 1
                return
            try:
                val, j = dec.raw_decode(s, i)
            except ValueError:
                return
            k = ws(j)
            if k < len(s) and s[k] not in ",}":
                return
            if k >= len(s) and not isinstance(val, list):
                return
            yield key, val
            if k >= len(s) or s[k] == "}":
                return
            i = k + 1

    start = s.find("{")
    if start < 0:
        return {}
    for key, pos in walk(start + 1, outer_key):
        if key == outer_key and isinstance(pos, int) and not isinstance(pos, bool):
            return {k: v for k, v in walk(pos, None) if isinstance(v, list)}
    return {}


_ESC = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _unescape_partial(raw: str) -> str:
    """JSON-unescape the complete escape sequences of a string cut off mid-way; a dangling partial escape is dropped."""
    out, i = [], 0
    while i < len(raw):
        ch = raw[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= len(raw):
            break
        nxt = raw[i + 1]
        if nxt == "u":
            hexs = raw[i + 2:i + 6]
            if len(hexs) < 4:
                break
            try:
                out.append(chr(int(hexs, 16)))
            except ValueError:
                pass
            i += 6
        else:
            out.append(_ESC.get(nxt, nxt))
            i += 2
    return "".join(out)


def _strip_closers(part: str) -> str:
    """Remove a trailing code fence, closing brace and closing quote (in that order) left on a cut-off value."""
    p = part.rstrip()
    if p.endswith("```"):
        p = p[:-3].rstrip()
    if p.endswith("}"):
        p = p[:-1].rstrip()
    if p.endswith('"'):
        p = p[:-1]
    return p


def salvage_string_field(text: str, key: str):
    """The string value of the top-level member `key` of a JSON reply, also when the reply was cut off inside that
    string (max_tokens). Returns (value, complete) or MISSING. Earlier members must be well-formed (strict walk, like
    salvage_answer); a complete string wins, a cut-off one yields the text read so far. Lenient where a model's JSON is
    sloppy: literal newlines inside the string are accepted (strict=False, this function only), a trailing closing
    quote / brace / code fence is stripped from a cut-off value, and a string with unescaped inner quotes
    ({"gist": "He said "hi" then left."}) is read up to the last quote before the closing brace."""
    s = text or ""
    dec = json.JSONDecoder(strict=False)
    i = s.find("{")
    if i < 0:
        return MISSING

    def ws(j):
        while j < len(s) and s[j] in " \t\r\n":
            j += 1
        return j

    i += 1
    while True:
        i = ws(i)
        if i >= len(s) or s[i] != '"':
            return MISSING
        try:
            k, i = dec.raw_decode(s, i)
        except ValueError:
            return MISSING
        i = ws(i)
        if i >= len(s) or s[i] != ":":
            return MISSING
        i = ws(i + 1)
        if k == key:
            if i >= len(s) or s[i] != '"':
                return MISSING
            try:
                v, j = dec.raw_decode(s, i)
                k2 = ws(j)
                if k2 >= len(s) or s[k2] in ",}":
                    return v, True
                # more text follows the first closing quote: unescaped quotes inside the value
                close = s.rfind("}")
                if close > j:
                    q = s.rfind('"', i + 1, close)
                    if q > i:
                        part = _unescape_partial(s[i + 1:q])
                        return (part, False) if part.strip() else MISSING
                part = _strip_closers(_unescape_partial(s[i + 1:]))
                return (part, False) if part.strip() else MISSING
            except ValueError:
                part = _strip_closers(_unescape_partial(s[i + 1:]))
                return (part, False) if part.strip() else MISSING
        try:
            _, j = dec.raw_decode(s, i)
        except ValueError:
            return MISSING
        j = ws(j)
        if j >= len(s) or s[j] != ",":
            return MISSING
        i = j + 1


def answer_text(v) -> str:
    """The prediction string for an answer value: None / missing is empty, everything else (including 0 and false) is str(v)."""
    return "" if v is None or v is MISSING else str(v)


def ask_json(chat: Chat, system: str, user: str, max_tokens: int, usage: Usage, log=None, retries: int = 1) -> dict:
    """Ask for one JSON object; on a parse failure re-ask once with a reminder. Returns {} if still unparseable."""
    text = chat.ask(system, user, max_tokens, usage, log)
    obj = parse_json(text)
    tries = 0
    while obj is None and tries < retries:
        tries += 1
        text = chat.ask(system, user + "\n\nYour previous reply was not a single valid JSON object. Reply with JSON only.",
                        max_tokens, usage, log)
        obj = parse_json(text)
    if obj is None and log:
        log({"action": "parse_error", "reply_head": text[:200]})
    return obj or {}
