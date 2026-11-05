"""Provider-agnostic LLM client with a disk cache, a hard call cap and a throttle.

Cost controls, in the order they apply:
1. Disk cache: a prompt is only ever sent once. Reruns and duplicate prompts are free.
2. Hard cap: real attempts are counted on disk, and the run stops at MAX_API_CALLS.
3. Throttle: calls are spaced out to stay inside provider rate limits.
"""
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass

from .config import (CACHE_DIR, DEFAULT_MODELS, MAX_API_CALLS, MIN_CALL_INTERVAL_S,
                     MODEL_OVERRIDE, PROVIDER)
from .prompts import REFUSAL


class BudgetExceeded(RuntimeError):
    """Raised instead of making a call that would go over the hard cap."""


class QuotaExhausted(RuntimeError):
    """Raised when the provider keeps refusing calls. Finished work stays in the cache."""


@dataclass
class Reply:
    text: str
    latency_s: float       # time of the original API call, kept on cache hits
    input_tokens: int
    output_tokens: int
    cached: bool


class LLM:
    def __init__(self, provider: str | None = None, cache_dir=None, max_calls: int = MAX_API_CALLS):
        self.provider = (provider or PROVIDER).lower()
        self.model = MODEL_OVERRIDE if MODEL_OVERRIDE and self.provider != "mock" \
            else DEFAULT_MODELS[self.provider]
        self.cache_dir = cache_dir or CACHE_DIR / "llm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage_path = self.cache_dir / "usage.json"
        self.max_calls = max_calls
        self._last_call = 0.0

    # ---- cache and budget ----
    def _cache_path(self, prompt: str):
        key = hashlib.sha256(f"{self.provider}|{self.model}|{prompt}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def is_cached(self, prompt: str) -> bool:
        return self.provider == "mock" or self._cache_path(prompt).exists()

    def calls_made(self) -> int:
        if not self.usage_path.exists():
            return 0
        return json.loads(self.usage_path.read_text())["attempts"]

    def _count_attempt(self) -> None:
        attempts = self.calls_made()
        if attempts >= self.max_calls:
            raise BudgetExceeded(f"Hard cap of {self.max_calls} API attempts reached.")
        self.usage_path.write_text(json.dumps({"attempts": attempts + 1}))

    def _throttle(self) -> None:
        wait = self._last_call + MIN_CALL_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    # ---- public API ----
    def generate(self, prompt: str, json_mode: bool = False) -> Reply:
        if self.provider == "mock":
            return Reply(self._mock(prompt), 0.0, 0, 0, cached=False)
        path = self._cache_path(prompt)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Reply(data["response"], data["latency_s"], data["input_tokens"],
                         data["output_tokens"], cached=True)

        text, latency, tokens_in, tokens_out = self._call_with_retry(prompt, json_mode)
        path.write_text(json.dumps({"provider": self.provider, "model": self.model,
                                    "response": text, "latency_s": latency,
                                    "input_tokens": tokens_in, "output_tokens": tokens_out}),
                        encoding="utf-8")
        return Reply(text, latency, tokens_in, tokens_out, cached=False)

    def _call_with_retry(self, prompt: str, json_mode: bool, retries: int = 2):
        """Retry short rate-limit or overload errors twice, then stop the whole run."""
        for attempt in range(retries + 1):
            self._count_attempt()
            self._throttle()
            start = time.perf_counter()
            try:
                text, tokens_in, tokens_out = getattr(self, f"_{self.provider}")(prompt, json_mode)
            except Exception as err:  # SDKs raise different types; classify by message
                message = str(err)
                transient = re.search(r"\b(429|500|502|503|504)\b|RESOURCE_EXHAUSTED|UNAVAILABLE",
                                      message)
                if not transient:
                    raise
                if attempt == retries:
                    raise QuotaExhausted(message[:300]) from err
                time.sleep(30 * (attempt + 1))
                continue
            if not text:
                raise RuntimeError("Provider returned an empty response.")
            return text, round(time.perf_counter() - start, 3), tokens_in, tokens_out

    # ---- providers (SDKs imported lazily so only the chosen one must be installed) ----
    def _gemini(self, prompt: str, json_mode: bool):
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        config = {"temperature": 0}
        if json_mode:
            config["response_mime_type"] = "application/json"
        resp = client.models.generate_content(model=self.model, contents=prompt, config=config)
        usage = resp.usage_metadata
        return ((resp.text or "").strip(), usage.prompt_token_count or 0,
                usage.candidates_token_count or 0)

    def _anthropic(self, prompt: str, json_mode: bool):
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        resp = client.messages.create(model=self.model, max_tokens=2048,
                                      messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    def _openai(self, prompt: str, json_mode: bool):
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], **kwargs)
        return (resp.choices[0].message.content.strip(), resp.usage.prompt_tokens,
                resp.usage.completion_tokens)

    @staticmethod
    def _mock(prompt: str) -> str:
        """Offline stand-in: quote the opening of the first passage, or refuse if none."""
        match = re.search(r"\[1\]\n(.+?)(?:\n\n\[2\]|\n\nQuestion:)", prompt, re.S)
        if not match:
            return REFUSAL
        return " ".join(match.group(1).split())[:300]
