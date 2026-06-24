"""OpenAI client with rate limiting and retry/backoff."""

from __future__ import annotations

import json
import random
import threading
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from honeypot.config import get_openai_api_key
from honeypot.prompts import REPAIR_PROMPT_SUFFIX, SYSTEM_PROMPT, build_user_prompt
from honeypot.schema import HoneypotJudgment, parse_judgment_json


class RateLimiter:
    """Token-bucket style limiter: max N acquisitions per 60-second window."""

    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = max(1, requests_per_minute)
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                window_start = now - 60.0
                self._timestamps = [t for t in self._timestamps if t > window_start]
                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return
                sleep_for = self._timestamps[0] - window_start
            time.sleep(max(sleep_for, 0.05))


class HoneypotLLMClient:
    def __init__(
        self,
        *,
        model: str,
        requests_per_minute: int,
        max_retries: int,
        initial_backoff_sec: float,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.initial_backoff_sec = initial_backoff_sec
        self.verbose = verbose
        self._client = OpenAI(api_key=get_openai_api_key())
        self._rate_limiter = RateLimiter(requests_per_minute)
        self._stats_lock = threading.Lock()
        self.retries_total = 0

    def _backoff_seconds(self, attempt: int, error: Exception) -> float:
        if isinstance(error, RateLimitError):
            retry_after = getattr(error, "retry_after", None)
            if retry_after is not None:
                try:
                    return float(retry_after) + random.uniform(0.1, 0.5)
                except (TypeError, ValueError):
                    pass
        base = self.initial_backoff_sec * (2**attempt)
        return base + random.uniform(0, 1.0)

    def _call_api(self, user_prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._rate_limiter.acquire()
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("empty response content")
                return content
            except (
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
            ) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                with self._stats_lock:
                    self.retries_total += 1
                delay = self._backoff_seconds(attempt, exc)
                print(
                    f"  [retry {attempt + 1}/{self.max_retries}] "
                    f"{type(exc).__name__}: sleeping {delay:.1f}s"
                )
                time.sleep(delay)
        raise RuntimeError("OpenAI call failed after retries") from last_error

    def judge_candidate(
        self,
        record: dict[str, Any],
        *,
        pass_number: int,
        pass1_judgment: dict[str, Any] | None = None,
    ) -> HoneypotJudgment:
        candidate_id = str(record.get("candidate_id", ""))
        user_prompt = build_user_prompt(
            record,
            pass_number=pass_number,
            pass1_judgment=pass1_judgment,
        )

        if self.verbose:
            print(
                f"    [api] pass {pass_number} request -> {candidate_id}",
                flush=True,
            )
        raw_text = self._call_api(user_prompt)
        try:
            raw = json.loads(raw_text)
            return parse_judgment_json(raw, expected_id=candidate_id)
        except (json.JSONDecodeError, ValueError) as first_err:
            if self.verbose:
                print(
                    f"    [api] pass {pass_number} repair -> {candidate_id}",
                    flush=True,
                )
            repair_prompt = user_prompt + REPAIR_PROMPT_SUFFIX
            raw_text = self._call_api(repair_prompt)
            try:
                raw = json.loads(raw_text)
                return parse_judgment_json(raw, expected_id=candidate_id)
            except (json.JSONDecodeError, ValueError) as second_err:
                raise ValueError(
                    f"schema validation failed for {candidate_id}: {second_err}"
                ) from first_err
