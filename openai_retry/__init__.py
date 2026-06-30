"""Production-grade retry/backoff for the OpenAI Python SDK.

``call_with_retry`` wraps any OpenAI SDK call and retries only the failures
that can actually recover - 429 rate limits, 5xx server errors, and
connection/timeout errors - using bounded, jittered exponential backoff. A
hard billing cap (``insufficient_quota``) is detected and surfaced as a typed
:class:`AIQuotaExhaustedError` instead of being retried into the ground.
"""

from __future__ import annotations

import logging
import os
import random
import time
from functools import lru_cache
from typing import Callable, Optional, TypeVar

from openai import (
    APIConnectionError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

__all__ = [
    "call_with_retry",
    "get_openai_client",
    "AIQuotaExhaustedError",
    "RETRYABLE_AI_ERRORS",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")

# --- Bounded retry policy for transient OpenAI failures ----------------------
# The defaults are deliberately conservative so the wrapper is safe to use
# inside a request/response path (a web handler, a webhook receiver) where a
# long retry storm would blow the latency budget. With 3 attempts (1 initial
# try + 2 retries), a 1s base and a 4s per-delay cap, the worst-case added
# sleep is about 1s + 2s = 3s (plus or minus 25% jitter). Override any of
# these with the AI_RETRY_* environment variables for batch/offline work where
# a larger budget is fine.
AI_RETRY_MAX_ATTEMPTS = int(os.getenv("AI_RETRY_MAX_ATTEMPTS", "3"))
AI_RETRY_BASE_DELAY_SECONDS = float(os.getenv("AI_RETRY_BASE_DELAY_SECONDS", "1.0"))
AI_RETRY_MAX_DELAY_SECONDS = float(os.getenv("AI_RETRY_MAX_DELAY_SECONDS", "4.0"))

# Only transient failures are retried:
#   RateLimitError       -> HTTP 429
#   InternalServerError  -> HTTP 5xx
#   APIConnectionError   -> network/connection failure (APITimeoutError is a
#                           subclass, so request timeouts are covered too)
# Everything else (auth errors, 4xx invalid requests, your own ValueErrors or
# pydantic ValidationErrors) propagates immediately: retrying those can never
# succeed and only adds latency.
RETRYABLE_AI_ERRORS: tuple[type[Exception], ...] = (
    RateLimitError,
    InternalServerError,
    APIConnectionError,
)

# Module-level alias so tests can monkeypatch openai_retry._sleep without
# touching the global time module (and without real sleeps in the suite).
_sleep = time.sleep


# --- Quota-exhaustion (`insufficient_quota`) handling ------------------------
#
# OpenAI marks a hard billing/credit cap with the error code
# "insufficient_quota". That is a wall, not a blip: retrying only burns the
# latency budget and never succeeds. The wrapper detects it, lets you record
# it through an optional callback, and then surfaces a typed
# ``AIQuotaExhaustedError`` so callers can branch to a quota-specific fallback.
#
# AIQuotaExhaustedError subclasses RateLimitError so existing
# ``except RateLimitError`` / ``except OpenAIError`` blocks keep catching this
# case. New code can ``except AIQuotaExhaustedError`` first to apply messaging
# that is distinct from the transient-429 path.

_INSUFFICIENT_QUOTA_CODE = "insufficient_quota"


class AIQuotaExhaustedError(RateLimitError):
    """OpenAI hard quota / billing-cap failure (``insufficient_quota``).

    Distinct from the transient 429 retry/backoff path: a quota-exhausted
    error means the account is out of credits or has hit a cap, so retrying
    would only consume latency without changing the outcome.

    Subclasses :class:`openai.RateLimitError` so handlers that catch
    ``RateLimitError`` / ``OpenAIError`` still treat this gracefully. New
    callers can ``except AIQuotaExhaustedError`` first to surface a clean,
    quota-specific fallback that is distinct from a transient blip.

    Constructed by wrapping the original ``RateLimitError`` so the SDK's
    standard ``response`` / ``body`` introspection still works. The original
    exception is also chained as ``__cause__`` (via ``raise ... from exc``).
    """

    def __init__(
        self,
        original: RateLimitError,
        *,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(
            f"OpenAI quota exhausted ({_INSUFFICIENT_QUOTA_CODE}): {original}",
            response=original.response,
            body=original.body,
        )
        # APIError.__init__ resets .code from body.get('code') (a top-level
        # 'code' field that OpenAI's responses don't populate - the real code
        # lives under body['error']['code']). Override AFTER the super() call
        # so .code reliably reflects the quota marker.
        self.code = code or _INSUFFICIENT_QUOTA_CODE
        self.original = original


def _extract_openai_error_code(exc: BaseException) -> Optional[str]:
    """Best-effort extraction of OpenAI's ``error.code`` from a RateLimitError.

    Returns the lowercased code string when present, else ``None``. Never
    raises - a malformed body must not turn a quota check into an error of its
    own. Inspects (in order):

      1. ``exc.code`` (top-level attribute the SDK sets from body['code']).
      2. ``exc.body['error']['code']`` (canonical shape in OpenAI's JSON
         responses; not populated at the top level so we have to dig).
    """
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code.lower()
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            inner_code = err.get("code")
            if isinstance(inner_code, str) and inner_code:
                return inner_code.lower()
    return None


def _is_quota_exhausted(exc: RateLimitError) -> bool:
    """Return ``True`` when ``exc`` represents an OpenAI hard-quota failure.

    Checked in order:
      1. Structured error code via :func:`_extract_openai_error_code`.
      2. Substring fallback against ``exc.message`` / ``str(exc)`` - defends
         against SDK / proxy reshuffles that could silently strip the
         structured marker and otherwise downgrade us to the transient
         retry path.
    """
    code = _extract_openai_error_code(exc)
    if code == _INSUFFICIENT_QUOTA_CODE:
        return True
    message = getattr(exc, "message", None) or str(exc)
    return _INSUFFICIENT_QUOTA_CODE in message.lower()


@lru_cache(maxsize=1)
def get_openai_client(timeout: float = 30.0) -> OpenAI:
    """Return a cached :class:`openai.OpenAI` client built from ``OPENAI_API_KEY``.

    The per-call ``timeout`` (seconds) is applied to every request made
    through the client, so a hung OpenAI request cannot stall a worker
    indefinitely. This is a convenience only - ``call_with_retry`` works with
    any client or any zero-argument callable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=api_key, timeout=timeout)


def _compute_backoff_delay(attempt: int) -> float:
    """Exponential backoff capped at ``AI_RETRY_MAX_DELAY_SECONDS`` with +/-25% jitter."""
    base = min(
        AI_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
        AI_RETRY_MAX_DELAY_SECONDS,
    )
    # +/-25% jitter so concurrent workers hitting the same 429 don't retry in
    # lock-step and re-collide.
    return base * (1.0 + random.uniform(-0.25, 0.25))


def call_with_retry(
    operation: Callable[[], T],
    *,
    description: str = "OpenAI call",
    on_quota_exhausted: Optional[Callable[[str, Optional[str]], None]] = None,
) -> T:
    """Run *operation*, retrying transient OpenAI failures with exponential backoff.

    Retries on 429 / 5xx / connection errors (see ``RETRYABLE_AI_ERRORS``).
    Non-retryable exceptions propagate immediately. After
    ``AI_RETRY_MAX_ATTEMPTS`` total attempts, the last transient exception is
    re-raised so existing callers' except-blocks behave exactly as before -
    the retries only reduce how often they fire.

    Special case: a ``RateLimitError`` whose error code is
    ``insufficient_quota`` is a HARD billing/credit cap, NOT a transient blip.
    Such errors are:

      - NOT retried (saves the backoff latency on a wall that won't move).
      - Reported through the optional ``on_quota_exhausted(description, code)``
        callback so you can record a distinct metric/event and tell a real
        billing wall apart from transient blips. Any exception the callback
        raises is swallowed and logged so an observability failure can never
        mask the original quota error.
      - Re-raised as :class:`AIQuotaExhaustedError` (a ``RateLimitError``
        subclass) so existing handlers still catch them, and new callers can
        match on the typed subclass for a quota-specific fallback.

    :param operation: A zero-argument callable that performs the OpenAI call,
        e.g. ``lambda: client.chat.completions.create(...)``.
    :param description: Human-readable label used in log lines and passed to
        the quota callback.
    :param on_quota_exhausted: Optional ``(description, code) -> None`` hook
        invoked once when a hard quota cap is detected, before the typed error
        is raised.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, AI_RETRY_MAX_ATTEMPTS + 1):
        try:
            return operation()
        except RateLimitError as exc:
            # Quota-exhausted: hard wall, do not retry. Report it + raise the
            # typed subclass that still satisfies `except RateLimitError`.
            if _is_quota_exhausted(exc):
                code = _extract_openai_error_code(exc)
                logger.error(
                    "%s hit OpenAI quota cap (code=%s) on attempt %d - "
                    "not retrying; raising AIQuotaExhaustedError.",
                    description,
                    code or _INSUFFICIENT_QUOTA_CODE,
                    attempt,
                )
                if on_quota_exhausted is not None:
                    # The callback must never mask the original quota error:
                    # swallow and log anything it raises, then re-raise the
                    # typed error below so the caller can fall back cleanly.
                    try:
                        on_quota_exhausted(description, code)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "on_quota_exhausted callback raised; "
                            "AIQuotaExhaustedError will still be raised so the "
                            "caller can apply its fallback."
                        )
                raise AIQuotaExhaustedError(exc, code=code) from exc
            last_exc = exc
            if attempt >= AI_RETRY_MAX_ATTEMPTS:
                logger.warning(
                    "%s failed with RateLimitError on attempt %d/%d - giving up.",
                    description,
                    attempt,
                    AI_RETRY_MAX_ATTEMPTS,
                )
                break
            delay = _compute_backoff_delay(attempt)
            logger.warning(
                "%s failed with RateLimitError (attempt %d/%d) - retrying in %.2fs.",
                description,
                attempt,
                AI_RETRY_MAX_ATTEMPTS,
                delay,
            )
            _sleep(delay)
        except (InternalServerError, APIConnectionError) as exc:
            last_exc = exc
            if attempt >= AI_RETRY_MAX_ATTEMPTS:
                logger.warning(
                    "%s failed with %s on attempt %d/%d - giving up.",
                    description,
                    type(exc).__name__,
                    attempt,
                    AI_RETRY_MAX_ATTEMPTS,
                )
                break
            delay = _compute_backoff_delay(attempt)
            logger.warning(
                "%s failed with %s (attempt %d/%d) - retrying in %.2fs.",
                description,
                type(exc).__name__,
                attempt,
                AI_RETRY_MAX_ATTEMPTS,
                delay,
            )
            _sleep(delay)
    assert last_exc is not None  # the loop only breaks after setting last_exc
    raise last_exc
