"""Unit tests for openai_retry.

No real network and no real sleeping: ``_sleep`` is monkeypatched to record
the delays it would have taken, and the OpenAI SDK errors are constructed
against fake httpx responses so the tests run fast and offline.
"""

from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError, InternalServerError, RateLimitError

import openai_retry
from openai_retry import AIQuotaExhaustedError, call_with_retry


# --- helpers to build real SDK errors without a network call -----------------

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _rate_limit_error(*, body=None, message="rate limited") -> RateLimitError:
    response = httpx.Response(429, request=_REQUEST)
    return RateLimitError(message, response=response, body=body)


def _server_error(message="boom") -> InternalServerError:
    response = httpx.Response(500, request=_REQUEST)
    return InternalServerError(message, response=response, body=None)


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=_REQUEST)


def _quota_body() -> dict:
    return {"error": {"code": "insufficient_quota", "message": "You exceeded your quota."}}


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Replace _sleep with a recorder so the suite never actually sleeps."""
    calls: list[float] = []
    monkeypatch.setattr(openai_retry, "_sleep", lambda d: calls.append(d))
    return calls


# --- happy path --------------------------------------------------------------

def test_returns_value_without_sleeping(no_real_sleep):
    result = call_with_retry(lambda: 42)
    assert result == 42
    assert no_real_sleep == []


def test_retries_then_succeeds(no_real_sleep):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _server_error()
        return "ok"

    assert call_with_retry(flaky) == "ok"
    assert attempts["n"] == 3
    # Two failures before the success means two backoff sleeps.
    assert len(no_real_sleep) == 2


# --- giving up ---------------------------------------------------------------

def test_gives_up_and_reraises_last_transient_error(monkeypatch, no_real_sleep):
    monkeypatch.setattr(openai_retry, "AI_RETRY_MAX_ATTEMPTS", 3)

    def always_429():
        raise _rate_limit_error()

    with pytest.raises(RateLimitError):
        call_with_retry(always_429)
    # 3 attempts means 2 sleeps between them, none after the final failure.
    assert len(no_real_sleep) == 2


def test_connection_errors_are_retried(no_real_sleep):
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _connection_error()
        return "recovered"

    assert call_with_retry(flaky) == "recovered"
    assert len(no_real_sleep) == 1


# --- non-retryable -----------------------------------------------------------

def test_non_retryable_error_propagates_immediately(no_real_sleep):
    def bad():
        raise ValueError("not an OpenAI transient error")

    with pytest.raises(ValueError):
        call_with_retry(bad)
    assert no_real_sleep == []


# --- quota exhaustion --------------------------------------------------------

def test_quota_exhausted_is_not_retried_and_is_typed(no_real_sleep):
    seen = []

    def quota():
        raise _rate_limit_error(body=_quota_body())

    with pytest.raises(AIQuotaExhaustedError) as exc_info:
        call_with_retry(quota, on_quota_exhausted=lambda desc, code: seen.append((desc, code)))

    # Never retried, so never slept.
    assert no_real_sleep == []
    # Callback fired exactly once with the detected code.
    assert seen == [("OpenAI call", "insufficient_quota")]
    assert exc_info.value.code == "insufficient_quota"


def test_quota_error_is_caught_by_rate_limit_handlers():
    # Legacy handlers that only know about RateLimitError keep working.
    def quota():
        raise _rate_limit_error(body=_quota_body())

    with pytest.raises(RateLimitError):
        call_with_retry(quota)


def test_quota_detected_via_message_when_body_missing(no_real_sleep):
    def quota():
        raise _rate_limit_error(message="insufficient_quota: out of credits")

    with pytest.raises(AIQuotaExhaustedError):
        call_with_retry(quota)
    assert no_real_sleep == []


def test_callback_failure_does_not_mask_quota_error(no_real_sleep):
    def quota():
        raise _rate_limit_error(body=_quota_body())

    def exploding_callback(desc, code):
        raise RuntimeError("metrics backend down")

    # The callback blowing up must not change what the caller sees.
    with pytest.raises(AIQuotaExhaustedError):
        call_with_retry(quota, on_quota_exhausted=exploding_callback)


# --- backoff math ------------------------------------------------------------

def test_backoff_is_capped_and_jittered(monkeypatch):
    monkeypatch.setattr(openai_retry, "AI_RETRY_BASE_DELAY_SECONDS", 1.0)
    monkeypatch.setattr(openai_retry, "AI_RETRY_MAX_DELAY_SECONDS", 4.0)
    # A high attempt number would explode without the cap; jitter is +/-25%,
    # so the result must stay within [cap*0.75, cap*1.25].
    delay = openai_retry._compute_backoff_delay(attempt=10)
    assert 3.0 <= delay <= 5.0
