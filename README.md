# openai-retry

Production-grade retry and backoff for the OpenAI Python SDK.

## The problem

Calls to the OpenAI API fail transiently all the time under real load: 429
rate limits, occasional 5xx server errors, and dropped connections or request
timeouts. A naive `try/except` either gives up on the first blip or, worse,
retries everything in a tight loop. The second failure mode is the dangerous
one, because some errors will never recover no matter how many times you try.
The clearest example is `insufficient_quota`, which means the account has hit a
hard billing or credit cap. Hammering that wall just burns your latency budget
and still fails.

This package is a small wrapper that retries the failures that can actually
recover, and refuses to retry the one that can't.

## Quickstart

```bash
pip install openai-retry
```

```python
from openai import OpenAI
from openai_retry import call_with_retry

client = OpenAI()

response = call_with_retry(
    lambda: client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    ),
    description="chat completion",
)
```

`call_with_retry` takes a zero-argument callable so you keep full control over
the request. Transient errors are retried with jittered exponential backoff;
anything non-transient propagates immediately.

If you want a ready-made client with a sane request timeout, there is a cached
helper:

```python
from openai_retry import get_openai_client

client = get_openai_client(timeout=30.0)  # reads OPENAI_API_KEY
```

## Handling a hard quota cap

A `RateLimitError` with the code `insufficient_quota` is treated as terminal:
it is not retried, and it is re-raised as `AIQuotaExhaustedError` so you can
branch on it specifically.

```python
from openai_retry import call_with_retry, AIQuotaExhaustedError
from openai import RateLimitError

def on_quota(description, code):
    metrics.increment("openai.quota_exhausted")  # your observability hook

try:
    result = call_with_retry(do_call, on_quota_exhausted=on_quota)
except AIQuotaExhaustedError:
    result = friendly_fallback()   # the account is out of credits
except RateLimitError:
    result = generic_fallback()    # a transient 429 that exhausted retries
```

Because `AIQuotaExhaustedError` subclasses `openai.RateLimitError`, code that
already catches `RateLimitError` keeps working without changes.

## Design decisions

**Only transient errors are retried.** The retryable set is exactly
`RateLimitError` (429), `InternalServerError` (5xx), and `APIConnectionError`
(connection failures, with timeouts covered because `APITimeoutError` is a
subclass). Auth failures, invalid-request 4xxs, and your own validation errors
fall straight through. Retrying those can never change the outcome and only
adds latency.

**The retry budget is bounded and the defaults are conservative.** Three total
attempts, a one second base delay, and a four second per-delay cap. That keeps
the worst-case added latency around three seconds, which is safe to use inside
a request or webhook handler. Batch and offline jobs can raise the limits with
environment variables.

**Backoff is jittered.** Delays grow exponentially but carry plus or minus 25%
random jitter, so a fleet of workers that all hit the same 429 do not retry in
lock-step and immediately collide again.

**`insufficient_quota` is terminal, not transient.** This is the core judgment
call. A quota cap is a wall, not a blip, so the wrapper detects it, optionally
reports it through a callback, and raises a typed error instead of retrying.
Detection checks the structured `error.code` first and falls back to a string
match on the message, which guards against SDK or proxy changes that might drop
the structured field.

**The quota error is typed but backward compatible.** `AIQuotaExhaustedError`
subclasses `RateLimitError` on purpose: existing handlers keep catching it,
while new code can catch the subclass first for quota-specific messaging.

**Observability never masks the original failure.** The `on_quota_exhausted`
callback is optional, and any exception it raises is swallowed and logged so a
broken metrics backend can't turn a clean quota error into a confusing one.

## Configuration

All settings are read from the environment at import time.

| Variable | Default | Meaning |
| --- | --- | --- |
| `AI_RETRY_MAX_ATTEMPTS` | `3` | Total attempts (1 initial try plus retries). |
| `AI_RETRY_BASE_DELAY_SECONDS` | `1.0` | Base delay for exponential backoff. |
| `AI_RETRY_MAX_DELAY_SECONDS` | `4.0` | Per-delay cap before jitter is applied. |

## Testing

The retry logic is fully unit tested without any network calls or real sleeps.
The module exposes `_sleep` as a patch point precisely so tests can record the
backoff delays instead of waiting them out.

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
