#!/usr/bin/env python3
"""Bounded, standard-library OpenAI SSE serving benchmark.

Measures requests and token-service timing, NOT trained-model numerical
equivalence. Use identical pinned model/quant/tokenizer/prompt inputs and
a separate reference oracle before claiming vLLM performance parity.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Outcome:
    completed: bool
    finish_reason: str | None
    elapsed: float
    ttft: float | None
    itls: list[float]
    tokens: int | None
    error: str | None
    output_sha256: str | None = None


def percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None
    series = sorted(values)
    return series[min(len(series) - 1, max(0, math.ceil(len(series) * percentage) - 1))]


def stream_one(url: str, model: str, prompt: str, max_tokens: int,
               timeout: float, key: str | None) -> Outcome:
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1, "n": 1, "stream": True,
        "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
    }, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", data=body, headers=headers,
        method="POST",
    )
    started = time.monotonic()
    ttft = None
    itls = []
    previous_token = None
    count = None
    reason = None
    ended = False
    output_hash = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}")
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                text = line[6:].strip()
                if text == b"[DONE]":
                    ended = True
                    break
                parsed = json.loads(text)
                if "error" in parsed:
                    raise ValueError(f"provider error {parsed['error'].get('code', 'unknown')}")
                for choice in parsed.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content", "") if isinstance(delta, dict) else choice.get("text", "")
                    if content:
                        output_hash.update(content.encode("utf-8"))
                        now = time.monotonic()
                        if ttft is None:
                            ttft = now - started
                        elif previous_token is not None:
                            # SSE text chunks aren't guaranteed to be one token.
                            # This is a chunk interval, not a verified true ITL.
                            itls.append(now - previous_token)
                        previous_token = now
                    if choice.get("finish_reason") is not None:
                        reason = choice["finish_reason"]
                usage = parsed.get("usage")
                if usage and isinstance(usage.get("completion_tokens"), int):
                    count = usage["completion_tokens"]
        if not ended or reason is None:
            raise ValueError("missing SSE [DONE] or terminal choice")
        return Outcome(True, reason, time.monotonic() - started,
                       ttft, itls, count, None, output_hash.hexdigest())
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError, TimeoutError) as exc:
        return Outcome(False, None, time.monotonic() - started, ttft, itls, count, type(exc).__name__)


def bench(url: str, model: str, prompt: str, count: int, concurrency: int,
          max_tokens: int, timeout: float, key: str | None) -> dict:
    if count < 1 or count > 1024 or concurrency < 1 or concurrency > 64 or concurrency > count:
        raise ValueError("bounded requests/concurrency required (1..1024, 1..64)")
    if not (1 <= max_tokens <= 512) or not (1 <= timeout <= 600):
        raise ValueError("bounded token/deadline limits")
    if not prompt or len(prompt.encode()) > 32 * 1024:
        raise ValueError("nonempty prompt up to 32 KiB required")
    # Submit every offered attempt, include rejected/expired responses.
    began = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(
            lambda _i: stream_one(url, model, prompt, max_tokens, timeout, key), range(count)
        ))
    elapsed = time.monotonic() - began
    completed = [o for o in outcomes if o.completed]
    ttfts = [o.ttft for o in completed if o.ttft is not None]
    chunk_intervals = [i for o in completed for i in o.itls]
    latency = [o.elapsed for o in completed]
    known_tokens = sum(o.tokens for o in completed if o.tokens is not None)
    unknown_tokens = sum(o.tokens is None for o in completed)
    errors = {}
    for o in outcomes:
        if o.error:
            errors[o.error] = errors.get(o.error, 0) + 1
    return {
        "schema": "zerogravity.serving.benchmark.v1",
        "endpoint": url.rstrip("/"),
        "model": model,
        "concurrency": concurrency,
        "offered_requests": count,
        "completed_requests": len(completed),
        "unsuccessful_requests": count - len(completed),
        "errors": errors,
        "output_sha256_by_request": [o.output_sha256 for o in outcomes],
        "wall_seconds": round(elapsed, 5),
        "latency_p50_seconds": percentile(latency, .50),
        "latency_p95_seconds": percentile(latency, .95),
        "ttft_p50_seconds": percentile(ttfts, .50),
        "ttft_p95_seconds": percentile(ttfts, .95),
        "sse_chunk_interval_p95_seconds": percentile(chunk_intervals, .95),
        "usage_reported_completion_tokens": known_tokens,
        "completions_without_usage": unknown_tokens,
        "reported_tokens_per_offered_second": round(known_tokens / elapsed, 4) if elapsed > 0 and not unknown_tokens else None,
        "numerical_source_equivalence": "not_verified",
        "fused_kernels_or_vllm_parity": "not_inferred",
        "note": "SSE chunk intervals do not establish token-level ITL; use provider token events or tracing.",
    }


def self_test() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            assert json.loads(raw)["model"] == "fixture"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in [
                {"choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
                {"choices": [], "usage": {"completion_tokens": 1}},
            ]:
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.close_connection = True
        def log_message(self, _format, *_args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    task = threading.Thread(target=server.serve_forever, daemon=True)
    task.start()
    try:
        result = bench(f"http://127.0.0.1:{server.server_port}", "fixture",
                       "hello", 4, 2, 1, 8, None)
        assert result["offered_requests"] == 4
        assert result["completed_requests"] == 4
        assert result["reported_tokens_per_offered_second"] is not None
        assert result["numerical_source_equivalence"] == "not_verified"
        assert len(result["output_sha256_by_request"])==4
        assert len(set(result["output_sha256_by_request"]))==1
        assert result["sse_chunk_interval_p95_seconds"] is None
        assert percentile([1., 2., 3., 4.], .95) == 4.
        print("SERVING_BENCH_SOURCE_FREE_SELF_TEST_PASS")
    finally:
        server.shutdown()
        server.server_close()
        task.join(timeout=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--prompt", default="Hi.")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--api-key-env", default="ZEROGRAVITY_API_KEY")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.model:
        parser.error("--model is required")
    import os
    print(json.dumps(bench(args.endpoint, args.model, args.prompt,
                           args.requests, args.concurrency, args.max_tokens,
                           args.timeout, os.environ.get(args.api_key_env)),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
