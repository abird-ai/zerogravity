#!/usr/bin/env python3
"""Bounded OpenAI SSE measurements; no inferred source or vLLM equivalence.

Default requests are closed-loop. --workload supplies open-loop arrivals.
SSE intervals measure chunks. Only separately bound provider token events
can supply token ITL; parsing them cannot independently prove commits/math.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import ipaddress
import json
import math
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Outcome:
    completed: bool
    finish_reason: str | None
    elapsed: float
    ttft: float | None
    itls: list[float]  # Compatibility field: SSE chunk intervals only.
    tokens: int | None
    error: str | None
    output_sha256: str | None = None
    classification: str = "completed"
    http_status: int | None = None
    http_accepted: bool = False
    provider_response_id: str | None = None
    chunk_times: list[float] = field(default_factory=list)
    provider_error_code: str | None = None
    token_observation: dict | None = None


def percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None
    series = sorted(values)
    return series[min(len(series) - 1, max(0, math.ceil(len(series) * percentage) - 1))]


def strict_json(path: str) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    with Path(path).open("rb") as source:
        raw = source.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("JSON input exceeds 8 MiB")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def number(value: object, lower: float, upper: float) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and lower <= value <= upper)


def label(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", value) is not None


def workload_records(value: object, max_tokens: int) -> list[dict]:
    if isinstance(value, dict):
        if set(value) != {"requests"}:
            raise ValueError("workload object requires only requests")
        value = value["requests"]
    if not isinstance(value, list) or not 1 <= len(value) <= 1024:
        raise ValueError("workload requires 1..1024 requests")
    result, seen, total = [], set(), 0
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) - {
            "request_id", "prompt", "arrival_seconds", "tenant", "max_tokens"
        }:
            raise ValueError("unknown workload request field")
        identity = item.get("request_id", f"request-{index}")
        prompt, arrival = item.get("prompt"), item.get("arrival_seconds")
        tenant, limit = item.get("tenant", "default"), item.get("max_tokens", max_tokens)
        if (not label(identity) or identity in seen or not label(tenant)
                or not isinstance(prompt, str) or not prompt or len(prompt.encode()) > 32 * 1024
                or not number(arrival, 0, 600) or not isinstance(limit, int)
                or isinstance(limit, bool) or not 1 <= limit <= 512):
            raise ValueError("invalid/duplicate bounded workload request")
        total += len(prompt.encode())
        if total > 8 * 1024 * 1024:
            raise ValueError("workload prompts exceed 8 MiB")
        seen.add(identity)
        result.append(dict(request_id=identity, prompt=prompt, arrival_seconds=float(arrival),
                           tenant=tenant, max_tokens=limit, input_index=index))
    return result


def wire_token_observation(groups: list[dict], extension: object, aggregate: object,
                           wire_id: str | None, provider_id: str | None,
                           count: int | None, maximum: int) -> dict:
    """Validate token IDs and distinct clock domains; never derive IDs from text."""
    if (type(maximum) is not int or not 1 <= maximum <= 512
            or (count is not None and (type(count) is not int or not 0 <= count <= maximum))):
        raise ValueError("invalid token-ID usage/capacity")
    identities, arrivals = [], []
    for group in groups:
        ids, at = group.get("token_ids"), group.get("client_received_at_seconds")
        if (not isinstance(ids, list) or not ids or len(ids) > maximum
                or not number(at, 0, 600) or (arrivals and at < arrivals[-1])
                or any(type(token) is not int or not 0 <= token <= 0xFFFFFFFF for token in ids)):
            raise ValueError("invalid bounded token-ID delivery group")
        identities.extend(ids)
        arrivals.append(at)
    if len(identities) > maximum or len(identities) > 512:
        raise ValueError("token-ID observations exceed requested capacity")
    if count is not None and groups and len(identities) != count:
        raise ValueError("actual token-ID count differs from terminal usage")
    # Pinned vLLM _base_request_id honors X-Request-ID, then uses this prefix.
    valid_bindings = all(isinstance(identity, str)
        and re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", identity) is not None
        for identity in [wire_id, provider_id])
    bound = valid_bindings and provider_id in {f"chatcmpl-{wire_id}", f"cmpl-{wire_id}"}
    commit_times, records = [], []
    last_received, last_written, last_gpu, last_cohort = None, None, None, None
    commit_status = "absent"
    if extension is not None:
        if (not valid_bindings or not isinstance(extension, dict)
                or extension.get("schema") != "zerogravity.serving.request-token-metrics.v1"
                or extension.get("clock") != "request_monotonic_seconds"
                or extension.get("wire_request_id") != wire_id
                or extension.get("provider_response_id") != provider_id
                or extension.get("complete") is not True
                or type(extension.get("completion_tokens")) is not int
                or extension.get("completion_tokens") != count
                or not isinstance(extension.get("tokens"), list)
                or len(extension["tokens"]) != count or count is None or count > 512):
            raise ValueError("provider token metadata binding/count/clock mismatch")
        bound = True
        for index, token in enumerate(extension["tokens"]):
            if (not isinstance(token, dict) or type(token.get("index")) is not int
                    or token["index"] != index or index >= len(identities)
                    or type(token.get("token_id")) is not int or token["token_id"] != identities[index]):
                raise ValueError("provider token metadata differs from delivered token IDs/order")
            committed, received, written = (token.get("commit_observed_at_seconds"),
                token.get("output_received_at_seconds"), token.get("http_write_completed_at_seconds"))
            if (not number(committed, 0, 600) or not number(received, committed, 600)
                    or (written is not None and not number(written, received, 600))
                    or (commit_times and committed < commit_times[-1])
                    or (last_received is not None and received < last_received)
                    or (last_written is not None and received < last_written)
                    or (last_written is not None and written is not None and written < last_written)):
                raise ValueError("provider token metadata clock/order mismatch")
            last_received = received
            if written is not None:
                last_written = written
            gpu = token.get("gpu")
            if gpu is not None:
                if (not isinstance(gpu, dict) or type(gpu.get("producer_frontier")) is not int
                        or not 1 <= gpu["producer_frontier"] <= 0xFFFFFFFFFFFFFFFF
                        or (index and (last_gpu is None or gpu["producer_frontier"] != last_gpu + 1))):
                    raise ValueError("invalid/nonconsecutive GPU producer frontier")
                last_gpu = gpu["producer_frontier"]
                cohort = gpu.get("native_cohort")
                if cohort is not None:
                    if (not isinstance(cohort, dict) or type(cohort.get("publication_sequence")) is not int
                            or not 1 <= cohort["publication_sequence"] <= 0xFFFFFFFFFFFFFFFF
                            or type(cohort.get("width")) is not int or cohort["width"] not in {2,4,8}
                            or (last_cohort is not None and cohort["publication_sequence"] <= last_cohort)):
                        raise ValueError("invalid/repeated native cohort publication observation")
                    last_cohort = cohort["publication_sequence"]
            elif last_gpu is not None:
                raise ValueError("GPU producer observation disappeared within one request")
            commit_times.append(committed)
            records.append(token)
        commit_status = "provider_bound_complete_commit_observations"
    complete_ids = bound and count is not None and len(identities) == count
    individual = complete_ids and all(len(group["token_ids"]) == 1 for group in groups)
    if aggregate is not None:
        if not isinstance(aggregate, dict):
            raise ValueError("invalid provider aggregate metrics")
        for key in ["time_to_first_token_ms", "generation_time_ms", "queue_time_ms", "mean_itl_ms", "tokens_per_second"]:
            value = aggregate.get(key)
            if value is not None and not number(value, 0, 1e12):
                raise ValueError("nonfinite/invalid provider aggregate metric")
    return {"status": "provider_bound_complete_ids" if complete_ids else "incomplete_or_unbound_ids",
        "token_ids": identities, "client_delivery_groups": groups,
        "client_delivery_status": "individual_token_ids_observed" if individual else "unresolved_token_groups_or_binding",
        "client_first_token_seconds": arrivals[0] if individual and arrivals else None,
        "client_token_delivery_intervals_seconds": [b-a for a,b in zip(arrivals, arrivals[1:])] if individual else None,
        "provider_commit_observation_status": commit_status,
        "provider_commit_observation_intervals_seconds": [b-a for a,b in zip(commit_times, commit_times[1:])] if commit_status != "absent" else None,
        "provider_first_commit_observed_seconds": commit_times[0] if commit_times else None,
        "provider_token_records": records, "provider_aggregate_metrics": aggregate,
        "semantics": "client parsed actual token-ID events; coalesced multi-ID groups have unresolved individual delivery; provider commit observations and mean ITL are distinct clocks/scopes; no independent physical/numerical proof"}


def stream_one(url: str, model: str, prompt: str, max_tokens: int,
               timeout: float, key: str | None, *, request_id: str | None = None,
               return_token_ids: bool = False) -> Outcome:
    payload = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1, "n": 1, "stream": True,
        "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
    }
    if return_token_ids:
        payload["return_token_ids"] = True
    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if request_id:
        headers["X-Request-ID"] = request_id
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                     data=body, headers=headers, method="POST")
    started = time.monotonic()
    ttft, previous, count, reason, status, provider_id, provider_code = (None,) * 7
    intervals, chunk_times = [], []
    ended, accepted, terminals, usages = False, False, 0, 0
    groups, extension, aggregate, observation = [], None, None, None
    data_events = 0
    output_hash = hashlib.sha256()

    def outcome(completed, classification, error):
        observed = observation
        if return_token_ids and observed is None:
            observed = {"status":"incomplete_response", "token_ids":[token for group in groups for token in group["token_ids"]],
                "client_delivery_groups":groups, "provider_aggregate_metrics":aggregate,
                "provider_commit_observation_status":"not_established"}
        return Outcome(completed, reason, time.monotonic() - started, ttft,
                       intervals, count, error, output_hash.hexdigest() if completed else None,
                       classification, status, accepted, provider_id, chunk_times, provider_code, observed)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, accepted = response.status, response.status == 200
            if not accepted:
                raise ValueError(f"HTTP {status}")
            for line in response:
                if len(line) > 1024 * 1024:
                    raise ValueError("SSE line exceeds 1 MiB")
                if time.monotonic() - started > timeout:
                    raise TimeoutError("request deadline")
                if not line.startswith(b"data: "):
                    continue
                text = line[6:].strip()
                if text == b"[DONE]":
                    ended = True
                    break
                data_events += 1
                if data_events > 2048:
                    raise ValueError("too many bounded SSE data events")
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise ValueError("SSE event is not an object")
                if "error" in parsed:
                    error = parsed["error"]
                    provider_code = str(error.get("code", "unknown")) if isinstance(error, dict) else "unknown"
                    kind = ("cancelled" if provider_code in {"cancelled", "canceled", "client_closed"} else
                            "timed_out" if provider_code in {"timeout", "deadline_exceeded", "request_timeout"} else "provider_error")
                    return outcome(False, kind, f"provider:{provider_code}")
                current = parsed.get("id")
                if current is not None:
                    if not isinstance(current, str) or not current or len(current) > 256:
                        raise ValueError("invalid SSE response ID")
                    if provider_id is not None and current != provider_id:
                        raise ValueError("SSE response ID changed")
                    provider_id = current
                choices = parsed.get("choices", [])
                now = time.monotonic() - started
                if parsed.get("zerogravity_token_metrics") is not None:
                    if extension is not None:
                        raise ValueError("duplicate final token metadata")
                    extension = parsed["zerogravity_token_metrics"]
                if parsed.get("metrics") is not None:
                    if aggregate is not None:
                        raise ValueError("duplicate final provider aggregate metrics")
                    aggregate = parsed["metrics"]
                if not isinstance(choices, list):
                    raise ValueError("invalid SSE choices")
                for choice in choices:
                    if not isinstance(choice, dict) or terminals:
                        raise ValueError("invalid/post-terminal SSE choice")
                    if return_token_ids and choice.get("token_ids") is not None:
                        ids = choice["token_ids"]
                        if (type(choice.get("index")) is not int or choice["index"] != 0 or not isinstance(ids, list) or len(ids) > max_tokens
                                or any(type(token) is not int or not 0 <= token <= 0xFFFFFFFF for token in ids)):
                            raise ValueError("invalid actual token IDs/choice")
                        if ids:
                            groups.append(dict(token_ids=ids, client_received_at_seconds=now))
                            if sum(len(group["token_ids"]) for group in groups) > max_tokens:
                                raise ValueError("too many actual token IDs")
                    delta = choice.get("delta")
                    content = delta.get("content", "") if isinstance(delta, dict) else choice.get("text", "")
                    if content:
                        if not isinstance(content, str):
                            raise ValueError("SSE content is not text")
                        output_hash.update(content.encode())
                        if ttft is None:
                            ttft = now
                        if previous is not None:
                            intervals.append(now - previous)
                        previous = now
                        chunk_times.append(now)
                    if choice.get("finish_reason") is not None:
                        reason = choice["finish_reason"]
                        if not isinstance(reason, str) or not reason or len(reason) > 64:
                            raise ValueError("invalid terminal finish reason")
                        terminals += 1
                usage = parsed.get("usage")
                if usage is not None:
                    value = usage.get("completion_tokens") if isinstance(usage, dict) else None
                    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= max_tokens:
                        raise ValueError("invalid completion usage")
                    usages += 1
                    if usages > 1:
                        raise ValueError("duplicate terminal usage")
                    count = value
        if not ended or terminals != 1:
            raise ValueError("missing/duplicate SSE terminal choice or DONE")
        if return_token_ids:
            observation = wire_token_observation(groups, extension, aggregate, request_id, provider_id, count, max_tokens)
        return outcome(True, "completed", None)
    except urllib.error.HTTPError as exc:
        status = exc.code
        kind = ("rejected_429" if status == 429 else "rejected_503" if status == 503 else
                "cancelled" if status == 499 else "timed_out" if status in {408, 504} else "http_error")
        return outcome(False, kind, f"HTTP{status}")
    except (urllib.error.URLError, OSError, ValueError, TypeError, KeyError) as exc:
        cause = getattr(exc, "reason", exc)
        kind = "timed_out" if isinstance(cause, TimeoutError) else (
            "protocol_error" if isinstance(exc, (ValueError, TypeError, KeyError)) else "transport_error")
        return outcome(False, kind, type(exc).__name__)


def snapshot(url: str | None, key: str | None) -> dict | None:
    if url is None:
        return None
    parsed = urllib.parse.urlsplit(url)
    try:
        local = ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        local = False
    if parsed.scheme != "http" or not local or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("snapshot URL requires literal loopback HTTP without credentials/fragment")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args, **_kwargs):
            raise ValueError("provider snapshot redirects are refused")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=5) as response:
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("snapshot exceeds 1 MiB")
            return {"ok": True, "url": url, "value": json.loads(raw)}
    except (OSError, ValueError) as exc:
        return {"ok": False, "url": url, "error": type(exc).__name__}


def validate_token_trace(trace: object, records: list[dict], run_id: str) -> dict:
    """v1 handshake, not independent physical/source attestation.

    Envelope: schema=zerogravity.serving.token-trace.v1, exact run_id,
    clock=provider_monotonic_seconds, event_semantics=committed_output_token.
    requests[]: request_id, wire_request_id, provider_response_id, complete=true,
    tokens[]. Tokens require contiguous index, u32 token_id, committed=true,
    nondecreasing committed_at_seconds; count equals terminal completion usage.
    Provider clock differences give ITL only, never client TTFT.
    """
    result = dict(status="absent", trust="provider_claim_not_independent_commit_or_source_proof",
                  requests={}, errors=[], intervals_seconds=[])
    if trace is None:
        return result
    try:
        if (not isinstance(trace, dict) or trace.get("schema") != "zerogravity.serving.token-trace.v1"
                or trace.get("run_id") != run_id or trace.get("clock") != "provider_monotonic_seconds"
                or trace.get("event_semantics") != "committed_output_token"
                or not isinstance(trace.get("requests"), list) or len(trace["requests"]) > 1024):
            raise ValueError("token trace envelope/run/clock/semantics mismatch")
        expected = {r["request_id"]: r for r in records}
        indexed = {}
        for entry in trace["requests"]:
            if not isinstance(entry, dict):
                raise ValueError("token trace request is not an object")
            identity = entry.get("request_id")
            if not isinstance(identity, str) or identity not in expected or identity in indexed:
                raise ValueError("foreign/duplicate token trace request")
            indexed[identity] = entry
        for identity, record in expected.items():
            if not record["completed"]:
                continue
            entry = indexed.get(identity)
            tokens = entry.get("tokens") if entry else None
            if (entry is None or entry.get("complete") is not True
                    or entry.get("wire_request_id") != record["wire_request_id"]
                    or not record["provider_response_id"]
                    or entry.get("provider_response_id") != record["provider_response_id"]
                    or not isinstance(tokens, list) or record["usage_reported_completion_tokens"] != len(tokens)):
                result["errors"].append(f"{identity}: incomplete/binding/count mismatch")
                continue
            times, ids = [], []
            for index, token in enumerate(tokens):
                if (not isinstance(token, dict) or token.get("index") != index
                        or isinstance(token.get("index"), bool) or not isinstance(token.get("token_id"), int)
                        or isinstance(token.get("token_id"), bool) or not 0 <= token["token_id"] <= 0xffffffff
                        or token.get("committed") is not True
                        or not number(token.get("committed_at_seconds"), 0, 1e15)
                        or (times and token["committed_at_seconds"] < times[-1])):
                    raise ValueError(f"{identity}: token order/ID/commit/clock mismatch")
                times.append(float(token["committed_at_seconds"]))
                ids.append(token["token_id"])
            intervals = [b - a for a, b in zip(times, times[1:])]
            result["requests"][identity] = dict(token_ids=ids, intervals_seconds=intervals)
            result["intervals_seconds"].extend(intervals)
        complete = sum(r["completed"] for r in records)
        result["status"] = ("provider_bound_complete" if complete > 0 and len(result["requests"]) == complete
                            else "incomplete")
    except (ValueError, TypeError, KeyError) as exc:
        result["status"] = "invalid"
        result["errors"].append(str(exc))
        result["requests"], result["intervals_seconds"] = {}, []
    return result


def bench(url: str, model: str, prompt: str, count: int, concurrency: int,
          max_tokens: int, timeout: float, key: str | None, *, workload: list[dict] | None = None,
          ttft_slo_seconds: float | None = None, chunk_slo_seconds: float | None = None,
          itl_slo_seconds: float | None = None, token_trace: object = None,
          token_trace_path: str | None = None, snapshot_url: str | None = None,
          run_id: str | None = None, max_pending: int | None = None,
          launch_deadline_seconds: float | None = None,
          return_token_ids: bool = False) -> dict:
    if not 1 <= count <= 1024 or not 1 <= concurrency <= min(64, count):
        raise ValueError("bounded requests/concurrency required (1..1024, 1..64)")
    if not 1 <= max_tokens <= 512 or not number(timeout, 1, 600):
        raise ValueError("bounded token/deadline limits")
    for bound in [ttft_slo_seconds, chunk_slo_seconds, itl_slo_seconds, launch_deadline_seconds]:
        if bound is not None and not number(bound, .000001, 600):
            raise ValueError("positive SLO/deadline <=600 seconds required")
    if token_trace is not None and token_trace_path is not None:
        raise ValueError("choose a token trace object or path")
    run_id = run_id or uuid.uuid4().hex
    if not label(run_id):
        raise ValueError("invalid bounded run ID")
    pending_limit = max_pending if max_pending is not None else min(count, concurrency + 64)
    if not concurrency <= pending_limit <= 1024:
        raise ValueError("max pending must be concurrency..1024")
    if workload is not None:
        items = workload_records(workload, max_tokens)
        if len(items) != count:
            raise ValueError("workload count differs from offered count")
        mode = "open_loop_scheduled_arrivals"
    else:
        if not prompt or len(prompt.encode()) > 32 * 1024:
            raise ValueError("nonempty prompt up to 32 KiB required")
        items = [dict(request_id=f"request-{i}", prompt=prompt, arrival_seconds=0., tenant="default",
                      max_tokens=max_tokens, input_index=i) for i in range(count)]
        mode = "closed_loop_bounded_concurrency"
    before = snapshot(snapshot_url, key)
    began = time.monotonic()
    records = []

    def make_record(item, out, offered, submitted, dispatched, wire_id):
        backlog = max(0., dispatched - offered)
        ttft = None if out.ttft is None else backlog + out.ttft
        chunks = out.itls
        ttft_met = None if ttft_slo_seconds is None else ttft is not None and ttft <= ttft_slo_seconds
        chunk_met = None if chunk_slo_seconds is None or not chunks else max(chunks) <= chunk_slo_seconds
        return dict(request_id=item["request_id"], wire_request_id=wire_id, tenant=item["tenant"],
                    input_index=item["input_index"], prompt_sha256=hashlib.sha256(item["prompt"].encode()).hexdigest(),
                    max_tokens=item["max_tokens"], offered_at_seconds=offered, submitted_at_seconds=submitted,
                    dispatched_at_seconds=dispatched, client_backlog_seconds=backlog,
                    service_elapsed_seconds=out.elapsed, elapsed_seconds=backlog + out.elapsed,
                    ttft_seconds=ttft, service_ttft_seconds=out.ttft,
                    sse_chunk_times_from_dispatch_seconds=out.chunk_times, sse_chunk_intervals_seconds=chunks,
                    ttft_slo_met=ttft_met, chunk_slo_met=chunk_met, completed=out.completed,
                    classification=out.classification, http_status=out.http_status, http_accepted=out.http_accepted,
                    http_attempted=out.classification not in {"client_overload", "launch_deadline"},
                    finish_reason=out.finish_reason, provider_response_id=out.provider_response_id,
                    provider_error_code=out.provider_error_code, usage_reported_completion_tokens=out.tokens,
                    error=out.error, output_sha256=out.output_sha256,
                    wire_token_observation=out.token_observation)

    def execute(item, submitted, planned):
        dispatched = time.monotonic() - began
        offered = dispatched if planned is None else planned
        wire_id = f"{run_id}:{item['request_id']}"
        if planned is not None and launch_deadline_seconds is not None and dispatched - planned >= launch_deadline_seconds:
            out = Outcome(False, None, 0, None, [], None, "launch_deadline", classification="launch_deadline")
        else:
            out = stream_one(url, model, item["prompt"], item["max_tokens"], timeout, key, request_id=wire_id,
                             return_token_ids=return_token_ids)
        return make_record(item, out, offered, submitted, dispatched, wire_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        if workload is None:
            remaining, pending = iter(items), set()
            for _ in range(concurrency):
                pending.add(pool.submit(execute, next(remaining), time.monotonic() - began, None))
            while pending:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    records.append(future.result())
                    item = next(remaining, None)
                    if item is not None:
                        pending.add(pool.submit(execute, item, time.monotonic() - began, None))
        else:
            pending = set()
            for item in sorted(items, key=lambda r: (r["arrival_seconds"], r["input_index"])):
                delay = began + item["arrival_seconds"] - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                done = {f for f in pending if f.done()}
                records.extend(f.result() for f in done)
                pending -= done
                submitted = time.monotonic() - began
                if len(pending) >= pending_limit:
                    out = Outcome(False, None, 0, None, [], None, "client_pending_limit", classification="client_overload")
                    records.append(make_record(item, out, item["arrival_seconds"], submitted, submitted,
                                               f"{run_id}:{item['request_id']}"))
                else:
                    pending.add(pool.submit(execute, item, submitted, item["arrival_seconds"]))
            records.extend(f.result() for f in concurrent.futures.as_completed(pending))
    elapsed = time.monotonic() - began
    records.sort(key=lambda r: r["input_index"])
    after = snapshot(snapshot_url, key)
    if token_trace_path:
        token_trace = strict_json(token_trace_path)
    trace = validate_token_trace(token_trace, records, run_id)
    completed = [r for r in records if r["completed"]]
    known = sum(r["usage_reported_completion_tokens"] for r in completed if r["usage_reported_completion_tokens"] is not None)
    unknown = sum(r["usage_reported_completion_tokens"] is None for r in completed)
    errors, classes = {}, {}
    for row in records:
        classes[row["classification"]] = classes.get(row["classification"], 0) + 1
        if row["error"]:
            errors[row["error"]] = errors.get(row["error"], 0) + 1
    proxy = [r for r in completed if r["ttft_slo_met"] is True and r["chunk_slo_met"] is True]
    trace_ready, true_slo = trace["status"] == "provider_bound_complete", []
    for row in records:
        evidence = trace["requests"].get(row["request_id"])
        row["provider_token_trace_complete"] = evidence is not None
        row["provider_token_itl_slo_met"] = (None if evidence is None or itl_slo_seconds is None else
                                            all(x <= itl_slo_seconds for x in evidence["intervals_seconds"]))
        if row["completed"] and row["ttft_slo_met"] is True and row["provider_token_itl_slo_met"] is True:
            true_slo.append(row)
    slo_configured = ttft_slo_seconds is not None and itl_slo_seconds is not None
    trace_tokens = sum(r["usage_reported_completion_tokens"] for r in true_slo)
    individual_rows, commit_rows, delivery_slo = [], [], []
    for row in completed:
        observed = row["wire_token_observation"] or {}
        individual = observed.get("client_delivery_status") == "individual_token_ids_observed"
        commit = observed.get("provider_commit_observation_status") == "provider_bound_complete_commit_observations"
        first = observed.get("client_first_token_seconds")
        row["client_token_ttft_seconds"] = row["client_backlog_seconds"] + first if first is not None else None
        row["client_token_itl_slo_met"] = (all(x <= itl_slo_seconds for x in observed["client_token_delivery_intervals_seconds"])
            if individual and itl_slo_seconds is not None else None)
        if individual:
            individual_rows.append(row)
            if (ttft_slo_seconds is not None and row["client_token_ttft_seconds"] is not None
                    and row["client_token_ttft_seconds"] <= ttft_slo_seconds and row["client_token_itl_slo_met"] is True):
                delivery_slo.append(row)
        if commit:
            commit_rows.append(row)
    fairness = []
    for tenant in sorted({r["tenant"] for r in records}):
        rows = [r for r in records if r["tenant"] == tenant]
        good = [r for r in rows if r["completed"]]
        fairness.append(dict(tenant=tenant, offered_requests=len(rows), completed_requests=len(good),
            completion_fraction=len(good) / len(rows),
            http_accepted_requests=sum(r["http_accepted"] for r in rows),
            unsuccessful_requests=len(rows) - len(good),
            http_429_requests=sum(r["http_status"] == 429 for r in rows),
            http_503_requests=sum(r["http_status"] == 503 for r in rows),
            cancelled_requests=sum(r["classification"] == "cancelled" for r in rows),
            timed_out_requests=sum(r["classification"] == "timed_out" for r in rows),
            latency_p95_seconds=percentile([r["elapsed_seconds"] for r in good], .95),
            ttft_p95_seconds=percentile([r["ttft_seconds"] for r in good if r["ttft_seconds"] is not None], .95),
            interpretation="workload_labels_not_authenticated_server_tenants"))
    return {
        "schema": "zerogravity.serving.benchmark.v2", "run_id": run_id, "endpoint": url.rstrip("/"),
        "model": model, "concurrency": concurrency, "arrival_mode": mode, "max_pending": pending_limit,
        "launch_deadline_seconds": launch_deadline_seconds, "offered_requests": count,
        "completed_requests": len(completed), "http_accepted_requests": sum(r["http_accepted"] for r in records),
        "http_429_requests": sum(r["http_status"] == 429 for r in records),
        "http_503_requests": sum(r["http_status"] == 503 for r in records),
        "rejected_requests": sum(r["classification"] in {"rejected_429", "rejected_503", "http_error"} for r in records),
        "cancelled_requests": classes.get("cancelled", 0), "timed_out_requests": classes.get("timed_out", 0),
        "client_overload_requests": classes.get("client_overload", 0), "launch_deadline_requests": classes.get("launch_deadline", 0),
        "unsuccessful_requests": count - len(completed), "classifications": classes, "errors": errors,
        "output_sha256_by_request": [r["output_sha256"] for r in records], "wall_seconds": round(elapsed, 5),
        "offered_window_seconds": elapsed,
        "admission_accounting": "HTTP_status_only_server_admission_requires_provider_trace",
        "denominator": "all_scheduled_offers_through_last_completion_including_client_overload_and_timeouts",
        "latency_p50_seconds": percentile([r["elapsed_seconds"] for r in completed], .50),
        "latency_p95_seconds": percentile([r["elapsed_seconds"] for r in completed], .95),
        "ttft_p50_seconds": percentile([r["ttft_seconds"] for r in completed if r["ttft_seconds"] is not None], .50),
        "ttft_p95_seconds": percentile([r["ttft_seconds"] for r in completed if r["ttft_seconds"] is not None], .95),
        "sse_chunk_interval_p95_seconds": percentile([x for r in completed for x in r["sse_chunk_intervals_seconds"]], .95),
        "usage_reported_completion_tokens": known, "completions_without_usage": unknown,
        "reported_tokens_per_offered_second": round(known / elapsed, 4) if elapsed > 0 and not unknown else None,
        "proxy_slo": {"semantics": "TTFT_and_SSE_chunks_not_token_ITL_or_qualified_goodput",
                      "ttft_limit_seconds": ttft_slo_seconds, "chunk_limit_seconds": chunk_slo_seconds,
                      "requests_meeting_both": len(proxy),
                      "usage_tokens_meeting_both": sum(r["usage_reported_completion_tokens"] or 0 for r in proxy)},
        "provider_token_trace": trace,
        "wire_token_metrics": {"requested":return_token_ids,
            "completed_requests_with_individual_token_delivery":len(individual_rows),
            "completed_requests_with_provider_commit_observations":len(commit_rows),
            "client_token_ttft_p50_seconds":percentile([r["client_token_ttft_seconds"] for r in individual_rows if r["client_token_ttft_seconds"] is not None], .50),
            "client_token_ttft_p95_seconds":percentile([r["client_token_ttft_seconds"] for r in individual_rows if r["client_token_ttft_seconds"] is not None], .95),
            "client_token_delivery_itl_p50_seconds":percentile([x for r in individual_rows for x in r["wire_token_observation"]["client_token_delivery_intervals_seconds"]], .50),
            "client_token_delivery_itl_p95_seconds":percentile([x for r in individual_rows for x in r["wire_token_observation"]["client_token_delivery_intervals_seconds"]], .95),
            "provider_commit_observed_itl_p95_seconds":percentile([x for r in commit_rows for x in r["wire_token_observation"]["provider_commit_observation_intervals_seconds"]], .95),
            "client_token_delivery_slo_requests":len(delivery_slo) if slo_configured else None,
            "client_token_delivery_slo_usage_tokens_per_offered_second":sum(r["usage_reported_completion_tokens"] for r in delivery_slo) / elapsed if slo_configured and elapsed > 0 else None,
            "qualified_source_correct_slo_goodput":None,
            "semantics":"actual single-ID delivery events, not content chunks; server aggregate mean ITL and provider commit observations remain separate; unresolved/coalesced/absent IDs cannot establish individual token ITL"},
        "provider_reported_true_token_itl_p95_seconds": percentile(trace["intervals_seconds"], .95) if trace_ready else None,
        "provider_trace_slo": {"ttft_limit_seconds": ttft_slo_seconds, "token_itl_limit_seconds": itl_slo_seconds,
            "status": "provider_bound_complete" if trace_ready and slo_configured else "not_established",
            "requests_meeting_both": len(true_slo) if trace_ready and slo_configured else None,
            "provider_reported_committed_tokens_per_offered_second": trace_tokens / elapsed if trace_ready and slo_configured and elapsed > 0 else None,
            "qualified_source_correct_slo_goodput": None},
        "fairness_by_workload_tenant": fairness,
        "provider_snapshots": {"before": before, "after": after,
            "interpretation": "raw_provider_JSON_not_independent_physical_memory_or_dispatch_proof"},
        "requests": records, "numerical_source_equivalence": "not_verified", "fused_kernels_or_vllm_parity": "not_inferred",
        "note": "SSE chunks do not establish token ITL. Provider traces require independent commit/source qualification.",
    }


def self_test() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import copy
    groups = [dict(token_ids=[4],client_received_at_seconds=.1),dict(token_ids=[5],client_received_at_seconds=.3)]
    metadata = dict(schema="zerogravity.serving.request-token-metrics.v1", clock="request_monotonic_seconds",
        wire_request_id="run:request",provider_response_id="chatcmpl-zg-1",complete=True,completion_tokens=2,
        tokens=[dict(index=i,token_id=4+i,commit_observed_at_seconds=.01+i*.01,
            output_received_at_seconds=.04+i*.1,http_write_completed_at_seconds=.1+i*.2) for i in range(2)])
    observed = wire_token_observation(groups,metadata,{"mean_itl_ms":99},"run:request","chatcmpl-zg-1",2,2)
    assert observed["status"] == "provider_bound_complete_ids"
    assert abs(observed["client_token_delivery_intervals_seconds"][0]-.2) < 1e-9
    assert abs(observed["provider_commit_observation_intervals_seconds"][0]-.01) < 1e-9
    assert observed["provider_aggregate_metrics"]["mean_itl_ms"] == 99
    gpu_metadata = copy.deepcopy(metadata)
    for index,record in enumerate(gpu_metadata["tokens"]):
        record["gpu"] = dict(producer_frontier=7+index,
            native_cohort=dict(publication_sequence=2+index,width=2))
    gpu_observed = wire_token_observation(groups,gpu_metadata,None,"run:request","chatcmpl-zg-1",2,2)
    assert gpu_observed["provider_token_records"][1]["gpu"]["producer_frontier"] == 8
    for field,value in [("producer_frontier",7),("producer_frontier",True),("native_cohort",dict(publication_sequence=2,width=2)),("native_cohort",dict(publication_sequence=3,width=True)),("native_cohort",dict(publication_sequence=3,width=3))]:
        bad = copy.deepcopy(gpu_metadata);bad["tokens"][1]["gpu"][field]=value
        try:
            wire_token_observation(groups,bad,None,"run:request","chatcmpl-zg-1",2,2)
            raise AssertionError("invalid/repeated GPU producer observation accepted")
        except ValueError:
            pass
    for count in [True,-1,3,1.5]:
        try:
            wire_token_observation(groups,None,None,"run:request","chatcmpl-run:request",count,2)
            raise AssertionError("invalid token usage accepted")
        except ValueError:
            pass
    coalesced = wire_token_observation([dict(token_ids=[4,5],client_received_at_seconds=.3)],metadata,None,"run:request","chatcmpl-zg-1",2,2)
    assert coalesced["client_token_delivery_intervals_seconds"] is None
    assert coalesced["client_delivery_status"] == "unresolved_token_groups_or_binding"
    vllm = wire_token_observation(groups,None,{"mean_itl_ms":99},"run:request","chatcmpl-run:request",2,2)
    assert vllm["client_delivery_status"] == "individual_token_ids_observed"
    assert vllm["provider_commit_observation_intervals_seconds"] is None
    missing = wire_token_observation([],None,None,"run:request","chatcmpl-run:request",2,2)
    assert missing["client_token_delivery_intervals_seconds"] is None
    for field,value in [("wire_request_id","foreign"),("provider_response_id","foreign"),("completion_tokens",1),("clock","wallclock")]:
        bad = copy.deepcopy(metadata); bad[field]=value
        try:
            wire_token_observation(groups,bad,None,"run:request","chatcmpl-zg-1",2,2)
            raise AssertionError("foreign/incomplete metadata accepted")
        except ValueError:
            pass
    for wire,provider in [(None,None),("", ""),("run:request",None),(None,"chatcmpl-zg-1")]:
        bad = copy.deepcopy(metadata); bad["wire_request_id"]=wire;bad["provider_response_id"]=provider
        try:
            wire_token_observation(groups,bad,None,wire,provider,2,2)
            raise AssertionError("null/empty identity promoted to bound evidence")
        except ValueError:
            pass
    for field,value in [("index",0),("token_id",6),("commit_observed_at_seconds",-.1),("output_received_at_seconds",.001),("output_received_at_seconds",.03),("output_received_at_seconds",.05),("http_write_completed_at_seconds",.001)]:
        bad = copy.deepcopy(metadata); bad["tokens"][1][field]=value
        try:
            wire_token_observation(groups,bad,None,"run:request","chatcmpl-zg-1",2,2)
            raise AssertionError("invalid token record accepted")
        except ValueError:
            pass
    for invalid_id in [-1,0x100000000,True,1.5]:
        try:
            wire_token_observation([dict(token_ids=[invalid_id],client_received_at_seconds=.1)],None,None,"r","chatcmpl-r",1,1)
            raise AssertionError("invalid token ID accepted")
        except ValueError:
            pass

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["model"] == "fixture"
            prompt = body["messages"][0]["content"]
            if prompt in {"reject429", "reject503", "cancel499"}:
                self.send_error(int(prompt[-3:]))
                return
            if prompt == "slow":
                time.sleep(.05)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = [
                {"id": self.headers.get("X-Request-ID"), "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
                {"choices": [], "usage": {"completion_tokens": 2}},
            ]
            if body.get("return_token_ids"):
                wire = self.headers["X-Request-ID"]
                provider = f"chatcmpl-{wire}"
                for chunk in chunks:
                    chunk["id"] = provider
                chunks[0]["choices"][0]["delta"]["content"] = ""
                chunks[1]["choices"][0]["delta"]["content"] = "é"
                chunks[0]["choices"][0]["token_ids"] = [4]
                chunks[1]["choices"][0]["token_ids"] = [5]
                if prompt == "boolindex":
                    chunks[0]["choices"][0]["index"] = False
                chunks[3]["metrics"] = {"mean_itl_ms":99}
            try:
                for chunk in chunks:
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
            except (BrokenPipeError, ConnectionResetError):
                # The invalid-index fixture deliberately closes its stream early.
                if prompt != "boolindex":
                    raise
            self.close_connection = True

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    task = threading.Thread(target=server.serve_forever, daemon=True)
    task.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        result = bench(endpoint, "fixture", "hello", 4, 2, 2, 8, None, ttft_slo_seconds=8, chunk_slo_seconds=8)
        assert result["offered_requests"] == result["completed_requests"] == 4
        assert result["http_accepted_requests"] == 4 and result["reported_tokens_per_offered_second"] is not None
        assert len(set(result["output_sha256_by_request"])) == 1
        assert result["proxy_slo"]["requests_meeting_both"] == 4
        assert result["provider_trace_slo"]["status"] == "not_established"
        assert result["provider_reported_true_token_itl_p95_seconds"] is None
        actual_ids = bench(endpoint,"fixture","wire",1,1,2,8,None,return_token_ids=True,
            ttft_slo_seconds=8,itl_slo_seconds=8)
        assert actual_ids["wire_token_metrics"]["completed_requests_with_individual_token_delivery"] == 1
        assert actual_ids["wire_token_metrics"]["completed_requests_with_provider_commit_observations"] == 0
        assert actual_ids["requests"][0]["wire_token_observation"]["token_ids"] == [4,5]
        assert actual_ids["wire_token_metrics"]["client_token_delivery_slo_requests"] == 1
        assert actual_ids["wire_token_metrics"]["qualified_source_correct_slo_goodput"] is None
        assert actual_ids["requests"][0]["client_token_ttft_seconds"] <= actual_ids["requests"][0]["ttft_seconds"]
        bad_index = bench(endpoint,"fixture","boolindex",1,1,2,8,None,return_token_ids=True)
        assert bad_index["completed_requests"] == 0 and bad_index["classifications"]["protocol_error"] == 1
        work = [dict(request_id=f"r{i}", prompt=p, arrival_seconds=0, tenant=f"tenant-{i % 2}")
                for i, p in enumerate(["hello", "reject429", "reject503", "cancel499"])]
        mixed = bench(endpoint, "fixture", "unused", 4, 2, 2, 8, None, workload=work)
        assert mixed["offered_requests"] == 4 and mixed["completed_requests"] == 1
        assert mixed["http_429_requests"] == mixed["http_503_requests"] == 1
        assert mixed["rejected_requests"] == 2 and mixed["cancelled_requests"] == 1
        assert mixed["http_accepted_requests"] == 1 and len(mixed["requests"]) == 4
        assert abs(mixed["reported_tokens_per_offered_second"] - 2 / mixed["offered_window_seconds"]) < .0001
        overloaded = bench(endpoint, "fixture", "unused", 4, 1, 2, 8, None,
            workload=[dict(request_id=f"r{i}", prompt="slow", arrival_seconds=0) for i in range(4)], max_pending=1)
        assert overloaded["offered_requests"] == 4 and overloaded["client_overload_requests"] == 3
        assert len(overloaded["requests"]) == 4 and overloaded["completed_requests"] == 1
        trace = dict(schema="zerogravity.serving.token-trace.v1", run_id=result["run_id"],
            clock="provider_monotonic_seconds", event_semantics="committed_output_token", requests=[])
        for row in result["requests"]:
            trace["requests"].append(dict(request_id=row["request_id"], wire_request_id=row["wire_request_id"],
                provider_response_id=row["provider_response_id"], complete=True,
                tokens=[dict(index=i, token_id=10 + i, committed=True, committed_at_seconds=float(i)) for i in range(2)]))
        assert validate_token_trace(trace, result["requests"], result["run_id"])["status"] == "provider_bound_complete"
        trace["requests"][0]["provider_response_id"] = "foreign-provider-request"
        assert validate_token_trace(trace, result["requests"], result["run_id"])["status"] == "incomplete"
        trace["requests"][0]["provider_response_id"] = result["requests"][0]["provider_response_id"]
        trace["requests"][0]["tokens"][1]["committed_at_seconds"] = -1
        assert validate_token_trace(trace, result["requests"], result["run_id"])["status"] == "invalid"
        trace["requests"][0]["tokens"][1]["committed_at_seconds"] = 1
        trace["requests"].pop()
        assert validate_token_trace(trace, result["requests"], result["run_id"])["status"] == "incomplete"
        one_trace = dict(schema="zerogravity.serving.token-trace.v1", run_id="fixture-trace",
            clock="provider_monotonic_seconds", event_semantics="committed_output_token",
            requests=[dict(request_id="request-0", wire_request_id="fixture-trace:request-0",
                provider_response_id="fixture-trace:request-0", complete=True,
                tokens=[dict(index=i, token_id=10 + i, committed=True, committed_at_seconds=float(i)) for i in range(2)])])
        traced = bench(endpoint, "fixture", "hello", 1, 1, 2, 8, None,
            run_id="fixture-trace", token_trace=one_trace,
            ttft_slo_seconds=8, chunk_slo_seconds=8, itl_slo_seconds=.25)
        assert traced["proxy_slo"]["requests_meeting_both"] == 1
        assert traced["provider_trace_slo"]["status"] == "provider_bound_complete"
        assert traced["provider_trace_slo"]["requests_meeting_both"] == 0
        assert traced["provider_trace_slo"]["provider_reported_committed_tokens_per_offered_second"] == 0
        assert traced["provider_trace_slo"]["qualified_source_correct_slo_goodput"] is None
        for bad in [[dict(request_id="same", prompt="a", arrival_seconds=0)] * 2,
                    [dict(prompt="a", arrival_seconds=float("nan"))],
                    [dict(prompt="a", arrival_seconds=True)],
                    [dict(prompt="a", arrival_seconds=0, unknown=True)]]:
            try:
                workload_records(bad, 2)
                raise AssertionError("invalid workload accepted")
            except ValueError:
                pass
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
    parser.add_argument("--workload", help="strict JSON requests: request_id,prompt,arrival_seconds,tenant,max_tokens")
    parser.add_argument("--ttft-slo", type=float)
    parser.add_argument("--chunk-slo", type=float, help="SSE chunk proxy; never token ITL")
    parser.add_argument("--itl-slo", type=float, help="requires complete provider token trace")
    parser.add_argument("--token-events", help="provider token-trace.v1 JSON; read after requests complete")
    parser.add_argument("--return-token-ids", action="store_true", help="request actual token IDs; validate individual delivery and optional provider commit observations")
    parser.add_argument("--provider-snapshot-url", help="raw before/after literal loopback HTTP JSON")
    parser.add_argument("--run-id", help="trace binding; default unique random run")
    parser.add_argument("--max-pending", type=int)
    parser.add_argument("--launch-deadline", type=float, help="open-loop client wait deadline before dispatch")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.model:
        parser.error("--model is required")
    import os
    try:
        work = workload_records(strict_json(args.workload), args.max_tokens) if args.workload else None
        if work is not None:
            work = [{k: v for k, v in row.items() if k != "input_index"} for row in work]
        result = bench(args.endpoint, args.model, args.prompt, len(work) if work is not None else args.requests,
            args.concurrency, args.max_tokens, args.timeout, os.environ.get(args.api_key_env), workload=work,
            ttft_slo_seconds=args.ttft_slo, chunk_slo_seconds=args.chunk_slo, itl_slo_seconds=args.itl_slo,
            token_trace_path=args.token_events, snapshot_url=args.provider_snapshot_url, run_id=args.run_id,
            max_pending=args.max_pending, launch_deadline_seconds=args.launch_deadline,
            return_token_ids=args.return_token_ids)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Cannot benchmark: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
