#!/usr/bin/env python3
"""Paired, source-attested OpenAI serving comparison with vLLM.

This script benchmarks two *already running* OpenAI chat endpoints.
It installs no SDK/models, does not infer model source identity from
display names, and does not promote operator-stage timings to
full-model/vLLM parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import statistics
from pathlib import Path
import urllib.request


BENCH = runpy.run_path(str(Path(__file__).with_name("serve-bench.py")))
bench = BENCH["bench"]
stream_one = BENCH["stream_one"]
strict_json = BENCH["strict_json"]
workload_records = BENCH["workload_records"]


def valid_sha(value: str | None) -> bool:
    return value is not None and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def preflight(url: str, model: str, key: str | None) -> None:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    request = urllib.request.Request(url.rstrip("/") + "/v1/models", headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        data = json.load(response)
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError(f"{url}: not an OpenAI model listing")
    if model not in {m.get("id") for m in data["data"] if isinstance(m, dict)}:
        raise ValueError(f"{url}: required route {model!r} is not registered")


def comparison(
    zg: str, vllm: str, zg_model: str, vllm_model: str, prompt: str,
    requests: int, concurrencies: list[int], max_tokens: int,
    zg_key: str | None, vllm_key: str | None,
    source_hashes: tuple[str | None, str | None],
    tokenizer_hashes: tuple[str | None, str | None],
    repetitions: int, timeout: float, *, workload: list[dict] | None = None,
    ttft_slo_seconds: float | None = None, chunk_slo_seconds: float | None = None,
    itl_slo_seconds: float | None = None,
    snapshot_urls: tuple[str | None, str | None] = (None, None),
    token_trace_paths: tuple[str | None, str | None] = (None, None),
    max_pending: int | None = None, launch_deadline_seconds: float | None = None,
    return_token_ids: bool = False,
) -> dict:
    if zg.rstrip("/") == vllm.rstrip("/"):
        raise ValueError("comparator needs two distinct endpoints")
    if (not 1 <= repetitions <= 4 or not 1 <= len(concurrencies) <= 8
            or len(set(concurrencies)) != len(concurrencies)):
        raise ValueError("repetitions 1..4; 1..8 distinct concurrency values")
    if requests < 1 or requests > 128 or not all(
        1 <= n <= min(requests, 8) for n in concurrencies
    ):
        raise ValueError("comparison requests 1..128 and concurrent 1..8")
    if not 1 <= max_tokens <= 512 or not 1 <= timeout <= 600:
        raise ValueError("comparison timeout/token bounds")
    if not prompt or len(prompt.encode()) > 32 * 1024:
        raise ValueError("comparison prompt out of bounds")
    if workload is not None:
        checked = workload_records(workload, max_tokens)
        if len(checked) != requests:
            raise ValueError("workload count differs from comparison requests")
        workload = [{k: v for k, v in row.items() if k != "input_index"} for row in checked]

    endpoints = {
        "zerogravity": (zg, zg_model, zg_key),
        "vllm": (vllm, vllm_model, vllm_key),
    }
    for url, model, key in endpoints.values():
        preflight(url, model, key)
        warm = stream_one(url, model, workload[0]["prompt"] if workload else prompt,
                          workload[0]["max_tokens"] if workload else max_tokens, timeout, key,
                          request_id="serve-compare-warmup", return_token_ids=return_token_ids)
        if not warm.completed:
            raise ValueError(f"warmup failed for {model}: {warm.error}")

    matched_source = (
        all(valid_sha(x) for x in source_hashes)
        and source_hashes[0] == source_hashes[1]
    )
    matched_tokenizer = (
        all(valid_sha(x) for x in tokenizer_hashes)
        and tokenizer_hashes[0] == tokenizer_hashes[1]
    )
    rows = []
    for concurrency in concurrencies:
        for rep in range(repetitions):
            # Each ABBA has equal order opportunities, no simultaneous
            # GPU interference. Alternate first service by repetition.
            order = ["zerogravity", "vllm", "vllm", "zerogravity"]
            if rep % 2:
                order = ["vllm", "zerogravity", "zerogravity", "vllm"]
            for position, label in enumerate(order):
                url, model, key = endpoints[label]
                provider_index = 0 if label == "zerogravity" else 1
                result = bench(
                    url, model, prompt, requests, concurrency,
                    max_tokens, timeout, key, workload=workload,
                    ttft_slo_seconds=ttft_slo_seconds, chunk_slo_seconds=chunk_slo_seconds,
                    itl_slo_seconds=itl_slo_seconds, snapshot_url=snapshot_urls[provider_index],
                    token_trace_path=token_trace_paths[provider_index], max_pending=max_pending,
                    launch_deadline_seconds=launch_deadline_seconds,
                    return_token_ids=return_token_ids,
                )
                result["provider"] = label
                result["repetition"] = rep
                result["order_position"] = position
                result["paired_order"] = "BAAB" if rep % 2 else "ABBA"
                rows.append(result)

    summaries = []
    for concurrency in concurrencies:
        group = [r for r in rows if r["concurrency"] == concurrency]
        per = {
            label: [r for r in group if r["provider"] == label]
            for label in endpoints
        }
        successful = all(
            r["completed_requests"] == requests
            and r["completions_without_usage"] == 0
            and r["reported_tokens_per_offered_second"] is not None
            for samples in per.values() for r in samples
        )
        # Match each workload identity separately: different prompts may have
        # different correct outputs. Usage/finish equality is a separate gate.
        def signature(sample, include_usage=False):
            return {r["request_id"]: (r["prompt_sha256"], r["max_tokens"], r["output_sha256"])
                    + ((r["usage_reported_completion_tokens"], r["finish_reason"]) if include_usage else ())
                    for r in sample["requests"]}
        baseline = signature(per["zerogravity"][0])
        baseline_usage = signature(per["zerogravity"][0], True)
        exact_text = successful and all(signature(r) == baseline for r in group)
        exact_usage = successful and all(signature(r, True) == baseline_usage for r in group)
        trace_complete = all(r["provider_token_trace"]["status"] == "provider_bound_complete" for r in group)
        def token_signature(sample):
            return {identity: evidence["token_ids"] for identity, evidence in
                    sample["provider_token_trace"]["requests"].items()}
        exact_tokens = (trace_complete and all(token_signature(r) == token_signature(group[0]) for r in group))
        wire_ids_complete = successful and all(
            row.get("wire_token_observation") is not None
            and row["wire_token_observation"]["status"] == "provider_bound_complete_ids"
            for sample in group for row in sample["requests"])
        def wire_signature(sample):
            return {row["request_id"]: row["wire_token_observation"]["token_ids"] for row in sample["requests"]}
        wire_ids_match = wire_ids_complete and all(wire_signature(sample) == wire_signature(group[0]) for sample in group)
        comparable = (successful and exact_text and exact_usage and matched_source and matched_tokenizer
                      and (not return_token_ids or wire_ids_match))
        medians = {
            label: statistics.median(
                r["reported_tokens_per_offered_second"] for r in samples
                if r["reported_tokens_per_offered_second"] is not None
            ) if all(r["reported_tokens_per_offered_second"] is not None for r in samples)
            else None
            for label, samples in per.items()
        }
        rate_zg, rate_vllm = medians["zerogravity"], medians["vllm"]
        summaries.append({
            "concurrency": concurrency,
            "completed_all_requests": successful,
            "exact_output_text_sha256_matched": bool(exact_text),
            "exact_usage_and_finish_matched": bool(exact_usage),
            "provider_bound_output_token_ids_matched": bool(exact_tokens) if trace_complete else None,
            "response_bound_output_token_ids_matched": bool(wire_ids_match) if wire_ids_complete else None,
            "response_token_ids_complete": wire_ids_complete,
            "client_token_delivery_itl_p95_seconds_median_by_provider": {
                label: statistics.median(sample["wire_token_metrics"]["client_token_delivery_itl_p95_seconds"] for sample in samples)
                if all(sample["wire_token_metrics"]["completed_requests_with_individual_token_delivery"] == requests
                    and sample["wire_token_metrics"]["client_token_delivery_itl_p95_seconds"] is not None for sample in samples) else None
                for label,samples in per.items()},
            "source_sha256_declared_identical": bool(matched_source),
            "tokenizer_sha256_declared_identical": bool(matched_tokenizer),
            "comparable_with_caller_supplied_source_attestation": bool(comparable),
            "independent_source_numeric_state_equivalence": "NOT_ESTABLISHED",
            "zerogravity_tokens_per_second_median": rate_zg,
            "vllm_tokens_per_second_median": rate_vllm,
            "zerogravity_over_vllm_median_goodput_ratio":
                round(rate_zg / rate_vllm, 5)
                if comparable and rate_zg is not None and rate_vllm else None,
            "legacy_goodput_ratio_semantics": "usage_reported_throughput_not_SLO_goodput",
            "zerogravity_over_vllm_reported_throughput_ratio":
                round(rate_zg / rate_vllm, 5)
                if comparable and rate_zg is not None and rate_vllm else None,
            "qualified_source_correct_slo_goodput_ratio": None,
            "vllm_equivalent": "NOT_ESTABLISHED",
        })
    return {
        "schema": "zerogravity.serving.vllm-comparison.v2",
        "benchmark_input_sha256": hashlib.sha256(
            json.dumps(workload, sort_keys=True, separators=(",", ":")).encode()
            if workload is not None else prompt.encode()).hexdigest(),
        "model_routes": {"zerogravity": zg_model, "vllm": vllm_model},
        "attestation": {
            "source_hashes": list(source_hashes),
            "tokenizer_hashes": list(tokenizer_hashes),
            "trust": "caller_provided_not_independently_cryptographically_verified",
        },
        "run_order": "alternating_ABBA_BAAB",
        "order_balance_complete": repetitions % 2 == 0,
        "provider_order_by_repetition": [
            ["vllm", "zerogravity", "zerogravity", "vllm"] if rep % 2 else
            ["zerogravity", "vllm", "vllm", "zerogravity"] for rep in range(repetitions)],
        "warmups_per_endpoint": 1,
        "runs": rows,
        "summaries": summaries,
        "limitations": [
            "Model name equality does not establish identical learned weights.",
            "Caller source/tokenizer hashes must be independently verified.",
            "SSE chunks do not establish token-level ITL.",
            "Provider token traces validate binding/count/order, not independent physical commits or numerical correctness.",
            "Throughput ratios, including the legacy goodput field, are not qualified SLO-goodput ratios.",
            "Equivalent quantization, physical GPU, driver and policies need external qualification.",
            "No vLLM-equivalent claim follows from faster median rates alone.",
            "Confidence intervals are not estimated; retain raw paired runs for independent statistical analysis.",
        ],
    }


def self_test() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    actual_calls = []
    call_lock = threading.Lock()

    class Fake(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/v1/models":
                self.send_error(404)
                return
            raw = json.dumps({"object": "list", "data": [{"id": "fixture"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            with call_lock:
                actual_calls.append(self.server.provider_label)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["model"] == "fixture"
            prompt = body["messages"][0]["content"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            docs = [
                {"choices": [{"index": 0, "delta": {"content": "yes" if prompt == "Hi." else prompt}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
                {"choices": [], "usage": {"completion_tokens": 1}},
            ]
            if body.get("return_token_ids"):
                provider_id = "chatcmpl-" + self.headers["X-Request-ID"]
                for doc in docs:
                    doc["id"] = provider_id
                docs[0]["choices"][0]["token_ids"] = [10]
                docs[-1]["metrics"] = {"mean_itl_ms":None}
            for doc in docs:
                self.wfile.write(b"data: " + json.dumps(doc).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *_args):
            pass

    servers = [ThreadingHTTPServer(("127.0.0.1", 0), Fake) for _ in range(2)]
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
    for index, server in enumerate(servers):
        server.provider_label = "zerogravity" if index == 0 else "vllm"
    for thread in threads:
        thread.start()
    try:
        a,b = [f"http://127.0.0.1:{s.server_port}" for s in servers]
        result = comparison(
            a,b,"fixture","fixture","Hi.",2,[1,2],1,None,None,
            ("a"*64,"a"*64),("b"*64,"b"*64),2,10
        )
        assert len(result["runs"]) == 16
        expected_order = ["zerogravity", "vllm", "vllm", "zerogravity",
                          "vllm", "zerogravity", "zerogravity", "vllm"] * 2
        assert [row["provider"] for row in result["runs"]] == expected_order
        # Verify real HTTP endpoint order, excluding one warmup per provider.
        assert actual_calls[:2] == ["zerogravity", "vllm"]
        assert actual_calls[2:] == [provider for provider in expected_order for _ in range(2)]
        assert all(row["completed_all_requests"] for row in result["summaries"])
        assert all(row["exact_output_text_sha256_matched"] for row in result["summaries"])
        assert all(row["comparable_with_caller_supplied_source_attestation"]
                   for row in result["summaries"])
        assert all(row["vllm_equivalent"]=="NOT_ESTABLISHED" for row in result["summaries"])
        assert all(row["qualified_source_correct_slo_goodput_ratio"] is None for row in result["summaries"])
        varied = comparison(a,b,"fixture","fixture","unused",2,[2],1,None,None,
            ("a"*64,"a"*64),("b"*64,"b"*64),1,10,
            workload=[dict(request_id="left", prompt="Left", arrival_seconds=0),
                      dict(request_id="right", prompt="Right", arrival_seconds=0)])
        assert varied["summaries"][0]["exact_output_text_sha256_matched"]
        assert len(set(varied["runs"][0]["output_sha256_by_request"])) == 2
        observed = comparison(a,b,"fixture","fixture","Hi.",1,[1],1,None,None,
            ("a"*64,"a"*64),("b"*64,"b"*64),1,10,return_token_ids=True)
        assert observed["summaries"][0]["response_bound_output_token_ids_matched"] is True
        assert observed["summaries"][0]["independent_source_numeric_state_equivalence"] == "NOT_ESTABLISHED"
        assert all(sample["wire_token_metrics"]["completed_requests_with_provider_commit_observations"] == 0 for sample in observed["runs"])
        different = comparison(
            a,b,"fixture","fixture","Hi.",1,[1],1,None,None,
            (None,None),(None,None),1,10
        )
        assert not different["summaries"][0]["comparable_with_caller_supplied_source_attestation"]
        try:
            comparison(a,a,"fixture","fixture","Hi.",1,[1],1,
                       None,None,(None,None),(None,None),1,10)
            raise AssertionError("same endpoint was accepted")
        except ValueError:
            pass
        print("SERVING_VLLM_COMPARATOR_SOURCE_FREE_SELF_TEST_PASS")
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=3)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zg-endpoint",default="http://127.0.0.1:8000")
    parser.add_argument("--vllm-endpoint",default="http://127.0.0.1:8001")
    parser.add_argument("--zg-model")
    parser.add_argument("--vllm-model")
    parser.add_argument("--prompt",default="Hi.")
    parser.add_argument("--requests",type=int,default=4)
    parser.add_argument("--concurrency",default="1,2")
    parser.add_argument("--max-tokens",type=int,default=8)
    parser.add_argument("--repetitions",type=int,default=1)
    parser.add_argument("--timeout",type=float,default=180.)
    parser.add_argument("--workload", help="shared strict JSON open-loop request workload")
    parser.add_argument("--ttft-slo", type=float)
    parser.add_argument("--chunk-slo", type=float, help="SSE chunk proxy, not token ITL")
    parser.add_argument("--itl-slo", type=float, help="requires complete bound provider token events")
    parser.add_argument("--zg-provider-snapshot-url")
    parser.add_argument("--vllm-provider-snapshot-url")
    parser.add_argument("--zg-token-events")
    parser.add_argument("--vllm-token-events")
    parser.add_argument("--return-token-ids", action="store_true", help="request and validate actual token IDs from both endpoints; coalesced ID groups do not establish individual delivery ITL")
    parser.add_argument("--max-pending", type=int)
    parser.add_argument("--launch-deadline", type=float)
    parser.add_argument("--zg-source-sha256")
    parser.add_argument("--vllm-source-sha256")
    parser.add_argument("--zg-tokenizer-sha256")
    parser.add_argument("--vllm-tokenizer-sha256")
    parser.add_argument("--zg-key-env",default="ZEROGRAVITY_API_KEY")
    parser.add_argument("--vllm-key-env",default="VLLM_API_KEY")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.zg_model or not args.vllm_model:
        parser.error("--zg-model and --vllm-model are required")
    import os
    try:
        concurrencies=list(dict.fromkeys(
            int(item.strip()) for item in args.concurrency.split(",")
        ))
        work = workload_records(strict_json(args.workload), args.max_tokens) if args.workload else None
        if work is not None:
            work = [{k: v for k, v in row.items() if k != "input_index"} for row in work]
        result=comparison(
            args.zg_endpoint,args.vllm_endpoint,args.zg_model,args.vllm_model,
            args.prompt,len(work) if work is not None else args.requests,concurrencies,args.max_tokens,
            os.environ.get(args.zg_key_env),os.environ.get(args.vllm_key_env),
            (args.zg_source_sha256,args.vllm_source_sha256),
            (args.zg_tokenizer_sha256,args.vllm_tokenizer_sha256),
            args.repetitions,args.timeout, workload=work,
            ttft_slo_seconds=args.ttft_slo, chunk_slo_seconds=args.chunk_slo,
            itl_slo_seconds=args.itl_slo,
            snapshot_urls=(args.zg_provider_snapshot_url, args.vllm_provider_snapshot_url),
            token_trace_paths=(args.zg_token_events, args.vllm_token_events),
            max_pending=args.max_pending, launch_deadline_seconds=args.launch_deadline,
            return_token_ids=args.return_token_ids,
        )
    except (OSError,ValueError,KeyError) as err:
        parser.exit(2,f"Cannot compare endpoints: {err}\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
