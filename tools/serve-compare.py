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
    repetitions: int, timeout: float,
) -> dict:
    if zg.rstrip("/") == vllm.rstrip("/"):
        raise ValueError("comparator needs two distinct endpoints")
    if not 1 <= repetitions <= 4 or not concurrencies:
        raise ValueError("repetitions must be 1..4 and concurrency list nonempty")
    if requests < 1 or requests > 128 or not all(
        1 <= n <= min(requests, 8) for n in concurrencies
    ):
        raise ValueError("comparison requests 1..128 and concurrent 1..8")
    if not 1 <= max_tokens <= 512 or not 1 <= timeout <= 600:
        raise ValueError("comparison timeout/token bounds")
    if not prompt or len(prompt.encode()) > 32 * 1024:
        raise ValueError("comparison prompt out of bounds")

    endpoints = {
        "zerogravity": (zg, zg_model, zg_key),
        "vllm": (vllm, vllm_model, vllm_key),
    }
    for url, model, key in endpoints.values():
        preflight(url, model, key)
        warm = stream_one(url, model, prompt, max_tokens, timeout, key)
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
                order.reverse()
            for label in order:
                url, model, key = endpoints[label]
                result = bench(
                    url, model, prompt, requests, concurrency,
                    max_tokens, timeout, key
                )
                result["provider"] = label
                result["repetition"] = rep
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
        outputs = {
            label: [hash for sample in samples
                    for hash in sample["output_sha256_by_request"]]
            for label, samples in per.items()
        }
        # Same prompt + deterministic greedy, so every success should
        # produce the same exact bytes, irrespective of provider and order.
        exact_text = successful and all(outputs.values()) and len(set(
            outputs["zerogravity"] + outputs["vllm"]
        )) == 1
        comparable = successful and exact_text and matched_source and matched_tokenizer
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
            "source_sha256_declared_identical": bool(matched_source),
            "tokenizer_sha256_declared_identical": bool(matched_tokenizer),
            "comparable_with_caller_supplied_source_attestation": bool(comparable),
            "zerogravity_tokens_per_second_median": rate_zg,
            "vllm_tokens_per_second_median": rate_vllm,
            "zerogravity_over_vllm_median_goodput_ratio":
                round(rate_zg / rate_vllm, 5)
                if comparable and rate_zg is not None and rate_vllm else None,
            "vllm_equivalent": "NOT_ESTABLISHED",
        })
    return {
        "schema": "zerogravity.serving.vllm-comparison.v1",
        "benchmark_input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model_routes": {"zerogravity": zg_model, "vllm": vllm_model},
        "attestation": {
            "source_hashes": list(source_hashes),
            "tokenizer_hashes": list(tokenizer_hashes),
            "trust": "caller_provided_not_independently_cryptographically_verified",
        },
        "run_order": "balanced_ABBA_BAAB",
        "warmups_per_endpoint": 1,
        "runs": rows,
        "summaries": summaries,
        "limitations": [
            "Model name equality does not establish identical learned weights.",
            "Caller source/tokenizer hashes must be independently verified.",
            "SSE chunks do not establish token-level ITL.",
            "Equivalent quantization, physical GPU, driver and policies need external qualification.",
            "No vLLM-equivalent claim follows from faster median rates alone.",
        ],
    }


def self_test() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

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
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["model"] == "fixture"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for doc in [
                {"choices": [{"index": 0, "delta": {"content": "yes"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
                {"choices": [], "usage": {"completion_tokens": 1}},
            ]:
                self.wfile.write(b"data: " + json.dumps(doc).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *_args):
            pass

    servers = [ThreadingHTTPServer(("127.0.0.1", 0), Fake) for _ in range(2)]
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
    for thread in threads:
        thread.start()
    try:
        a,b = [f"http://127.0.0.1:{s.server_port}" for s in servers]
        result = comparison(
            a,b,"fixture","fixture","Hi.",2,[1,2],1,None,None,
            ("a"*64,"a"*64),("b"*64,"b"*64),1,10
        )
        assert len(result["runs"]) == 8
        assert all(row["completed_all_requests"] for row in result["summaries"])
        assert all(row["exact_output_text_sha256_matched"] for row in result["summaries"])
        assert all(row["comparable_with_caller_supplied_source_attestation"]
                   for row in result["summaries"])
        assert all(row["vllm_equivalent"]=="NOT_ESTABLISHED" for row in result["summaries"])
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
        result=comparison(
            args.zg_endpoint,args.vllm_endpoint,args.zg_model,args.vllm_model,
            args.prompt,args.requests,concurrencies,args.max_tokens,
            os.environ.get(args.zg_key_env),os.environ.get(args.vllm_key_env),
            (args.zg_source_sha256,args.vllm_source_sha256),
            (args.zg_tokenizer_sha256,args.vllm_tokenizer_sha256),
            args.repetitions,args.timeout
        )
    except (OSError,ValueError,KeyError) as err:
        parser.exit(2,f"Cannot compare endpoints: {err}\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
