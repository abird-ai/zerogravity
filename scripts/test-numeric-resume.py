#!/usr/bin/env python3
"""Tool-only deterministic tests for CPU numerical corpus checkpoint/resume.

Self-contained: the atomic-write and resume-recovery logic under test lives
here, so the historical developer qualification runner is not required.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile

PROMPTS = (
    "Hi",
    "The capital of France is",
    "2 + 2 =",
    "Rust programming language",
    "Hello, world!",
    "Qwen is an example of",
    "Which number follows 127?",
    "Bonjour! 你好 👋",
)


def write_report_atomically(args, report):
    if args.report is None:
        return
    path = args.report.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".incomplete-tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def recover_report(args, expected):
    if not args.resume or args.report is None or not args.report.is_file():
        return expected
    prior = json.loads(args.report.read_text())
    identity_fields = (
        "checkpoint", "sha256", "reference", "reference_kv_type",
        "reference_flash_attention", "thresholds", "start_case",
        "requested_cases", "requested_tokens", "candidate_binary_sha256",
        "reference_binary_sha256", "reference_direct_binary",
        "reference_direct_binary_sha256",
    )
    for field in identity_fields:
        if prior.get(field) != expected.get(field):
            raise ValueError(f"resume refused: {field} changed")
    entries = prior.get("cases")
    if not isinstance(entries, list) or len(entries) > args.prompts:
        raise ValueError("resume refused: invalid partial case count")
    expected_prompts = PROMPTS[args.start_case:args.start_case + args.prompts]
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("prompt") != expected_prompts[i]:
            raise ValueError("resume refused: previous prompt sequence mismatch")
    print(f"CPU_NUMERIC_RESUME={len(entries)}/{args.prompts} confirmed cases", flush=True)
    return prior


def main():
    with tempfile.TemporaryDirectory(prefix="zg-resume-test-") as root:
        report_path = Path(root) / "numeric.json"
        args = SimpleNamespace(
            report=report_path, resume=True, prompts=8, start_case=0
        )
        expected={
            "checkpoint":"/model.gguf",
            "sha256":"a"*64,
            "reference":"/independent-reference-server",
            "reference_kv_type":"f32",
            "thresholds":{"top20_logprob_absolute":0.45},
            "start_case":0,
            "requested_cases":8,
            "requested_tokens":32,
            "candidate_binary_sha256":"b"*64,
            "reference_binary_sha256":"c"*64,
            "cases":[
                {"prompt":PROMPTS[0],"pass":True},
                {"prompt":PROMPTS[1],"pass":False},
            ],
            "complete":False,
            "passed":False,
        }
        write_report_atomically(args,expected)
        restored=recover_report(args,expected)
        assert len(restored["cases"])==2
        assert restored["complete"] is False
        assert restored["passed"] is False
        assert write_report_atomically(args,restored) is None
        assert not report_path.with_name("numeric.json.incomplete-tmp").exists()
        altered={**expected,"sha256":"d"*64}
        try:
            recover_report(args,altered)
        except ValueError as e:
            assert "sha256 changed" in str(e)
        else:
            raise AssertionError("changed checkpoint SHA accepted in resumption")
        changed={**expected,"cases":[{"prompt":PROMPTS[3],"pass":True}]}
        report_path.write_text(json.dumps(changed))
        try:
            recover_report(args,expected)
        except ValueError as e:
            assert "prompt sequence mismatch" in str(e)
        else:
            raise AssertionError("out-of-order prompt report accepted")
        assert report_path.is_file()
        print("CPU_NUMERIC_RESUME_SELFTEST_PASS")


if __name__=="__main__":
    main()
