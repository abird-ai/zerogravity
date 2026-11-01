#!/usr/bin/env python3
"""Compare independent per-layer full-f32 Qwen numerical traces.

The candidate and independent reference must be captured from exactly the
same source checkpoint, input token IDs, layer stage definitions, model mode
and CPU precision. This tool enforces alignment and compares every f32
element. It does not generate an independent reference and does not claim
source correctness from two copies of the same trace.

Full reference capture is a separate required acceptance gate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

STAGES = (
    "attention_input_norm",
    "attention_output",
    "post_attention_residual",
    "ffn_input_norm",
    "ffn_output",
    "post_ffn_residual",
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    result = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
            position, layer, stage = row["position"], row["layer"], row["stage"]
            values = row["values"]
            if not isinstance(position, int) or position < 0:
                raise ValueError("invalid position")
            if not isinstance(layer, int) or layer < 0:
                raise ValueError("invalid layer")
            if not isinstance(stage, str):
                raise ValueError("invalid stage")
            if not isinstance(values, list) or not values or len(values) > 131072:
                raise ValueError("unbounded/empty stage vector")
            if not all(isinstance(x, (float, int)) and
                       not isinstance(x, bool) and math.isfinite(x) for x in values):
                raise ValueError("nonfinite/non-numeric stage values")
            key = (position, layer, stage)
            if key in result:
                raise ValueError("duplicate stage key")
            result[key] = list(map(float, values))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}:{number}: {e}") from e
    if not result:
        raise ValueError(f"{path}: no stage vectors")
    return result


def compare(candidate, reference, atol, rtol, rms_tol):
    ours, expected = read(candidate), read(reference)
    if ours.keys() != expected.keys():
        a, b = set(ours), set(expected)
        raise ValueError(f"stage mismatch (missing={sorted(b-a)[:4]}, extra={sorted(a-b)[:4]})")
    summary = []
    failed = 0
    for key in sorted(expected):
        actual = ours[key]
        oracle = expected[key]
        if len(actual) != len(oracle):
            raise ValueError(f"shape mismatch at {key}: {len(actual)} vs {len(oracle)}")
        if key[2] not in STAGES and key[2] not in ("embedding", "final_norm"):
            raise ValueError(f"unknown stage key {key}")
        diffs = [a-r for a,r in zip(actual,oracle)]
        max_abs = max(map(abs,diffs))
        rmse = math.sqrt(math.fsum(d*d for d in diffs)/len(diffs))
        worst = max(range(len(diffs)),key=lambda j:abs(diffs[j]))
        over = sum(
            abs(a-r) > atol + rtol*abs(r) for a,r in zip(actual,oracle)
        )
        passed = over == 0 and rmse <= rms_tol
        failed += int(not passed)
        summary.append({
            "position":key[0], "layer":key[1], "stage":key[2],
            "elements":len(oracle), "max_abs_error":max_abs,
            "rmse":rmse, "elements_outside_tolerance":over,
            "worst_element_index":worst, "passed":passed
        })
    return {
        "passed":failed==0,
        "passed_stages":len(summary)-failed,
        "failed_stages":failed,
        "stage_count":len(summary),
        "all_element_count":sum(row["elements"] for row in summary),
        "max_abs_error":max(row["max_abs_error"] for row in summary),
        "max_rmse":max(row["rmse"] for row in summary),
        "worst_stages":sorted(summary,key=lambda x:x["max_abs_error"],reverse=True)[:15],
        "stages":summary,
    }


def self_test():
    with tempfile.TemporaryDirectory(prefix="zg-layer-compare-") as directory:
        folder = Path(directory)
        candidate, reference = folder/"candidate.jsonl", folder/"reference.jsonl"
        reference.write_text(json.dumps({
            "position":2, "layer":0, "stage":"attention_output",
            "values":[1.0,2.0,3.0,4.0]
        })+"\n")
        candidate.write_text(reference.read_text())
        assert compare(candidate,reference,1e-5,1e-5,1e-5)["passed"]
        candidate.write_text(json.dumps({
            "position":2, "layer":0, "stage":"attention_output",
            "values":[1.0,2.0,30.0,4.0]
        })+"\n")
        assert not compare(candidate,reference,1e-5,1e-5,1e-5)["passed"]
        try:
            compare(reference,folder/"absent.jsonl",1e-5,1e-5,1e-5)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing independent trace accepted")
        print("QWEN_LAYER_COMPARATOR_SELFTEST_PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--candidate",type=Path)
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--reference-provenance",
                        help="independent reference runtime, source revision and capture command")
    parser.add_argument("--abs-tol",type=float,default=1e-3)
    parser.add_argument("--rel-tol",type=float,default=1e-3)
    parser.add_argument("--rms-tol",type=float,default=5e-4)
    parser.add_argument("--report",type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required=(args.candidate,args.reference,args.checkpoint,
              args.sha256,args.reference_provenance)
    if not all(required):
        parser.error("independent comparison requires --candidate, --reference, "
                     "--checkpoint, --sha256 and --reference-provenance")
    for name, tol in (
        ("abs-tol", args.abs_tol),("rel-tol", args.rel_tol),("rms-tol",args.rms_tol)
    ):
        if not math.isfinite(tol) or tol < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if len(args.sha256)!=64 or not all(x in "0123456789abcdef"
                                     for x in args.sha256.lower()):
        parser.error("invalid source SHA256 digest")
    if args.candidate.resolve()==args.reference.resolve():
        parser.error("candidate and independent reference must be distinct files")
    actual_sha=sha256(args.checkpoint)
    if actual_sha != args.sha256.lower():
        raise ValueError("checkpoint SHA-256 mismatch: refusing wrong source")
    result=compare(args.candidate,args.reference,args.abs_tol,args.rel_tol,args.rms_tol)
    result.update({
        "candidate":str(args.candidate.resolve()),
        "reference":str(args.reference.resolve()),
        "checkpoint_sha256":actual_sha,
        "independent_reference_provenance":args.reference_provenance,
        "thresholds":{
            "abs_tol":args.abs_tol,"rel_tol":args.rel_tol,"rms_tol":args.rms_tol
        },
    })
    print(
        "LAYER_COMPARISON_RESULT="+("PASS" if result["passed"] else "FAIL"),
        f"{result['passed_stages']}/{result['stage_count']} stages,",
        f"{result['all_element_count']} elements,",
        f"max_abs={result['max_abs_error']:.6g}",
        flush=True,
    )
    for row in result["worst_stages"][:5]:
        print("LAYER_DELTA",json.dumps(row,sort_keys=True),flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return 0 if result["passed"] else 4


if __name__=="__main__":
    try:
        sys.exit(main())
    except (OSError,ValueError) as error:
        print("LAYER_COMPARISON_ERROR="+str(error),file=sys.stderr)
        sys.exit(2)
