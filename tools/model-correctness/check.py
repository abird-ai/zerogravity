"""Compare independently collected model outputs; never execute a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from accuracy import compare_generation, full_logits_metrics

SCHEMA = "zg.model-output.v1"
BACKENDS = {"cpu", "cuda", "rocm", "vulkan"}
MODEL_KEYS = {"source_sha256", "tokenizer_sha256", "template_sha256", "precision", "quantization"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def read(path):
    raw = Path(path).read_bytes()
    def nonfinite(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def finite_float(value):
        parsed = float(value)
        require(math.isfinite(parsed), "nonfinite JSON number: " + value)
        return parsed
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite, parse_float=finite_float)
    return value, {"path": str(Path(path).absolute()), "bytes": len(raw),
                   "sha256": hashlib.sha256(raw).hexdigest()}


def validate(document):
    require(isinstance(document, dict) and document.get("schema") == SCHEMA,
            "model-output schema required")
    representation = document.get("text_representation")
    require(representation in ("utf8", "sha256"), "explicit output text representation")
    require(document.get("complete") is True, "interrupted or incomplete output")
    closure = document.get("owner_closure")
    require(isinstance(closure, dict) and closure.get("complete") is True
            and closure.get("errors") == [], "original owner closure required")
    model = document.get("model")
    require(isinstance(model, dict) and set(model) == MODEL_KEYS, "complete model identity")
    for key in ("source_sha256", "tokenizer_sha256", "template_sha256"):
        require(digest(model[key]), "model identity digest: " + key)
    for key in ("precision", "quantization"):
        require(isinstance(model[key], str) and model[key].strip(), "explicit " + key)
    execution = document.get("execution")
    require(isinstance(execution, dict) and execution.get("backend") in BACKENDS,
            "explicit actual backend")
    for key in ("runtime_revision", "numerical_profile"):
        require(isinstance(execution.get(key), str) and execution[key].strip(),
                "explicit execution identity: " + key)
    require(execution.get("runtime_artifact_kind") in ("executable", "container-config"),
            "explicit runtime artifact kind")
    require(digest(execution.get("runtime_artifact_sha256")), "runtime artifact digest")
    require(type(execution.get("batch_invariant")) is bool, "explicit invariance mode")
    capacity, vocabulary = execution.get("context_capacity"), execution.get("vocabulary_size")
    require(integer(capacity, 1) and integer(vocabulary, 1), "context and vocabulary bounds")
    cases = document.get("cases")
    require(isinstance(cases, list) and cases, "nonempty complete case population")
    seen = set()
    for case in cases:
        require(isinstance(case, dict) and isinstance(case.get("id"), str)
                and case["id"] and case["id"] not in seen, "unique nonempty case id")
        seen.add(case["id"])
        ids = case.get("input_ids")
        require(isinstance(ids, list) and ids and all(integer(i) and i < vocabulary for i in ids),
                "complete original token-id input")
        new, cached = case.get("new_input_tokens"), case.get("cached_tokens")
        require(integer(new, 1) and integer(cached) and new + cached == len(ids),
                "new input and cached prefix lengths join actual input")
        generation = case.get("generation")
        require(isinstance(generation, dict) and isinstance(generation.get("token_ids"), list)
                and generation["token_ids"], "complete nonempty generation")
        require(all(integer(i) and i < vocabulary for i in generation["token_ids"]),
                "generated token IDs within vocabulary")
        if representation == "sha256":
            require(digest(generation.get("text")), "actual output text SHA256")
        rows = generation.get("top_logprobs")
        if rows is not None:
            require(isinstance(rows, list), "logprob rows must be a list")
            for row in rows:
                require(isinstance(row, list) and all(isinstance(e, dict)
                        and integer(e.get("token_id")) and e["token_id"] < vocabulary for e in row),
                        "top logprob token IDs within actual vocabulary")
        require(len(ids) + len(generation["token_ids"]) <= capacity,
                "full requested input/output fits declared context; no truncation")
    return model, execution, cases


def compare(reference, candidate, *, policy="vllm-default"):
    rm, re, rc = validate(reference)
    cm, ce, cc = validate(candidate)
    require(rm == cm, "same checkpoint, tokenizer/template, precision and quantization")
    require(reference["text_representation"] == candidate["text_representation"],
            "same raw text or text-digest representation")
    require(re["vocabulary_size"] == ce["vocabulary_size"], "same vocabulary")
    require(re["context_capacity"] == ce["context_capacity"], "same declared context capacity")
    require(re["batch_invariant"] == ce["batch_invariant"], "same declared invariance mode")
    require([c["id"] for c in rc] == [c["id"] for c in cc],
            "same complete ordered case population")
    results = []
    for left, right in zip(rc, cc):
        for key in ("input_ids", "new_input_tokens", "cached_tokens"):
            require(left[key] == right[key], "same original input/history/schedule: " + key)
        result = {"id": left["id"], "new_input_tokens": left["new_input_tokens"],
                  "cached_tokens": left["cached_tokens"],
                  "generation": compare_generation(left["generation"], right["generation"],
                                                   policy=policy, top_k=5)}
        has_logits = "full_logits" in left or "full_logits" in right
        if has_logits:
            require(all(integer(c.get("target_id"))
                        and c["target_id"] < re["vocabulary_size"] for c in (left, right)),
                    "paired strict integer target IDs within vocabulary")
            require("full_logits" in left and "full_logits" in right
                    and left.get("target_id") == right.get("target_id"),
                    "paired full-vocabulary logits and same target")
            result["logits_metrics"] = full_logits_metrics(left["full_logits"], right["full_logits"],
                left.get("target_id"), vocabulary_size=re["vocabulary_size"])
        results.append(result)
    return {"schema": "zg.model-output-comparison.v1", "policy": policy,
            "comparison_passed": all(r["generation"]["accepted"] for r in results),
            "text_equality_kind": reference["text_representation"],
            "model": rm, "reference_execution": re, "candidate_execution": ce,
            "cases": results, "logits_metrics_are_measurements_only": True,
            "declared_identities_independently_authenticated": False,
            "operator_state_physical_qualified": False, "inference_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--policy", choices=("vllm-default", "exact"), default="vllm-default")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        reference, rp = read(args.reference)
        candidate, cp = read(args.candidate)
        result = compare(reference, candidate, policy=args.policy)
        result["inputs"] = {"reference": rp, "candidate": cp}
        payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (OSError, ValueError, TypeError, KeyError) as error:
        result = {"schema": "zg.model-output-comparison.v1", "comparison_passed": False,
                  "error": str(error), "inference_qualified": False}
        payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # An existing reference/result path must never be overwritten by a report.
    with args.report.open("x") as stream:
        stream.write(payload)
    return 0 if result["comparison_passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
