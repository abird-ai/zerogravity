#!/usr/bin/env python3
"""Fail-closed Qwen3.5 CPU evidence gates, with separate qualification scopes.

--scope phase1 (the GPU *development admission* gate) requires exact-model
native CPU state integrity, independent short numerical/per-layer parity, and
real trained-model contexts through 8192 tokens for BOTH Qwen3.5 sizes.

--scope extended (the earlier full-context gate) additionally requires
16384 and 32768 genuine whole-model context evidence for BOTH. These are
**deferred**, not counted as passed or silently dropped, for Phase 1.

Passing phase1 authorizes engineering on whole-model CUDA/ROCm/Vulkan; it
does NOT imply a working or qualified GPU model backend, reliable 16K/32K
contexts, exhaustive sampling parity, or production release readiness.
Reference and model source hashes and existing numerical tolerances unchanged.
Missing evidence always fails the selected gate.
"""
import argparse
import json
from pathlib import Path
import sys

MODELS = {
    "qwen35-0.8b": "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf",
    "qwen35-2b": "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6",
}
PHASE1_CONTEXT_COUNTS = (128,512,1024,4096,8192)
DEFERRED_CONTEXT_COUNTS = (16384,32768)
CONTEXT_COUNTS = PHASE1_CONTEXT_COUNTS + DEFERRED_CONTEXT_COUNTS
REQUIRED_PROMPTS = 8
REQUIRED_TOKENS = 32


def read_json(path, observations):
    if not path.is_file():
        observations.append(f"missing evidence: {path}")
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        observations.append(f"invalid JSON evidence {path}: {e}")
        return None


def numeric_ok(record, model, expected_sha, problems):
    if not isinstance(record, dict):
        return False
    if record.get("sha256","").lower() != expected_sha:
        problems.append(f"{model}: numerical report checkpoint digest mismatch")
        return False
    cases=record.get("cases",[])
    if not isinstance(cases,list) or len(cases)!=REQUIRED_PROMPTS:
        problems.append(f"{model}: incomplete numerical corpus; need 8 prompts")
        return False
    if record.get("start_case") not in (0,None):
        problems.append(f"{model}: report does not start at corpus first prompt")
        return False
    if not record.get("passed"):
        problems.append(f"{model}: numerical strict gate recorded FAIL")
        return False
    for i,case in enumerate(cases):
        if not isinstance(case,dict) or case.get("tokens")!=REQUIRED_TOKENS or not case.get("pass"):
            problems.append(f"{model}: numerical case {i} is incomplete or failed")
            return False
        if case.get("hard_top1_mismatches",1) or case.get("missing_reference_top5_steps",1):
            problems.append(f"{model}: numerical case {i} has hard choice/top5 mismatches")
            return False
        if case.get("unverified_fragment_positions",1):
            problems.append(f"{model}: numerical case {i} contains unverified UTF8 rows")
            return False
        if case.get("minimum_top20_overlap",0)<14:
            problems.append(f"{model}: numerical case {i} shared rank overlap is too small")
            return False
        if case.get("max_shared_logprob_absolute_error",999)>0.45:
            problems.append(f"{model}: numerical case {i} failed the predeclared top20 gate")
            return False
        # Preserve actual high-probability distribution accuracy independently
        # of rare, near-zero tail logits. Greedy rank agreement alone must not
        # qualify a model with a materially different sampled distribution.
        if case.get("max_top1_probability_absolute_error",999)>0.05:
            problems.append(
                f"{model}: numerical case {i} top-token probability drift "
                f"{case.get('max_top1_probability_absolute_error')} exceeds 0.05"
            )
            return False
        if case.get("max_truncated_union_top20_mass_l1",999)>0.06:
            problems.append(
                f"{model}: numerical case {i} truncated top20 probability mass drift "
                f"{case.get('max_truncated_union_top20_mass_l1')} exceeds 0.06"
            )
            return False
    return True


def context_ok(record,model,expected_sha,n,problems):
    if not isinstance(record,dict):
        return False
    if record.get("source_sha256","").lower()!=expected_sha:
        problems.append(f"{model}:{n}: wrong source SHA256 in context report")
        return False
    if record.get("actual_reference_prompt_tokens")!=n:
        problems.append(f"{model}:{n}: full-model context not actually {n} tokens")
        return False
    if record.get("qualified_length_type")!="real full-model CPU token-sequential prefill":
        problems.append(f"{model}:{n}: synthetic operator positions are not real full-model context")
        return False
    if record.get("pass") is not True or record.get("next_token_exact") is not True:
        problems.append(f"{model}:{n}: independent next token mismatch / gate failed")
        return False
    if record.get("top20_shared_ids",0)<14 or record.get("max_shared_logprob_error",999)>0.45:
        problems.append(f"{model}:{n}: numerical top20 error/coverage rejected")
        return False
    if not (isinstance(record.get("rust_published_roots"),int)
            and 1<=record["rust_published_roots"]<=n):
        problems.append(f"{model}:{n}: missing/bad committed CPU StateRoot evidence")
        return False
    return True


def layer_ok(record,model,expected_sha,problems):
    if not isinstance(record,dict):
        return False
    if record.get("checkpoint_sha256","").lower()!=expected_sha:
        problems.append(f"{model}: independent layer evidence checkpoint mismatch")
        return False
    reference=record.get("independent_reference_provenance","")
    if not isinstance(reference,str) or len(reference.strip())<16:
        problems.append(f"{model}: no independently documented per-layer oracle provenance")
        return False
    if not record.get("passed") or record.get("failed_stages",1)!=0:
        problems.append(f"{model}: independent per-layer comparison failed")
        return False
    if record.get("stage_count",0)<146:
        problems.append(f"{model}: too few observed trained model layer stages")
        return False
    if record.get("all_element_count",0)<146*1024:
        problems.append(f"{model}: incomplete layer vector coverage")
        return False
    return True


def check(folder,scope="extended"):
    if scope not in ("phase1","extended"):
        raise ValueError("unrecognized CPU qualification scope")
    required_contexts=PHASE1_CONTEXT_COUNTS if scope=="phase1" else CONTEXT_COUNTS
    deferred_contexts=DEFERRED_CONTEXT_COUNTS if scope=="phase1" else ()
    problems=[]
    passed=[]
    for model,sha in MODELS.items():
        prefix=folder/model
        num=read_json(prefix/"numeric-32.json",problems)
        if numeric_ok(num,model,sha,problems):
            passed.append(f"{model}:numeric-32")
        for length in required_contexts:
            rec=read_json(prefix/f"context-{length}.json",problems)
            if context_ok(rec,model,sha,length,problems):
                passed.append(f"{model}:context-{length}")
        layer=read_json(prefix/"independent-layers.json",problems)
        if layer_ok(layer,model,sha,problems):
            passed.append(f"{model}:independent-layers")
        state=read_json(prefix/"state-invariants.json",problems)
        if state is not None:
            if state.get("checkpoint_sha256","").lower()==sha \
                    and state.get("passed") is True \
                    and state.get("cross_session_rejection") is True \
                    and state.get("atomic_rollback") is True \
                    and state.get("replay_equivalence") is True \
                    and state.get("chunk_equivalence") is True:
                passed.append(f"{model}:state-invariants")
            else:
                problems.append(f"{model}: state invariants report incomplete/failing")
    return {
        "status": (
            ("PHASE1_CPU_READY_FOR_GPU_DEVELOPMENT" if scope=="phase1"
             else "EXTENDED_CONTEXT_QUALIFIED") if not problems
            else ("BLOCKED_PHASE1_CPU_FIRST" if scope=="phase1"
                  else "BLOCKED_EXTENDED_CONTEXT")
        ),
        "ready_for_gpu":len(problems)==0,
        "scope":scope,
        "gpu_development_only":scope=="phase1",
        "gpu_inference_implemented":False,
        "extended_context_qualified":scope=="extended" and len(problems)==0,
        "source_models": MODELS,
        "real_context_lengths_required": required_contexts,
        "deferred_context_lengths":deferred_contexts,
        "deferred_gates":[f"{model}:context-{length}" for model in MODELS
                          for length in deferred_contexts],
        "required_numeric":{
            "prompts":REQUIRED_PROMPTS,"generated_tokens":REQUIRED_TOKENS,
            "max_shared_top20_logprob_absolute_error":0.45,
            "max_top1_probability_absolute_error":0.05,
            "max_truncated_top20_union_l1":0.06,
            "hard_top1_mismatches":0,
        },
        "completed_gates":passed,
        "unmet_gates":problems,
    }


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory(prefix="zg-cpu-gate-") as root:
        report=check(Path(root))
        assert not report["ready_for_gpu"],"empty evidence wrongly authorized GPU phase"
        assert len(report["unmet_gates"])>=20,"required gates missing"
        phase1=check(Path(root),scope="phase1")
        assert not phase1["ready_for_gpu"],"empty evidence wrongly cleared Phase 1"
        assert len(phase1["unmet_gates"])==16,"Phase 1 must need exactly sixteen gates"
        assert len(phase1["deferred_gates"])==4,"16K/32K must be explicitly deferred"
        assert not phase1["extended_context_qualified"]
        path=Path(root)/"qwen35-2b"
        path.mkdir()
        (path/"context-32768.json").write_text(json.dumps({
            "qualified_length_type":"synthetic 32768 operator steps", "pass":True,
            "actual_reference_prompt_tokens":32768,
            "source_sha256":MODELS["qwen35-2b"]
        }))
        report=check(Path(root))
        assert not report["ready_for_gpu"],"synthetic 32K wrongly accepted"

        # Unit-only synthesized evidence verifies that Phase 1 can advance
        # with the four extended gates explicitly deferred, never passed.
        # This is NOT external model evidence or a qualification artifact.
        for model,sha in MODELS.items():
            folder=Path(root)/model
            folder.mkdir(exist_ok=True)
            numeric={"sha256":sha,"passed":True,"start_case":0,"cases":[{
                "tokens":REQUIRED_TOKENS,"pass":True,
                "hard_top1_mismatches":0,"missing_reference_top5_steps":0,
                "unverified_fragment_positions":0,"minimum_top20_overlap":20,
                "max_shared_logprob_absolute_error":0.0,
                "max_top1_probability_absolute_error":0.0,
                "max_truncated_union_top20_mass_l1":0.0,
            } for _ in range(REQUIRED_PROMPTS)]}
            (folder/"numeric-32.json").write_text(json.dumps(numeric))
            for length in PHASE1_CONTEXT_COUNTS:
                sample={
                    "source_sha256":sha,"actual_reference_prompt_tokens":length,
                    "qualified_length_type":"real full-model CPU token-sequential prefill",
                    "pass":True,"next_token_exact":True,"top20_shared_ids":20,
                    "max_shared_logprob_error":0.0,"rust_published_roots":1,
                }
                (folder/f"context-{length}.json").write_text(json.dumps(sample))
            layers={
                "checkpoint_sha256":sha,
                "independent_reference_provenance":"unit-test-only mock independent oracle",
                "passed":True,"failed_stages":0,
                "stage_count":146,"all_element_count":146*1024,
            }
            (folder/"independent-layers.json").write_text(json.dumps(layers))
            state={"checkpoint_sha256":sha,"passed":True,
                   "cross_session_rejection":True,"atomic_rollback":True,
                   "replay_equivalence":True,"chunk_equivalence":True}
            (folder/"state-invariants.json").write_text(json.dumps(state))
        phase1=check(Path(root),scope="phase1")
        extended=check(Path(root),scope="extended")
        assert phase1["ready_for_gpu"] and len(phase1["completed_gates"])==16
        assert not phase1["extended_context_qualified"]
        assert len(phase1["deferred_gates"])==4
        assert not extended["ready_for_gpu"] and len(extended["unmet_gates"])==4
        # A synthetic 8K marker must never be accepted as a real context.
        path=Path(root)/"qwen35-0.8b"/"context-8192.json"
        bad=json.loads(path.read_text())
        bad["qualified_length_type"]="synthetic operator/position check"
        path.write_text(json.dumps(bad))
        assert not check(Path(root),scope="phase1")["ready_for_gpu"]
        print("CPU_FIRST_FAIL_CLOSED_SELFTEST_PASS")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir",type=Path,default=Path("target/qualification/cpu-gate"))
    parser.add_argument("--report",type=Path)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--scope",choices=("phase1","extended"),default="phase1",
        help="phase1=8K GPU development admission; extended=8K+16K+32K full-context evidence")
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    result=check(args.evidence_dir,scope=args.scope)
    print(result["status"])
    print(f"GATES_PASSED={len(result['completed_gates'])} GATES_UNMET={len(result['unmet_gates'])}")
    if result["deferred_gates"]:
        print(f"DEFERRED_NOT_QUALIFIED={len(result['deferred_gates'])}")
        for item in result["deferred_gates"]:
            print("DEFERRED",item)
    for issue in result["unmet_gates"]:
        print("BLOCKER",issue)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return 0 if result["ready_for_gpu"] else 4


if __name__=="__main__":
    sys.exit(main())
