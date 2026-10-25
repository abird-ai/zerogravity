#!/usr/bin/env python3
"""Fresh-process controller for the explicit Vulkan Gemma E2B diagnostic.

Every invocation is one process: the original native-context constructor is
one-shot and the source/provider envelopes are process-lifetime for unknown
cases. The controller never claims performance, serving or numerical eligibility.
It validates the reports the actual binaries write; it does not synthesize them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = "resume-all-five-vulkan-20261025-2"
STATE = Path("/home/pvl/.local/state/abird/zerogravity") / BASE

GIB = 1024**3
MIB = 1024**2
MATRIX_STATUS = "PASS_EXACT_DIAGNOSTIC_MATRIX"
UNKNOWN_STATUS = "ACTUAL_UNKNOWN_RETAINED"
FINAL_STATUS = "PASS_ACTUAL_FINAL_OWNER"
PARTIAL_STATUS = "PASS_ACTUAL_PARTIAL_NATIVE_RETAINED"
PEER_STATUS = "PASS_ACTUAL_LEARNED_SAME_PCI_PEER"
FALSE_PERMITS = (
    "trained_native_serving_or_performance_qualified",
    "trained_native_natural_serving_or_performance_qualified",
)
HOST_BASELINE = 10 * GIB
DIAGNOSTIC_HOST = 32 * MIB
PEER_PROVIDER = 64 * MIB
PEER_DEVICE = 2 * GIB
VULKAN_PROVIDER = 20 * GIB
REQUIRED = {
    "zg.gemma4-e2b-original-vulkan-complete-qualification.v1": {
        "schema": str, "status": str, "exact": bool, "events": list,
        "inventory": str, "managed_source_closed": bool, "physical_reserved": int,
        "host_reserved": int, "diagnostic_host_reserved": int,
        "physical_fenced": bool, "host_fenced": bool,
        "expected_persistent_context_device": int,
        "expected_persistent_context_host": int,
        "persistent_compiler_host": int,
        "expected_persistent_host_baseline": int,
        "context_retirement": str, "storage": str, "source_page_policy": str,
        "trained_native_serving_or_performance_qualified": bool,
    },
    "zg.gemma4-original-vulkan-unknown-native-owner.v1": {
        "schema": str, "status": str, "source_close_refused": bool,
        "physical_fenced": bool, "physical_reserved": int,
        "host_reserved": int,
        "trained_native_serving_or_performance_qualified": bool,
    },
    "zg.gemma4-original-vulkan-final-feedback-owner.v1": {
        "schema": str, "status": str, "actual_completed_full35_calls": int,
        "actual_completed_layers": int,
        "actual_completed_proposed_state_planes": int,
        "actual_completed_fullV": int, "commits": int, "actual_selected": list,
        "source_frontier": int, "source_close_refused": bool,
        "physical_reserved": int, "physical_fenced": bool,
        "host_reserved": int, "host_fenced": bool,
        "diagnostic_host_reserved": int,
        "expected_persistent_context_host": int,
        "persistent_compiler_host": int,
        "expected_persistent_host_baseline": int,
        "original_final_owner": dict,
        "trained_native_natural_serving_or_performance_qualified": bool,
    },
    "zg.gemma4-original-vulkan-learned-same-pci-peer.v1": {
        "schema": str, "status": str, "direction": str,
        "same_process": bool, "same_pci": bool, "same_arbiter": bool,
        "same_queue": bool, "distinct_services": bool,
        "learned_vulkan_owner": bool, "finite_hip_peer": bool,
        "source_host_limit": int, "host_limit": int, "host_reserved": int,
        "host_pre_admitted": bool, "physical_limit": int,
        "physical_reserved": int, "vulkan_provider_device": int,
        "vulkan_context_host": int, "vulkan_context_device": int,
        "vulkan_native_host": int, "vulkan_native_device": int,
        "hip_source_host": int, "hip_provider_device": int,
        "hip_context_host": int, "hip_compiler_host": int,
        "hip_context_device": int, "hip_compiler_device": int,
        "submitted_fault": bool, "submitter_retained_handles": int,
        "peer_refused": bool, "peer_new_service_refused": bool,
        "peer_root_unchanged": bool, "peer_pending_unchanged": bool,
        "peer_resident_unchanged": bool, "peer_uploads_unchanged": bool,
        "peer_queue_unchanged": bool, "vulkan_native_retained": bool,
        "vulkan_set_key": int, "vulkan_actor": str,
        "vulkan_native_modules": int, "vulkan_native_code_bytes": int,
        "vulkan_native_host_bytes": int, "vulkan_native_device_bytes": int,
        "vulkan_managed_handles": int, "hip_source_retained": bool,
        "hip_service_retained": bool, "hip_managed_handles": int,
        "hip_managed_handles_added": int, "physical_fenced": bool,
        "original_source_retained": bool, "original_native_retained": bool,
        "original_handle_retained": bool,
        "source_close_refused": bool,
        "trained_native_serving_or_performance_qualified": bool,
        "trained_native_natural_serving_or_performance_qualified": bool,
    },
    "zg.gemma4-original-vulkan-partial-native-owner.v1": {
        "schema": str, "status": str, "partial_native_error": str,
        "partial_modules_constructed_before_fault": int,
        "source_close_refused": bool, "physical_fenced": bool,
        "physical_reserved": int, "host_reserved": int, "host_fenced": bool,
        "trained_native_natural_serving_or_performance_qualified": bool,
    },
}
RECEIPT_FIELDS = {
    "wrapper": str, "fault": bool, "source": str, "target_lease": str,
    "service_generation": int, "target_lease_bytes": int,
    "aliases_inside_only_wrapper": int, "candidate_owners_before_drop": int,
    "model_owners_before_drop": int, "candidate_workspace_target_aliases": int,
    "candidate_retired": bool, "target_final_reader_hits": int,
    "native_retirement_attempts": int,
    "reader_hits_before_native_retirement": int,
    "actual_native_modules_before": int,
    "actual_native_spirv_representation_bytes_before": int,
    "actual_native_host_bytes_before": int,
    "actual_native_device_bytes_before": int,
    "native_admission_retained_after": bool,
    "actual_native_modules_after": int,
    "original_service_retained_after": bool,
    "actual_quarantined_handles_after": int,
    "source_frontier": int,
}


def read_report(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), f"report object: {path}"
    return value


def require_fields(report: dict, fields: dict[str, type]) -> None:
    for key, kind in fields.items():
        assert key in report, f"missing typed field: {key}"
        assert type(report[key]) is kind, f"{key} must be {kind.__name__}"


def reject_true_permits(value: object, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            joined = f"{path}.{key}" if path else key
            if ("qualified" in key or "permit" in key) and item is not False:
                raise AssertionError(f"permit must be false: {joined}")
            reject_true_permits(item, joined)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_true_permits(item, f"{path}[{index}]")


def false_permits(report: dict, required: tuple[str, ...] = FALSE_PERMITS) -> None:
    for key in required:
        assert key in report, f"missing false-permit field: {key}"
        assert report[key] is False, f"{key} must stay false"
    reject_true_permits(report)


def check_12b_feedback(report: dict) -> None:
    selected = report.get("actual_selected")
    frontier = report.get("source_frontier")
    assert isinstance(selected, list) and len(selected) == 2, "actual 8B selection"
    assert all(
        type(v) is int and 0 <= v <= 0xFFFF_FFFF for v in selected
    ), "actual selected IDs"
    assert type(frontier) is int and 0 <= frontier <= 0xFFFF_FFFF
    assert report.get("commits") == 1, "exactly one committed feedback packet"


def check_schema(report: dict, schema: str) -> None:
    assert report.get("schema") == schema, report.get("schema")
    require_fields(report, REQUIRED[schema])


def check_matrix_events(report: dict) -> None:
    events = report["events"]
    assert events, "matrix must contain actual events"
    kinds = [e.get("kind") for e in events if isinstance(e, dict)]
    assert all(isinstance(e, dict) for e in events), "typed event objects"
    assert kinds.count("actual-completed-proposal") >= 2
    assert kinds.count("actual-committed-call") >= 2
    assert kinds.count("source-ID-decoded-output") >= 1
    assert kinds.count("resident-normal-parity") >= 1
    hits = [e for e in events if e.get("kind") == "resident-ple-row-reuse"]
    assert len(hits) == 2, "warm and observed PLE reuse events"
    assert hits[0].get("observed") is False
    assert hits[0].get("full_kv_exact") is False
    assert hits[1].get("observed") is True
    assert hits[1].get("full_kv_exact") is True
    assert hits[1].get("learned_full35") is True
    assert hits[1].get("uploads") == 0
    assert hits[1].get("reuses", 0) > 0
    assert hits[1].get("ple_row_h2d") == 0
    for event in events:
        assert set(("kind",)) <= set(event), "event kind"


def check_healthy(report: dict) -> None:
    check_schema(
        report, "zg.gemma4-e2b-original-vulkan-complete-qualification.v1"
    )
    assert report["status"] == MATRIX_STATUS, report.get("status")
    assert report["exact"] is True
    assert report["managed_source_closed"] is True
    assert report["physical_reserved"] == 4 * GIB
    assert report["host_reserved"] == HOST_BASELINE + DIAGNOSTIC_HOST
    assert report["diagnostic_host_reserved"] == DIAGNOSTIC_HOST
    assert report["physical_fenced"] is False
    assert report["host_fenced"] is False
    assert report["expected_persistent_host_baseline"] == HOST_BASELINE
    assert report["persistent_compiler_host"] == 8 * GIB
    assert report["expected_persistent_context_host"] == 2 * GIB
    check_matrix_events(report)
    false_permits(report)


def check_unknown(report: dict) -> None:
    check_schema(report, "zg.gemma4-original-vulkan-unknown-native-owner.v1")
    assert report["status"] == UNKNOWN_STATUS, report.get("status")
    assert report["source_close_refused"] is True
    assert report["physical_fenced"] is True
    assert report["physical_reserved"] == 20 * GIB
    assert report["host_reserved"] > 2 * GIB
    false_permits(report)


def need_retained_owner(receipt: object) -> None:
    assert isinstance(receipt, dict), "actual final owner receipt"
    require_fields(receipt, RECEIPT_FIELDS)
    assert receipt["fault"] is True
    assert receipt["actual_native_modules_before"] > 0
    assert receipt["actual_native_spirv_representation_bytes_before"] > 0
    assert receipt["native_admission_retained_after"] is True
    assert receipt["original_service_retained_after"] is True
    assert receipt["actual_native_modules_after"] >= receipt[
        "actual_native_modules_before"
    ]


def check_final(report: dict, fault: bool) -> None:
    check_schema(report, "zg.gemma4-original-vulkan-final-feedback-owner.v1")
    assert report["status"] == FINAL_STATUS, report.get("status")
    check_12b_feedback(report)
    assert report["actual_completed_full35_calls"] == 1
    assert report["actual_completed_layers"] == 35
    assert report["actual_completed_proposed_state_planes"] == 30
    assert report["actual_completed_fullV"] == 1
    assert report["source_close_refused"] is fault
    assert report["physical_fenced"] is fault
    # The retained host baseline is not a GPU-service fence. This probe only
    # fences the physical device; host source/compiler leases stay resident.
    assert report["host_fenced"] is False
    assert report["physical_reserved"] == (20 * GIB if fault else 4 * GIB)
    if fault:
        assert report["host_reserved"] > HOST_BASELINE
        need_retained_owner(report["original_final_owner"])
    else:
        assert report["host_reserved"] == HOST_BASELINE + DIAGNOSTIC_HOST
        receipt = report["original_final_owner"]
        require_fields(receipt, RECEIPT_FIELDS)
        assert receipt["fault"] is False
        assert receipt["candidate_retired"] is True
        assert receipt["target_final_reader_hits"] >= 1
        assert receipt["reader_hits_before_native_retirement"] >= 1
        assert receipt["native_admission_retained_after"] is False
        assert receipt["actual_native_modules_after"] == 0
        assert receipt["original_service_retained_after"] is False
        assert receipt["aliases_inside_only_wrapper"] == 1
        assert receipt["candidate_workspace_target_aliases"] == 0
        assert receipt["actual_native_modules_before"] > 0
        assert receipt["actual_native_spirv_representation_bytes_before"] > 0
    false_permits(report)


def check_partial(report: dict) -> None:
    check_schema(report, "zg.gemma4-original-vulkan-partial-native-owner.v1")
    assert report["status"] == PARTIAL_STATUS, report.get("status")
    assert "actual partial native preparation fault" in report[
        "partial_native_error"
    ]
    assert report["partial_modules_constructed_before_fault"] > 0
    assert report["source_close_refused"] is True
    assert report["physical_fenced"] is True
    assert report["physical_reserved"] == 20 * GIB
    # Actual partial native failure retains the host baseline but does not
    # claim a host-service fence that the source owner never performed.
    assert report["host_fenced"] is False
    assert report["host_reserved"] > HOST_BASELINE
    false_permits(report)


def supervised_launch(argv: list[str], env: dict[str, str] | None = None) -> None:
    """Only the accepted launch.py supervisor may own a physical child."""
    result = subprocess.run(argv, env=env, check=False)
    if result.returncode:
        raise SystemExit(f"failed ({result.returncode}): {' '.join(argv)}")


def launch_case(base: Path, mode: str, suffix: str = "") -> dict:
    """One fresh process through the original group/EOF supervisor and 70 FDs."""
    output = STATE / f"case-{mode}{suffix}"
    receipt = STATE / f"launch-{mode}{suffix}.json"
    supervised_launch([
        sys.executable, "-B",
        str(base / "source/tools/vulkan-gemma-qualification/runner/launch.py"),
        "--base", str(base), "--output", str(output),
        "--receipt", str(receipt), "--mode", mode,
    ])
    value = read_report(receipt)
    assert value.get("status") == "CLOSED_PASS_SCOPED", receipt
    assert value.get("log", {}).get("eof") is True
    assert value.get("log", {}).get("closed") is True
    return read_report(output / (mode + ".json"))


def matrix_case(base: Path, mode: str, unknown: bool) -> dict:
    value = launch_case(base, mode)
    check_unknown(value) if unknown else check_healthy(value)
    return value


def final_case(base: Path, mode: str, fault: bool) -> dict:
    value = launch_case(base, mode)
    check_partial(value) if mode == "partial-native" else check_final(value, fault)
    return value


def check_peer(report: dict) -> None:
    check_schema(report, "zg.gemma4-original-vulkan-learned-same-pci-peer.v1")
    assert report["status"] == PEER_STATUS, report.get("status")
    direction = report["direction"]
    assert direction in ("vulkan-then-hip", "hip-then-vulkan")
    assert report["same_process"] is True
    assert report["same_pci"] is True
    assert report["same_arbiter"] is True
    assert report["same_queue"] is True
    assert report["distinct_services"] is True
    assert report["learned_vulkan_owner"] is True
    assert report["finite_hip_peer"] is True
    assert report["source_host_limit"] == 16 * GIB
    assert report["host_limit"] == 16 * GIB
    assert 0 < report["host_reserved"] <= report["host_limit"]
    assert report["host_pre_admitted"] is True
    assert report["physical_limit"] == 64 * GIB
    assert report["vulkan_provider_device"] == (
        VULKAN_PROVIDER + report["vulkan_context_device"]
    ), "Vulkan provider envelope plus its admitted context"
    assert report["physical_reserved"] == (
        report["vulkan_provider_device"] + PEER_PROVIDER + PEER_DEVICE
    ), "both owned physical envelopes"
    assert report["vulkan_context_host"] == 2 * GIB
    assert report["vulkan_context_device"] == 4 * GIB
    assert report["vulkan_native_host"] > 0
    assert report["vulkan_native_device"] > 0
    assert report["hip_source_host"] > 0
    assert report["hip_provider_device"] == PEER_PROVIDER
    assert report["hip_context_host"] == 2 * GIB
    assert report["hip_compiler_host"] == GIB
    assert report["hip_context_device"] == GIB
    assert report["hip_compiler_device"] == GIB
    assert report["submitted_fault"] is True
    assert report["submitter_retained_handles"] > 0
    assert report["peer_refused"] is True
    assert report["peer_new_service_refused"] is True
    assert report["peer_root_unchanged"] is True
    assert report["peer_pending_unchanged"] is True
    assert report["peer_resident_unchanged"] is True
    assert report["peer_uploads_unchanged"] is True
    assert report["peer_queue_unchanged"] is True
    assert report["vulkan_native_retained"] is True
    assert report["vulkan_set_key"] > 0
    assert report["vulkan_native_modules"] > 0
    assert report["vulkan_native_code_bytes"] > 0
    assert report["vulkan_native_host_bytes"] > 0
    assert report["vulkan_native_device_bytes"] > 0
    if direction == "vulkan-then-hip":
        assert report["vulkan_managed_handles"] > 0
        assert report["hip_managed_handles_added"] == 0
    else:
        assert report["vulkan_managed_handles"] == 0
        assert report["hip_managed_handles_added"] > 0
    assert report["hip_managed_handles"] >= 0
    assert report["hip_source_retained"] is True
    assert report["hip_service_retained"] is True
    assert report["physical_fenced"] is True
    assert report["original_source_retained"] is True
    assert report["original_native_retained"] is True
    assert report["original_handle_retained"] is True
    assert report["source_close_refused"] is True
    false_permits(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument(
        "--peer-report",
        type=Path,
        action="append",
        default=[],
        help=(
            "validate an externally produced single-process same-PCI peer "
            "receipt; no two-process substitute is accepted"
        ),
    )
    parser.add_argument(
        "--check-report",
        type=Path,
        action="append",
        default=[],
        help="validate an existing report only; no subprocess is launched",
    )
    args = parser.parse_args()
    if args.check_report:
        for path in args.check_report:
            value = read_report(path)
            schema = value.get("schema", "")
            if schema.endswith("partial-native-owner.v1"):
                check_partial(value)
            elif schema.endswith("final-feedback-owner.v1"):
                check_final(value, fault=bool(value.get("source_close_refused")))
            elif schema.endswith("unknown-native-owner.v1"):
                check_unknown(value)
            elif schema.endswith("learned-same-pci-peer.v1"):
                check_peer(value)
            else:
                check_healthy(value)
        print("VULKAN_GEMMA_REPORTS_CHECKED", len(args.check_report))
        return 0
    assert args.base is not None
    assert args.base.is_absolute() and args.base.parent == (
        Path("/home/pvl/.cache/abird/zerogravity/qualification")
    )
    if args.build:
        supervised_launch([
            sys.executable, "-B",
            str(args.base / "source/tools/vulkan-gemma-qualification/runner/launch.py"),
            "--base", str(args.base), "--output", str(STATE / "case-build"),
            "--receipt", str(STATE / "launch-build.json"), "--mode", "build",
        ])
    matrix_case(args.base, "resident", False)
    matrix_case(args.base, "resident-unknown", True)
    matrix_case(args.base, "resident-row-unknown", True)
    for mode, fault in (
        ("session-known", False),
        ("snapshot-known", False),
        ("session-release-fault", True),
        ("snapshot-release-fault", True),
        ("partial-native", True),
    ):
        final_case(args.base, mode, fault)
    # Both provider directions are separate fresh processes. An externally
    # supplied peer receipt remains inspection-only and cannot replace either.
    for mode in ("peer-vulkan-then-hip", "peer-hip-then-vulkan"):
        check_peer(launch_case(args.base, mode))
    if args.peer_report:
        for path in args.peer_report:
            check_peer(read_report(path))
    print("VULKAN_GEMMA_CONTROLLER_OK reports=", STATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
