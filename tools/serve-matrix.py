#!/usr/bin/env python3
"""Serial matrix driver for existing paired serve-compare lifecycle runs.

Strict matrix input:
  {"schema":"zerogravity.serving.comparison-matrix.v1",
   "compare_argv":[...],
   "canonical_workload":"...json", "ragged_workload":"...json",
   "context_cells":[{"name":"context-...","workload":"...json","concurrency":N}]}

Workloads and optional --provider-lifecycle paths remain external inputs.
Canonical and ragged workloads each run C1/C4/C8/C16/C32/C64. Context cells
are appended in listed order. Every cell is a separate serialized invocation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
BENCH = runpy.run_path(str(HERE / "serve-bench.py"))
strict_json = BENCH["strict_json"]
load_workload = BENCH["load_workload"]
MATRIX_SCHEMA = "zerogravity.serving.comparison-matrix.v1"
RESULT_SCHEMA = "zerogravity.serving.comparison-matrix-result.v1"
CONCURRENCIES = (1, 4, 8, 16, 32, 64)
MAX_REPORT_BYTES = 256 * 1024 * 1024


class CellFailure(Exception):
    def __init__(self, message: str, returncode: int | None = None):
        super().__init__(message)
        self.returncode = returncode


def label(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", value) is not None


def matrix_cells(value: object, base: Path) -> tuple[list[str], list[dict]]:
    required = {"schema", "compare_argv", "canonical_workload", "ragged_workload", "context_cells"}
    if not isinstance(value, dict) or set(value) != required or value.get("schema") != MATRIX_SCHEMA:
        raise ValueError("invalid strict comparison-matrix.v1 envelope")
    argv = value["compare_argv"]
    if (not isinstance(argv, list) or not argv or len(argv) > 256
            or any(not isinstance(item, str) or not item or "\0" in item
                   or len(item.encode()) > 16384 for item in argv)
            or sum(len(item.encode()) for item in argv) > 128 * 1024):
        raise ValueError("compare_argv requires bounded direct arguments")
    controlled = {"--workload", "--concurrency", "--requests", "--self-test"}
    if any(item in controlled or any(item.startswith(flag + "=") for flag in controlled) for item in argv):
        raise ValueError("matrix owns workload, request population and concurrency arguments")
    contexts = value["context_cells"]
    if not isinstance(contexts, list) or len(contexts) > 64:
        raise ValueError("context_cells must be a list of at most 64 cells")

    def path(raw: object) -> Path:
        if not isinstance(raw, str) or not raw or "\0" in raw or len(raw.encode()) > 16384:
            raise ValueError("matrix workload path is invalid")
        candidate = Path(raw)
        return (candidate if candidate.is_absolute() else base / candidate).resolve()

    cells = []
    for kind in ("canonical", "ragged"):
        workload = path(value[kind + "_workload"])
        cells.extend(dict(name=f"{kind}-c{concurrency}", kind=kind,
                          workload=str(workload), concurrency=concurrency)
                     for concurrency in CONCURRENCIES)
    for item in contexts:
        if (not isinstance(item, dict) or set(item) != {"name", "workload", "concurrency"}
                or not label(item.get("name")) or type(item.get("concurrency")) is not int
                or not 1 <= item["concurrency"] <= 64):
            raise ValueError("invalid strict bounded context cell")
        cells.append(dict(name=item["name"], kind="context", workload=str(path(item["workload"])),
                          concurrency=item["concurrency"]))
    names, work = set(), set()
    for cell in cells:
        identity = (cell["workload"], cell["concurrency"])
        if cell["name"] in names or identity in work:
            raise ValueError("duplicate matrix cell name or workload/concurrency work")
        names.add(cell["name"])
        work.add(identity)
    # Refuse undersized/invalid populations before running any earlier cell.
    populations = {}
    for cell in cells:
        if cell["workload"] not in populations:
            populations[cell["workload"]] = len(load_workload(cell["workload"], 512))
        if populations[cell["workload"]] < cell["concurrency"]:
            raise ValueError(f"{cell['name']}: workload population is below concurrency")
        cell["population"] = populations[cell["workload"]]
    return list(argv), cells


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, value: object, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".writing", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as target:
            json.dump(value, target, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        sync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def cell_result_path(output: Path, cell: dict) -> Path:
    return output.with_name(f"{output.name}.{cell['name']}.json")


def validate_outputs(output: Path, cells: list[dict], matrix_path: Path | None,
                     overwrite: bool) -> list[Path]:
    targets = [output, *(cell_result_path(output, cell) for cell in cells)]
    resolved_targets = [path.resolve() for path in targets]
    inputs = {Path(cell["workload"]).resolve() for cell in cells}
    if matrix_path is not None:
        inputs.add(matrix_path.resolve())
    if len(set(resolved_targets)) != len(resolved_targets):
        raise ValueError("matrix output paths alias each other")
    if any(path in inputs for path in resolved_targets):
        raise ValueError("matrix outputs must not alias the matrix or workload inputs")
    existing_targets = [path for path in targets if path.exists()]
    if any(os.path.samefile(left,right) for index,left in enumerate(existing_targets)
           for right in existing_targets[:index]):
        raise ValueError("matrix output paths alias each other")
    if any(os.path.samefile(target,source) for target in existing_targets for source in inputs):
        raise ValueError("matrix outputs must not alias the matrix or workload inputs")
    for path in targets:
        exists = path.exists() or path.is_symlink()
        if exists and not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace it: {path}")
        if exists and not path.is_file():
            raise ValueError(f"matrix output is not a file: {path}")
    return targets


def validate_report(cell: dict, report: object) -> tuple[list, dict]:
    summaries = report.get("summaries") if isinstance(report, dict) else None
    summary = summaries[0] if isinstance(summaries, list) and len(summaries) == 1 else None
    if (not isinstance(report, dict) or report.get("schema") != "zerogravity.serving.vllm-comparison.v2"
            or not isinstance(summary, dict) or summary.get("concurrency") != cell["concurrency"]):
        raise CellFailure("serve-compare output does not describe exactly this matrix cell")
    return summaries, summary


def invoke_compare(cell: dict, compare_argv: list[str], cwd: Path,
                   result_path: Path, overwrite: bool) -> dict:
    command = [sys.executable, "-B", str(HERE / "serve-compare.py"), *compare_argv,
               "--workload", cell["workload"], "--concurrency", str(cell["concurrency"])]
    try:
        with tempfile.TemporaryDirectory(prefix="serve-matrix-") as temporary:
            stdout, stderr = Path(temporary) / "stdout.json", Path(temporary) / "stderr.log"
            with stdout.open("wb") as out, stderr.open("wb") as err:
                completed = subprocess.run(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                           stdout=out, stderr=err, check=False)
            if completed.returncode:
                detail = stderr.read_bytes()[:16384].decode("utf-8", "replace").strip()
                raise CellFailure(detail or f"serve-compare exited {completed.returncode}", completed.returncode)
            try:
                report = strict_json(str(stdout), MAX_REPORT_BYTES)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                raise CellFailure(f"invalid serve-compare output: {type(error).__name__}") from error
            validate_report(cell, report)
            try:
                # Preserve the complete report before its temporary stdout is removed.
                write_json(result_path, report, replace=overwrite)
            except OSError as error:
                raise CellFailure(f"cannot retain serve-compare output: {type(error).__name__}") from error
    except OSError as error:
        raise CellFailure(f"cannot execute serve-compare: {type(error).__name__}") from error
    return report


def run_matrix(value: object, output: Path, base: Path, invoke=None, *,
               matrix_path: Path | None = None, overwrite: bool = False) -> bool:
    compare_argv, cells = matrix_cells(value, base)
    output, *result_paths = validate_outputs(output.absolute(), cells, matrix_path, overwrite)
    aggregate = {"schema":RESULT_SCHEMA, "matrix_schema":MATRIX_SCHEMA,
                 "status":"running", "cells":[]}
    write_json(output, aggregate, replace=overwrite)
    for cell, result_path in zip(cells, result_paths):
        row = {**cell, "status":"running"}
        aggregate["cells"].append(row)
        write_json(output, aggregate)
        try:
            if invoke is None:
                report = invoke_compare(cell, compare_argv, base, result_path, overwrite)
            else:
                report = invoke(cell)
                validate_report(cell, report)
                write_json(result_path, report, replace=overwrite)
            row["comparison_file"] = result_path.name
            summaries, summary = validate_report(cell, report)
            if summary.get("completed_all_requests") is not True:
                raise CellFailure("comparison returned a partial or mismatched cell")
            compact = {key:report[key] for key in ["schema", "benchmark_input_sha256", "model_routes",
                       "run_order", "order_balance_complete"] if key in report}
            compact["summaries"] = summaries
            compact["run_count"] = len(report.get("runs", [])) if isinstance(report.get("runs", []), list) else None
            row.update(status="completed", comparison=compact)
        except CellFailure as error:
            row.update(status="failed", error=str(error), returncode=error.returncode)
            aggregate["status"] = "failed"
            write_json(output, aggregate)
            return False
        except OSError as error:
            row.update(status="failed", error=f"cannot retain serve-compare output: {type(error).__name__}",
                       returncode=None)
            aggregate["status"] = "failed"
            write_json(output, aggregate)
            return False
        write_json(output, aggregate)
    aggregate["status"] = "completed"
    write_json(output, aggregate)
    return True


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="serve-matrix-self-test-") as temporary:
        root = Path(temporary)
        requests = [dict(request_id=f"r{i}",prompt="fixture",arrival_seconds=0) for i in range(64)]
        workload_document = {"schema":BENCH["WORKLOAD_SCHEMA"],"requests":requests}
        for name in ["canonical", "ragged", "context"]:
            (root/f"{name}.json").write_text(json.dumps(workload_document))
        spec = {"schema":MATRIX_SCHEMA, "compare_argv":["--zg-model","z","--vllm-model","v"],
                "canonical_workload":"canonical.json", "ragged_workload":"ragged.json",
                "context_cells":[{"name":"context-256","workload":"context.json","concurrency":16}]}
        matrix_path = root/"matrix.json"
        matrix_path.write_text(json.dumps(spec))

        retained = {"schema":"zerogravity.serving.vllm-comparison.v2",
                    "summaries":[{"concurrency":1,"completed_all_requests":False}],
                    "runs":[{"failure_details":{"phase":"measurement"}}],
                    "provider_lifecycle":{"failure_details":["shutdown"]}}
        real_run = subprocess.run
        def fake_run(command, **kwargs):
            kwargs["stdout"].write(json.dumps(retained).encode())
            return subprocess.CompletedProcess(command,0)
        subprocess.run = fake_run
        direct_result = root/"direct.json"
        try:
            report = invoke_compare({"name":"direct","workload":str(root/"canonical.json"),
                                     "concurrency":1},[],root,direct_result,False)
        finally:
            subprocess.run = real_run
        assert report == retained and json.loads(direct_result.read_text()) == retained

        calls = []
        reports = {}
        def complete(cell):
            calls.append((cell["workload"],cell["concurrency"]))
            report = {"schema":"zerogravity.serving.vllm-comparison.v2",
                      "summaries":[{"concurrency":cell["concurrency"],"completed_all_requests":True}],
                      "runs":[{"failure_details":{"message":"retained"},"cell":cell["name"]}],
                      "provider_lifecycle":{"runs":[{"status":"stopped"}]},
                      "limitations":["retained verbatim"]}
            reports[cell["name"]] = report
            return report
        output = root/"aggregate.json"
        assert run_matrix(spec,output,root,complete,matrix_path=matrix_path)
        result = json.loads(output.read_text())
        assert result["status"] == "completed" and len(result["cells"]) == 13
        assert len(calls) == len(set(calls)) == 13
        assert {row["concurrency"] for row in result["cells"] if row["kind"] != "context"} == set(CONCURRENCIES)
        for row in result["cells"]:
            assert row["comparison_file"] == f"aggregate.json.{row['name']}.json"
            assert json.loads((root/row["comparison_file"]).read_text()) == reports[row["name"]]

        partial_calls = []
        partial_reports = {}
        def partial(cell):
            partial_calls.append(cell["name"])
            good = len(partial_calls) < 3
            report = {"schema":"zerogravity.serving.vllm-comparison.v2",
                      "summaries":[{"concurrency":cell["concurrency"],"completed_all_requests":good}],
                      "runs":[{"completed":good,"failure":"synthetic" if not good else None}],
                      "provider_lifecycle":{"failure_details":["retained"]}}
            partial_reports[cell["name"]] = report
            return report
        partial_output = root/"partial.json"
        assert not run_matrix(spec,partial_output,root,partial,matrix_path=matrix_path)
        partial_result = json.loads(partial_output.read_text())
        assert partial_calls == ["canonical-c1","canonical-c4","canonical-c8"]
        assert [row["status"] for row in partial_result["cells"]] == ["completed","completed","failed"]
        failed_row = partial_result["cells"][-1]
        assert json.loads((root/failed_row["comparison_file"]).read_text()) == partial_reports[failed_row["name"]]

        failed_calls = []
        def failed(cell):
            failed_calls.append(cell["name"])
            raise CellFailure("synthetic subprocess failure", 7)
        failed_output = root/"failed.json"
        assert not run_matrix(spec,failed_output,root,failed,matrix_path=matrix_path)
        assert failed_calls == ["canonical-c1"]
        assert json.loads(failed_output.read_text())["cells"][0]["returncode"] == 7
        assert not cell_result_path(failed_output,{"name":"canonical-c1"}).exists()

        try:
            run_matrix(spec,output,root,complete,matrix_path=matrix_path)
            raise AssertionError("existing outputs were overwritten without --overwrite")
        except FileExistsError:
            pass
        assert run_matrix(spec,output,root,complete,matrix_path=matrix_path,overwrite=True)

        preexisting = root/"preexisting.json"
        sibling = cell_result_path(preexisting,{"name":"canonical-c1"})
        sibling.write_text("do not replace")
        before = len(calls)
        try:
            run_matrix(spec,preexisting,root,complete,matrix_path=matrix_path)
            raise AssertionError("an existing cell output was accepted")
        except FileExistsError:
            pass
        assert len(calls) == before and sibling.read_text() == "do not replace"
        assert not preexisting.exists()

        for alias in [matrix_path,root/"canonical.json"]:
            try:
                run_matrix(spec,alias,root,complete,matrix_path=matrix_path,overwrite=True)
                raise AssertionError("matrix/input output alias was accepted")
            except ValueError:
                pass
        generated_alias = root/"alias.json.canonical-c1.json"
        generated_alias.write_text(json.dumps(workload_document))
        alias_spec = {**spec,"canonical_workload":generated_alias.name}
        try:
            run_matrix(alias_spec,root/"alias.json",root,complete,matrix_path=matrix_path,overwrite=True)
            raise AssertionError("generated cell/input output alias was accepted")
        except ValueError:
            pass

        duplicate = dict(spec)
        duplicate["context_cells"] = [{"name":"duplicate","workload":"canonical.json","concurrency":1}]
        for bad in [duplicate, {**spec,"context_cells":[{"name":"too-wide","workload":"context.json","concurrency":65}]},
                    {**spec,"compare_argv":["--self-test"]}]:
            try:
                matrix_cells(bad,root)
                raise AssertionError("invalid/duplicate matrix work accepted")
            except ValueError:
                pass
        print("SERVING_COMPARISON_MATRIX_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", help="strict comparison-matrix.v1 JSON")
    parser.add_argument("--output", help="aggregate JSON path")
    parser.add_argument("--overwrite", action="store_true", help="replace existing aggregate/cell outputs")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.matrix or not args.output:
        parser.error("--matrix and --output are required")
    matrix_path = Path(args.matrix).resolve()
    try:
        value = strict_json(str(matrix_path))
        ok = run_matrix(value, Path(args.output).absolute(), matrix_path.parent,
                        matrix_path=matrix_path, overwrite=args.overwrite)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Cannot run comparison matrix: {error}\n")
    if not ok:
        parser.exit(1, f"Comparison matrix stopped on failure; inspect {Path(args.output).resolve()}\n")


if __name__ == "__main__":
    main()
