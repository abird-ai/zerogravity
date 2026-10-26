"""Serial owned original CUDA references; no downloads, arithmetic override or permit."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from contract import read_json, read_report_json, request, require, sha
from discovery import Discovery, HeldFiles, command_complete, compose_layouts
from late_layout import LateLayoutCapture

HERE = Path(__file__).resolve().parent
OWNER_SHA = "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
OBSERVER_SHA = "f648eff0ecb09fde0223d570979e9fd505786b90fdd22c2df08a51ff9d3dc93b"
IMAGE = "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe"
IMAGE_MANIFEST = "sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634"
CANCELLATION_SHA = "00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128"


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stage_pins(mode):
    require(mode in ("discover", "capture"), "explicit reference ancestry stage")
    discovery = read_json(HERE / "original-pins.json")
    require(len(discovery) == 42, "complete original discovery source ancestry")
    if mode == "discover":
        return discovery
    captured = read_json(HERE / "capture-pins.json")
    require(len(captured) == 50, "complete retained full-ABI source ancestry")
    changed = {name for name in discovery if captured.get(name) != discovery[name]}
    require(changed == {"run_matrix.py", *["package/" + name for name in (
        "collector.cpp", "host_test.py", "matrix_worker.py", "module_join.h", "native_trace.py",
        "pins.json", "test_native_metadata.py", "validate_native.py", "validate_trajectory.py")]},
        "exact declared full-ABI stage ancestry changes")
    require(set(captured) - set(discovery) == {"package/" + name for name in (
        "capture_catalog.py", "capture_controls.py", "capture_controls_validation.py", "capture_layouts.py",
        "offline_formals.py", "test_capture_catalog.py", "test_capture_layouts.py", "test_capture_resume.py")},
        "exact declared full-ABI stage additions")
    require(set(discovery) <= set(captured), "all original inherited source retained")
    return captured


def verify_controller(pins):
    for name, digest in pins.items():
        require(sha(HERE / name) == digest, "reference successor source changed during collection")


def verify_original(root, mode="discover"):
    pins = stage_pins(mode)
    for relative, pin in pins.items():
        path = root / relative
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == pin["bytes"]
                and sha(path) == pin["sha256"], "changed original dependency: " + relative)
    return pins


def prepare_package(root, output, value, mode="discover"):
    pins = verify_original(root, mode)
    target = output / "package"
    target.mkdir(mode=0o700)
    for relative in pins:
        parts = Path(relative).parts
        if parts[0] != "package":
            continue
        destination = target.joinpath(*parts[1:])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
    for source, name in (("contract.py", "reference_request.py"),
                         ("matrix_contract.py", "matrix_contract.py"),
                         ("tokenize_prompt.py", "tokenize_prompt.py")):
        shutil.copyfile(HERE / source, target / name)
    # Change only the old finite workload membership, preserving every scope,
    # source, interval, packet, resource and extent comparison below it.
    native = target / "validate_native.py"
    old = "prompt_rows in (47,57,62,67,72)"
    original_native = native.read_text()
    require(original_native.count(old) == 1, "exact original native workload guard")
    native.write_text(original_native.replace(old, "prompt_rows == PROMPT_ROWS"))
    write(target / "request.json", value)
    write(output / "source-inheritance.json", {
        "original_files": pins, "original_root": str(root),
        "ancestry_stage": mode,
        "ancestor_manifest_sha256": sha(HERE / ("original-pins.json" if mode == "discover" else "capture-pins.json")),
        "replaced_original_files": ["package/matrix_contract.py", "package/validate_native.py"],
        "new_files": ["reference_request.py", "tokenize_prompt.py", "request.json"],
        "original_observers_unchanged": True,
        "native_validator_change": {"old": old, "new": "prompt_rows == PROMPT_ROWS"},
        "remaining_original_validator_bytes_unchanged": True,
        "native_recipes_qualified": False})
    verify_original(root, mode)
    return target


def verify_primary(package):
    rows = read_json(package / "primary-runtime-pins.json")
    require(len(rows) == 20 and len({r["mounted"] for r in rows}) == len(rows),
            "complete original primary runtime pins")
    for row in rows:
        path = Path(row["path"])
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], "changed primary runtime source")
    return {"namespace": "host", "files_verified": len(rows)}


def verify_collector(root, collector):
    built = read_json(collector / "build-result.json")
    original = root / "package"
    library = collector / "libzg_driver_capture.so"
    require(built["status"] == "BUILT_ONLY_NO_RUNTIME_OR_GPU_QUALIFICATION"
            and built["source_sha256"] == sha(original / "collector.cpp")
            and built["pins_sha256"] == sha(original / "pins.json")
            and built["source_headers"] == {name: sha(original / name)
                for name in ("public_input.h", "preserved_input.h", "module_join.h")}
            and not library.is_symlink() and library.stat().st_size == built["bytes"]
            and sha(library) == built["sha256"], "exact stage source-matching original collector")
    return built


def verify_observer(args):
    actual = sha(args.observer)
    if actual == OBSERVER_SHA:
        require(args.observer_receipt is None, "original observer needs no successor receipt")
        return {"artifact_sha256": actual, "original_artifact_sha256": OBSERVER_SHA,
                "kind": "original_artifact"}
    require(args.observer_receipt is not None, "rebuilt observer requires explicit source/build lineage")
    receipt = read_json(args.observer_receipt)
    require(receipt["schema"] == "zerogravity.qwen35-cuda-reference.observer.v1"
            and receipt["original_artifact_sha256"] == OBSERVER_SHA,
            "declared original observer ancestry")
    require(receipt["original_source"]["sha256"] == "07a10e650bd6efbb982b7f2509a2b6b610b90cf1f3092722e3ac1e4076aaa073"
            and receipt["original_source"]["bytes"] == 21240,
            "exact original observer C++ source, no changed-cap derivative")
    for field in ("artifact", "original_source", "build_provenance"):
        pin = receipt[field]
        require(type(pin) is dict and {"path", "bytes", "sha256"} <= set(pin),
                "complete declared observer artifact/source/build pin")
        path = Path(pin["path"])
        require(path.is_absolute() and not path.is_symlink() and path.is_file()
                and path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"],
                "held declared observer artifact/source/build pin")
    require(Path(receipt["artifact"]["path"]) == args.observer
            and receipt["artifact"]["sha256"] == actual, "actual rebuilt observer artifact")
    build = read_json(receipt["build_provenance"]["path"])
    require(build["source_sha256"] == receipt["original_source"]["sha256"]
            and build["artifact_sha256"] == actual and type(build["argv"]) is list
            and bool(build["argv"]) and all(type(word) is str for word in build["argv"])
            and type(build["dependencies"]) is list and bool(build["dependencies"]),
            "actual retained compiler/source/artifact build relationship")
    for pin in [build["compiler"], *build["dependencies"]]:
        path = Path(pin["path"])
        require(path.is_absolute() and not path.is_symlink() and path.is_file()
                and path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"],
                "unchanged declared observer compiler/dependency")
    return receipt


def bind_command_evidence(helper, lifecycle, directory):
    """Keep original nonce-bound container cleanup; expose each command's evidence."""
    directory.mkdir(mode=0o700)
    sequence = 0

    def command(argv, timeout=10):
        nonlocal sequence
        label = f"command-{sequence:04d}"
        sequence += 1

        def checkpoint(stage, record):
            try:
                write(directory / (label + ".json"), {"stage": stage, "record": record})
                return True
            except Exception:
                # Cleanup must still stop/remove the exact owned container.
                return False

        owner = getattr(helper, "_qwen35_owner", {})
        cid = None
        cidfile = owner.get("cidfile")
        if cidfile is not None and cidfile.is_file():
            value = cidfile.read_text().strip()
            if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
                cid = value
        record = lifecycle.owned_command(argv, label, cid, timeout, checkpoint,
            directory=directory, container_name=owner.get("name"), log_limit_bytes=1024 * 1024)
        require(record["evidence_complete"] and record["result_persisted"]
                and record["leader_reaped"] and record["owned_process_group_gone"]
                and record["status"] == "finished" and not record.get("error_type")
                and not record["errors"] and record["complete"]
                and record["leader_reap_attempts"] == 1 and record["all_nonzero_signals_before_reap"]
                and not record["signals"]
                and type(record["returncode"]) is int
                and not record["timed_out"], "incomplete owned Podman command evidence")
        return subprocess.CompletedProcess(argv, record["returncode"],
            Path(record["stdout"]["path"]).read_bytes(), Path(record["stderr"]["path"]).read_bytes())

    helper.command = command


class OriginalLogs:
    """Separate retained streams over the unchanged original observation interface."""
    def __init__(self, helper):
        self.helper = helper
        self.logs = {}
        self.process = None

    def open(self, process, directory):
        self.process = process
        for name in ("stdout", "stderr"):
            self.logs[name] = self.helper.Log(getattr(process, name), directory / (name + ".log"))

    @property
    def total(self):
        return sum(log.total for log in self.logs.values())

    @property
    def retained(self):
        return sum(log.retained for log in self.logs.values())

    @property
    def errors(self):
        return [error for log in self.logs.values() for error in log.errors]

    @property
    def eof(self):
        return len(self.logs) == 2 and all(log.eof for log in self.logs.values())

    @property
    def closed(self):
        return all(getattr(self.process, stream).closed for stream in ("stdout", "stderr"))

    def drain(self, seconds):
        for log in self.logs.values():
            log.drain(seconds / 2)

    def observe(self, process, seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            self.drain(.05)
            require(all(log.total <= self.helper.CAP for log in self.logs.values()) and not self.errors,
                    "owned original stream failed/truncated")
            status = self.helper.leader_status_unreaped(process)
            if status is not None:
                return status
        raise TimeoutError("owned foreground exit deadline; leader remains unreaped")

    def finish(self):
        failures = []
        for log in self.logs.values():
            try:
                log.finish()
            except BaseException as error:
                failures.append(repr(error))
        for stream in ("stdout", "stderr"):
            pipe = getattr(self.process, stream)
            if not pipe.closed:
                pipe.close()
        require(not failures, "owned stream closure: " + repr(failures))


def invoke(helper, args, package, lane, cpus, cancel):
    # Adapted routing only from original run_matrix.py::invoke; same physical owner.
    directory = args.output / lane
    directory.mkdir(mode=0o700)
    hooks = directory / "empty-oci-hooks"
    hooks.mkdir(mode=0o700)
    if lane == "observed":
        (directory / "driver").mkdir(mode=0o700)
    cache = directory / "runtime-cache"
    for name in ("cache", "triton"):
        (cache / name).mkdir(parents=True, mode=0o700)
    nonce = uuid.uuid4().hex
    name = "zg-qwen35-reference-" + lane + "-" + nonce
    cid = directory / "owned-container.cid"
    environment = {
        "ZG_PROBE_ALLOWED_CPUS": ",".join(map(str, cpus)), "ZG_SHORT_HF_MODEL": "0.8b",
        "ZG_SHORT_HF_ROWS": str(args.request_value["prompt_tokens"]), "ZG_CANONICAL_STAGE": "reference",
        "PYTHONPATH": "/capture:/capture/guards/0.8b", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "VLLM_PLUGINS": "", "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_CACHE_PATH": "/root/.cache/nv",
        "TRITON_CACHE_DIR": "/root/.triton", "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torchinductor",
        "LD_AUDIT": "", "LD_PRELOAD": "/interposer/libzg_gemm_observer.so" if lane == "observed" else "",
        "ZG_DRIVER_PASS": args.mode if lane == "observed" else "plain"}
    if lane == "tokenize":
        environment.update(NVIDIA_VISIBLE_DEVICES="void", CUDA_VISIBLE_DEVICES="")
    argv = helper.PODMAN + ["--hooks-dir", str(hooks), "run", "--name", name, "--label", helper.LABEL + "=" + nonce,
        "--cidfile", str(cid), "--rm", "--pull=never", "--network=none", "--read-only",
        "--group-add", "keep-groups", "--cap-drop=all", "--security-opt=no-new-privileges",
        "--memory=12g", "--memory-swap=12g", "--cpus=4", "--pids-limit=512", "--shm-size=1g",
        "--ulimit", "fsize=67108864:67108864", "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g"]
    if lane != "tokenize":
        argv += ["--device", "nvidia.com/gpu=all"]
    mounts = [(package, "/capture:ro"),
              (directory, "/output:rw"), (cache / "cache", "/root/.cache:rw"),
              (cache / "triton", "/root/.triton:rw"), (args.source, "/source:ro"),
              (args.generation, "/canonical/common-generation:ro")]
    if lane == "observed":
        mounts += [(args.observer.parent, "/interposer:ro"), (args.collector, "/collector:ro")]
        if args.mode == "capture":
            environment["ZG_DRIVER_LAYOUT_SHA"] = sha(args.discovery / "layouts.json")
            environment["ZG_DRIVER_LAYOUT_CATALOG_SHA"] = sha(args.output / "layout-catalog.json")
            mounts += [(args.discovery, "/discovery:ro"),
                       (args.output / "layout-catalog.json", "/layout-catalog.json:ro")]
    for key, value in environment.items():
        argv += ["-e", key + "=" + value]
    for source, target in mounts:
        argv += ["-v", str(source) + ":" + target]
    script = "tokenize_prompt.py" if lane == "tokenize" else "capture_matrix.py"
    argv += ["--entrypoint", "python3", IMAGE, "/capture/" + script]
    if lane != "tokenize":
        argv.append(lane)
    receipt = {"status": "UNQUALIFIED", "mode": lane, "argv": argv, "owned_name": name,
               "owned_nonce": nonce, "image_id": IMAGE, "expected_cpus": cpus,
               "helper_sha256": OWNER_SHA, "existing_observer_sha256": sha(args.observer),
               "model": "0.8b", "prompt_rows": args.request_value["prompt_tokens"],
               "request": args.request_value, "performance_qualified": False,
               "observer_lineage": args.observer_lineage,
               "primary_source_before": verify_primary(package)}
    helper._qwen35_owner = {"name": name, "cidfile": cid}
    write(directory / "invocation.json", receipt)
    child = log = failure = None
    started = time.monotonic()
    try:
        cancel.check()
        require(not list(hooks.iterdir()), "empty explicit OCI hooks namespace")
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, start_new_session=True)
        log = OriginalLogs(helper)
        log.open(child, directory)
        receipt["owned_cli_pid"] = child.pid
        write(directory / "invocation.json", receipt)
        receipt["exit_code"] = cancel.observe(log, child, 900)
    except BaseException as caught:
        failure = caught
    finally:
        receipt["cleanup_errors"] = helper.cleanup(child, name, nonce, cid, receipt) if child is not None else []
        try:
            if log is not None:
                log.finish()
        except BaseException as caught:
            receipt["cleanup_errors"].append(repr(caught))
        finally:
            if log is not None:
                receipt["log_cleanup"] = {"pipe_eof": log.eof, "all_closed": log.closed,
                                          "total_bytes": log.total, "retained_bytes": log.retained}
                receipt["streams"] = {name: {"path": str(current.path), "bytes_seen": current.total,
                    "bytes_retained": current.retained, "capture_complete": current.eof and current.closed
                    and not current.errors and current.total == current.retained,
                    "sha256_retained": sha(current.path)} for name, current in log.logs.items()}
            elif child is not None and child.stdout is not None:
                child.stdout.close()
                if child.stderr is not None:
                    child.stderr.close()
        try:
            receipt["primary_source_after"] = verify_primary(package)
        except BaseException as caught:
            failure = caught if failure is None else failure
        receipt["elapsed_seconds_not_benchmark"] = time.monotonic() - started
        if failure is not None:
            receipt["failure"] = repr(failure)
        if failure is None and receipt.get("exit_code") == 0 and not receipt["cleanup_errors"]:
            receipt["status"] = "CAPTURE_COMPLETED_REQUIRES_VALIDATION"
        write(directory / "invocation.json", receipt)
    cancel.check()
    require(receipt["status"] == "CAPTURE_COMPLETED_REQUIRES_VALIDATION",
            "owned original failed; retained complete failure scope")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for option in ("reference-root", "owner", "observer", "cancellation", "collector",
                   "source", "generation", "request", "output", "lifecycle-source"):
        result.add_argument("--" + option, type=Path, required=True)
    result.add_argument("--mode", choices=("discover", "capture"), required=True)
    result.add_argument("--discovery", type=Path)
    result.add_argument("--layout-discovery", type=Path, action="append", default=[],
                        help="additional passing discovery receipt used only for exact whole-image ABI union")
    for option in ("late-layout-capture", "late-layout-controller", "late-layout-pins"):
        result.add_argument("--" + option, type=Path)
    result.add_argument("--late-layout-controller-sha256")
    result.add_argument("--late-layout-pins-sha256")
    result.add_argument("--observer-receipt", type=Path)
    result.add_argument("--lifecycle-sha256", required=True)
    return result


def main(args, cancel):
    with ExitStack() as held:
        return run(args, cancel, held)


def run(args, cancel, held):
    command_evidence = held.enter_context(HeldFiles())
    controller_pins = {path.name: sha(path) for path in HERE.iterdir()
                       if path.suffix in (".py", ".json") and path.is_file()}
    for value in vars(args).values():
        if isinstance(value, Path):
            require(value.is_absolute(), "explicit absolute source/output paths")
    require(not args.output.exists(), "fresh output; retain failed/interrupted originals")
    args.request_value = request(read_json(args.request))
    require(all(path.is_absolute() for path in args.layout_discovery), "absolute additional discovery paths")
    verify_original(args.reference_root, args.mode)
    require(sha(args.owner) == OWNER_SHA, "exact original physical owner")
    args.observer_lineage = verify_observer(args)
    require(sha(args.lifecycle_source) == args.lifecycle_sha256, "reviewed shared command lifecycle source")
    lifecycle = load(args.lifecycle_source, "qwen35_reference_command_lifecycle")
    helper = load(args.owner, "qwen35_reference_owned_container")
    args.output.mkdir(mode=0o700)
    bind_command_evidence(helper, lifecycle, args.output / "commands")
    inspected = helper.command(helper.PODMAN + ["image", "inspect", IMAGE], 15)
    require(inspected.returncode == 0, "existing approved image only")
    image, = json.loads(inspected.stdout)
    require(image["Id"] == IMAGE.removeprefix("sha256:") and image["Digest"] == IMAGE_MANIFEST,
            "immutable original vLLM/Torch runtime")
    built = verify_collector(args.reference_root, args.collector)
    require((args.mode == "capture") == (args.discovery is not None)
            and (args.mode == "capture" or not args.layout_discovery), "capture requires same-request discovery")
    discoveries = []
    late_sources = []
    late_arguments = (args.late_layout_capture, args.late_layout_controller, args.late_layout_controller_sha256,
                      args.late_layout_pins, args.late_layout_pins_sha256)
    require(not any(value is not None for value in late_arguments)
            or (all(value is not None for value in late_arguments) and args.mode == "capture"),
            "late-reflection capture/controller/pinset paths and digests supplied together for capture only")
    if args.discovery is not None:
        require(len(args.layout_discovery) <= 3
                and len(set([args.discovery, *args.layout_discovery])) == 1 + len(args.layout_discovery),
                "finite distinct passing discovery receipts")
        discoveries = [held.enter_context(Discovery(args.discovery, args.request_value))]
        discoveries += [held.enter_context(Discovery(path)) for path in args.layout_discovery]
        if args.late_layout_capture is not None:
            late_sources = [held.enter_context(LateLayoutCapture(args.late_layout_capture,
                args.late_layout_controller, args.late_layout_controller_sha256,
                args.late_layout_pins, args.late_layout_pins_sha256, discoveries))]
            write(args.output / "failed-late-layout-facts.json", late_sources[0].provenance())
        write(args.output / "layout-catalog.json", compose_layouts([*discoveries, *late_sources]))
        require((args.output / "layout-catalog.json").stat().st_size <= 4 * 1024 * 1024,
                "original full-ABI capture catalog bound")
    cpus = sorted(os.sched_getaffinity(0))[:4]
    require(len(cpus) == 4, "four original allowed CPUs")
    write(args.output / "image-inspect.json", image)
    package = prepare_package(args.reference_root, args.output, args.request_value, args.mode)
    initial_package_pins = {str(p.relative_to(package)): sha(p) for p in package.rglob("*") if p.is_file()}
    invoke(helper, args, package, "tokenize", cpus, cancel)
    for relative, digest in initial_package_pins.items():
        require(sha(package / relative) == digest, "reference source changed during tokenizer capture")
    tokenized = request(read_json(args.output / "tokenize/tokenized-request.json"), tokenized=True)
    require({k: v for k, v in tokenized.items() if k != "prompt_token_ids"} == args.request_value,
            "same requested source/prompt/generation after actual tokenization")
    if args.discovery is not None:
        require(discoveries[0].result["tokenized_request"] == tokenized, "same actual tokenizer-derived discovery operands")
    write(package / "request.json", tokenized)
    package_pins = {str(p.relative_to(package)): sha(p) for p in package.rglob("*") if p.is_file()}
    write(args.output / "prepared-package.json", package_pins)
    for lane in ("plain", "observed"):
        verify_controller(controller_pins)
        verify_original(args.reference_root, args.mode)
        verify_collector(args.reference_root, args.collector)
        verify_observer(args)
        require(sha(args.lifecycle_source) == args.lifecycle_sha256 and sha(args.owner) == OWNER_SHA,
                "same reviewed process/command lifecycle source")
        for discovery in [*discoveries, *late_sources]:
            discovery.check()
        invoke(helper, args, package, lane, cpus, cancel)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(package) + ":" + str(package / "guards/0.8b"))
    def validation_checkpoint(stage, record):
        try:
            write(args.output / "validation-command.json", {"stage": stage, "record": record})
            return True
        except Exception:
            return False

    checked = lifecycle.owned_command([sys.executable, "-B", str(package / "validate_trajectory.py"),
        str(args.output), args.mode], "validation", None, 900, validation_checkpoint,
        directory=args.output, container_name=None, log_limit_bytes=64 * 1024 * 1024, env=environment)
    for relative, digest in package_pins.items():
        require(sha(package / relative) == digest, "prepared reference source changed")
    verify_controller(controller_pins)
    verify_original(args.reference_root, args.mode)
    verify_collector(args.reference_root, args.collector)
    verify_observer(args)
    require(sha(args.lifecycle_source) == args.lifecycle_sha256 and sha(args.owner) == OWNER_SHA,
            "process/command lifecycle source unchanged")
    for discovery in [*discoveries, *late_sources]:
        discovery.check()
    command_complete({"record": checked}, command_evidence, args.output)
    trajectory = read_report_json(args.output / "trajectory-validation.json")
    comparison_sha = None
    if discoveries:
        def comparison_checkpoint(stage, record):
            try:
                write(args.output / "comparison-command.json", {"stage": stage, "record": record})
                return True
            except Exception:
                return False

        comparison = lifecycle.owned_command([sys.executable, "-B", str(HERE / "discovery.py"),
            "--discovery", str(args.discovery), "--capture", str(args.output),
            "--output", str(args.output / "discovery-comparison.json")],
            "comparison", None, 900, comparison_checkpoint, directory=args.output,
            container_name=None, log_limit_bytes=1024 * 1024,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        command_complete({"record": comparison}, command_evidence, args.output, "comparison")
        verify_controller(controller_pins)
        for relative, digest in package_pins.items():
            require(sha(package / relative) == digest, "prepared reference source changed during comparison")
        for discovery in [*discoveries, *late_sources]:
            discovery.check()
        retained = held.enter_context(HeldFiles())
        compared = retained.document(args.output / "discovery-comparison.json")
        require(compared["status"] == "PASS_COMPLETE_SAME_REQUEST_ORIGINAL_BYTES"
                and all(compared[name] is False for name in
                    ("native_recipes_qualified", "model_execution_qualified", "performance_qualified")),
                "passing complete original byte comparison, no implementation qualification")
        for pin in compared["held_capture_inputs"]:
            path = Path(pin["path"])
            require(path.is_absolute() and path.is_relative_to(args.output) and ".." not in path.parts,
                    "comparison owns only this capture's actual inputs")
            retained.hold(path, expected_sha=pin["sha256"], expected_bytes=pin["bytes"])
        retained.check()
        comparison_sha = sha(args.output / "discovery-comparison.json")
    command_evidence.check()
    write(args.output / "reference-result.json", {
        "schema": "zerogravity.qwen35-cuda-reference.result.v1", "request": args.request_value,
        "tokenized_request": tokenized, "mode": args.mode, "original_trajectory_validation": "PASS",
        "trajectory_sha256": sha(args.output / "trajectory-validation.json"),
        "layouts_sha256": sha(args.output / "layouts.json") if args.mode == "discover" else None,
        "prepared_package_sha256": sha(args.output / "prepared-package.json"),
        "ancestry_stage": args.mode,
        "ancestor_manifest_sha256": sha(HERE / ("original-pins.json" if args.mode == "discover" else "capture-pins.json")),
        "collector_build": built,
        "successor_source_sha256": controller_pins,
        "discovery_comparison_sha256": comparison_sha,
        "layout_catalog_sha256": sha(args.output / "layout-catalog.json") if discoveries else None,
        "held_discovery_receipts": [{"root": str(item.root), "inputs": item.pins()} for item in discoveries],
        "failed_late_layout_source": late_sources[0].provenance() if late_sources else None,
        "observer_lineage": args.observer_lineage,
        "command_lifecycle_sha256": args.lifecycle_sha256,
        "natural_calls": trajectory["outputs"], "eight_calls_complete": trajectory["eight_calls_complete"],
        "native_recipes_qualified": False, "model_execution_qualified": False, "performance_qualified": False})


if __name__ == "__main__":
    arguments = parser().parse_args()
    require(sha(arguments.cancellation) == CANCELLATION_SHA, "original record-only signal contract")
    signals = load(arguments.cancellation, "qwen35_reference_cancellation")
    with signals.SigtermCancellation() as active:
        main(arguments, active)
