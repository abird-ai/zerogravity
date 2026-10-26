"""Held original discovery ancestry, finite layout union and complete raw comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from contract import SOURCE_MANIFEST, pairs, read_json, request, require, sha, stamp

MIB = 1024 * 1024
HERE = Path(__file__).resolve().parent


class HeldFiles:
    def __init__(self):
        self.files = {}

    def close(self):
        for item in self.files.values():
            os.close(item["fd"])
        self.files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def digest(fd, size):
        result = hashlib.sha256()
        offset = 0
        while offset < size:
            data = os.pread(fd, min(MIB, size - offset), offset)
            require(bool(data), "complete held reference stream")
            result.update(data)
            offset += len(data)
        require(os.pread(fd, 1, size) == b"", "held reference EOF")
        return result.hexdigest()

    def hold(self, path, expected_sha=None, expected_bytes=None, limit=64 * MIB):
        path = Path(path)
        key = str(path)
        if key in self.files:
            item = self.files[key]
        else:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                before = os.fstat(fd)
                require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit,
                        "bounded regular discovery/reference input")
                item = {"fd": fd, "path": path, "stamp": stamp(before), "bytes": before.st_size,
                        "sha256": self.digest(fd, before.st_size)}
                require(item["stamp"] == stamp(os.fstat(fd)) == stamp(path.lstat()),
                        "unchanged held reference identity")
                self.files[key] = item
            except BaseException:
                os.close(fd)
                raise
        require(item["bytes"] <= limit
                and (expected_bytes is None or item["bytes"] == expected_bytes)
                and (expected_sha is None or item["sha256"] == expected_sha),
                "exact retained discovery/reference size/SHA")
        return item

    def document(self, path, expected_sha=None, limit=64 * MIB):
        item = self.hold(path, expected_sha=expected_sha, limit=limit)
        data = os.pread(item["fd"], item["bytes"], 0)
        require(len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"],
                "same complete held metadata bytes")
        return json.loads(data, object_pairs_hook=pairs,
                          parse_constant=lambda _: require(False, "nonfinite reference JSON"))

    def content(self, path):
        item = self.hold(path)
        data = os.pread(item["fd"], item["bytes"], 0)
        require(len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"],
                "same complete retained source bytes")
        return data

    def check(self):
        for item in self.files.values():
            require(item["stamp"] == stamp(os.fstat(item["fd"])) == stamp(item["path"].lstat())
                    and self.digest(item["fd"], item["bytes"]) == item["sha256"]
                    and item["stamp"] == stamp(os.fstat(item["fd"])),
                    "held original reference changed")

    def pins(self):
        return [{key: str(value) if key == "path" else value for key, value in item.items()
                 if key in ("path", "bytes", "sha256")} for item in self.files.values()]


def streams(held, records, directory, prefix="", command=False):
    require(set(records) == {"stdout", "stderr"}, "separate complete original evidence streams")
    total = 0
    for name, record in records.items():
        require(type(record["bytes_seen"]) is type(record["bytes_retained"]) is int
                and 0 <= record["bytes_seen"] == record["bytes_retained"] <= 64 * MIB
                and record["capture_complete"] is True,
                "original stream complete, no dropped or truncated suffix")
        if command:
            require(record["eof"] is True and record["reader_complete"] is True
                    and record["reader_error"] is None and record["truncated"] is False
                    and type(record["eof_observed"]) is dict and bool(record["eof_observed"]),
                    "command stream has actual complete EOF/reader evidence")
        path = Path(record["path"])
        expected = directory / (prefix + name + ".log")
        require(path.is_absolute() and path.resolve(strict=True) == expected.resolve(strict=True),
                "retained stream is this original owner's actual evidence file")
        for candidate in {path, expected}:
            held.hold(candidate, expected_sha=record["sha256_retained"],
                      expected_bytes=record["bytes_retained"])
        total += record["bytes_retained"]
    return total


def command_complete(document, held=None, directory=None, label="validation"):
    record = document["record"]
    require(record["status"] == "finished" and record["returncode"] == 0
            and record["evidence_complete"] and record["result_persisted"]
            and record["complete"] and record["leader_reaped"] and record["owned_process_group_gone"]
            and record["leader_reap_attempts"] == 1 and record["all_nonzero_signals_before_reap"]
            and not record.get("error_type") and not record["signals"] and not record["errors"]
            and not record["timed_out"],
            "passing owned original validator command")
    if held is not None:
        streams(held, {name: record[name] for name in ("stdout", "stderr")}, directory,
                prefix=label + ".", command=True)


def owned_invocation(held, directory, failed=False):
    invocation = held.document(directory / "invocation.json", limit=MIB)
    require(invocation["status"] == ("UNQUALIFIED" if failed else "CAPTURE_COMPLETED_REQUIRES_VALIDATION")
            and invocation["exit_code"] == (1 if failed else 0) and not invocation["cleanup_errors"]
            and invocation["owned_container_absent_after_reap"]
            and invocation["foreground_cleanup"]["leader_reaped"]
            and invocation["foreground_cleanup"]["group_absent"]
            and invocation["foreground_cleanup"]["all_nonzero_signals_before_reap"]
            and not invocation["foreground_cleanup"].get("stage_errors")
            and invocation["log_cleanup"]["pipe_eof"] and invocation["log_cleanup"]["all_closed"]
            and invocation["primary_source_before"] == invocation["primary_source_after"]
            == {"namespace": "host", "files_verified": 20},
            "complete original process/source owner, not outer adapter status")
    total = streams(held, invocation["streams"], directory)
    require(total == invocation["log_cleanup"]["total_bytes"] == invocation["log_cleanup"]["retained_bytes"],
            "complete combined original stream extent")
    return invocation


def archive(held, directory, receipt):
    require(receipt["file"] == "raw.bin" and type(receipt["bytes"]) is int
            and 0 <= receipt["bytes"] <= 32 * MIB and type(receipt["records"]) is list,
            "bounded complete original raw archive")
    offset, records = 0, []
    for record in receipt["records"]:
        require(type(record["offset"]) is int and record["offset"] == offset
                and type(record["bytes"]) is int and 0 < record["bytes"] <= MIB,
                "ordered complete original raw record")
        records.append({key: record.get(key) for key in
                        ("label", "case", "call", "layer", "dtype", "shape", "stride", "offset", "bytes", "sha256")})
        offset += record["bytes"]
    require(offset == receipt["bytes"], "no omitted raw record suffix")
    item = held.hold(directory / "raw.bin", expected_sha=receipt["sha256"],
                     expected_bytes=receipt["bytes"], limit=32 * MIB)
    return {"item": item, "records": records}


def archives(held, root):
    result, completions = {}, {}
    for mode in ("plain", "observed"):
        directory = root / mode
        owned_invocation(held, directory)
        worker = held.document(directory / "worker-report.json")
        generated = held.document(directory / "result.json", limit=MIB)["cases"]
        require(len(generated) == 1, "one actual request per source reference")
        completions[mode] = {key: generated[0][key] for key in
                             ("case", "prompt_ids", "completion_ids", "finish_reason", "stop_reason")}
        states = held.document(directory / "all42/report.json")
        require(states["active_domains"] == 42 and states["outputs"] == len(generated[0]["completion_ids"])
                and set(states["archives"]) == set(states["prior_archives"]) == {str(i) for i in range(24)},
                "complete all24/all42 active and consumed-prior archive cohort")
        for key, subdir, receipt in (("common", directory, worker["raw"]),
                                     ("parameters", directory / "parameters", worker["parameter_raw"]),
                                     ("final", directory / "final", worker["final_head"]["raw"])):
            result[(mode, key)] = archive(held, subdir, receipt)
        for layer in range(24):
            result[(mode, "all42", layer)] = archive(
                held, directory / "all42" / str(layer), states["archives"][str(layer)])
            result[(mode, "prior", layer)] = archive(
                held, directory / "all42-prior" / str(layer), states["prior_archives"][str(layer)])
        for name in ("worker-source-after.json", "main-source-after.json"):
            source = held.document(directory / name, limit=MIB)
            require(source["manifest_sha256"] == SOURCE_MANIFEST and source["text_roles"] == 320
                    and source["all_tensors"] == 488 and source["full_payload_freshly_verified"] is True,
                    "same fully verified original BF16 checkpoint")
    require(completions["plain"] == completions["observed"], "paired natural original token histories")
    return result, completions


def prepared_discovery(held, root, prepared, tokenized):
    original = read_json(HERE / "original-pins.json")
    expected = {str(Path(name).relative_to("package")): pin for name, pin in original.items()
                if name.startswith("package/")}
    require(set(prepared) == set(expected) | {"reference_request.py", "tokenize_prompt.py", "request.json"},
            "exact inherited discovery source package cohort")
    for relative, pin in expected.items():
        path = root / "package" / relative
        if relative not in ("matrix_contract.py", "validate_native.py"):
            held.hold(path, expected_sha=pin["sha256"], expected_bytes=pin["bytes"])
    native = held.content(root / "package/validate_native.py").decode("utf-8")
    require(native.count("prompt_rows == PROMPT_ROWS") == 1
            and hashlib.sha256(native.replace("prompt_rows == PROMPT_ROWS",
                "prompt_rows in (47,57,62,67,72)").encode()).hexdigest()
            == expected["validate_native.py"]["sha256"], "only declared original finite prompt guard changed")
    for name in ("matrix_contract.py", "tokenize_prompt.py"):
        held.hold(root / "package" / name, expected_sha=sha(HERE / name))
    # First completed47 used the same request logic before bounded report reading
    # was added. Preserve that accepted byte identity explicitly.
    require(prepared["reference_request.py"] in {
        "0df35c710140686b49b90e7770bf5beafe5f76ac1fcc111ae2c7858a6fcf79a3", sha(HERE / "contract.py")},
        "authenticated successor request contract ancestry")
    require(held.document(root / "package/request.json", limit=MIB) == tokenized,
            "actual discovery package used the same tokenizer-derived request")


def native_inputs(held, root, mode):
    observed = root / "observed"
    native = held.document(observed / "native-metadata-validation.json")
    driver = held.document(observed / "driver/capture.json")
    result = held.document(observed / "driver-result.json", limit=MIB)
    require(native["mode"] == mode and driver["mode"] == (1 if mode == "discover" else 2)
            and driver["failures"] == 0 and result["pass"] == mode and result["collector_status"] == 0
            and native["complete_selected_argument_fields"] is (mode == "capture"),
            "actual validated original native stage, not a collector alias")
    held.hold(observed / "driver/data.bin", expected_sha=native["parameter_arena_sha256"],
              expected_bytes=driver["data_bytes"])
    for module in driver["modules"]:
        require(module["ready"] is True, "complete materialized original native image")
        held.hold(observed / "driver" / f'module-{module["index"]:04d}.cubin',
                  expected_sha=module["sha256"], expected_bytes=module["bytes"], limit=32 * MIB)
    if mode == "capture":
        held.document(observed / "selected-native-input-validation.json")
    return native, driver


def layout_key(row):
    require(set(row) == {"callback_name", "name", "module_sha256", "params"}, "complete original reflected layout fields")
    key = (row["callback_name"], row["name"], row["module_sha256"])
    require(all(type(value) is str and value for value in key)
            and len(key[2]) == 64 and all(c in "0123456789abcdef" for c in key[2]),
            "actual callback/entry/whole image SHA")
    end = 0
    require(type(row["params"]) is list and 0 < len(row["params"]) <= 64, "bounded complete reflected formals")
    for offset, size in row["params"]:
        require(type(offset) is type(size) is int and offset >= end and size > 0 and offset + size <= 4096,
                "complete ordered reflected ABI fields")
        end = offset + size
    return key


class Discovery(HeldFiles):
    def __init__(self, root, expected_request=None):
        super().__init__()
        self.root = Path(root)
        try:
            self.result = self.document(self.root / "reference-result.json", limit=MIB)
            require(self.result["schema"] == "zerogravity.qwen35-cuda-reference.result.v1"
                    and self.result["mode"] == "discover" and self.result["original_trajectory_validation"] == "PASS"
                    and all(self.result[name] is False for name in
                            ("native_recipes_qualified", "model_execution_qualified", "performance_qualified")),
                    "passing original discovery, no implementation qualification")
            request(self.result["request"])
            request(self.result["tokenized_request"], tokenized=True)
            require({k: v for k, v in self.result["tokenized_request"].items() if k != "prompt_token_ids"}
                    == self.result["request"], "tokenized request preserves exact original request")
            require(expected_request is None or self.result["request"] == expected_request,
                    "same exact requested source/prompt/generation/batch")
            command_complete(self.document(self.root / "validation-command.json", limit=MIB), self, self.root)
            self.trajectory = self.document(self.root / "trajectory-validation.json", self.result["trajectory_sha256"])
            self.layout = self.document(self.root / "layouts.json", self.result["layouts_sha256"], limit=MIB)
            prepared = self.document(self.root / "prepared-package.json", self.result["prepared_package_sha256"], limit=MIB)
            for relative, digest in prepared.items():
                path = Path(relative)
                require(not path.is_absolute() and ".." not in path.parts, "bounded retained package member")
                self.hold(self.root / "package" / path, expected_sha=digest)
            prepared_discovery(self, self.root, prepared, self.result["tokenized_request"])
            tokenizer = owned_invocation(self, self.root / "tokenize")
            require(tokenizer["request"] == self.result["request"]
                    and self.document(self.root / "tokenize/tokenized-request.json", limit=MIB)
                    == self.result["tokenized_request"], "complete original tokenizer request ancestry")
            self.document(self.root / "tokenize/tokenization.json", limit=MIB)
            require(self.trajectory["model"] == "0.8b" and self.trajectory["prompt_rows"] == self.result["request"]["prompt_tokens"]
                    and self.trajectory["original_source_and_lifetime_validation"] == "PASS"
                    and self.trajectory["common_final_fullv_greedy_validation"] == "PASS"
                    and self.trajectory["all24_all42_post_validation"] == "PASS"
                    and self.trajectory["native"]["mode"] == "discover",
                    "complete same-source original discovery trajectory")
            self.raw, self.completions = archives(self, self.root)
            outputs = len(self.completions["plain"]["completion_ids"])
            require(self.completions["plain"]["prompt_ids"] == self.result["tokenized_request"]["prompt_token_ids"]
                    and self.result["natural_calls"] == self.trajectory["outputs"] == outputs
                    and self.result["eight_calls_complete"] == self.trajectory["eight_calls_complete"] == (outputs == 8),
                    "actual full raw history is the declared tokenizer-derived request")
            for lane in ("plain", "observed"):
                invocation = self.document(self.root / lane / "invocation.json", limit=MIB)
                require(invocation["request"] == self.result["request"],
                        "original invocation used the same exact request contract")
            observed = self.root / "observed"
            native, driver = native_inputs(self, self.root, "discover")
            require(native == self.trajectory["native"], "native discovery metadata is the actual validated original")
            functions = {row["index"]: row for row in driver["functions"]}
            modules = {row["index"]: row for row in driver["modules"]}
            actual, launches = {}, native["launches"]
            for launch in launches:
                function = functions[launch["function"]]
                module = modules[function["module"]]
                require(function["reflected"] is True and module["ready"] is True
                        and launch["module"] == function["module"]
                        and module["sha256"] == launch["module_sha256"]
                        and function["name"] == launch["entry"] and launch["parameters"] is None,
                        "actual discovered selected whole image/function/ABI, no invented packet")
                row = {key: function[key] for key in ("callback_name", "name", "params")}
                row["module_sha256"] = module["sha256"]
                key = layout_key(row)
                require(key not in actual or actual[key] == row, "consistent actual selected reflected layout")
                actual[key] = row
                self.hold(observed / "driver" / f'module-{module["index"]:04d}.cubin',
                          expected_sha=module["sha256"], expected_bytes=module["bytes"], limit=32 * MIB)
            require(self.layout["schema"] == "zg.vllm.blas-layouts.v1" and 0 < len(actual) <= 128
                    and len(self.layout["layouts"]) == len(actual)
                    and {layout_key(row): row for row in self.layout["layouts"]} == actual,
                    "layout union is exactly the validated actual selected resource cohort")
            self.check()
        except BaseException:
            self.close()
            raise


def compose_layouts(discoveries):
    rows = {}
    for discovery in discoveries:
        discovery.check()
        for row in discovery.layout["layouts"]:
            key = layout_key(row)
            require(key not in rows or rows[key] == row, "conflicting exact original fullimage ABI")
            rows[key] = row
    require(0 < len(rows) <= 128, "finite original capture layout capacity")
    return {"schema": "zg.short-hf.capture-layout-union.v1", "layouts": [rows[key] for key in sorted(rows)],
            "provenance": [item.provenance() if hasattr(item, "provenance") else
                           {"discovery": str(item.root), "held_inputs": item.pins()} for item in discoveries],
            "native_memory_qualified": False, "mathematical_replay_qualified": False,
            "model_qualified": False, "performance_qualified": False}


def compare_archives(expected_archives, current):
    require(set(current) == set(expected_archives), "same complete raw archive cohort")
    total = 0
    for key, actual in current.items():
        expected = expected_archives[key]
        require(actual["records"] == expected["records"]
                and actual["item"]["bytes"] == expected["item"]["bytes"],
                "same complete original source/state/final/fullV record metadata")
        count = actual["item"]["bytes"]
        offset = 0
        while offset < count:
            size = min(MIB, count - offset)
            left = os.pread(actual["item"]["fd"], size, offset)
            right = os.pread(expected["item"]["fd"], size, offset)
            require(len(left) == len(right) == size and left == right,
                    "capture changed original complete source/state/final/fullV/prior bytes")
            offset += size
        total += count
    return total


def compare(discovery, captured):
    with HeldFiles() as held:
        captured = Path(captured)
        current, completions = archives(held, captured)
        native, _ = native_inputs(held, captured, "capture")
        trajectory = held.document(captured / "trajectory-validation.json")
        require(trajectory["native"] == native and all(trajectory[name] == "PASS" for name in
                ("original_source_and_lifetime_validation", "common_final_fullv_greedy_validation",
                 "all24_all42_post_validation")), "actual original packed capture trajectory passed")
        command_complete(held.document(captured / "validation-command.json", limit=MIB), held, captured)
        require(completions == discovery.completions,
                "capture preserves exact original natural history")
        total = compare_archives(discovery.raw, current)
        held.check()
        discovery.check()
        return {"schema": "zerogravity.qwen35-cuda-reference.discovery-comparison.v1",
                "status": "PASS_COMPLETE_SAME_REQUEST_ORIGINAL_BYTES", "archive_count": len(current),
                "compared_bytes": total, "discovery": str(discovery.root),
                "held_discovery_inputs": discovery.pins(), "held_capture_inputs": held.pins(),
                "native_recipes_qualified": False, "model_execution_qualified": False, "performance_qualified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with Discovery(args.discovery) as original:
        result = compare(original, args.capture)
    with args.output.open("x", encoding="utf-8") as stream:
        os.chmod(args.output, 0o600)
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
