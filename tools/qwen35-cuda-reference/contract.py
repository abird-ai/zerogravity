"""Finite original BF16 reference requests; no inference or native permit."""
import hashlib
import json
import os
from pathlib import Path
import stat

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
SOURCE_MANIFEST = "c4c1d349743c0e0ba51ab07f769348e6b13f9896c354303c7af5e839c11dbf09"
GENERATION = {"max_new_tokens": 8, "eos_token_id": 248046, "temperature": 0.0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def read_json(path):
    return _read_json(path, 1024 * 1024)


def read_report_json(path):
    """Original full trajectory reports retain the original64MiB report bound."""
    return _read_json(path, 64 * 1024 * 1024)


def _read_json(path, limit):
    path = Path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= limit,
                "bounded regular JSON")
        data = stream.read(limit + 1)
        require(len(data) == before.st_size and stream.read(1) == b""
                and stamp(before) == stamp(os.fstat(stream.fileno())) == stamp(path.lstat()),
                "complete unchanged held JSON/EOF")
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "nonfinite JSON"))


def sha(path):
    path = Path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), "regular authenticated source/artifact")
        result = hashlib.file_digest(stream, "sha256").hexdigest()
        require(stream.read(1) == b""
                and stamp(before) == stamp(os.fstat(stream.fileno())) == stamp(path.lstat()),
                "complete unchanged held source/artifact/EOF")
        return result


def stamp(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def request(value, tokenized=False):
    fields = {"schema", "model", "source_revision", "phase", "prompt_tokens",
              "cached_tokens", "request_batch_size", "prompt", "generation"}
    if tokenized:
        fields.add("prompt_token_ids")
    require(type(value) is dict and set(value) == fields, "exact reference request fields")
    require(value["schema"] == "zerogravity.qwen35-cuda-reference.request.v1"
            and value["model"] == MODEL and value["source_revision"] == REVISION,
            "original .8B model/revision")
    require(value["phase"] == "prefill" and type(value["cached_tokens"]) is int
            and value["cached_tokens"] == 0, "fresh prompt; cached decode is separate")
    require(type(value["prompt_tokens"]) is int and value["prompt_tokens"] in (47, 48, 49, 50)
            and type(value["request_batch_size"]) is int and value["request_batch_size"] == 1,
            "finite fresh47–50 single-request source profile")
    require(type(value["prompt"]) is str and 0 < len(value["prompt"].encode()) <= 16384,
            "bounded actual prompt text")
    generation = value["generation"]
    require(type(generation) is dict and set(generation) == set(GENERATION)
            and type(generation["max_new_tokens"]) is int
            and type(generation["eos_token_id"]) is int
            and type(generation["temperature"]) in (int, float)
            and generation == GENERATION, "original natural8 greedy/EOS generation")
    if tokenized:
        ids = value["prompt_token_ids"]
        require(type(ids) is list and len(ids) == value["prompt_tokens"]
                and all(type(token) is int and 0 <= token < 248320 for token in ids),
                "actual complete tokenizer-derived IDs")
    return value


def rendered(value):
    return "<|im_start|>user\n" + value["prompt"] + "<|im_end|>\n<|im_start|>assistant\n"
