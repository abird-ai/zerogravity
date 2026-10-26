"""Owned CPU-only container entrypoint, before any original model execution."""
import importlib.metadata
import hashlib
from pathlib import Path
import os
import stat

from reference_request import SOURCE_MANIFEST, read_json, rendered, request, require, sha


def stamp(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def held_hash(fd):
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def main():
    require(os.environ.get("NVIDIA_VISIBLE_DEVICES") == "void"
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and not list(Path("/dev").glob("nvidia*")) and not Path("/dev/kfd").exists()
            and not list(Path("/dev/dri").glob("render*"))
            and not list(Path("/dev/dri").glob("card*")),
            "tokenizer namespace has no GPU device nodes or visible devices")
    value = request(read_json("/capture/request.json"))
    source = Path("/source")
    manifest = source / "source-receipt.json"
    require(not manifest.is_symlink() and sha(manifest) == SOURCE_MANIFEST,
            "pinned original source receipt before tokenizer")
    document = read_json(manifest)
    require(document["repo"] == value["model"] and document["revision"] == value["source_revision"],
            "same original source revision")
    held = []
    try:
        receipt_fd = os.open(manifest, os.O_RDONLY | os.O_NOFOLLOW)
        receipt_before = os.fstat(receipt_fd)
        held.append((receipt_fd, manifest,
                     {"bytes": receipt_before.st_size, "sha256": SOURCE_MANIFEST}, receipt_before))
        require(held_hash(receipt_fd) == SOURCE_MANIFEST, "held original source receipt")
        require(len(document["files"]) == 11 and len({row["name"] for row in document["files"]}) == 11,
                "complete distinct original source files")
        for row in document["files"]:
            require(Path(row["name"]).name == row["name"], "source basename")
            path = source / row["name"]
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            held.append((fd, path, row, os.fstat(fd)))
            require(stat.S_ISREG(os.fstat(fd).st_mode)
                    and os.fstat(fd).st_size == row["bytes"] and held_hash(fd) == row["sha256"],
                    "complete original source bytes before tokenization")
        # Exact source tokenizer; original V1 tokenizer independently checks these IDs.
        from tokenizers import Tokenizer
        ids = Tokenizer.from_file(str(source / "tokenizer.json")).encode(
            rendered(value), add_special_tokens=False).ids
        value["prompt_token_ids"] = ids
        request(value, tokenized=True)
        for fd, path, row, before in held:
            after = os.fstat(fd)
            require(stamp(before) == stamp(after) == stamp(path.lstat())
                    and held_hash(fd) == row["sha256"] and stamp(before) == stamp(os.fstat(fd)),
                    "held original source unchanged through tokenization")
        require(sha(manifest) == SOURCE_MANIFEST, "source receipt unchanged")
        from support import write
        write("/output/tokenized-request.json", value)
        write("/output/tokenization.json", {
            "source_manifest_sha256": SOURCE_MANIFEST,
            "request": value, "tokenizers_version": importlib.metadata.version("tokenizers"),
            "source_files_verified": len(held), "gpu_device_mounted": False,
            "original_v1_tokenizer_check_pending": True})
    finally:
        for fd, _, _, _ in held:
            os.close(fd)


if __name__ == "__main__":
    main()
