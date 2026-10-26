#!/usr/bin/env python3
"""Developer-only extraction of a source-verified Qwen GGUF Q4_0 matvec fixture.

This is NOT a model execution backend, GPU kernel implementation, or runtime
dependency. It only packages exact GGUF source bytes and scalar CPU oracle values
for testing one Rust-authored CubeCL quantized projection across devices.
"""
import argparse
import hashlib
import pathlib
import struct
import sys


def checked_bytes(f, n):
    data = f.read(n)
    if len(data) != n:
        raise ValueError("truncated GGUF")
    return data


def u32(f):
    return struct.unpack("<I", checked_bytes(f, 4))[0]


def u64(f):
    return struct.unpack("<Q", checked_bytes(f, 8))[0]


def text(f):
    count = u64(f)
    if count > 8 * 1024 * 1024:
        raise ValueError("unbounded metadata string")
    return checked_bytes(f, count).decode("utf-8")


def skip_value(f, ty):
    sizes = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4,
             6: 4, 10: 8, 11: 8, 12: 8}
    if ty == 8:
        n = u64(f)
        if n > 8 * 1024 * 1024:
            raise ValueError("metadata string too long")
        f.seek(n, 1)
    elif ty == 9:
        elem = u32(f)
        n = u64(f)
        if n > 4_000_000:
            raise ValueError("array count too large")
        if elem in sizes:
            f.seek(n * sizes[elem], 1)
        else:
            for _ in range(n):
                skip_value(f, elem)
    elif ty in sizes:
        f.seek(sizes[ty], 1)
    else:
        raise ValueError(f"unknown GGUF metadata type {ty}")


def metadata(f):
    if checked_bytes(f, 4) != b"GGUF" or u32(f) != 3:
        raise ValueError("only GGUF v3 is qualified")
    tensor_count = u64(f)
    meta_count = u64(f)
    if tensor_count > 10000 or meta_count > 10000:
        raise ValueError("GGUF fixture count exceeds limit")
    align = 32
    for _ in range(meta_count):
        key = text(f)
        typ = u32(f)
        if key == "general.alignment":
            if typ not in (4, 10):
                raise ValueError("invalid alignment type")
            align = u32(f) if typ == 4 else u64(f)
        else:
            skip_value(f, typ)
    names = {}
    for _ in range(tensor_count):
        name = text(f)
        rank = u32(f)
        if rank not in (1, 2, 3, 4):
            raise ValueError("invalid tensor rank")
        dims = [u64(f) for _ in range(rank)]
        dtype = u32(f)
        offset = u64(f)
        if name in names:
            raise ValueError("duplicate tensor")
        names[name] = (dims, dtype, offset)
    start = (f.tell() + align - 1) // align * align
    return names, start


def read_floats(f, position, count, dtype):
    if dtype == 8:  # ggml Q8_0, 32 weights in a 34-byte block
        if count % 32:
            raise ValueError("Q8 dimension")
        f.seek(position)
        data = checked_bytes(f, (count // 32) * 34)
        values = []
        for i in range(0, len(data), 34):
            d = struct.unpack_from("<e", data, i)[0]
            values.extend(d * q for q in struct.unpack_from("<32b", data, i + 2))
        return values
    if dtype == 14:  # ggml Q6_K (256 weights, 210 bytes)
        if count % 256:
            raise ValueError("Q6_K dimension")
        f.seek(position)
        data = checked_bytes(f, (count // 256) * 210)
        values = []
        for b in range(0, len(data), 210):
            block = data[b:b + 210]
            d = struct.unpack_from("<e", block, 208)[0]
            sub = [0.0] * 256
            for half in range(2):
                low = block[half * 64:half * 64 + 64]
                high = block[128 + half * 32:160 + half * 32]
                scales = struct.unpack_from("<8b", block, 192 + half * 8)
                for lane in range(32):
                    for g in range(4):
                        nibble = ((low[lane + (g % 2) * 32] & 15)
                                  if g < 2 else
                                  (low[lane + (g % 2) * 32] >> 4))
                        q = (nibble | (((high[lane] >> (g * 2)) & 3) << 4)) - 32
                        sub[half * 128 + g * 32 + lane] = d * scales[g * 2 + lane // 16] * q
            values.extend(sub)
        return values
    raise ValueError(f"embedding quant type {dtype} not supported by fixture")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--token", type=int, default=0)
    a = parser.parse_args()
    if not 1 <= a.rows <= 32:
        raise ValueError("probe supports 1..32 projection rows")
    digest = hashlib.sha256()
    with a.source.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != a.sha256:
        raise ValueError("checkpoint SHA-256 mismatch: refusing fixture")
    with a.source.open("rb") as f:
        tensors, data_start = metadata(f)
        embedding = tensors["token_embd.weight"]
        projection = tensors["blk.0.attn_gate.weight"]
        shape, dtype, offset = projection
        if dtype != 2 or len(shape) != 2 or shape[0] not in (1024, 2048) or shape[1] < a.rows:
            raise ValueError(f"unqualified projection layout: {projection}")
        columns = shape[0]
        if columns % 32:
            raise ValueError("Q4_0 row misalignment")
        e_shape, e_type, e_offset = embedding
        if e_shape[0] != columns or a.token >= e_shape[1]:
            raise ValueError("embedding geometry or token ID mismatch")
        embedding_byte_row = {8: columns // 32 * 34,
                              14: columns // 256 * 210}[e_type]
        input_vec = read_floats(
            f, data_start + e_offset + a.token * embedding_byte_row,
            columns, e_type)
        blocks = columns // 32
        length = blocks * 18 * a.rows
        f.seek(data_start + offset)
        quantized = checked_bytes(f, length)
    scales = []
    nibble_bytes = []
    output = []
    for r in range(a.rows):
        accum = 0.0
        for block_index in range(blocks):
            i = (r * blocks + block_index) * 18
            scale = struct.unpack_from("<e", quantized, i)[0]
            packed = quantized[i + 2:i + 18]
            scales.append(scale)
            nibble_bytes.extend(packed)
            base = block_index * 32
            for j, packed_byte in enumerate(packed):
                accum += ((packed_byte & 15) - 8) * scale * input_vec[base + j]
                accum += ((packed_byte >> 4) - 8) * scale * input_vec[base + 16 + j]
        output.append(accum)
    # Header magic + logical column/row dimensions, then separately packed
    # Q4_0 scale/quant/source-input/oracle arrays. The original GGUF itself
    # remains ignored/outside Git and is never transmitted to an external peer.
    payload = b"ZQ4P" + struct.pack("<II", columns, a.rows)
    for values in (scales, input_vec, output):
        payload += struct.pack("<" + "f" * len(values), *values)
    # Packed 4-bit values are contiguous, arranged by output row, then blocks.
    payload += bytes(nibble_bytes)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(payload)
    print(f"SOURCE_SHA256={digest.hexdigest()}")
    print(f"Q4_PROBE={a.output} BYTES={len(payload)} COLS={columns} ROWS={a.rows}")
    print(f"CPU_ORACLE={output[:5]}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as e:
        print(f"fixture extraction refused: {e}", file=sys.stderr)
        raise SystemExit(2)
