//! Original accepted CPU-only oracle; read only after actual GPU producer.
use super::io::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::Arc,
};
use zerogravity::accelerator::program::{ConvDomain, RecurrentDomain, StateSlot};
use zerogravity::json_allocation::{self, ValueConstruction};
use zerogravity_gpu::{QwenSourceFrame, QwenSourceFrameKind, QwenSourceObserver};
fn string<'a>(value: &'a Value, key: &str) -> Proof<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("original reference string {key}").into())
}
const RAW_BYTES: u64 = 934_776_928;
/// Declared cap of the held manifest input (also the transient admission extent
/// charged by `natural_admission_preflight`). Never raised to fit an observation.
const MANIFEST_CAP: u64 = 16 * 1024 * 1024;
const RECEIPT_CAP: u64 = 1024 * 1024;
const RECORD_COUNT: usize = 1040;
const HELD_OWNERS: usize = 5;
const HELD_PATH_CAP: usize = 4096;
const HELD_SHA_CAP: usize = 64;
const DIAGNOSTIC_HOST_BYTES: u64 = 384 * 1024 * 1024;
/// Source-derived caps for the one owner/helper phase. The role/module strings
/// are generated from the frozen source table; their capacities are the exact
/// longest structural names, never observation or allocator guesses.
const SOURCE_PREFIX_CAP: usize = "model.language_model.layers.".len() + 2;
const SOURCE_ROLE_SUFFIX_CAP: usize = ".post_attention_layernorm.weight".len();
const SOURCE_ROLE_TOP_CAP: usize = "model.language_model.embed_tokens.weight".len();
const SOURCE_ROLE_CAP: usize = if SOURCE_PREFIX_CAP + SOURCE_ROLE_SUFFIX_CAP > SOURCE_ROLE_TOP_CAP {
    SOURCE_PREFIX_CAP + SOURCE_ROLE_SUFFIX_CAP
} else {
    SOURCE_ROLE_TOP_CAP
};
const MODULE_STRING_CAP: usize = "embedding".len();
const _: () = assert!(SOURCE_ROLE_CAP == 62);
const _: () = assert!(MODULE_STRING_CAP == 9);
const MODULE_COUNT: usize = 67;
const SHAPE_VALUES: usize = 4;
const VOCAB: u64 = 248_320;
/// Largest actual raw state record: one 48x128x128 F32 recurrent state.
/// Derived from the source descriptor shape, not a budget slack constant.
const MAX_RAW_RECORD_BYTES: u64 = 48 * 128 * 128 * 4;
const _: () = assert!(MAX_RAW_RECORD_BYTES >= VOCAB * 2);
const NATURAL_ARGV_MAX: usize = 72;
fn checked_bytes(count: usize, each: usize, why: &str) -> Proof<u64> {
    (count as u64)
        .checked_mul(each as u64)
        .ok_or_else(|| format!("{why} extent overflow").into())
}
/// Owner/helper bound charged before any accepted pin path join or digest copy.
/// It covers the five fixed held owners, both generated role/module vectors, the
/// exact natural argv string vector and the one record-shape Value built by each
/// validation pass. Held path/digest capacities are fixed by `HELD_*_CAP`.
fn owner_metadata_extent() -> Proof<u64> {
    checked_bytes(
        HELD_OWNERS,
        std::mem::size_of::<Held>() + HELD_PATH_CAP + HELD_SHA_CAP,
        "held owner layout",
    )
}
fn owner_helper_phase_bound() -> Proof<u64> {
    let owners = owner_metadata_extent()?;
    let roles = checked_bytes(
        SOURCE_ROLES,
        std::mem::size_of::<String>() + SOURCE_ROLE_CAP,
        "source role helper",
    )?;
    let prefixes = checked_bytes(
        64,
        std::mem::size_of::<String>() + SOURCE_PREFIX_CAP,
        "source role prefix helper",
    )?;
    let modules = checked_bytes(
        MODULE_COUNT,
        std::mem::size_of::<String>() + MODULE_STRING_CAP,
        "module order helper",
    )?;
    let argv = checked_bytes(
        NATURAL_ARGV_MAX,
        std::mem::size_of::<String>() + HELD_PATH_CAP,
        "natural argv helper",
    )?;
    let argv_refs = checked_bytes(
        NATURAL_ARGV_MAX,
        std::mem::size_of::<&str>(),
        "natural argv reference helper",
    )?;
    let order_refs = checked_bytes(
        MODULE_COUNT,
        std::mem::size_of::<&str>(),
        "module order reference helper",
    )?;
    let fullv = checked_bytes(
        2,
        std::mem::size_of::<String>() + HELD_SHA_CAP,
        "trajectory fullV digest helper",
    )?;
    let cpu_env = checked_bytes(
        4,
        std::mem::size_of::<String>() + 16,
        "cpu environment helper",
    )?;
    let cpu_vec = checked_bytes(4, std::mem::size_of::<u64>(), "cpu cohort helper")?;
    let shapes = checked_bytes(
        RECORD_COUNT,
        SHAPE_VALUES * (std::mem::size_of::<Value>() + std::mem::size_of::<usize>()),
        "record shape helper",
    )?;
    owners
        .checked_add(roles)
        .and_then(|n| n.checked_add(prefixes))
        .and_then(|n| n.checked_add(modules))
        .and_then(|n| n.checked_add(argv))
        .and_then(|n| n.checked_add(argv_refs))
        .and_then(|n| n.checked_add(order_refs))
        .and_then(|n| n.checked_add(fullv))
        .and_then(|n| n.checked_add(cpu_env))
        .and_then(|n| n.checked_add(cpu_vec))
        .and_then(|n| n.checked_add(shapes))
        .ok_or_else(|| Box::<dyn std::error::Error>::from("owner/helper phase bound"))
}
const TEACHER_ACCEPTED_BYTES: u64 = 13_214;
const TEACHER_ACCEPTED_SHA: &str =
    "2a81216ede102c8b0c22c082332f62f38bf0bb2f326298e519f34ed20fc3e3d7";
const NATURAL_ACCEPTED_FILE: &str = "accepted-original-natural-full-window-1.json";
const SNAPSHOT: &str = "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e";
const INITIAL_PRIOR_PLANES: usize = 96;
const CONV_WORDS: usize = 10240 * 4;
const RECURRENT_WORDS: usize = 48 * 128 * 128;
const INITIAL_PRIOR_WORDS: usize = 48 * (CONV_WORDS + RECURRENT_WORDS);
const SOURCE_ROLES: usize = 851;
const SOURCE_FDS: usize = 18;
const SOURCE_EVENTS: usize = 268;
const SOURCE_PARAMETER_BYTES: u64 = 53_791_996_928;
const SOURCE_MAPPED_BYTES: u64 = 53_793_594_360;
const MAXIMUM_MAPPED_BYTES: u64 = 2_542_798_560;
const TEACHER_REPOSITORY: &str = "Qwen/Qwen3.8-27B";
const TEACHER_REVISION: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
const TEACHER_TORCH: &str = "2.13.0+cu129";
const TEACHER_TORCH_GIT: &str = "cf30153c4c131c8164ee7798e5022d810682e2cb";
const TEACHER_CACHE_UTILS_SHA: &str =
    "702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c";
const TEACHER_CONFIG_SHA: &str = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab";
const TEACHER_INDEX_SHA: &str = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df";
// Complete accepted original source/config/tokenizer pins (bytes, SHA256).
const SOURCE_FILES: [(&str, u64, &str); 31] = [
    (
        "LICENSE",
        11_544,
        "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
    ),
    (
        "README.md",
        65_012,
        "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641",
    ),
    (
        "chat_template.jinja",
        8_952,
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    ),
    (
        "config.json",
        4_312,
        "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
    ),
    (
        "crc32.txt",
        238,
        "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83",
    ),
    (
        "generation_config.json",
        202,
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    ),
    (
        "merges.txt",
        3_353_259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    (
        "model-00001-of-00018.safetensors",
        3_966_730_552,
        "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c",
    ),
    (
        "model-00002-of-00018.safetensors",
        3_043_080_328,
        "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb",
    ),
    (
        "model-00003-of-00018.safetensors",
        2_542_796_952,
        "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968",
    ),
    (
        "model-00004-of-00018.safetensors",
        3_988_973_152,
        "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a",
    ),
    (
        "model-00005-of-00018.safetensors",
        2_099_339_864,
        "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701",
    ),
    (
        "model-00006-of-00018.safetensors",
        3_979_553_696,
        "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0",
    ),
    (
        "model-00007-of-00018.safetensors",
        2_108_759_344,
        "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892",
    ),
    (
        "model-00008-of-00018.safetensors",
        3_979_553_696,
        "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79",
    ),
    (
        "model-00009-of-00018.safetensors",
        2_108_759_344,
        "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8",
    ),
    (
        "model-00010-of-00018.safetensors",
        3_979_553_696,
        "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9",
    ),
    (
        "model-00011-of-00018.safetensors",
        2_108_759_344,
        "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8",
    ),
    (
        "model-00012-of-00018.safetensors",
        3_979_553_696,
        "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55",
    ),
    (
        "model-00013-of-00018.safetensors",
        2_108_759_344,
        "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a",
    ),
    (
        "model-00014-of-00018.safetensors",
        3_979_553_696,
        "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b",
    ),
    (
        "model-00015-of-00018.safetensors",
        2_108_759_344,
        "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22",
    ),
    (
        "model-00016-of-00018.safetensors",
        3_979_564_040,
        "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe",
    ),
    (
        "model-00017-of-00018.safetensors",
        2_108_759_344,
        "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8",
    ),
    (
        "model-00018-of-00018.safetensors",
        3_392_197_344,
        "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06",
    ),
    (
        "model.safetensors.index.json",
        112_216,
        "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
    ),
    (
        "preprocessor_config.json",
        390,
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    ),
    (
        "tokenizer.json",
        12_809_320,
        "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    ),
    (
        "tokenizer_config.json",
        17_928,
        "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    ),
    (
        "video_preprocessor_config.json",
        385,
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    ),
    (
        "vocab.json",
        6_722_759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    ),
];
// Original read-only reference package (declared pins; verifier rehashes files).
const REFERENCE_PACKAGE: [(&str, u64, &str); 9] = [
    (
        "archive.py",
        2_312,
        "7e66bce11e67aa93da8a9f9ccab669398e702e31770f8ca819ccaeef21aa7c89",
    ),
    (
        "capture.py",
        8_657,
        "9bfe8722f2f6c9797d2b6e7f24d2dfe9d83177b9e1f1b69295845801119dfcb3",
    ),
    (
        "contract.py",
        12_712,
        "6cdcba6a57fbb47c580ddeecf1bae26ae5f6338bb3f6fb155fa5ca97e81e9b0f",
    ),
    (
        "inputs.json",
        1_374,
        "6b7485a4670fe24d681884ace8d61f565cf110ab6e9fac2d55e06542219e90a9",
    ),
    (
        "owned_container_v4.py",
        11_840,
        "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d",
    ),
    (
        "run_reference.py",
        7_178,
        "4a2926b0d8cd3b605a97bbcf33889c7ce91fc22be9dc53c993eb3516e2cc106c",
    ),
    (
        "source-pin.json",
        7_058,
        "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a",
    ),
    (
        "source_owner.py",
        7_614,
        "3e07c5d1b6b9f1182ed985757245826ee187b61b96ddc20bac36f913b0fabdce",
    ),
    (
        "validate_reference.py",
        13_876,
        "75f2d4a62380da383f54fb8cf3dbf30850c439bf0b486d441e4438b71f3da298",
    ),
];
// Fresh derivative natural reference package: same 9 tool roles, new bytes.
// Declared pins; the runtime rehashes the held files and never trusts names.
const NATURAL_PACKAGE: [(&str, u64, &str); 9] = [
    (
        "archive.py",
        2312,
        "7e66bce11e67aa93da8a9f9ccab669398e702e31770f8ca819ccaeef21aa7c89",
    ),
    (
        "capture.py",
        10815,
        "2c0e0692a51383c2527fe994a0e8a05a8b490e8d43988effcc815264885ddfd3",
    ),
    (
        "contract.py",
        13923,
        "882abbd934966f4f0aadd4d8ec24953b19efbc8a135e3392ccf2c9e71034f8eb",
    ),
    (
        "inputs.json",
        1444,
        "59064199956314944d1434ffaf845b4cc38a5357398388091573f9ff0a348ec9",
    ),
    (
        "owned_container_v4.py",
        11840,
        "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d",
    ),
    (
        "run_reference.py",
        13312,
        "54641525b9525982e9682bb82c3dc483724357ccf9f53176741b3167b847d5ca",
    ),
    (
        "source-pin.json",
        7058,
        "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a",
    ),
    (
        "source_owner.py",
        7614,
        "3e07c5d1b6b9f1182ed985757245826ee187b61b96ddc20bac36f913b0fabdce",
    ),
    (
        "validate_reference.py",
        18197,
        "2574c7fe4eaf5d472db4694b0127bf47d7a9f3ce69bbe7edf7a3b4e41b176c15",
    ),
];
// Original immutable reference runtime (unchanged by this derivative).
const IMAGE: &str = "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe";
const IMAGE_DIGEST: &str =
    "sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634";
const NATURAL_OWNED_NAME: &str = "zg-original27-natural-full-window-";
const PODMAN_ROOT: &str =
    "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers";
const PODMAN_RUNROOT: &str = "/run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers";
const OWNER_LABEL: &str = "abird.zerogravity.vllm-gpu-oracle-owner";
const SOURCE_DIRECTORY: &str = "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files";
const NATURAL_TMPFS: &str = "/tmp:rw,nosuid,nodev,size=1g";
/// Fixed `-e K=V` cohort the original worker requires, in emit order.
const NATURAL_ENV: [(&str, &str); 16] = [
    ("CUDA_VISIBLE_DEVICES", "-1"),
    ("NVIDIA_VISIBLE_DEVICES", "void"),
    ("ATEN_CPU_CAPABILITY", "avx2"),
    ("USE_HUB_KERNELS", "0"),
    ("HF_HUB_OFFLINE", "1"),
    ("TRANSFORMERS_OFFLINE", "1"),
    ("OMP_NUM_THREADS", "4"),
    ("MKL_NUM_THREADS", "4"),
    ("OPENBLAS_NUM_THREADS", "4"),
    ("OMP_DYNAMIC", "FALSE"),
    ("MKL_DYNAMIC", "FALSE"),
    ("PYTHONPATH", "/probe"),
    ("PYTHONDONTWRITEBYTECODE", "1"),
    ("LD_PRELOAD", ""),
    ("LD_AUDIT", ""),
    ("VLLM_PLUGINS", ""),
];
/// Dynamic env slots are compared by shape, not against an invented constant.
const ARGV_CPUS: &str = "@cpus";
const ARGV_PACKAGE: &str = "@package";
const ARGV_PROBE: &str = "@probe";
const NATURAL_FSIZE: &str = "fsize=1342177280:1342177280";
/// Full exact original argv arity: 12 fixed/name-label-cidfile fields, 14
/// resource/namespace fields, five fixed env pairs, the remaining env pairs and
/// the ten source/entrypoint fields. It is checked before every indexed access.
const NATURAL_ARGV_LEN: usize = 12 + 14 + 10 + 2 * (NATURAL_ENV.len() - 3) + 10;
const _: () = assert!(NATURAL_ARGV_MAX == NATURAL_ARGV_LEN);
/// Finite exact derivative argv: original podman root/runroot, one `run` with
/// the owner's actual separate option/value syntax, the fixed worker env cohort,
/// the three source mounts and the original entrypoint. Duplicates, unknown or
/// reordered options, a different output/cidfile and a stale image all refuse.
fn natural_argv(args: &[&str], cpus: &[u64], nonce: &str) -> Proof<()> {
    need(
        args.len() == NATURAL_ARGV_LEN,
        format!(
            "exact original natural argv arity {} = {NATURAL_ARGV_LEN}",
            args.len()
        ),
    )?;
    need(
        args.first() == Some(&"podman")
            && args[1] == "--root"
            && args[2] == PODMAN_ROOT
            && args[3] == "--runroot"
            && args[4] == PODMAN_RUNROOT
            && args[5] == "run",
        "original podman container root/runroot/run",
    )?;
    need(
        args.iter().all(|arg| arg.len() <= HELD_PATH_CAP),
        "bounded natural argv element before expected vector construction",
    )?;
    // Capture output is the directory that owns the actual cidfile/mount pair.
    let cidfile = Path::new(args[11]);
    need(
        cidfile.is_absolute() && cidfile.file_name().and_then(|n| n.to_str()) == Some("owned.cid"),
        "exact capture cidfile basename",
    )?;
    let output = cidfile.parent().ok_or("capture output parent")?;
    need(
        output
            .as_os_str()
            .len()
            .checked_add("/output:rw".len())
            .is_some_and(|n| n <= HELD_PATH_CAP),
        "exact capture output mount within the held path capacity",
    )?;
    let cpu_env = cpus
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let probe = format!("{}:/probe:ro", ARGV_PROBE);
    let mut expected = Vec::with_capacity(NATURAL_ARGV_LEN);
    expected.extend([
        "podman".to_string(),
        "--root".to_string(),
        PODMAN_ROOT.to_string(),
        "--runroot".to_string(),
        PODMAN_RUNROOT.to_string(),
        "run".to_string(),
        "--name".to_string(),
        format!("{NATURAL_OWNED_NAME}{nonce}"),
        "--label".to_string(),
        format!("{OWNER_LABEL}={nonce}"),
        "--cidfile".to_string(),
        output.join("owned.cid").display().to_string(),
        "--rm".to_string(),
        "--pull=never".to_string(),
        "--network=none".to_string(),
        "--read-only".to_string(),
        "--cap-drop=all".to_string(),
        "--security-opt=no-new-privileges".to_string(),
        "--memory=12g".to_string(),
        "--memory-swap=12g".to_string(),
        "--cpus=4".to_string(),
        "--pids-limit=256".to_string(),
        "--ulimit".to_string(),
        NATURAL_FSIZE.to_string(),
        "--tmpfs".to_string(),
        NATURAL_TMPFS.to_string(),
        "-e".to_string(),
        "CUDA_VISIBLE_DEVICES=-1".to_string(),
        "-e".to_string(),
        "NVIDIA_VISIBLE_DEVICES=void".to_string(),
        "-e".to_string(),
        "ATEN_CPU_CAPABILITY=avx2".to_string(),
        "-e".to_string(),
        format!("ZG_ALLOWED_CPUS={ARGV_CPUS}"),
        "-e".to_string(),
        format!("ZG_PACKAGE_SHA={ARGV_PACKAGE}"),
    ]);
    for (name, value) in NATURAL_ENV.iter().skip(3) {
        expected.push("-e".to_string());
        expected.push(format!("{name}={value}"));
    }
    need(
        cpu_env.split(',').count() == 4,
        "four actual allowed capture cpus",
    )?;
    expected.extend([
        "-v".to_string(),
        probe,
        "-v".to_string(),
        format!("{SOURCE_DIRECTORY}:/source:ro"),
        "-v".to_string(),
        format!("{}:/output:rw", output.display()),
        "--entrypoint".to_string(),
        "python3".to_string(),
        IMAGE.to_string(),
        "/probe/capture.py".to_string(),
    ]);
    need(
        expected.len() == NATURAL_ARGV_LEN && expected.capacity() == NATURAL_ARGV_LEN,
        "derived natural argv remains at the exact arity/capacity",
    )?;
    for (index, (actual, want)) in args.iter().zip(&expected).enumerate() {
        if want == &format!("ZG_ALLOWED_CPUS={ARGV_CPUS}") {
            need(
                actual.strip_prefix("ZG_ALLOWED_CPUS=") == Some(cpu_env.as_str()),
                "argv allowed CPU cohort equals the held capture cpus",
            )?;
            continue;
        }
        if want == &format!("ZG_PACKAGE_SHA={ARGV_PACKAGE}") {
            let value = actual
                .strip_prefix("ZG_PACKAGE_SHA=")
                .ok_or("argv worker package digest")?;
            need(
                value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()),
                "argv worker package digest shape",
            )?;
            continue;
        }
        if want.ends_with(":/probe:ro") {
            let source = actual
                .strip_suffix(":/probe:ro")
                .ok_or("argv probe mount")?;
            need(
                Path::new(source).is_absolute()
                    && Path::new(source).file_name().and_then(|n| n.to_str())
                        == Some("qwen27b-natural-full-window-reference"),
                "argv probe mount is the frozen natural tool package",
            )?;
            continue;
        }
        need(
            actual == want,
            format!("exact original argv element {index}: {actual} != {want}"),
        )?;
    }
    Ok(())
}
/// Actual original invocation/image join for the fresh natural corpus: the two
/// held files are byte-exact copies of the capture output, and every declared
/// runtime/resource/ownership field is revalidated, not merely SHA-shaped.
fn validate_natural_corpus(
    accepted: &Value,
    artifacts: Option<&(Held, Held)>,
    admission: &mut Admission,
) -> Proof<Vec<u64>> {
    need(
        accepted["image_id"] == IMAGE && accepted["image_digest"] == IMAGE_DIGEST,
        "accepted natural receipt binds the original immutable runtime",
    )?;
    // Both retained owners were opened once by `open_natural`; the same two
    // descriptors are censused, parsed and leased, never reopened by path.
    let (invocation, inspected) = artifacts.ok_or("two retained natural artifact owners")?;
    let (inv, invocation_tree) = held_json(invocation, admission, "natural invocation")?;
    need(
        inv["source_before"]
            .as_array()
            .is_some_and(|rows| rows.len() == SOURCE_FILES.len())
            && inv["source_before"] == accepted["source_files"]
            && inv["source_after"] == inv["source_before"],
        "actual invocation source-before/after joins the accepted source rows",
    )?;
    need(
        inv["status"] == "EXECUTED_ORIGINAL27B_NATURAL_FULL_WINDOW_REQUIRES_VALIDATION"
            && inv["exit_code"].as_u64() == Some(0)
            && inv["cleanup_errors"] == json!([])
            && inv["source_FD_cleanup_errors"] == json!([])
            && inv["source_FDs_closed"] == true
            && inv["source_before"] == inv["source_after"]
            && inv["owned_container_absent_after_reap"] == true
            && inv["image_id"] == IMAGE
            && inv["image_digest"] == IMAGE_DIGEST
            && inv["package"] == accepted["package"]
            && inv["scope"] == accepted["scope"],
        "actual natural capture invocation status/package/source closure",
    )?;
    let cleanup = &inv["foreground_cleanup"];
    need(
        cleanup["leader_reaped"] == true
            && cleanup["group_absent"] == true
            && cleanup["all_nonzero_signals_before_reap"] == true
            && cleanup["term_sent"] == true
            && cleanup["kill_sent_before_reap"] == true
            && cleanup["kill_attempts_before_reap"].as_u64() == Some(2)
            && cleanup["leader_reap_attempts"].as_u64() == Some(1)
            && cleanup["stage_errors"] == json!([])
            && inv["log_cleanup"]["pipe_eof"] == true
            && inv["log_cleanup"]["all_closed"] == true,
        "one TERM, two KILL before one Reap, group absence, full EOF",
    )?;
    let nonce = string(&inv, "owned_nonce")?;
    need(
        nonce.len() == 32
            && nonce.bytes().all(|b| b.is_ascii_hexdigit())
            && string(&inv, "owned_name")? == format!("{NATURAL_OWNED_NAME}{nonce}"),
        "actual owned natural container name/nonce",
    )?;
    let cpus_json = inv["expected_cpus"]
        .as_array()
        .ok_or("actual original cpus")?;
    need(cpus_json.len() == 4, "exact four original capture cpus")?;
    let mut cpus = Vec::with_capacity(4);
    for value in cpus_json {
        cpus.push(
            value
                .as_u64()
                .filter(|c| *c < 4096)
                .ok_or("bounded original cpus")?,
        );
    }
    for i in 0..cpus.len() {
        for j in (i + 1)..cpus.len() {
            need(cpus[i] != cpus[j], "distinct original capture cpus")?;
        }
    }
    let argv = inv["argv"]
        .as_array()
        .ok_or("actual natural invocation argv")?;
    need(
        argv.len() == NATURAL_ARGV_LEN,
        "exact original natural argv arity before indexed access",
    )?;
    let mut args = Vec::with_capacity(argv.len());
    for arg in argv {
        let text = arg
            .as_str()
            .ok_or("every actual natural argv element is a string")?;
        need(
            text.len() <= HELD_PATH_CAP,
            "every actual natural argv element is a bounded string",
        )?;
        args.push(text);
    }
    natural_argv(&args, &cpus, nonce)?;
    let (image, image_tree) = held_json(inspected, admission, "natural image inspect")?;
    let rows = image.as_array().ok_or("actual image inspect list")?;
    need(
        rows.len() == 1
            && rows[0]["Id"]
                == IMAGE
                    .strip_prefix("sha256:")
                    .ok_or("original image id form")?
            && rows[0]["Digest"] == IMAGE_DIGEST,
        "same immutable original image Id/Digest from actual inspect",
    )?;
    drop(inv);
    drop(image);
    admission.release(invocation_tree, "natural invocation tree")?;
    admission.release(image_tree, "natural image inspect tree")?;
    Ok(cpus)
}
fn role_name(prefix: &str, name: &str) -> Proof<String> {
    let dot = usize::from(!prefix.is_empty());
    let length = prefix
        .len()
        .checked_add(dot)
        .and_then(|n| n.checked_add(name.len()))
        .ok_or("source role length overflow")?;
    need(
        length <= SOURCE_ROLE_CAP,
        "source role length inside its derived capacity before append",
    )?;
    let mut out = String::with_capacity(SOURCE_ROLE_CAP);
    out.push_str(prefix);
    if dot == 1 {
        out.push('.');
    }
    out.push_str(name);
    need(
        out.len() == length && out.capacity() <= SOURCE_ROLE_CAP,
        "source role string inside its derived capacity",
    )?;
    Ok(out)
}
fn source_roles() -> Proof<Vec<String>> {
    let mut out = Vec::with_capacity(SOURCE_ROLES);
    out.push(role_name("", "lm_head.weight")?);
    out.push(role_name("", "model.language_model.embed_tokens.weight")?);
    for layer in 0..64 {
        let mut prefix = String::with_capacity(SOURCE_PREFIX_CAP);
        prefix.push_str("model.language_model.layers.");
        prefix.push_str(&layer.to_string());
        need(
            prefix.len() <= SOURCE_PREFIX_CAP && prefix.capacity() <= SOURCE_PREFIX_CAP,
            "source layer prefix inside its derived capacity",
        )?;
        out.push(role_name(&prefix, "input_layernorm.weight")?);
        let attention: &[&str] = if layer % 4 == 3 {
            &[
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ]
        } else {
            &[
                "linear_attn.A_log",
                "linear_attn.conv1d.weight",
                "linear_attn.dt_bias",
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.norm.weight",
                "linear_attn.out_proj.weight",
            ]
        };
        for name in attention {
            out.push(role_name(&prefix, name)?);
        }
        for name in [
            "mlp.down_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "post_attention_layernorm.weight",
        ] {
            out.push(role_name(&prefix, name)?);
        }
    }
    out.push(role_name("", "model.language_model.norm.weight")?);
    need(
        out.len() == SOURCE_ROLES && out.capacity() == SOURCE_ROLES,
        "exact source role vector capacity",
    )?;
    out.sort_unstable();
    need(
        out.windows(2).all(|pair| pair[0] != pair[1]),
        "unique source role table",
    )?;
    Ok(out)
}
fn module_name(layer: usize) -> Proof<String> {
    let mut out = String::with_capacity(MODULE_STRING_CAP);
    out.push_str("layer.");
    out.push_str(&layer.to_string());
    need(
        out.len() <= MODULE_STRING_CAP && out.capacity() <= MODULE_STRING_CAP,
        "module name inside its derived capacity",
    )?;
    Ok(out)
}
fn module_order() -> Proof<Vec<String>> {
    let mut out = Vec::with_capacity(MODULE_COUNT);
    out.push("embedding".to_string());
    for layer in 0..64 {
        out.push(module_name(layer)?);
    }
    out.push("final".to_string());
    out.push("head".to_string());
    need(
        out.len() == MODULE_COUNT && out.capacity() == MODULE_COUNT,
        "exact module order vector capacity",
    )?;
    Ok(out)
}
fn validate_source(
    accepted: &Value,
    manifest: &Value,
    fresh: bool,
    artifacts: Option<&(Held, Held)>,
    admission: &mut Admission,
) -> Proof<()> {
    let teacher_receipt = accepted["schema"] == "zg.actual-original27b-full-window-reference.v1"
        && accepted["status"] == "ACTUAL_ORIGINAL27B_FULL_WINDOW_REFERENCE_ONLY";
    let fresh_receipt = accepted["schema"]
        == "zg.actual-original27b-full-window-natural-reference.v1"
        && accepted["status"] == "ACTUAL_ORIGINAL27B_FULL_WINDOW_NATURAL_REFERENCE_ONLY"
        && accepted["independent_original_natural_trajectory"] == true
        && accepted["scope"]["natural_feedback_qualified"] == true
        && accepted["scope"]["teacher_forcing_only"] == false;
    need(
        if fresh {
            fresh_receipt
        } else {
            teacher_receipt
        },
        "original CPU-only accepted receipt kind",
    )?;
    let scope = &accepted["scope"];
    need(
        scope.as_object().is_some_and(|m| m.len() == 10)
            && scope["full_hf_conv4"] == true
            && scope["original_full_text_forward"] == true
            && scope["all_851_text_source_roles"] == true
            && scope["mixed_source_parameters_preserved"] == true
            && scope["natural_feedback_qualified"] == fresh
            && scope["teacher_forcing_only"] == !fresh
            && scope["all_bf16_loader_equivalence_qualified"] == false
            && scope["mkl_exp_producer_qualified"] == false
            && scope["rust_numerical_parity_qualified"] == false
            && scope["state_root_native_serving_performance_qualified"] == false,
        "complete original source/feedback scope",
    )?;
    need(
        accepted["layers"] == 64
            && accepted["records"] == 1040
            && accepted["text_roles"] == SOURCE_ROLES
            && accepted["cache_domains"] == 112
            && accepted["full_hf_conv_width"] == 4
            && accepted["original_repetitions"] == 2
            && accepted["raw_bytes"] == RAW_BYTES
            && accepted["snapshot"] == SNAPSHOT
            && accepted["repository"] == TEACHER_REPOSITORY
            && accepted["revision"] == TEACHER_REVISION
            && accepted["parameter_mode"] == "raw-source-assign-no-cast"
            && accepted["config_sha256"] == TEACHER_CONFIG_SHA
            && accepted["index_sha256"] == TEACHER_INDEX_SHA
            && accepted["original_source_and_owned_lifetime_closed"] == true
            && accepted["mkl_exp_producer_qualified"] == false
            && accepted["rust_numerical_parity_qualified"] == false
            && accepted["state_root_native_serving_performance_qualified"] == false,
        "original accepted receipt source/record/parity identity",
    )?;
    let files = accepted["source_files"]
        .as_array()
        .ok_or("original source file list")?;
    need(
        files.len() == SOURCE_FILES.len(),
        "all31 original source files",
    )?;
    for (name, bytes, sha) in SOURCE_FILES {
        let mut rows = files.iter().filter(|file| file["name"] == name);
        let row = rows
            .next()
            .ok_or_else(|| format!("original source pin {name}"))?;
        need(
            rows.next().is_none()
                && row["bytes"].as_u64() == Some(bytes)
                && row["sha256"] == sha
                && row["fd_identity"]
                    .as_array()
                    .is_some_and(|id| id.len() == 5),
            format!("exact original source/config/tokenizer pin {name}"),
        )?;
    }
    let package = accepted["package"]
        .as_object()
        .ok_or("original reference package")?;
    // Separate authority: the fresh natural package must carry its own 9 pins,
    // not the accepted teacher-forcing package identity.
    let table = if fresh {
        need(
            NATURAL_PACKAGE != REFERENCE_PACKAGE,
            "fresh derivative natural tool package identity",
        )?;
        &NATURAL_PACKAGE[..]
    } else {
        &REFERENCE_PACKAGE[..]
    };
    need(
        package.len() == table.len(),
        "all9 held reference tool pins",
    )?;
    for (name, bytes, sha) in table {
        let pin = package
            .get(*name)
            .ok_or_else(|| format!("reference tool pin {name}"))?;
        need(
            pin["bytes"].as_u64() == Some(*bytes) && pin["sha256"] == *sha,
            format!("exact reference tool pin {name}"),
        )?;
    }
    // Fresh affinity is the actual captured CPU cohort, not a fixed 0..3 claim.
    let affinity = if fresh {
        validate_natural_corpus(accepted, artifacts, admission)?
    } else {
        vec![0, 1, 2, 3]
    };
    for key in ["manifest", "raw"] {
        let pin = &accepted[key];
        let path = Path::new(string(pin, "path")?);
        need(
            path.is_absolute() && path.file_name().and_then(|n| n.to_str()).is_some(),
            format!("declared original {key} basename"),
        )?;
    }
    // The fresh producer and validator both name this exact schema.
    let schema = if fresh {
        "zg.original27b-natural-full-window-capture.v1"
    } else {
        "zg.original27b-full-window-capture.v1"
    };
    need(
        manifest["schema"] == schema
            && manifest["snapshot"] == SNAPSHOT
            && manifest["scope"] == accepted["scope"]
            && manifest["repository"] == TEACHER_REPOSITORY
            && manifest["revision"] == TEACHER_REVISION
            && manifest["parameter_mode"] == "raw-source-assign-no-cast"
            && manifest["torch"] == TEACHER_TORCH
            && manifest["torch_git"] == TEACHER_TORCH_GIT
            && manifest["cache_utils_sha256"] == TEACHER_CACHE_UTILS_SHA
            && manifest["threads"].as_u64() == Some(4)
            && manifest["interop_threads"].as_u64() == Some(1)
            && manifest["affinity"].as_array().is_some_and(|a| {
                a.len() == affinity.len()
                    && a.iter()
                        .zip(&affinity)
                        .all(|(c, want)| c.as_u64() == Some(*want))
            })
            && manifest["original_full_forward"] == true
            && manifest["cache_domains"] == 112
            && manifest["full_hf_conv_width"] == 4
            && manifest["all_851_source_parameters_raw"] == true
            && manifest["all_original_source_and_mapped_bytes_unchanged"] == true
            && manifest["source_owner_retained_until_process_exit"] == true
            && manifest["maximum_mapped_bytes"].as_u64() == Some(MAXIMUM_MAPPED_BYTES),
        "actual original source/cache/tool/manifest identity",
    )?;
    let roles = source_roles()?;
    need(roles.len() == SOURCE_ROLES, "exact851 original role table")?;
    for key in ["source_views", "source_readsets", "source_counts"] {
        let map = manifest[key]
            .as_object()
            .ok_or_else(|| format!("original {key}"))?;
        need(
            map.len() == SOURCE_ROLES && roles.iter().all(|role| map.contains_key(role)),
            format!("all851 exact original {key} roles"),
        )?;
    }
    let (mut parameter_bytes, mut mapped_bytes) = (0u64, 0u64);
    for (name, view) in manifest["source_views"]
        .as_object()
        .ok_or("original source views")?
    {
        let bytes = view["bytes"].as_u64().ok_or("source view bytes")?;
        let mapped = view["mapping_bytes"]
            .as_u64()
            .ok_or("source view mapping")?;
        let elements = view["shape"]
            .as_array()
            .ok_or("source view shape")?
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(d.as_u64()?))
            .ok_or("source view element count")?;
        need(
            view["dtype"] == "BF16"
                && elements > 0
                && bytes == 2 * elements
                && mapped >= bytes
                && view["absolute"].as_u64().is_some()
                && view["view_offset"].as_u64().is_some(),
            format!("raw BF16 source role {name}"),
        )?;
        parameter_bytes += bytes;
        mapped_bytes += mapped;
    }
    need(
        parameter_bytes == SOURCE_PARAMETER_BYTES && mapped_bytes == SOURCE_MAPPED_BYTES,
        "complete851 original source bytes",
    )?;
    need(
        manifest["source_counts"]
            .as_object()
            .is_some_and(|m| m.values().all(|v| v.as_u64() == Some(4))),
        "each original source role bound in both repeats/calls",
    )?;
    let fds = manifest["source_FDs"]
        .as_object()
        .ok_or("original held source FDs")?;
    need(fds.len() == SOURCE_FDS, "all18 original held source FDs")?;
    for (name, bytes, _) in SOURCE_FILES.iter().filter(|f| f.0.starts_with("model-")) {
        let fd = fds
            .get(*name)
            .ok_or_else(|| format!("held source FD {name}"))?
            .as_array()
            .ok_or("held FD identity")?;
        need(
            fd.len() == 5 && fd[2].as_u64() == Some(*bytes),
            format!("held source FD extent {name}"),
        )?;
    }
    let events = manifest["source_events"]
        .as_array()
        .ok_or("original source events")?;
    need(
        events.len() == SOURCE_EVENTS,
        "all268 original per-call source events",
    )?;
    let order = module_order()?;
    let mut at = 0usize;
    for repeat in 0..2u64 {
        for call in 0..2u64 {
            let mut bound = 0u64;
            for group in &order {
                let event = events.get(at).ok_or("original source event absent")?;
                at += 1;
                need(
                    event["repeat"].as_u64() == Some(repeat)
                        && event["call"].as_u64() == Some(call)
                        && event["position"].as_u64() == Some(call)
                        && event["group"] == group.as_str()
                        && event["source_bound"] == true,
                    format!("original source event order repeat{repeat}/call{call}/{group}"),
                )?;
                let roles = event["roles"].as_u64().ok_or("source event roles")?;
                let mapped = event["mapping_bytes"]
                    .as_u64()
                    .ok_or("source event bytes")?;
                need(
                    roles > 0 && mapped > 0 && mapped <= MAXIMUM_MAPPED_BYTES,
                    "bounded original source event",
                )?;
                bound += roles;
            }
            need(
                bound == SOURCE_ROLES as u64,
                "each original call binds all851 source roles",
            )?;
            // Every source event of this call carries that call's actual input
            // token tag: teacher-forced and natural calls are distinguished by
            // the held receipt, never by an invented field.
            let teacher = manifest["calls"][(repeat * 2 + call) as usize]["teacher"]
                .as_u64()
                .ok_or("source event actual call teacher")?;
            for event in &events[at - order.len()..at] {
                need(
                    event["teacher"].as_u64() == Some(teacher),
                    "source event teacher equals this call's actual input token",
                )?;
            }
        }
    }
    need(at == SOURCE_EVENTS, "complete original source event cohort")
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Codec {
    Bf16,
    F32,
    I64,
}
impl Codec {
    fn width(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
            Self::I64 => 8,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::I64 => "I64",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Label {
    Embedding,
    Positions,
    LayerOutput,
    Normalized,
    Logits,
    BeforeConv4,
    BeforeRecurrent,
    BeforeKey,
    BeforeValue,
    AfterConv4,
    AfterRecurrent,
    AfterKey,
    AfterValue,
}
impl Label {
    fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Positions => "positions",
            Self::LayerOutput => "layer_output",
            Self::Normalized => "normalized",
            Self::Logits => "logits",
            Self::BeforeConv4 => "before.conv4",
            Self::BeforeRecurrent => "before.recurrent",
            Self::BeforeKey => "before.key",
            Self::BeforeValue => "before.value",
            Self::AfterConv4 => "after.conv4",
            Self::AfterRecurrent => "after.recurrent",
            Self::AfterKey => "after.key",
            Self::AfterValue => "after.value",
        }
    }
}
/// Compact finite record descriptor: no retained JSON string per record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Spec {
    repeat: usize,
    call: usize,
    layer: Option<usize>,
    label: Label,
    shape: [usize; 4],
    rank: u8,
    codec: Codec,
}
impl Spec {
    fn elements(&self) -> usize {
        self.shape[..self.rank as usize].iter().product()
    }
    fn bytes(&self) -> usize {
        self.elements() * self.codec.width()
    }
    fn shape(&self) -> Vec<usize> {
        self.shape[..self.rank as usize].to_vec()
    }
}
/// Validated raw interval identity; the manifest JSON tree is released.
struct RecordRow {
    offset: u64,
    sha: [u8; 32],
}
fn digest(raw: &[u8]) -> [u8; 32] {
    Sha256::digest(raw).into()
}
fn hex(bits: &[u8; 32]) -> String {
    bits.iter().map(|b| format!("{b:02x}")).collect()
}
fn digest_from_hex(text: &str) -> Proof<[u8; 32]> {
    need(
        text.len() == 64 && text.bytes().all(|b| b.is_ascii_hexdigit()),
        "64 hex digest",
    )?;
    let mut out = [0u8; 32];
    for (at, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        let word = std::str::from_utf8(pair)?;
        out[at] = u8::from_str_radix(word, 16)?;
    }
    Ok(out)
}
/// Retained compact typed metadata plus every fixed `Reference` header exactly
/// once and the five held owners' path/digest heap capacities exactly once.
/// The inline `Held` structs are already inside `size_of::<Reference>()`.
fn retained_allocation_bound() -> Proof<u64> {
    u64::try_from(std::mem::size_of::<Reference>())?
        .checked_add(typed_extent()?)
        .and_then(|n| {
            n.checked_add((HELD_OWNERS as u64).checked_mul((HELD_PATH_CAP + HELD_SHA_CAP) as u64)?)
        })
        .ok_or_else(|| Box::<dyn std::error::Error>::from("retained allocation bound"))
}
fn retained_metadata_bound() -> Proof<u64> {
    retained_allocation_bound()
}
fn typed_extent() -> Proof<u64> {
    (RECORD_COUNT as u64)
        .checked_mul((std::mem::size_of::<Spec>() + std::mem::size_of::<RecordRow>()) as u64)
        .ok_or_else(|| Box::<dyn std::error::Error>::from("typed metadata extent"))
}
fn first_generation() -> Proof<u64> {
    (RECORD_COUNT as u64)
        .checked_mul(std::mem::size_of::<Spec>() as u64)
        .ok_or_else(|| Box::<dyn std::error::Error>::from("first typed generation"))
}
/// Both descriptor generations live at once: the first validated geometry plus
/// the final specs/records built from the same held manifest tree.
fn typed_generation_peak() -> Proof<u64> {
    first_generation()?
        .checked_add(typed_extent()?)
        .ok_or_else(|| Box::<dyn std::error::Error>::from("two typed descriptor generations"))
}
/// Live allocation accounting against the already-acquired fixed harness. The
/// high-water is the actual admission extent, distinct from retained metadata.
struct Admission {
    limit: u64,
    live: u64,
    high: u64,
}
impl Admission {
    fn new(limit: u64) -> Self {
        Self {
            limit,
            live: 0,
            high: 0,
        }
    }
    /// Start with the already source-derived owner/helper phase reservation.
    /// Tree/raw charges then see only the remaining allowance, and the high-water
    /// includes the reservation from the first allocation through report.
    fn new_reserved(limit: u64, reserved: u64) -> Proof<Self> {
        need(
            reserved <= limit,
            "owner/helper reservation inside the 384MiB diagnostic harness",
        )?;
        Ok(Self {
            limit,
            live: reserved,
            high: reserved,
        })
    }
    fn charge(&mut self, bytes: u64, why: &str) -> Proof<()> {
        let next = self
            .live
            .checked_add(bytes)
            .ok_or_else(|| format!("{why} extent overflow"))?;
        need(
            next <= self.limit,
            format!("{why} above the 384MiB diagnostic harness"),
        )?;
        self.live = next;
        self.high = self.high.max(next);
        Ok(())
    }
    fn release(&mut self, bytes: u64, why: &str) -> Proof<()> {
        self.live = self
            .live
            .checked_sub(bytes)
            .ok_or_else(|| format!("{why} release"))?;
        Ok(())
    }
}
/// Charge the two-stage bootstrap (input + escaped-string scratch), census the
/// held bytes, charge the derived tree upper, then construct the `Value`.
/// Input/scratch are released immediately after parse; the tree stays charged
/// until the caller drops the value.
fn held_json(held: &Held, admission: &mut Admission, label: &str) -> Proof<(Value, u64)> {
    let len = usize::try_from(held.bytes).map_err(|_| "held document extent")?;
    let input = u64::try_from(len).map_err(|_| "held document extent")?;
    let scratch = u64::try_from(json_allocation::census_scratch_peak(len)?)
        .map_err(|_| "census scratch extent")?;
    admission.charge(input, label)?;
    admission.charge(scratch, label)?;
    let bytes = match held.read_whole() {
        Ok(bytes) => bytes,
        Err(error) => {
            admission.release(input, label)?;
            admission.release(scratch, label)?;
            return Err(error);
        }
    };
    let parsed = (|| {
        // One borrowing census over these exact retained bytes; Qwen refuses
        // decoded private serde sentinel keys before the normal Value parse.
        let allocation = json_allocation::census_json(&bytes, ValueConstruction::SerdeValue)?;
        let tree = u64::try_from(allocation.tree_heap_upper()).map_err(|_| "census tree extent")?;
        admission.charge(tree, label)?;
        let value: Value = serde_json::from_slice(allocation.input())
            .map_err(|error| format!("{label} parse refused: {error}"))?;
        Ok((value, tree))
    })();
    // Drop the input/scratch owners before releasing their charged extents.
    drop(bytes);
    admission.release(input, label)?;
    admission.release(scratch, label)?;
    parsed
}
/// Pre-allocation proof of the fixed 384MiB harness: one largest declared
/// document bootstrap, both typed descriptor generations, two simultaneous raw
/// state records and the retained five-owner lease fit before any input buffer
/// exists. Returns the harness extent the caller must acquire first.
pub(super) fn natural_admission_preflight() -> Proof<u64> {
    let cap = usize::try_from(MANIFEST_CAP).map_err(|_| "manifest cap extent")?;
    let scratch = u64::try_from(json_allocation::census_scratch_peak(cap)?)
        .map_err(|_| "manifest census scratch")?;
    let bootstrap = MANIFEST_CAP
        .checked_add(scratch)
        .ok_or_else(|| Box::<dyn std::error::Error>::from("manifest bootstrap"))?;
    let raw_pair = 2u64
        .checked_mul(MAX_RAW_RECORD_BYTES)
        .ok_or_else(|| Box::<dyn std::error::Error>::from("two actual raw state records"))?;
    let early_logits = VOCAB
        .checked_mul(2)
        .ok_or_else(|| Box::<dyn std::error::Error>::from("early fullV logits"))?;
    let typed = typed_generation_peak()?;
    let admission = owner_helper_phase_bound()?
        .checked_add(bootstrap)
        .and_then(|n| n.checked_add(typed))
        .and_then(|n| n.checked_add(raw_pair))
        .and_then(|n| n.checked_add(early_logits))
        .ok_or_else(|| Box::<dyn std::error::Error>::from("natural admission preflight"))?;
    need(
        admission <= DIAGNOSTIC_HOST_BYTES,
        "natural metadata admission fits the fixed 384MiB harness",
    )?;
    Ok(DIAGNOSTIC_HOST_BYTES)
}
fn spec(
    repeat: usize,
    call: usize,
    layer: Option<usize>,
    label: Label,
    shape: [usize; 4],
    rank: u8,
    codec: Codec,
) -> Spec {
    Spec {
        repeat,
        call,
        layer,
        label,
        shape,
        rank,
        codec,
    }
}
fn states(out: &mut Vec<Spec>, repeat: usize, call: usize, before: bool) {
    if before && call == 0 {
        return;
    }
    let rows = call + usize::from(!before);
    for layer in 0..64 {
        let fields = if layer % 4 == 3 {
            vec![
                (
                    if before {
                        Label::BeforeKey
                    } else {
                        Label::AfterKey
                    },
                    [1, 4, rows, 256],
                    4,
                    Codec::Bf16,
                ),
                (
                    if before {
                        Label::BeforeValue
                    } else {
                        Label::AfterValue
                    },
                    [1, 4, rows, 256],
                    4,
                    Codec::Bf16,
                ),
            ]
        } else {
            vec![
                (
                    if before {
                        Label::BeforeConv4
                    } else {
                        Label::AfterConv4
                    },
                    [1, 10240, 4, 0],
                    3,
                    Codec::Bf16,
                ),
                (
                    if before {
                        Label::BeforeRecurrent
                    } else {
                        Label::AfterRecurrent
                    },
                    [1, 48, 128, 128],
                    4,
                    Codec::F32,
                ),
            ]
        };
        for (label, shape, rank, codec) in fields {
            out.push(spec(repeat, call, Some(layer), label, shape, rank, codec));
        }
    }
}
fn specs() -> Vec<Spec> {
    let mut out = Vec::with_capacity(1040);
    for repeat in 0..2 {
        for call in 0..2 {
            states(&mut out, repeat, call, true);
            for (label, shape, rank, codec) in [
                (Label::Embedding, [1, 1, 5120, 0], 3, Codec::Bf16),
                (Label::Positions, [3, 1, 1, 0], 3, Codec::I64),
            ] {
                out.push(spec(repeat, call, None, label, shape, rank, codec));
            }
            for layer in 0..64 {
                out.push(spec(
                    repeat,
                    call,
                    Some(layer),
                    Label::LayerOutput,
                    [1, 1, 5120, 0],
                    3,
                    Codec::Bf16,
                ));
            }
            for (label, count) in [(Label::Normalized, 5120), (Label::Logits, 248320)] {
                out.push(spec(
                    repeat,
                    call,
                    None,
                    label,
                    [1, 1, count, 0],
                    3,
                    Codec::Bf16,
                ));
            }
            states(&mut out, repeat, call, false);
        }
    }
    out
}
fn validate_records(
    manifest: &Value,
    teachers: Option<&[[u32; 2]; 2]>,
) -> Proof<(Vec<Spec>, Vec<RecordRow>)> {
    let specs = specs();
    let records = manifest["records"].as_array().ok_or("full64 record list")?;
    need(
        specs.len() == 1040 && records.len() == specs.len(),
        "exact full64 record cohort",
    )?;
    let mut rows = Vec::with_capacity(specs.len());
    let mut offset = 0u64;
    for (r, s) in records.iter().zip(&specs) {
        let shape: Value = s.shape().into();
        need(
            r["repeat"] == s.repeat
                && r["call"] == s.call
                && r["position"] == s.call
                && teachers
                    .is_none_or(|t| r["teacher"].as_u64() == Some(u64::from(t[s.repeat][s.call])))
                && r["layer"] == s.layer.map_or(Value::Null, |n| n.into())
                && r["label"] == s.label.name()
                && r["shape"] == shape
                && r["dtype"] == s.codec.name()
                && r["device"] == "cpu"
                && r["offset"] == offset
                && r["bytes"] == s.bytes(),
            format!(
                "full producer repeat{}/call{}/layer{:?}/{} metadata",
                s.repeat,
                s.call,
                s.layer,
                s.label.name()
            ),
        )?;
        let strides = r["original_stride"].as_array().ok_or("original strides")?;
        need(
            strides.len() == s.rank as usize,
            "exact original stride rank",
        )?;
        let mut span = 0u64;
        for (d, stride) in s.shape().iter().zip(strides) {
            span = span
                .checked_add(
                    ((*d - 1) as u64)
                        .checked_mul(stride.as_u64().ok_or("nonnegative stride")?)
                        .ok_or("stride multiplication")?,
                )
                .ok_or("stride sum")?;
        }
        let digest = digest_from_hex(string(r, "sha256")?)?;
        need(
            span < s.elements() as u64,
            "bounded actual stride in elements/full-record SHA",
        )?;
        rows.push(RecordRow {
            offset,
            sha: digest,
        });
        offset = offset
            .checked_add(s.bytes() as u64)
            .ok_or("record extent")?;
    }
    need(
        offset == RAW_BYTES
            && manifest["raw"]["bytes"] == RAW_BYTES
            && manifest["raw"]["file"] == "raw.bin",
        "all raw intervals/EOF/no gaps",
    )?;
    Ok((specs, rows))
}
#[allow(dead_code)]
struct Scope {
    declared: BTreeSet<(usize, usize)>,
    required: BTreeSet<usize>,
    total: usize,
    floats: usize,
    words: usize,
    positions: usize,
}
#[allow(dead_code)]
fn scope_totals(specs: &[Spec], proposals: &[(usize, usize)]) -> Proof<Scope> {
    need(
        !proposals.is_empty(),
        "nonempty declared shared proposal scope",
    )?;
    let mut declared = BTreeSet::new();
    for &(repeat, call) in proposals {
        need(
            repeat < 2 && call < 2 && declared.insert((repeat, call)),
            "bounded unique declared shared proposal scope",
        )?;
    }
    let mut required = BTreeSet::new();
    let (mut total, mut floats, mut words, mut positions) = (0, 0, 0, 0);
    for (index, spec) in specs.iter().enumerate() {
        if !declared.contains(&(spec.repeat, spec.call)) {
            continue;
        }
        total += 1;
        if spec.codec == Codec::I64 {
            positions += spec.elements();
        } else {
            required.insert(index);
            floats += 1;
            words += spec.elements();
        }
    }
    need(
        !required.is_empty() && floats == required.len(),
        "declared shared proposal has original float producers",
    )?;
    Ok(Scope {
        declared,
        required,
        total,
        floats,
        words,
        positions,
    })
}
#[allow(dead_code)]
fn verify_scope(
    specs: &[Spec],
    scope: &Scope,
    records: &[usize],
    pending: usize,
    words: usize,
    positions: usize,
    mismatches: usize,
) -> Proof<()> {
    need(
        pending == 0,
        "no in-progress producer chunks outside completed proposals",
    )?;
    let mut seen = BTreeSet::new();
    for index in records {
        let spec = specs
            .get(*index)
            .ok_or("original record index in complete cohort")?;
        need(
            spec.codec != Codec::I64,
            "completed original proposal/record identity/exact extent",
        )?;
        need(
            scope.required.contains(index) && seen.insert(*index),
            "completed records exactly declared proposal set once",
        )?;
    }
    need(
        seen == scope.required && records.len() == scope.floats,
        "declared shared proposal record set complete/no extra/duplicate",
    )?;
    need(
        words == scope.words && positions == scope.positions && mismatches == 0,
        "declared shared proposal word/position totals exact",
    )
}
struct Held {
    path: PathBuf,
    file: File,
    stamp: [u64; 7],
    bytes: u64,
    sha: String,
}
impl Held {
    fn new(root: &Path, pin: &Value, cap: u64) -> Proof<Self> {
        let declared = string(pin, "path")?;
        need(
            declared.len() <= HELD_PATH_CAP,
            "declared held path within 4096 bytes before join",
        )?;
        let name = Path::new(declared)
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("expected basename")?;
        let bytes = pin["bytes"].as_u64().ok_or("expected extent")?;
        Self::new_external(root, name, bytes, string(pin, "sha256")?, cap)
    }
    /// The one external-pin constructor. The path and digest caps are enforced
    /// before `join`/copy, including the checked root+relative result. The
    /// descriptor opened here is the only owner and is verified on its own bytes.
    fn new_external(root: &Path, name: &str, bytes: u64, sha: &str, cap: u64) -> Proof<Self> {
        need(root.is_absolute(), "absolute held root")?;
        need(
            bytes > 0 && bytes <= cap,
            "bounded expected file extent before allocation",
        )?;
        need(
            !name.is_empty()
                && name.len() <= HELD_PATH_CAP
                && !name.contains('/')
                && name != "."
                && name != "..",
            "bounded relative held basename before join",
        )?;
        need(
            sha.len() == HELD_SHA_CAP && sha.bytes().all(|b| b.is_ascii_hexdigit()),
            "64 hex external pin digest before copy",
        )?;
        let joined = root
            .as_os_str()
            .len()
            .checked_add(1)
            .and_then(|n| n.checked_add(name.len()))
            .ok_or("held root/relative path length")?;
        need(
            joined <= HELD_PATH_CAP,
            "checked held root+relative path within 4096 bytes",
        )?;
        let mut path = PathBuf::with_capacity(HELD_PATH_CAP);
        path.push(root);
        path.push(name);
        need(
            path.as_os_str().len() <= HELD_PATH_CAP && path.capacity() <= HELD_PATH_CAP,
            "constructed held path within its retained capacity",
        )?;
        let mut digest = String::with_capacity(HELD_SHA_CAP);
        digest.push_str(sha);
        need(
            digest.capacity() <= HELD_SHA_CAP,
            "digest inside retained capacity",
        )?;
        let file = open(&path)?;
        let before = stamp(&file)?;
        let held = Self {
            path,
            file,
            stamp: before,
            bytes,
            sha: digest,
        };
        held.verify()?;
        Ok(held)
    }
    fn verify(&self) -> Proof<()> {
        // Retained-FD authority only: the held descriptor is the lease. A path
        // reopen is never a replacement for the original owner.
        need(
            stamp(&self.file)? == self.stamp && hash(&self.file, self.bytes)? == self.sha,
            "same original held expected FD/fullSHA/EOF",
        )
    }
    fn retained_heap(&self) -> usize {
        self.path.capacity() + self.sha.capacity()
    }
    fn read_whole(&self) -> Proof<Vec<u8>> {
        let bytes = usize::try_from(self.bytes).map_err(|_| "held document extent")?;
        need(
            bytes > 0 && self.bytes <= MANIFEST_CAP,
            "bounded whole held document",
        )?;
        need(
            stamp(&self.file)? == self.stamp,
            "same held FD before whole read",
        )?;
        let mut out = vec![0u8; bytes];
        self.file.read_exact_at(&mut out, 0)?;
        need(
            stamp(&self.file)? == self.stamp,
            "same held FD after whole read",
        )?;
        Ok(out)
    }
    fn read(&self, offset: u64, bytes: usize) -> Proof<Vec<u8>> {
        need(
            bytes > 0
                && bytes <= 4 * 1024 * 1024
                && offset
                    .checked_add(bytes as u64)
                    .is_some_and(|n| n <= self.bytes),
            "complete bounded original expected record",
        )?;
        need(
            stamp(&self.file)? == self.stamp,
            "same expected FD before record",
        )?;
        let mut out = vec![0u8; bytes];
        self.file.read_exact_at(&mut out, offset)?;
        need(
            stamp(&self.file)? == self.stamp,
            "same expected FD after record",
        )?;
        Ok(out)
    }
}
/// The accepted receipt extent/SHA is an external caller operand supplied by the
/// trusted launch boundary (`natural_data.load_authority`); it is never measured
/// or minted here. Fast refusal happens before any read, path reopen, context,
/// model or GPU work; the fixed harness is acquired before bytes are hashed.
pub(super) fn natural_receipt_present(root: &Path, external: (u64, &str)) -> Proof<()> {
    need(root.is_absolute(), "absolute accepted natural receipt root")?;
    need(
        external.0 > 0 && external.0 <= RECEIPT_CAP,
        "external accepted natural receipt extent within the 1MiB cap",
    )?;
    need(
        external.1.len() == HELD_SHA_CAP && external.1.bytes().all(|b| b.is_ascii_hexdigit()),
        "external accepted natural receipt SHA shape",
    )?;
    let joined = root
        .as_os_str()
        .len()
        .checked_add(1)
        .and_then(|n| n.checked_add(NATURAL_ACCEPTED_FILE.len()))
        .ok_or("accepted natural receipt path length")?;
    need(
        joined <= HELD_PATH_CAP,
        "accepted natural receipt path within 4096 bytes before join",
    )?;
    let metadata = std::fs::symlink_metadata(root.join(NATURAL_ACCEPTED_FILE))?;
    need(
        metadata.file_type().is_file() && metadata.len() == external.0,
        "held accepted natural receipt extent equals the external pin",
    )
}
pub(super) struct DiagnosticInventory {
    pub root_bytes: u64,
    pub readback_window_bytes: u64,
    pub domain_metadata_bytes: u64,
    pub reference_bytes: u64,
    pub manifest_input_bytes: u64,
    pub manifest_tree_bytes: u64,
    pub admission_bytes: u64,
    pub observed_bytes: u64,
    pub snapshot_bytes: u64,
    pub committed_bytes: u64,
    pub report_bytes: u64,
    pub peak_bytes: u64,
}
pub(super) fn natural_reference(
    root: &Path,
    pin: (u64, &str),
) -> Proof<(Arc<Reference>, DiagnosticInventory)> {
    let reference = Reference::open_natural(root, pin)?;
    let inventory = reference.inventory()?;
    need(
        inventory.reference_bytes <= retained_metadata_bound()?,
        "actual retained reference inside the proved metadata bound",
    )?;
    need(
        inventory.admission_bytes <= DIAGNOSTIC_HOST_BYTES,
        "actual admission high-water inside the fixed 384MiB harness",
    )?;
    Ok((Arc::new(reference), inventory))
}
pub(super) struct Reference {
    cache_domains: u64,
    records: Vec<RecordRow>,
    manifest_input_bytes: usize,
    manifest_tree_bytes: usize,
    specs: Vec<Spec>,
    manifest_owner: Held,
    raw: Held,
    accepted: Option<Held>,
    invocation: Option<Held>,
    image: Option<Held>,
    admission_bytes: u64,
    natural: bool,
    compositional: bool,
    teachers: [[u32; 2]; 2],
    selections: [[(u32, u32); 2]; 2],
}
impl Reference {
    fn open(root: &Path) -> Proof<Self> {
        let mut admission =
            Admission::new_reserved(DIAGNOSTIC_HOST_BYTES, owner_helper_phase_bound()?)?;
        let owner = Held::new(
            root,
            &json!({"path":"accepted-original-full-window-1.json",
                "bytes":TEACHER_ACCEPTED_BYTES,"sha256":TEACHER_ACCEPTED_SHA}),
            RECEIPT_CAP,
        )?;
        let (accepted, tree) = held_json(&owner, &mut admission, "teacher accepted receipt")?;
        Self::open_common(
            root,
            accepted,
            tree,
            false,
            false,
            None,
            owner,
            None,
            &mut admission,
        )
    }
    fn open_natural(root: &Path, pin: (u64, &str)) -> Proof<Self> {
        need(
            pin.0 > 0 && pin.0 <= RECEIPT_CAP && pin.1.len() == 64,
            "bounded external natural accepted reference pin",
        )?;
        // The accepted receipt is opened as one retained owner from the external
        // caller pin and read from that exact descriptor; the path is never
        // reopened as a replacement.
        let mut admission =
            Admission::new_reserved(DIAGNOSTIC_HOST_BYTES, owner_helper_phase_bound()?)?;
        let owner = Held::new_external(root, NATURAL_ACCEPTED_FILE, pin.0, pin.1, RECEIPT_CAP)?;
        let (accepted, tree) = held_json(&owner, &mut admission, "natural accepted receipt")?;
        for (key, name) in [
            ("invocation", "invocation.json"),
            ("image_inspect", "image-inspect.json"),
        ] {
            need(
                Path::new(string(&accepted[key], "path")?)
                    .file_name()
                    .and_then(|n| n.to_str())
                    == Some(name),
                format!("declared natural artifact basename {name}"),
            )?;
        }
        let invocation = Held::new(root, &accepted["invocation"], MANIFEST_CAP)?;
        let image = Held::new(root, &accepted["image_inspect"], RECEIPT_CAP)?;
        Self::open_common(
            root,
            accepted,
            tree,
            true,
            false,
            Some(pin),
            owner,
            Some((invocation, image)),
            &mut admission,
        )
    }
    /// Unchanged held teacher receipt; the independent fullV scan must prove the
    /// call1 input actually recorded by the original capture. No new model call.
    fn open_compositional(root: &Path) -> Proof<Self> {
        let mut admission =
            Admission::new_reserved(DIAGNOSTIC_HOST_BYTES, owner_helper_phase_bound()?)?;
        let owner = Held::new(
            root,
            &json!({"path":"accepted-original-full-window-1.json",
                "bytes":TEACHER_ACCEPTED_BYTES,"sha256":TEACHER_ACCEPTED_SHA}),
            RECEIPT_CAP,
        )?;
        let (accepted, tree) = held_json(&owner, &mut admission, "teacher accepted receipt")?;
        Self::open_common(
            root,
            accepted,
            tree,
            true,
            true,
            None,
            owner,
            None,
            &mut admission,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn open_common(
        root: &Path,
        accepted: Value,
        accepted_tree: u64,
        natural: bool,
        compositional: bool,
        natural_pin: Option<(u64, &str)>,
        accepted_owner: Held,
        artifacts: Option<(Held, Held)>,
        admission: &mut Admission,
    ) -> Proof<Self> {
        let teacher_receipt = accepted["schema"]
            == "zg.actual-original27b-full-window-reference.v1"
            && accepted["status"] == "ACTUAL_ORIGINAL27B_FULL_WINDOW_REFERENCE_ONLY";
        let natural_receipt = accepted["schema"]
            == "zg.actual-original27b-full-window-natural-reference.v1"
            && accepted["status"] == "ACTUAL_ORIGINAL27B_FULL_WINDOW_NATURAL_REFERENCE_ONLY"
            && accepted["scope"]["natural_feedback_qualified"] == true
            && accepted["scope"]["teacher_forcing_only"] == false
            && accepted["independent_original_natural_trajectory"] == true;
        need(
            match (natural, compositional) {
                (false, false) => teacher_receipt && natural_pin.is_none() && artifacts.is_none(),
                (true, false) => natural_receipt && natural_pin.is_some() && artifacts.is_some(),
                (true, true) => teacher_receipt && natural_pin.is_none() && artifacts.is_none(),
                (false, true) => false,
            },
            "accepted complete original CPU source-only reference cohort",
        )?;
        need(
            accepted["snapshot"] == SNAPSHOT
                && accepted["records"] == 1040
                && accepted["raw_bytes"] == RAW_BYTES
                && accepted["original_source_and_owned_lifetime_closed"] == true,
            "accepted complete original CPU source-only reference",
        )?;
        need(
            Path::new(string(&accepted["manifest"], "path")?)
                .file_name()
                .and_then(|name| name.to_str())
                == Some("manifest.json")
                && Path::new(string(&accepted["raw"], "path")?)
                    .file_name()
                    .and_then(|name| name.to_str())
                    == Some("raw.bin"),
            "expected root has manifest.json/raw.bin basenames",
        )?;
        let manifest_owner = Held::new(root, &accepted["manifest"], MANIFEST_CAP)?;
        let (manifest, manifest_tree) = held_json(&manifest_owner, admission, "original manifest")?;
        validate_source(
            &accepted,
            &manifest,
            natural && !compositional,
            artifacts.as_ref(),
            admission,
        )?;
        let raw = Held::new(root, &accepted["raw"], RAW_BYTES)?;
        // The fresh natural cohort retains all five original held owners for the
        // whole lease: accepted receipt, manifest, raw, invocation and image.
        let (invocation, image) = match artifacts {
            Some((invocation, image)) => (Some(invocation), Some(image)),
            None => (None, None),
        };
        need(
            raw.bytes == RAW_BYTES && manifest["raw"]["sha256"] == raw.sha,
            "same complete original raw artifact",
        )?;
        // The accepted receipt tree is no longer needed: the retained owners are
        // open, validated and independent of the transient parse tree.
        drop(accepted);
        admission.release(accepted_tree, "accepted receipt tree")?;
        admission.charge(typed_generation_peak()?, "two typed descriptor generations")?;
        let (geometry, _) = validate_records(&manifest, None)?;
        // Price the actual largest fullV logits record before the first record
        // buffer exists; it stays live through the complete selection phase.
        let early_logits = VOCAB
            .checked_mul(2)
            .ok_or("bounded early fullV logits record")?;
        admission.charge(early_logits, "early fullV logits window")?;
        let mut selections = [[(0u32, 0u32); 2]; 2];
        for (repeat, row) in selections.iter_mut().enumerate() {
            for (call, slot) in row.iter_mut().enumerate() {
                let index = geometry
                    .iter()
                    .position(|s| s.repeat == repeat && s.call == call && s.label == Label::Logits)
                    .ok_or("original fullV record absent")?;
                let bytes = record_bytes_manifest(&manifest, &raw, index, &geometry[index])?;
                *slot = bf16_argmax(&bytes)?;
            }
        }
        let mut teachers = [[11u32, 353u32]; 2];
        if natural {
            let held = geometry
                .iter()
                .find(|s| s.call == 1 && s.label == Label::Embedding)
                .ok_or("original call1 embedding record absent")?;
            let held = manifest["records"][spec_index(&manifest, held)?]["teacher"]
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .filter(|v| *v < 248320)
                .ok_or("bounded original call1 teacher input")?;
            for repeat in 0..2 {
                let seed = geometry
                    .iter()
                    .position(|s| s.repeat == repeat && s.call == 0)
                    .ok_or("original seed record")?;
                need(
                    manifest["records"][seed]["teacher"].as_u64() == Some(11),
                    "natural call0 is original seed11",
                )?;
                teachers[repeat] = [11, held];
            }
            if !compositional {
                for repeat in 0..2 {
                    need(
                        held == selections[repeat][0].0,
                        "independent natural call1 input equals accepted call0 selection",
                    )?;
                }
            }
        }
        let (specs, records) = validate_records(&manifest, Some(&teachers))?;
        drop(geometry);
        admission.release(first_generation()?, "first typed descriptor generation")?;
        admission.release(early_logits, "early fullV logits window")?;
        // The fresh natural comparator reads two actual raw state records at the
        // same time; charge the pair before either record buffer exists.
        let raw_record_bytes = specs
            .iter()
            .map(|spec| spec.bytes() as u64)
            .max()
            .ok_or("bounded original raw record")?;
        let raw_pair = 2u64
            .checked_mul(raw_record_bytes)
            .ok_or("two simultaneous raw state records")?;
        if natural {
            admission.charge(raw_pair, "two simultaneous raw state records")?;
        }
        let manifest_input_bytes =
            usize::try_from(manifest_owner.bytes).map_err(|_| "manifest extent fits usize")?;
        // Compact typed metadata only: 1040 finite descriptors plus their raw
        // interval digests. The transient parse tree is released below and is
        // bounded by the declared manifest input cap, never a per-node guess.
        let manifest_tree_bytes = specs
            .len()
            .checked_mul(std::mem::size_of::<Spec>())
            .and_then(|n| {
                n.checked_add(
                    records
                        .len()
                        .checked_mul(std::mem::size_of::<RecordRow>())?,
                )
            })
            .ok_or("compact typed metadata extent")?;
        let reference = Self {
            cache_domains: manifest["cache_domains"]
                .as_u64()
                .filter(|v| *v == 112)
                .ok_or("held manifest 112 cache domains")?,
            records,
            manifest_input_bytes,
            manifest_tree_bytes,
            specs,
            manifest_owner,
            raw,
            accepted: Some(accepted_owner),
            invocation,
            image,
            admission_bytes: admission.high,
            natural,
            compositional,
            teachers,
            selections,
        };
        reference.verify_lease()?;
        if natural {
            reference.natural_trajectory(&manifest)?;
            admission.release(raw_pair, "two simultaneous raw state records")?;
        }
        // Release the validation-only JSON tree before any native work.
        drop(manifest);
        admission.release(manifest_tree, "manifest JSON tree")?;
        Ok(reference)
    }
    fn record_bytes_for(&self, spec: &Spec) -> Proof<Vec<u8>> {
        let index = self
            .specs
            .iter()
            .position(|s| s == spec)
            .ok_or("record spec in cohort")?;
        self.record_bytes(index)
    }
    fn record_bytes(&self, index: usize) -> Proof<Vec<u8>> {
        let spec = self.specs.get(index).ok_or("record index in cohort")?;
        let row = self.records.get(index).ok_or("record row in cohort")?;
        let bytes = self.raw.read(row.offset, spec.bytes())?;
        need(
            digest(&bytes) == row.sha,
            "whole original record SHA before comparison",
        )?;
        Ok(bytes)
    }
    /// Computed retained-reference extent after validation: typed descriptors,
    /// fixed record digests, selections/teachers, the five held owners and the
    /// actual capacities of every retained path/String heap allocation.
    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.specs.capacity() * std::mem::size_of::<Spec>()
            + self.records.capacity() * std::mem::size_of::<RecordRow>()
            + [
                Some(&self.manifest_owner),
                Some(&self.raw),
                self.accepted.as_ref(),
                self.invocation.as_ref(),
                self.image.as_ref(),
            ]
            .into_iter()
            .flatten()
            .map(Held::retained_heap)
            .sum::<usize>()
    }
    /// Before/after lease recheck of the exact five original held FD owners.
    pub(super) fn verify_lease(&self) -> Proof<()> {
        self.manifest_owner.verify()?;
        self.raw.verify()?;
        if self.natural {
            need(
                self.accepted.is_some() && self.invocation.is_some() && self.image.is_some(),
                "all five original natural held FD owners retained",
            )?;
        }
        for held in [
            self.accepted.as_ref(),
            self.invocation.as_ref(),
            self.image.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            held.verify()?;
        }
        Ok(())
    }
    /// Five-FD identity evidence for the final report; never a path substitution.
    pub(super) fn lease_evidence(&self) -> Value {
        let one = |held: &Held| {
            json!({"path":held.path.display().to_string(),"bytes":held.bytes,
                "sha256":held.sha,
                "fd_identity":{"dev":held.stamp[0],"ino":held.stamp[1],
                    "mtime":held.stamp[3],"mtime_nsec":held.stamp[4]}})
        };
        let owners = [one(&self.manifest_owner), one(&self.raw)]
            .into_iter()
            .chain(self.accepted.iter().map(one))
            .chain(self.invocation.iter().map(one))
            .chain(self.image.iter().map(one))
            .collect::<Vec<_>>();
        json!({"owners":owners,
            "independent_artifact_pins":5,
            "retained_across_validation_and_report":true,
            "mutable_path_reopen_as_replacement":false})
    }
    /// Phase-owned extents: each is the computed maximum of one live phase,
    /// never their sum and never a guessed comparator allowance. The admission
    /// extent is the actual charged high-water from the census/parse stages;
    /// `reference_bytes` is the separate retained typed metadata.
    pub(super) fn inventory(&self) -> Proof<DiagnosticInventory> {
        const ROOT: u64 = 159_121_408;
        const WINDOW: u64 = MAX_RAW_RECORD_BYTES;
        const DOMAINS: u64 = 112;
        const DOMAIN_META: u64 = DOMAINS
            * (std::mem::size_of::<(StateSlot, Vec<f32>)>() as u64
                + 3 * std::mem::size_of::<usize>() as u64
                + std::mem::size_of::<StateSlot>() as u64
                + 2 * std::mem::size_of::<usize>() as u64);
        let metadata = u64::try_from(self.retained_bytes())?;
        let typed = u64::try_from(self.manifest_tree_bytes)?;
        let input = u64::try_from(self.manifest_input_bytes)?;
        let max_raw = self
            .specs
            .iter()
            .map(|s| s.bytes() as u64)
            .max()
            .unwrap_or(0);
        let max_expanded = self
            .specs
            .iter()
            .map(|s| s.elements() as u64 * 4)
            .max()
            .unwrap_or(0);
        let admission = self.admission_bytes;
        need(
            admission <= DIAGNOSTIC_HOST_BYTES,
            "actual admission high-water inside the diagnostic harness",
        )?;
        let observed = metadata
            .checked_add(4 * 1024 * 1024)
            .and_then(|n| n.checked_add(max_raw))
            .and_then(|n| n.checked_add(max_expanded))
            .ok_or("observed extent")?;
        let snapshot = ROOT
            .checked_mul(2)
            .and_then(|n| n.checked_add(2 * WINDOW))
            .and_then(|n| n.checked_add(DOMAIN_META))
            .and_then(|n| n.checked_add(metadata))
            .ok_or("snapshot extent")?;
        let committed = ROOT
            .checked_add(max_expanded)
            .and_then(|n| n.checked_add(metadata))
            .ok_or("committed extent")?;
        // 1MiB JSON output cap plus the same-size serializer scratch; domain
        // vectors and unused scopes are dropped before this phase.
        let report = metadata
            .checked_add(2 * 1024 * 1024)
            .and_then(|n| n.checked_add(self.records.len() as u64 * 8))
            .ok_or("report extent")?;
        let peak = admission
            .max(observed)
            .max(snapshot)
            .max(committed)
            .max(report);
        Ok(DiagnosticInventory {
            root_bytes: ROOT,
            readback_window_bytes: WINDOW,
            domain_metadata_bytes: DOMAIN_META,
            reference_bytes: metadata,
            manifest_input_bytes: input,
            manifest_tree_bytes: typed,
            admission_bytes: admission,
            observed_bytes: observed,
            snapshot_bytes: snapshot,
            committed_bytes: committed,
            report_bytes: report,
            peak_bytes: peak,
        })
    }
    fn full_v(&self, repeat: usize, call: usize) -> Proof<(u32, u32)> {
        need(repeat < 2 && call < 2, "bounded original fullV call")?;
        let index = self
            .specs
            .iter()
            .position(|s| s.repeat == repeat && s.call == call && s.label == Label::Logits)
            .ok_or("original fullV record absent")?;
        let bytes = self.record_bytes(index)?;
        let selection = bf16_argmax(&bytes)?;
        need(
            selection == self.selections[repeat][call],
            "same independently computed original fullV selection",
        )?;
        Ok(selection)
    }
    fn record_sha(&self, repeat: usize, call: usize, label: Label) -> Proof<String> {
        let index = self
            .specs
            .iter()
            .position(|s| s.repeat == repeat && s.call == call && s.label == label)
            .ok_or_else(|| format!("original record {} absent", label.name()))?;
        Ok(hex(&digest(&self.record_bytes(index)?)))
    }
    /// Prove the held call1 input actually recorded by the original capture. For
    /// a teacher-only receipt this cannot establish an independent natural
    /// trajectory: the recorded call1 input is the teacher token, while natural
    /// feedback requires the independently selected call0 token.
    pub(super) fn join_evidence(&self, actual: Option<[[(u32, u32); 2]; 2]>) -> Proof<Value> {
        need(self.natural, "natural join evidence only")?;
        let mut repeats = Vec::new();
        let mut natural_feedback = true;
        for repeat in 0..2 {
            let seed = self.selections[repeat][0];
            let held = self.selections[repeat][1];
            if let Some(actual) = actual {
                need(
                    actual[repeat][0] == seed && actual[repeat][1] == held,
                    "actual original natural selections equal independent fullV",
                )?;
            }
            let required = seed.0;
            let held_input = self.teachers[repeat][1];
            let feedback = held_input == required;
            natural_feedback &= feedback;
            let available = feedback && !self.compositional;
            repeats.push(json!({
                "repeat":repeat,
                "call0_seed":11,
                "call0_independent_selection":[seed.0,seed.1],
                "call0_embedding_sha256":self.record_sha(repeat,0,Label::Embedding)?,
                "call1_held_teacher_input":held_input,
                "call1_independent_selection":[held.0,held.1],
                "call1_embedding_sha256":self.record_sha(repeat,1,Label::Embedding)?,
                "natural_feedback_input_required":required,
                "held_call1_input_is_natural_feedback":feedback,
                "natural_call_available":available,
            }));
        }
        let independent = !self.compositional && natural_feedback;
        Ok(json!({
            "independent_original_natural_trajectory":independent,
            "actual_observed":actual.is_some(),
            "newcalls":0,
            "teacher_forcing_only":!natural_feedback,
            "held_receipt_complete":true,
            "source_roles_per_call":SOURCE_ROLES,
            "repeats":repeats,
            "reason":if independent {
                "accepted receipt records independent natural feedback at every call1".to_string()
            } else {
                let required = self.selections[0][0].0;
                let held = self.teachers[0][1];
                format!("held teacher-forced call1 input {held} differs from independent natural feedback input {required}; accepted receipt natural_feedback_qualified=false, so a fresh natural capture is required")
            },
        }))
    }
    /// Finite fresh-natural trajectory contract over the actual held bytes:
    /// exact original module order, both calls and both repeats, per-call fullV
    /// identity and predecessor joins, byte-exact prior-state joins, canonical
    /// I64 positions and a byte-identical repetition. Teacher-forced or
    /// composited receipts are not natural authority and never reach here.
    fn natural_trajectory(&self, manifest: &Value) -> Proof<()> {
        need(
            self.natural && !self.compositional,
            "fresh independent natural reference cohort only",
        )?;
        let calls = manifest["calls"]
            .as_array()
            .ok_or("natural original call list")?;
        need(calls.len() == 4, "two independent natural original cohorts")?;
        let groups = module_order()?;
        for repeat in 0..2 {
            let seed = self.selections[repeat][0];
            let call1 = self.selections[repeat][1];
            need(
                seed.0 < 248320 && call1.0 < 248320,
                "finite original natural selections",
            )?;
            let mut fullv: Vec<String> = Vec::with_capacity(2);
            for call in 0..2 {
                let index = repeat * 2 + call;
                let row = &calls[index];
                let identity = self.record_sha(repeat, call, Label::Logits)?;
                let order = row["original_module_order"]
                    .as_array()
                    .ok_or("natural original module order")?;
                need(
                    row["repeat"].as_u64() == Some(repeat as u64)
                        && row["call"].as_u64() == Some(call as u64)
                        && row["position"].as_u64() == Some(call as u64)
                        && row["cache_frontier"].as_u64() == Some(call as u64 + 1)
                        && row["teacher"].as_u64() == Some(u64::from(self.teachers[repeat][call]))
                        && row["greedy_id_diagnostic_only"].as_u64()
                            == Some(u64::from(self.selections[repeat][call].0))
                        && row["selected_max_bits"].as_u64()
                            == Some(u64::from(self.selections[repeat][call].1))
                        && row["logits_record_sha256"] == identity.as_str()
                        && row["natural_feedback_used"] == (call == 1)
                        && order.len() == groups.len()
                        && order
                            .iter()
                            .zip(&groups)
                            .all(|(value, group)| value.as_str() == Some(group.as_str())),
                    format!(
                        "actual original natural repeat{repeat}/call{call} order/positions/frontier/fullV identity"
                    ),
                )?;
                if call == 0 {
                    need(
                        row["predecessor_logits_sha256"].is_null(),
                        "seed call has no natural predecessor",
                    )?;
                } else {
                    need(
                        row["predecessor_logits_sha256"] == fullv[0].as_str(),
                        "call1 predecessor is this cohort actual call0 fullV record",
                    )?;
                }
                fullv.push(identity);
            }
            need(
                self.teachers[repeat][1] == seed.0,
                "actual natural call1 input equals this cohort original call0 selection",
            )?;
            // Every call1-before plane is the held call0-after plane, byte-exact.
            for spec in self.specs.iter().filter(|s| {
                s.repeat == repeat
                    && s.call == 1
                    && matches!(
                        s.label,
                        Label::BeforeConv4
                            | Label::BeforeRecurrent
                            | Label::BeforeKey
                            | Label::BeforeValue
                    )
            }) {
                let after = match spec.label {
                    Label::BeforeConv4 => Label::AfterConv4,
                    Label::BeforeRecurrent => Label::AfterRecurrent,
                    Label::BeforeKey => Label::AfterKey,
                    Label::BeforeValue => Label::AfterValue,
                    _ => return Err("natural before plane".into()),
                };
                let index0 = self
                    .specs
                    .iter()
                    .position(|s| {
                        s.repeat == repeat
                            && s.call == 0
                            && s.layer == spec.layer
                            && s.label == after
                    })
                    .ok_or("natural call0 after record absent")?;
                need(
                    self.record_bytes(index0)?
                        == self.record_bytes(
                            self.specs
                                .iter()
                                .position(|s| s == spec)
                                .ok_or("natural spec index")?,
                        )?,
                    "natural call1 before equals natural call0 after actual bytes",
                )?;
            }
        }
        // Both calls and both repeats are byte-identical: same seed and weights,
        // including the canonical I64 position records.
        for spec in self.specs.iter().filter(|s| s.repeat == 0) {
            let other = self
                .specs
                .iter()
                .position(|s| {
                    s.repeat == 1
                        && s.call == spec.call
                        && s.layer == spec.layer
                        && s.label == spec.label
                })
                .ok_or("repeated original record absent")?;
            need(
                self.record_bytes(
                    self.specs
                        .iter()
                        .position(|s| s == spec)
                        .ok_or("repeated spec index")?,
                )? == self.record_bytes(other)?,
                format!(
                    "repeated original call{}/{} bytes identical",
                    spec.call,
                    spec.label.name()
                ),
            )?;
        }
        // Positions are the actual original call index in every cohort.
        for spec in self.specs.iter().filter(|s| s.codec == Codec::I64) {
            let bytes = self.record_bytes(
                self.specs
                    .iter()
                    .position(|s| s == spec)
                    .ok_or("position spec index")?,
            )?;
            need(
                bytes.chunks_exact(8).all(|word| {
                    i64::from_le_bytes(word.try_into().expect("8-byte word")) == spec.call as i64
                }),
                "actual original canonical positions per call",
            )?;
        }
        Ok(())
    }
}
fn record_bytes_manifest(
    manifest: &Value,
    raw: &Held,
    index: usize,
    spec: &Spec,
) -> Proof<Vec<u8>> {
    let row = manifest["records"]
        .as_array()
        .and_then(|rows| rows.get(index))
        .ok_or("record row in held manifest")?;
    let bytes = raw.read(row["offset"].as_u64().ok_or("record offset")?, spec.bytes())?;
    need(
        sha(&bytes) == string(row, "sha256")?,
        "whole original record SHA before comparison",
    )?;
    Ok(bytes)
}
fn spec_index(manifest: &Value, spec: &Spec) -> Proof<usize> {
    let records = manifest["records"].as_array().ok_or("record list")?;
    records
        .iter()
        .position(|r| {
            r["repeat"].as_u64() == Some(spec.repeat as u64)
                && r["call"].as_u64() == Some(spec.call as u64)
                && r["layer"] == spec.layer.map_or(Value::Null, |n| n.into())
                && r["label"] == spec.label.name()
        })
        .ok_or_else(|| "spec record absent".into())
}
#[allow(dead_code)]
pub(super) fn packet_matches(selected: u32, max_bits: u32, expected: (u32, u32)) -> bool {
    selected == expected.0 && max_bits == expected.1
}
fn bf16_argmax(raw: &[u8]) -> Proof<(u32, u32)> {
    need(
        !raw.is_empty() && raw.len().is_multiple_of(2),
        "complete BF16 fullV words",
    )?;
    // Exact numeric order: sign-magnitude source bits rank as +magnitude and
    // -magnitude, so both signed zeros tie at one numeric zero and keep the
    // lowest index (and that word's own bits), negatives order by magnitude and
    // subnormals keep their source bits. NaN/inf are refused above.
    let key = |bits: u32| -> i64 {
        let magnitude = i64::from((bits >> 16) & 0x7fff);
        if bits & 0x8000_0000 == 0 {
            magnitude
        } else {
            -magnitude
        }
    };
    let mut best: Option<(i64, (u32, u32))> = None;
    for (index, word) in raw.chunks_exact(2).enumerate() {
        let bits = u32::from(u16::from_le_bytes(word.try_into().unwrap())) << 16;
        need(
            f32::from_bits(bits).is_finite(),
            "finite original fullV word",
        )?;
        let rank = key(bits);
        if best.is_none_or(|(prior, _)| rank > prior) {
            best = Some((rank, (index as u32, bits)));
        }
    }
    best.map(|(_, selection)| selection)
        .ok_or_else(|| "nonempty original fullV".into())
}
/// Exact interrupted partition: the completed declared prefix plus exactly the
/// one armed observed call, each proposal once, and no overlap.
fn interrupted_scope(
    scope: &[(usize, usize)],
    active: (usize, usize),
    proposals: &[(usize, usize)],
) -> Proof<BTreeSet<(usize, usize)>> {
    let declared = scope.iter().copied().collect::<BTreeSet<_>>();
    let mut requested = proposals.iter().copied().collect::<BTreeSet<_>>();
    need(
        proposals.len() == requested.len()
            && requested.len() == scope.len() + 1
            && requested.remove(&active),
        "interrupted closure declares the completed proposals plus exactly the armed call",
    )?;
    need(
        declared == requested && !declared.contains(&active),
        "completed prefix and armed call are disjoint declared proposals",
    )?;
    Ok(declared)
}
#[derive(Default)]
struct Progress {
    words: usize,
    mismatches: usize,
    first: Option<Value>,
}
pub(super) struct Expected {
    root: PathBuf,
    reference: Option<Arc<Reference>>,
    natural: bool,
    natural_pin: Option<(u64, String)>,
    compositional: bool,
    armed: bool,
    repeat: usize,
    call: usize,
    progress: BTreeMap<usize, Progress>,
    records: Vec<usize>,
    scope: Vec<(usize, usize)>,
    scope_records: Vec<usize>,
    words: usize,
    mismatches: usize,
    positions: usize,
    scope_words: usize,
    scope_mismatches: usize,
    scope_positions: usize,
    initial_zero_words: usize,
    initial_total_words: usize,
    initial_prior: BTreeMap<(StateSlot, usize), (usize, usize)>,
    initial_order: Vec<(StateSlot, usize)>,
}
impl Expected {
    pub(super) fn new(root: PathBuf) -> Proof<Self> {
        need(root.is_absolute(), "explicit original expected directory")?;
        Ok(Self {
            root,
            reference: None,
            natural: false,
            natural_pin: None,
            compositional: false,
            armed: false,
            repeat: 0,
            call: 0,
            progress: BTreeMap::new(),
            records: Vec::new(),
            scope: Vec::new(),
            scope_records: Vec::new(),
            words: 0,
            mismatches: 0,
            positions: 0,
            scope_words: 0,
            scope_mismatches: 0,
            scope_positions: 0,
            initial_zero_words: 0,
            initial_total_words: 0,
            initial_prior: BTreeMap::new(),
            initial_order: Vec::new(),
        })
    }
    #[allow(dead_code)]
    pub(super) fn new_natural(root: PathBuf, pin: Option<(u64, String)>) -> Proof<Self> {
        need(root.is_absolute(), "explicit natural expected directory")?;
        need(
            pin.as_ref()
                .is_some_and(|(bytes, sha)| *bytes > 0 && *bytes <= 1024 * 1024 && sha.len() == 64),
            "fresh pinned natural accepted reference required",
        )?;
        Ok(Self {
            root,
            reference: None,
            natural: true,
            natural_pin: pin,
            compositional: false,
            armed: false,
            repeat: 0,
            call: 0,
            progress: BTreeMap::new(),
            records: Vec::new(),
            scope: Vec::new(),
            scope_records: Vec::new(),
            words: 0,
            mismatches: 0,
            positions: 0,
            scope_words: 0,
            scope_mismatches: 0,
            scope_positions: 0,
            initial_zero_words: 0,
            initial_total_words: 0,
            initial_prior: BTreeMap::new(),
            initial_order: Vec::new(),
        })
    }
    /// Held teacher receipt as a compositional natural adapter. No new original
    /// call is made; call0 is the unchanged seed and call1 is teacher-forced, so
    /// the runtime must refuse natural feedback with the exact join evidence.
    #[allow(dead_code)]
    pub(super) fn new_compositional(root: PathBuf) -> Proof<Self> {
        let mut expected = Self::new(root)?;
        expected.natural = true;
        expected.compositional = true;
        Ok(expected)
    }
    fn open_reference(&mut self) -> Proof<()> {
        if self.reference.is_none() {
            self.reference = Some(Arc::new(if self.natural {
                match &self.natural_pin {
                    Some(pin) => Reference::open_natural(&self.root, (pin.0, pin.1.as_str()))?,
                    None => Reference::open_compositional(&self.root)?,
                }
            } else {
                Reference::open(&self.root)?
            }));
        }
        Ok(())
    }
    /// One immutable reference shared by the preflight/main/late/replay/final
    /// comparators; only their own comparison state is separate.
    pub(super) fn new_shared(reference: &Arc<Reference>, natural: bool) -> Proof<Self> {
        need(
            natural && reference.natural,
            "shared natural reference cohort",
        )?;
        let mut expected = Self::new(PathBuf::from("/shared-reference"))?;
        expected.natural = true;
        expected.reference = Some(reference.clone());
        Ok(expected)
    }
    /// Bind one observed proposal before any device/resource effect. Callers
    /// that need the held reference open first use `begin`; the split exists so
    /// the source-only control can prove a binding cannot be published from a
    /// proposed frontier without the held call, and cannot be re-bound.
    #[allow(dead_code)]
    pub(super) fn bind(&mut self, repeat: usize, call: usize) -> Proof<()> {
        need(
            repeat < 2 && call < 2 && !self.armed && self.progress.is_empty(),
            "original expected session/call boundary",
        )?;
        need(
            !self.scope.contains(&(repeat, call)),
            "declared original proposal bound once",
        )?;
        self.repeat = repeat;
        self.call = call;
        self.armed = true;
        if call == 0 {
            self.initial_prior.clear();
            self.initial_order.clear();
            self.initial_zero_words = 0;
        }
        self.scope.push((repeat, call));
        Ok(())
    }
    /// Every observed proposal binds its repeat/call before any effect, and the
    /// independent fullV selection is available before the producer runs.
    pub(super) fn begin(&mut self, repeat: usize, call: usize) -> Proof<()> {
        self.open_reference()?;
        self.bind(repeat, call)
    }
    /// Actual mismatches accumulated for the currently armed call, if any.
    #[allow(dead_code)]
    pub(super) fn active_mismatches(&self) -> Proof<usize> {
        need(self.armed, "no active original proposal")?;
        Ok(self
            .progress
            .values()
            .map(|progress| progress.mismatches)
            .sum())
    }
    #[allow(dead_code)]
    pub(super) fn is_compositional(&self) -> bool {
        self.compositional
    }
    #[allow(dead_code)]
    pub(super) fn natural_available(&mut self) -> Proof<bool> {
        self.open_reference()?;
        Ok(self.reference.as_ref().is_some_and(|r| !r.compositional))
    }
    #[allow(dead_code)]
    pub(super) fn join_evidence(&mut self, actual: Option<[[(u32, u32); 2]; 2]>) -> Proof<Value> {
        self.open_reference()?;
        self.reference
            .as_ref()
            .ok_or("natural reference absent".into())
            .and_then(|r| r.join_evidence(actual))
    }
    /// Ordinary observer=None probes are packet-only: the held reference must
    /// still be byte-identical, no comparison may be left open, and no completed
    /// record may exist. Then release the declared scope without any claim.
    #[allow(dead_code)]
    pub(super) fn close_ordinary(&mut self) -> Proof<()> {
        // The armed/claimed boundary is checked before the held reference so an
        // ordinary caller cannot close a claimed proposal by construction.
        need(
            !self.armed && self.progress.is_empty(),
            "ordinary probes close only between packet-only proposals",
        )?;
        need(
            self.scope_records.is_empty()
                && self.scope_words == 0
                && self.scope_mismatches == 0
                && self.scope_positions == 0,
            "no ordinary expected comparison is claimed",
        )?;
        let r = self
            .reference
            .as_ref()
            .ok_or("ordinary probe without held reference")?;
        for (repeat, call) in &self.scope {
            need(
                r.specs
                    .iter()
                    .any(|s| s.repeat == *repeat && s.call == *call),
                "ordinary packet-only proposal is a declared original call",
            )?;
        }
        r.verify_lease()?;
        self.scope.clear();
        Ok(())
    }
    /// Post-comparison held verification of the whole accepted corpus and the
    /// exact completed cohort; no retirement is invented.
    #[allow(dead_code)]
    pub(super) fn verify_held(&self) -> Proof<()> {
        let r = self
            .reference
            .as_ref()
            .ok_or("held original reference absent")?;
        r.verify_lease()?;
        need(
            self.mismatches == 0
                && self.records.len() == 1036
                && self.words == 240_896_000
                && self.positions == 12
                && self.scope_records.is_empty(),
            "all1040 held records exact after final ordinary read",
        )
    }
    fn compare(&mut self, frame: QwenSourceFrame, actual: &[u8]) -> Proof<()> {
        need(
            self.armed
                && frame.position == self.call
                && actual.len().is_multiple_of(4)
                && !actual.is_empty()
                && frame.offset_words + actual.len() / 4 <= frame.total_words,
            "actual completed frame geometry/call binding",
        )?;
        if let QwenSourceFrameKind::State {
            slot,
            plane,
            prior: true,
        } = frame.kind
            && self.call == 0
        {
            need(
                actual.iter().all(|b| *b == 0),
                "actual original initial state is positive zero",
            )?;
            let words = actual.len() / 4;
            let key = (slot, plane);
            let entry = self.initial_prior.entry(key).or_insert_with(|| {
                self.initial_order.push(key);
                (0, frame.total_words)
            });
            need(
                entry.1 == frame.total_words && entry.0 == frame.offset_words,
                "initial prior plane ordered contiguous chunks exactly once",
            )?;
            entry.0 = entry
                .0
                .checked_add(words)
                .ok_or("initial prior chunk sum")?;
            self.initial_zero_words += words;
            return Ok(());
        }
        // Nothing in this object is called from a GPU operand/parameter path.
        let reference = self
            .reference
            .as_ref()
            .ok_or("expected bound before producer")?;
        let (layer, label, kv) = match frame.kind {
            QwenSourceFrameKind::Value { layer, stage } => match stage {
                "embedding" => (None, Label::Embedding, false),
                "layer_output" => (Some(layer), Label::LayerOutput, false),
                "final_norm" => (None, Label::Normalized, false),
                "logits" => (None, Label::Logits, false),
                _ => return Err("unknown original producer boundary".into()),
            },
            QwenSourceFrameKind::State { slot, plane, prior } => {
                let (layer, label, kv) = match (slot, prior) {
                    (StateSlot::Conv(d), false) => (d.0, Label::AfterConv4, false),
                    (StateSlot::Conv(d), true) => (d.0, Label::BeforeConv4, false),
                    (StateSlot::Recurrent(d), false) => (d.0, Label::AfterRecurrent, false),
                    (StateSlot::Recurrent(d), true) => (d.0, Label::BeforeRecurrent, false),
                    (StateSlot::Kv(d), false) => (
                        d.0,
                        if plane == 0 {
                            Label::AfterKey
                        } else {
                            Label::AfterValue
                        },
                        true,
                    ),
                    (StateSlot::Kv(d), true) => (
                        d.0,
                        if plane == 0 {
                            Label::BeforeKey
                        } else {
                            Label::BeforeValue
                        },
                        true,
                    ),
                };
                need(
                    plane == 0 || kv && plane == 1,
                    "original state component index",
                )?;
                (Some(layer), label, kv)
            }
        };
        let index = reference
            .specs
            .iter()
            .position(|s| {
                s.repeat == self.repeat
                    && s.call == self.call
                    && s.layer == layer
                    && s.label == label
            })
            .ok_or("actual producer not in accepted complete cohort")?;
        let spec = &reference.specs[index];
        need(
            spec.elements() == frame.total_words && spec.codec != Codec::I64,
            "complete original producer extent",
        )?;
        let expected = reference.record_bytes(index)?;
        let progress = self.progress.entry(index).or_default();
        need(
            progress.words == frame.offset_words,
            "original producer chunk order/no duplicate",
        )?;
        let active = if kv { frame.total_words / 1024 } else { 0 };
        for (j, word) in actual.chunks_exact(4).enumerate() {
            let at = frame.offset_words + j;
            let original = if kv {
                let token = at / 1024;
                let head = (at / 256) % 4;
                let lane = at % 256;
                head * active * 256 + token * 256 + lane
            } else {
                at
            };
            let actual = u32::from_le_bytes(word.try_into().unwrap());
            let expected = match spec.codec {
                Codec::Bf16 => {
                    u32::from(u16::from_le_bytes(
                        expected[original * 2..original * 2 + 2].try_into().unwrap(),
                    )) << 16
                }
                Codec::F32 => {
                    u32::from_le_bytes(expected[original * 4..original * 4 + 4].try_into().unwrap())
                }
                Codec::I64 => return Err("position record is not a float producer".into()),
            };
            if actual != expected {
                progress.mismatches += 1;
                if progress.first.is_none() {
                    progress.first = Some(
                        json!({"actual_index":at,"original_index":original,"actual_bits":format!("{actual:08x}"),
                    "expected_bits":format!("{expected:08x}")}),
                    );
                }
            }
        }
        progress.words += actual.len() / 4;
        if self.compositional && self.call == 1 {
            if let Some(p) = self.progress.get(&index)
                && p.mismatches > 0
            {
                let held = reference.teachers[self.repeat][1];
                let required = reference.selections[self.repeat][0].0;
                let label_name = label.name();
                return Err(format!(
                    "held natural join unavailable: teacher-forced original call1 input {held} differs from independent natural feedback input {required} at {label_name}; no accepted natural call1 record (receipt natural_feedback_qualified=false)"
                )
                .into());
            }
        }
        Ok(())
    }
    pub(super) fn finish(&mut self) -> Proof<Value> {
        let before_records = self.records.len();
        let reference = self
            .reference
            .as_ref()
            .ok_or("expected only opens after real computation")?;
        let mut count = 0;
        let (mut words, mut mismatches, mut positions) = (0, 0, 0);
        for (index, spec) in reference.specs.iter().enumerate() {
            if spec.repeat != self.repeat || spec.call != self.call {
                continue;
            }
            count += 1;
            if spec.codec == Codec::I64 {
                let actual = reference.record_bytes(index)?;
                need(
                    actual
                        .chunks_exact(8)
                        .all(|b| i64::from_le_bytes(b.try_into().unwrap()) == self.call as i64),
                    "three original current positions",
                )?;
                positions += 3;
                continue;
            }
            let p = self
                .progress
                .get(&index)
                .ok_or("missing actual full64 producer/state record")?;
            need(
                p.words == spec.elements(),
                "incomplete actual producer chunks",
            )?;
            words += p.words;
            mismatches += p.mismatches;
            self.records.push(index);
        }
        need(
            count == if self.call == 0 { 196 } else { 324 },
            "complete original proposal record count",
        )?;
        if self.call == 0 {
            self.verify_initial_prior()?;
        }
        need(self.armed, "original proposal completion bound to begin")?;
        self.armed = false;
        self.progress.clear();
        self.words += words;
        self.mismatches += mismatches;
        self.positions += positions;
        self.scope_records
            .extend_from_slice(&self.records[before_records..]);
        self.scope_words += words;
        self.scope_mismatches += mismatches;
        self.scope_positions += positions;
        Ok(
            json!({"repeat":self.repeat,"call":self.call,"records":count,"float_words":words,
            "positions":positions,"mismatches":mismatches,"exact":mismatches==0}),
        )
    }
    /// Exact interrupted-prefix closure: declared proposals must match, every
    /// completed record is an explicit member of that scope, and the pending
    /// prefix is bounded inside the declared extent. No completeness claim.
    #[allow(dead_code)]
    pub(super) fn finish_partial(&mut self, proposals: &[(usize, usize)]) -> Proof<Value> {
        let r = self
            .reference
            .as_ref()
            .ok_or("original comparison never opened")?;
        r.verify_lease()?;
        need(
            self.armed,
            "interrupted closure requires the one armed caller-owned call",
        )?;
        let active = (self.repeat, self.call);
        interrupted_scope(&self.scope, active, proposals)?;
        let scope = scope_totals(&r.specs, proposals)?;
        let mut seen = BTreeSet::new();
        for index in &self.scope_records {
            let spec = r
                .specs
                .get(*index)
                .ok_or("interrupted record in complete cohort")?;
            need(
                spec.codec != Codec::I64 && scope.required.contains(index) && seen.insert(*index),
                "interrupted completed record is in declared scope exactly once",
            )?;
        }
        for (index, spec) in r.specs.iter().enumerate() {
            if spec.codec != Codec::I64 || !scope.declared.contains(&(spec.repeat, spec.call)) {
                continue;
            }
            let actual = r.record_bytes(index)?;
            need(
                actual
                    .chunks_exact(8)
                    .all(|b| i64::from_le_bytes(b.try_into().unwrap()) == spec.call as i64),
                "declared interrupted original positions",
            )?;
        }
        // Pending prefix: every index is owned by the declared scope, disjoint
        // from the completed prefix, and carries no mismatch.
        need(
            !self.progress.is_empty(),
            "interrupted closure needs the actually observed pending prefix",
        )?;
        let mut pending = BTreeSet::new();
        let mut pending_layers = BTreeSet::new();
        for (index, progress) in &self.progress {
            let spec = r
                .specs
                .get(*index)
                .ok_or("interrupted pending record in complete cohort")?;
            need(
                spec.codec != Codec::I64
                    && scope.required.contains(index)
                    && pending.insert(*index)
                    && !seen.contains(index)
                    && progress.mismatches == 0
                    && progress.words > 0
                    && progress.words <= spec.elements()
                    && spec.repeat == active.0
                    && spec.call == active.1
                    && pending_layers.insert((spec.layer, spec.label)),
                "interrupted pending prefix owns disjoint declared extent of the armed call",
            )?;
        }
        // Exact full prefix domain hashes: every completed record is rehashed
        // from the held raw artifact, and every declared position record is
        // rehashed with its exact call value.
        for index in &self.scope_records {
            r.record_bytes(*index)?;
        }
        r.verify_lease()?;
        let pending_words: usize = self.progress.values().map(|p| p.words).sum();
        need(
            self.scope_words
                .checked_add(pending_words)
                .is_some_and(|n| n <= scope.words),
            "interrupted prefix inside declared original extent",
        )?;
        need(
            self.scope_mismatches == 0 && pending.len() <= scope.required.len(),
            "interrupted prefix holds no mismatch",
        )?;
        let value = json!({"declared_proposals":scope.declared.iter().map(|&(repeat,call)|json!([repeat,call])).collect::<Vec<_>>(),
            "required_float_records":scope.floats,"completed_float_records":seen.len(),
            "required_float_words":scope.words,"completed_float_words":self.scope_words,
            "pending_float_words":pending_words,
            "unavailable_float_words":scope.words-self.scope_words-pending_words,
            "observed":{"repeat":active.0,"call":active.1,"records":pending.len(),
                "float_words":pending_words,"positions":scope.positions},
            "positions":scope.positions,
            "mismatches":self.scope_mismatches,"interrupted":true,
            "exact_completed_prefix":self.scope_mismatches==0});
        self.armed = false;
        self.progress.clear();
        self.scope.clear();
        self.scope_records.clear();
        self.scope_words = 0;
        self.scope_positions = 0;
        self.scope_mismatches = 0;
        Ok(value)
    }
    #[allow(dead_code)]
    pub(super) fn finish_scoped(&mut self, proposals: &[(usize, usize)]) -> Proof<Value> {
        let r = self
            .reference
            .as_ref()
            .ok_or("original comparison never opened")?;
        need(
            !self.armed && self.progress.is_empty(),
            "scope closes only between completed proposals",
        )?;
        let declared = self.scope.iter().copied().collect::<BTreeSet<_>>();
        let requested = proposals.iter().copied().collect::<BTreeSet<_>>();
        need(
            proposals.len() == requested.len() && !declared.is_empty() && declared == requested,
            "exact expected-only scope closure equals declared proposals",
        )?;
        r.verify_lease()?;
        let scope = scope_totals(&r.specs, proposals)?;
        for (index, spec) in r.specs.iter().enumerate() {
            if spec.codec != Codec::I64 || !scope.declared.contains(&(spec.repeat, spec.call)) {
                continue;
            }
            let actual = r.record_bytes(index)?;
            need(
                actual
                    .chunks_exact(8)
                    .all(|b| i64::from_le_bytes(b.try_into().unwrap()) == spec.call as i64),
                "declared original current positions",
            )?;
        }
        verify_scope(
            &r.specs,
            &scope,
            &self.scope_records,
            self.progress.len(),
            self.scope_words,
            self.scope_positions,
            self.scope_mismatches,
        )?;
        r.verify_lease()?;
        let value = json!({"declared_proposals":scope.declared.iter().map(|&(repeat,call)|json!([repeat,call])).collect::<Vec<_>>(),"records":scope.total,"float_records":scope.floats,"float_words":scope.words,"positions":scope.positions,"mismatches":self.scope_mismatches,"exact":self.scope_mismatches==0});
        self.scope.clear();
        self.scope_records.clear();
        self.scope_words = 0;
        self.scope_positions = 0;
        self.scope_mismatches = 0;
        Ok(value)
    }
    #[allow(dead_code)]
    pub(super) fn in_progress(&self) -> (usize, usize) {
        (
            self.progress.len(),
            self.progress.values().map(|p| p.words).sum(),
        )
    }
    fn verify_initial_prior(&mut self) -> Proof<()> {
        need(
            self.initial_prior.len() == INITIAL_PRIOR_PLANES
                && self.initial_order.len() == INITIAL_PRIOR_PLANES,
            "all96 initial prior planes once",
        )?;
        let mut expected = Vec::with_capacity(INITIAL_PRIOR_PLANES);
        let mut words = 0usize;
        // Actual engine order derives from the ordered StateSlot set: every Conv
        // slot precedes every Recurrent slot, regardless of model layer order.
        for layer in 0..64 {
            if layer % 4 != 3 {
                expected.push((StateSlot::Conv(ConvDomain(layer)), 0));
            }
        }
        for layer in 0..64 {
            if layer % 4 != 3 {
                expected.push((StateSlot::Recurrent(RecurrentDomain(layer)), 0));
            }
        }
        need(
            self.initial_order == expected,
            "initial prior planes ordered by actual StateSlot (48 Conv then 48 Recurrent)",
        )?;
        for (key, (next, total)) in &self.initial_prior {
            let plane = match key.0 {
                StateSlot::Conv(_) => CONV_WORDS,
                StateSlot::Recurrent(_) => RECURRENT_WORDS,
                StateSlot::Kv(_) => return Err("unexpected initial KV prior plane".into()),
            };
            need(
                *total == plane && next == total,
                format!("complete initial prior plane {key:?}"),
            )?;
            words = words.checked_add(*total).ok_or("initial prior word sum")?;
        }
        need(
            words == INITIAL_PRIOR_WORDS && self.initial_zero_words == INITIAL_PRIOR_WORDS,
            "all39714816 initial prior words positive zero",
        )?;
        self.initial_total_words = self
            .initial_total_words
            .checked_add(words)
            .ok_or("initial prior total word sum")?;
        Ok(())
    }
    #[allow(dead_code)]
    pub(super) fn selection(&self, repeat: usize, call: usize) -> Proof<(u32, u32)> {
        self.reference
            .as_ref()
            .ok_or("expected opened only after actual producer")?
            .full_v(repeat, call)
    }
    #[allow(dead_code)]
    pub(super) fn selection_open(&mut self, repeat: usize, call: usize) -> Proof<(u32, u32)> {
        self.open_reference()?;
        self.selection(repeat, call)
    }
    /// Pre-read geometry for the one full-root ordinary readback: the natural
    /// two-position capacity is fully committed, the held manifest declares its
    /// actual 112 cache domains and the resulting root extent is the measured
    /// 159_121_408 B. Checked before `read_domains` allocates, never after.
    #[allow(dead_code)]
    pub(super) fn preflight_domains(&self, call: usize, frontier: u64, capacity: u64) -> Proof<()> {
        let reference = self.reference.as_ref().ok_or("committed expected absent")?;
        let words = INITIAL_PRIOR_WORDS + 16 * 2 * 1024 * (call + 1);
        need(
            call < 2
                && capacity == 2
                && frontier == (call + 1) as u64
                && reference.cache_domains == 112
                && words as u64 * 4 == 159_121_408,
            "expected frontier/capacity/112-domain geometry before full-root readback",
        )
    }
    #[allow(dead_code)]
    pub(super) fn compare_committed(
        &self,
        repeat: usize,
        call: usize,
        domains: &[(StateSlot, Vec<f32>)],
    ) -> Proof<Value> {
        need(
            repeat < 2 && call < 2 && domains.len() == 112,
            "all112 committed state domains for natural call",
        )?;
        let reference = self.reference.as_ref().ok_or("committed expected absent")?;
        reference.verify_lease()?;
        let mut seen = BTreeSet::new();
        let (mut words, mut mismatches) = (0usize, 0usize);
        let mut first = None;
        for (slot, actual) in domains {
            need(seen.insert(*slot), "committed state domain once")?;
            let (label, expected) = match slot {
                StateSlot::Conv(d) => {
                    let spec = reference
                        .specs
                        .iter()
                        .position(|s| {
                            s.repeat == repeat
                                && s.call == call
                                && s.layer == Some(d.0)
                                && s.label == Label::AfterConv4
                        })
                        .ok_or("committed conv4 record absent")?;
                    let spec = &reference.specs[spec];
                    ("conv4", expand_bf16(&reference.record_bytes_for(spec)?)?)
                }
                StateSlot::Recurrent(d) => {
                    let spec = reference
                        .specs
                        .iter()
                        .position(|s| {
                            s.repeat == repeat
                                && s.call == call
                                && s.layer == Some(d.0)
                                && s.label == Label::AfterRecurrent
                        })
                        .ok_or("committed recurrent record absent")?;
                    let spec = &reference.specs[spec];
                    ("recurrent", f32_words(&reference.record_bytes_for(spec)?)?)
                }
                StateSlot::Kv(d) => {
                    let active = call + 1;
                    let key = reference
                        .specs
                        .iter()
                        .position(|s| {
                            s.repeat == repeat
                                && s.call == call
                                && s.layer == Some(d.0)
                                && s.label == Label::AfterKey
                        })
                        .ok_or("committed key record absent")?;
                    let value = reference
                        .specs
                        .iter()
                        .position(|s| {
                            s.repeat == repeat
                                && s.call == call
                                && s.layer == Some(d.0)
                                && s.label == Label::AfterValue
                        })
                        .ok_or("committed value record absent")?;
                    let key = expand_bf16(&reference.record_bytes(key)?)?;
                    let value = expand_bf16(&reference.record_bytes(value)?)?;
                    need(
                        key.len() == value.len() && key.len() == active * 1024,
                        "committed KV active frontier extent",
                    )?;
                    let mut flat = Vec::with_capacity(key.len() * 2);
                    for row in [&key, &value] {
                        for at in 0..active * 1024 {
                            let original =
                                ((at / 256) % 4) * active * 256 + (at / 1024) * 256 + at % 256;
                            flat.push(row[original]);
                        }
                    }
                    ("kv", flat)
                }
            };
            need(
                actual.len() == expected.len() && actual.iter().all(|v| v.is_finite()),
                format!("committed {label} domain extent/finite"),
            )?;
            words += actual.len();
            for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                if a.to_bits() != e.to_bits() {
                    mismatches += 1;
                    if first.is_none() {
                        first = Some(json!({"slot":format!("{slot:?}"),"index":index,
                            "actual_bits":format!("{:08x}",a.to_bits()),
                            "expected_bits":format!("{:08x}",e.to_bits())}));
                    }
                }
            }
        }
        need(
            mismatches == 0 && words == INITIAL_PRIOR_WORDS + 16 * 2 * 1024 * (call + 1),
            "all112 committed domains exactly original natural state/root extent",
        )?;
        Ok(
            json!({"repeat":repeat,"call":call,"domains":domains.len(),"words":words,
            "mismatches":mismatches,"first":first}),
        )
    }
    #[allow(dead_code)]
    pub(super) fn after_natural(&mut self, actual: [[(u32, u32); 2]; 2]) -> Proof<Value> {
        self.after()?;
        let full4 = self.finish_scoped(&[(0, 0), (0, 1), (1, 0), (1, 1)])?;
        let reference = self.reference.as_ref().ok_or("natural reference absent")?;
        let join = reference.join_evidence(Some(actual))?;
        Ok(json!({"full4":true,"full4_scope":full4,"join":join,
            "selections":reference.selections,
            "natural_source_events_per_call":SOURCE_EVENTS / 4}))
    }
    pub(super) fn after(&self) -> Proof<()> {
        let r = self
            .reference
            .as_ref()
            .ok_or("original comparison never opened")?;
        r.manifest_owner.verify()?;
        r.raw.verify()?;
        need(
            self.words == 240896000
                && self.positions == 12
                && self.mismatches == 0
                && self.records.len() == 1036,
            "all1040 records/240896000float+12position words exact",
        )
    }
    pub(super) fn summary(&self) -> Value {
        let records = self
            .reference
            .as_ref()
            .map(|r| {
                self.records
                    .iter()
                    .filter_map(|index| {
                        r.specs.get(*index).map(|s| {
                            json!({"repeat":s.repeat,"call":s.call,"original_record_index":index,
                            "layer":s.layer,"label":s.label.name(),"words":s.elements(),
                            "mismatches":0})
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        json!({"records":records,"float_words":self.words,"positions":self.positions,
            "mismatches":self.mismatches,"initial_zero_words":self.initial_total_words,
            "initial_prior_planes":self.initial_order.len(),
            "natural":self.natural,
            "expected_opened_before_actual_producer":self.reference.is_some(),
            "compositional_original_adapter":self.compositional})
    }
}
impl QwenSourceObserver for Expected {
    fn completed(&mut self, frame: QwenSourceFrame, actual: &[u8]) -> zerogravity::Result<()> {
        self.compare(frame, actual)
            .map_err(|e| zerogravity::Error::State(format!("original full64 comparison: {e}")))
    }
}
#[allow(dead_code)]
fn expand_bf16(raw: &[u8]) -> Proof<Vec<f32>> {
    need(raw.len().is_multiple_of(2), "complete BF16 state words")?;
    Ok(raw
        .chunks_exact(2)
        .map(|w| f32::from_bits(u32::from(u16::from_le_bytes(w.try_into().unwrap())) << 16))
        .collect())
}
#[allow(dead_code)]
fn f32_words(raw: &[u8]) -> Proof<Vec<f32>> {
    need(raw.len().is_multiple_of(4), "complete F32 state words")?;
    Ok(raw
        .chunks_exact(4)
        .map(|w| f32::from_bits(u32::from_le_bytes(w.try_into().unwrap())))
        .collect())
}
/// Source-only fixture reports for the Python semantic dispatch. These exercise
/// the same serde shapes and typed selection tuples as the runtime builders,
/// but carry no model/device/GPU evidence and can never be a capture receipt.
#[allow(dead_code)]
pub(super) fn natural_report_fixtures() -> Value {
    let sel = |id: u32, bits: u32| json!([id, bits]);
    // Same seed input on both independent cohorts: identical selection tuples.
    let seed0 = (17u32, 0x3f80_0000u32);
    let seed1 = seed0;
    let next0 = (23u32, 0x4000_0000u32);
    let next1 = next0;
    let pin = json!({"bytes":20000,"sha256":"a".repeat(64)});
    let inv = json!({"root_bytes":159_121_408u64,"readback_window_bytes":3*1024*1024u64,
        "domain_metadata_bytes":112u64*(16+3*8+8+2*8),
        "reference_bytes":2_000_000u64,"manifest_input_bytes":100_000u64,
        "manifest_tree_bytes":4_000_000u64,"admission_bytes":8_000_000u64,
        "observed_bytes":10_000_000u64,"snapshot_bytes":300_000_000u64,
        "committed_bytes":200_000_000u64,"report_bytes":3_000_000u64,
        "peak_bytes":300_000_000u64,"host_bytes":384*1024*1024u64});
    let closure = |call: usize| {
        json!({"declared_proposals":[[0,call]],"records":196,
        "float_records":195,"float_words":1000,"positions":3,"mismatches":0,"exact":true})
    };
    let receipt = |fault: bool, wrapper: &str| {
        json!({
        "wrapper":wrapper,"fault":fault,"source":"source","lease":"lease",
        "service_generation":1,"lease_bytes":12,"aliases_inside_wrapper":1,
        "candidate_owners_before_drop":1,"model_owners_before_drop":1,
        "candidate_workspace_target_aliases":0,"candidate_retired":true,"release_hits":1,
        "native_retirement_attempts":1,"reader_hits_before_native_retirement":1,
        "native_modules_before":4,"native_code_bytes_before":1000,
        "native_host_bytes_before":64*1024*1024,"native_device_bytes_before":1000,
        "native_admission_retained_after":fault,"native_modules_after":if fault {4} else {0},
        "original_service_retained_after":fault,
        "quarantined_handles_after":if fault {2} else {0}})
    };
    let host = |fault: bool| {
        if fault {
            2 * 1024 * 1024 * 1024u64
                + 384 * 1024 * 1024
                + 64 * 1024 * 1024
                + 4 * 1024 * 1024
                + 64 * 1024 * 1024
        } else {
            2 * 1024 * 1024 * 1024u64 + 384 * 1024 * 1024
        }
    };
    let mut modes = serde_json::Map::new();
    for mode in [
        "healthy",
        "cached-unknown",
        "session-known",
        "session-release-fault",
        "snapshot-known",
        "snapshot-release-fault",
    ] {
        let fault = mode.ends_with("fault") || mode == "cached-unknown";
        let final_mode =
            mode.contains("known") || mode.contains("fault") && mode.contains("session");
        let final_mode = final_mode || mode.starts_with("snapshot");
        let actual = if mode == "cached-unknown" {
            json!({"first":{"call":0,"selected_id":seed0.0,"max_bits":seed0.1},
                "selected_output_tuples":{"seed_input":11,"seed":sel(seed0.0,seed0.1),
                    "next":Value::Null,"distinct_from_seed_input":true},
                "cached_unknown_root_unchanged":true,"published_cached_roots":0,
                "cached_expected_records":4,"cached_expected_words":128,
                "cached_expected_closure":{"exact_completed_prefix":true,"interrupted":true,
                    "mismatches":0,"unavailable_float_words":64,
                    "observed":{"repeat":0,"call":1,"records":4,"float_words":128,"positions":3}}})
        } else if final_mode {
            let wrapper = if mode.starts_with("snapshot") {
                "Snapshot"
            } else {
                "Session"
            };
            json!({"first":{"call":0,"selected_id":seed0.0,"max_bits":seed0.1},
                "selection":sel(seed0.0,seed0.1),"scope_closure":closure(0),
                "selected_output_tuples":{"seed_input":11,"seed":sel(seed0.0,seed0.1),
                    "next":sel(seed0.0,seed0.1),"distinct_from_seed_input":true},
                "original_final_reader":receipt(fault, wrapper)})
        } else {
            json!({"actual_full64_calls":9,"source_capacity":2,
                "diagnostic_committed_state_compared":true,"eos_text_policy_implemented":false,
                "ordinary_packet":{"seed":sel(seed0.0,seed0.1),"next":sel(next0.0,next0.1),
                    "seed_readback_bytes":16,"next_readback_bytes":16},
                "selected_output_tuples":{"seed_input":11,"seed":sel(seed0.0,seed0.1),
                    "next":sel(next0.0,next0.1),"distinct_from_seed_input":true},
                "first":{"call":0,"selected_id":seed0.0,"max_bits":seed0.1},
                "cached":{"call":0,"selected_id":next0.0,"max_bits":next0.1},
                "independent_peer":{"call":0,"selected_id":seed1.0,"max_bits":seed1.1},
                "independent_peer_cached":{"call":0,"selected_id":next1.0,"max_bits":next1.1},
                "first_selection":sel(seed0.0,seed0.1),"cached_selection":sel(next0.0,next0.1),
                "peer_selection":sel(seed1.0,seed1.1),"peer_cached_selection":sel(next1.0,next1.1),
                "late_cancel":{"scope":json!({"selected_id":next0.0,"max_bits":next0.1}),
                    "scope_closure":closure(1),"selection":sel(next0.0,next0.1)},
                "restore_replay":{"scope":json!({"selected_id":next0.0,"max_bits":next0.1}),
                    "scope_closure":closure(1),"selection":sel(next0.0,next0.1)},
                "ordinary_committed":{"repeat":0,"call":1,"domains":112,"words":39_780_352u64,
                    "mismatches":0,"first":Value::Null},
                "full4":{"full4":true,"natural_source_events_per_call":67,
                    "full4_scope":{"declared_proposals":[[0,0],[0,1],[1,0],[1,1]],
                        "records":1040,"float_records":1036,"float_words":240_896_000u64,
                        "positions":12,"mismatches":0,"exact":true},
                    "selections":[[sel(seed0.0,seed0.1),sel(next0.0,next0.1)],
                        [sel(seed1.0,seed1.1),sel(next1.0,next1.1)]],
                    "join":{"independent_original_natural_trajectory":true,
                        "actual_observed":true,"teacher_forcing_only":false,"newcalls":0}}})
        };
        let value = json!({"schema":"zg.qwen27b-natural-two-position-diagnostic.v2",
            "status":"PASS_DIAGNOSTIC_NATURAL_UNQUALIFIED","mode":mode,"failure":Value::Null,
            "retention_mode":if fault {"unknown"} else {"known"},
            "typed_final_retention_join":true,
            "reference_lease":json!({"owners":(0..5).map(|i| json!({
                "path":format!("/held/{i}"),"bytes":1000+i,"sha256":"b".repeat(64),
                "fd_identity":{"dev":1,"ino":i+2,"mtime":3,"mtime_nsec":4}}))
                .collect::<Vec<_>>(),
                "independent_artifact_pins":5,
                "retained_across_validation_and_report":true,
                "mutable_path_reopen_as_replacement":false}),
            "retained_source_context_bytes":if fault {4*1024*1024u64} else {0},
            "retained_source_metadata_bytes":if fault {64*1024*1024u64} else {0},
            "retained_native_inventory":if fault {json!("inventory")} else {Value::Null},
            "released_host_assertion":if fault {Value::Null} else {json!(true)},
            "actual":actual,"source_close_refused":fault,
            "physical_reserved":if fault {16*1024*1024*1024u64} else {4*1024*1024*1024u64},
            "physical_fenced":fault,"host_reserved":host(fault),"host_fenced":false,
            "diagnostic_host_bytes":384*1024*1024u64,"diagnostic_inventory":inv.clone(),
            "natural_accepted_pin":pin.clone(),
            "mathematical_original_reference_compared":mode=="healthy",
            "ordinary_full_state_compared":mode=="healthy","full4_compared":mode=="healthy",
            "lifecycle_cases_qualified":false,
            "trained_native_natural_serving_or_performance_qualified":false});
        modes.insert(mode.to_string(), value);
    }
    Value::Object(modes)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_specs_keep_full4_and_rectangular_kv_mapping() {
        let s = specs();
        assert_eq!(s.len(), 1040);
        assert_eq!(
            s.iter()
                .filter(|s| s.codec != Codec::I64)
                .map(Spec::elements)
                .sum::<usize>(),
            240896000
        );
        for active in [1, 2] {
            let mut seen = vec![false; active * 1024];
            for at in 0..active * 1024 {
                let original = ((at / 256) % 4) * active * 256 + (at / 1024) * 256 + at % 256;
                assert!(!seen[original]);
                seen[original] = true;
            }
            assert!(seen.into_iter().all(|v| v));
        }
        assert!(
            s.iter()
                .filter(|s| matches!(s.label, Label::AfterConv4 | Label::BeforeConv4))
                .all(|s| s.elements() == 40960)
        );
    }
    #[test]
    fn source_only_fixtures_serialize_all_six_modes() {
        let fixtures = natural_report_fixtures();
        let modes = fixtures.as_object().unwrap();
        assert_eq!(modes.len(), 6);
        assert!(modes["healthy"]["actual"]["ordinary_packet"]["seed"][0] != 11);
        for mode in [
            "healthy",
            "cached-unknown",
            "session-known",
            "session-release-fault",
            "snapshot-known",
            "snapshot-release-fault",
        ] {
            let value = &modes[mode];
            assert_eq!(
                value["reference_lease"]["owners"].as_array().unwrap().len(),
                5
            );
            assert_eq!(value["reference_lease"]["independent_artifact_pins"], 5);
            assert_eq!(
                value["retention_mode"],
                if mode.ends_with("fault") || mode == "cached-unknown" {
                    "unknown"
                } else {
                    "known"
                }
            );
            assert_eq!(value["typed_final_retention_join"], true);
            if mode != "cached-unknown" {
                assert_eq!(value["actual"]["selected_output_tuples"]["seed_input"], 11);
                assert_ne!(value["actual"]["selected_output_tuples"]["seed"][0], 11);
            }
        }
        assert!(
            modes["healthy"]["actual"]["full4"]["selections"][0][0]
                == modes["healthy"]["actual"]["ordinary_packet"]["seed"]
        );
        if let Ok(path) = std::env::var("ZG_NATURAL_FIXTURES") {
            std::fs::write(path, serde_json::to_vec_pretty(&fixtures).unwrap()).unwrap();
        }
    }
    #[test]
    fn two_cohort_fixtures_are_identical_output_tuples() {
        let fixtures = natural_report_fixtures();
        let healthy = &fixtures["healthy"]["actual"];
        assert_eq!(
            healthy["ordinary_packet"]["seed"],
            healthy["peer_selection"]
        );
        assert_eq!(
            healthy["full4"]["selections"][0],
            healthy["full4"]["selections"][1]
        );
    }
    #[test]
    fn two_independent_seeds_complete_identical_lifecycles() {
        let (reference, _dir) = synthetic_reference();
        let mut values = Vec::new();
        for seed in 0..2 {
            let mut e = Expected::new_shared(&reference, true).unwrap();
            e.begin(seed, 0).unwrap();
            feed_initial(&mut e, 0);
            feed_records(&mut e, &reference, seed, 0);
            values.push(e.finish().unwrap());
        }
        for field in ["records", "float_words", "positions", "mismatches", "exact"] {
            assert_eq!(values[0][field], values[1][field]);
        }
    }
    #[test]
    fn pristine_lease_mutation_is_refused_after_lifecycle() {
        let (reference, dir) = synthetic_reference();
        reference.verify_lease().unwrap();
        let mut e = Expected::new_shared(&reference, true).unwrap();
        e.begin(0, 0).unwrap();
        feed_initial(&mut e, 0);
        feed_records(&mut e, &reference, 0, 0);
        e.finish().unwrap();
        reference.verify_lease().unwrap();
        // Mutate the retained raw artifact in place: the held descriptor's own
        // identity/digest must refuse; no path reopen can substitute authority.
        let raw = dir.join("raw.bin");
        let mut bytes = std::fs::read(&raw).unwrap();
        bytes[0] ^= 1;
        std::fs::write(&raw, &bytes).unwrap();
        assert!(reference.verify_lease().is_err());
        assert!(e.selection(0, 0).is_err());
    }
    #[test]
    fn pristine_natural_accepted_owner_mutation_is_refused() {
        let (reference, dir) = synthetic_reference();
        reference.verify_lease().unwrap();
        // The natural-only accepted receipt is an independent held owner too.
        std::fs::write(dir.join("accepted.json"), b"{ }").unwrap();
        assert!(reference.verify_lease().is_err());
    }
    fn synthetic_specs() -> Vec<Spec> {
        // Full 1040-record finite structure, tiny artificial extents: this is a
        // comparator/control fixture, never accepted model authority.
        specs()
            .into_iter()
            .map(|mut s| {
                match s.codec {
                    Codec::I64 => {
                        s.shape = [3, 1, 1, 0];
                        s.rank = 3;
                    }
                    _ => {
                        s.shape = [1, 2, 0, 0];
                        s.rank = 2;
                    }
                }
                s
            })
            .collect()
    }
    fn synthetic_reference() -> (Arc<Reference>, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "zg-qwen-natural-synthetic-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let specs = synthetic_specs();
        let mut raw = Vec::new();
        let mut records = Vec::with_capacity(specs.len());
        for s in &specs {
            let bytes = if s.codec == Codec::I64 {
                (0..3)
                    .flat_map(|_| (s.call as i64).to_le_bytes())
                    .collect::<Vec<_>>()
            } else {
                vec![0u8; s.bytes()]
            };
            records.push(RecordRow {
                offset: raw.len() as u64,
                sha: digest(&bytes),
            });
            raw.extend_from_slice(&bytes);
        }
        let raw_path = dir.join("raw.bin");
        std::fs::write(&raw_path, &raw).unwrap();
        let manifest_path = dir.join("manifest.json");
        std::fs::write(&manifest_path, b"{}").unwrap();
        let pin = |path: &Path, bytes: usize| {
            let data = std::fs::read(path).unwrap();
            assert_eq!(data.len(), bytes);
            json!({"path":path.display().to_string(),"bytes":bytes,"sha256":sha(&data)})
        };
        // Two further retained owners so this fixture exercises the exact
        // five-FD natural lease without a capture receipt.
        let mut owners = Vec::new();
        for name in ["invocation.json", "image-inspect.json"] {
            let path = dir.join(name);
            std::fs::write(&path, b"{}").unwrap();
            owners.push(Held::new(&dir, &pin(&path, 2), 1024).unwrap());
        }
        let owner = |kind: &str| {
            let path = dir.join(format!("{kind}.json"));
            std::fs::write(&path, b"{}").unwrap();
            Held::new(&dir, &pin(&path, 2), 1024).unwrap()
        };
        let accepted = owner("accepted");
        let mut owners = owners.into_iter();
        let reference = Arc::new(Reference {
            cache_domains: 112,
            records,
            manifest_input_bytes: 2,
            manifest_tree_bytes: 8,
            specs,
            manifest_owner: Held::new(&dir, &pin(&manifest_path, 2), 1024).unwrap(),
            raw: Held::new(&dir, &pin(&raw_path, raw.len()), raw.len() as u64).unwrap(),
            accepted: Some(accepted),
            invocation: owners.next(),
            image: owners.next(),
            admission_bytes: 0,
            natural: true,
            compositional: false,
            teachers: [[11, 11]; 2],
            selections: [[(0, 0); 2]; 2],
        });
        (reference, dir)
    }
    fn kind(s: &Spec) -> QwenSourceFrameKind {
        use zerogravity_gpu::QwenSourceFrameKind as K;
        match (s.layer, s.label) {
            (None, Label::Embedding) => K::Value {
                layer: 0,
                stage: "embedding",
            },
            (None, Label::Normalized) => K::Value {
                layer: 0,
                stage: "final_norm",
            },
            (None, Label::Logits) => K::Value {
                layer: 0,
                stage: "logits",
            },
            (Some(layer), Label::LayerOutput) => K::Value {
                layer,
                stage: "layer_output",
            },
            (Some(layer), label) => K::State {
                slot: match label {
                    Label::AfterConv4 | Label::BeforeConv4 => StateSlot::Conv(ConvDomain(layer)),
                    Label::AfterRecurrent | Label::BeforeRecurrent => {
                        StateSlot::Recurrent(RecurrentDomain(layer))
                    }
                    Label::AfterKey | Label::BeforeKey | Label::AfterValue | Label::BeforeValue => {
                        StateSlot::Kv(zerogravity::accelerator::program::KvDomain(layer))
                    }
                    _ => panic!("synthetic layer label"),
                },
                plane: usize::from(matches!(label, Label::AfterValue | Label::BeforeValue)),
                prior: matches!(
                    label,
                    Label::BeforeConv4
                        | Label::BeforeRecurrent
                        | Label::BeforeKey
                        | Label::BeforeValue
                ),
            },
            _ => panic!("synthetic kind"),
        }
    }
    fn feed_records(e: &mut Expected, reference: &Reference, repeat: usize, call: usize) {
        for (index, s) in reference.specs.iter().enumerate() {
            if s.repeat != repeat || s.call != call || s.codec == Codec::I64 {
                continue;
            }
            let raw = reference.record_bytes(index).unwrap();
            // Producer frames carry expanded 4-byte words; BF16 records are
            // widened exactly as the comparator expects.
            let bytes = if s.codec == Codec::Bf16 {
                expand_bf16(&raw)
                    .unwrap()
                    .iter()
                    .flat_map(|w| w.to_bits().to_le_bytes())
                    .collect::<Vec<u8>>()
            } else {
                raw
            };
            e.completed(
                QwenSourceFrame {
                    position: call,
                    kind: kind(s),
                    offset_words: 0,
                    total_words: s.elements(),
                },
                &bytes,
            )
            .unwrap();
        }
    }
    fn feed_initial(e: &mut Expected, call: usize) {
        let zero = vec![0u8; 4096 * 4];
        let plane = |e: &mut Expected, slot: StateSlot, total: usize| {
            let mut at = 0;
            while at < total {
                let words = (total - at).min(4096);
                e.completed(
                    QwenSourceFrame {
                        position: call,
                        kind: QwenSourceFrameKind::State {
                            slot,
                            plane: 0,
                            prior: true,
                        },
                        offset_words: at,
                        total_words: total,
                    },
                    &zero[..words * 4],
                )
                .unwrap();
                at += words;
            }
        };
        for layer in 0..64 {
            if layer % 4 != 3 {
                plane(&mut *e, StateSlot::Conv(ConvDomain(layer)), CONV_WORDS);
            }
        }
        for layer in 0..64 {
            if layer % 4 != 3 {
                plane(
                    &mut *e,
                    StateSlot::Recurrent(RecurrentDomain(layer)),
                    RECURRENT_WORDS,
                );
            }
        }
    }
    #[test]
    fn actual_completed_lifecycle_resets_seed_and_keeps_cumulative_words() {
        let (reference, _dir) = synthetic_reference();
        let mut e = Expected::new_shared(&reference, true).unwrap();
        e.begin(0, 0).unwrap();
        feed_initial(&mut e, 0);
        feed_records(&mut e, &reference, 0, 0);
        let first = e.finish().unwrap();
        assert_eq!(first["records"], 196);
        assert_eq!(e.initial_order.len(), 96);
        let first_words = e.words;
        e.begin(1, 0).unwrap();
        assert!(e.initial_order.is_empty() && e.initial_prior.is_empty());
        feed_initial(&mut e, 0);
        feed_records(&mut e, &reference, 1, 0);
        let second = e.finish().unwrap();
        assert_eq!(second["records"], 196);
        assert_eq!(e.words, first_words * 2);
        let closure = e.finish_scoped(&[(0, 0), (1, 0)]).unwrap();
        assert_eq!(closure["float_records"], 390);
        assert_eq!(closure["mismatches"], 0);
    }
    #[test]
    fn actual_completed_refuses_wrong_position_gap_duplicate_and_contamination() {
        let (reference, _dir) = synthetic_reference();
        let mut e = Expected::new_shared(&reference, true).unwrap();
        e.begin(0, 1).unwrap();
        let s = reference
            .specs
            .iter()
            .find(|s| s.repeat == 0 && s.call == 1 && s.codec != Codec::I64)
            .unwrap();
        let bytes = reference
            .record_bytes(reference.specs.iter().position(|x| x == s).unwrap())
            .unwrap();
        let frame = |position: usize, offset: usize| QwenSourceFrame {
            position,
            kind: kind(s),
            offset_words: offset,
            total_words: s.elements(),
        };
        assert!(e.completed(frame(0, 0), &bytes).is_err());
        assert!(e.completed(frame(1, 1), &bytes).is_err());
        e.completed(frame(1, 0), &bytes).unwrap();
        assert!(e.completed(frame(1, 0), &bytes).is_err());
    }
    #[test]
    fn actual_completed_refuses_missing_initial_prior_plane() {
        let (reference, _dir) = synthetic_reference();
        let mut e = Expected::new_shared(&reference, true).unwrap();
        e.begin(0, 0).unwrap();
        let zero = vec![0u8; 4096 * 4];
        // All planes except the first ordered Conv plane.
        let mut first = true;
        for layer in 0..64 {
            if layer % 4 != 3 {
                if first {
                    first = false;
                    continue;
                }
                let total = CONV_WORDS;
                let mut at = 0;
                while at < total {
                    let words = (total - at).min(4096);
                    e.completed(
                        QwenSourceFrame {
                            position: 0,
                            kind: QwenSourceFrameKind::State {
                                slot: StateSlot::Conv(ConvDomain(layer)),
                                plane: 0,
                                prior: true,
                            },
                            offset_words: at,
                            total_words: total,
                        },
                        &zero[..words * 4],
                    )
                    .unwrap();
                    at += words;
                }
            }
        }
        feed_records(&mut e, &reference, 0, 0);
        assert!(e.finish().is_err());
    }
    fn bf16(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    #[test]
    fn original_fullv_argmax_keeps_lowest_id_and_integer_signed_zero() {
        let expected = (7, 0x3f80_0000);
        assert!(packet_matches(7, 0x3f80_0000, expected));
        assert!(!packet_matches(8, 0x3f80_0000, expected));
        assert!(!packet_matches(7, 0x3f80_0001, expected));
        assert_eq!(
            bf16_argmax(&bf16(&[0x3f80, 0x4000, 0x4000])).unwrap(),
            (1, 0x4000_0000)
        );
        assert_eq!(
            bf16_argmax(&bf16(&[0x8000, 0x0000, 0xbf80])).unwrap(),
            (0, 0x8000_0000)
        );
        assert_eq!(
            bf16_argmax(&bf16(&[0x0000, 0x8000, 0x0000])).unwrap(),
            (0, 0x0000_0000)
        );
        assert_eq!(
            bf16_argmax(&bf16(&[0x8000, 0x8000])).unwrap(),
            (0, 0x8000_0000)
        );
        assert_eq!(bf16_argmax(&bf16(&[0xc000, 0xbf80])).unwrap().0, 1);
        assert_eq!(
            bf16_argmax(&bf16(&[0x3f80, 0x3f80])).unwrap(),
            (0, 0x3f80_0000)
        );
        assert_eq!(
            bf16_argmax(&bf16(&[0x0001, 0x0002, 0x8001])).unwrap(),
            (1, 0x0002_0000)
        );
        assert_eq!(
            bf16_argmax(&bf16(&[0x8001, 0x8002])).unwrap(),
            (0, 0x8001_0000)
        );
        assert_eq!(
            bf16_argmax(&bf16(&[0x8002, 0x8001])).unwrap(),
            (1, 0x8001_0000)
        );
        assert_eq!(
            bf16_argmax(&bf16(&[0x0000, 0x8000])).unwrap(),
            (0, 0x0000_0000)
        );
        assert!(bf16_argmax(&bf16(&[0x7fc0, 0x3f80])).is_err());
        assert!(bf16_argmax(&[0, 1, 2]).is_err());
        assert!(bf16_argmax(&[]).is_err());
    }
    #[test]
    fn source_only_binding_is_one_armed_call_and_never_completes() {
        let mut expected = Expected::new("/nonexistent/original-expected".into()).unwrap();
        assert!(expected.bind(0, 0).is_ok());
        assert_eq!(expected.active_mismatches().unwrap(), 0);
        assert!(expected.bind(0, 1).is_err());
        assert!(expected.bind(1, 0).is_err());
        assert!(expected.bind(0, 0).is_err());
        assert!(expected.finish().is_err());
        assert!(expected.join_evidence(None).is_err());
    }
    #[test]
    fn ordinary_closure_refuses_armed_or_claimed_and_needs_a_held_reference() {
        let mut expected = Expected::new("/nonexistent/original-expected".into()).unwrap();
        assert!(expected.bind(0, 0).is_ok());
        assert!(expected.close_ordinary().is_err());
        let mut absent = Expected::new("/nonexistent/original-expected".into()).unwrap();
        assert!(absent.close_ordinary().is_err());
        let mut claimed = Expected::new("/nonexistent/original-expected".into()).unwrap();
        claimed.scope_words = 1;
        assert!(claimed.close_ordinary().is_err());
    }
    #[test]
    fn interrupted_scope_requires_armed_call_and_exact_partition() {
        assert_eq!(
            interrupted_scope(&[(0, 0)], (0, 1), &[(0, 0), (0, 1)]).unwrap(),
            BTreeSet::from([(0usize, 0usize)])
        );
        for bad in [
            vec![(0, 0)],
            vec![(0, 1)],
            vec![(0, 0), (0, 0), (0, 1)],
            vec![(0, 0), (0, 1), (1, 0)],
        ] {
            assert!(
                interrupted_scope(&[(0, 0)], (0, 1), &bad).is_err(),
                "{bad:?}"
            );
        }
        assert!(interrupted_scope(&[(0, 0), (0, 1)], (0, 1), &[(0, 0), (0, 1)]).is_err());
        assert!(interrupted_scope(&[(0, 0)], (0, 0), &[(0, 0)]).is_err());
    }
    #[test]
    fn metadata_census_counts_structure_and_is_parse_equivalent() {
        let doc = br#"{"a":{"b":[1,2,3,{"c":"d"}]},"a":"\u00e9"}"#;
        let allocation = json_allocation::census_json(doc, ValueConstruction::SerdeValue).unwrap();
        assert_eq!(allocation.input(), doc);
        assert!(allocation.tree_heap_upper() > 4 * std::mem::size_of::<Value>());
        assert_eq!(
            allocation.parse_extra_peak(),
            allocation.tree_heap_upper() + json_allocation::census_scratch_peak(doc.len()).unwrap()
        );
        assert!(serde_json::from_slice::<Value>(doc).is_ok());
        let empty =
            json_allocation::census_json(b"{\"a\":[],\"b\":{}}", ValueConstruction::SerdeValue)
                .unwrap();
        assert_eq!(
            empty.tree_heap_upper(),
            4 * 776 + 2,
            "two objects, two keys"
        );
        assert!(empty.tree_heap_upper() < allocation.tree_heap_upper());
    }
    #[test]
    fn metadata_census_charges_duplicate_occurrences_and_array_slots() {
        let single =
            json_allocation::census_json(b"{\"a\":1}", ValueConstruction::SerdeValue).unwrap();
        let duplicate =
            json_allocation::census_json(b"{\"a\":1,\"a\":2}", ValueConstruction::SerdeValue)
                .unwrap();
        assert_eq!(
            duplicate.tree_heap_upper() - single.tree_heap_upper(),
            776 + 1
        );
        let array =
            json_allocation::census_json(b"[0,1,2,3,4,5,6,7,8]", ValueConstruction::SerdeValue)
                .unwrap();
        // Nine elements grow a Vec to capacity 16; the factor 2 is the
        // old+new realloc overlap, not an element count.
        assert_eq!(
            array.tree_heap_upper(),
            2 * 16 * std::mem::size_of::<Value>()
        );
        assert_eq!(
            json_allocation::census_json(b"[]", ValueConstruction::SerdeValue)
                .unwrap()
                .tree_heap_upper(),
            0
        );
    }
    #[test]
    fn metadata_census_refuses_private_serde_sentinels_at_any_depth() {
        for doc in [
            br#"{"$serde_json::private::RawValue":"{}"}"#.as_slice(),
            br#"{"$serde_json::private::Number":1}"#.as_slice(),
            br#"[{"x":{"$serde_json::private::RawValue":"1"}}]"#.as_slice(),
            br#"{"$serde_json::private\u003a\u003aRawValue":"{}"}"#.as_slice(),
        ] {
            assert!(
                json_allocation::census_json(doc, ValueConstruction::SerdeValue).is_err(),
                "{}",
                String::from_utf8_lossy(doc)
            );
            assert!(
                json_allocation::census_json(doc, ValueConstruction::DirectValueTree).is_ok(),
                "{}",
                String::from_utf8_lossy(doc)
            );
        }
        // The same spelling as an ordinary value is not a transport key.
        assert!(
            json_allocation::census_json(
                br#"{"a":"$serde_json::private::RawValue"}"#,
                ValueConstruction::SerdeValue
            )
            .is_ok()
        );
    }
    #[test]
    fn metadata_census_refuses_junk_escapes_surrogates_and_depth() {
        for doc in [
            br#"{}x"#.as_slice(),
            br#"{"a":1,}"#.as_slice(),
            br#""\q""#.as_slice(),
            br#""\ud800""#.as_slice(),
        ] {
            assert!(
                json_allocation::census_json(doc, ValueConstruction::SerdeValue).is_err(),
                "{}",
                String::from_utf8_lossy(doc)
            );
        }
        assert!(
            json_allocation::census_json(br#""\ud800\udc00""#, ValueConstruction::SerdeValue)
                .is_ok()
        );
        let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
        assert!(
            json_allocation::census_json(deep.as_bytes(), ValueConstruction::SerdeValue).is_err()
        );
        assert_eq!(
            natural_admission_preflight().unwrap(),
            DIAGNOSTIC_HOST_BYTES
        );
    }
    fn temporary_root(tag: &str) -> PathBuf {
        let at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("zg-{tag}-{}-{at}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        root
    }
    fn complete_natural_argv() -> Vec<String> {
        let nonce = "a".repeat(32);
        let output = "/tmp/zg-natural-arity";
        let mut args = vec![
            "podman".to_string(),
            "--root".to_string(),
            PODMAN_ROOT.to_string(),
            "--runroot".to_string(),
            PODMAN_RUNROOT.to_string(),
            "run".to_string(),
            "--name".to_string(),
            format!("{NATURAL_OWNED_NAME}{nonce}"),
            "--label".to_string(),
            format!("{OWNER_LABEL}={nonce}"),
            "--cidfile".to_string(),
            format!("{output}/owned.cid"),
            "--rm".to_string(),
            "--pull=never".to_string(),
            "--network=none".to_string(),
            "--read-only".to_string(),
            "--cap-drop=all".to_string(),
            "--security-opt=no-new-privileges".to_string(),
            "--memory=12g".to_string(),
            "--memory-swap=12g".to_string(),
            "--cpus=4".to_string(),
            "--pids-limit=256".to_string(),
            "--ulimit".to_string(),
            NATURAL_FSIZE.to_string(),
            "--tmpfs".to_string(),
            NATURAL_TMPFS.to_string(),
            "-e".to_string(),
            "CUDA_VISIBLE_DEVICES=-1".to_string(),
            "-e".to_string(),
            "NVIDIA_VISIBLE_DEVICES=void".to_string(),
            "-e".to_string(),
            "ATEN_CPU_CAPABILITY=avx2".to_string(),
            "-e".to_string(),
            "ZG_ALLOWED_CPUS=0,1,2,3".to_string(),
            "-e".to_string(),
            format!("ZG_PACKAGE_SHA={}", "b".repeat(64)),
        ];
        for (name, value) in NATURAL_ENV.iter().skip(3) {
            args.push("-e".to_string());
            args.push(format!("{name}={value}"));
        }
        args.extend([
            "-v".to_string(),
            "/frozen/qwen27b-natural-full-window-reference:/probe:ro".to_string(),
            "-v".to_string(),
            format!("{SOURCE_DIRECTORY}:/source:ro"),
            "-v".to_string(),
            format!("{output}:/output:rw"),
            "--entrypoint".to_string(),
            "python3".to_string(),
            IMAGE.to_string(),
            "/probe/capture.py".to_string(),
        ]);
        args
    }
    #[test]
    fn natural_argv_checks_exact_arity_before_any_indexed_access() {
        let args = complete_natural_argv();
        assert_eq!(args.len(), NATURAL_ARGV_LEN);
        let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
        natural_argv(&refs, &[0, 1, 2, 3], &"a".repeat(32)).unwrap();
        for prefix in [9usize, 10, 11] {
            assert!(
                natural_argv(&refs[..prefix], &[0, 1, 2, 3], &"a".repeat(32)).is_err(),
                "truncated argv {prefix}"
            );
        }
        assert!(
            natural_argv(
                &refs[..NATURAL_ARGV_LEN - 1],
                &[0, 1, 2, 3],
                &"a".repeat(32)
            )
            .is_err()
        );
        let mut long = refs.clone();
        long.push("extra");
        assert!(natural_argv(&long, &[0, 1, 2, 3], &"a".repeat(32)).is_err());
    }
    #[test]
    fn external_receipt_authority_binds_extent_then_same_fd_digest() {
        let root = temporary_root("natural-external");
        let path = root.join(NATURAL_ACCEPTED_FILE);
        let original = b"{\"a\":1}\n";
        std::fs::write(&path, original).unwrap();
        let good = hex(&digest(original));
        assert!(natural_receipt_present(&root, (original.len() as u64, &good)).is_ok());
        // The presence gate checks the supplied extent, never a local hash.
        std::fs::write(&path, b"{\"a\":2}\n").unwrap();
        assert!(natural_receipt_present(&root, (original.len() as u64, &good)).is_ok());
        assert!(
            Held::new_external(
                &root,
                NATURAL_ACCEPTED_FILE,
                original.len() as u64,
                &good,
                RECEIPT_CAP
            )
            .is_err()
        );
        // Growth with the old external extent refuses before transport/GPU.
        std::fs::write(&path, b"{\"a\":2}\n ").unwrap();
        assert!(natural_receipt_present(&root, (original.len() as u64, &good)).is_err());
        // Path replacement: same extent via a symlink still refuses on the
        // retained no-follow descriptor, not on a path reopen.
        std::fs::remove_file(&path).unwrap();
        let target = root.join("a1234567");
        std::fs::write(&target, original).unwrap();
        std::os::unix::fs::symlink(&target, &path).unwrap();
        assert!(
            Held::new_external(
                &root,
                NATURAL_ACCEPTED_FILE,
                original.len() as u64,
                &good,
                RECEIPT_CAP
            )
            .is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn held_external_pin_caps_path_and_digest_before_join_or_open() {
        let root = temporary_root("natural-caps");
        let long_name = "a".repeat(HELD_PATH_CAP + 1);
        assert!(
            Held::new_external(&root, &long_name, 1, &"b".repeat(HELD_SHA_CAP), RECEIPT_CAP)
                .is_err()
        );
        let long_sha = "b".repeat(HELD_SHA_CAP + 1);
        assert!(Held::new_external(&root, "absent", 1, &long_sha, RECEIPT_CAP).is_err());
        let long_root = PathBuf::from(format!("/{}", "c".repeat(HELD_PATH_CAP - 2)));
        assert!(
            Held::new_external(
                &long_root,
                "abcd",
                1,
                &"b".repeat(HELD_SHA_CAP),
                RECEIPT_CAP
            )
            .is_err()
        );
        assert!(
            Held::new_external(
                Path::new("relative"),
                "absent",
                1,
                &"b".repeat(HELD_SHA_CAP),
                RECEIPT_CAP
            )
            .is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn canonical_source_roles_and_module_order_are_complete() {
        let roles = source_roles().unwrap();
        assert_eq!(roles.len(), SOURCE_ROLES);
        let mut expected = std::collections::BTreeSet::new();
        for name in [
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
        ] {
            expected.insert(name.to_string());
        }
        for layer in 0..64 {
            let prefix = format!("model.language_model.layers.{layer}");
            let suffix: &[&str] = if layer % 4 == 3 {
                &[
                    "input_layernorm.weight",
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "self_attn.q_norm.weight",
                    "self_attn.k_norm.weight",
                    "mlp.down_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "post_attention_layernorm.weight",
                ]
            } else {
                &[
                    "input_layernorm.weight",
                    "linear_attn.A_log",
                    "linear_attn.conv1d.weight",
                    "linear_attn.dt_bias",
                    "linear_attn.in_proj_a.weight",
                    "linear_attn.in_proj_b.weight",
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.in_proj_z.weight",
                    "linear_attn.norm.weight",
                    "linear_attn.out_proj.weight",
                    "mlp.down_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "post_attention_layernorm.weight",
                ]
            };
            for suffix in suffix {
                expected.insert(format!("{prefix}.{suffix}"));
            }
        }
        for role in &roles {
            assert!(!role.starts_with('.'));
            assert!(role.len() <= SOURCE_ROLE_CAP);
        }
        assert_eq!(
            roles
                .iter()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>(),
            expected
        );
        let modules = module_order().unwrap();
        assert_eq!(modules.len(), MODULE_COUNT);
        assert_eq!(modules.first().map(String::as_str), Some("embedding"));
        assert_eq!(modules.get(1).map(String::as_str), Some("layer.0"));
        assert_eq!(modules.get(10).map(String::as_str), Some("layer.9"));
        assert_eq!(modules.get(11).map(String::as_str), Some("layer.10"));
        assert_eq!(modules.get(64).map(String::as_str), Some("layer.63"));
        assert_eq!(modules.get(65).map(String::as_str), Some("final"));
        assert_eq!(modules.get(66).map(String::as_str), Some("head"));
        assert!(modules.iter().all(|name| name.len() <= MODULE_STRING_CAP));
    }
    #[test]
    fn retained_five_owner_inventory_fits_complete_reference_bound() {
        let (reference, dir) = synthetic_reference();
        assert_eq!(reference.specs.capacity(), RECORD_COUNT);
        assert_eq!(reference.records.capacity(), RECORD_COUNT);
        let owners = [
            Some(&reference.manifest_owner),
            Some(&reference.raw),
            reference.accepted.as_ref(),
            reference.invocation.as_ref(),
            reference.image.as_ref(),
        ];
        assert_eq!(owners.iter().flatten().count(), HELD_OWNERS);
        for held in owners.into_iter().flatten() {
            assert_eq!(held.path.capacity(), HELD_PATH_CAP);
            assert_eq!(held.sha.capacity(), HELD_SHA_CAP);
        }
        let actual = u64::try_from(reference.retained_bytes()).unwrap();
        let bound = retained_allocation_bound().unwrap();
        assert!(actual <= bound, "actual {actual} <= bound {bound}");
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn metadata_tree_bound_is_derived_before_value_and_refuses_amplification() {
        assert_eq!(json_allocation::census_scratch_peak(0).unwrap(), 17 + 104);
        let keys = (0..20_000)
            .map(|n| format!("\"{n}\":0"))
            .collect::<Vec<_>>()
            .join(",");
        let doc = format!("{{{keys}}}");
        let allocation =
            json_allocation::census_json(doc.as_bytes(), ValueConstruction::SerdeValue).unwrap();
        let tree = allocation.tree_heap_upper();
        assert!(tree > doc.len(), "node bound covers key/value slots");
        let mut small = Admission::new(tree as u64 / 2);
        assert!(small.charge(tree as u64, "tree").is_err());
        assert_eq!(small.high, 0, "a refused allocation leaves no high-water");
        let mut exact = Admission::new(tree as u64);
        exact.charge(tree as u64, "tree").unwrap();
        assert_eq!(exact.high, tree as u64);
        assert!(serde_json::from_slice::<Value>(doc.as_bytes()).is_ok());
        // Retained typed metadata stays below the transient admission preflight.
        assert!(retained_metadata_bound().unwrap() < natural_admission_preflight().unwrap());
    }
}
