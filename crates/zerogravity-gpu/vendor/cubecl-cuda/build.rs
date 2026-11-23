use cudarc::driver::sys::CUDA_VERSION;
use std::{
    env,
    ffi::OsStr,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    println!("cargo::rustc-check-cfg=cfg(cuda_12050)");
    println!("cargo::rustc-check-cfg=cfg(cuda_12080)");

    if CUDA_VERSION >= 12050 {
        println!("cargo:rustc-cfg=cuda_12050");
    }
    if CUDA_VERSION >= 12080 {
        println!("cargo:rustc-cfg=cuda_12080");
    }

    if env::var_os("CARGO_FEATURE_FLASH_ATTN").is_some() {
        build_flash_attention();
    }
}

fn required_path(name: &str) -> PathBuf {
    println!("cargo:rerun-if-env-changed={name}");
    env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
        .unwrap_or_else(|| panic!("{name} must point to the pinned source directory"))
}

fn run(program: impl AsRef<OsStr>, args: &[impl AsRef<OsStr>]) {
    let program = program.as_ref();
    let status = Command::new(program)
        .args(args.iter().map(AsRef::as_ref))
        .status()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", Path::new(program).display()));
    assert!(
        status.success(),
        "{} failed with {status}",
        Path::new(program).display()
    );
}

fn build_flash_attention() {
    let source = required_path("ZEROGRAVITY_FLASH_ATTN_SRC");
    let cutlass = required_path("ZEROGRAVITY_FLASH_ATTN_CUTLASS");
    let cudart = required_path("ZEROGRAVITY_CUDART_LIB");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));

    let bridge = PathBuf::from("src/owned_command/flash_attn/bridge.cu");
    let shim = PathBuf::from("src/owned_command/flash_attn/shim");
    let kernel = source.join("csrc/flash_attn/src/flash_fwd_hdim256_bf16_causal_sm80.cu");
    let kernel_object = out.join("flash_fwd_hdim256_bf16_causal_sm80.o");
    let bridge_object = out.join("flash_attn_bridge.o");
    let archive = out.join("libzg_flash_attn_fa2.a");

    println!("cargo:rerun-if-changed={}", bridge.display());
    println!("cargo:rerun-if-changed={}", shim.display());

    let mut common = vec![
        "-std=c++17".into(),
        "-O3".into(),
        "--use_fast_math".into(),
        "--expt-relaxed-constexpr".into(),
        "-lineinfo".into(),
        "-DFLASHATTENTION_DISABLE_BACKWARD".into(),
        "-DFLASHATTENTION_DISABLE_DROPOUT".into(),
        "-DFLASHATTENTION_DISABLE_ALIBI".into(),
        "-DFLASHATTENTION_DISABLE_SOFTCAP".into(),
        "-DFLASHATTENTION_DISABLE_LOCAL".into(),
        "-DFLASHATTENTION_DISABLE_UNEVEN_K".into(),
        // vLLM FA2's Ampere policy: one SM80 image plus forward-compatible PTX.
        "-gencode".into(),
        "arch=compute_80,code=sm_80".into(),
        "-gencode".into(),
        "arch=compute_80,code=compute_80".into(),
    ];
    for include in [
        shim,
        source.join("csrc/flash_attn"),
        source.join("csrc/flash_attn/src"),
        source.join("csrc/common"),
        cutlass.join("include"),
    ] {
        common.push("-I".into());
        common.push(include.into_os_string());
    }

    let nvcc = env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let mut kernel_args = common.clone();
    kernel_args.extend([
        "-c".into(),
        kernel.into_os_string(),
        "-o".into(),
        kernel_object.clone().into_os_string(),
    ]);
    run(&nvcc, &kernel_args);

    let mut bridge_args = common;
    bridge_args.extend([
        "-c".into(),
        bridge.into_os_string(),
        "-o".into(),
        bridge_object.clone().into_os_string(),
    ]);
    run(&nvcc, &bridge_args);

    let ar = env::var_os("AR").unwrap_or_else(|| "ar".into());
    run(
        &ar,
        &[
            "crs".into(),
            archive.into_os_string(),
            kernel_object.into_os_string(),
            bridge_object.into_os_string(),
        ],
    );

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=zg_flash_attn_fa2");
    println!("cargo:rustc-link-search=native={}", cudart.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
