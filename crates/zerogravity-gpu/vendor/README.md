# CubeCL native target and compiler dependency support

These five crates are the published CubeCL 0.11.0 sources. Four carry narrow
local capability-table and optional compiler dependency patches; `cubecl-wgpu`
additionally carries the local sealed Vulkan/SPIR-V native-catalogue
implementation described below. Cargo can use them through ordinary
`[patch.crates-io]` entries. The global Cargo registry is unchanged.

All five packages identify upstream commit
`52c5086d3c5e73e56b3ee4cb711bdda495a360f0` in their published
`.cargo_vcs_info.json`, from <https://github.com/tracel-ai/cubecl>:

| Package | Upstream directory | Published `.crate` SHA-256 |
| --- | --- | --- |
| `cubecl-ir` 0.11.0 | `crates/cubecl-ir` | `9e7ca48e7231089b8376a12c3322fce4390a272ed41fe2e659a790615b2d8f9d` |
| `cubecl-cpp` 0.11.0 | `crates/cubecl-cpp` | `1983253e91442adcff6a72ac30d9003fe44446dd0bca7d7bb41d39834da99e0b` |
| `cubecl-cuda` 0.11.0 | `crates/cubecl-cuda` | `0f10277e142dfaf556d24875cf5b347322892b7d7a0650f85b7590bae95e6a86` |
| `cubecl-hip` 0.11.0 | `crates/cubecl-hip` | `d3cf95f0ff97999ab1bbbad2f91aacd81c73642dc0be41bc46dce41fcdd5b424` |
| `cubecl-wgpu` 0.11.0 | `crates/cubecl-wgpu` | `2b8c6f4eb4af39ed3bbb3e4126373d8132cbe018e669172d8d4f1c775b828497` |
| `cubecl` 0.11.0 (license texts only) | `crates/cubecl` | `b423fd1c34d0f034556c4e662c815e814ffd48b5b85f05341b04dcc14276b73f` |

The archives already existed in the local Cargo cache. Their SHA-256 digests were
verified against the repository's locked registry checksums before copying any
source. No source or SDK was downloaded. The published `.crate` digests in the
table above pin the exact upstream archives, and the files edited by Zerogravity
are listed below.

`cubecl-wgpu` retains its complete `build.rs`, published normalized
`Cargo.toml` and `README.md`; its manifest matches the registry copy byte for
byte. Its `src/` is the published tree plus the local catalogue changes listed
below. Each other package retains its complete `src/`, published normalized
`Cargo.toml` and `README.md`. The CPP and CUDA packages also retain their required `build.rs`. Upstream lock
files, the unused original manifests, registry markers and VCS metadata are
omitted. The IR archive's `LICENSE-APACHE` and `LICENSE-MIT` contain literal
`../../LICENSE-APACHE` and `../../LICENSE-MIT` path placeholders. Those upstream
bytes remain unchanged in that package. This directory contains the actual
10,868-byte Apache and 1,105-byte MIT license texts copied from the cached
`cubecl` 0.11.0 archive, verified against its locked registry checksum and the
same upstream commit. The CPP archive declares the same `MIT OR Apache-2.0`
license but does not package license texts. The shared license provenance is
recorded in this directory. Existing source notices are preserved.

The local `cubecl-wgpu` changes are:

- `src/compute/native.rs`: add the sealed source-owned catalogue, its bounded
  inventory, admission trait, actual-environment observation and
  drain-before-retirement set. Unknown preparation or retirement retains the
  partial/full modules and owners to the process boundary.
- `src/compute/server.rs`: preserve each compiled SPIR-V source and its actual
  binding/IO descriptors beside the pipeline; add the cold catalogue
  preparation, live check, environment and retirement paths.
- `src/backend/base.rs`: carry the actual cached SPIR-V source into the same
  pipeline entry.
- `src/backend/vulkan.rs`: keep the effective storage-buffer/binding bound when
  it is smaller than the device-local heap, instead of replacing it with the
  heap size.
- `src/runtime.rs`: capture that effective bound before backend substitution and
  pass it to the server.
- `src/compute/mod.rs`: expose the sealed catalogue internally.

The three AMD capability-table changes are:

- `cubecl-ir/src/amd.rs`: add the exact `GFX90C` family and parser entry, with
  wave64 and no WMMA; test case folding, target-feature suffixes and rejection of
  unknown neighboring targets.
- `cubecl-cpp/src/hip/arch.rs`: test that this family reports wave64 without WMMA
  or MFMA and that unknown targets retain no usable wave width.
- `cubecl-cpp/src/hip/mma/rocwmma_compiler.rs`: return no matrix combinations for
  `GFX90C`, with a focused test.

This recognizes the real driver target. It neither changes the driver-reported
architecture nor maps the device to a CDNA family with unsupported matrix
instructions. Unknown targets retain the existing refusal behavior. Source
vendoring and capability tests alone do not qualify native kernel execution or
numerical parity; the repository's physical qualification remains required.

The CUDA package changes exactly four upstream files. HIP changes the same
runtime/compiler/context files plus its crate export:

- `Cargo.toml`: make `cubecl-llvm` optional behind `llvm`, retain that feature in
  upstream defaults, and use weak `pliron-dump` forwarding so diagnostics do not
  enable an unused compiler.
- `src/compiler.rs`: gate LLVM imports, representations and dispatch branches.
  Require either `cpp` or `llvm`; retain CPP selection when both are enabled.
- `src/runtime.rs`: gate LLVM-only feature restrictions and CUDA PTX-version
  selection. CPP runtime properties and compiler options retain their original
  behavior.
- `src/compute/context.rs`: gate LLVM-only artifact loading and preserve the
  existing CPP cache namespace. Focused host tests cover compiler selection and
  CPP cache keys.
- `src/lib.rs` (HIP only): re-export the actual driver PCI preflight beside
  `HipRuntime`, so the selected ordinal-0 adapter identity is available before
  any client/context/queue effect. `src/runtime.rs` adds
  `preflight_physical(AmdDevice)` using `hipDeviceGetPCIBusId`; it does not
  enumerate another adapter, create a client or compile a kernel.

Zerogravity selects CubeCL with global defaults disabled, explicitly preserves
its existing std/stdlib and core/server features, and selects `cuda-cpp` or
`hip-cpp`. That selected normal/build dependency graph excludes `cubecl-llvm`,
`pliron-llvm`, `llvm-sys` and the LLVM bundler. Bare `pliron` remains the shared
CubeCL IR dependency. The published default LLVM behavior is still available to
other consumers; Cargo feature unification can enable it explicitly.

The qualified NVRTC/HIPRTC CPP compiler, cache loading, runtime launch and source
arithmetic bodies are unchanged. This dependency isolation removes an unused
LLVM SDK prerequisite and final-executable archive-ordering shim; it does not
introduce another compiler or qualify LLVM device code generation. The isolated
source audit records byte-identical CPP dispatch fragments, runtime fingerprint
assignments and 17 compiler/load/launch function bodies. Actual library links,
consumer compatibility and physical qualification are separate acceptance gates.

## Workspace and external consumer patch contract

The production workspace excludes these five vendor packages from its member
list and selects them through its root patch table and one effective lockfile.
Cargo does not propagate workspace patches through library dependencies. An
external unpublished path consumer must declare the same five verified patches
in its own root manifest, in addition to its selected GPU dependency:

```toml
[dependencies]
zerogravity-gpu = { path = "/path/to/zerogravity/crates/zerogravity-gpu", default-features = false, features = ["cuda"] }

[patch.crates-io]
cubecl-ir = { path = "/path/to/zerogravity/crates/zerogravity-gpu/vendor/cubecl-ir" }
cubecl-cpp = { path = "/path/to/zerogravity/crates/zerogravity-gpu/vendor/cubecl-cpp" }
cubecl-cuda = { path = "/path/to/zerogravity/crates/zerogravity-gpu/vendor/cubecl-cuda" }
cubecl-hip = { path = "/path/to/zerogravity/crates/zerogravity-gpu/vendor/cubecl-hip" }
cubecl-wgpu = { path = "/path/to/zerogravity/crates/zerogravity-gpu/vendor/cubecl-wgpu" }
```

Choose `hip` or `vulkan` as appropriate. The bounded external link qualification
uses its own explicit patch table; a plain GPU path dependency does not inherit
these patches and is not covered by that proof. Cargo features are additive, so
an external dependency that independently selects upstream LLVM defaults can
reintroduce that separate compiler requirement. No additional vendoring or
implicit SDK acquisition is performed to conceal this Cargo behavior.
