# Initial hardware targets and capability optimization

16 October 2026. **User decision:** the initial target fleet is the current
machine plus SSH targets `pvl-a1` and `pvl-x2`: three CPUs and five physical GPUs.
Optimize all of them after native CUDA/ROCm enablement. The user explicitly
requests subagents for implementation work going forward. This document records
the target decision and planned acceptance; it does not qualify new backends,
models, context sizes, ISA kernels or performance superiority.

## Target fleet inventory

These are the initial discovered identities/capabilities, not a current free
memory budget or a new qualification. Recheck the selected profile before
hardware work. Completion and open work are in the [GPU/CPU TODO](../plans/cpu-gpu-optimization/TODO.md).

| Machine | CPU, topology and reported cache | Observed CPU instructions |
| --- | --- | --- |
| Current machine | Ryzen 7 5800H, 8 cores/16 threads, one NUMA node; L1d 32 KiB/core, L2 512 KiB/core, L3 16 MiB | AVX2, FMA, F16C; AVX512 absent |
| `pvl-a1` | Ryzen AI 9 HX 370, 12 cores/24 threads, one NUMA node; L1d 48 KiB/core, L2 1 MiB/core, L3 16+8 MiB. Topology reports four cores with 5.158 GHz maximum and eight with 3.289 GHz maximum | AVX2/FMA/F16C, AVX512F/DQ/BW/VL/IFMA, AVX-VNNI, AVX512-VNNI/BF16/VBMI/VBMI2/VPOPCNTDQ |
| `pvl-x2` | Ryzen AI MAX+ 395, 16 cores/32 threads, one NUMA node; L1d 48 KiB/core, L2 1 MiB/core, L3 2×32 MiB | AVX2/FMA/F16C, AVX512F/DQ/BW/VL/IFMA, AVX-VNNI, AVX512-VNNI/BF16/VBMI/VBMI2/VPOPCNTDQ |

| Physical GPU | Observed architecture and execution width | Memory and relevant limits |
| --- | --- | --- |
| Current RTX 3060 Laptop, PCI `0000:01:00.0` | CUDA compute capability 8.6 (`sm86`) | 6144 MiB dedicated VRAM |
| Current AMD Renoir | Actual HIP/COMGR target `gfx90c`, 8 CU, wave64 | Shared system memory; no WMMA/MFMA capability inferred |
| `pvl-a1` RTX 4060 Laptop, PCI `0000:64:00.0`, `10de:28e0` | Compute capability 8.9 (`sm89`); Vulkan subgroup32 | 8188 MiB dedicated VRAM; Vulkan shared workgroup memory 48 KiB, maximum 1024 invocations |
| `pvl-a1` Radeon 890M, PCI `0000:66:00.0`, `1002:150e` | Actual ROCm `gfx1150`, 16 CU, wave32; RADV subgroup64 default, controllable 32–64 | Shared RAM; sysfs 512 MiB VRAM aperture and 15.23 GiB GTT, not additive capacity; LDS 64 KiB, maximum 1024 invocations |
| `pvl-x2` Radeon 8060S, PCI `0000:c6:00.0` | Actual ROCm `gfx1151`, 40 CU, wave32; RADV subgroup64 default, controllable 32–64 | Nominal 64 GiB shared GPU pool; usable host RAM 67.04 GB decimal; LDS 64 KiB, maximum 1024 invocations |

Both remote AMD Vulkan devices report Mesa 26.1.8/Vulkan 1.4.354, FP64/F16,
I8/I16/I64, integer dot8 and cooperative matrices, plus FP32 signed-zero/NaN,
denormal and round-to-nearest preservation. The remote NVIDIA driver is
595.91.07; Vulkan 1.4.329 reports FP64, subgroup32 and FP32 denormal-preserve
false. These are advertised capabilities; actual kernel behavior remains a
separate gate. Vulkan CPU llvmpipe is excluded from the five-GPU count.

The qualified native a1 HIP profile uses its existing ROCm/runtime closure with
an explicit glibc2.44/GCC16.2 loader. The default2.42 loader cannot replace that
profile. Exact approved model identities are in [the accepted baseline](../plans/accepted-baseline/TODO.md).
SDK/compiler versions, usable memory, swap/load and power telemetry must be
measured for a new test; no global cache purge, downloads or device recovery is
implied. Old H24 discovery JSON/scripts/stderr are retired and are not required
execution inputs. The local gfx90c launch hold remains.

## Existing CPU execution and concrete opportunities

Read-only source audit: `crates/zerogravity-cpu/src/weights.rs` prepares source Q8 activation blocks
once per projection, reads original packed GGUF rows and uses Rayon parallel
rows. The integer dots and F32 dots express the qualified source's lane/FMA
reduction as portable Rust; there is no explicit runtime AVX2/AVX512 dispatcher,
CPU intrinsic implementation or model-specific topology-aware worker pool.
Compiler autovectorization is possible and must be inspected/measured rather
than assumed absent. `crates/zerogravity-model-qwen35/src/lib.rs` owns ordered GDN/attention dots and norms; the math module owns the source polynomial/FMA exponential schedule.

Implement capability-selected kernels inside those existing numerical
boundaries, retaining a qualified portable path. AVX2 and AVX512/VNNI can decode
packed weights, form exact integer lanes and process independent rows without
changing source scale/FMA/reduction order. Benchmark vector activation max,
round-to-even conversion, half conversion and quantization; preserve first
signed-maximum ties, NaN-to-integer behavior, subnormals and signed zeros.
Avoid saturating integer dot instructions unless the exact intermediate bounds
prove saturation impossible. BF16 availability does not authorize replacing
source FP32 operations with lower-precision arithmetic.

Measure thread counts, SMT on/off placement, cache-local row tiles, immutable
source-derived repacking and activation reuse. Admission must include derived
layouts; immutable source identity remains the original checkpoint. Evaluate
four fast-core versus all-core placements on the HX 370 using observed
frequency/cache groups and measured load; preserve small-operation serial
thresholds and reserve CPU capacity for GPU submission/tokenizer/serving work.
Tune the two L3 groups on MAX+ 395 independently of NUMA (one node was observed).
Do not add another scheduler authority or pin processes/change governors as an
unrequested persistent machine configuration.

## Ordered implementation and acceptance

After native qualification, follow the approved
[workspace/layout redesign](../plans/accepted-baseline/TODO.md),
then the immediately following [left-right redesign](../plans/cpu-gpu-optimization/TODO.md).
Implement measured optimization variants within that user-approved ordering.

1. **Complete native enablement first.** Follow
   [P07-H24](../plans/accepted-baseline/TODO.md); qualify compilation,
   physical identity and exact operators before complete CUDA/HIP models.
   Reuse one Rust model schedule and existing physical arbiters/leases/events.
   Never infer `gfx1150`/`gfx1151` support from a different RDNA target or count
   runtime initialization as a trained-model pass. Establish remote assets and
   their hashes before remote model qualification. The fleet request authorizes
   implementation and bounded temporary binary/evidence transfers and qualification
   workloads over SSH. Model/SDK downloads, persistent service/configuration or
   driver/device privilege changes, push and release retain their approval limits.
2. **Record end-to-end baselines on all targets.** Pin source/tokenizer,
   compiler/runtime/backend, numeric policy, topology, power policy, workload
   and resource limits. Separate cold compilation/load/prefill from warm cached
   prefill/decode, fresh versus reused prefixes, interactive versus batched
   concurrency, and CPU/GPU/mixed placement. Measure TTFT, inter-token latency,
   completed useful tokens/s, p50/p95/p99, queue/admission delay, host core time,
   actual transfer bytes, memory residency and cancellation latency. Retain
   short and long contexts only within independently qualified boundaries.
3. **Tune CPU variants and GPU kernels in bounded units.** Qualify AVX2 then
   AVX512/VNNI candidates, cache/thread policies and source-quantized GEMV;
   tune GPU row/block tiling, subgroup widths, vector loads, admitted packing,
   occupancy and memory reuse per observed architecture. Compare exact fused
   and prepared execution before adding complexity. Consider tensor/MMA/WMMA
   or accelerated integer dot only when their accumulation, saturation and
   rounding match the selected source contract; FP16/BF16/TF32 substitution
   cannot satisfy the current exact FP32 contract by changing tolerances.
4. **Qualify shared-memory and heterogeneous placement.** Couple CPU and APU
   pressure against the same physical RAM pool; Vulkan heaps, HIP free memory,
   VRAM apertures and GTT counters may overlap and must never be summed. Charge
   immutable sources, candidate/committed roots, staging, compiler/workspace
   and quarantine; retain OS/desktop headroom. Exercise CPU/APU/dGPU concurrency,
   bandwidth contention, cross-backend access to the same PCI device and fair
   bounded service. Multiple GPUs are placement choices, not permission to
   shard a stateful model without the existing distributed/ownership contracts.
5. **Reduce complete-request overhead after measured bottlenecks.** Use the
   [GPU text/control roadmap](GPU_TEXT_FRONTEND_AND_DEVICE_ORCHESTRATION_ROADMAP_2026-10-16.md)
   for graph replay, bounded device generation, tokenizer/decoder placement and
   overlap. All additions remain placements of one engine and session authority.

Every optimization must pass unchanged independent original-source criteria,
complete logits/all state domains/ordered traces and zero-sign reporting,
multi-token nonzero state, replay/checkpoint/rollback, stale ownership refusal,
physical readback/admission and real cancellation/fencing/original quarantine.
Preserve known independent CPU 8K drift and deferred larger-context boundaries.
Then require a repeated end-to-end benefit under comparable conditions, retain
negative results and report scope per CPU/GPU/backend/workload. Kernel timing,
peak utilization, fewer launches or advertised hardware features cannot prove
that the hardware is fully used or another runtime is outperformed.

Runtime selection keys must be capabilities: ISA/OS vector support, topology,
cache groups, PCI identity, backend/architecture/subgroup, numeric features,
memory domain and qualified kernel/compiler version. Hostnames belong only in
fleet discovery and benchmark receipts. The goal is the best demonstrated
useful performance for these targets; each improvement remains empirically gated.
