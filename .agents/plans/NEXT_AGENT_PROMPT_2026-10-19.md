# Starting prompt — Zerogravity comprehensive takeover, 19 October 2026

Copy the text below into the next agent. The active task is the P12 batch-native resident GPU forward implementation; the handoff also preserves the complete original architecture and all later model/SSD/serving priorities. This is an implementation handoff, NOT the earlier 17 October documentation-only pause.

~~~text
You are taking over the EXISTING live Zerogravity Rust repository at:

  /home/pvl/spaces/abird/src/zerogravity

Use the connected @dotlink-l5 local machine connector; specify that exact directory in every tool call. This is an established, source-verified engine, not a greenfield project and not an invitation to rewrite it.

FIRST: Read the NEW complete handoff progressively, without truncating its context:

  .agents/docs/FULL_COMPREHENSIVE_AGENT_HANDOFF_2026-10-19.md

Read its up-to-date first 16 sections, Appendix A containing the complete previous 3,430-line October 17 historical handoff and decision/recovery register, Appendix B containing live-plan TODO snapshots, and Appendix C containing the exhaustive docs/plans/research/prior-art reading catalog. The embedded October 17 pause and draft statuses are HISTORICAL and were superseded by the user's explicit implementation continuation. Do not resume older unmerged worktrees or the C3/C4 acquisition drafts.

Then read current:
  AGENTS.md
  .agents/plans/README.md
  .agents/plans/TODO.md
  .agents/plans/serving-and-speculation/{README,TODO,FUSED_GPU_BATCHING}.md
  .agents/docs/JOINT_CONCURRENT_SERVING_OPTIMIZATION_AND_GOODPUT_CONTRACT_2026-10-15.md
  .agents/research/VLLM_PARITY_AND_COMBINATORY_SPECIALIZATION_ACCEPTANCE_2026-10-15.md
  .agents/plans/IMPLEMENTATION_ROADMAP.md
  .agents/plans/ZEROGRAVITY_MASTER_PLAN_2026-10-10.md

Consult other canonical specs listed in the handoff when their ownership is relevant. These source files and active TODOs outrank a dated handoff.

REVALIDATE BEFORE ANY CHANGE:
  git status --short --branch
  git log -12 --oneline
  git worktree list
  git diff --check
  ps for Cargo/rustc/active GPU-serving processes
  relevant source ownership and configured device/model/capacity
  active agent-owned changes/branches/logs; do not overwrite them.

At handoff creation, the authoritative signed HEAD was
  3ec431a2df62bbcae5091a048f285801f151da7e
  feat(gpu): qualify native batched GGUF quantized projections

MAIN was clean and nine commits ahead of local origin/main, with no other worktree. It may have moved; current Git wins. The root README is user-authored: DO NOT EDIT it.

OWNER'S CURRENT PRIORITIES, IN ORDER:
1. Genuine vLLM-equivalent fused continuous GPU batching and physically verified OpenAI-compatible serving, then actual source-matched vLLM goodput benchmark.
2. Source-preserving SSD/GPU/RAM tensor residency integrated into the SAME engine, with real GPU readiness/lease and trained resident-vs-SSD StateRoot/logit proof.
3. Original Qwen3.8 27B CPU/BF16/FLA model source, trained numerical and large-source/SSD GPU execution.
4. Gemma4 E2B, then 26B-A4B and dense 31B, each separately source/numerically qualified; Flash-Next and all other multimodal/distributed/speculative work remain on their own plans.

WHAT HAS ALREADY PASSED (DO NOT REIMPLEMENT):
- Original real Qwen3.5 0.8B/2B CPU GGUF and 512-consumed-token native resident GPU bounded profiles, source/tokenizer/greedy/state/cancel/quarantine and full-vocab/all42-domain parity.
- Real OpenAI local CPU HTTP and resident CUDA GPU serving, multiple parallel sessions/alias-sharing, strict greedy-only JSON/SSE, queues, cancellation and resource admission. HTTP state-cohort publication is now verified for compatible one-token final prefill; the GPU model graphs still execute serially per request.
- Native B=2/4/8 GPU CUDA fused isolated F32 and GGUF Q4_0/Q4_1/Q8_0/Q6_K projection operators, batched elementwise/norm/GDN/KV/attention, source-exact versus B1.
- Real CUDA Qwen3.5 0.8B B2/B4/B8 ResidentBatchRows and PreparedResidentCohort: exact 248,320 logits and all42 state domains PER independent session, source/Ready/epoch/root/state arbitration, aborted/stale/foreign rollback and reclaimed leases. Full model GPU kernel fusion is NOT yet implemented; HTTP must keep gpu_fused_batches=0, and batch_mode=fused must refuse.
- Bounded checksum-verified immutable SSD source chunk/window IO and leased staging; no trained model streamed from SSD.
- YAML model/pipeline compiler/context filters/grants/decision adapter scaffolds, original Qwen3.8 BF16 mathematical diagnostics and pinned reference-trace comparator. No trained 27B or Gemma4 inference.

THE IMMEDIATE TASK: Implement the next physically qualified P12 slice, not another documentation or mock-only phase.

1. Inspect exact current source:
   crates/zerogravity-core/src/accelerator/program.rs
   crates/zerogravity-core/src/session_root.rs
   crates/zerogravity-gpu/src/resident_program.rs
   crates/zerogravity-gpu/src/resident_inference.rs
   crates/zerogravity-gpu/src/resident_batch.rs
   crates/zerogravity-gpu/src/batch_{projection,quant_projection,elementwise,recurrent,kv,attention}.rs
   crates/zerogravity-cli/src/serve/{gpu_engine,gpu_batch,http,engine,mod}.rs
   tools/serve-bench.py, tools/serve-compare.py

2. Introduce the smallest source-qualified B>1 batch-native ForwardPlan/ResidentProgram execution slice that actually CALLS the accepted multi-row GPU kernels for compatible same-source model lanes. Preserve the immutable weight source once, independent per-row prior/candidate KV/GDN, masks and causal frontiers, original SessionAuthority, ResourceArbiter, exact BufferLease/Ready/fence/quarantine, and distinct per-request output/greedy. DO NOT fork the Qwen token loop or invent another StateRoot issuer.

3. Qualify original trained Qwen3.5 0.8B physical CUDA B2 full-model complete logits/all42 domains against the accepted CPU/B1 references, one real fused multi-row kernel per appropriate graph stage, unequal prefixes, EOS, cancellation/stale epoch/foreign row and physical completion. Expand to B4/B8, Qwen3.5 2B and qualified quant codecs only after their smaller source/physical gates pass. Report actual kernel-launch counts, not grouped scheduler waves or model forward API calls.

4. Keep HTTP batch_mode=fused disabled until the complete B>1 resident model graph has passed all advertised-backend source/state/cancellation gates. Later integrate source-safe paged/hybrid KV and finite fair continuous batching; compare to true vLLM V1 on same weights/quant/tokenizer/hardware under matched cold/warm offered and committed load. tools/serve-compare.py is only a harness; vLLM was NOT installed or running on this host at handoff. Do not create fake comparison results.

5. Run focused tests, then the complete isolated gate:
   CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=4 python3 scripts/native-gpu.py -- bash scripts/check.sh

   Then source-appropriate strict all-target Clippy -D warnings and ACTUAL physical GPU original-model tests if needed. Run the separate approved Qwen3.5 .8B/2B original cache/direct AND HF-import release parity routes from handoff section 12. Verify full-vocabulary logits/all42 domains and status/exit receipts; a compiler pass is not a device pass.

6. Observe approval boundaries: no model/SDK downloads, held Renoir/gfx90c AMD launches or resets, external host privilege changes, destructive Git operations, root README edits, push/tag/release/deploy unless separately authorized. CUDA runtime/CCCL/NVRTC are present on the qualified machine via already-installed Nix packages; set CUDA_PATH correctly before physical CUDA, not by acquiring new SDKs. Avoid unqualified CUDA/Vulkan fallback. The separate offline Nix source-integrity derivation has been blocked by an absent pinned nixpkgs cache/SSL trust, while Nix formatting passes; report this accurately.

7. Keep the plan workflow simple:
   .agents/plans/TODO.md = one live queue;
   .agents/plans/serving-and-speculation/TODO.md = current P0 owner;
   .agents/runs/<phase>-<agent>/TODO.md = work in progress.
   Commit only reviewed implemented source/tests/docs with signed commits and verify the exact staged paths. Do not push. Record genuinely remaining work as unchecked items, leaving already-completed items under their signed commits.

Use independent subagents for focused review, implementation and testing where available, with disjoint file scopes and one MAIN Cargo owner; otherwise do the work with the available tools. Make progress autonomously. On an interruption, inspect actual live tree and test logs and continue; never rewrite already-qualified work or claim a phase finished from partial results.

Report succinctly: current verified HEAD, physically passing acceptance, exact unqualified fusion/vLLM/SSD/model gates, test exits, changed paths and next execution step. Do not confuse multi-root atomic publication with native multi-row model fusion.
~~~
