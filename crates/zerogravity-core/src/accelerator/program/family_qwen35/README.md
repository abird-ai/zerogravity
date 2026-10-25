# Qwen3.5 execution recipes

Family-specific immutable mathematical descriptors and source geometry only.
Their Rust modules remain `accelerator::program::{full_model_profile,short_hf,vllm_model,vllm_native}`
via explicit `#[path]` declarations. This preserves existing type/plan identities
while isolating Qwen assumptions in one source directory. The model's authenticated
source, numerical policies, prepared hardware implementation and session/flight
ownership remain independent, and this folder cannot grant model eligibility.

New model families should supply their own verified mathematical adapter into
the existing `ForwardPlan`/`TextCallGeometry`; do not append their hardcoded
shapes to these Qwen recipes or duplicate shared execution/storage.
