//! Completed source/state observations; no model, root or publication authority.
// Previous names: Qwen08ModelReadback, Qwen08DomainReadback.
use crate::SourceBindings;
use cubecl::bytes::Bytes;
use zerogravity_core::accelerator::program::StateSlot;
pub struct SourceDomainReadback<'a> {
    pub slot: StateSlot,
    pub buffers: &'a [Bytes],
}
pub struct SourceModelReadback<'a> {
    pub phase: usize,
    pub record: &'a SourceBindings,
    pub values: &'a [Bytes],
    pub input_rms: &'a [Bytes],
    pub rms: &'a [(usize, &'a [Bytes])],
    pub prior_domains: &'a [SourceDomainReadback<'a>],
    pub domains: &'a [SourceDomainReadback<'a>],
    pub greedy: &'a [Bytes],
}
