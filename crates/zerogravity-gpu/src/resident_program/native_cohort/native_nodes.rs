//! Source-typed native operands, one binding per node in the original loop.
use super::super::*;
use super::{attention::CohortAttention, head_input::CohortHeadInput, values::CohortValues};
use crate::resident_inference::native_fragment::{PrefixPhaseEnqueue, short_hf::ShortHfCallInput};
use crate::resident_leases::{ResidentLease, TiedWeight};
use crate::vllm_cuda_qualification::cohort_native::{CohortNativeLibrary, CohortNativeSite, Role};
use std::sync::atomic::AtomicBool;
use zerogravity_core::accelerator::program::{StateEffect, Vllm08Kind, Vllm08Projection};
use zerogravity_core::device_events::DeviceAccess;
use zerogravity_model_qwen35::original_binding::QwenMatrixRole;
pub(super) struct CohortNativeNodes {
    values: Arc<CohortValues>,
    _library: Arc<CohortNativeLibrary>,
    _head: Arc<CohortHeadInput>,
    _tied: Arc<TiedWeight>,
    attention: BTreeMap<usize, Arc<CohortAttention>>,
    nodes: BTreeMap<usize, CohortNativeSite>,
}
/// Per-node census: source node -> concrete access list.
type CohortNodeCensus = Vec<(usize, Vec<(ResidentLease, DeviceAccess)>)>;

impl CohortNativeNodes {
    pub(super) fn bind(
        inputs: &[Arc<ShortHfCallInput>],
        values: Arc<CohortValues>,
        library: Arc<CohortNativeLibrary>,
        head: Arc<CohortHeadInput>,
        tied: Arc<TiedWeight>,
        attention: BTreeMap<usize, Arc<CohortAttention>>,
        status: ResidentLease,
    ) -> Result<Arc<Self>> {
        values.require_inputs(inputs)?;
        let mut nodes = BTreeMap::new();
        for (index, node) in inputs[0].record().plan().nodes().iter().enumerate() {
            let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
                continue;
            };
            let kind = descriptor.kind();
            if !matches!(
                kind,
                Vllm08Kind::Projection(_)
                    | Vllm08Kind::SiluAndMul
                    | Vllm08Kind::AttentionQk
                    | Vllm08Kind::KvAppend
                    | Vllm08Kind::FlashAttention
                    | Vllm08Kind::Sigmoid
                    | Vllm08Kind::Multiply
            ) {
                continue;
            }
            let mut roles = if let Some(layer) = descriptor.layer().filter(|l| {
                l.is_attention()
                    && matches!(
                        kind,
                        Vllm08Kind::AttentionQk
                            | Vllm08Kind::KvAppend
                            | Vllm08Kind::FlashAttention
                            | Vllm08Kind::Sigmoid
                            | Vllm08Kind::Multiply
                    )
            }) {
                attention
                    .get(&layer.index())
                    .ok_or_else(|| Error::State("B4 original attention binding absent".into()))?
                    .roles(index)?
            } else {
                BTreeMap::new()
            };
            for input in inputs {
                let actual = &input.record().plan().nodes()[index];
                let PlanOperation::TypedVllm08 {
                    descriptor: actual_descriptor,
                    ..
                } = &actual.operation
                else {
                    return Err(Error::State("B4 every real native source node".into()));
                };
                actual_descriptor.validate()?;
                if actual_descriptor.call() != input.call()
                    || actual_descriptor.kind() != kind
                    || actual_descriptor.layer() != descriptor.layer()
                    || actual.inputs != node.inputs
                    || actual.outputs != node.outputs
                {
                    return Err(Error::State(
                        "B4 per-row native descriptor/producers differ".into(),
                    ));
                }
                if let Vllm08Kind::Projection(projection) = kind {
                    let weight = if projection == Vllm08Projection::Head {
                        let matrices = actual_descriptor.matrices();
                        if matrices.len() != 1
                            || !Arc::ptr_eq(input.source().source(), tied.source())
                        {
                            return Err(Error::State(
                                "B4 full Head original tied source role".into(),
                            ));
                        }
                        input
                            .source()
                            .source()
                            .validate_matrix(QwenMatrixRole::Output, &matrices[0])?;
                        input
                            .source()
                            .source()
                            .validate_matrix(QwenMatrixRole::Embedding, &matrices[0])?;
                        tied.matrix().require(
                            input.source().source().source(),
                            values.service(),
                            248320,
                            input.call().model_profile().hidden_size(),
                        )?;
                        tied.matrix().lease()
                    } else {
                        library
                            .weights()
                            .require_operator(actual_descriptor)?
                            .lease()
                    };
                    if let Some(old) = roles.insert(Role::W, weight.clone())
                        && old.id() != weight.id()
                    {
                        return Err(Error::State("B4 rows differ in immutable W owner".into()));
                    }
                }
            }
            if matches!(kind, Vllm08Kind::Projection(_) | Vllm08Kind::SiluAndMul) {
                let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State(
                        "B4 native actual unary projection/ACT SSA".into(),
                    ));
                };
                roles.insert(
                    Role::X,
                    if kind == Vllm08Kind::Projection(Vllm08Projection::Head) {
                        head.lease().clone()
                    } else {
                        values.lease(*input)?.clone()
                    },
                );
                roles.insert(Role::Y, values.lease(*output)?.clone());
            }
            roles.insert(Role::Workspace, library.workspace().clone());
            nodes.insert(index, library.bind(inputs, index, &roles, status.clone())?);
        }
        if nodes.len() != 169 || attention.len() != 6 {
            return Err(Error::State(
                "B4 all115 projection/24ACT/30attention sites".into(),
            ));
        }
        Ok(Arc::new(Self {
            values,
            _library: library,
            _head: head,
            _tied: tied,
            attention,
            nodes,
        }))
    }
    pub(super) fn contains(&self, node: usize) -> bool {
        self.nodes.contains_key(&node)
    }
    pub(super) fn enqueue(
        &self,
        node: usize,
        permits: &[PrefixPhaseEnqueue<'_>],
        effects: &[StateEffect],
        cancelled: Option<&AtomicBool>,
    ) -> Result<(usize, usize, usize, usize)> {
        self.values.service().ensure_usable()?;
        let native = self
            .nodes
            .get(&node)
            .ok_or_else(|| Error::State("B4 actual native node absent".into()))?;
        let mut copies = 0;
        let mut grouped = 0;
        // Each adapter ignores unrelated node indices; KV copies validate all
        // per-row current state/effects before touching a shared native page.
        for attention in self.attention.values() {
            let (launches, transports) = attention.before(node, permits, effects)?;
            copies += launches;
            grouped += transports;
        }
        let (n, c, z) = native.enqueue(node, permits, cancelled)?;
        copies += c;
        for attention in self.attention.values() {
            let (launches, transports) = attention.after(node, permits, effects)?;
            copies += launches;
            grouped += transports;
        }
        Ok((n, copies, z, grouped))
    }
    /// Per-node census over the 169 native sites plus the attention grouped KV
    /// transport attribution at each layer's KvAppend source node.
    pub(super) fn census(&self) -> Result<CohortNodeCensus> {
        let mut result = Vec::with_capacity(self.nodes.len());
        for (&node, site) in &self.nodes {
            result.push((node, site.accesses().to_vec()));
        }
        for attention in self.attention.values() {
            let node = attention.kv_append_node();
            let entry = result
                .iter_mut()
                .find(|(at, _)| *at == node)
                .ok_or_else(|| Error::State("B4 census KV-append native site absent".into()))?;
            entry.1.extend(attention.kv_transport_accesses());
        }
        Ok(result)
    }
    pub(super) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.nodes
            .values()
            .flat_map(|node| node.accesses().iter().cloned())
            .chain(self.attention.values().flat_map(|a| a.accesses()))
            .collect()
    }
}
