//! One finite47 record pool, fresh source-bound operands at each original node.
use super::*;
use crate::cuda::native::attention_cubin::{fixed_prompt_layer3_packets, fixed_prompt_packets};
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;

pub(super) struct Programs {
    cut: Arc<FixedPromptPrefixCut>,
    values: Arc<ModelValues>,
    attention: BTreeMap<usize, Arc<attention::Binding>>,
    attention_kernels: Vec<(Vec<fixed_prompt_packets::Role>, Arc<OwnedCudaKernel>)>,
    gates: [Arc<OwnedCudaKernel>; 2],
    counts: BTreeMap<usize, [AtomicUsize; 4]>,
    gate_counts: BTreeMap<usize, [AtomicUsize; 2]>,
    head: Arc<head::FixedPromptHeadPrograms>,
    _metadata: Arc<WindowWorkspace>,
}

fn require_packet_extents(
    packet: &Packet,
    node: &Node,
    values: &ModelValues,
    workspace: &ResidentLease,
) -> Result<()> {
    if packet.roles.len() != packet.manifest.bindings.len() {
        return Err(Error::State(
            "wholeInitial47 packet role count differs".into(),
        ));
    }
    for (role, binding) in packet.roles.iter().zip(&packet.manifest.bindings) {
        let bytes = match role {
            Binding::W => node
                .projection
                .as_ref()
                .ok_or_else(|| Error::State("whole actual W role absent".into()))?
                .weight()
                .bytes(),
            Binding::X => values.lease(node.input)?.bytes(),
            Binding::Y => values.lease(node.output)?.bytes(),
            Binding::Workspace => workspace.bytes(),
        };
        // Workspace packets may use only their declared finite prefix; public/W are exact.
        if (matches!(role, Binding::Workspace) && binding.bytes > bytes as u64)
            || (!matches!(role, Binding::Workspace) && binding.bytes != bytes as u64)
        {
            return Err(Error::State(
                "wholeInitial47 source role extent differs pooled record".into(),
            ));
        }
    }
    Ok(())
}
impl FixedPromptModelNativePrograms {
    #[expect(
        clippy::too_many_arguments,
        reason = "same original source/value/set/init owners"
    )]
    pub(crate) fn prepare_whole(
        cut: Arc<FixedPromptPrefixCut>,
        values: Arc<ModelValues>,
        weights: Arc<ModelWeights>,
        workspace: ResidentLease,
        images: Arc<PreparedImages>,
        attention: BTreeMap<usize, Arc<attention::Binding>>,
        head: Arc<head::FixedPromptHeadPrograms>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        cut.require_whole()?;
        head.require_images(&images)?;
        cut.scope().require_values(&values)?;
        weights.require_fixed_prompt_prefix(&cut)?;
        init.require_service(values.service())?;
        images.require_service(values.service())?;
        if attention.keys().copied().collect::<Vec<_>>() != [3, 7, 11, 15, 19, 23]
            || !head.retained_values(&cut, &values)?
            || !Arc::ptr_eq(workspace.service(), values.service())
            || workspace.service_generation() != values.service().generation()
            || workspace.source_identity() != values.service().source_identity()
            || workspace.bytes() != WORKSPACE_BYTES
            || workspace.role() != BufferRole::Scratch
        {
            return Err(Error::State(
                "wholeInitial47 exact native/source/workspace owner cohort".into(),
            ));
        }
        let metadata = Arc::new(cut.scope().source().reserve_workspace(2 * 1024 * 1024)?);
        init.retain_completion_owner(metadata.clone())?;
        init.retain_completion_owner(cut.clone())?;
        init.retain_completion_owner(values.clone())?;
        init.retain_completion_owner(weights.clone())?;
        init.retain_completion_owner(head.clone())?;
        for (layer, binding) in &attention {
            binding.require_values(&cut, &values)?;
            if binding.whole_site()?.layer() != *layer {
                return Err(Error::State(
                    "wholeInitial47 six exact source attention sites".into(),
                ));
            }
            init.retain_completion_owner(binding.clone())?;
        }
        let mut nodes = BTreeMap::new();
        let mut all = BTreeMap::new();
        let mut per_layer = BTreeMap::<usize, BTreeSet<Operation>>::new();
        for (index, node) in cut.nodes().iter().enumerate() {
            let Some(descriptor) = cut.scope().record().operator(index)? else {
                continue;
            };
            let Ok(operation) = Operation::of(descriptor) else {
                continue;
            };
            let layer = layer_of(cut.scope().record(), index)?;
            let ([x], [y]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                return Err(Error::State("fixed_prompt native actual unary SSA".into()));
            };
            let finite_site = layer < 24
                && if layer % 4 == 3 {
                    matches!(
                        operation,
                        Operation::AttentionQkv
                            | Operation::Out
                            | Operation::GateUp
                            | Operation::Activation
                            | Operation::Down
                    )
                } else {
                    operation != Operation::AttentionQkv
                };
            if !finite_site
                || descriptor.call() != cut.scope().record().call()
                || !per_layer.entry(layer).or_default().insert(operation)
            {
                return Err(Error::State(
                    "fixed_prompt native actual source/node/layer cohort".into(),
                ));
            }
            let (input, output) = (values.lease(*x)?, values.lease(*y)?);
            let projection = if operation == Operation::Activation {
                None
            } else {
                Some(
                    cut.scope()
                        .bind_projection(values.clone(), &weights, index)?,
                )
            };
            if input.id() == output.id()
                || input.id() == workspace.id()
                || output.id() == workspace.id()
                || input.bytes() != descriptor.input_elements()?[0] * 2
                || output.bytes() != descriptor.output_elements()?[0] * 2
            {
                return Err(Error::State(
                    "fixed_prompt native full actual producer extents/nonalias".into(),
                ));
            }
            if let Some(binding) = &projection {
                binding.require(&values, descriptor, input, output)?;
                if binding.weight().id() == workspace.id() {
                    return Err(Error::State("fixed_prompt W/workspace alias".into()));
                }
                insert(&mut all, binding.weight().clone(), DeviceAccess::Read);
            }
            insert(&mut all, input.clone(), DeviceAccess::Read);
            insert(&mut all, output.clone(), DeviceAccess::Write);
            nodes.insert(
                index,
                Node {
                    operation,
                    descriptor: descriptor.clone(),
                    gdn_output: operation == Operation::Out
                        && is_gdn_output(cut.scope().record(), *x)?,
                    input: *x,
                    output: *y,
                    projection,
                },
            );
        }
        let gdn_ops = Operation::ALL.into_iter().collect::<BTreeSet<_>>();
        let attention_ops = BTreeSet::from([
            Operation::AttentionQkv,
            Operation::Out,
            Operation::GateUp,
            Operation::Activation,
            Operation::Down,
        ]);
        if nodes.len() != 138
            || per_layer.len() != 24
            || (0..24).any(|layer| {
                per_layer.get(&layer)
                    != Some(if layer % 4 == 3 {
                        &attention_ops
                    } else {
                        &gdn_ops
                    })
            })
        {
            return Err(Error::State(
                "wholeInitial47 exact138 actual projection/activation sites".into(),
            ));
        }
        insert(&mut all, workspace.clone(), DeviceAccess::Write);
        init.retain_completion_owner(cut.clone())?;
        init.retain_completion_owner(values.clone())?;
        init.retain_completion_owner(weights.clone())?;
        let mut kernels = BTreeMap::new();
        let mut operations = Operation::ALL.to_vec();
        operations.push(Operation::AttentionQkv);
        for operation in operations {
            let mut prepared = Vec::new();
            let records = if operation == Operation::AttentionQkv {
                let binding = nodes
                    .get(&32)
                    .and_then(|node| node.projection.as_ref())
                    .ok_or_else(|| {
                        Error::State("fixed_prompt actual node32 projection owner absent".into())
                    })?;
                super::super::fixed_prompt_attention_projection::model_packets(binding)?
            } else {
                packets(operation)?
            };
            for packet in records {
                for node in nodes.values().filter(|node| node.operation == operation) {
                    require_packet_extents(&packet, node, &values, &workspace)?;
                }
                not_cancelled(cancelled)?;
                let image = match packet.image {
                    Some(kind) => images.image(kind)?,
                    None => images.activation()?,
                };
                // SAFETY: unchanged accepted finite47 packet, same original image/set/init.
                let kernel = unsafe { images.set().prepare(image, packet.manifest) }
                    .map_err(native_error)?;
                init.retain_completion_owner(kernel.clone())?;
                prepared.push(Kernel {
                    roles: packet.roles,
                    kernel,
                });
            }
            kernels.insert(operation, prepared);
        }
        let whole = Programs::prepare(
            cut.clone(),
            values.clone(),
            attention,
            head,
            metadata,
            &images,
            init,
            cancelled,
        )?;
        for (lease, access) in whole.accesses()? {
            insert(&mut all, lease, access);
        }
        if kernels.values().map(Vec::len).sum::<usize>() + whole.kernels().len() != 16 {
            return Err(Error::State(
                "wholeInitial47 exact16 shared native records".into(),
            ));
        }
        let result = Arc::new(Self {
            whole: Some(whole),
            attention: None,
            tail: None,
            tail_counts: std::array::from_fn(|_| AtomicUsize::new(0)),
            cut,
            values,
            weights,
            workspace,
            nodes,
            kernels,
            accesses: all.into_values().collect(),
            images,
            enqueued_records: AtomicUsize::new(0),
            qkv_enqueues: AtomicUsize::new(0),
            _metadata: None,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    fn whole(&self) -> Result<&Arc<Programs>> {
        self.whole
            .as_ref()
            .ok_or_else(|| Error::State("wholeInitial47 pooled binding absent".into()))
    }
    pub(crate) fn head(&self) -> Result<&Arc<head::FixedPromptHeadPrograms>> {
        Ok(&self.whole()?.head)
    }
    pub(crate) fn attention_bindings(&self) -> Result<&BTreeMap<usize, Arc<attention::Binding>>> {
        Ok(&self.whole()?.attention)
    }
    pub(crate) fn whole_counts(&self) -> Result<[[usize; 4]; 6]> {
        let whole = self.whole()?;
        Ok([3, 7, 11, 15, 19, 23].map(|l| {
            whole.counts[&l]
                .each_ref()
                .map(|c| c.load(Ordering::Acquire))
        }))
    }
    pub(crate) fn whole_outputs_unpromoted(&self) -> Result<bool> {
        self.whole()?.outputs_unpromoted()
    }
    pub(crate) fn whole_graph_retained(&self) -> Result<bool> {
        self.whole()?.head.require_images(&self.images)?;
        self.whole()?.require_values(&self.cut, &self.values)?;
        Ok(self.weights.source().same_issuer(self.cut.scope().source())
            && self.nodes.len() == 138
            && self.kernels.values().map(Vec::len).sum::<usize>() == 10
            && self.whole()?.kernels().len() == 6)
    }
    pub(crate) fn enqueue_whole(
        &self,
        index: usize,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
        fault: usize,
    ) -> Result<usize> {
        self.cut.require_whole()?;
        if !matches!(fault, 0 | 17..=21) {
            return Err(Error::State("wholeInitial47 fault scope".into()));
        }
        if self.nodes.contains_key(&index) {
            return self.enqueue_inner(index, cancelled, 0);
        }
        self.whole()?
            .enqueue(index, permit, cancelled, fault, &self.enqueued_records)
    }
}
impl Programs {
    #[expect(
        clippy::too_many_arguments,
        reason = "same original initializer and retained pool owners"
    )]
    fn prepare(
        cut: Arc<FixedPromptPrefixCut>,
        values: Arc<ModelValues>,
        attention: BTreeMap<usize, Arc<attention::Binding>>,
        head: Arc<head::FixedPromptHeadPrograms>,
        metadata: Arc<WindowWorkspace>,
        images: &Arc<PreparedImages>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        let mut attention_kernels = Vec::with_capacity(3);
        for packet in fixed_prompt_packets::packets_with_fa(true)? {
            not_cancelled(cancelled)?;
            let image = images.image(packet.image)?;
            // SAFETY: original set/init; each enqueue supplies its sealed site operands.
            let kernel =
                unsafe { images.set().prepare(image, packet.manifest) }.map_err(native_error)?;
            init.retain_completion_owner(kernel.clone())?;
            attention_kernels.push((packet.roles, kernel));
        }
        let mut gates = Vec::with_capacity(2);
        for (kind, manifest) in fixed_prompt_layer3_packets::packets()? {
            not_cancelled(cancelled)?;
            let image = images.image(kind)?;
            // SAFETY: same accepted finite47 Sig/Mul packets and original set/init.
            let kernel = unsafe { images.set().prepare(image, manifest) }.map_err(native_error)?;
            init.retain_completion_owner(kernel.clone())?;
            gates.push(kernel);
        }
        if attention_kernels.len() != 3 {
            return Err(Error::State("wholeInitial47 QK/KV/FA record count".into()));
        }
        let gates: [_; 2] = gates
            .try_into()
            .map_err(|_| Error::State("wholeInitial47 gate record count".into()))?;
        let result = Arc::new(Self {
            cut,
            values,
            attention,
            attention_kernels,
            gates,
            head,
            _metadata: metadata,
            counts: [3, 7, 11, 15, 19, 23]
                .into_iter()
                .map(|l| (l, std::array::from_fn(|_| AtomicUsize::new(0))))
                .collect(),
            gate_counts: [3, 7, 11, 15, 19, 23]
                .into_iter()
                .map(|l| (l, std::array::from_fn(|_| AtomicUsize::new(0))))
                .collect(),
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    fn accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let mut all = BTreeMap::new();
        for binding in self.attention.values() {
            for (lease, access) in binding.accesses() {
                insert(&mut all, lease, access);
            }
            let ids = binding.whole_site()?.values();
            for (id, access) in [
                (ids[3], DeviceAccess::Read),
                (ids[4], DeviceAccess::Read),
                (ids[5], DeviceAccess::Write),
                (ids[6], DeviceAccess::Write),
            ] {
                insert(&mut all, self.values.lease(id)?.clone(), access);
            }
        }
        for (lease, access) in self.head.accesses() {
            insert(&mut all, lease, access);
        }
        Ok(all.into_values().collect())
    }
    pub(super) fn require_values(
        &self,
        cut: &Arc<FixedPromptPrefixCut>,
        values: &Arc<ModelValues>,
    ) -> Result<()> {
        if !Arc::ptr_eq(cut, &self.cut)
            || !Arc::ptr_eq(values, &self.values)
            || !self.head.retained_values(cut, values)?
            || self.attention.len() != 6
        {
            return Err(Error::State(
                "wholeInitial47 immutable pool graph differs".into(),
            ));
        }
        cut.require_whole()?;
        cut.scope().require_values(values)?;
        for (layer, binding) in &self.attention {
            if binding.whole_site()?.layer() != *layer
                || !binding.whole_graph_retained(cut, values)?
            {
                return Err(Error::State(
                    "wholeInitial47 retained attention source graph differs".into(),
                ));
            }
        }
        Ok(())
    }
    fn outputs_unpromoted(&self) -> Result<bool> {
        for id in 1..327 {
            if self.values.lease(ValueId(id))?.is_initialized() {
                return Ok(false);
            }
        }
        for binding in self.attention.values() {
            if !binding.fa_output_unpromoted()? {
                return Ok(false);
            }
        }
        self.head.output_unpromoted()
    }
    pub(super) fn kernels(&self) -> Vec<Arc<OwnedCudaKernel>> {
        self.attention_kernels
            .iter()
            .map(|(_, k)| k.clone())
            .chain(self.gates.iter().cloned())
            .chain(std::iter::once(self.head.kernel().clone()))
            .collect()
    }
    fn enqueue(
        &self,
        index: usize,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
        fault: usize,
        total: &AtomicUsize,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        if index == 254 {
            let late = AtomicBool::new(fault == 20);
            not_cancelled(if late.load(Ordering::Acquire) {
                Some(&late)
            } else {
                cancelled
            })?;
            let count = self.head.enqueue(index, cancelled)?;
            total.fetch_add(count, Ordering::AcqRel);
            if fault == 21 {
                return Err(Error::State(
                    "deliberate fixed_prompt whole unknown after Head".into(),
                ));
            }
            return Ok(count);
        }
        let (layer, binding, position) = self
            .attention
            .iter()
            .find_map(|(layer, binding)| {
                let nodes = binding.whole_site().ok()?.nodes();
                nodes
                    .iter()
                    .position(|i| *i == index)
                    .map(|position| (*layer, binding, position))
            })
            .ok_or_else(|| Error::State("wholeInitial47 actual attention cursor absent".into()))?;
        binding.require_enqueue(permit, cancelled)?;
        let counts = &self.counts[&layer];
        let actual = counts.each_ref().map(|c| c.load(Ordering::Acquire));
        let expected = match position {
            1 => [0, 0, 0, 0],
            2 => [1, 0, 0, 0],
            3..=5 => [1, 1, 1, usize::from(position >= 4)],
            _ => {
                return Err(Error::State(
                    "wholeInitial47 finite attention record cursor".into(),
                ));
            }
        };
        if actual != expected || !self.outputs_unpromoted()? {
            return Err(Error::State(
                "wholeInitial47 actual attention order/output state differs".into(),
            ));
        }
        if position <= 3 {
            let slot = position - 1;
            let mut transports = 0;
            if position == 2 {
                transports += binding.before_whole_append(permit, cancelled)?;
            }
            let (roles, kernel) = &self.attention_kernels[slot];
            let handles = binding.handles(roles)?;
            // SAFETY: original body armed flight owns this exact source/site/state graph.
            unsafe { kernel.enqueue(handles) }.map_err(native_error)?;
            counts[if position == 3 { 3 } else { slot }].fetch_add(1, Ordering::AcqRel);
            total.fetch_add(1, Ordering::AcqRel);
            if position == 2 {
                let outgoing = binding.after_whole_append(permit, cancelled)?;
                counts[2].fetch_add(outgoing, Ordering::AcqRel);
                transports += outgoing;
            }
            if position == 3 && layer == 23 && fault == 19 {
                return Err(Error::State(
                    "deliberate fixed_prompt whole unknown after last FA".into(),
                ));
            }
            return Ok(1 + transports);
        }
        let which = position - 4;
        let gates = &self.gate_counts[&layer];
        if gates.each_ref().map(|c| c.load(Ordering::Acquire))
            != if which == 0 { [0, 0] } else { [1, 0] }
        {
            return Err(Error::State(
                "wholeInitial47 exact Sigmoid/Multiply order".into(),
            ));
        }
        let ids = binding.whole_site()?.values();
        let operands = if which == 0 {
            vec![ids[5], ids[3]]
        } else {
            vec![ids[6], ids[4], ids[5]]
        };
        let handles = operands
            .into_iter()
            .map(|id| self.values.lease(id).map(ResidentLease::handle))
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: same finite Sig/Mul records; actual two-input Mul SSA is sealed above.
        unsafe { self.gates[which].enqueue(handles) }.map_err(native_error)?;
        gates[which].fetch_add(1, Ordering::AcqRel);
        total.fetch_add(1, Ordering::AcqRel);
        Ok(1)
    }
    pub(super) unsafe fn retire(&self) -> Result<()> {
        for (_, kernel) in &self.attention_kernels {
            unsafe { kernel.retire_after_physical_completion() }.map_err(native_error)?;
        }
        for kernel in &self.gates {
            unsafe { kernel.retire_after_physical_completion() }.map_err(native_error)?;
        }
        // Head retires its kernel only; the parent retires the single shared set last.
        unsafe { self.head.retire() }
    }
}
