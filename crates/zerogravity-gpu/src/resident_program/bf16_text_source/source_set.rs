//! Complete source-step lowering with one shared workspace/resident matrix set.
//! Position records cover the arithmetic domain, not captured prompts or cases.
use super::*;
use layout::{Layout, aligned, sum};

#[derive(Clone, Debug)]
pub struct SourceSetInventory {
    pub resident_weight_bytes: usize,
    pub workspace_bytes: usize,
    pub root_bytes: usize,
    pub root_count: usize,
    pub model_device_upper_bytes: usize,
    pub context_device_bytes: usize,
    pub physical_device_upper_bytes: usize,
    pub host_upper_bytes: usize,
}
pub(crate) struct PendingSourceSet {
    bodies: Vec<PendingBf16TextSource>,
    matrices: Vec<(
        Arc<PreparedBf16Projection>,
        Option<crate::resident_leases::resident_matrix::PendingResidentMatrix>,
    )>,
}
impl PendingSourceSet {
    pub(crate) fn bind(self) -> Result<Vec<Arc<Bf16TextSourceBody>>> {
        // Upload coverage does not publish initialization. The original Ready
        // barrier must mark every lease before any operand/body can bind.
        for (operand, pending) in self.matrices {
            operand.bind_initialized(pending)?;
        }
        self.bodies
            .into_iter()
            .map(|pending| pending.bind().map(|(_, body)| body))
            .collect()
    }
}
pub(crate) struct SourceStepSet {
    source: VerifiedTensorSource,
    records: Vec<Arc<TextStepBindings>>,
    layouts: Vec<Layout>,
    operands: Arc<BTreeMap<String, Arc<PreparedBf16Projection>>>,
    slots: Vec<usize>,
    scratch: Vec<usize>,
    max_rows: usize,
    inventory: SourceSetInventory,
    _metadata: WindowWorkspace,
}
fn same_states(a: &[StateRequirement], b: &[StateRequirement]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            a.slot == b.slot
                && a.physical == b.physical
                && a.bytes == b.bytes
                && match (a.layout, b.layout) {
                    (StateLayout::Vector { elements: a }, StateLayout::Vector { elements: b }) => {
                        a == b
                    }
                    (
                        StateLayout::Kv {
                            slab: a,
                            capacity: x,
                        },
                        StateLayout::Kv {
                            slab: b,
                            capacity: y,
                        },
                    ) => a == b && x == y,
                    _ => false,
                }
        })
}
impl SourceStepSet {
    pub(crate) fn new(
        source: VerifiedTensorSource,
        records: Vec<Arc<TextStepBindings>>,
        root_count: usize,
    ) -> Result<Self> {
        if records.is_empty()
            || records.len() > 64
            || !(2..=258).contains(&root_count)
            || records[0].capacity() != records.len()
        {
            return Err(Error::Limit("complete source step/root population".into()));
        }
        let metadata = source.reserve_workspace(4 * 1024 * 1024)?;
        let mut layouts: Vec<Layout> = Vec::with_capacity(records.len());
        let mut operands = BTreeMap::<String, Arc<PreparedBf16Projection>>::new();
        let mut slots = Vec::<usize>::new();
        let mut scratch = vec![0; math::scratch_layout().len()];
        for (position, record) in records.iter().enumerate() {
            if record.position() != position
                || record.capacity() != records.len()
                || record.geometry() != records[0].geometry()
            {
                return Err(Error::State(
                    "source records cover each actual frontier exactly".into(),
                ));
            }
            require_source_bindings(&source, record)?;
            let l = Layout::derive_source(record)?;
            if let Some(first) = layouts.first()
                && (first.parameters != l.parameters || !same_states(&first.states, &l.states))
            {
                return Err(Error::State(
                    "source geometry/parameter identity changes across positions".into(),
                ));
            }
            slots.resize(slots.len().max(l.slots.len()), 0);
            for (all, n) in slots.iter_mut().zip(&l.slots) {
                *all = (*all).max(*n);
            }
            for (all, n) in scratch.iter_mut().zip(&l.math) {
                *all = (*all).max(*n);
            }
            for node in record.plan().nodes() {
                if let PlanOperation::TypedProject { tensor, .. }
                | PlanOperation::TypedEmbed { tensor, .. } = &node.operation
                {
                    let matrix = source.matrix(&tensor.name, tensor.axes)?;
                    if let Some(old) = operands.get(&tensor.name) {
                        if !old.matrix().same_binding(&matrix) {
                            return Err(Error::State("source matrix identity changes".into()));
                        }
                    } else {
                        operands.insert(
                            tensor.name.clone(),
                            PreparedBf16Projection::new(
                                source.clone(),
                                matrix,
                                Bf16ProjectionStorage::Resident,
                            )?,
                        );
                    }
                }
            }
            layouts.push(l);
        }
        let first = &layouts[0];
        let max_rows = operands
            .values()
            .map(|p| p.max_launch_rows())
            .max()
            .ok_or_else(|| Error::State("source matrices absent".into()))?;
        let weights = sum(operands
            .values()
            .map(|p| {
                p.matrix().byte_range().and_then(|r| {
                    usize::try_from(r.end - r.start)
                        .map_err(|_| Error::Limit("source resident weight extent".into()))
                })
            })
            .collect::<Result<Vec<_>>>()?)?;
        let tables = sum(math::immutable_tables()
            .iter()
            .map(|t| aligned(t.len()))
            .collect::<Result<Vec<_>>>()?)?;
        let max_columns = layouts.iter().map(|l| l.max_columns).max().unwrap();
        let owned = sum([
            sum(slots.iter().copied())?,
            first.parameter_bytes,
            sum(scratch.iter().copied())?,
            source_page::PAGE_BYTES,
            aligned(max_columns * 2)?,
            2 * aligned(max_rows * 4)?,
            2 * layout::ALIGNMENT,
            tables,
            weights,
        ])?;
        let roots = first
            .root_bytes
            .checked_mul(root_count)
            .ok_or_else(|| Error::Limit("source root pool total".into()))?;
        let (native_host, native_device) = crate::hip::native::source_preparation_upper_bytes();
        let root_planes = first
            .states
            .iter()
            .map(|s| s.layout.planes().map(|p| p.0))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .sum();
        let root_metadata = crate::resident_leases::backing_pool::ResidentPool::metadata_bytes(
            root_planes,
            root_count,
        )?;
        let host = sum([
            source.metadata_reserved_bytes() as usize,
            source.staging_chunk_bytes() * 3,
            12 * 1024 * 1024,
            records.len() * (layout::RECORD_BYTES + layout::METADATA_BYTES),
            1024 * 1024,
            (root_count / 2 + 1) * 32 * 1024 * 1024,
            6 * first.root_bytes,
            root_metadata,
            source_page::transport_host_bytes(&source)?,
            3 * first.parameter_bytes,
            native_host,
            2 * 1024 * 1024 * 1024,
        ])?;
        if host as u64 > source.workspace_limit_bytes()? {
            return Err(Error::Limit(
                "source resident complete host preflight".into(),
            ));
        }
        let model_device_upper_bytes = sum([owned, roots, native_device])?;
        let (context_device_bytes, physical_device_upper_bytes) =
            crate::hip::native::resident_source_device_inventory(model_device_upper_bytes)?;
        let inventory = SourceSetInventory {
            resident_weight_bytes: weights,
            workspace_bytes: owned - weights,
            root_bytes: first.root_bytes,
            root_count,
            model_device_upper_bytes,
            context_device_bytes,
            physical_device_upper_bytes,
            host_upper_bytes: host,
        };
        Ok(Self {
            source,
            records,
            layouts,
            operands: Arc::new(operands),
            slots,
            scratch,
            max_rows,
            inventory,
            _metadata: metadata,
        })
    }
    pub(crate) fn inventory(&self) -> &SourceSetInventory {
        &self.inventory
    }
    pub(crate) fn catalogue(
        &self,
        client: &Client,
    ) -> Result<(
        Vec<Box<dyn cubecl_server::kernel::CubeKernel>>,
        WindowWorkspace,
    )> {
        let c = math::prepared::enumerate_records(
            client,
            &self.source,
            &self.records,
            false,
            Some(&self.operands),
        )?;
        Ok((c.kernels, c.metadata))
    }
    pub(crate) async fn prepare(
        self,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        native: Arc<crate::hip::native::PreparedSourceNative>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<PendingSourceSet> {
        init.require_service(&service)?;
        native.require_live(service.client(), self.source.source_identity())?;
        if service.source_identity() != self.source.source_identity()
            || service.client().is_capturing()
        {
            return Err(Error::State(
                "source-set service/source/capture state differs".into(),
            ));
        }
        if self.inventory.model_device_upper_bytes > service.capacity() {
            return Err(Error::resource(
                "GPU provider memory / VRAM budget",
                "source-set resident model/native/root admission",
                self.inventory.model_device_upper_bytes as u64,
                service.capacity() as u64,
            ));
        }
        // Retain compilation/source authority before the first device transfer.
        init.retain_completion_owner(native.clone())?;
        let mut matrices = Vec::with_capacity(self.operands.len());
        for operand in self.operands.values() {
            operand.require_capability(&service)?;
            let pending = operand.upload(init, cancelled).await?;
            matrices.push((operand.clone(), pending));
        }
        let page =
            source_page::ResidentSourcePage::allocate(self.source.clone(), service.clone(), init)?;
        let frame = ResidentFrameCheckpoint::new(self.source.clone())?;
        let parameters = upload_parameters(&self.source, &self.layouts[0].parameters, init)?;
        let slots = self
            .slots
            .iter()
            .map(|n| init.allocate_empty(*n, BufferRole::Activation))
            .collect::<Result<Vec<_>>>()?;
        let scratch = self
            .scratch
            .iter()
            .map(|n| init.allocate_empty(*n, BufferRole::Scratch))
            .collect::<Result<Vec<_>>>()?;
        let tables = math::immutable_tables()
            .into_iter()
            .map(|mut bytes| {
                bytes.resize(aligned(bytes.len())?, 0);
                init.upload_bytes(&bytes, BufferRole::ImmutableWeights)
            })
            .collect::<Result<Vec<_>>>()?;
        let max_columns = self.layouts.iter().map(|l| l.max_columns).max().unwrap();
        let packed = init.allocate_empty(aligned(max_columns * 2)?, BufferRole::Scratch)?;
        let accumulation = init.allocate_empty(aligned(self.max_rows * 4)?, BufferRole::Scratch)?;
        let projection_flags =
            init.allocate_empty(aligned(self.max_rows * 4)?, BufferRole::Scratch)?;
        let status = init.allocate_empty(layout::ALIGNMENT, BufferRole::Scratch)?;
        let zero = init.upload_zeroed(layout::ALIGNMENT, BufferRole::ImmutableWeights)?;
        let mut accesses = Vec::new();
        accesses.extend(
            slots
                .iter()
                .chain(&scratch)
                .cloned()
                .map(|l| (l, DeviceAccess::Write)),
        );
        accesses.extend(
            parameters
                .values()
                .chain(&tables)
                .chain([&zero])
                .cloned()
                .map(|l| (l, DeviceAccess::Read)),
        );
        accesses.extend(
            [
                &packed,
                &accumulation,
                &projection_flags,
                &status,
                page.lease(),
            ]
            .into_iter()
            .cloned()
            .map(|l| (l, DeviceAccess::Write)),
        );
        for (_, pending) in &matrices {
            accesses.push((
                pending
                    .as_ref()
                    .ok_or_else(|| Error::State("source resident upload absent".into()))?
                    .lease()
                    .clone(),
                DeviceAccess::Read,
            ));
        }
        let owned = sum(accesses.iter().map(|(l, _)| l.bytes()))?;
        if owned != self.inventory.resident_weight_bytes + self.inventory.workspace_bytes {
            return Err(Error::State(
                "source actual shared ownership differs from admission".into(),
            ));
        }
        let mut bodies = Vec::with_capacity(self.records.len());
        for (recorded, mut l) in self.records.into_iter().zip(self.layouts) {
            l.slots = self.slots.clone();
            l.value_bytes = sum(self.slots.iter().copied())?;
            l.math = self.scratch.clone();
            l.owned_bytes = owned;
            let values = recorded
                .plan()
                .values()
                .iter()
                .zip(&l.bindings)
                .map(|(value, b)| {
                    Ok(DeviceBuffer::new(
                        trimmed(&slots[b.slot], 0, b.bytes)?,
                        value.capacity_elements()?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            let body = Arc::new(Bf16TextSourceBody {
                values,
                service: service.clone(),
                recorded,
                source: self.source.clone(),
                layout: l,
                slots: slots.clone(),
                parameters: parameters.clone(),
                projections: self.operands.clone(),
                scratch: scratch.clone(),
                tables: tables.clone(),
                page: page.clone(),
                frame: frame.clone(),
                packed: packed.clone(),
                accumulation: accumulation.clone(),
                projection_flags: projection_flags.clone(),
                status: status.clone(),
                zero: zero.clone(),
                accesses: accesses.clone(),
                epochs: Mutex::new(Vec::new()),
                next: AtomicUsize::new(0),
                _metadata: self.source.reserve_workspace(layout::METADATA_BYTES)?,
                _native: native.clone(),
            });
            init.retain_completion_owner(body.clone())?;
            bodies.push(PendingBf16TextSource { body });
        }
        Ok(PendingSourceSet { bodies, matrices })
    }
}
