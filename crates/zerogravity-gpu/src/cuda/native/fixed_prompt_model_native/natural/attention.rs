//! Current source Q/K/KV/FA/gates, pooled by finite geometry on the same set.
use super::*;
use crate::cuda::native::{
    attention_cubin::{
        ModelAttentionParameters,
        fixed_prompt_natural::{self as packets, Role, Stage},
    },
    model_cache::SharedModelCache,
    model_kv_transport::fixed_prompt_natural::{FixedPromptKvStorage, FixedPromptKvTransport},
    model_rotary::ModelRotary,
};
use crate::resident_leases::ResidentReadback;
use crate::resident_program::text_positions::fixed_prompt_natural::{
    FixedPromptPositions, FixedPromptPositionsStorage,
};
use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;

struct Controls {
    bytes: Arc<BTreeMap<Role, Vec<u8>>>,
    leases: BTreeMap<Role, ResidentLease>,
}
pub(crate) struct AttentionBuffers {
    parameters: BTreeMap<usize, Arc<ModelAttentionParameters>>,
    caches: BTreeMap<usize, Arc<SharedModelCache>>,
    kv: BTreeMap<usize, Arc<FixedPromptKvStorage>>,
    controls: BTreeMap<usize, Controls>,
    _positions: Arc<FixedPromptPositionsStorage>,
    rotary: Arc<ModelRotary>,
    lse: [ResidentLease; 2],
    private_o: ResidentLease,
    _metadata: Arc<WindowWorkspace>,
}
struct Launch {
    roles: Vec<Role>,
    kernel: Arc<OwnedCudaKernel>,
}
pub(crate) struct AttentionStorage {
    buffers: Arc<AttentionBuffers>,
    kernels: BTreeMap<(usize, Stage), Vec<Launch>>,
    images: Arc<PreparedImages>,
}
struct Site {
    nodes: [usize; 7],
    roles: BTreeMap<Role, ResidentLease>,
    transport: Arc<FixedPromptKvTransport>,
}
pub(crate) struct AttentionCall {
    storage: Arc<AttentionStorage>,
    input: Arc<FixedPromptCallInput>,
    state: Arc<FixedPromptCallState>,
    values: Arc<ModelValues>,
    positions: Arc<FixedPromptPositions>,
    sites: BTreeMap<usize, Site>,
    index: BTreeMap<usize, (usize, Stage)>,
    _metadata: WindowWorkspace,
}
fn site(
    record: &crate::resident_program::source_bindings::SourceBindings,
    layer: usize,
) -> Result<[usize; 7]> {
    ModelAttentionParameters::nodes(record, layer)
}

impl AttentionBuffers {
    pub(crate) fn prepare(
        source: &VerifiedTensorSource,
        parameters: BTreeMap<usize, Arc<ModelAttentionParameters>>,
        caches: BTreeMap<usize, Arc<SharedModelCache>>,
        positions: Arc<FixedPromptPositionsStorage>,
        rotary: Arc<ModelRotary>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let layers = [3, 7, 11, 15, 19, 23];
        if parameters.keys().copied().collect::<Vec<_>>() != layers
            || caches.keys().copied().collect::<Vec<_>>() != layers
        {
            return Err(Error::State(
                "canonical six independent source attention owners".into(),
            ));
        }
        let metadata = Arc::new(source.reserve_workspace(256 * 1024)?);
        init.retain_completion_owner(metadata.clone())?;
        init.retain_completion_owner(positions.clone())?;
        init.retain_completion_owner(rotary.clone())?;
        for p in parameters.values() {
            init.retain_completion_owner(p.clone())?;
        }
        for c in caches.values() {
            init.retain_completion_owner(c.clone())?;
        }
        let mut kv = BTreeMap::new();
        for (layer, cache) in &caches {
            kv.insert(*layer, FixedPromptKvStorage::prepare(cache.clone(), init)?);
        }
        let mut controls = BTreeMap::new();
        for base in std::iter::once(0).chain(47..=53) {
            let rows = if base == 0 { 47 } else { 1 };
            let mut bytes = BTreeMap::from([
                (
                    Role::SlotsI64,
                    (0..rows)
                        .flat_map(|j| ((4 * 544 + base + j) as i64).to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
                (Role::KScale, 1f32.to_le_bytes().to_vec()),
                (Role::VScale, 1f32.to_le_bytes().to_vec()),
                (Role::DummyCuKI32, vec![0; 8]),
                (
                    Role::SeqUsedI32,
                    ((base + rows) as i32).to_le_bytes().to_vec(),
                ),
                (
                    Role::BlockTableI32,
                    [4i32, 0, 0, 0]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect(),
                ),
            ]);
            if base == 0 {
                bytes.insert(
                    Role::CuQI32,
                    [0i32, 47].into_iter().flat_map(i32::to_le_bytes).collect(),
                );
            }
            let bytes = Arc::new(bytes);
            init.retain_completion_owner(bytes.clone())?;
            let mut leases = BTreeMap::new();
            for (role, data) in bytes.iter() {
                leases.insert(*role, init.upload_bytes(data, BufferRole::Activation)?);
            }
            controls.insert(base, Controls { bytes, leases });
        }
        let lse = [
            init.allocate_empty(1504, BufferRole::Scratch)?,
            init.allocate_empty(32, BufferRole::Scratch)?,
        ];
        let private_o = init.allocate_empty(4096, BufferRole::Scratch)?;
        let result = Arc::new(Self {
            parameters,
            caches,
            kv,
            controls,
            _positions: positions,
            rotary,
            lse,
            private_o,
            _metadata: metadata,
        });
        // Controls' original Vecs must outlive actual asynchronous H2D; init keeps
        // the final owner before it can complete or any native record activates.
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
}
impl AttentionStorage {
    pub(crate) fn prepare(
        buffers: Arc<AttentionBuffers>,
        images: Arc<PreparedImages>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        init.retain_completion_owner(buffers.clone())?;
        init.retain_completion_owner(images.clone())?;
        let mut kernels = BTreeMap::new();
        let mut reused = BTreeMap::<Vec<u8>, Arc<OwnedCudaKernel>>::new();
        for base in std::iter::once(0).chain(47..=53) {
            let stages = if base == 0 {
                vec![
                    Stage::Qk,
                    Stage::Kv,
                    Stage::Fa,
                    Stage::Sigmoid,
                    Stage::Multiply,
                ]
            } else {
                vec![
                    Stage::Qkv,
                    Stage::Qk,
                    Stage::Kv,
                    Stage::Fa,
                    Stage::Sigmoid,
                    Stage::Multiply,
                    Stage::Out,
                ]
            };
            for stage in stages {
                let mut ls = Vec::new();
                let ps = if base == 0 {
                    packets::prefill_packets(stage)?
                } else {
                    packets::cached_packets(base, stage)?
                };
                for packet in ps {
                    not_cancelled(cancelled)?;
                    let image = images.image(packet.image)?;
                    let key = format!("{:?}{:?}", packet.image, packet.manifest).into_bytes();
                    let kernel = if let Some(k) = reused.get(&key) {
                        k.clone()
                    } else {
                        let k = unsafe { images.set().prepare(image, packet.manifest) }
                            .map_err(native_error)?;
                        init.retain_completion_owner(k.clone())?;
                        reused.insert(key, k.clone());
                        k
                    };
                    ls.push(Launch {
                        roles: packet.roles,
                        kernel,
                    });
                }
                kernels.insert((base, stage), ls);
            }
        }
        let result = Arc::new(Self {
            buffers,
            kernels,
            images,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn bind(
        self: &Arc<Self>,
        input: Arc<FixedPromptCallInput>,
        state: Arc<FixedPromptCallState>,
        values: Arc<ModelValues>,
        positions: Arc<FixedPromptPositions>,
        weights: &ModelWeights,
        workspace: &ResidentLease,
    ) -> Result<Arc<AttentionCall>> {
        if !Arc::ptr_eq(state.input(), &input)
            || !positions.retained(&input, values.service())
            || !Arc::ptr_eq(positions.lease().service(), values.service())
        {
            return Err(Error::State(
                "canonical current attention source/state/positions map".into(),
            ));
        }
        input.scope().require_values(&values)?;
        self.images.require_service(values.service())?;
        self.buffers
            .rotary
            .require_call(input.source(), input.call(), values.service())?;
        let metadata = input.source().reserve_workspace(512 * 1024)?;
        let mut sites = BTreeMap::new();
        let mut index = BTreeMap::new();
        for layer in [3, 7, 11, 15, 19, 23] {
            let nodes = site(input.record(), layer)?;
            let plan = input.record().plan();
            let qkv = &plan.nodes()[nodes[0]];
            let qk = &plan.nodes()[nodes[1]];
            let kv = &plan.nodes()[nodes[2]];
            let fa = &plan.nodes()[nodes[3]];
            let sig = &plan.nodes()[nodes[4]];
            let mul = &plan.nodes()[nodes[5]];
            let out = &plan.nodes()[nodes[6]];
            if qk.inputs != qkv.outputs
                || qk.outputs.len() != 3
                || kv.inputs != [qk.outputs[1], qkv.outputs[0]]
                || fa.inputs != [qk.outputs[0]]
                || sig.inputs != [qk.outputs[2]]
                || mul.inputs != [fa.outputs[0], sig.outputs[0]]
                || out.inputs != mul.outputs
            {
                return Err(Error::State(
                    "canonical exact original QKV/QK/KV/FA/two-input gate SSA".into(),
                ));
            }
            let domains = state.states()[layer].clone();
            domains.require_fixed_prompt_published(input.scope())?;
            self.buffers.caches[&layer].require_binding(input.scope(), &domains)?;
            let [qg, kg] = self.buffers.parameters[&layer]
                .require_fixed_prompt_call(&input, &values, nodes[1])?;
            let mut roles = BTreeMap::from([
                (Role::QkvInput, values.lease(qkv.inputs[0])?.clone()),
                (Role::PackedQkv, values.lease(qkv.outputs[0])?.clone()),
                (Role::Q, values.lease(qk.outputs[0])?.clone()),
                (Role::K, values.lease(qk.outputs[1])?.clone()),
                (Role::Gate, values.lease(qk.outputs[2])?.clone()),
                (Role::AttentionOutput, values.lease(fa.outputs[0])?.clone()),
                (Role::SigmoidGate, values.lease(sig.outputs[0])?.clone()),
                (Role::OutInput, values.lease(mul.outputs[0])?.clone()),
                (Role::Projected, values.lease(out.outputs[0])?.clone()),
                (Role::QGamma, qg),
                (Role::KGamma, kg),
                (Role::PositionsI64Pitch65, positions.lease().clone()),
                (
                    Role::CosSinSourcePrefix67,
                    self.buffers.rotary.lease().clone(),
                ),
                (
                    Role::OriginalKvState,
                    self.buffers.caches[&layer].lease().clone(),
                ),
                (
                    Role::LseF32,
                    self.buffers.lse[usize::from(input.rows() == 1)].clone(),
                ),
            ]);
            for (role, node_index) in [(Role::QkvW, nodes[0]), (Role::OutW, nodes[6])] {
                let Some(descriptor) = input.record().operator(node_index)? else {
                    return Err(Error::State(
                        "canonical current attention learned descriptor".into(),
                    ));
                };
                roles.insert(role, weights.require_operator(descriptor)?.lease().clone());
            }
            roles.insert(Role::Workspace, workspace.clone());
            if input.rows() == 1 {
                roles.insert(Role::PrivateO, self.buffers.private_o.clone());
            }
            for (r, l) in &self.buffers.controls[&input.base()].leases {
                roles.insert(*r, l.clone());
            }
            for stage in [
                Stage::Qkv,
                Stage::Qk,
                Stage::Kv,
                Stage::Fa,
                Stage::Sigmoid,
                Stage::Multiply,
                Stage::Out,
            ] {
                if input.base() == 0 && matches!(stage, Stage::Qkv | Stage::Out) {
                    continue;
                }
                let ps = if input.base() == 0 {
                    packets::prefill_packets(stage)?
                } else {
                    packets::cached_packets(input.base(), stage)?
                };
                for p in ps {
                    if p.roles.len() != p.manifest.bindings.len() {
                        return Err(Error::State("canonical attention role count".into()));
                    }
                    for (r, b) in p.roles.iter().zip(&p.manifest.bindings) {
                        let l = roles.get(r).ok_or_else(|| {
                            Error::State("canonical required attention role absent".into())
                        })?;
                        if l.bytes() as u64 != b.bytes
                            || l.source_identity() != input.call().source()
                            || !Arc::ptr_eq(l.service(), values.service())
                            || l.service_generation() != values.service().generation()
                        {
                            return Err(Error::State(
                                "canonical fresh attention exact role extent/generation".into(),
                            ));
                        }
                    }
                }
            }
            let transport = self.buffers.kv[&layer].bind(input.clone(), domains)?;
            for (stage, node) in [
                Stage::Qkv,
                Stage::Qk,
                Stage::Kv,
                Stage::Fa,
                Stage::Sigmoid,
                Stage::Multiply,
                Stage::Out,
            ]
            .into_iter()
            .zip(nodes)
            {
                index.insert(node, (layer, stage));
            }
            sites.insert(
                layer,
                Site {
                    nodes,
                    roles,
                    transport,
                },
            );
        }
        Ok(Arc::new(AttentionCall {
            storage: self.clone(),
            input,
            state,
            values,
            positions,
            sites,
            index,
            _metadata: metadata,
        }))
    }
}
impl AttentionCall {
    pub(crate) fn accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let mut all = BTreeMap::new();
        for s in self.sites.values() {
            for (role, l) in &s.roles {
                let write = matches!(
                    role,
                    Role::Q
                        | Role::K
                        | Role::Gate
                        | Role::AttentionOutput
                        | Role::SigmoidGate
                        | Role::OutInput
                        | Role::Projected
                        | Role::LseF32
                        | Role::PrivateO
                        | Role::OriginalKvState
                        | Role::Workspace
                );
                insert(
                    &mut all,
                    l.clone(),
                    if write {
                        DeviceAccess::Write
                    } else {
                        DeviceAccess::Read
                    },
                );
            }
            for (l, a) in s.transport.accesses() {
                insert(&mut all, l, a);
            }
        }
        for (l, a) in self.positions.accesses() {
            insert(&mut all, l, a);
        }
        Ok(all.into_values().collect())
    }
    pub(crate) fn enqueue(
        &self,
        index: usize,
        session: &ResidentSession,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Submitted> {
        not_cancelled(cancelled)?;
        self.input.require_session(session)?;
        self.positions.require_enqueue(session, permit, cancelled)?;
        let (layer, stage) = *self
            .index
            .get(&index)
            .ok_or_else(|| Error::State("current actual attention cursor absent".into()))?;
        let s = &self.sites[&layer];
        let mut result = Submitted::default();
        let effect = self.state.states()[layer].effect(false)?;
        if stage == Stage::Kv {
            result.copies += s.transport.before(permit, effect, cancelled)?;
        }
        for launch in &self.storage.kernels[&(self.input.base(), stage)] {
            let handles = launch
                .roles
                .iter()
                .map(|r| {
                    s.roles.get(r).map(ResidentLease::handle).ok_or_else(|| {
                        Error::State("current attention exact native role absent".into())
                    })
                })
                .collect::<Result<Vec<Handle>>>()?;
            unsafe { launch.kernel.enqueue(handles) }.map_err(native_error)?;
            result.native += 1;
        }
        if stage == Stage::Kv {
            result.copies += s.transport.after(permit, effect, cancelled)?;
        }
        if stage == Stage::Fa && self.input.rows() == 1 {
            not_cancelled(cancelled)?;
            let kernel = &self.storage.kernels[&(self.input.base(), stage)]
                .last()
                .unwrap()
                .kernel;
            unsafe {
                kernel.copy_fixed_prompt_attention_output(
                    s.roles[&Role::PrivateO].handle(),
                    s.roles[&Role::AttentionOutput].handle(),
                )
            }
            .map_err(native_error)?;
            result.copies += 1;
        }
        Ok(result)
    }
    pub(crate) fn extra_readbacks(&self) -> Result<Vec<ResidentReadback>> {
        let mut out = vec![self.positions.readback()?];
        for s in self.sites.values() {
            out.push(s.transport.status_readback()?);
        }
        Ok(out)
    }
    pub(crate) fn validate_extra(&self, raw: &[Bytes]) -> Result<()> {
        if raw.len() != 7 {
            return Err(Error::State(
                "canonical positions plus six KV statuses".into(),
            ));
        }
        self.positions.validate(&raw[0])?;
        for b in &raw[1..] {
            crate::cuda::native::model_kv_transport::ModelKvTransport::validate_status(b)?;
        }
        Ok(())
    }
    pub(crate) fn retained(&self) -> Result<bool> {
        for (layer, s) in &self.sites {
            if s.nodes != site(self.input.record(), *layer)?
                || !s
                    .transport
                    .retained(&self.input, &self.state.states()[*layer])?
            {
                return Ok(false);
            }
        }
        Ok(self.positions.retained(&self.input, self.values.service()))
    }
}

impl AttentionStorage {
    pub(crate) fn immutable_controls(&self) -> Vec<(ResidentLease, &[u8])> {
        self.buffers
            .controls
            .values()
            .flat_map(|c| {
                c.bytes
                    .iter()
                    .map(|(r, b)| (c.leases[r].clone(), b.as_slice()))
            })
            .collect()
    }
    pub(super) unsafe fn retire(&self) -> Result<()> {
        let mut seen = std::collections::BTreeSet::new();
        for l in self.kernels.values().flatten() {
            if seen.insert(Arc::as_ptr(&l.kernel) as usize) {
                unsafe { l.kernel.retire_after_physical_completion() }.map_err(native_error)?;
            }
        }
        Ok(())
    }
}

impl AttentionStorage {
    pub(super) fn all_kernels(&self) -> Vec<Arc<OwnedCudaKernel>> {
        let mut seen = std::collections::BTreeSet::new();
        self.kernels
            .values()
            .flatten()
            .filter(|l| seen.insert(Arc::as_ptr(&l.kernel) as usize))
            .map(|l| l.kernel.clone())
            .collect()
    }
    pub(super) fn storage_accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut out = Vec::new();
        for c in self.buffers.caches.values() {
            out.push((c.lease().clone(), DeviceAccess::Write));
        }
        for k in self.buffers.kv.values() {
            out.push((k.status().clone(), DeviceAccess::Write));
        }
        for c in self.buffers.controls.values() {
            out.extend(c.leases.values().cloned().map(|l| (l, DeviceAccess::Read)));
        }
        out.extend(
            self.buffers
                .lse
                .iter()
                .cloned()
                .map(|l| (l, DeviceAccess::Write)),
        );
        out.push((self.buffers.private_o.clone(), DeviceAccess::Write));
        out
    }
}
