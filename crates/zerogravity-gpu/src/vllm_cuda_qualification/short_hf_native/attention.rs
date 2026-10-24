//! Actual six source QK parameters/caches and fresh current-call transport.
use super::super::{
    model_rotary::ModelRotary, model_values::ModelValues, projection_cubin::not_cancelled,
    short_hf_model::ShortHfAttentionSite,
};
use super::{
    ShortHfNativeGeometry,
    packets::{Catalogue, Key, Role},
    transport::{CACHE_BYTES, Definitions, Transport},
};
use crate::{
    resident_inference::{
        ResidentSession,
        native_fragment::{
            PrefixPhaseEnqueue,
            model_state::BoundModelState,
            short_hf::{ShortHfCallInput, ShortHfCallState},
        },
    },
    resident_leases::{ResidentInitialization, ResidentLease, ResidentReadback, ResidentService},
    resident_program::text_positions::short_hf::{ShortHfPositions, ShortHfPositionsStorage},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use zerogravity_artifacts::source_weights::{ParameterWindow, WindowWorkspace};
use zerogravity_core::{
    Error, Result,
    device_events::{BufferRole, DeviceAccess},
};
use zerogravity_model_qwen35::original_binding::{BoundFullModelSource, QwenParameterRole};

const LAYERS: [usize; 6] = [3, 7, 11, 15, 19, 23];
struct ZeroSource {
    bytes: Vec<u8>,
    _source: BoundFullModelSource,
    _host: Vec<WindowWorkspace>,
}
struct GammaPayload {
    windows: [ParameterWindow; 2],
    _source: BoundFullModelSource,
    _host: WindowWorkspace,
}
struct Parameters {
    leases: [ResidentLease; 2],
    payload: Arc<GammaPayload>,
}
struct ControlPayload {
    bytes: Vec<Vec<u8>>,
    _host: Arc<WindowWorkspace>,
}
struct Controls {
    definitions: Arc<Definitions>,
    payload: Arc<ControlPayload>,
    leases: BTreeMap<Role, ResidentLease>,
}
pub(super) struct Preparation<'a> {
    pub(super) source: BoundFullModelSource,
    pub(super) geometries: &'a [ShortHfNativeGeometry],
    pub(super) rotary: Arc<ModelRotary>,
    pub(super) positions: Arc<ShortHfPositionsStorage>,
    pub(super) catalogue: &'a Catalogue,
    pub(super) service: Arc<ResidentService>,
    pub(super) private_layout: &'a [(Role, usize)],
}
pub(super) struct Storage {
    source: BoundFullModelSource,
    service: Arc<ResidentService>,
    parameters: BTreeMap<usize, Parameters>,
    caches: BTreeMap<usize, ResidentLease>,
    statuses: BTreeMap<usize, ResidentLease>,
    controls: BTreeMap<(usize, usize), Controls>,
    private: BTreeMap<(Role, usize), ResidentLease>,
    private_roles: BTreeMap<Key, Vec<(Role, usize)>>,
    rotary: Arc<ModelRotary>,
    positions: Arc<ShortHfPositionsStorage>,
    _metadata: Arc<WindowWorkspace>,
}
struct CurrentSite {
    site: ShortHfAttentionSite,
    state: Arc<BoundModelState>,
    roles: BTreeMap<Role, ResidentLease>,
    transport: Transport,
}
pub(super) struct Current {
    storage: Arc<Storage>,
    input: Arc<ShortHfCallInput>,
    state: Arc<ShortHfCallState>,
    values: Arc<ModelValues>,
    positions: Arc<ShortHfPositions>,
    sites: BTreeMap<usize, CurrentSite>,
    _metadata: WindowWorkspace,
}
fn private(role: Role) -> bool {
    matches!(
        role,
        Role::Lse | Role::PrivateO | Role::PartialO | Role::PartialLse
    )
}
impl Storage {
    pub(super) fn cohort_parameters(
        &self,
        inputs: &[Arc<ShortHfCallInput>],
        layer: usize,
    ) -> Result<[ResidentLease; 2]> {
        if inputs.len() != 4 {
            return Err(Error::State("B4 all-row QK source parameters".into()));
        }
        let parameters = self
            .parameters
            .get(&layer)
            .ok_or_else(|| Error::State("B4 actual original QK layer absent".into()))?;
        for input in inputs {
            if !Arc::ptr_eq(input.source().source(), self.source.source()) {
                return Err(Error::State("B4 QK source parameter owner".into()));
            }
            let node = input.scope().attention_site(layer)?.nodes[1];
            let zerogravity_core::accelerator::program::PlanOperation::TypedVllm08 {
                descriptor,
                ..
            } = &input.record().plan().nodes()[node].operation
            else {
                return Err(Error::State("B4 source QK descriptor absent".into()));
            };
            if descriptor.call() != input.call() || descriptor.parameters().len() != 2 {
                return Err(Error::State(
                    "B4 QK every real row parameter descriptor".into(),
                ));
            }
            for (index, role) in [
                QwenParameterRole::QueryNorm(layer),
                QwenParameterRole::KeyNorm(layer),
            ]
            .into_iter()
            .enumerate()
            {
                let parameter = &descriptor.parameters()[index];
                self.source.source().validate_parameter(role, parameter)?;
                if parameters.payload.windows[index]
                    .parameter()
                    .source_parameter()
                    != *parameter
                    || !parameters.leases[index].is_initialized()
                    || parameters.leases[index].bytes() != 512
                    || !Arc::ptr_eq(parameters.leases[index].service(), &self.service)
                {
                    return Err(Error::State(
                        "B4 exact immutable raw gamma source/lease".into(),
                    ));
                }
            }
        }
        Ok(parameters.leases.clone())
    }
    pub(super) fn prepare(
        p: Preparation<'_>,
        init: &mut ResidentInitialization<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancel)?;
        init.require_service(&p.service)?;
        let metadata = Arc::new(
            p.source
                .source()
                .source()
                .reserve_workspace(2 * 1024 * 1024)?,
        );
        init.retain_completion_owner(metadata.clone())?;
        init.retain_completion_owner(p.rotary.clone())?;
        init.retain_completion_owner(p.positions.clone())?;
        // One zero source, six actual cache uploads. Price all outstanding
        // upload copies/staging conservatively until this real init completes.
        let zero_host =
            super::storage::admit_host_cohort(&p.source, 6 * 3 * CACHE_BYTES + 64 * 1024)?;
        let zero = Arc::new(ZeroSource {
            bytes: vec![0; CACHE_BYTES],
            _source: p.source.clone(),
            _host: zero_host,
        });
        init.retain_completion_owner(zero.clone())?;
        let mut caches = BTreeMap::new();
        let mut statuses = BTreeMap::new();
        let mut parameters = BTreeMap::new();
        for layer in LAYERS {
            not_cancelled(cancel)?;
            caches.insert(
                layer,
                init.upload_bytes(&zero.bytes, BufferRole::RecurrentState)?,
            );
            statuses.insert(layer, init.upload_bytes(&[0; 4], BufferRole::Scratch)?);
            let host = p.source.source().source().reserve_workspace(64 * 1024)?;
            let mut windows = Vec::with_capacity(2);
            for role in [
                QwenParameterRole::QueryNorm(layer),
                QwenParameterRole::KeyNorm(layer),
            ] {
                let descriptor = p.source.source().parameter_descriptor(role)?;
                p.source.source().validate_parameter(role, &descriptor)?;
                let parameter = p.source.source().source().parameter(&descriptor.name)?;
                if descriptor.bytes != 512 || parameter.source_parameter() != descriptor {
                    return Err(Error::State(
                        "short-HF authentic QK gamma256 source parameter".into(),
                    ));
                }
                let window = match cancel {
                    Some(flag) => p.source.source().source().read_parameter_cancellable(
                        &parameter,
                        0..256,
                        flag,
                    )?,
                    None => p
                        .source
                        .source()
                        .source()
                        .read_parameter(&parameter, 0..256)?,
                };
                if !window.is_complete() || !super::super::finite_bf16(window.as_slice()) {
                    return Err(Error::Malformed(
                        "short-HF finite complete raw QK gamma".into(),
                    ));
                }
                windows.push(window);
            }
            let windows: [ParameterWindow; 2] = windows
                .try_into()
                .map_err(|_| Error::State("short-HF exact two source gamma windows".into()))?;
            let payload = Arc::new(GammaPayload {
                windows,
                _source: p.source.clone(),
                _host: host,
            });
            init.retain_completion_owner(payload.clone())?;
            let leases = [
                init.upload_bytes(payload.windows[0].as_slice(), BufferRole::ImmutableWeights)?,
                init.upload_bytes(payload.windows[1].as_slice(), BufferRole::ImmutableWeights)?,
            ];
            parameters.insert(layer, Parameters { leases, payload });
        }
        for g in p.geometries {
            p.rotary
                .require_call(p.source.source(), g.scope.call().as_vllm_call(), &p.service)?;
        }
        let mut controls = BTreeMap::new();
        let mut private_extents = BTreeSet::new();
        let mut private_roles = BTreeMap::new();
        for geometry in p
            .geometries
            .iter()
            .filter(|g| g.scope.call().span().base() == 0)
        {
            let workload = geometry.scope.call().span().workload();
            let prompt = workload.prompt_rows();
            for call in 0..8 {
                let base = if call == 0 { 0 } else { prompt + call - 1 };
                let rows = if call == 0 { prompt } else { 1 };
                let host = Arc::new(p.source.source().source().reserve_workspace(
                    64 * 1024 + 3 * 4096 + Definitions::metadata_bound(workload, call)?,
                )?);
                let definitions = Definitions::prepare(workload, call, host.clone())?;
                let roles = [
                    Role::Slots,
                    Role::KScale,
                    Role::VScale,
                    Role::CuQ,
                    Role::DummyCuK,
                    Role::SeqUsed,
                    Role::Table,
                ];
                let bytes = vec![
                    (base..base + rows)
                        .flat_map(|n| (4 * 544i64 + n as i64).to_le_bytes())
                        .collect(),
                    1f32.to_le_bytes().to_vec(),
                    1f32.to_le_bytes().to_vec(),
                    [0i32, rows as i32]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect(),
                    vec![0; 8],
                    i32::try_from(base + rows)
                        .map_err(|_| Error::Limit("short-HF finite sequence length".into()))?
                        .to_le_bytes()
                        .to_vec(),
                    [4i32, 0, 0, 0]
                        .into_iter()
                        .flat_map(i32::to_le_bytes)
                        .collect(),
                ];
                if bytes.iter().map(Vec::capacity).sum::<usize>() > 4096 {
                    return Err(Error::Limit("short-HF exact admitted control bytes".into()));
                }
                let payload = Arc::new(ControlPayload { bytes, _host: host });
                init.retain_completion_owner(payload.clone())?;
                let mut leases = BTreeMap::new();
                for (role, data) in roles.into_iter().zip(&payload.bytes) {
                    leases.insert(role, init.upload_bytes(data, BufferRole::ImmutableWeights)?);
                }
                controls.insert(
                    (prompt, call),
                    Controls {
                        definitions,
                        payload,
                        leases,
                    },
                );
            }
        }
        for site in p.catalogue.sites() {
            let mut roles = BTreeSet::new();
            for packet in &site.packets {
                for (role, binding) in packet.roles.iter().zip(&packet.manifest.bindings) {
                    if private(*role) {
                        let n = usize::try_from(binding.bytes).map_err(|_| {
                            Error::Limit("short-HF private native role overflow".into())
                        })?;
                        private_extents.insert((*role, n));
                        roles.insert((*role, n));
                    }
                }
            }
            private_roles.insert(site.key, roles.into_iter().collect());
        }
        if private_extents.iter().copied().collect::<Vec<_>>() != p.private_layout {
            return Err(Error::State(
                "short-HF private FA allocation differs from SAME original inventory".into(),
            ));
        }
        let mut private = BTreeMap::new();
        for (role, n) in private_extents {
            if n == 0 || n > 8 * 1024 * 1024 {
                return Err(Error::Limit(
                    "short-HF bounded private FA role extent".into(),
                ));
            }
            private.insert((role, n), init.allocate_empty(n, BufferRole::Scratch)?);
        }
        let result = Arc::new(Self {
            source: p.source,
            service: p.service,
            parameters,
            caches,
            statuses,
            controls,
            private,
            private_roles,
            rotary: p.rotary,
            positions: p.positions,
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(super) fn bind(
        self: &Arc<Self>,
        input: Arc<ShortHfCallInput>,
        state: Arc<ShortHfCallState>,
        values: Arc<ModelValues>,
        positions: Arc<ShortHfPositions>,
    ) -> Result<Arc<Current>> {
        input.scope().require_values(&values)?;
        if !Arc::ptr_eq(state.input(), &input)
            || !self.retained_source(input.source(), values.service())?
            || !positions.retained(&input, values.service())
            || positions.lease().id() != self.positions.lease().id()
        {
            return Err(Error::State(
                "short-HF attention actual source/current/cache/positions seal".into(),
            ));
        }
        self.rotary
            .require_call(self.source.source(), input.call(), values.service())?;
        let metadata = self
            .source
            .source()
            .source()
            // Fresh current handles/owners only; definitions stay in Controls.
            .reserve_workspace(512 * 1024)?;
        let control = self
            .controls
            .get(&(input.workload().prompt_rows(), input.call_ordinal()))
            .ok_or_else(|| Error::State("short-HF exact current attention control key".into()))?;
        let mut sites = BTreeMap::new();
        for layer in LAYERS {
            let site = input.scope().attention_site(layer)?.clone();
            let state = state
                .states()
                .get(layer)
                .ok_or_else(|| Error::State("short-HF current attention state absent".into()))?
                .clone();
            state.require_short_hf(input.scope())?;
            let params = self
                .parameters
                .get(&layer)
                .ok_or_else(|| Error::State("short-HF source QK gamma owner absent".into()))?;
            let node = &input.record().plan().nodes()[site.nodes[1]];
            let zerogravity_core::accelerator::program::PlanOperation::TypedVllm08 {
                descriptor,
                ..
            } = &node.operation
            else {
                return Err(Error::State("short-HF QK actual source descriptor".into()));
            };
            if descriptor.parameters().len() != 2 {
                return Err(Error::State(
                    "short-HF exact QK source parameter arity".into(),
                ));
            }
            for (i, role) in [
                QwenParameterRole::QueryNorm(layer),
                QwenParameterRole::KeyNorm(layer),
            ]
            .into_iter()
            .enumerate()
            {
                self.source
                    .source()
                    .validate_parameter(role, &descriptor.parameters()[i])?;
                if params.payload.windows[i].parameter().source_parameter()
                    != descriptor.parameters()[i]
                {
                    return Err(Error::State(
                        "short-HF current QK SAME source parameter member".into(),
                    ));
                }
            }
            let mut roles = control.leases.clone();
            for (role, id) in [
                (Role::Packed, site.operands[1]),
                (Role::Q, site.operands[2]),
                (Role::K, site.operands[3]),
                (Role::Gate, site.operands[4]),
                (Role::O, site.operands[5]),
            ] {
                roles.insert(role, values.lease(id)?.clone());
            }
            roles.insert(Role::QGamma, params.leases[0].clone());
            roles.insert(Role::KGamma, params.leases[1].clone());
            roles.insert(Role::Rotary, self.rotary.lease().clone());
            roles.insert(Role::Positions, positions.lease().clone());
            let cache = self.caches[&layer].clone();
            let status = self.statuses[&layer].clone();
            roles.insert(Role::Cache, cache.clone());
            let transport = Transport::bind(
                control.definitions.clone(),
                input.clone(),
                state.clone(),
                cache,
                status,
            )?;
            sites.insert(
                layer,
                CurrentSite {
                    site,
                    state,
                    roles,
                    transport,
                },
            );
        }
        Ok(Arc::new(Current {
            storage: self.clone(),
            input,
            state,
            values,
            positions,
            sites,
            _metadata: metadata,
        }))
    }
    pub(super) fn immutable_controls(&self) -> Vec<(ResidentLease, &[u8])> {
        let mut all = Vec::new();
        for params in self.parameters.values() {
            for (lease, window) in params.leases.iter().zip(&params.payload.windows) {
                all.push((lease.clone(), window.as_slice()));
            }
        }
        let roles = [
            Role::Slots,
            Role::KScale,
            Role::VScale,
            Role::CuQ,
            Role::DummyCuK,
            Role::SeqUsed,
            Role::Table,
        ];
        for control in self.controls.values() {
            for (role, bytes) in roles.iter().zip(&control.payload.bytes) {
                all.push((control.leases[role].clone(), bytes.as_slice()));
            }
        }
        all
    }
    pub(super) fn storage_accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.caches
            .values()
            .chain(self.statuses.values())
            .chain(self.private.values())
            .cloned()
            .map(|l| (l, DeviceAccess::Write))
            .chain(
                self.immutable_controls()
                    .into_iter()
                    .map(|(l, _)| (l, DeviceAccess::Read)),
            )
            .collect()
    }
    pub(super) fn retained_source(
        &self,
        source: &BoundFullModelSource,
        service: &Arc<ResidentService>,
    ) -> Result<bool> {
        Ok(Arc::ptr_eq(self.source.source(), source.source())
            && self.source.model_profile() == source.model_profile()
            && Arc::ptr_eq(&self.service, service)
            && self.caches.len() == 6
            && self.statuses.len() == 6
            && self.parameters.len() == 6
            && self.controls.len() == 40
            && self.rotary.lease().bytes() == 79 * 128
            && self.positions.lease().bytes() == 12312
            && [self.rotary.lease(), self.positions.lease()]
                .iter()
                .all(|l| {
                    Arc::ptr_eq(l.service(), service)
                        && l.source_identity() == source.source().source_identity()
                        && l.service_generation() == service.generation()
                })
            && self.storage_accesses().iter().all(|(l, _)| {
                Arc::ptr_eq(l.service(), service)
                    && l.source_identity() == source.source().source_identity()
                    && l.service_generation() == service.generation()
            }))
    }
}
impl Current {
    fn site(&self, node: usize) -> Option<&CurrentSite> {
        self.sites.values().find(|s| s.site.nodes.contains(&node))
    }
    pub(super) fn roles(&self, node: usize) -> Result<BTreeMap<Role, ResidentLease>> {
        let Some(site) = self.site(node) else {
            return Ok(BTreeMap::new());
        };
        let mut roles = site.roles.clone();
        let ids = site.site.operands;
        if node == site.site.nodes[4] {
            roles.insert(Role::X, self.values.lease(ids[4])?.clone());
            roles.insert(Role::Y, self.values.lease(ids[6])?.clone());
        }
        if node == site.site.nodes[5] {
            roles.insert(Role::X, self.values.lease(ids[5])?.clone());
            roles.insert(Role::Gate, self.values.lease(ids[6])?.clone());
            roles.insert(Role::Y, self.values.lease(ids[7])?.clone());
        }
        // Private native owners are selected by this exact compiled site's roles,
        // never by caller-provided bytes, raw addresses or public-value aliasing.
        let key = Key::current(self.input.scope(), node)?;
        for &(role, n) in
            self.storage.private_roles.get(&key).ok_or_else(|| {
                Error::State("short-HF exact source private role key absent".into())
            })?
        {
            roles.insert(
                role,
                self.storage
                    .private
                    .get(&(role, n))
                    .ok_or_else(|| {
                        Error::State("short-HF actual private role owner missing".into())
                    })?
                    .clone(),
            );
        }
        Ok(roles)
    }
    pub(super) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.sites
            .values()
            .flat_map(|s| s.transport.accesses())
            .collect()
    }
    pub(super) fn storage_accesses(
        &self,
        node: usize,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        Ok(self
            .site(node)
            .filter(|s| node == s.site.nodes[2])
            .map_or_else(Vec::new, |s| s.transport.accesses()))
    }
    pub(super) fn before(
        &self,
        node: usize,
        session: &ResidentSession,
        permit: &PrefixPhaseEnqueue<'_>,
        cancel: Option<&AtomicBool>,
        counts: &[AtomicUsize; 3],
    ) -> Result<()> {
        not_cancelled(cancel)?;
        self.input.require_session(session)?;
        if let Some(site) = self.site(node) {
            self.positions.require_enqueue(session, permit, cancel)?;
            if node == site.site.nodes[2] {
                let effect = self.input.record().plan().nodes()[node]
                    .state
                    .ok_or_else(|| Error::State("short-HF real KV append effect absent".into()))?;
                let n = site.transport.before(permit, effect)?;
                counts[2].fetch_add(n, Ordering::Release);
            }
        }
        Ok(())
    }
    pub(super) fn after(
        &self,
        node: usize,
        session: &ResidentSession,
        permit: &PrefixPhaseEnqueue<'_>,
        cancel: Option<&AtomicBool>,
        counts: &[AtomicUsize; 3],
    ) -> Result<()> {
        not_cancelled(cancel)?;
        self.input.require_session(session)?;
        if let Some(site) = self.site(node).filter(|s| node == s.site.nodes[2]) {
            let effect = self.input.record().plan().nodes()[node]
                .state
                .ok_or_else(|| Error::State("short-HF current KV effect absent".into()))?;
            let n = site.transport.after(permit, effect)?;
            counts[2].fetch_add(n, Ordering::Release);
        }
        Ok(())
    }
    pub(super) fn status_readbacks(&self) -> Result<Vec<ResidentReadback>> {
        LAYERS
            .iter()
            .map(|l| self.sites[l].transport.status().readback(0, 4))
            .collect()
    }
    pub(super) fn validate_statuses(&self, raw: &[cubecl::bytes::Bytes]) -> Result<()> {
        if raw.len() != 6
            || raw.iter().any(|b| b[..] != [0; 4])
            || self.sites.values().any(|s| !s.transport.output_submitted())
        {
            return Err(Error::State(
                "short-HF exact six actually submitted finite KV transports".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn retained(
        &self,
        input: &Arc<ShortHfCallInput>,
        state: &Arc<ShortHfCallState>,
        values: &Arc<ModelValues>,
    ) -> Result<bool> {
        Ok(Arc::ptr_eq(input, &self.input)
            && Arc::ptr_eq(state, &self.state)
            && Arc::ptr_eq(values, &self.values)
            && self.positions.retained(input, values.service())
            && self.sites.len() == 6
            && self.sites.iter().all(|(layer, s)| {
                s.state.layer().index() == *layer && s.transport.retained(input, &s.state)
            }))
    }
}

impl Storage {
    pub(super) fn prepare_finite_code(
        &self,
        queue: &cubecl::cuda::OwnedCudaQueue,
    ) -> Result<usize> {
        self.controls.values().try_fold(0usize, |n, control| {
            Ok(n + control.definitions.prepare_code(queue)?)
        })
    }
}
