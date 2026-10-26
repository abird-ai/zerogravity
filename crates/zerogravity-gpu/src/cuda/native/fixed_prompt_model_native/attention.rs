//! Genuine node33/34 operands; original body owns the only traversal/flight.
use super::*;
use crate::cuda::native::attention_cubin::fixed_prompt_packets::{self, Role};
use crate::cuda::native::{
    attention_cubin::ModelAttentionParameters, model_cache::ModelCache,
    model_kv_transport::ModelKvTransport, model_rotary::ModelRotary,
};
use crate::resident_inference::native_fragment::{
    PrefixPhaseEnqueue, model_state::BoundModelState,
};
use crate::resident_leases::ResidentReadback;
use crate::resident_program::text_positions::TextPositionsPrepared;
use cubecl::bytes::Bytes;

struct Controls {
    bytes: Vec<Vec<u8>>,
    _metadata: WindowWorkspace,
}
pub(crate) struct Inputs {
    pub(crate) cut: Arc<FixedPromptPrefixCut>,
    pub(crate) values: Arc<ModelValues>,
    pub(crate) parameters: Arc<ModelAttentionParameters>,
    pub(crate) positions: Arc<TextPositionsPrepared>,
    pub(crate) rotary: Arc<ModelRotary>,
    pub(crate) domains: Arc<BoundModelState>,
    pub(crate) cache: Arc<ModelCache>,
}
pub(crate) struct Binding {
    site: Option<super::super::fixed_prompt_model_prefix::AttentionSite>,
    cut: Arc<FixedPromptPrefixCut>,
    values: Arc<ModelValues>,
    parameters: Arc<ModelAttentionParameters>,
    positions: Arc<TextPositionsPrepared>,
    rotary: Arc<ModelRotary>,
    domains: Arc<BoundModelState>,
    _cache: Arc<ModelCache>,
    transport: Arc<ModelKvTransport>,
    roles: BTreeMap<Role, ResidentLease>,
    _controls: Arc<Controls>,
}
impl Binding {
    /// All real values/params/positions/rotary/cache owners already initialized.
    /// Actual control uploads and outgoing transport status precede module load.
    pub(crate) fn prepare(
        input: Inputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        fixed_prompt_packets::require_memory_qualification()?;
        not_cancelled(cancelled)?;
        let Inputs {
            cut,
            values,
            parameters,
            positions,
            rotary,
            domains,
            cache,
        } = input;
        cut.scope().require_values(&values)?;
        if !cut.is_attention_cut() || !Arc::ptr_eq(values.record(), cut.scope().record()) {
            return Err(Error::State(
                "fixed_prompt exact computed45-value QK/KV source owner".into(),
            ));
        }
        init.require_service(values.service())?;
        domains.require_fixed_prompt_attention_kv(cut.scope())?;
        cache.require_binding(&domains)?;
        positions.require_invocation(values.record().call())?;
        rotary.require_call(
            cut.scope().source(),
            values.record().call(),
            values.service(),
        )?;
        if !positions.prepared_lease().is_initialized() {
            return Err(Error::State(
                "fixed_prompt positions actual source upload incomplete".into(),
            ));
        }
        let [qgamma, kgamma] = parameters.require_fixed_prompt_qk(&cut, &values)?;
        let mut roles = BTreeMap::from([
            (Role::Packed, values.lease(ValueId(41))?.clone()),
            (Role::Q, values.lease(ValueId(42))?.clone()),
            (Role::K, values.lease(ValueId(43))?.clone()),
            (Role::Gate, values.lease(ValueId(44))?.clone()),
            (Role::QGamma, qgamma),
            (Role::KGamma, kgamma),
            (Role::Rotary, rotary.lease().clone()),
            (Role::Positions, positions.prepared_lease().clone()),
            (Role::Cache, cache.lease().clone()),
        ]);
        init.retain_completion_owner(cut.clone())?;
        init.retain_completion_owner(values.clone())?;
        init.retain_completion_owner(parameters.clone())?;
        init.retain_completion_owner(positions.clone())?;
        init.retain_completion_owner(rotary.clone())?;
        init.retain_completion_owner(domains.clone())?;
        init.retain_completion_owner(cache.clone())?;
        let metadata = cut
            .scope()
            .source()
            .reserve_workspace(if cut.includes_fa() {
                128 * 1024
            } else {
                64 * 1024
            })?;
        let slots = (0..47i64)
            .flat_map(|row| (4 * 544 + row).to_le_bytes())
            .collect::<Vec<_>>();
        let mut control_roles = vec![Role::Slots, Role::KScale, Role::VScale];
        let mut bytes = vec![
            slots,
            1f32.to_le_bytes().to_vec(),
            1f32.to_le_bytes().to_vec(),
        ];
        if cut.includes_fa() {
            cut.require_fa_read()?;
            roles.insert(Role::O, values.lease(ValueId(45))?.clone());
            roles.insert(Role::Lse, init.allocate_empty(1504, BufferRole::Scratch)?);
            control_roles.extend([Role::CuQ, Role::DummyCuK, Role::SeqUsed, Role::Table]);
            bytes.extend([
                [0i32, 47].into_iter().flat_map(i32::to_le_bytes).collect(),
                [0i32, 0].into_iter().flat_map(i32::to_le_bytes).collect(),
                47i32.to_le_bytes().to_vec(),
                [4i32, 0, 0, 0]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect(),
            ]);
        }
        if bytes.iter().map(Vec::capacity).sum::<usize>() > 1024
            || bytes.len() != control_roles.len()
        {
            return Err(Error::Limit(
                "fixed_prompt exact admitted control payload ceiling".into(),
            ));
        }
        let controls = Arc::new(Controls {
            bytes,
            _metadata: metadata,
        });
        init.retain_completion_owner(controls.clone())?;
        for (role, bytes) in control_roles.into_iter().zip(&controls.bytes) {
            roles.insert(role, init.upload_bytes(bytes, BufferRole::Activation)?);
        }
        let packets = fixed_prompt_packets::packets_with_fa(cut.includes_fa())?;
        let mut unique = BTreeSet::new();
        for (role, lease) in &roles {
            if !Arc::ptr_eq(lease.service(), values.service())
                || lease.service_generation() != values.service().generation()
                || lease.source_identity() != values.record().call().source()
                || !unique.insert(lease.id())
            {
                return Err(Error::State(
                    "fixed_prompt QK/KV same source/service/distinct original leases".into(),
                ));
            }
            let expected = packets
                .iter()
                .flat_map(|p| p.roles.iter().zip(&p.manifest.bindings))
                .filter(|(r, _)| *r == role)
                .map(|(_, b)| b.bytes)
                .collect::<BTreeSet<_>>();
            if expected != BTreeSet::from([lease.bytes() as u64]) {
                return Err(Error::State(
                    "fixed_prompt actual QK/KV role extents differ packet".into(),
                ));
            }
        }
        let transport = ModelKvTransport::prepare_fixed_prompt(
            &cut,
            domains.clone(),
            cache.clone(),
            init,
            cancelled,
        )?;
        let bound = Arc::new(Self {
            site: None,
            cut,
            values,
            parameters,
            positions,
            rotary,
            domains,
            _cache: cache,
            transport,
            roles,
            _controls: controls,
        });
        init.retain_completion_owner(bound.clone())?;
        Ok(bound)
    }
    pub(crate) fn require_values(
        &self,
        cut: &Arc<FixedPromptPrefixCut>,
        values: &Arc<ModelValues>,
    ) -> Result<()> {
        if !Arc::ptr_eq(cut, &self.cut) || !Arc::ptr_eq(values, &self.values) {
            return Err(Error::State(
                "fixed_prompt QK/KV body changed original cut/value seal".into(),
            ));
        }
        if self.site.is_some() {
            self.domains.require_fixed_prompt_whole(cut)
        } else {
            self.domains.require_fixed_prompt_attention_kv(cut.scope())
        }
    }
    pub(crate) fn domains(&self) -> &Arc<BoundModelState> {
        &self.domains
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut all = BTreeMap::new();
        for (role, lease) in &self.roles {
            insert(
                &mut all,
                lease.clone(),
                if matches!(
                    role,
                    Role::Q | Role::K | Role::Gate | Role::Cache | Role::O | Role::Lse
                ) {
                    DeviceAccess::Write
                } else {
                    DeviceAccess::Read
                },
            );
        }
        for (lease, access) in self.transport.accesses() {
            insert(&mut all, lease, access);
        }
        all.into_values().collect()
    }
    pub(crate) fn extra_readbacks(&self) -> Result<Vec<ResidentReadback>> {
        Ok(vec![
            self.positions.readback()?,
            self.transport.status_readback()?,
        ])
    }
    pub(crate) fn validate_extra(&self, raw: &[Bytes]) -> Result<()> {
        if raw.len() != 2 {
            return Err(Error::State(
                "fixed_prompt actual positions/KV status cohort".into(),
            ));
        }
        self.positions.validate_readback(&raw[..1])?;
        ModelKvTransport::validate_status(&raw[1][..])
    }
    pub(crate) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        let mut all = self.parameters.immutable_weights();
        all.push((self.rotary.lease().clone(), self.rotary.source_bytes()));
        if self.cut.includes_fa() {
            for (role, bytes) in [Role::CuQ, Role::DummyCuK, Role::SeqUsed, Role::Table]
                .into_iter()
                .zip(&self._controls.bytes[3..])
            {
                all.push((self.roles[&role].clone(), bytes.as_slice()));
            }
        }
        all
    }
    pub(super) fn require_enqueue(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<()> {
        not_cancelled(cancelled)?;
        self.domains
            .require_enqueue(self.values.service(), permit)?;
        self.positions
            .require_enqueue(permit.session(), Some(permit), cancelled)?;
        self.rotary.require_call(
            self.cut.scope().source(),
            self.values.record().call(),
            self.values.service(),
        )
    }
    pub(crate) fn fa_output_unpromoted(&self) -> Result<bool> {
        Ok(self.cut.includes_fa()
            && !self
                .values
                .lease(self.site.as_ref().map_or(ValueId(45), |s| s.values()[4]))?
                .is_initialized()
            && self
                .roles
                .get(&Role::Lse)
                .is_some_and(|lease| !lease.is_initialized()))
    }
    pub(crate) fn outputs_unpromoted(&self) -> Result<(bool, bool)> {
        let mut qk = true;
        let ids = self
            .site
            .as_ref()
            .map_or([ValueId(42), ValueId(43), ValueId(44)], |s| {
                [s.values()[1], s.values()[2], s.values()[3]]
            });
        for id in ids {
            qk &= !self.values.lease(id)?.is_initialized();
        }
        let leases = self.domains.leases();
        Ok((
            qk,
            !leases[1].0.is_initialized() && !leases[3].0.is_initialized(),
        ))
    }
}
struct Kernel {
    roles: Vec<Role>,
    kernel: Arc<OwnedCudaKernel>,
}
pub(super) struct Programs {
    binding: Arc<Binding>,
    kernels: Vec<Kernel>,
    qk: AtomicUsize,
    kv: AtomicUsize,
    transports: AtomicUsize,
    fa: AtomicUsize,
}
impl Programs {
    pub(super) fn prepare(
        binding: Arc<Binding>,
        images: &Arc<PreparedImages>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        init.retain_completion_owner(binding.clone())?;
        let mut kernels = Vec::with_capacity(2 + usize::from(binding.cut.includes_fa()));
        for packet in fixed_prompt_packets::packets_with_fa(binding.cut.includes_fa())? {
            not_cancelled(cancelled)?;
            let image = images.image(packet.image)?;
            // SAFETY: same original set/init, exact source roles already bound.
            let kernel =
                unsafe { images.set().prepare(image, packet.manifest) }.map_err(native_error)?;
            init.retain_completion_owner(kernel.clone())?;
            kernels.push(Kernel {
                roles: packet.roles,
                kernel,
            });
        }
        if kernels.len() != 2 + usize::from(binding.cut.includes_fa()) {
            return Err(Error::State(
                "fixed_prompt exact two or three attention records".into(),
            ));
        }
        let prepared = Arc::new(Self {
            binding,
            kernels,
            qk: AtomicUsize::new(0),
            kv: AtomicUsize::new(0),
            transports: AtomicUsize::new(0),
            fa: AtomicUsize::new(0),
        });
        init.retain_completion_owner(prepared.clone())?;
        Ok(prepared)
    }
    pub(super) fn enqueue(
        &self,
        index: usize,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
        fault: usize,
    ) -> Result<usize> {
        self.binding.require_enqueue(permit, cancelled)?;
        if index == 35 {
            return self.enqueue_fa(cancelled, fault);
        }
        let which = match index {
            33 => 0,
            34 => 1,
            _ => {
                return Err(Error::State(
                    "fixed_prompt exact node33/34 native cursor".into(),
                ));
            }
        };
        if !self.binding.outputs_unpromoted()?.0 || !self.binding.outputs_unpromoted()?.1 {
            return Err(Error::State(
                "fixed_prompt QK/KV outputs promoted before actual completion".into(),
            ));
        }
        if which == 1 && self.qk.load(Ordering::Acquire) != 1 {
            return Err(Error::State(
                "fixed_prompt KV requires actual preceding QK enqueue".into(),
            ));
        }
        if (which == 0 && self.qk.load(Ordering::Acquire) != 0)
            || (which == 1 && self.kv.load(Ordering::Acquire) != 0)
        {
            return Err(Error::State(
                "fixed_prompt immutable QK/KV binding is one use".into(),
            ));
        }
        let late = AtomicBool::new(which == 1 && fault == 6);
        let cancel = if late.load(Ordering::Acquire) {
            Some(&late)
        } else {
            cancelled
        };
        let mut transports = 0;
        if which == 1 {
            transports += self.binding.transport.before_append(
                permit,
                self.binding.domains.effect(false)?,
                cancel,
            )?;
        }
        not_cancelled(cancel)?;
        let kernel = &self.kernels[which];
        let handles = kernel
            .roles
            .iter()
            .map(|role| {
                self.binding
                    .roles
                    .get(role)
                    .map(ResidentLease::handle)
                    .ok_or_else(|| Error::State("fixed_prompt actual native role absent".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: original armed body flight retains same source/value/state/control graph.
        unsafe { kernel.kernel.enqueue(handles) }.map_err(native_error)?;
        if which == 0 {
            self.qk.fetch_add(1, Ordering::AcqRel);
        } else {
            self.kv.fetch_add(1, Ordering::AcqRel);
        }
        if (which == 0 && fault == 5) || (which == 1 && fault == 7) {
            return Err(Error::State(
                if which == 0 {
                    "deliberate fixed_prompt unknown after actual QK"
                } else {
                    "deliberate fixed_prompt unknown after actual KV append"
                }
                .into(),
            ));
        }
        if which == 1 {
            if fault == 8 {
                late.store(true, Ordering::Release);
            }
            transports += self.binding.transport.after_append(
                permit,
                self.binding.domains.effect(false)?,
                if late.load(Ordering::Acquire) {
                    Some(&late)
                } else {
                    cancelled
                },
            )?;
        }
        self.transports.fetch_add(transports, Ordering::AcqRel);
        Ok(1 + transports)
    }
    fn enqueue_fa(&self, cancelled: Option<&AtomicBool>, fault: usize) -> Result<usize> {
        if !self.binding.cut.includes_fa()
            || !matches!(fault, 0 | 9 | 10)
            || self.counts() != (1, 1, 1)
            || self.fa.load(Ordering::Acquire) != 0
            || !self.binding.fa_output_unpromoted()?
        {
            return Err(Error::State(
                "fixed_prompt FA requires same appendedKv3 and unpublished45/LSE".into(),
            ));
        }
        self.binding.cut.require_fa_read()?;
        let late = AtomicBool::new(fault == 10);
        not_cancelled(if fault == 10 { Some(&late) } else { cancelled })?;
        let kernel = self
            .kernels
            .get(2)
            .ok_or_else(|| Error::State("fixed_prompt actual FA record absent".into()))?;
        let handles = kernel
            .roles
            .iter()
            .map(|role| {
                self.binding
                    .roles
                    .get(role)
                    .map(ResidentLease::handle)
                    .ok_or_else(|| Error::State("fixed_prompt FA original role absent".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: SAME original armed body flight retains cache/currentattempt/controls/images.
        unsafe { kernel.kernel.enqueue(handles) }.map_err(native_error)?;
        self.fa.fetch_add(1, Ordering::AcqRel);
        if !self.binding.fa_output_unpromoted()? {
            return Err(Error::State(
                "fixed_prompt FA prematurely initialized output/LSE".into(),
            ));
        }
        if fault == 9 {
            return Err(Error::State(
                "deliberate fixed_prompt unknown after actual FA".into(),
            ));
        }
        Ok(1)
    }
    pub(super) fn fa_enqueues(&self) -> usize {
        self.fa.load(Ordering::Acquire)
    }
    pub(super) fn record_count(&self) -> usize {
        self.kernels.len()
    }
    pub(super) fn counts(&self) -> (usize, usize, usize) {
        (
            self.qk.load(Ordering::Acquire),
            self.kv.load(Ordering::Acquire),
            self.transports.load(Ordering::Acquire),
        )
    }
    pub(super) fn binding(&self) -> &Arc<Binding> {
        &self.binding
    }
    pub(super) fn kernels(&self) -> Vec<Arc<OwnedCudaKernel>> {
        self.kernels.iter().map(|k| k.kernel.clone()).collect()
    }
    pub(super) unsafe fn retire(&self) -> Result<()> {
        for kernel in &self.kernels {
            unsafe { kernel.kernel.retire_after_physical_completion() }.map_err(native_error)?;
        }
        Ok(())
    }
}

mod whole;
