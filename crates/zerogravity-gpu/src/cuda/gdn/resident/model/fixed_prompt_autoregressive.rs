//! Cold transport status, then fresh current-state handle binding; no fake root.
use super::*;

pub(crate) struct ModelGdnTransportStorage {
    cache: ResidentLease,
    status: ResidentLease,
    phase: GdnPhase,
    slot: GdnCacheSlot,
    definitions: [StateTransportDefinition; 4],
    source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
    _metadata: WindowWorkspace,
}
impl ModelGdnTransportStorage {
    pub(crate) fn host_metadata_bound() -> usize {
        StateTransportDefinition::metadata_bound() + std::mem::size_of::<Self>() + 32
    }
    fn definitions(
        source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
    ) -> Result<([StateTransportDefinition; 4], WindowWorkspace)> {
        let metadata = source.reserve_workspace(Self::host_metadata_bound())?;
        let definitions = [
            StateCopy::ConvIn,
            StateCopy::ConvOut,
            StateCopy::RecurrentIn,
            StateCopy::RecurrentOut,
        ]
        .into_iter()
        .map(StateTransportDefinition::prepare)
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| Error::State("four cold state transport definitions".into()))?;
        Ok((definitions, metadata))
    }
    pub(crate) fn prepare(
        prepared: &Arc<GdnPrepared>,
        init: &mut crate::resident_leases::ResidentInitialization<'_>,
        source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
    ) -> Result<Arc<Self>> {
        init.require_service(&prepared.service)?;
        if !matches!(
            prepared.phase,
            GdnPhase::FixedPrompt | GdnPhase::CachedDecode
        ) || prepared.slot != GdnCacheSlot::One
        {
            return Err(Error::Unsupported(
                "canonical cold transport finite source phase/plane".into(),
            ));
        }
        if source.source_identity() != prepared.service.source_identity() {
            return Err(Error::State(
                "cold GDN transport immutable source/service".into(),
            ));
        }
        let (definitions, metadata) = Self::definitions(source)?;
        let status = init.upload_bytes(&[0; 4], BufferRole::Scratch)?;
        let result = Arc::new(Self {
            cache: prepared.bound_lease(GdnBuffer::CacheArena)?.clone(),
            status,
            phase: prepared.phase,
            slot: prepared.slot,
            definitions,
            source: source.clone(),
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn prepare_text(
        prepared: &Arc<GdnPrepared>,
        init: &mut crate::resident_leases::ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        init.require_service(&prepared.service)?;
        let super::super::ComponentCode::Text(code) = &prepared.code else {
            return Err(Error::Unsupported(
                "text transport needs original source-bound GDN code".into(),
            ));
        };
        code.require(&prepared.service, prepared.phase)?;
        code.require_arithmetic(prepared.arithmetic())?;
        if prepared.slot != GdnCacheSlot::One {
            return Err(Error::State("text selected private cache plane".into()));
        }
        let source = code.scope().source();
        if source.source_identity() != prepared.service.source_identity() {
            return Err(Error::State(
                "cold text transport immutable source/service".into(),
            ));
        }
        let (definitions, metadata) = Self::definitions(source)?;
        let status = init.upload_bytes(&[0; 4], BufferRole::Scratch)?;
        let result = Arc::new(Self {
            cache: prepared.bound_lease(GdnBuffer::CacheArena)?.clone(),
            status,
            phase: prepared.phase,
            slot: prepared.slot,
            definitions,
            source: source.clone(),
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn bind(
        self: &Arc<Self>,
        binding: Arc<ModelGdnBinding>,
        domains: Arc<BoundModelState>,
        prepared: &Arc<GdnPrepared>,
    ) -> Result<Arc<ModelGdnAdapter>> {
        binding.require_prepared(prepared, &domains)?;
        let service = self.cache.service();
        service.ensure_usable()?;
        if self.source.source_identity() != binding.source().source_identity()
            || prepared.phase != self.phase
            || prepared.slot != self.slot
            || self.cache.id() != prepared.bound_lease(GdnBuffer::CacheArena)?.id()
            || !self.cache.is_initialized()
            || !self.status.is_initialized()
            || self.status.bytes() != 4
            || self.status.role() != BufferRole::Scratch
            || !Arc::ptr_eq(self.status.service(), service)
            || self.status.service_generation() != service.generation()
            || self.status.source_identity() != binding.source().source_identity()
        {
            return Err(Error::State(
                "canonical existing GDN transport resources differ".into(),
            ));
        }
        let leases = domains.leases();
        for (index, (lease, _)) in leases.iter().enumerate() {
            let bytes = if index < 2 {
                CONV_ELEMENTS * 4
            } else {
                H_ELEMENTS * 4
            };
            if lease.bytes() != bytes
                || !Arc::ptr_eq(lease.service(), service)
                || lease.source_identity() != binding.source().source_identity()
                || lease.service_generation() != service.generation()
                || [self.cache.id(), self.status.id()].contains(&lease.id())
            {
                return Err(Error::State(
                    "canonical fresh actual Conv/H transport state".into(),
                ));
            }
        }
        let metadata = binding.source().reserve_workspace(64 * 1024)?;
        let conv = plane_handle(
            &self.cache,
            self.slot.convolution_offset(),
            CONV_STATE_BYTES,
        )?;
        let recurrent = plane_handle(
            &self.cache,
            self.slot.recurrent_offset(),
            RECURRENT_STATE_BYTES,
        )?;
        let copies = [
            self.definitions[StateCopy::ConvIn.tag()].bind(
                service,
                StateCopy::ConvIn,
                leases[0].0.handle(),
                conv.clone(),
                &self.status,
            )?,
            self.definitions[StateCopy::ConvOut.tag()].bind(
                service,
                StateCopy::ConvOut,
                conv,
                leases[1].0.handle(),
                &self.status,
            )?,
            self.definitions[StateCopy::RecurrentIn.tag()].bind(
                service,
                StateCopy::RecurrentIn,
                leases[2].0.handle(),
                recurrent.clone(),
                &self.status,
            )?,
            self.definitions[StateCopy::RecurrentOut.tag()].bind(
                service,
                StateCopy::RecurrentOut,
                recurrent,
                leases[3].0.handle(),
                &self.status,
            )?,
        ];
        Ok(Arc::new(ModelGdnAdapter {
            copies,
            domains,
            binding,
            cache: self.cache.clone(),
            status: self.status.clone(),
            slot: self.slot,
            _transport: Some(self.clone()),
            _metadata: metadata,
        }))
    }
}

impl ModelGdnTransportStorage {
    pub(crate) fn leases(&self) -> [ResidentLease; 2] {
        [self.cache.clone(), self.status.clone()]
    }
}
