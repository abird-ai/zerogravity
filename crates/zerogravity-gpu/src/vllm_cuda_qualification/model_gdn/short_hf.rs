//! Same source parameters, GDN partitions and bit transports; no new evaluator.
use super::super::short_hf_model::ShortHfModelBinding;
use super::*;
use crate::vllm_cuda_gdn::{
    ModelGdnTransportStorage, ShortHfGdnCode, ShortHfGdnGeometry, ShortHfGdnWorkspace,
};

fn operators(
    scope: &ShortHfModelBinding,
    layer: usize,
) -> Result<(&Vllm08Operator, &Vllm08Operator)> {
    let site = scope.gdn_site(layer)?;
    let nodes = scope.record().plan().nodes();
    match (
        &nodes[site.nodes[0]].operation,
        &nodes[site.nodes[1]].operation,
    ) {
        (
            PlanOperation::TypedVllm08 { descriptor: c, .. },
            PlanOperation::TypedVllm08 { descriptor: r, .. },
        ) => Ok((c, r)),
        _ => Err(Error::State(
            "short-HF actual source GDN pair absent".into(),
        )),
    }
}
impl ModelGdnParameters {
    pub(crate) fn prepare_short_hf(
        scope: &Arc<ShortHfModelBinding>,
        layer: usize,
        service: &Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::super::projection_cubin::not_cancelled(cancelled)?;
        let (conv, recurrent) = operators(scope, layer)?;
        ShortHfGdnGeometry::for_call(scope.call())?.require_catalog()?;
        if scope.source().source().source_identity() != service.source_identity() {
            return Err(Error::State(
                "short-HF GDN actual source/service before parameter upload".into(),
            ));
        }
        Self::prepare_source(
            scope.source().source().clone(),
            service,
            conv,
            recurrent,
            init,
            Some(scope.call().model_profile()),
            cancelled,
        )
    }
    pub(crate) fn require_short_hf(&self, scope: &ShortHfModelBinding, layer: usize) -> Result<()> {
        let (conv, recurrent) = operators(scope, layer)?;
        if self.payload.short_hf_profile != Some(scope.call().model_profile())
            || self.payload.layer != scope.record().call().layer(layer)?
            || !Arc::ptr_eq(&self.payload.source, scope.source().source())
        {
            return Err(Error::State(
                "short-HF four learned parameter source/role/profile owner".into(),
            ));
        }
        self.require(conv, recurrent)
    }
    pub(crate) fn short_hf_leases(&self) -> &[ResidentLease; 4] {
        &self.leases
    }
}

pub(crate) struct ColdShortHfGdn {
    code: Arc<ShortHfGdnCode>,
    prepared: Arc<GdnPrepared>,
    transport: Arc<ModelGdnTransportStorage>,
    _metadata: WindowWorkspace,
}
impl ColdShortHfGdn {
    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<usize> {
        self.prepared.prepare_code(queue)
    }
    /// Root-free preparation over genuine source/value/cache owners. Caller
    /// retains all model storage before native modules activate.
    pub(crate) async fn prepare(
        scope: Arc<ShortHfModelBinding>,
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        workspace: Arc<ShortHfGdnWorkspace>,
        layer: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::super::projection_cubin::not_cancelled(cancelled)?;
        let cache = workspace.cache()?.clone();
        let code = ShortHfGdnCode::bind(scope, values, parameters, layer, workspace)?;
        let metadata = code
            .scope()
            .source()
            .source()
            .source()
            .reserve_workspace(code.geometry().metadata_bound()?)?;
        let prepared = GdnPrepared::prepare_short_hf_outputs(
            code.clone(),
            code.values().service().clone(),
            code.inputs(cache),
            code.outputs(),
            cancelled,
        )
        .await?;
        let mut init = code.values().service().begin_initialization(0)?;
        init.retain_completion_owner(code.clone())?;
        init.retain_completion_owner(prepared.clone())?;
        let transport = ModelGdnTransportStorage::prepare_short_hf(&prepared, &mut init)?;
        drop(init.complete_sync().await?);
        Ok(Arc::new(Self {
            code,
            prepared,
            transport,
            _metadata: metadata,
        }))
    }
    pub(crate) fn storage_accesses(
        &self,
        node: usize,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let site = self.code.scope().gdn_site(self.code.layer())?;
        let recurrent = if node == site.nodes[0] {
            false
        } else if node == site.nodes[1] {
            true
        } else {
            return Err(Error::State(
                "short-HF actual GDN lowerer node absent".into(),
            ));
        };
        let mut result = self.prepared.storage_accesses_partition(recurrent)?;
        result.extend(
            self.transport
                .leases()
                .into_iter()
                .map(|l| (l, DeviceAccess::Write)),
        );
        // Root-free native status/cache owners only. Actual prior/candidate
        // transport leases are added by the bound current-attempt actor.
        Ok(result)
    }
    /// Fresh current-attempt domains and actual record are checked before the
    /// existing transport binds. No prior Ready or cached token is retained.
    pub(crate) fn bind_call(
        &self,
        scope: Arc<ShortHfModelBinding>,
        values: Arc<ModelValues>,
        domains: Arc<BoundModelState>,
    ) -> Result<Arc<ModelGdnPrepared>> {
        scope.require_values(&values)?;
        let layer = self.code.layer();
        if domains.call() != scope.record().call()
            || domains.layer() != scope.record().call().layer(layer)?
            || ShortHfGdnGeometry::for_call(scope.call())? != self.code.geometry()
            || !Arc::ptr_eq(scope.source().source(), self.code.scope().source().source())
        {
            return Err(Error::State(
                "short-HF cold geometry versus actual root/source/layer".into(),
            ));
        }
        self.code.require_arithmetic(self.prepared.arithmetic())?;
        self.code.parameters().require_short_hf(&scope, layer)?;
        let site = scope.gdn_site(layer)?;
        let operands: [ResidentLease; 5] = site
            .operands
            .into_iter()
            .map(|id| values.lease(id).cloned())
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::State("short-HF five actual GDN producers".into()))?;
        for (old, new) in self.code.operands().iter().zip(&operands) {
            crate::resident_program::natural_binding::same_lease(old, new)?;
        }
        let nodes = scope.record().plan().nodes();
        for (i, recurrent) in [(0, false), (1, true)] {
            domains.require_effect(
                nodes[site.nodes[i]]
                    .state
                    .ok_or_else(|| Error::State("short-HF actual GDN state effect".into()))?,
                recurrent,
            )?;
        }
        let prepared =
            if Arc::ptr_eq(&scope, self.code.scope()) && Arc::ptr_eq(&values, self.code.values()) {
                self.prepared.clone()
            } else {
                let code = self.code.bind_call(scope.clone(), values.clone())?;
                self.prepared.bind_short_hf_outputs(code)?
            };
        let binding = Arc::new(ModelGdnBinding {
            arithmetic: prepared.arithmetic().clone(),
            values,
            parameters: self.code.parameters().clone(),
            nodes: site.nodes,
            operands,
        });
        binding.require_prepared(&prepared, &domains)?;
        let adapter = self
            .transport
            .bind(binding.clone(), domains.clone(), &prepared)?;
        let result = Arc::new(ModelGdnPrepared {
            binding,
            prepared,
            adapter,
            domains,
            _arena: RetainedArena {
                _legacy: None,
                _shared: None,
            },
            pending: Mutex::new(None),
            _metadata: scope
                .source()
                .source()
                .source()
                .reserve_workspace(64 * 1024)?,
            canonical47: None,
            short_hf: Some(scope.clone()),
        });
        result.require_short_hf_binding(&scope)?;
        Ok(result)
    }
}
impl ModelGdnPrepared {
    pub(crate) fn require_short_hf_binding(&self, scope: &Arc<ShortHfModelBinding>) -> Result<()> {
        if self
            .short_hf
            .as_ref()
            .is_none_or(|s| !Arc::ptr_eq(s, scope))
            || self.canonical47.is_some()
            || self.binding.call() != scope.record().call()
            || !Arc::ptr_eq(self.binding.source(), scope.source().source())
            || self.binding.nodes != scope.gdn_site(self.binding.layer().index())?.nodes
        {
            return Err(Error::State(
                "short-HF SAME whole model GDN source/call/producer seal".into(),
            ));
        }
        scope.require_values(&self.binding.values)?;
        self.binding
            .parameters
            .require_short_hf(scope, self.binding.layer().index())?;
        self.binding
            .require_prepared(&self.prepared, &self.domains)?;
        self.prepared.arithmetic().require_instance(
            ShortHfGdnGeometry::for_call(scope.call())?.phase,
            GdnCacheSlot::One,
            true,
        )?;
        self.prepared.arithmetic().family().require_same(
            &crate::vllm_cuda_gdn::GdnFamilyIdentity::short_hf(scope.call().model_profile()),
        )
    }
}

impl ModelGdnPrepared {
    pub(crate) fn short_hf_storage_accesses(
        &self,
        node: usize,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        if self.short_hf.is_none() {
            return Err(Error::Unsupported(
                "short-HF lowerer source binding absent".into(),
            ));
        }
        let recurrent = if node == self.binding.nodes[0] {
            false
        } else if node == self.binding.nodes[1] {
            true
        } else {
            return Err(Error::State("short-HF actual GDN node absent".into()));
        };
        let mut result = self.prepared.storage_accesses_partition(recurrent)?;
        result.extend(self.adapter.storage_accesses_partition(recurrent));
        // Keep the declared BA dependency in both nodes. Prefill packs A/B
        // during Conv; cached recurrence reads BA directly. This adds no command.
        result.push((self.binding.operands[1].clone(), DeviceAccess::Read));
        Ok(result)
    }
}

impl ColdShortHfGdn {
    pub(crate) fn retained_storage_accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let site = self.code.scope().gdn_site(self.code.layer())?;
        let mut all = self.storage_accesses(site.nodes[0])?;
        all.extend(self.storage_accesses(site.nodes[1])?);
        Ok(all)
    }
}

impl ColdShortHfGdn {
    pub(crate) fn allocation_layout(scope: &ShortHfModelBinding) -> Result<Vec<usize>> {
        let geometry = ShortHfGdnGeometry::for_call(scope.call())?;
        geometry.require_catalog()?;
        let mut result = Vec::with_capacity(36);
        for layer in (0..24).filter(|l| l % 4 != 3) {
            operators(scope, layer)?;
            // SAME remaining per-layer private role and actual transport status.
            result.push(crate::vllm_cuda_gdn::GdnBuffer::Rstd.bytes(geometry.phase));
            result.push(4);
        }
        Ok(result)
    }
    pub(crate) fn host_bytes(scope: &ShortHfModelBinding) -> Result<usize> {
        let geometry = ShortHfGdnGeometry::for_call(scope.call())?;
        geometry
            .metadata_bound()?
            .checked_add(64 * 1024)
            .and_then(|n| n.checked_add(ModelGdnTransportStorage::host_metadata_bound()))
            .and_then(|n| n.checked_mul(18))
            .ok_or_else(|| Error::Limit("short-HF18 actual GDN actor source owners".into()))
    }
}
impl ModelGdnParameters {
    pub(crate) fn allocation_layout(scope: &ShortHfModelBinding) -> Result<Vec<usize>> {
        let mut result = Vec::with_capacity(72);
        for layer in (0..24).filter(|l| l % 4 != 3) {
            let (c, r) = operators(scope, layer)?;
            let descriptors = [
                c.parameters()[0].clone(),
                r.parameters()[0].clone(),
                r.parameters()[1].clone(),
                r.parameters()[2].clone(),
            ];
            result.extend(super::parameter_allocation_layout(&descriptors)?);
        }
        Ok(result)
    }
}
