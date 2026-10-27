//! Same source parameters, GDN partitions and bit transports; no new evaluator.
use super::super::text_model::TextModelBinding;
use super::*;
use crate::cuda::gdn::{ModelGdnTransportStorage, TextGdnCode, TextGdnGeometry, TextGdnWorkspace};

fn operators(scope: &TextModelBinding, layer: usize) -> Result<(SourceOperator, SourceOperator)> {
    let site = scope.gdn_site(layer)?;
    match (
        scope.record().operator(site.nodes[0])?,
        scope.record().operator(site.nodes[1])?,
    ) {
        (Some(conv), Some(recurrent)) => Ok((conv.clone(), recurrent.clone())),
        _ => Err(Error::State("text actual source GDN pair absent".into())),
    }
}
impl ModelGdnParameters {
    pub(crate) fn prepare_text(
        scope: &Arc<TextModelBinding>,
        layer: usize,
        service: &Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::super::projection_cubin::not_cancelled(cancelled)?;
        let (conv, recurrent) = operators(scope, layer)?;
        TextGdnGeometry::for_call(scope.call(), scope.gdn_implementation())?
            .require_implementation()?;
        if scope.source().source_identity() != service.source_identity() {
            return Err(Error::State(
                "text GDN actual source/service before parameter upload".into(),
            ));
        }
        Self::prepare_source(
            scope.source().clone(),
            service,
            &conv,
            &recurrent,
            init,
            Some(crate::cuda::gdn::GdnFamilyIdentity::text(
                scope.gdn_implementation().config(),
            )),
            cancelled,
        )
    }
    pub(crate) fn require_text(&self, scope: &TextModelBinding, layer: usize) -> Result<()> {
        let (conv, recurrent) = operators(scope, layer)?;
        if self.payload.text_implementation
            != Some(crate::cuda::gdn::GdnFamilyIdentity::text(
                scope.gdn_implementation().config(),
            ))
            || self.payload.site != layer
            || !self.payload.source.same_issuer(scope.source())
        {
            return Err(Error::State(
                "text four learned parameter source/role/profile owner".into(),
            ));
        }
        self.require(&conv, &recurrent)
    }
    pub(crate) fn text_leases(&self) -> &[ResidentLease; 4] {
        &self.leases
    }
}

pub(crate) struct ColdTextGdn {
    code: Arc<TextGdnCode>,
    prepared: Arc<GdnPrepared>,
    transport: Arc<ModelGdnTransportStorage>,
    _metadata: WindowWorkspace,
}
impl ColdTextGdn {
    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<usize> {
        self.prepared.prepare_code(queue)
    }
    /// Root-free preparation over genuine source/value/cache owners. Caller
    /// retains all model storage before native modules activate.
    pub(crate) async fn prepare(
        scope: Arc<TextModelBinding>,
        values: Arc<ModelValues>,
        parameters: Arc<ModelGdnParameters>,
        workspace: Arc<TextGdnWorkspace>,
        layer: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::super::projection_cubin::not_cancelled(cancelled)?;
        let cache = workspace.cache()?.clone();
        let code = TextGdnCode::bind(scope, values, parameters, layer, workspace)?;
        let metadata = code
            .scope()
            .source()
            .reserve_workspace(code.geometry().metadata_bound()?)?;
        let prepared = GdnPrepared::prepare_text_outputs(
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
        let transport = ModelGdnTransportStorage::prepare_text(&prepared, &mut init)?;
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
            return Err(Error::State("text actual GDN lowerer node absent".into()));
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
        scope: Arc<TextModelBinding>,
        values: Arc<ModelValues>,
        domains: Arc<BoundModelState>,
    ) -> Result<Arc<ModelGdnPrepared>> {
        scope.require_values(&values)?;
        let layer = self.code.layer();
        if domains.call() != scope.record().call()
            || domains.layer() != layer
            || TextGdnGeometry::for_call(scope.call(), scope.gdn_implementation())?
                != self.code.geometry()
            || !scope.source().same_issuer(self.code.scope().source())
        {
            return Err(Error::State(
                "text cold geometry versus actual root/source/layer".into(),
            ));
        }
        self.code.require_arithmetic(self.prepared.arithmetic())?;
        self.code.parameters().require_text(&scope, layer)?;
        let site = scope.gdn_site(layer)?;
        let operands: [ResidentLease; 5] = site
            .operands
            .into_iter()
            .map(|id| values.lease(id).cloned())
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::State("text five actual GDN producers".into()))?;
        for (old, new) in self.code.operands().iter().zip(&operands) {
            crate::resident_program::natural_binding::same_lease(old, new)?;
        }
        let nodes = scope.record().plan().nodes();
        for (i, recurrent) in [(0, false), (1, true)] {
            domains.require_effect(
                nodes[site.nodes[i]]
                    .state
                    .ok_or_else(|| Error::State("text actual GDN state effect".into()))?,
                recurrent,
            )?;
        }
        let prepared =
            if Arc::ptr_eq(&scope, self.code.scope()) && Arc::ptr_eq(&values, self.code.values()) {
                self.prepared.clone()
            } else {
                let code = self.code.bind_call(scope.clone(), values.clone())?;
                self.prepared.bind_text_outputs(code)?
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
            _metadata: scope.source().reserve_workspace(64 * 1024)?,
            fixed_prompt: None,
            text: Some(scope.clone()),
        });
        result.require_text_binding(&scope)?;
        Ok(result)
    }
}
impl ModelGdnPrepared {
    pub(crate) fn require_text_binding(&self, scope: &Arc<TextModelBinding>) -> Result<()> {
        if self.text.as_ref().is_none_or(|s| !Arc::ptr_eq(s, scope))
            || self.fixed_prompt.is_some()
            || self.binding.call() != scope.record().call()
            || !self.binding.source().same_issuer(scope.source())
            || self.binding.nodes != scope.gdn_site(self.binding.layer())?.nodes
        {
            return Err(Error::State(
                "text SAME whole model GDN source/call/producer seal".into(),
            ));
        }
        scope.require_values(&self.binding.values)?;
        self.binding
            .parameters
            .require_text(scope, self.binding.layer())?;
        self.binding
            .require_prepared(&self.prepared, &self.domains)?;
        self.prepared.arithmetic().require_instance(
            TextGdnGeometry::for_call(scope.call(), scope.gdn_implementation())?.phase,
            GdnCacheSlot::One,
            true,
        )?;
        self.prepared.arithmetic().family().require_same(
            &crate::cuda::gdn::GdnFamilyIdentity::text(scope.gdn_implementation().config()),
        )
    }
}

impl ModelGdnPrepared {
    pub(crate) fn text_storage_accesses(
        &self,
        node: usize,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        if self.text.is_none() {
            return Err(Error::Unsupported(
                "text lowerer source binding absent".into(),
            ));
        }
        let recurrent = if node == self.binding.nodes[0] {
            false
        } else if node == self.binding.nodes[1] {
            true
        } else {
            return Err(Error::State("text actual GDN node absent".into()));
        };
        let mut result = self.prepared.storage_accesses_partition(recurrent)?;
        result.extend(self.adapter.storage_accesses_partition(recurrent));
        // Keep the declared BA dependency in both nodes. Prefill packs A/B
        // during Conv; cached recurrence reads BA directly. This adds no command.
        result.push((self.binding.operands[1].clone(), DeviceAccess::Read));
        Ok(result)
    }
}

impl ColdTextGdn {
    pub(crate) fn retained_storage_accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let site = self.code.scope().gdn_site(self.code.layer())?;
        let mut all = self.storage_accesses(site.nodes[0])?;
        all.extend(self.storage_accesses(site.nodes[1])?);
        Ok(all)
    }
}

impl ColdTextGdn {
    pub(crate) fn allocation_layout(scope: &TextModelBinding) -> Result<Vec<usize>> {
        let geometry = TextGdnGeometry::for_call(scope.call(), scope.gdn_implementation())?;
        geometry.require_implementation()?;
        let count = scope
            .gdn_sites()
            .len()
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("GDN allocation role count".into()))?;
        let mut result = Vec::with_capacity(count);
        for (layer, _) in scope.gdn_sites() {
            operators(scope, layer)?;
            // SAME remaining per-layer private role and actual transport status.
            result.push(crate::cuda::gdn::GdnBuffer::Rstd.bytes(geometry.phase));
            result.push(4);
        }
        Ok(result)
    }
    pub(crate) fn host_bytes(scope: &TextModelBinding) -> Result<usize> {
        let geometry = TextGdnGeometry::for_call(scope.call(), scope.gdn_implementation())?;
        geometry
            .metadata_bound()?
            .checked_add(64 * 1024)
            .and_then(|n| n.checked_add(ModelGdnTransportStorage::host_metadata_bound()))
            .and_then(|n| n.checked_mul(scope.gdn_sites().len()))
            .ok_or_else(|| Error::Limit("text actual GDN actor source owners".into()))
    }
}
impl ModelGdnParameters {
    pub(crate) fn allocation_layout(scope: &TextModelBinding) -> Result<Vec<usize>> {
        let count = scope
            .gdn_sites()
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("GDN parameter role count".into()))?;
        let mut result = Vec::with_capacity(count);
        for (layer, _) in scope.gdn_sites() {
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
