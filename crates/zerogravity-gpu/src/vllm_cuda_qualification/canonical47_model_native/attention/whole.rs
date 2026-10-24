//! Same QK/KV/FA physical binding algorithm, exact source layer sites.
use super::*;
impl Binding {
    pub(crate) fn prepare_whole(
        input: Inputs,
        layer: usize,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        canonical47_packets::require_memory_qualification()?;
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
        cut.require_whole()?;
        let site = cut.attention_site(layer)?.clone();
        let ids = site.values();
        cut.scope().require_values(&values)?;
        if !cut.is_whole() || !Arc::ptr_eq(values.record(), cut.scope().record()) {
            return Err(Error::State(
                "canonical47 exact computed45-value QK/KV source owner".into(),
            ));
        }
        init.require_service(values.service())?;
        domains.require_canonical47_whole(&cut)?;
        if domains.layer() != site.layer() {
            return Err(Error::State(
                "whole QK/KV actual source layer/state differs".into(),
            ));
        }
        cache.require_binding(&domains)?;
        positions.require_invocation(values.record().call().request())?;
        rotary.require_call(
            cut.scope().source().source(),
            values.record().call(),
            values.service(),
        )?;
        if !positions.prepared_lease().is_initialized() {
            return Err(Error::State(
                "canonical47 positions actual source upload incomplete".into(),
            ));
        }
        let [qgamma, kgamma] = parameters.require_whole_qk(&cut, &values, layer)?;
        let mut roles = BTreeMap::from([
            (Role::Packed, values.lease(ids[0])?.clone()),
            (Role::Q, values.lease(ids[1])?.clone()),
            (Role::K, values.lease(ids[2])?.clone()),
            (Role::Gate, values.lease(ids[3])?.clone()),
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
        let metadata =
            cut.scope()
                .source()
                .source()
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
            cut.require_whole()?;
            roles.insert(Role::O, values.lease(ids[4])?.clone());
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
                "canonical47 exact admitted control payload ceiling".into(),
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
        let packets = canonical47_packets::packets_with_fa(cut.includes_fa())?;
        let mut unique = BTreeSet::new();
        for (role, lease) in &roles {
            if !Arc::ptr_eq(lease.service(), values.service())
                || lease.service_generation() != values.service().generation()
                || lease.source_identity() != values.record().call().source()
                || !unique.insert(lease.id())
            {
                return Err(Error::State(
                    "canonical47 QK/KV same source/service/distinct original leases".into(),
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
                    "canonical47 actual QK/KV role extents differ packet".into(),
                ));
            }
        }
        let transport =
            ModelKvTransport::prepare_whole(&cut, domains.clone(), cache.clone(), init, cancelled)?;
        let bound = Arc::new(Self {
            site: Some(site),
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

    pub(crate) fn whole_site(
        &self,
    ) -> Result<&super::super::super::canonical47_model_prefix::AttentionSite> {
        self.site
            .as_ref()
            .ok_or_else(|| Error::State("whole actual attention site absent".into()))
    }
    pub(crate) fn handles(&self, roles: &[Role]) -> Result<Vec<cubecl::server::Handle>> {
        roles
            .iter()
            .map(|r| {
                self.roles
                    .get(r)
                    .map(ResidentLease::handle)
                    .ok_or_else(|| Error::State("whole actual role absent".into()))
            })
            .collect()
    }
    pub(crate) fn before_whole_append(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.transport
            .before_append(permit, self.whole_site()?.effect(), cancelled)
    }
    pub(crate) fn after_whole_append(
        &self,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.transport
            .after_append(permit, self.whole_site()?.effect(), cancelled)
    }
    pub(crate) fn whole_graph_retained(
        &self,
        cut: &Arc<Canonical47PrefixCut>,
        values: &Arc<ModelValues>,
    ) -> Result<bool> {
        self.require_values(cut, values)?;
        let site = self.whole_site()?;
        Ok(self.domains.layer() == site.layer()
            && site.nodes() == cut.attention_site(site.layer().index())?.nodes()
            && self.roles[&Role::Cache].id() == self._cache.lease().id()
            && self._cache.retained_whole_binding(cut, &self.domains)?)
    }
}
