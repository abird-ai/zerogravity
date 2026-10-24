//! First3 cut in the original full-model body, with no attention/Head owner.
use super::*;
use crate::vllm_cuda_gdn::Canonical47GdnArithmetic;
use crate::vllm_cuda_qualification::{
    canonical47_model_native::Canonical47ModelNativePrograms,
    canonical47_model_prefix::Canonical47PrefixCut,
};

pub(super) struct PrefixPhase {
    pub(super) cut: Arc<Canonical47PrefixCut>,
    pub(super) native: Arc<Canonical47ModelNativePrograms>,
}
impl BoundPhase {
    pub(super) fn full_library(&self) -> Result<&Arc<FullModelLibrary>> {
        if self.canonical47_prefix.is_some() {
            return Err(Error::Unsupported(
                "canonical47 first3 cannot enter full-model/attention/Head library".into(),
            ));
        }
        self.library
            .as_ref()
            .ok_or_else(|| Error::State("full-model native library absent".into()))
    }
}
impl FullModelProgram {
    /// Called after all original source/cache/native preparation flights completed.
    pub(crate) fn bind_canonical47_prefix(
        &self,
        cut: Arc<Canonical47PrefixCut>,
        native: Arc<Canonical47ModelNativePrograms>,
        states: Vec<Arc<BoundModelState>>,
        gdn: BTreeMap<usize, Arc<ModelGdnPrepared>>,
        arithmetic: Canonical47GdnArithmetic,
    ) -> Result<()> {
        if cut.is_whole() {
            return Err(Error::Unsupported(
                "wholeInitial47 requires distinct actual24-state binder".into(),
            ));
        }
        arithmetic.require_owned()?;
        let scope = self
            .canonical47
            .as_ref()
            .ok_or_else(|| Error::State("canonical47 model source scope absent".into()))?;
        if !Arc::ptr_eq(scope, cut.scope())
            || !Arc::ptr_eq(&self.record, scope.record())
            || !Arc::ptr_eq(&self.source, scope.source().source())
            || self.bound.get().is_some()
            || self.launched.load(Ordering::Acquire)
            || self.next_node.load(Ordering::Acquire) != 0
            || self.output_submitted.load(Ordering::Acquire)
            || self.greedy.get().is_some()
            || self.natural.is_some()
            || self.fused.len() != cut.duals()
            || states.len() != 3 + usize::from(cut.includes_kv())
            || gdn.len() != 3
        {
            return Err(Error::State(
                "canonical47 exactly once-bound real first3 model body".into(),
            ));
        }
        scope.require_values(&self.values)?;
        native.require_values(&cut, &self.values)?;
        for id in cut.values() {
            self.values.lease(*id)?;
        }
        if self
            .values
            .lease(ValueId(cut.values().len() as u32))
            .is_ok()
        {
            return Err(Error::State(
                "canonical47 prefix cannot retain future uncomputed value leases".into(),
            ));
        }
        let mut layers = BTreeSet::new();
        for state in &states {
            if state.call() != self.record.call()
                || !layers.insert(state.layer().index())
                || states
                    .first()
                    .is_some_and(|first| !first.same_request(state))
            {
                return Err(Error::State(
                    "canonical47 same original schedule/source/layer states".into(),
                ));
            }
            if state.layer().is_attention() {
                let attention = native.attention_binding().ok_or_else(|| {
                    Error::State("canonical47 actual Kv3 native owner absent".into())
                })?;
                state.require_canonical47_attention_kv(scope)?;
                if !cut.includes_kv()
                    || state.layer().index() != 3
                    || !Arc::ptr_eq(state, attention.domains())
                {
                    return Err(Error::State(
                        "canonical47 same original Kv3 state/native owner".into(),
                    ));
                }
            } else {
                state.require_canonical47_model_prefix(scope)?;
                let prepared = gdn
                    .get(&state.layer().index())
                    .ok_or_else(|| Error::State("canonical47 actual indexed GDN absent".into()))?;
                prepared.require_value_state(&self.values, state)?;
                prepared.require_canonical47_binding(scope, arithmetic)?;
            }
        }
        if layers != (0..3 + usize::from(cut.includes_kv())).collect()
            || gdn.keys().copied().collect::<BTreeSet<_>>() != (0..3).collect()
        {
            return Err(Error::State(
                "canonical47 exact actual GDN0..2 and optional Kv3 owners".into(),
            ));
        }
        let gdn_arithmetic = gdn
            .values()
            .next()
            .ok_or_else(|| Error::State("canonical47 GDN arithmetic absent".into()))?
            .arithmetic()
            .clone();
        self.gdn_family.require_same(gdn_arithmetic.family())?;
        for prepared in gdn.values() {
            gdn_arithmetic.require_same(prepared.arithmetic())?;
        }
        let expected_fused = cut
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(i, node)| {
                matches!(&node.operation,PlanOperation::TypedVllm08 {descriptor,..}
                if matches!(descriptor.kind(),Vllm08Kind::ResidualRms(_)))
                .then_some(i)
            })
            .collect::<BTreeSet<_>>();
        if self.fused.keys().copied().collect::<BTreeSet<_>>() != expected_fused {
            return Err(Error::State(
                "canonical47 original five indexed dual RMS actors".into(),
            ));
        }
        let mut accesses = BTreeMap::new();
        for (lease, access) in self
            .values
            .accesses()
            .into_iter()
            .chain(self.rms.accesses().iter().cloned())
        {
            insert_access(&mut accesses, lease, access);
        }
        for fused in self.fused.values() {
            for (lease, access) in fused.accesses() {
                insert_access(&mut accesses, lease.clone(), *access);
            }
        }
        for state in &states {
            for (lease, access) in state.leases() {
                insert_access(&mut accesses, lease, access);
            }
        }
        for prepared in gdn.values() {
            for (lease, access) in prepared.accesses() {
                insert_access(&mut accesses, lease, access);
            }
        }
        for (lease, access) in native.accesses() {
            insert_access(&mut accesses, lease.clone(), *access);
        }
        self.bound
            .set(BoundPhase {
                gdn_arithmetic,
                phase: 0,
                library: None,
                canonical47_prefix: Some(PrefixPhase { cut, native }),
                canonical_call: None,
                short_hf: None,
                states,
                gdn,
                attention: BTreeMap::new(),
                accesses: accesses.into_values().collect(),
            })
            .map_err(|_| Error::State("canonical47 model prefix already bound".into()))
    }
    pub(crate) fn execution_nodes(&self) -> Result<&[PlanNode]> {
        let bound = self
            .bound
            .get()
            .ok_or_else(|| Error::State("original model execution before binding".into()))?;
        Ok(match &bound.canonical47_prefix {
            Some(p) => p.cut.nodes(),
            None => self.record.plan().nodes(),
        })
    }
    pub(super) fn execution_value_count(&self) -> Result<u32> {
        let bound = self
            .bound
            .get()
            .ok_or_else(|| Error::State("original model readback before binding".into()))?;
        Ok(match &bound.canonical47_prefix {
            Some(p) => p.cut.values().len() as u32,
            None => 327,
        })
    }
    pub(super) fn finish_canonical47_prefix_nodes(&self) -> Result<()> {
        let prefix = self
            .bound
            .get()
            .and_then(|b| b.canonical47_prefix.as_ref())
            .ok_or_else(|| Error::State("canonical47 finite prefix binding absent".into()))?;
        if !self.launched.load(Ordering::Acquire) {
            return Err(Error::State("canonical47 prefix never launched".into()));
        }
        prefix
            .cut
            .require_finished(self.next_node.load(Ordering::Acquire))
    }
    pub(crate) fn canonical47_prefix_carries(&self) -> Result<Vec<StateSlot>> {
        let bound = self
            .bound
            .get()
            .ok_or_else(|| Error::State("original model carry before binding".into()))?;
        match &bound.canonical47_prefix {
            Some(prefix) => {
                self.finish_canonical47_prefix_nodes()?;
                Ok(prefix.cut.carries().to_vec())
            }
            None => Ok(Vec::new()),
        }
    }
    pub(super) fn validate_canonical47_prefix_readbacks(
        &self,
        raw: &[cubecl::bytes::Bytes],
    ) -> Result<()> {
        let cut = &self
            .bound
            .get()
            .and_then(|b| b.canonical47_prefix.as_ref())
            .ok_or_else(|| Error::State("canonical47 exact readback cut absent".into()))?
            .cut;
        let status = if cut.is_whole() { 19 } else { 4 };
        let values = cut.values().len();
        if raw.len() != status + 3 + values + cut.duals() * 4
            || raw[..status].iter().any(|r| r[..] != [0; 4])
        {
            return Err(Error::State(
                "canonical47 exactly original status/39 values/five dual readbacks".into(),
            ));
        }
        self.rms.validate_readback(&raw[status..status + 3])?;
        for (index, fused) in self.fused.values().enumerate() {
            let start = status + 3 + values + index * 4;
            fused.validate_readback(&raw[start..start + 4])?;
        }
        for (index, actual) in raw[status + 3..status + 3 + values].iter().enumerate() {
            if actual.len() != self.values.lease(ValueId(index as u32))?.bytes() {
                return Err(Error::State(
                    "canonical47 complete actual computed value bytes".into(),
                ));
            }
        }
        Ok(())
    }
}
impl ResidentProgram {
    pub(crate) async fn prepare_canonical47_model_prefix_values(
        service: Arc<ResidentService>,
        cut: Arc<Canonical47PrefixCut>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Self> {
        let source = cut.scope().source().source().clone();
        let plan = cut.scope().record().plan().clone();
        let (body, values, preflight) =
            FullModelProgram::prepare_canonical47_prefix_values(service.clone(), cut, cancel)
                .await?;
        Ok(Self {
            _source_owner: ResidentProgramSource::Windows(source),
            plan,
            preflight,
            values,
            status: body.status().handle(),
            owned_ranges: OwnedRanges::new(service.client().service_id()),
            body: ResidentProgramBody::NativeFull(body),
        })
    }
}
