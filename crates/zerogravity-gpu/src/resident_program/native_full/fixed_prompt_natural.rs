//! Cold327-value storage has no attempt/state authority and is never dispatched.
use super::*;
use crate::cuda::native::fixed_prompt_model::FixedPromptModelBinding;
use crate::resident_inference::native_fragment::fixed_prompt_natural::{
    FixedPromptCallInput, FixedPromptCallState,
};
use crate::resident_leases::TiedWeight;

pub(crate) struct CanonicalColdValues {
    pub(crate) body: Arc<FullModelProgram>,
    pub(crate) buffers: Vec<DeviceBuffer>,
    pub(crate) preflight: ResidentPreflight,
    greedy: Arc<PreparedBf16Greedy>,
}

impl FullModelProgram {
    /// Root-free arithmetic/storage only. Neither fake initial publication nor
    /// a Decode47 BoundModelState is required to prepare the N1 geometry.
    pub(crate) async fn prepare_fixed_prompt_unbound_values(
        service: Arc<ResidentService>,
        scope: Arc<FixedPromptModelBinding>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<CanonicalColdValues> {
        let (body, buffers, preflight) = Self::prepare_values_impl(
            service.clone(),
            scope.record().clone(),
            Some(scope),
            None,
            cancelled,
        )
        .await?;
        let mut init = service.begin_initialization(0)?;
        init.retain_completion_owner(body.clone())?;
        let greedy = PreparedBf16Greedy::prepare(
            body.source.clone(),
            body.values.clone(),
            &mut init,
            cancelled,
        )?;
        init.retain_completion_owner(greedy.clone())?;
        drop(init.complete_sync().await?);
        Ok(CanonicalColdValues {
            body,
            buffers,
            preflight,
            greedy,
        })
    }
}

impl CanonicalColdValues {
    /// Caller holds the original workspace through real completion and has
    /// consumed the fresh state's shared submission claim before this binding.
    pub(crate) fn bind_values(
        &self,
        input: &Arc<FixedPromptCallInput>,
        state: &Arc<FixedPromptCallState>,
        table: Arc<TiedWeight>,
    ) -> Result<Arc<FullModelProgram>> {
        state.require_submission_claimed()?;
        if !Arc::ptr_eq(input, state.input())
            || self.body.bound.get().is_some()
            || self.body.launched.load(Ordering::Acquire)
            || self.body.next_node.load(Ordering::Acquire) != 0
            || self.body.output_submitted.load(Ordering::Acquire)
            || self.body.greedy.get().is_some()
            || self.body.natural.is_some()
        {
            return Err(Error::State(
                "canonical binding requires root-free unused cold storage".into(),
            ));
        }
        crate::resident_program::fixed_prompt_binding::same_recipe(
            input.source(),
            self.body.record(),
            input.record(),
        )?;
        let metadata = input.source().reserve_workspace(4 * 1024 * 1024)?;
        let values = ModelValues::bind(
            input.record().clone(),
            &self.buffers,
            self.body.values.service().clone(),
            self.body.status().clone(),
        )?;
        let rms = self.body.rms.bind_fixed_prompt_call(input.clone(), table)?;
        let mut fused = BTreeMap::new();
        for (&node, cold) in &self.body.fused {
            let [a, b] = input.record().plan().nodes()[node].inputs.as_slice() else {
                return Err(Error::State(
                    "canonical current dual RMS actual inputs".into(),
                ));
            };
            fused.insert(
                node,
                cold.bind_fixed_prompt_call(input, node, values.lease(*a)?, values.lease(*b)?)?,
            );
        }
        let greedy = self.greedy.bind_fixed_prompt_call(values.clone())?;
        let result = Arc::new(FullModelProgram {
            attention_sites: self.body.attention_sites.clone(),
            gdn_family: self.body.gdn_family,
            fixed_prompt: Some(input.scope().clone()),
            source: input.source().clone(),
            record: input.record().clone(),
            values,
            rms,
            _ordinary_binding: input.scope().rms(1)?.clone(),
            fused,
            certificates: self.body.certificates.clone(),
            bound: OnceLock::new(),
            greedy: OnceLock::new(),
            launched: AtomicBool::new(false),
            next_node: AtomicUsize::new(0),
            output_submitted: AtomicBool::new(false),
            fault: AtomicUsize::new(0),
            fault_cut: AtomicBool::new(false),
            convolution_dispatches: AtomicUsize::new(0),
            internal_cancel: AtomicBool::new(false),
            native_launches: AtomicUsize::new(0),
            workspace_clears: AtomicUsize::new(0),
            device_copies: AtomicUsize::new(0),
            _readback: self.body._readback.clone(),
            _metadata: Arc::new(metadata),
            natural: None,
        });
        result
            .greedy
            .set(greedy)
            .map_err(|_| Error::State("canonical fresh output consumer already bound".into()))?;
        Ok(result)
    }
}

pub(super) struct CurrentCall {
    pub(super) input: Arc<FixedPromptCallInput>,
    pub(super) state: Arc<FixedPromptCallState>,
    pub(super) native:
        Arc<crate::cuda::native::fixed_prompt_model_native::natural::FixedPromptNativeCall>,
    pub(super) positions:
        Arc<crate::resident_program::text_positions::fixed_prompt_natural::FixedPromptPositions>,
}
impl FullModelProgram {
    pub(crate) fn bind_fixed_prompt_call(
        &self,
        input: Arc<FixedPromptCallInput>,
        state: Arc<FixedPromptCallState>,
        native: Arc<crate::cuda::native::fixed_prompt_model_native::natural::FixedPromptNativeCall>,
        positions: Arc<
            crate::resident_program::text_positions::fixed_prompt_natural::FixedPromptPositions,
        >,
        gdn: BTreeMap<usize, Arc<ModelGdnPrepared>>,
    ) -> Result<()> {
        state.require_submission_claimed()?;
        if self.bound.get().is_some()
            || self.launched.load(Ordering::Acquire)
            || self.next_node.load(Ordering::Acquire) != 0
            || !Arc::ptr_eq(state.input(), &input)
            || !Arc::ptr_eq(&self.record, input.record())
            || self.greedy.get().is_none()
            || self.natural.is_some()
            || self.fused.len() != 48
            || gdn.len() != 18
            || state.states().len() != 24
            || !positions.retained(&input, self.values.service())
        {
            return Err(Error::State(
                "canonical exactly once original current body binding".into(),
            ));
        }
        native.require_values(&input, &self.values)?;
        let mut all = BTreeMap::new();
        for (l, a) in self
            .values
            .accesses()
            .into_iter()
            .chain(self.rms.accesses().iter().cloned())
        {
            insert_access(&mut all, l, a);
        }
        for f in self.fused.values() {
            for (l, a) in f.accesses() {
                insert_access(&mut all, l.clone(), *a);
            }
        }
        for (layer, s) in state.states().iter().enumerate() {
            s.require_fixed_prompt_published(input.scope())?;
            if s.layer() != layer {
                return Err(Error::State(
                    "canonical exact current layer state cohort".into(),
                ));
            }
            for (l, a) in s.leases() {
                insert_access(&mut all, l, a);
            }
            if layer % 4 != 3 {
                gdn.get(&layer)
                    .ok_or_else(|| Error::State("canonical indexed GDN absent".into()))?
                    .require_value_state(&self.values, s)?;
            }
        }
        let arithmetic = gdn
            .values()
            .next()
            .ok_or_else(|| Error::State("canonical GDN arithmetic absent".into()))?
            .arithmetic()
            .clone();
        self.gdn_family.require_same(arithmetic.family())?;
        for p in gdn.values() {
            arithmetic.require_same(p.arithmetic())?;
            for (l, a) in p.accesses() {
                insert_access(&mut all, l, a);
            }
        }
        for (l, a) in native.accesses() {
            insert_access(&mut all, l.clone(), *a);
        }
        for (l, a) in self.greedy.get().unwrap().accesses() {
            insert_access(&mut all, l.clone(), *a);
        }
        self.bound
            .set(BoundPhase {
                gdn_arithmetic: arithmetic,
                phase: 0,
                library: None,
                fixed_prompt_prefix: None,
                states: state.states().to_vec(),
                gdn,
                attention: BTreeMap::new(),
                accesses: all.into_values().collect(),
                text: None,
                canonical_call: Some(CurrentCall {
                    input,
                    state,
                    native,
                    positions,
                }),
            })
            .map_err(|_| Error::State("canonical current body already bound".into()))
    }
    pub(crate) fn fixed_prompt_call_extra_reads(&self) -> Result<Vec<ResidentReadback>> {
        self.bound
            .get()
            .and_then(|b| b.canonical_call.as_ref())
            .ok_or_else(|| Error::State("canonical current binding absent".into()))?
            .native
            .extra_readbacks()
    }
    pub(crate) fn validate_fixed_prompt_call_extra(
        &self,
        raw: &[cubecl::bytes::Bytes],
    ) -> Result<()> {
        self.bound
            .get()
            .and_then(|b| b.canonical_call.as_ref())
            .ok_or_else(|| Error::State("canonical current binding absent".into()))?
            .native
            .validate_extra(raw)
    }
    pub(super) fn validate_fixed_prompt_call_readbacks(
        &self,
        raw: &[cubecl::bytes::Bytes],
    ) -> Result<()> {
        const HEADER: usize = 19;
        if raw.len() != 543
            || raw[0][..] != [0; 4]
            || raw[1..HEADER].iter().any(|b| b[..] != [0; 4])
        {
            return Err(Error::State(
                "canonical actual source statuses/543body descriptors".into(),
            ));
        }
        self.rms.validate_readback(&raw[HEADER..HEADER + 3])?;
        for (i, f) in self.fused.values().enumerate() {
            f.validate_readback(&raw[349 + i * 4..349 + (i + 1) * 4])?;
        }
        self.greedy
            .get()
            .ok_or_else(|| Error::State("canonical sampler absent".into()))?
            .validate_readback(&raw[541..543])?;
        Ok(())
    }
    pub(crate) fn fixed_prompt_call_status_reads(&self) -> Result<Vec<ResidentReadback>> {
        let b = self
            .bound
            .get()
            .and_then(|b| b.canonical_call.as_ref())
            .ok_or_else(|| Error::State("canonical current binding absent".into()))?;
        let mut r = vec![self.status().readback(0, 4)?];
        for p in self.bound.get().unwrap().gdn.values() {
            r.push(p.status_readback()?);
        }
        for read in b.native.extra_readbacks()?.into_iter().skip(1) {
            r.push(read);
        }
        r.extend(self.greedy.get().unwrap().readbacks()?);
        Ok(r)
    }
    pub(crate) fn fixed_prompt_call_sample_minimal(
        &self,
        raw: &[cubecl::bytes::Bytes],
    ) -> Result<(u32, u32)> {
        self.require_publication_finished()?;
        if raw.len() != 27 || raw[..25].iter().any(|b| b[..] != [0; 4]) {
            return Err(Error::State(
                "canonical actual transport/source status failed".into(),
            ));
        }
        self.greedy.get().unwrap().validate_readback(&raw[25..])
    }
    pub(crate) fn fixed_prompt_call_retained(&self) -> Result<bool> {
        let b = self
            .bound
            .get()
            .and_then(|b| b.canonical_call.as_ref())
            .ok_or_else(|| Error::State("canonical current binding absent".into()))?;
        Ok(Arc::ptr_eq(b.state.input(), &b.input)
            && b.native.retained()?
            && b.positions.retained(&b.input, self.values.service()))
    }
    pub(crate) fn fixed_prompt_readback_nodes(&self) -> Vec<usize> {
        self.fused.keys().copied().collect()
    }
}

impl FullModelProgram {
    pub(crate) fn arm_fixed_prompt_call_fault(&self, mode: usize) -> Result<()> {
        let current = self
            .bound
            .get()
            .and_then(|b| b.canonical_call.as_ref())
            .ok_or_else(|| Error::State("canonical fault before current binding".into()))?;
        if !matches!(mode, 1 | 2 | 19)
            || current.input.base() != 47
            || current.input.rows() != 1
            || self.launched.load(Ordering::Acquire)
            || self.next_node.load(Ordering::Acquire) != 0
            || self.fault.load(Ordering::Acquire) != 0
        {
            return Err(Error::State(
                "canonical cached-first once-only fault armer".into(),
            ));
        }
        self.fault.store(mode, Ordering::Release);
        Ok(())
    }
}

impl CanonicalColdValues {
    pub(crate) fn storage_accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut all = BTreeMap::new();
        for (l, a) in self
            .body
            .values
            .accesses()
            .into_iter()
            .chain(self.body.rms.accesses().iter().cloned())
            .chain(self.greedy.accesses().iter().cloned())
        {
            insert_access(&mut all, l, a);
        }
        for f in self.body.fused.values() {
            for (l, a) in f.accesses() {
                insert_access(&mut all, l.clone(), *a);
            }
        }
        all.into_values().collect()
    }
}
