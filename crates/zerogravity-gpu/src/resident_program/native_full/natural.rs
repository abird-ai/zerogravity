//! Fresh one-use call binding; all native/storage/math resources are original.
use super::super::natural_binding::{same_lease, same_recipe};
use super::*;
use crate::resident_inference::native_fragment::natural_publication::{
    NaturalCallInput, NaturalCallState,
};
use crate::resident_leases::TiedWeight;
use cubecl::bytes::Bytes;

#[derive(Clone, Copy)]
enum NaturalStorageBinding {
    ClosedFive,
    Reusable,
}

struct NaturalBinding<'a> {
    storage: NaturalStorageBinding,
    cancel: Option<&'a AtomicBool>,
}

impl FullModelProgram {
    pub(crate) fn bind_natural(
        self: &Arc<Self>,
        input: Arc<NaturalCallInput>,
        state: Arc<NaturalCallState>,
        buffers: &[DeviceBuffer],
        positions: Arc<crate::resident_program::text_positions::TextPositionsPrepared>,
        table: Arc<TiedWeight>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        self.bind_natural_storage(
            input,
            state,
            buffers,
            positions,
            table,
            NaturalBinding {
                storage: NaturalStorageBinding::ClosedFive,
                cancel,
            },
        )
    }
    pub(crate) fn bind_reusable(
        self: &Arc<Self>,
        input: Arc<NaturalCallInput>,
        state: Arc<NaturalCallState>,
        buffers: &[DeviceBuffer],
        positions: Arc<crate::resident_program::text_positions::TextPositionsPrepared>,
        table: Arc<TiedWeight>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        state.require_reusable_submission_claimed()?;
        self.bind_natural_storage(
            input,
            state,
            buffers,
            positions,
            table,
            NaturalBinding {
                storage: NaturalStorageBinding::Reusable,
                cancel,
            },
        )
    }
    fn bind_natural_storage(
        self: &Arc<Self>,
        input: Arc<NaturalCallInput>,
        state: Arc<NaturalCallState>,
        buffers: &[DeviceBuffer],
        positions: Arc<crate::resident_program::text_positions::TextPositionsPrepared>,
        table: Arc<TiedWeight>,
        binding: NaturalBinding<'_>,
    ) -> Result<Arc<Self>> {
        let NaturalBinding { storage, cancel } = binding;
        cancelled(cancel)?;
        if let Some(scope) = &self.canonical47 {
            scope.require_native_activation()?;
        }
        if self.launched.load(Ordering::Acquire)
            || self.next_node.load(Ordering::Acquire) != 0
            || self.output_submitted.load(Ordering::Acquire)
            || self.natural.is_some()
            || !Arc::ptr_eq(&input, state.input())
            || !Arc::ptr_eq(&self.source, input.source())
        {
            return Err(Error::State(
                "natural binding requires UNUSED original cold body and exact input/state".into(),
            ));
        }
        Self::require_record(&self.source, input.record())?;
        same_recipe(&self.record, input.record())?;
        let old = self
            .bound
            .get()
            .ok_or_else(|| Error::State("natural original cold phase unbound".into()))?;
        let phase = super::super::natural_binding::slot(input.call())?;
        if old.phase != phase || state.states().len() != 24 {
            return Err(Error::State(
                "natural original physical phase/state count".into(),
            ));
        }
        // New canonical wrappers may carry the new original RootAttempt, but
        // the old route keeps its EXACT original prior/candidate physical leases.
        // Reuse binds genuine fresh canonical owners through the original adapter.
        for (a, b) in old.states.iter().zip(state.states()) {
            if a.layer() != b.layer() {
                return Err(Error::State(
                    "natural original indexed state layer changed".into(),
                ));
            }
            let (a, b) = (a.leases(), b.leases());
            if a.len() != b.len() {
                return Err(Error::State("natural state physical cohort size".into()));
            }
            for ((a, aa), (b, ba)) in a.iter().zip(&b) {
                if aa != ba {
                    return Err(Error::State("natural state access role changed".into()));
                }
                if matches!(storage, NaturalStorageBinding::ClosedFive) {
                    same_lease(a, b)?;
                } else if !Arc::ptr_eq(a.service(), b.service())
                    || a.source_identity() != b.source_identity()
                    || a.service_generation() != b.service_generation()
                    || a.bytes() != b.bytes()
                    || a.role() != b.role()
                {
                    return Err(Error::State(
                        "reusable actual canonical source/service/role differs".into(),
                    ));
                }
            }
        }
        let metadata = self.source.source().reserve_workspace(4 * 1024 * 1024)?;
        let values = ModelValues::bind(
            input.record().clone(),
            buffers,
            self.values.service().clone(),
            self.status().clone(),
        )?;
        let rms = self.rms.bind_natural(input.clone(), table)?;
        let ordinary = descriptor(&input.record().plan().nodes()[1])?;
        let ordinary_binding = BoundModelRmsSource::bind(self.source.clone(), ordinary.clone())?;
        let mut fused = BTreeMap::new();
        for (&node, cold) in &self.fused {
            let n = &input.record().plan().nodes()[node];
            let op = descriptor(n)?;
            let [a, b] = n.inputs.as_slice() else {
                return Err(Error::State("natural dual RMS arity".into()));
            };
            fused.insert(
                node,
                cold.bind_natural(
                    BoundModelRmsSource::bind(self.source.clone(), op.clone())?,
                    values.lease(*a)?,
                    values.lease(*b)?,
                )?,
            );
        }
        let original_greedy = self
            .greedy
            .get()
            .ok_or_else(|| Error::State("natural original reducer storage absent".into()))?;
        let greedy = match storage {
            NaturalStorageBinding::ClosedFive => original_greedy.bind_natural(values.clone())?,
            NaturalStorageBinding::Reusable => original_greedy.bind_reusable(values.clone())?,
        };
        let result = Arc::new(Self {
            gdn_family: self.gdn_family,
            source: self.source.clone(),
            record: input.record().clone(),
            values: values.clone(),
            rms,
            _ordinary_binding: ordinary_binding,
            fused,
            certificates: self.certificates.clone(),
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
            _readback: self._readback.clone(),
            _metadata: Arc::new(metadata),
            natural: Some(state.clone()),
            canonical47: None,
        });
        result
            .greedy
            .set(greedy)
            .map_err(|_| Error::State("natural output consumer fresh once binding".into()))?;
        let mut gdn = BTreeMap::new();
        let mut attention = BTreeMap::new();
        for domains in state.states() {
            let layer = domains.layer().index();
            if domains.layer().is_attention() {
                attention.insert(
                    layer,
                    old.attention
                        .get(&layer)
                        .ok_or_else(|| Error::State("natural cold attention absent".into()))?
                        .bind_natural(values.clone(), domains.clone(), positions.clone())?,
                );
            } else {
                gdn.insert(
                    layer,
                    old.gdn
                        .get(&layer)
                        .ok_or_else(|| Error::State("natural cold GDN absent".into()))?
                        .bind_natural(values.clone(), domains.clone())?,
                );
            }
        }
        let old_library = old.full_library()?;
        let library = Arc::new(FullModelLibrary {
            source: self.source.clone(),
            projections: old_library
                .projections
                .bind_natural(phase, values.clone())?,
            attention: old_library.attention.clone(),
            head: old_library
                .head
                .bind_natural(phase, values, result.final_normalizer()?)?,
            images: old_library.images.clone(),
        });
        result.bind_phase(phase, library, state.states().to_vec(), gdn, attention)?;
        old.gdn_arithmetic.require_same(
            &result
                .bound
                .get()
                .ok_or_else(|| Error::State("natural GDN arithmetic unbound".into()))?
                .gdn_arithmetic,
        )?;
        Ok(result)
    }
}
impl ResidentProgram {
    pub(crate) fn natural_cold_slot(
        &self,
    ) -> Result<super::super::natural_binding::ColdProgramSlot> {
        let body = self.full_model_body()?.clone();
        if body.launched.load(Ordering::Acquire) {
            return Err(Error::State("natural template was already executed".into()));
        }
        Ok(super::super::natural_binding::ColdProgramSlot::new(
            body,
            self.values.clone(),
            self.preflight.clone(),
        ))
    }
    pub(crate) fn from_natural_body(
        source: Arc<OriginalWindowModelSource>,
        body: Arc<FullModelProgram>,
        values: Vec<DeviceBuffer>,
        preflight: ResidentPreflight,
    ) -> Result<Self> {
        if values.len() != 327 {
            return Err(Error::State(
                "natural original physical value coverage".into(),
            ));
        }
        Ok(Self {
            _source_owner: ResidentProgramSource::Windows(source),
            plan: body.record().plan().clone(),
            preflight,
            values,
            status: body.status().handle(),
            owned_ranges: OwnedRanges::new(body.values().service().client().service_id()),
            body: ResidentProgramBody::NativeFull(body),
        })
    }
}
impl FullModelProgram {
    pub(crate) fn publication_extra_reads(&self) -> Result<Vec<ResidentReadback>> {
        if self.natural.is_none() {
            return Err(Error::State("natural extra reads on forced body".into()));
        }
        let bound = self
            .bound
            .get()
            .ok_or_else(|| Error::State("natural actual position binding absent".into()))?;
        let mut read = None;
        for binding in bound.attention.values() {
            let next = binding.publication_position_readback()?;
            if read.as_ref().is_some_and(|old: &ResidentReadback| {
                old.lease_id() != next.lease_id()
                    || old.offset() != next.offset()
                    || old.bytes() != next.bytes()
            }) {
                return Err(Error::State(
                    "natural all six attention consumers require SAME original positions".into(),
                ));
            }
            read = Some(next);
        }
        Ok(vec![read.ok_or_else(|| {
            Error::State("natural actual source positions absent".into())
        })?])
    }
    pub(crate) fn validate_publication_extra(&self, raw: &[Bytes]) -> Result<()> {
        let bound = self
            .bound
            .get()
            .ok_or_else(|| Error::State("natural actual position binding absent".into()))?;
        if self.natural.is_none() || bound.attention.len() != 6 {
            return Err(Error::State(
                "natural actual positions cohort absent".into(),
            ));
        }
        for binding in bound.attention.values() {
            binding.validate_publication_positions(raw)?;
        }
        Ok(())
    }
}
