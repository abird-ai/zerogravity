//! All six source layers share the SAME26 exact original attention records.
use super::super::model::ModelAttentionBinding;
use super::*;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use crate::resident_leases::ResidentLease;
use zerogravity_core::device_events::DeviceAccess;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ModelAttentionSubmitted {
    pub(crate) native_launches: usize,
    pub(crate) device_copies: usize,
    pub(crate) transports: usize,
}
pub(crate) struct ModelAttentionPrograms {
    programs: Arc<AttentionPrograms>,
    // Native leases/record identities only: never a phase/previous-Ready owner.
    bindings: BTreeMap<(usize, usize), Arc<AttentionBindings>>,
    accesses: Vec<Vec<(ResidentLease, DeviceAccess)>>,
}
impl ModelAttentionPrograms {
    pub(crate) fn prepare(
        images: Arc<PreparedImages>,
        bindings: Vec<Arc<ModelAttentionBinding>>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        if bindings.len() != 30 || bindings.capacity() > 32 {
            return Err(Error::State(
                "full24 six attention layers times five actual phases".into(),
            ));
        }
        let mut keyed = BTreeMap::new();
        let mut native = Vec::with_capacity(30);
        let mut accesses = vec![BTreeMap::new(); 5];
        for binding in bindings {
            let phase = AttentionCase::ALL
                .iter()
                .position(|case| *case == binding.case())
                .ok_or_else(|| Error::State("full24 original attention case absent".into()))?;
            let layer = binding.layer();
            if ![3, 7, 11, 15, 19, 23].contains(&layer) || keyed.contains_key(&(phase, layer)) {
                return Err(Error::State(
                    "full24 exact indexed attention layer/case union".into(),
                ));
            }
            init.retain_completion_owner(binding.clone())?;
            native.push(binding.native().clone());
            for (lease, access) in binding.accesses()? {
                let entry = accesses[phase]
                    .entry(lease.id())
                    .or_insert((lease.clone(), access));
                if access == DeviceAccess::Write {
                    entry.1 = access;
                }
            }
            keyed.insert((phase, layer), binding.native().clone());
        }
        // Original factory deduplicates full image+ABI identities before native
        // prepare. Repeated source-layer bindings do not create156 records.
        let programs = AttentionPrograms::prepare_bindings(images, native, init, cancelled)?;
        let result = Arc::new(Self {
            programs,
            bindings: keyed,
            accesses: accesses
                .into_iter()
                .map(|a| a.into_values().collect())
                .collect(),
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn accesses(&self, phase: usize) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        self.accesses
            .get(phase)
            .cloned()
            .ok_or_else(|| Error::State("full24 attention access phase absent".into()))
    }
    pub(crate) fn require_phase(
        &self,
        phase: usize,
        bindings: &BTreeMap<usize, Arc<ModelAttentionBinding>>,
    ) -> Result<()> {
        if bindings.len() != 6 {
            return Err(Error::State(
                "full24 exactly six current-phase attention bindings".into(),
            ));
        }
        for (&layer, binding) in bindings {
            let native = self.bindings.get(&(phase, layer)).ok_or_else(|| {
                Error::State("full24 native attention phase/layer missing".into())
            })?;
            if !Arc::ptr_eq(native, binding.native())
                || AttentionCase::ALL.get(phase) != Some(&binding.case())
            {
                return Err(Error::State(
                    "full24 native attention record operands differ from SAME phase body".into(),
                ));
            }
        }
        Ok(())
    }
    /// Called once for the actual typed node by the original node loop.
    pub(crate) fn enqueue(
        &self,
        phase: usize,
        layer: usize,
        node: usize,
        binding: &ModelAttentionBinding,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ModelAttentionSubmitted> {
        not_cancelled(cancelled)?;
        let native = self
            .bindings
            .get(&(phase, layer))
            .ok_or_else(|| Error::State("full24 unbound attention phase/layer".into()))?;
        if !Arc::ptr_eq(native, binding.native())
            || binding.layer() != layer
            || AttentionCase::ALL.get(phase) != Some(&binding.case())
        {
            return Err(Error::State(
                "full24 bound phase body differs from sealed native record operands".into(),
            ));
        }
        binding.require_enqueue(permit.session(), permit, cancelled)?;
        let stage = binding.stage(node)?;
        let mut transports = 0;
        if stage == AttentionStage::Kv {
            transports += binding.before_kv(permit, cancelled)?;
        }
        // SAFETY: caller's original armed flight retains self and the current
        // phase-only access/outputs. Source wrappers seal actual producer leases.
        let actual = unsafe {
            self.programs
                .enqueue_bindings(binding.native(), binding.case(), stage, cancelled, None)
        }?;
        if stage == AttentionStage::Kv {
            transports += binding.after_kv(permit, cancelled)?;
        }
        Ok(ModelAttentionSubmitted {
            native_launches: actual.native_launches,
            device_copies: actual.device_copies,
            transports,
        })
    }
    pub(crate) unsafe fn retire_kernels_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        unsafe { self.programs.retire_kernels_after_physical_completion() }
    }
}
