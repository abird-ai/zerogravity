//! Same BF16 exponent test; only proved retained ranges may share a launch.
use super::*;
use crate::vllm_cuda_qualification::model_values::{ModelValues, bounded::FINITE_CODE_GROUP_WIDTH};
use std::sync::Weak;
use std::sync::atomic::AtomicU8;
use zerogravity_core::accelerator::program::ValueId;

const WIDTH: usize = FINITE_CODE_GROUP_WIDTH;
/// Cold metadata bound. B4 envelope arithmetic (derived, not silently
/// raised): at most ceil(327/8)=41 candidate groups, each contributing at
/// most the 4096-byte generated source, 9 Handle clones (8 members +
/// status), the member chunk, one kernel key and 512 bytes of Vec slack.
/// The whole sum stays under 256 KiB for every cubecl Handle layout; the
/// runtime check refuses beyond.
const METADATA_LIMIT: usize = 256 * 1024;
/// Per-run ledger bound. B4 arithmetic (derived): at most 327 pending entries
/// priced at 128 bytes of BTree node slack, 41 masks at 32 bytes and one run
/// struct: 327*128 + 41*32 + size_of::<CertificationRun>() <= 128 KiB.
const LEDGER_LIMIT: usize = 128 * 1024;
mod qualification;
pub(crate) use qualification::qualify;

/// One cold candidate group: kernel, grid and immutable member ValueIds.
type ColdCohortGroup = (PinnedKernel, u32, Arc<[ValueId]>);

struct Cohort {
    members: Vec<(ValueId, usize)>,
    launch: crate::vllm_ptx_inline::PreparedPinnedLaunch,
}
pub(super) struct CohortSet {
    groups: Vec<Cohort>,
    metadata_upper_bytes: usize,
}
struct Pending {
    producer: usize,
    group: usize,
    bit: u8,
}

/// Sealed, bounded retained-range binding. Concrete leases/status/service plus
/// the immutable candidate membership and the eligible sole-producer members;
/// no caller-supplied `all_retained=true` proof exists here.
pub(in crate::resident_program) struct FiniteCohortLeases {
    service: Arc<ResidentService>,
    status: ResidentLease,
    candidates: Box<[Box<[ValueId]>]>,
    leases: BTreeMap<u32, ResidentLease>,
    /// Eligible member lease ids -> (lease, SSA producer node); once-only.
    eligible: BTreeMap<zerogravity_core::device_events::BufferLeaseId, (ResidentLease, usize)>,
}

impl FiniteCohortLeases {
    fn lease(&self, value: ValueId) -> Result<&ResidentLease> {
        self.leases
            .get(&value.0)
            .ok_or_else(|| Error::State("finite cohort candidate lease absent".into()))
    }
    fn eligible_values(&self) -> BTreeMap<u32, usize> {
        self.leases
            .iter()
            .filter_map(|(id, lease)| {
                self.eligible
                    .get(&lease.id())
                    .map(|(_, producer)| (*id, *producer))
            })
            .collect()
    }
    /// N1 sealed binding from the current bounded `ModelValues` issuer:
    /// candidate groups from the immutable dedicated geometry, eligible
    /// members from the complete sealed lowerer census. Refusal order is the
    /// original: proof issuer first, then group preparation.
    pub(in crate::resident_program) fn for_values(values: &Arc<ModelValues>) -> Result<Self> {
        let proof = values.retained_finite_proof()?;
        if !Arc::ptr_eq(proof.record(), values.record()) || proof.members().len() > 512 {
            return Err(Error::State("finite cohort current source issuer".into()));
        }
        let candidates = values
            .finite_code_groups()?
            .into_iter()
            .map(|group| group.into_boxed_slice())
            .collect::<Vec<_>>();
        let mut leases = BTreeMap::new();
        for value in candidates.iter().flatten() {
            leases.insert(value.0, values.lease(*value)?.clone());
        }
        let mut eligible = BTreeMap::new();
        for &(value, producer) in proof.members() {
            let lease = values.lease(value)?;
            if eligible
                .insert(lease.id(), (lease.clone(), producer))
                .is_some()
            {
                return Err(Error::State(
                    "finite cohort duplicate eligible owner".into(),
                ));
            }
        }
        Ok(Self {
            service: values.service().clone(),
            status: values.status().clone(),
            candidates: candidates.into(),
            leases,
            eligible,
        })
    }
    /// B4 sealed binding from the call-local cohort value leases, the cold
    /// candidate membership and the census-derived sole-producer proof.
    pub(in crate::resident_program) fn for_cohort(
        values: &Arc<crate::resident_program::native_cohort::values::CohortValues>,
        status: ResidentLease,
        candidates: Box<[Box<[ValueId]>]>,
        proof: &crate::vllm_cuda_qualification::model_values::bounded::RetainedFiniteProof,
    ) -> Result<Self> {
        if proof.members().len() > 512 {
            return Err(Error::State("finite cohort current source issuer".into()));
        }
        let service = values.service().clone();
        let mut leases = BTreeMap::new();
        for value in candidates.iter().flatten() {
            leases.insert(value.0, values.lease(*value)?.clone());
        }
        let mut eligible = BTreeMap::new();
        for &(value, producer) in proof.members() {
            let lease = values.lease(value)?;
            if eligible
                .insert(lease.id(), (lease.clone(), producer))
                .is_some()
            {
                return Err(Error::State(
                    "finite cohort duplicate eligible owner".into(),
                ));
            }
        }
        Ok(Self {
            service,
            status,
            candidates,
            leases,
            eligible,
        })
    }
}

/// Chronology witness (concrete enum, no trait objects). ONE definition of the
/// producer/flush storage semantics for both native bodies.
pub(in crate::resident_program) enum ProducerWitness {
    /// N1: the bounded storage cursor of the current record.
    Values(Arc<ModelValues>),
    /// B4: the call-local cohort node/storage cursors. The run is owned by the
    /// body; this Weak back-reference creates no Arc cycle.
    Cohort(Weak<crate::resident_program::native_cohort::FullModelCohort>),
}
impl ProducerWitness {
    /// The deferred observation happens exactly while the original loop is
    /// inside the producer node; no earlier or replayed observation passes.
    fn require_storage_node(&self, node: usize) -> Result<()> {
        match self {
            Self::Values(values) => values.require_storage_node(node),
            Self::Cohort(body) => {
                let body = body
                    .upgrade()
                    .ok_or_else(|| Error::State("B4 finite cohort body absent".into()))?;
                if body.producer_cursor_active(node) {
                    Ok(())
                } else {
                    Err(Error::State(
                        "B4 finite cohort producer is not the active original node".into(),
                    ))
                }
            }
        }
    }
    fn require_finished_storage(&self) -> Result<()> {
        match self {
            Self::Values(values) => values.require_finished_storage(),
            Self::Cohort(body) => {
                let body = body
                    .upgrade()
                    .ok_or_else(|| Error::State("B4 finite cohort body absent".into()))?;
                if body.storage_cursor_finished() {
                    Ok(())
                } else {
                    Err(Error::State(
                        "B4 finite cohort whole actual traversal absent".into(),
                    ))
                }
            }
        }
    }
}

/// Cold per-geometry grouped finite definitions: immutable membership, kernel
/// and grid derived once from the candidate element counts. Per-call bind maps
/// the membership onto current certificates/handles; group masks never
/// specialize kernel keys at runtime.
pub(in crate::resident_program) struct FiniteCohortGroups {
    groups: Arc<[ColdCohortGroup]>,
    metadata_upper_bytes: usize,
}
impl FiniteCohortGroups {
    pub(in crate::resident_program) fn groups(&self) -> &[ColdCohortGroup] {
        &self.groups
    }
    /// Cold constructor over the immutable candidate element counts.
    pub(in crate::resident_program) fn prepare(elements: &[u32]) -> Result<Self> {
        let dedicated = vec![true; elements.len()];
        let candidates =
            crate::vllm_cuda_qualification::model_values::bounded::finite_code_groups(&dedicated)?;
        let mut groups = Vec::with_capacity(candidates.len());
        let mut bytes = groups.capacity()
            * (std::mem::size_of::<PinnedKernel>() + 4 + std::mem::size_of::<Arc<[ValueId]>>())
            + std::mem::size_of::<Self>();
        for candidate in &candidates {
            let sizes = candidate
                .iter()
                .map(|value| {
                    elements
                        .get(value.0 as usize)
                        .copied()
                        .ok_or_else(|| Error::State("finite cohort cold member extent".into()))
                })
                .collect::<Result<Vec<_>>>()?;
            let (kernel, grid) = definition(&sizes)?;
            bytes = bytes
                .checked_add(
                    kernel.retained_metadata_bytes()?
                        + candidate.len() * std::mem::size_of::<ValueId>()
                        + 512,
                )
                .ok_or_else(|| Error::Limit("finite cohort cold metadata overflow".into()))?;
            if bytes > METADATA_LIMIT {
                return Err(Error::Limit("finite cohort cold metadata bound".into()));
            }
            groups.push((kernel, grid, candidate.clone().into()));
        }
        Ok(Self {
            groups: groups.into(),
            metadata_upper_bytes: bytes,
        })
    }
    pub(in crate::resident_program) fn metadata_upper_bytes(&self) -> usize {
        self.metadata_upper_bytes
    }
}

/// Current source record and producer observations. No initialized/Ready state.
pub(in crate::resident_program) struct CertificationRun {
    certificates: Arc<CertificateSet>,
    cohorts: Arc<CohortSet>,
    leases: FiniteCohortLeases,
    witness: ProducerWitness,
    pending: BTreeMap<usize, Pending>,
    observed: Vec<AtomicU8>,
    masks: Vec<u8>,
    started: AtomicBool,
    flushed: AtomicBool,
    immediate_launches: AtomicUsize,
    group_launches: AtomicUsize,
}

fn source(elements: &[u32]) -> Result<(Arc<str>, u32)> {
    if !(2..=WIDTH).contains(&elements.len()) || elements.contains(&0) {
        return Err(Error::Unsupported(
            "finite cohort exact2..8 nonempty BF16 ranges".into(),
        ));
    }
    let mut text = String::from(
        "// cubecl.cuda.dynamic_shared_bytes=0\nextern \"C\" __global__ void zg_native_bf16_finite_cohort(",
    );
    for i in 0..elements.len() {
        text.push_str(&format!("const unsigned short* x{i},"));
    }
    text.push_str("unsigned int* status){unsigned int b=blockIdx.x;unsigned int i;\n");
    let mut start = 0u32;
    for (member, &n) in elements.iter().enumerate() {
        let end = start
            .checked_add(n.div_ceil(128))
            .ok_or_else(|| Error::Limit("finite cohort grid overflow".into()))?;
        text.push_str(&format!("{}(b<{end}u){{i=(b-{start}u)*128u+threadIdx.x;if(i<{n}u && (x{member}[i]&0x7f80u)==0x7f80u)atomicOr(status,1u);}}\n", if member==0 {"if"} else {"else if"}));
        start = end;
    }
    text.push_str("}\n");
    if text.len() > 4096 {
        return Err(Error::Limit("finite cohort generated source bound".into()));
    }
    Ok((text.into(), start))
}

/// Source geometry candidate only. No retained/write/once-only witness is issued.
pub(in crate::resident_program) fn definition(elements: &[u32]) -> Result<(PinnedKernel, u32)> {
    let (text, grid) = source(elements)?;
    let mut bindings = elements
        .iter()
        .map(|&n| (n as usize * 2, BufferIOAttr::ReadOnly))
        .collect::<Vec<_>>();
    bindings.push((4, BufferIOAttr::ReadWrite));
    Ok((
        PinnedKernel::new("zg_native_bf16_finite_cohort", text, 128, bindings.into())?,
        grid,
    ))
}

fn observe(observed: &AtomicU8, bit: u8, producer: usize, node: usize) -> Result<()> {
    if node != producer || observed.fetch_or(bit, Ordering::AcqRel) & bit != 0 {
        return Err(Error::State(
            "finite cohort exact once-only original producer".into(),
        ));
    }
    Ok(())
}
fn begin_flush(observed: &[AtomicU8], masks: &[u8], started: &AtomicBool) -> Result<()> {
    if observed.len() != masks.len()
        || observed
            .iter()
            .zip(masks)
            .any(|(a, &b)| a.load(Ordering::Acquire) != b)
        || started.swap(true, Ordering::AcqRel)
    {
        return Err(Error::State(
            "finite cohort incomplete/replayed terminal flush".into(),
        ));
    }
    Ok(())
}

fn complete_group(members: &[(ValueId, usize)], eligible: &BTreeMap<u32, usize>) -> bool {
    members
        .iter()
        .all(|(value, _)| eligible.contains_key(&value.0))
}

impl CohortSet {
    fn prepare_inner(
        certificates: &CertificateSet,
        leases: &FiniteCohortLeases,
        cold: Option<&Arc<FiniteCohortGroups>>,
    ) -> Result<Self> {
        if let Some(cold) = cold
            && cold.groups.len() != leases.candidates.len()
        {
            return Err(Error::State("finite cohort cold membership differs".into()));
        }
        for (index, candidate) in leases.candidates.iter().enumerate() {
            if let Some(cold) = cold
                && cold.groups[index].2.as_ref() != candidate.as_ref()
            {
                return Err(Error::State("finite cohort cold membership differs".into()));
            }
        }
        let mut groups = Vec::with_capacity(leases.candidates.len());
        let mut bytes =
            groups.capacity() * std::mem::size_of::<Cohort>() + std::mem::size_of::<Self>();
        for candidate in leases.candidates.iter() {
            let mut members = Vec::with_capacity(candidate.len());
            for value in candidate {
                let lease = leases.lease(*value)?;
                let index = *certificates
                    .by_lease
                    .get(&lease.id())
                    .ok_or_else(|| Error::State("finite cohort certificate absent".into()))?;
                require_certificate_range(
                    certificates,
                    index,
                    &lease.handle(),
                    &leases.status.handle(),
                    &leases.status,
                )?;
                members.push((*value, index));
            }
            let chunk = members.as_slice();
            let mut handles = Vec::with_capacity(chunk.len() + 1);
            let mut sizes = Vec::with_capacity(chunk.len());
            for &(_, index) in chunk {
                let handle = certificates.entries[index].launch.handles()[0].clone();
                let n = u32::try_from(handle.size_in_used() / 2)
                    .map_err(|_| Error::Limit("finite cohort element extent".into()))?;
                if handle.size_in_used() != u64::from(n) * 2 || n == 0 {
                    return Err(Error::State("finite cohort BF16 whole range".into()));
                }
                sizes.push(n);
                handles.push(handle);
            }
            handles.push(leases.status.handle());
            let (text, grid) = source(&sizes)?;
            // Body's original metadata owner covers this bounded cold payload
            // and every current ledger; no device allocation is added.
            bytes = bytes
                .checked_add(
                    text.len()
                        + handles.len() * std::mem::size_of::<Handle>()
                        + std::mem::size_of_val(chunk)
                        + PinnedKernel::KEY_METADATA_BYTES
                        + 512,
                )
                .ok_or_else(|| Error::Limit("finite cohort metadata overflow".into()))?;
            if bytes > METADATA_LIMIT {
                return Err(Error::Limit("finite cohort cold metadata bound".into()));
            }
            let kernel = match cold {
                Some(cold) => {
                    let (kernel, cold_grid, _) = &cold.groups[groups.len()];
                    if *cold_grid != grid {
                        return Err(Error::State("finite definition grid differs".into()));
                    }
                    kernel.clone()
                }
                None => definition(&sizes)?.0,
            };
            groups.push(Cohort {
                members: chunk.to_vec(),
                launch: kernel.prepare(leases.service.client(), [grid, 1, 1], handles)?,
            });
        }
        Ok(Self {
            groups,
            metadata_upper_bytes: bytes,
        })
    }
}

impl CertificationRun {
    pub(in crate::resident_program) fn bind(
        certificates: Arc<CertificateSet>,
        leases: FiniteCohortLeases,
        witness: ProducerWitness,
        cold: Option<Arc<FiniteCohortGroups>>,
    ) -> Result<Self> {
        let eligible = leases.eligible_values();
        if certificates.cohorts.get().is_none() {
            let prepared = Arc::new(CohortSet::prepare_inner(
                &certificates,
                &leases,
                cold.as_ref(),
            )?);
            let _ = certificates.cohorts.set(prepared);
        }
        let cohorts = certificates.cohorts.get().unwrap().clone();
        let mut pending = BTreeMap::new();
        let mut masks = vec![0; cohorts.groups.len()];
        for (group, c) in cohorts.groups.iter().enumerate() {
            // A changed/hidden writer keeps the original immediate certificate.
            if complete_group(&c.members, &eligible) {
                let handles = c.launch.handles();
                if handles.len() != c.members.len() + 1
                    || !same_range(handles.last().unwrap(), &leases.status.handle())
                {
                    return Err(Error::State(
                        "finite cohort original stored status range".into(),
                    ));
                }
                for (bit, &(value, index)) in c.members.iter().enumerate() {
                    if !same_range(&handles[bit], &leases.lease(value)?.handle()) {
                        return Err(Error::State(
                            "finite cohort original stored value range".into(),
                        ));
                    }
                    require_certificate_range(
                        &certificates,
                        index,
                        &leases.lease(value)?.handle(),
                        &leases.status.handle(),
                        &leases.status,
                    )?;
                    let mask = 1u8 << bit;
                    if pending
                        .insert(
                            index,
                            Pending {
                                producer: eligible[&value.0],
                                group,
                                bit: mask,
                            },
                        )
                        .is_some()
                    {
                        return Err(Error::State(
                            "finite cohort duplicate physical range".into(),
                        ));
                    }
                    masks[group] |= mask;
                }
            }
        }
        // BTree node slack, producer observations and both small vectors.
        if pending.len() * 128 + masks.len() * 32 + std::mem::size_of::<Self>() > LEDGER_LIMIT {
            return Err(Error::Limit(
                "finite cohort current ledger metadata bound".into(),
            ));
        }
        Ok(Self {
            certificates,
            cohorts,
            leases,
            witness,
            pending,
            observed: (0..masks.len()).map(|_| AtomicU8::new(0)).collect(),
            masks,
            started: AtomicBool::new(false),
            flushed: AtomicBool::new(false),
            immediate_launches: AtomicUsize::new(0),
            group_launches: AtomicUsize::new(0),
        })
    }
    pub(in crate::resident_program) fn certify(
        &self,
        node: usize,
        value: &DeviceBuffer,
        status: &Handle,
    ) -> Result<usize> {
        if self.started.load(Ordering::Acquire) {
            return Err(Error::State(
                "finite cohort producer after terminal flush".into(),
            ));
        }
        let index = *self
            .certificates
            .by_lease
            .get(&value.original_lease()?.id())
            .ok_or_else(|| Error::State("finite cohort current certificate absent".into()))?;
        let launch = require_certificate(
            &self.certificates,
            index,
            value,
            status,
            &self.leases.status,
        )?;
        if let Some(p) = self.pending.get(&index) {
            self.witness.require_storage_node(node)?;
            observe(&self.observed[p.group], p.bit, p.producer, node)?;
            Ok(0)
        } else {
            launch.launch()?;
            self.immediate_launches.fetch_add(1, Ordering::Release);
            Ok(1)
        }
    }
    pub(in crate::resident_program) fn flush(&self, cancel: Option<&AtomicBool>) -> Result<usize> {
        self.witness.require_finished_storage()?;
        begin_flush(&self.observed, &self.masks, &self.started)?;
        let mut launches = 0;
        for (index, c) in self.cohorts.groups.iter().enumerate() {
            if self.masks[index] == 0 {
                continue;
            }
            cancelled(cancel)?;
            self.leases.service.ensure_usable()?;
            c.launch.launch()?;
            self.group_launches.fetch_add(1, Ordering::Release);
            launches += 1;
        }
        self.flushed.store(true, Ordering::Release);
        Ok(launches)
    }
    pub(in crate::resident_program) fn require_flushed(&self) -> Result<()> {
        if self.flushed.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(Error::State(
                "finite cohort required before sampler/completion".into(),
            ))
        }
    }
    /// Source-owned client enqueue counts, not hardware activity enumeration.
    pub(in crate::resident_program) fn metrics(&self) -> [usize; 5] {
        [
            self.pending.len(),
            self.masks.iter().filter(|&&m| m != 0).count(),
            self.immediate_launches.load(Ordering::Acquire),
            self.group_launches.load(Ordering::Acquire),
            self.cohorts.metadata_upper_bytes,
        ]
    }
}

#[cfg(test)]
mod tests;
