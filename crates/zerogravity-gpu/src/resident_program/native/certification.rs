//! Same BF16 exponent test; only proved retained ranges may share a launch.
use super::*;
use crate::vllm_cuda_qualification::model_values::{ModelValues, bounded::FINITE_CODE_GROUP_WIDTH};
use std::sync::atomic::AtomicU8;
use zerogravity_core::accelerator::program::ValueId;

const WIDTH: usize = FINITE_CODE_GROUP_WIDTH;
const METADATA_LIMIT: usize = 256 * 1024;
const LEDGER_LIMIT: usize = 128 * 1024;
mod qualification;
pub(crate) use qualification::qualify;

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
/// Current source record and producer observations. No initialized/Ready state.
pub(in crate::resident_program) struct CertificationRun {
    certificates: Arc<CertificateSet>,
    cohorts: Arc<CohortSet>,
    values: Arc<ModelValues>,
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
    fn prepare(certificates: &CertificateSet, values: &ModelValues) -> Result<Self> {
        let candidates = values.finite_code_groups()?;
        let mut groups = Vec::with_capacity(candidates.len());
        let mut bytes =
            groups.capacity() * std::mem::size_of::<Cohort>() + std::mem::size_of::<Self>();
        for candidate in candidates {
            let mut members = Vec::with_capacity(candidate.len());
            for value in candidate {
                let lease = values.lease(value)?;
                let index = *certificates
                    .by_lease
                    .get(&lease.id())
                    .ok_or_else(|| Error::State("finite cohort certificate absent".into()))?;
                require_certificate_range(
                    certificates,
                    index,
                    &lease.handle(),
                    &values.status().handle(),
                    values.status(),
                )?;
                members.push((value, index));
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
            handles.push(values.status().handle());
            let (text, grid) = source(&sizes)?;
            // Body's original4MiB metadata owner covers this bounded cold
            // payload and every current ledger; no device allocation is added.
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
            let (kernel, actual_grid) = definition(&sizes)?;
            if grid != actual_grid {
                return Err(Error::State("finite definition grid differs".into()));
            }
            groups.push(Cohort {
                members: chunk.to_vec(),
                launch: kernel.prepare(values.service().client(), [grid, 1, 1], handles)?,
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
        values: Arc<ModelValues>,
    ) -> Result<Self> {
        let proof = values.retained_finite_proof()?;
        if !Arc::ptr_eq(proof.record(), values.record()) || proof.members().len() > 512 {
            return Err(Error::State("finite cohort current source issuer".into()));
        }
        if certificates.cohorts.get().is_none() {
            let prepared = Arc::new(CohortSet::prepare(&certificates, &values)?);
            let _ = certificates.cohorts.set(prepared);
        }
        let cohorts = certificates.cohorts.get().unwrap().clone();
        let eligible = proof
            .members()
            .iter()
            .map(|&(value, node)| (value.0, node))
            .collect::<BTreeMap<_, _>>();
        let mut pending = BTreeMap::new();
        let mut masks = vec![0; cohorts.groups.len()];
        for (group, c) in cohorts.groups.iter().enumerate() {
            // A changed/hidden writer keeps the original immediate certificate.
            if complete_group(&c.members, &eligible) {
                let handles = c.launch.handles();
                if handles.len() != c.members.len() + 1
                    || !same_range(handles.last().unwrap(), &values.status().handle())
                {
                    return Err(Error::State(
                        "finite cohort original stored status range".into(),
                    ));
                }
                for (bit, &(value, index)) in c.members.iter().enumerate() {
                    if !same_range(&handles[bit], &values.lease(value)?.handle()) {
                        return Err(Error::State(
                            "finite cohort original stored value range".into(),
                        ));
                    }
                    require_certificate_range(
                        &certificates,
                        index,
                        &values.lease(value)?.handle(),
                        &values.status().handle(),
                        values.status(),
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
            values,
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
            self.values.status(),
        )?;
        if let Some(p) = self.pending.get(&index) {
            self.values.require_storage_node(node)?;
            observe(&self.observed[p.group], p.bit, p.producer, node)?;
            Ok(0)
        } else {
            launch.launch()?;
            self.immediate_launches.fetch_add(1, Ordering::Release);
            Ok(1)
        }
    }
    pub(in crate::resident_program) fn flush(&self, cancel: Option<&AtomicBool>) -> Result<usize> {
        self.values.require_finished_storage()?;
        begin_flush(&self.observed, &self.masks, &self.started)?;
        let mut launches = 0;
        for (index, c) in self.cohorts.groups.iter().enumerate() {
            if self.masks[index] == 0 {
                continue;
            }
            cancelled(cancel)?;
            self.values.service().ensure_usable()?;
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
