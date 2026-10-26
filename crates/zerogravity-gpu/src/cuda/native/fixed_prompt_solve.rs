//! Finite original47 solve variants. Explicit captured operands, no model permit.
use crate::cuda::native::RetainedSource;
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

mod faults;
mod qualification;
mod resident;
#[cfg(test)]
mod tests;
pub use faults::{
    FixedPromptSolveFault, FixedPromptSolveFaultOutcome, qualify_fixed_prompt_solve_fault,
};
pub use qualification::qualify_fixed_prompt_solve_leaves;

const MIB: usize = 1024 * 1024;
const PROVIDER: usize = 128 * MIB;
const INPUT_BYTES: usize = 583_568;
const OUTPUT_BYTES: usize = 481_280;
const FIXTURE_ADMISSION: usize = 4 * MIB;
const READBACK_HOST: usize = 16 * MIB;
const CANCELLED: &str = "original47 solve leaf cancelled";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedPromptSolveVariant {
    Warp2Stage2,
    Warp4Stage4,
}
impl FixedPromptSolveVariant {
    pub fn solve_ptx_sha256(self) -> &'static str {
        match self {
            Self::Warp2Stage2 => "8201e61290e057badac428ff441391ceaf45c075c67daee4ffe24e816f86bb02",
            Self::Warp4Stage4 => "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
        }
    }
}
/// Reserve SAME original source 4MiB ticket before constructing these Vecs.
/// No expected Ai/W/U field exists; all seven fields are declared leaf inputs.
pub struct FixedPromptSolveFixture {
    pub a: Vec<u8>,
    pub k: Vec<u8>,
    pub v: Vec<u8>,
    pub beta: Vec<u8>,
    pub g: Vec<u8>,
    pub cu: Vec<u8>,
    pub indices: Vec<u8>,
    pub admission: WindowWorkspace,
}
impl FixedPromptSolveFixture {
    fn values(&self) -> [&[u8]; 7] {
        [
            &self.a,
            &self.k,
            &self.v,
            &self.beta,
            &self.g,
            &self.cu,
            &self.indices,
        ]
    }
    fn capacity(&self) -> Option<usize> {
        [
            &self.a,
            &self.k,
            &self.v,
            &self.beta,
            &self.g,
            &self.cu,
            &self.indices,
        ]
        .into_iter()
        .try_fold(0usize, |n, v| n.checked_add(v.capacity()))
    }
}
/// Borrowed only by the trusted original completion callback.
pub struct FixedPromptSolveReadback<'a> {
    pub variant: FixedPromptSolveVariant,
    pub repeat: usize,
    pub inputs: [&'a [u8]; 7],
    pub ai: &'a [u8],
    pub w: &'a [u8],
    pub u: &'a [u8],
}
#[derive(Debug)]
pub struct FixedPromptSolveReceipt {
    pub completed_cohorts: usize,
    pub zero_dispatches: usize,
    pub solve_dispatches: usize,
    pub recompute_dispatches: usize,
    pub compared_output_bytes: usize,
    pub immutable_operand_bytes_checked: usize,
    pub uploaded_value_bytes: u64,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
    pub parent_fullmodel_accepted: bool,
}
fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State(CANCELLED.into()));
    }
    Ok(())
}
fn validate_values(values: [&[u8]; 7]) -> Result<()> {
    for (i, (raw, bytes)) in values
        .into_iter()
        .zip([192512, 192512, 192512, 3008, 3008, 8, 8])
        .enumerate()
    {
        let valid = if matches!(i, 1 | 2) {
            super::finite_bf16(raw)
        } else if i < 5 {
            raw.len().is_multiple_of(4)
                && raw
                    .chunks_exact(4)
                    .all(|v| u32::from_le_bytes(v.try_into().unwrap()) & 0x7f80_0000 != 0x7f80_0000)
        } else {
            true
        };
        if raw.len() != bytes || !valid {
            return Err(Error::Malformed(
                "finite exact original47 leaf operand extent/dtype".into(),
            ));
        }
    }
    if values[5] != [0, 0, 0, 0, 47, 0, 0, 0] || values[6] != [0; 8] {
        return Err(Error::State(
            "original47 CU[0,47]/indices[[0,0]] only".into(),
        ));
    }
    Ok(())
}
fn validate_unwritten_ai(raw: &[u8]) -> Result<()> {
    if raw.len() != 96256 {
        return Err(Error::State("fixed-prompt solve whole Ai extent".into()));
    }
    // Original solve stores only diagonal/lower16x16 blocks. Upper blocks,
    // including columns48..63 at T47, retain the preceding full zeros_like.
    for token in 0..47 {
        for head in 0..16 {
            for column in 0..64 {
                if column / 16 > token / 16 {
                    let at = ((token * 16 + head) * 64 + column) * 2;
                    if raw[at..at + 2] != [0, 0] {
                        return Err(Error::State(
                            "fixed-prompt solve unwritten upper/tail Ai lost original zero".into(),
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}
struct LeafOwner {
    bound: Arc<RetainedSource>,
    fixture: FixedPromptSolveFixture,
    _readback_and_code: WindowWorkspace,
}
impl LeafOwner {
    fn new(model: Arc<RetainedSource>, fixture: FixedPromptSolveFixture) -> Result<Arc<Self>> {
        let bound = model.clone();
        if model.source().staging_chunk_bytes() != MIB
            || !fixture.admission.source().same_issuer(model.source())
            || fixture.admission.bytes() != FIXTURE_ADMISSION
            || fixture.capacity().is_none_or(|n| n > MIB)
        {
            return Err(Error::Unsupported(
                "closed original .8B source/admitted47 leaf".into(),
            ));
        }
        validate_values(fixture.values())?;
        if resident::host_bound()? > READBACK_HOST {
            return Err(Error::Limit(
                "fixed-prompt solve closed code/transfer/metadata admission".into(),
            ));
        }
        let readback_and_code = model.source().reserve_workspace(READBACK_HOST)?;
        Ok(Arc::new(Self {
            bound,
            fixture,
            _readback_and_code: readback_and_code,
        }))
    }
}
