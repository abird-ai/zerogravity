//! Exact stored operands/handles; no caller override at enqueue.
use super::*;
use crate::cuda::gdn::resident::solve_leaf::InverseZero;
use crate::cuda::{gdn::GdnBuffer, kernel::PinnedKernel};
use crate::resident_leases::ResidentReadback;
use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use cubecl_runtime::kernel::BufferIOAttr;

const ROLES: [GdnBuffer; 7] = [
    GdnBuffer::Kkt,
    GdnBuffer::K,
    GdnBuffer::V,
    GdnBuffer::Beta,
    GdnBuffer::GCumulative,
    GdnBuffer::QueryStarts,
    GdnBuffer::ChunkIndices,
];
const OUTPUT_ROLES: [GdnBuffer; 3] = [GdnBuffer::Inverse, GdnBuffer::W, GdnBuffer::U];
pub(super) fn host_bound() -> Result<usize> {
    // Four cohorts and construction/cache source copies, one initialization
    // H2D cohort and one live complete readback cohort; runtime peak unclaimed.
    let code = include_str!("solve2.cu")
        .len()
        .checked_add(include_str!("../../gdn/solve.cu").len())
        .and_then(|n| n.checked_add(include_str!("../../gdn/recompute.cu").len()))
        .and_then(|n| n.checked_mul(8));
    code.and_then(|n| n.checked_add(3 * INPUT_BYTES))
        .and_then(|n| n.checked_add(3 * (INPUT_BYTES + OUTPUT_BYTES)))
        .and_then(|n| n.checked_add(MIB)) // descriptors/readback vectors/zero-copy text
        .ok_or_else(|| Error::Limit("fixed-prompt solve original host sizing overflow".into()))
}
pub(super) struct Job {
    pub(super) variant: FixedPromptSolveVariant,
    pub(super) repeat: usize,
    pub(super) outputs: [ResidentLease; 3],
    zero: InverseZero,
    solve: LeafLaunch,
    solve_handles: Vec<Handle>,
    recompute: LeafLaunch,
    recompute_handles: Vec<Handle>,
    pub(super) accesses: Vec<(ResidentLease, DeviceAccess)>,
    pub(super) reads: Vec<ResidentReadback>,
}
pub(super) struct Prepared {
    pub(super) service: Arc<ResidentService>,
    pub(super) owner: Arc<LeafOwner>,
    pub(super) inputs: Vec<ResidentLease>,
    pub(super) jobs: Vec<Job>,
}
#[derive(Clone, Copy)]
struct Code {
    entry: &'static str,
    source: &'static str,
    ptx_sha256: &'static str,
    threads: u32,
    shared: usize,
    grid: [u32; 3],
}
fn solve_code(variant: FixedPromptSolveVariant) -> Code {
    match variant {
        FixedPromptSolveVariant::Warp2Stage2 => Code {
            entry: "zg_vllm_8f235650d676e07a",
            source: include_str!("solve2.cu"),
            ptx_sha256: variant.solve_ptx_sha256(),
            threads: 64,
            shared: 10240,
            grid: [1, 16, 1],
        },
        FixedPromptSolveVariant::Warp4Stage4 => Code {
            entry: "zg_vllm_116d2ffe705b6643",
            source: include_str!("../../gdn/solve.cu"),
            ptx_sha256: variant.solve_ptx_sha256(),
            threads: 128,
            shared: 10240,
            grid: [1, 16, 1],
        },
    }
}
fn recompute_code() -> Code {
    Code {
        entry: "zg_vllm_a73872840c0a9697",
        source: include_str!("../../gdn/recompute.cu"),
        ptx_sha256: "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
        threads: 256,
        shared: 40960,
        grid: [1, 16, 1],
    }
}
struct LeafLaunch {
    kernel: PinnedKernel,
    grid: [u32; 3],
    bindings: Vec<(GdnBuffer, usize, BufferIOAttr)>,
}
fn launch(code: Code, bindings: Vec<(GdnBuffer, usize, BufferIOAttr)>) -> Result<LeafLaunch> {
    if code.grid != [1, 16, 1]
        || !code.source.starts_with(&format!(
            "// cubecl.cuda.dynamic_shared_bytes={}\n",
            code.shared
        ))
        || !code.source.contains(code.ptx_sha256)
        || !code
            .source
            .contains(&format!("__launch_bounds__({})", code.threads))
        || !code.source.contains(&format!("void {}(", code.entry))
    {
        return Err(Error::State(
            "exact finite fixed-prompt solve source/entry/PTX/threads/shared contract".into(),
        ));
    }
    let kernel = PinnedKernel::new(
        code.entry,
        Arc::from(code.source),
        code.threads,
        bindings
            .iter()
            .map(|(_, bytes, io)| (*bytes, *io))
            .collect::<Vec<_>>()
            .into(),
    )?;
    Ok(LeafLaunch {
        kernel,
        grid: code.grid,
        bindings,
    })
}
fn solve(variant: FixedPromptSolveVariant) -> Result<LeafLaunch> {
    use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
    launch(
        solve_code(variant),
        vec![
            (GdnBuffer::Kkt, 192512, R),
            (GdnBuffer::Inverse, 96256, W),
            (GdnBuffer::QueryStarts, 8, R),
            (GdnBuffer::ChunkIndices, 8, R),
        ],
    )
}
fn recompute() -> Result<LeafLaunch> {
    use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
    launch(
        recompute_code(),
        vec![
            (GdnBuffer::K, 192512, R),
            (GdnBuffer::V, 192512, R),
            (GdnBuffer::Beta, 3008, R),
            (GdnBuffer::W, 192512, W),
            (GdnBuffer::U, 192512, W),
            (GdnBuffer::Inverse, 96256, R),
            (GdnBuffer::GCumulative, 3008, R),
            (GdnBuffer::QueryStarts, 8, R),
            (GdnBuffer::ChunkIndices, 8, R),
        ],
    )
}
fn handles(spec: &LeafLaunch, buffers: &BTreeMap<GdnBuffer, ResidentLease>) -> Result<Vec<Handle>> {
    spec.bindings
        .iter()
        .map(|(role, bytes, _)| {
            let lease = buffers
                .get(role)
                .ok_or_else(|| Error::State("fixed-prompt solve original binding absent".into()))?;
            if *bytes != lease.bytes() {
                return Err(Error::State(
                    "fixed-prompt solve no partial or foreign binding".into(),
                ));
            }
            Ok(lease.handle())
        })
        .collect()
}
fn distinct(leases: &[ResidentLease]) -> Result<()> {
    let ids: BTreeSet<_> = leases.iter().map(ResidentLease::id).collect();
    if ids.len() != leases.len() {
        return Err(Error::State(
            "fixed-prompt solve duplicate actual lease".into(),
        ));
    }
    for (i, a) in leases.iter().enumerate() {
        let a = a.handle().binding();
        let ar = a.range();
        for b in &leases[i + 1..] {
            let b = b.handle().binding();
            let br = b.range();
            if a.service == b.service
                && a.memory.id() == b.memory.id()
                && ar.start < br.end
                && br.start < ar.end
            {
                return Err(Error::State(
                    "fixed-prompt solve actual original allocation alias".into(),
                ));
            }
        }
    }
    Ok(())
}
impl Prepared {
    pub(super) fn prepare(
        service: Arc<ResidentService>,
        owner: Arc<LeafOwner>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        init.require_service(&service)?;
        init.retain_completion_owner(owner.clone())?;
        let mut inputs = Vec::with_capacity(7);
        for raw in owner.fixture.values() {
            not_cancelled(cancelled)?;
            inputs.push(init.upload_bytes(raw, BufferRole::Activation)?);
        }
        let mut jobs = Vec::with_capacity(4);
        for variant in [
            FixedPromptSolveVariant::Warp2Stage2,
            FixedPromptSolveVariant::Warp4Stage4,
        ] {
            for repeat in 0..2 {
                not_cancelled(cancelled)?;
                let outputs = [
                    init.allocate_empty(96256, BufferRole::Scratch)?,
                    init.allocate_empty(192512, BufferRole::Activation)?,
                    init.allocate_empty(192512, BufferRole::Activation)?,
                ];
                let buffers: BTreeMap<_, _> = ROLES
                    .into_iter()
                    .zip(inputs.iter().cloned())
                    .chain(OUTPUT_ROLES.into_iter().zip(outputs.iter().cloned()))
                    .collect();
                let zero = InverseZero::prepare(&service, &outputs[0])?;
                let solve = solve(variant)?;
                let recompute = recompute()?;
                let solve_handles = handles(&solve, &buffers)?;
                let recompute_handles = handles(&recompute, &buffers)?;
                solve
                    .kernel
                    .validate(service.client(), solve.grid, &solve_handles)?;
                recompute
                    .kernel
                    .validate(service.client(), recompute.grid, &recompute_handles)?;
                let accesses = inputs
                    .iter()
                    .cloned()
                    .map(|l| (l, DeviceAccess::Read))
                    .chain(outputs.iter().cloned().map(|l| (l, DeviceAccess::Write)))
                    .collect();
                let reads = inputs
                    .iter()
                    .chain(&outputs)
                    .map(|l| l.readback(0, l.bytes()))
                    .collect::<Result<Vec<_>>>()?;
                jobs.push(Job {
                    variant,
                    repeat,
                    outputs,
                    zero,
                    solve,
                    solve_handles,
                    recompute,
                    recompute_handles,
                    accesses,
                    reads,
                });
            }
        }
        let prepared = Arc::new(Self {
            service,
            owner,
            inputs,
            jobs,
        });
        distinct(&prepared.leases())?;
        // Same positive owner is attached before any native source compilation/launch.
        init.retain_completion_owner(prepared.clone())?;
        Ok(prepared)
    }
    pub(super) fn leases(&self) -> Vec<ResidentLease> {
        self.inputs
            .iter()
            .cloned()
            .chain(self.jobs.iter().flat_map(|j| j.outputs.iter().cloned()))
            .collect()
    }
    pub(super) fn through_solve(
        &self,
        index: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.service.ensure_usable()?;
        let job = &self.jobs[index];
        job.zero.enqueue(self.service.client())?;
        job.solve
            .kernel
            .launch(self.service.client(), job.solve.grid, &job.solve_handles)?;
        Ok(2)
    }
    pub(super) fn recompute(&self, index: usize, cancelled: Option<&AtomicBool>) -> Result<()> {
        not_cancelled(cancelled)?;
        let job = &self.jobs[index];
        job.recompute.kernel.launch(
            self.service.client(),
            job.recompute.grid,
            &job.recompute_handles,
        )
    }
    pub(super) fn validate<'a>(
        &self,
        index: usize,
        raw: &'a [Bytes],
    ) -> Result<FixedPromptSolveReadback<'a>> {
        if raw.len() != 10
            || raw
                .iter()
                .take(7)
                .zip(self.owner.fixture.values())
                .any(|(a, b)| &a[..] != b)
        {
            return Err(Error::State(
                "fixed-prompt solve actual full input/control unchanged check".into(),
            ));
        }
        if raw[7].len() != 96256
            || raw[8].len() != 192512
            || raw[9].len() != 192512
            || raw[7..].iter().any(|v| !super::super::finite_bf16(&v[..]))
        {
            return Err(Error::State(
                "fixed-prompt solve full finite Ai/W/U including masked tail".into(),
            ));
        }
        validate_unwritten_ai(&raw[7][..])?;
        let j = &self.jobs[index];
        Ok(FixedPromptSolveReadback {
            variant: j.variant,
            repeat: j.repeat,
            inputs: std::array::from_fn(|i| &raw[i][..]),
            ai: &raw[7][..],
            w: &raw[8][..],
            u: &raw[9][..],
        })
    }
}

#[cfg(test)]
mod source_tests {
    use super::*;
    use cubecl_runtime::kernel::KernelMetadata;
    #[test]
    fn actual_variant_and_cache_identity_cannot_inherit_solve4() {
        let two = solve(FixedPromptSolveVariant::Warp2Stage2).unwrap();
        let four = solve(FixedPromptSolveVariant::Warp4Stage4).unwrap();
        assert_ne!(two.kernel.id(), four.kernel.id());
        assert_eq!(solve_code(FixedPromptSolveVariant::Warp2Stage2).threads, 64);
        assert_eq!(
            solve_code(FixedPromptSolveVariant::Warp4Stage4).threads,
            128
        );
        assert_eq!(two.grid, [1, 16, 1]);
        assert_eq!(four.grid, [1, 16, 1]);
        assert_eq!(two.bindings, four.bindings);
        assert_eq!(
            two.bindings.iter().map(|(r, _, _)| *r).collect::<Vec<_>>(),
            vec![
                GdnBuffer::Kkt,
                GdnBuffer::Inverse,
                GdnBuffer::QueryStarts,
                GdnBuffer::ChunkIndices
            ]
        );
    }
    #[test]
    fn wrong_source_thread_or_entry_contract_refuses_before_client() {
        let good = solve_code(FixedPromptSolveVariant::Warp2Stage2);
        for bad in [
            Code {
                threads: 128,
                ..good
            },
            Code { shared: 0, ..good },
            Code {
                entry: "zg_vllm_116d2ffe705b6643",
                ..good
            },
            Code {
                ptx_sha256: "wrong",
                ..good
            },
            Code {
                grid: [1, 15, 1],
                ..good
            },
        ] {
            assert!(launch(bad, vec![(GdnBuffer::Kkt, 192512, BufferIOAttr::ReadOnly)]).is_err());
        }
        assert_eq!(recompute_code().threads, 256);
        assert_eq!(recompute_code().shared, 40960);
        assert_eq!(recompute_code().grid, [1, 16, 1]);
    }
}
