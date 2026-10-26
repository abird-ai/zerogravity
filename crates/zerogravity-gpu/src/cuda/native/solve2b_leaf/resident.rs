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
    // Two cohorts and construction/cache source copies, one initialization
    // H2D cohort and one live complete readback cohort; runtime peak unclaimed.
    let code = include_str!("../../gdn/solve.cu")
        .len()
        .checked_add(include_str!("../../gdn/qwen2b_recompute.cu").len())
        .and_then(|n| n.checked_mul(8));
    code.and_then(|n| n.checked_add(3 * INPUT_BYTES))
        .and_then(|n| n.checked_add(3 * (INPUT_BYTES + OUTPUT_BYTES)))
        .and_then(|n| n.checked_add(MIB)) // descriptors/readback vectors/zero-copy text
        .ok_or_else(|| Error::Limit("solve2b original host sizing overflow".into()))
}
pub(super) struct Job {
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
fn solve_code() -> Code {
    Code {
        entry: "zg_vllm_116d2ffe705b6643",
        source: include_str!("../../gdn/solve.cu"),
        ptx_sha256: OWNED.solve_ptx_sha256(),
        threads: 128,
        shared: 10240,
        grid: [1, 16, 1],
    }
}
fn recompute_code() -> Code {
    Code {
        entry: "zg_vllm_eded6623852de87d",
        source: include_str!("../../gdn/qwen2b_recompute.cu"),
        ptx_sha256: OWNED.recompute_ptx_sha256(),
        threads: 256,
        shared: 32768,
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
            "exact finite solve2b source/entry/PTX/threads/shared contract".into(),
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
fn solve() -> Result<LeafLaunch> {
    use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
    launch(
        solve_code(),
        vec![
            (GdnBuffer::Kkt, 262144, R),
            (GdnBuffer::Inverse, 131072, W),
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
            (GdnBuffer::K, 262144, R),
            (GdnBuffer::V, 262144, R),
            (GdnBuffer::Beta, 4096, R),
            (GdnBuffer::W, 262144, W),
            (GdnBuffer::U, 262144, W),
            (GdnBuffer::Inverse, 131072, R),
            (GdnBuffer::GCumulative, 4096, R),
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
                .ok_or_else(|| Error::State("solve2b original binding absent".into()))?;
            if *bytes != lease.bytes() {
                return Err(Error::State("solve2b no partial or foreign binding".into()));
            }
            Ok(lease.handle())
        })
        .collect()
}
fn distinct(leases: &[ResidentLease]) -> Result<()> {
    let ids: BTreeSet<_> = leases.iter().map(ResidentLease::id).collect();
    if ids.len() != leases.len() {
        return Err(Error::State("solve2b duplicate actual lease".into()));
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
                    "solve2b actual original allocation alias".into(),
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
        let mut jobs = Vec::with_capacity(2);
        for repeat in 0..2 {
            not_cancelled(cancelled)?;
            let outputs = [
                init.allocate_empty(131072, BufferRole::Scratch)?,
                init.allocate_empty(262144, BufferRole::Activation)?,
                init.allocate_empty(262144, BufferRole::Activation)?,
            ];
            let buffers: BTreeMap<_, _> = ROLES
                .into_iter()
                .zip(inputs.iter().cloned())
                .chain(OUTPUT_ROLES.into_iter().zip(outputs.iter().cloned()))
                .collect();
            let zero = InverseZero::prepare_2b(&service, &outputs[0])?;
            let solve = solve()?;
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
    ) -> Result<Solve2bReadback<'a>> {
        if raw.len() != 10
            || raw
                .iter()
                .take(7)
                .zip(self.owner.fixture.values())
                .any(|(a, b)| &a[..] != b)
        {
            return Err(Error::State(
                "solve2b actual full input/control unchanged check".into(),
            ));
        }
        if raw[7].len() != 131072
            || raw[8].len() != 262144
            || raw[9].len() != 262144
            || raw[7..].iter().any(|v| !super::super::finite_bf16(&v[..]))
        {
            return Err(Error::State(
                "solve2b full finite Ai/W/U including masked tail".into(),
            ));
        }
        validate_unwritten_ai(&raw[7][..])?;
        let j = &self.jobs[index];
        Ok(Solve2bReadback {
            arithmetic: OWNED,
            original_reference: REFERENCE,
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
    #[test]
    fn only_actual_owned_sources_and_complete_t64_bindings() {
        let solve = solve().unwrap();
        let rec = recompute().unwrap();
        assert_eq!(solve_code().threads, 128);
        assert_eq!(solve_code().shared, 10240);
        assert_eq!(recompute_code().threads, 256);
        assert_eq!(recompute_code().shared, 32768);
        assert_eq!(solve.grid, [1, 16, 1]);
        assert_eq!(rec.grid, [1, 16, 1]);
        assert_eq!(
            solve
                .bindings
                .iter()
                .map(|(r, b, _)| (*r, *b))
                .collect::<Vec<_>>(),
            vec![
                (GdnBuffer::Kkt, 262144),
                (GdnBuffer::Inverse, 131072),
                (GdnBuffer::QueryStarts, 8),
                (GdnBuffer::ChunkIndices, 8)
            ]
        );
        assert_eq!(rec.bindings.len(), 9);
        assert_eq!(rec.bindings[5].1, 131072);
        assert_eq!(rec.bindings[6].1, 4096);
    }
    #[test]
    fn source_threads_entry_shared_or_grid_mutation_refuses_before_client() {
        let good = solve_code();
        for bad in [
            Code {
                threads: 64,
                ..good
            },
            Code { shared: 0, ..good },
            Code {
                entry: "other",
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
            assert!(launch(bad, vec![(GdnBuffer::Kkt, 262144, BufferIOAttr::ReadOnly)]).is_err());
        }
    }
}
