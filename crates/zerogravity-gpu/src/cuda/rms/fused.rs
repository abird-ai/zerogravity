//! Draft: original layer0 or bound prefix dual-output RMS, no profile authority.
//! Reuse existing nine RMS stages; replace only the input-widen stage.
#[cfg(all(feature = "cuda-native", target_os = "linux"))]
use super::final_binding::BoundFinalRmsSource;
use super::layer_binding::BoundRmsSource;
use super::model_binding::BoundModelRmsSource;
use super::{COLUMNS, Columns, Rows, Spec, Stage, specification, specification_for};
use crate::cuda::kernel::PinnedKernel;
use crate::physical_quarantine::CompletionOwner;
use crate::resident_leases::{ResidentLease, ResidentReadback, ResidentService};
use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use cubecl_runtime::kernel::BufferIOAttr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::source_weights::{
    ParameterWindow, VerifiedTensorSource, WindowWorkspace,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

const METADATA_CAP: usize = 2 * 1024 * 1024 + Stage::ORDER.len() * PinnedKernel::KEY_METADATA_BYTES;
const GAMMA: &str = "model.language_model.layers.0.post_attention_layernorm.weight";
pub(super) const ADD_SOURCE: &str = include_str!("fused_add.cu");

#[path = "fused/witness.rs"]
mod witness;
pub(crate) use witness::{FusedRmsRetentionState, FusedRmsRetentionWitness};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Buffer {
    GammaRaw,
    GammaF32,
    GammaShifted,
    Attention,
    PriorResidual,
    SumF32,
    Squares,
    Mean,
    Epsilon,
    Inverse,
    NormalizedF32,
    Weighted,
    NormalizedBf16,
    ResidualBf16,
}
impl Buffer {
    fn owned(rows: Rows, columns: Columns) -> [(Self, usize); 12] {
        let count = columns.count();
        let dense = rows.count() * count;
        [
            (Self::GammaRaw, count * 2),
            (Self::GammaF32, count * 4),
            (Self::GammaShifted, count * 4),
            (Self::SumF32, dense * 4),
            (Self::Squares, dense * 4),
            (Self::Mean, rows.count() * 4),
            (Self::Epsilon, rows.count() * 4),
            (Self::Inverse, rows.count() * 4),
            (Self::NormalizedF32, dense * 4),
            (Self::Weighted, dense * 4),
            (Self::NormalizedBf16, dense * 2),
            (Self::ResidualBf16, dense * 2),
        ]
    }
    fn stage(stage: Stage) -> &'static [Self] {
        match stage {
            Stage::GammaWiden => &[Self::GammaRaw, Self::GammaF32],
            Stage::GammaShift => &[Self::GammaF32, Self::GammaShifted],
            Stage::InputWiden => &[
                Self::Attention,
                Self::PriorResidual,
                Self::SumF32,
                Self::ResidualBf16,
            ],
            Stage::Square => &[Self::SumF32, Self::Squares],
            Stage::Mean => &[Self::Squares, Self::Mean],
            Stage::Epsilon => &[Self::Mean, Self::Epsilon],
            Stage::Inverse => &[Self::Epsilon, Self::Inverse],
            Stage::Normalize => &[Self::SumF32, Self::Inverse, Self::NormalizedF32],
            Stage::Weight => &[Self::NormalizedF32, Self::GammaShifted, Self::Weighted],
            Stage::Bf16Store => &[Self::Weighted, Self::NormalizedBf16],
        }
    }
    fn read_only(self) -> bool {
        matches!(self, Self::GammaRaw | Self::Attention | Self::PriorResidual)
    }
}

fn add_bindings(rows: Rows, columns: Columns) -> Arc<[(usize, BufferIOAttr)]> {
    use BufferIOAttr::{ReadOnly as R, WriteOnly as W};
    let dense = rows.count() * columns.count();
    Arc::from([
        (dense * 2, R),
        (dense * 2, R),
        (dense * 4, W),
        (dense * 2, W),
    ])
}

pub(super) fn fused_specification(stage: Stage, rows: Rows, columns: Columns) -> Result<Spec> {
    if matches!(rows, Rows::FixedPrompt | Rows::CohortB4 { initial: true })
        && columns != Columns::Qwen08
    {
        return Err(Error::Unsupported(
            "fixed_prompt dual RMS is original1024 only".into(),
        ));
    }
    if stage != Stage::InputWiden {
        return if columns == Columns::Qwen08 {
            specification(stage, rows)
        } else {
            specification_for(stage, rows, columns)
        };
    }
    let source: Arc<str> = format!(
        "#define ZG_ROWS {}\n#define ZG_COLUMNS {}\n{ADD_SOURCE}",
        rows.count(),
        columns.count()
    )
    .into();
    Ok(Spec {
        kernel: PinnedKernel::new(
            "zg_vllm_post_rms_add",
            source,
            256,
            add_bindings(rows, columns),
        )?,
        grid: [(rows.count() * columns.count()).div_ceil(256) as u32, 1, 1],
    })
}

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("post-attention RMS cancelled".into()));
    }
    Ok(())
}

fn validate_operands(
    service: &Arc<ResidentService>,
    rows: Rows,
    columns: Columns,
    attention: &ResidentLease,
    residual: &ResidentLease,
) -> Result<()> {
    let bytes = rows.count() * columns.count() * 2;
    if attention.id() == residual.id()
        || [attention, residual].into_iter().any(|lease| {
            !Arc::ptr_eq(lease.service(), service)
                || lease.source_identity() != service.source_identity()
                || lease.bytes() != bytes
        })
    {
        return Err(Error::State(
            "post-attention RMS original producer leases/service/extent".into(),
        ));
    }
    // Producer leases may be uninitialized at preparation; their original
    // enclosing flight computes them before this consumer's enqueue.
    Ok(())
}

enum RmsSource {
    #[cfg(target_os = "linux")]
    LeafFinal(super::leaf_source::RmsLeafSource),
    Legacy(VerifiedTensorSource),
    Bound(BoundRmsSource),
    Model(BoundModelRmsSource),
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    Final(BoundFinalRmsSource),
}
impl RmsSource {
    fn columns(&self) -> Columns {
        match self {
            Self::Model(binding) if binding.geometry().width == 2048 => Columns::Qwen2b,
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            Self::Final(binding) if binding.geometry().width == 2048 => Columns::Qwen2b,
            #[cfg(target_os = "linux")]
            Self::LeafFinal(source) => source.columns(),
            _ => Columns::Qwen08,
        }
    }
    fn source(&self) -> &VerifiedTensorSource {
        match self {
            #[cfg(target_os = "linux")]
            Self::LeafFinal(source) => source.source(),
            Self::Legacy(source) => source,
            Self::Bound(binding) => binding.source(),
            Self::Model(binding) => binding.source(),
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            Self::Final(binding) => binding.source(),
        }
    }
}

struct FusedRmsSource {
    source: VerifiedTensorSource,
    _parameter: ParameterWindow,
    gamma: Vec<u8>,
    specs: Arc<[Spec]>,
    rows: Rows,
    columns: Columns,
    #[cfg(target_os = "linux")]
    _leaf: Option<super::leaf_source::RmsLeafSource>,
    gamma_name: String,
    _binding: Option<BoundRmsSource>,
    _model_binding: Option<BoundModelRmsSource>,
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    _final_binding: Option<BoundFinalRmsSource>,
    // Actual payloads/specs drop before original host admission.
    _workspace: WindowWorkspace,
    _metadata: WindowWorkspace,
}
pub(super) fn pooled_metadata_bytes() -> Result<usize> {
    // Closed layout:14 leases/accesses,10 stages and at most4 handles/stage.
    // Original gamma/window/current-binding payloads retain separate charges.
    let parts = [
        std::mem::size_of::<FusedRmsSource>(),
        std::mem::size_of::<FusedRmsPrepared>(),
        14 * std::mem::size_of::<(Buffer, ResidentLease)>(),
        14 * std::mem::size_of::<(ResidentLease, DeviceAccess)>(),
        10 * std::mem::size_of::<crate::cuda::kernel::PreparedPinnedLaunch>(),
        40 * std::mem::size_of::<Handle>(),
        32 * (2 * std::mem::size_of::<usize>() + std::mem::align_of::<usize>()),
    ];
    parts
        .into_iter()
        .try_fold(0usize, |n, p| n.checked_add(p))
        .ok_or_else(|| Error::Limit("pooled RMS source/binding metadata extent".into()))
}
fn pooled_source_host_bytes(columns: usize) -> Result<usize> {
    columns
        .checked_mul(6)
        .and_then(|n| n.checked_add(64 * 1024))
        .ok_or_else(|| Error::Limit("pooled dual actual gamma payload capacity".into()))
}
fn source_owner(
    binding: RmsSource,
    rows: Rows,
    definitions: Option<Arc<super::definitions::RmsDefinitions>>,
) -> Result<Arc<FusedRmsSource>> {
    let pooled = definitions.is_some();
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    if pooled
        && !matches!(&binding,RmsSource::Model(b) if b.operator().call().rows().len() == rows.count())
    {
        return Err(Error::State(
            "pooled source budget requires actual text binding".into(),
        ));
    }
    #[cfg(not(all(feature = "cuda-native", target_os = "linux")))]
    if pooled {
        return Err(Error::Unsupported("pooled source unavailable".into()));
    }
    #[cfg(target_os = "linux")]
    let leaf = match &binding {
        RmsSource::LeafFinal(source) => Some(source.clone()),
        _ => None,
    };
    let columns = binding.columns();
    let count = columns.count();
    let metadata = binding.source().reserve_workspace(if pooled {
        pooled_metadata_bytes()?
    } else {
        METADATA_CAP
    })?;
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    let final_binding = match &binding {
        RmsSource::Final(binding) => Some(binding.clone()),
        _ => None,
    };
    let (source, bound, model_bound, gamma_name) = match binding {
        #[cfg(target_os = "linux")]
        RmsSource::LeafFinal(binding) => {
            binding.require_source(binding.source())?;
            (
                binding.source().clone(),
                None,
                None,
                binding.final_gamma_name().to_owned(),
            )
        }
        RmsSource::Legacy(source) => (source, None, None, GAMMA.to_owned()),
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        RmsSource::Final(binding) => {
            binding.require_source(binding.source())?;
            (
                binding.source().clone(),
                None,
                None,
                binding.gamma_name().to_owned(),
            )
        }
        RmsSource::Bound(binding) => {
            binding.require_source(binding.source())?;
            let name = binding.parameter().name.clone();
            (binding.source().clone(), Some(binding), None, name)
        }
        RmsSource::Model(binding) => {
            binding.require_source(binding.source())?;
            if binding.operator().call().rows().len() != rows.count() {
                return Err(Error::State("full-model RMS actual request rows".into()));
            }
            let name = binding.gamma_name().to_owned();
            (binding.source().clone(), None, Some(binding), name)
        }
    };
    let physical = Buffer::owned(rows, columns)
        .iter()
        .map(|(_, bytes)| bytes)
        .sum::<usize>();
    let readback = count * 2 + rows.count() * count * 8;
    let host = if pooled {
        pooled_source_host_bytes(count)?
    } else {
        physical * 3 + readback * 3 + 64 * 1024
    };
    let workspace = source.reserve_workspace(host)?;
    let parameter = source.parameter(&gamma_name)?;
    if parameter.tensor().name != gamma_name || parameter.tensor().shape != [count as u64] {
        return Err(Error::Malformed(format!(
            "original post-attention gamma{count} shape/name"
        )));
    }
    let parameter = source.read_parameter(&parameter, 0..count)?;
    if parameter.parameter().source_identity() != source.source_identity() {
        return Err(Error::State("original post-attention gamma source".into()));
    }
    // SAME actual loaded BF16 gamma conversion as original input RMS.
    let gamma = if columns == Columns::Qwen08 {
        super::resident::loaded_gamma(&parameter)?
    } else {
        super::resident::loaded_gamma_for(&parameter, columns)?
    };
    let specs = if let Some(definitions) = definitions {
        definitions.specs(rows, columns, true)?
    } else {
        Stage::ORDER
            .into_iter()
            .map(|stage| fused_specification(stage, rows, columns))
            .collect::<Result<Vec<_>>>()?
            .into()
    };
    Ok(Arc::new(FusedRmsSource {
        source,
        _parameter: parameter,
        gamma,
        specs,
        rows,
        columns,
        #[cfg(target_os = "linux")]
        _leaf: leaf,
        gamma_name,
        _binding: bound,
        _model_binding: model_bound,
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        _final_binding: final_binding,
        _workspace: workspace,
        _metadata: metadata,
    }))
}

/// Original borrowed producer leases only; no raw activation/input constructor.
pub(crate) struct FusedRmsPrepared {
    // Drop cloned physical handles before their lease/admission owners.
    stages: Arc<[crate::cuda::kernel::PreparedPinnedLaunch]>,
    service: Arc<ResidentService>,
    leases: Arc<[(Buffer, ResidentLease)]>,
    accesses: Arc<[(ResidentLease, DeviceAccess)]>,
    owner: Arc<FusedRmsSource>,
    model_binding: Option<BoundModelRmsSource>,
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pooled: Option<Arc<super::pooled::PooledRmsActor>>,
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    _text_binding: Option<Arc<WindowWorkspace>>,
}
impl FusedRmsPrepared {
    /// Caller holds its one original preparation fair slot. All allocation,
    /// gamma upload and readback construction precede native pool activation.
    pub(crate) async fn prepare(
        service: Arc<ResidentService>,
        source: VerifiedTensorSource,
        rows: usize,
        attention: ResidentLease,
        residual: ResidentLease,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        Self::prepare_source(
            service,
            RmsSource::Legacy(source),
            rows,
            attention,
            residual,
            correlation,
            cancelled,
        )
        .await
    }
    pub(crate) async fn prepare_bound(
        service: Arc<ResidentService>,
        binding: BoundRmsSource,
        rows: usize,
        branch: ResidentLease,
        residual: ResidentLease,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        Self::prepare_source(
            service,
            RmsSource::Bound(binding),
            rows,
            branch,
            residual,
            correlation,
            cancelled,
        )
        .await
    }
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) async fn prepare_final(
        service: Arc<ResidentService>,
        binding: BoundFinalRmsSource,
        rows: usize,
        branch: ResidentLease,
        residual: ResidentLease,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        Self::prepare_source(
            service,
            RmsSource::Final(binding),
            rows,
            branch,
            residual,
            correlation,
            cancelled,
        )
        .await
    }
    /// Same arithmetic and producer-lease path; full-layer source metadata only.
    pub(crate) async fn prepare_model(
        service: Arc<ResidentService>,
        binding: BoundModelRmsSource,
        branch: ResidentLease,
        residual: ResidentLease,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        let rows = Rows::new(binding.operator().call().rows().len())?;
        Self::prepare_model_rows(
            service,
            binding,
            rows,
            branch,
            residual,
            correlation,
            cancelled,
        )
        .await
    }

    /// Explicit row storage; source eligibility is checked before this boundary.
    pub(crate) async fn prepare_model_rows(
        service: Arc<ResidentService>,
        binding: BoundModelRmsSource,
        rows: Rows,
        branch: ResidentLease,
        residual: ResidentLease,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        if !matches!(
            binding.operator().kind(),
            zerogravity_core::accelerator::program::SourceOperatorKind::ResidualRms(_)
        ) || binding.operator().call().rows().len() != rows.count()
        {
            return Err(Error::State(
                "dual RMS actual operator/row storage differs".into(),
            ));
        }
        Self::prepare_rows(
            service,
            RmsSource::Model(binding),
            rows,
            branch,
            residual,
            correlation,
            cancelled,
        )
        .await
    }
    /// Every actual full-record RMS site; no component/layer0 retagging.
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) async fn prepare_fixed_prompt_model(
        service: Arc<ResidentService>,
        scope: &crate::cuda::native::fixed_prompt_model::FixedPromptModelBinding,
        node: usize,
        branch: ResidentLease,
        residual: ResidentLease,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let binding = scope.rms(node)?.clone();
        if !matches!(
            binding.operator().kind(),
            zerogravity_core::accelerator::program::SourceOperatorKind::ResidualRms(_)
        ) || binding.operator().call() != scope.record().call()
        {
            return Err(Error::State(
                "fixed_prompt actual dual RMS full-record source site".into(),
            ));
        }
        let rows = match scope.kind() {
            crate::cuda::native::fixed_prompt_model::FixedPromptCallKind::Prefill => {
                Rows::FixedPrompt
            }
            crate::cuda::native::fixed_prompt_model::FixedPromptCallKind::Decode { .. } => {
                Rows::One
            }
        };
        let prepared = Self::prepare_rows(
            service,
            RmsSource::Model(binding.clone()),
            rows,
            branch,
            residual,
            correlation,
            cancelled,
        )
        .await?;
        prepared.require_full_model_operator(binding.operator())?;
        Ok(prepared)
    }

    /// Authentic observed branch/residual LEAF, not source-produced layer math.
    #[cfg(target_os = "linux")]
    pub(super) async fn prepare_2b_final_leaf(
        service: Arc<ResidentService>,
        source: super::leaf_source::RmsLeafSource,
        rows: usize,
        branch: ResidentLease,
        residual: ResidentLease,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        source.require_source(source.source())?;
        Self::prepare_source(
            service,
            RmsSource::LeafFinal(source),
            rows,
            branch,
            residual,
            correlation,
            cancelled,
        )
        .await
    }
    /// Fixed47 actual source gamma; branch/residual remain explicit LEAF inputs.
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(super) async fn prepare_fixed_prompt_final_leaf(
        service: Arc<ResidentService>,
        source: super::leaf_source::RmsLeafSource,
        branch: ResidentLease,
        residual: ResidentLease,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::fixed_prompt::not_cancelled(cancelled)?;
        source.require_source(source.source())?;
        Self::prepare_rows(
            service,
            RmsSource::Final(source.final_binding().clone()),
            Rows::FixedPrompt,
            branch,
            residual,
            correlation,
            cancelled,
        )
        .await
    }

    async fn prepare_source(
        service: Arc<ResidentService>,
        binding: RmsSource,
        rows: usize,
        attention: ResidentLease,
        residual: ResidentLease,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let rows = Rows::new(rows)?;
        Self::prepare_rows(
            service,
            binding,
            rows,
            attention,
            residual,
            correlation,
            cancelled,
        )
        .await
    }
    async fn prepare_rows(
        service: Arc<ResidentService>,
        binding: RmsSource,
        rows: Rows,
        attention: ResidentLease,
        residual: ResidentLease,
        correlation: u64,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        Self::prepare_rows_storage(
            service,
            binding,
            rows,
            attention,
            residual,
            (correlation, cancelled),
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            None,
        )
        .await
    }

    async fn prepare_rows_storage(
        service: Arc<ResidentService>,
        binding: RmsSource,
        rows: Rows,
        attention: ResidentLease,
        residual: ResidentLease,
        control: (u64, Option<&AtomicBool>),
        #[cfg(all(feature = "cuda-native", target_os = "linux"))] pooled: Option<
            Arc<super::pooled::PooledRmsActor>,
        >,
    ) -> Result<Arc<Self>> {
        let (correlation, cancelled) = control;
        not_cancelled(cancelled)?;
        if rows == Rows::FixedPrompt && binding.columns() != Columns::Qwen08 {
            return Err(Error::Unsupported("fixed_prompt dual source width".into()));
        }
        if binding.source().source_identity() != service.source_identity() {
            return Err(Error::State(
                "post-attention RMS source/service identity".into(),
            ));
        }
        validate_operands(&service, rows, binding.columns(), &attention, &residual)?;
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        let _is_pooled = pooled.is_some();
        #[cfg(not(all(feature = "cuda-native", target_os = "linux")))]
        let _is_pooled = false;
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        let definitions = pooled.as_ref().map(|p| p.definitions());
        #[cfg(not(all(feature = "cuda-native", target_os = "linux")))]
        let definitions = None;
        let owner = source_owner(binding, rows, definitions)?;
        let layout = Buffer::owned(rows, owner.columns);
        let physical = layout
            .iter()
            .filter(|(role, _)| {
                #[cfg(all(feature = "cuda-native", target_os = "linux"))]
                if pooled.as_ref().is_some_and(|p| {
                    Self::scratch_role(*role).is_some()
                        || p.has_workspace_outputs()
                            && matches!(
                                role,
                                Buffer::SumF32 | Buffer::NormalizedBf16 | Buffer::ResidualBf16
                            )
                }) {
                    return false;
                }
                let _ = role;
                true
            })
            .map(|(_, bytes)| bytes)
            .sum::<usize>();
        if service.remaining_capacity()? < physical {
            return Err(Error::Limit(
                "post-attention RMS capacity before allocation".into(),
            ));
        }
        not_cancelled(cancelled)?;
        let mut init = service.begin_initialization(correlation)?;
        init.retain_completion_owner(owner.clone())?;
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        if let Some(pooled) = &pooled {
            init.retain_completion_owner(pooled.clone())?;
        }
        let mut leases = Vec::with_capacity(14);
        leases.push((Buffer::Attention, attention));
        leases.push((Buffer::PriorResidual, residual));
        for (role, bytes) in layout {
            not_cancelled(cancelled)?;
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            if let Some(shared) = pooled.as_ref().zip(Self::scratch_role(role)) {
                leases.push((role, shared.0.lease(shared.1, bytes)?));
                continue;
            }
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            if let Some(pooled) = &pooled {
                let output = match role {
                    Buffer::SumF32 => Some(super::pooled::RmsWorkspaceOutput::Sum),
                    Buffer::NormalizedBf16 => Some(super::pooled::RmsWorkspaceOutput::Normalized),
                    Buffer::ResidualBf16 => Some(super::pooled::RmsWorkspaceOutput::Residual),
                    _ => None,
                };
                if let Some(output_role) = output
                    && let Some(lease) = pooled.workspace_output(output_role, bytes)?
                {
                    leases.push((role, lease));
                    continue;
                }
            }
            let lease = match role {
                Buffer::GammaRaw => {
                    init.upload_bytes(&owner.gamma, BufferRole::ImmutableWeights)?
                }
                Buffer::NormalizedBf16 | Buffer::ResidualBf16 => {
                    init.allocate_empty(bytes, BufferRole::Output)?
                }
                _ => init.allocate_empty(bytes, BufferRole::Scratch)?,
            };
            if lease.bytes() != bytes {
                return Err(Error::State(
                    "post-attention RMS original physical extent".into(),
                ));
            }
            leases.push((role, lease));
        }
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        if let Some(pooled) = &pooled {
            pooled.require_storage(leases.iter().map(|(_, lease)| lease))?;
        }
        drop(init.complete_sync().await?);
        let mut stages = Vec::with_capacity(Stage::ORDER.len());
        for (stage, spec) in Stage::ORDER.into_iter().zip(owner.specs.iter()) {
            let handles = Buffer::stage(stage)
                .iter()
                .map(|role| leases.iter().find(|(r, _)| r == role).unwrap().1.handle())
                .collect::<Vec<_>>();
            stages.push(spec.kernel.prepare(service.client(), spec.grid, handles)?);
        }
        let accesses: Vec<_> = leases
            .iter()
            .map(|(role, lease)| {
                (
                    lease.clone(),
                    if role.read_only() {
                        DeviceAccess::Read
                    } else {
                        DeviceAccess::Write
                    },
                )
            })
            .collect();
        Ok(Arc::new(Self {
            service,
            leases: leases.into(),
            stages: stages.into(),
            accesses: accesses.into(),
            owner,
            model_binding: None,
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            pooled,
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            _text_binding: None,
        }))
    }
    fn lease(&self, role: Buffer) -> &ResidentLease {
        &self.leases.iter().find(|(r, _)| *r == role).unwrap().1
    }
    pub(crate) fn require_source_operator(
        &self,
        actual: &zerogravity_core::accelerator::program::SourceOperator,
    ) -> Result<()> {
        let binding = self
            .model_binding
            .as_ref()
            .or(self.owner._model_binding.as_ref())
            .ok_or_else(|| {
                Error::State("ordinary/prefix RMS cannot stand in for a full-model producer".into())
            })?;
        binding.require_source(&self.owner.source)?;
        if binding.operator() != actual {
            return Err(Error::State(
                "RMS actual mathematical call/descriptor differs".into(),
            ));
        }
        if actual.call().rows().len() != self.owner.rows.count()
            || self.service.source_identity() != actual.call().source()
        {
            return Err(Error::State(
                "full-model RMS original source/site/call producer differs".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn require_full_model_operator(
        &self,
        operator: &zerogravity_core::accelerator::program::SourceOperator,
    ) -> Result<()> {
        operator.validate()?;
        let binding = self
            .model_binding
            .as_ref()
            .or(self.owner._model_binding.as_ref())
            .ok_or_else(|| {
                Error::State("ordinary/prefix RMS cannot stand in for a full-model producer".into())
            })?;
        binding.require_source(&self.owner.source)?;
        if binding.operator() != operator
            || operator.call().rows().len() != self.owner.rows.count()
            || self.service.source_identity() != operator.call().source()
        {
            return Err(Error::State(
                "full-model RMS original source/site/call producer differs".into(),
            ));
        }
        Ok(())
    }
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn require_component_values(
        &self,
        values: &Arc<crate::resident_program::qwen2b_component::ComponentValues>,
    ) -> Result<()> {
        self.require_source_operator(values.recipe().operator(7)?)?;
        for (role, id) in [
            (Buffer::Attention, 7),
            (Buffer::PriorResidual, 0),
            (Buffer::NormalizedBf16, 8),
            (Buffer::ResidualBf16, 9),
        ] {
            let actual = self.lease(role);
            let expected = values.lease(zerogravity_core::accelerator::program::ValueId(id))?;
            if actual.id() != expected.id()
                || !Arc::ptr_eq(actual.service(), expected.service())
                || actual.service_generation() != expected.service_generation()
                || actual.bytes() != expected.bytes()
            {
                return Err(Error::State(
                    "postAttention SAME immutable source SSA operand/output map".into(),
                ));
            }
        }
        Ok(())
    }
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn cohort_gamma(self: &Arc<Self>) -> Result<super::cohort::CohortGamma> {
        let binding = self
            .model_binding
            .as_ref()
            .or(self.owner._model_binding.as_ref())
            .ok_or_else(|| Error::State("B4 borrowed genuine model RMS gamma binding".into()))?;
        binding.require_source(&self.owner.source)?;
        super::cohort::CohortGamma::bind(
            binding.operator().clone(),
            self.owner.source.clone(),
            self.lease(Buffer::GammaRaw).clone(),
            self.clone(),
        )
    }
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn immutable_gamma(&self) -> (ResidentLease, &[u8]) {
        (self.lease(Buffer::GammaRaw).clone(), &self.owner.gamma)
    }
    pub(crate) fn normalized(&self) -> &ResidentLease {
        self.lease(Buffer::NormalizedBf16)
    }
    pub(crate) fn residual(&self) -> &ResidentLease {
        self.lease(Buffer::ResidualBf16)
    }
    pub(crate) fn sum_f32(&self) -> &ResidentLease {
        self.lease(Buffer::SumF32)
    }
    pub(crate) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    pub(crate) fn completion_owner(self: &Arc<Self>) -> CompletionOwner {
        self.clone()
    }
    /// Caller retains this owner and all original upstream producers in the
    /// SAME armed flight; merge accesses by ID, with Write dominating Read.
    pub(crate) fn enqueue(&self, cancelled: Option<&AtomicBool>) -> Result<()> {
        not_cancelled(cancelled)?;
        self.service.ensure_usable()?;
        if self.owner.source.source_identity() != self.service.source_identity() {
            return Err(Error::State(
                "post-attention RMS retained source owner".into(),
            ));
        }
        if self.owner._parameter.parameter().tensor().name != self.owner.gamma_name {
            return Err(Error::State("bound RMS source gamma name changed".into()));
        }
        if let Some(binding) = &self.owner._binding {
            binding.require_source(&self.owner.source)?;
        }
        if let Some(binding) = &self.owner._model_binding {
            binding.require_source(&self.owner.source)?;
        }
        if let Some(binding) = &self.model_binding {
            binding.require_source(&self.owner.source)?;
        }
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        if let Some(binding) = &self.owner._final_binding {
            binding.require_source(&self.owner.source)?;
        }
        validate_operands(
            &self.service,
            self.owner.rows,
            self.owner.columns,
            self.lease(Buffer::Attention),
            self.lease(Buffer::PriorResidual),
        )?;
        #[cfg(target_os = "linux")]
        if let Some(source) = &self.owner._leaf {
            source.require_source(&self.owner.source)?;
        }
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        let borrow = self
            .pooled
            .as_ref()
            .map(|pooled| pooled.begin())
            .transpose()?;
        for stage in self.stages.iter() {
            not_cancelled(cancelled)?;
            stage.launch()?;
        }
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        if let Some(borrow) = borrow {
            borrow.complete()?;
        }
        Ok(())
    }
    /// Original gamma, computed FP32 sum, computed BF16 normalized/residual.
    pub(crate) fn readbacks(&self) -> Result<[ResidentReadback; 4]> {
        let roles = [
            Buffer::GammaRaw,
            Buffer::SumF32,
            Buffer::NormalizedBf16,
            Buffer::ResidualBf16,
        ];
        let read = |role| {
            let lease = self.lease(role);
            lease.readback(0, lease.bytes())
        };
        Ok([
            read(roles[0])?,
            read(roles[1])?,
            read(roles[2])?,
            read(roles[3])?,
        ])
    }
    /// All oracle comparison occurs only in the caller's post-arithmetic
    /// completion validator. This verifies source/shape/finite values only.
    pub(crate) fn validate_readback(&self, actual: &[Bytes]) -> Result<()> {
        if actual.len() != 4 {
            return Err(Error::State("dual RMS four readbacks required".into()));
        }
        self.validate_readback_words([
            &actual[0][..],
            &actual[1][..],
            &actual[2][..],
            &actual[3][..],
        ])
    }
    pub(crate) fn validate_readback_words(&self, actual: [&[u8]; 4]) -> Result<()> {
        let dense = self.owner.rows.count() * self.owner.columns.count();
        if actual[0] != self.owner.gamma.as_slice()
            || actual[1].len() != dense * 4
            || actual[2].len() != dense * 2
            || actual[3].len() != dense * 2
            || !actual[1]
                .chunks_exact(4)
                .all(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()).is_finite())
            || !super::resident::finite_bf16(actual[2])
            || !super::resident::finite_bf16(actual[3])
        {
            return Err(Error::State(
                "post-attention RMS original gamma/dual-output readback".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "fused_tests.rs"]
mod tests;

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
#[path = "fused/fixed_prompt_natural.rs"]
mod fixed_prompt_natural;
#[cfg(all(feature = "cuda-native", target_os = "linux"))]
#[path = "fused/natural.rs"]
mod natural;

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
#[path = "fused/pooled.rs"]
mod pooled;
