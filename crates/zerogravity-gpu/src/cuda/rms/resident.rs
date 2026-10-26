//! Preallocated original-source RMS producer for one enclosing physical flight.
use super::{Columns, Rows, Spec, Stage, specification_for};
use crate::cuda::kernel::PinnedKernel;
use crate::physical_quarantine::CompletionOwner;
use crate::resident_leases::{ResidentLease, ResidentReadback, ResidentService};
use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{
    ParameterWindow, TensorWindow, VerifiedMatrix, VerifiedParameter, VerifiedTensorSource,
    WindowWorkspace,
};
use zerogravity_core::accelerator::program::MatrixAxes;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::{Error, Result};

fn require_not_cancelled(cancelled: Option<&std::sync::atomic::AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
        return Err(Error::State("original RMS preparation cancelled".into()));
    }
    Ok(())
}

const VOCABULARY: usize = 248_320;
const METADATA_CAP: usize = 2 * 1024 * 1024 + Stage::ORDER.len() * PinnedKernel::KEY_METADATA_BYTES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Buffer {
    GammaRaw,
    GammaF32,
    GammaShifted,
    Embedding,
    InputF32,
    Squares,
    Mean,
    Epsilon,
    Inverse,
    Normalized,
    Weighted,
    Bf16Output,
}
impl Buffer {
    fn layout(rows: Rows, columns: Columns) -> [(Self, usize); 12] {
        let columns = columns.count();
        let dense = rows.count() * columns;
        [
            (Self::GammaRaw, columns * 2),
            (Self::GammaF32, columns * 4),
            (Self::GammaShifted, columns * 4),
            (Self::Embedding, dense * 2),
            (Self::InputF32, dense * 4),
            (Self::Squares, dense * 4),
            (Self::Mean, rows.count() * 4),
            (Self::Epsilon, rows.count() * 4),
            (Self::Inverse, rows.count() * 4),
            (Self::Normalized, dense * 4),
            (Self::Weighted, dense * 4),
            (Self::Bf16Output, dense * 2),
        ]
    }
    fn stage(stage: Stage) -> &'static [Self] {
        match stage {
            Stage::GammaWiden => &[Self::GammaRaw, Self::GammaF32],
            Stage::GammaShift => &[Self::GammaF32, Self::GammaShifted],
            Stage::InputWiden => &[Self::Embedding, Self::InputF32],
            Stage::Square => &[Self::InputF32, Self::Squares],
            Stage::Mean => &[Self::Squares, Self::Mean],
            Stage::Epsilon => &[Self::Mean, Self::Epsilon],
            Stage::Inverse => &[Self::Epsilon, Self::Inverse],
            Stage::Normalize => &[Self::InputF32, Self::Inverse, Self::Normalized],
            Stage::Weight => &[Self::Normalized, Self::GammaShifted, Self::Weighted],
            Stage::Bf16Store => &[Self::Weighted, Self::Bf16Output],
        }
    }
    fn read_only(self) -> bool {
        matches!(self, Self::GammaRaw | Self::Embedding)
    }
}

struct SourceOwner {
    source: VerifiedTensorSource,
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    _bindings: Option<Arc<crate::resident_program::source_bindings::SourceBindings>>,
    _parameter: ParameterWindow,
    _rows: Vec<TensorWindow>,
    _teacher_ids: Vec<u32>,
    rows: Rows,
    columns: Columns,
    #[cfg(target_os = "linux")]
    _leaf: Option<super::leaf_source::RmsLeafSource>,
    gamma: Vec<u8>,
    embedding: Vec<u8>,
    specs: Arc<[Spec]>,
    // Drop payloads before returning their host admission.
    _workspace: WindowWorkspace,
    _metadata: WindowWorkspace,
}

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
fn source_embedding(
    bindings: &crate::resident_program::source_bindings::SourceBindings,
) -> Result<VerifiedMatrix> {
    use zerogravity_core::accelerator::program::{SourceNumericalContract, SourceOperatorKind};
    let descriptor = bindings
        .operator(0)?
        .ok_or_else(|| Error::State("RMS actual embedding operator absent".into()))?;
    let [member] = descriptor.matrices() else {
        return Err(Error::State("RMS actual embedding member arity".into()));
    };
    if descriptor.kind() != SourceOperatorKind::EmbeddingRows
        || descriptor.call() != bindings.call()
        || descriptor.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1
    {
        return Err(Error::State(
            "RMS actual embedding source/call/contract differs".into(),
        ));
    }
    let matrix = bindings.source().matrix(&member.name, member.axes)?;
    if member.source != matrix.source_identity()
        || member.kind != matrix.tensor().kind
        || member.rows != matrix.rows()
        || member.columns != matrix.columns()
        || member.axes != matrix.axes()
    {
        return Err(Error::State(
            "RMS actual embedding source header differs".into(),
        ));
    }
    Ok(matrix)
}
pub(super) fn finite_bf16(bytes: &[u8]) -> bool {
    bytes.len().is_multiple_of(2)
        && bytes
            .chunks_exact(2)
            .all(|raw| u16::from_le_bytes([raw[0], raw[1]]) & 0x7f80 != 0x7f80)
}

pub(super) fn loaded_gamma(window: &ParameterWindow) -> Result<Vec<u8>> {
    loaded_gamma_for(window, Columns::Qwen08)
}

pub(super) fn loaded_gamma_for(window: &ParameterWindow, columns: Columns) -> Result<Vec<u8>> {
    let columns = columns.count();
    if !window.is_complete() || window.parameter().elements() != columns {
        return Err(Error::Malformed(format!(
            "complete original ordinary gamma{columns}"
        )));
    }
    let bytes = match window.parameter().tensor().kind {
        QuantKind::Bf16 => window.as_slice().to_vec(),
        QuantKind::F32 => {
            let mut output = Vec::with_capacity(columns * 2);
            for raw in window.as_slice().chunks_exact(4) {
                let bits = u32::from_le_bytes(raw.try_into().unwrap());
                if !f32::from_bits(bits).is_finite() {
                    return Err(Error::Malformed("nonfinite original ordinary gamma".into()));
                }
                let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
                output.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
            }
            output
        }
        _ => {
            return Err(Error::Unsupported(
                "original ordinary gamma BF16/F32 source".into(),
            ));
        }
    };
    if bytes.len() != columns * 2 || !finite_bf16(&bytes) {
        return Err(Error::Malformed("finite loaded ordinary BF16 gamma".into()));
    }
    Ok(bytes)
}

fn source_owner(
    source: VerifiedTensorSource,
    ids: &[u32],
    columns: Columns,
) -> Result<Arc<SourceOwner>> {
    let rows = Rows::new(ids.len())?;
    source_owner_rows(source, ids, columns, rows)
}

fn source_owner_rows(
    source: VerifiedTensorSource,
    ids: &[u32],
    columns: Columns,
    rows: Rows,
) -> Result<Arc<SourceOwner>> {
    if ids.len() != rows.count() || (rows == Rows::FixedPrompt && columns != Columns::Qwen08) {
        return Err(Error::Unsupported("closed source RMS rows/columns".into()));
    }
    source_owner_rows_storage(source, ids, columns, rows, None)
}
pub(super) fn pooled_metadata_bytes() -> Result<usize> {
    // Closed layout:12 leases/accesses,10 outputs/stages, at most4 handles per
    // stage;72 source rows/IDs. Payload bytes have their own workspace owner.
    let parts = [
        std::mem::size_of::<SourceOwner>(),
        std::mem::size_of::<RmsPrepared>(),
        12 * std::mem::size_of::<(Buffer, ResidentLease)>(),
        12 * std::mem::size_of::<(ResidentLease, DeviceAccess)>(),
        10 * std::mem::size_of::<ResidentLease>(),
        10 * std::mem::size_of::<crate::cuda::kernel::PreparedPinnedLaunch>(),
        40 * std::mem::size_of::<Handle>(),
        72 * (std::mem::size_of::<TensorWindow>() + std::mem::size_of::<u32>()),
        32 * (2 * std::mem::size_of::<usize>() + std::mem::align_of::<usize>()),
    ];
    parts
        .into_iter()
        .try_fold(0usize, |n, p| n.checked_add(p))
        .ok_or_else(|| Error::Limit("pooled ordinary RMS source/binding metadata extent".into()))
}
fn pooled_source_host_bytes(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_add(1)
        .and_then(|n| n.checked_mul(columns))
        .and_then(|n| n.checked_mul(6))
        .and_then(|n| n.checked_add(64 * 1024))
        .ok_or_else(|| Error::Limit("pooled RMS actual source payload capacity".into()))
}
fn source_owner_rows_storage(
    source: VerifiedTensorSource,
    ids: &[u32],
    columns: Columns,
    rows: Rows,
    definitions: Option<Arc<super::definitions::RmsDefinitions>>,
) -> Result<Arc<SourceOwner>> {
    let parameter = source.parameter("model.language_model.layers.0.input_layernorm.weight")?;
    let matrix = source.matrix(
        "model.language_model.embed_tokens.weight",
        MatrixAxes::OutputInput,
    )?;
    source_owner_rows_storage_bound(source, ids, columns, rows, definitions, matrix, parameter)
}
fn source_owner_rows_storage_bound(
    source: VerifiedTensorSource,
    ids: &[u32],
    columns: Columns,
    rows: Rows,
    definitions: Option<Arc<super::definitions::RmsDefinitions>>,
    matrix: VerifiedMatrix,
    parameter: VerifiedParameter,
) -> Result<Arc<SourceOwner>> {
    let pooled = definitions.is_some();
    if ids.len() != rows.count() {
        return Err(Error::State("RMS actual source rows".into()));
    }
    let count = columns.count();
    if ids.iter().any(|id| *id as usize >= VOCABULARY) {
        return Err(Error::Malformed(
            "original source teacher vocabulary".into(),
        ));
    }
    let metadata = source.reserve_workspace(if pooled {
        pooled_metadata_bytes()?
    } else {
        METADATA_CAP
    })?;
    let physical = Buffer::layout(rows, columns)
        .iter()
        .map(|(_, bytes)| bytes)
        .sum::<usize>();
    // Original shared diagnostic admission owns D2H; pooled scratch is empty
    // device storage. Only actual gamma/embedding payloads are retained here.
    let host = if pooled {
        pooled_source_host_bytes(rows.count(), count)?
    } else {
        physical * 3 + (ids.len() * count * 2 + count * 2) * 3 + 64 * 1024
    };
    let workspace = source.reserve_workspace(host)?;
    source.require_matrix_binding(&matrix)?;
    let parameter = source.read_parameter(&parameter, 0..count)?;
    if matrix.rows() != VOCABULARY
        || matrix.columns() != count
        || matrix.tensor().kind != QuantKind::Bf16
        || parameter.parameter().source_identity() != source.source_identity()
    {
        return Err(Error::Unsupported(
            "original .8B embedding/gamma source geometry".into(),
        ));
    }
    let mut windows = Vec::with_capacity(ids.len());
    let mut embedding = Vec::with_capacity(ids.len() * count * 2);
    for &id in ids {
        let window = source.read_rows(&matrix, id as usize..id as usize + 1)?;
        if window.as_slice().len() != count * 2 || !finite_bf16(window.as_slice()) {
            return Err(Error::Malformed(
                "finite complete original embedding row".into(),
            ));
        }
        embedding.extend_from_slice(window.as_slice());
        windows.push(window);
    }
    let gamma = if columns == Columns::Qwen08 {
        loaded_gamma(&parameter)?
    } else {
        loaded_gamma_for(&parameter, columns)?
    };
    let specs = if let Some(definitions) = definitions {
        definitions.specs(rows, columns, false)?
    } else {
        Stage::ORDER
            .into_iter()
            .map(|stage| specification_for(stage, rows, columns))
            .collect::<Result<Vec<_>>>()?
            .into()
    };
    Ok(Arc::new(SourceOwner {
        source,
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        _bindings: None,
        _parameter: parameter,
        _rows: windows,
        _teacher_ids: ids.to_vec(),
        rows,
        columns,
        #[cfg(target_os = "linux")]
        _leaf: None,
        gamma,
        embedding,
        specs,
        _workspace: workspace,
        _metadata: metadata,
    }))
}

/// All storage is private and source-owned; this is not a profile/model permit.
pub(crate) struct RmsPrepared {
    // Drop cloned physical handles before their lease/admission owners.
    stages: Arc<[crate::cuda::kernel::PreparedPinnedLaunch]>,
    service: Arc<ResidentService>,
    leases: Arc<[(Buffer, ResidentLease)]>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    outputs: Arc<[ResidentLease]>,
    owner: Arc<SourceOwner>,
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pooled: Option<Arc<super::pooled::PooledRmsActor>>,
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    natural: Option<Arc<natural::NaturalRows>>,
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    text: Option<Arc<text::EmbeddingRows>>,
}
impl RmsPrepared {
    /// Actual model-prepared embedding/RMS operands; no role names or reissuer.
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) async fn prepare_bound(
        service: Arc<ResidentService>,
        bindings: Arc<crate::resident_program::source_bindings::SourceBindings>,
        correlation: u64,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Arc<Self>> {
        let rows = Rows::new(bindings.call().rows().len())?;
        Self::prepare_rows_bound(service, bindings, rows, correlation, cancelled).await
    }

    /// Explicit physical row storage after the caller binds its capture recipe.
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) async fn prepare_rows_bound(
        service: Arc<ResidentService>,
        bindings: Arc<crate::resident_program::source_bindings::SourceBindings>,
        rows: Rows,
        correlation: u64,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Arc<Self>> {
        use zerogravity_core::accelerator::program::{PlanOperation, SourceOperatorKind, ValueId};
        require_not_cancelled(cancelled)?;
        let plan = bindings.plan();
        let (Some(embedding), Some(rms)) = (plan.nodes().first(), plan.nodes().get(1)) else {
            return Err(Error::State("ordinary source producer nodes absent".into()));
        };
        let (
            PlanOperation::SourceOperator {
                descriptor: ed,
                state: None,
            },
            PlanOperation::SourceOperator {
                descriptor: rd,
                state: None,
            },
        ) = (&embedding.operation, &rms.operation)
        else {
            return Err(Error::State(
                "ordinary source mathematical producers absent".into(),
            ));
        };
        let binding = super::BoundModelRmsSource::bind(bindings.source().clone(), (**rd).clone())?;
        let [matrix] = ed.matrices() else {
            return Err(Error::State(
                "ordinary source embedding member arity".into(),
            ));
        };
        if ed.kind() != SourceOperatorKind::EmbeddingRows
            || !matches!(rd.kind(), SourceOperatorKind::Rms(_))
            || ed.call() != bindings.call()
            || rd.call() != bindings.call()
            || embedding.outputs != [ValueId(0)]
            || !embedding.inputs.is_empty()
            || rms.inputs != [ValueId(0)]
            || rms.outputs != [ValueId(1)]
            || service.source_identity() != bindings.source().source_identity()
            || rows.count() != bindings.call().rows().len()
        {
            return Err(Error::State(
                "ordinary source actual embedding/RMS/SSA/call differs".into(),
            ));
        }
        let actual = bindings.source().matrix(&matrix.name, matrix.axes)?;
        let parameter = bindings.source().parameter(&binding.parameter().name)?;
        parameter.validate_source_parameter(binding.parameter())?;
        if matrix.source != actual.source_identity()
            || matrix.rows != actual.rows()
            || matrix.columns != actual.columns()
            || matrix.kind != actual.tensor().kind
            || matrix.axes != actual.axes()
        {
            return Err(Error::State(
                "ordinary source embedding header differs".into(),
            ));
        }
        let columns = match binding.geometry().width {
            1024 => Columns::Qwen08,
            2048 => Columns::Qwen2b,
            _ => return Err(Error::Unsupported("ordinary source RMS width".into())),
        };
        let mut owner = source_owner_rows_storage_bound(
            bindings.source().clone(),
            bindings.call().rows().ids(),
            columns,
            rows,
            None,
            actual,
            parameter,
        )?;
        Arc::get_mut(&mut owner)
            .ok_or_else(|| Error::State("fresh bound RMS source owner".into()))?
            ._bindings = Some(bindings);
        Self::prepare_owner(service, owner, correlation, cancelled).await
    }
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) fn cohort_gamma(
        self: &Arc<Self>,
        binding: &super::BoundModelRmsSource,
    ) -> Result<super::cohort::CohortGamma> {
        binding.require_source(&self.owner.source)?;
        if !matches!(
            binding.operator().kind(),
            zerogravity_core::accelerator::program::SourceOperatorKind::Rms(_)
        ) || binding.parameter() != &self.owner._parameter.parameter().source_parameter()
        {
            return Err(Error::State(
                "B4 actual ordinary source gamma role/header".into(),
            ));
        }
        super::cohort::CohortGamma::bind(
            binding.operator().clone(),
            self.owner.source.clone(),
            self.lease(Buffer::GammaRaw).clone(),
            self.clone(),
        )
    }
    /// Allocate/upload before the caller activates its native provider pool.
    /// No reference embedding/normalization input or alternate source issuer.
    /// Caller already holds its one original preparation fair slot.
    pub(crate) async fn prepare(
        service: Arc<ResidentService>,
        source: VerifiedTensorSource,
        teacher_ids: &[u32],
        correlation: u64,
    ) -> Result<Arc<Self>> {
        if source.source_identity() != service.source_identity() {
            return Err(Error::State("RMS producer source/service identity".into()));
        }
        let owner = source_owner(source, teacher_ids, Columns::Qwen08)?;
        Self::prepare_owner(service, owner, correlation, None).await
    }

    #[cfg(target_os = "linux")]
    pub(super) async fn prepare_2b(
        service: Arc<ResidentService>,
        source: super::leaf_source::RmsLeafSource,
        teacher_ids: &[u32],
        correlation: u64,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::qwen2b::not_cancelled(cancelled)?;
        source.require_source(source.source())?;
        if source.source().source_identity() != service.source_identity() {
            return Err(Error::State("2B RMS source/service issuer".into()));
        }
        if source.columns() != Columns::Qwen2b {
            return Err(Error::State("wide RMS leaf actual width differs".into()));
        }
        let (matrix, parameter) = source.ordinary_members()?;
        let mut owner = source_owner_rows_storage_bound(
            source.source().clone(),
            teacher_ids,
            Columns::Qwen2b,
            Rows::new(teacher_ids.len())?,
            None,
            matrix,
            parameter,
        )?;
        Arc::get_mut(&mut owner)
            .ok_or_else(|| Error::State("fresh2B RMS source owner".into()))?
            ._leaf = Some(source);
        Self::prepare_owner(service, owner, correlation, cancelled).await
    }

    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(super) async fn prepare_fixed_prompt(
        service: Arc<ResidentService>,
        source: super::leaf_source::RmsLeafSource,
        teacher_ids: &[u32; 47],
        correlation: u64,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::fixed_prompt::not_cancelled(cancelled)?;
        source.require_source(source.source())?;
        if source.source().source_identity() != service.source_identity() {
            return Err(Error::State(
                "fixed_prompt RMS source/service issuer".into(),
            ));
        }
        if source.columns() != Columns::Qwen08 {
            return Err(Error::State("fixed RMS leaf actual width differs".into()));
        }
        let (matrix, parameter) = source.ordinary_members()?;
        let mut owner = source_owner_rows_storage_bound(
            source.source().clone(),
            teacher_ids,
            Columns::Qwen08,
            Rows::FixedPrompt,
            None,
            matrix,
            parameter,
        )?;
        Arc::get_mut(&mut owner)
            .ok_or_else(|| Error::State("fresh fixed_prompt RMS source owner".into()))?
            ._leaf = Some(source);
        Self::prepare_owner(service, owner, correlation, cancelled).await
    }

    /// Same original source-row/RMS actor; explicit full-record canonical scope.
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(crate) async fn prepare_fixed_prompt_model(
        service: Arc<ResidentService>,
        binding: Arc<crate::cuda::native::fixed_prompt_model::FixedPromptModelBinding>,
        correlation: u64,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Arc<Self>> {
        let rows = match binding.kind() {
            crate::cuda::native::fixed_prompt_model::FixedPromptCallKind::Prefill => {
                Rows::FixedPrompt
            }
            crate::cuda::native::fixed_prompt_model::FixedPromptCallKind::Decode { .. } => {
                Rows::One
            }
        };
        Self::prepare_rows_bound(
            service,
            binding.record().clone(),
            rows,
            correlation,
            cancelled,
        )
        .await
    }

    async fn prepare_owner(
        service: Arc<ResidentService>,
        owner: Arc<SourceOwner>,
        correlation: u64,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Arc<Self>> {
        Self::prepare_owner_storage(
            service,
            owner,
            correlation,
            cancelled,
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            None,
        )
        .await
    }

    async fn prepare_owner_storage(
        service: Arc<ResidentService>,
        owner: Arc<SourceOwner>,
        correlation: u64,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
        #[cfg(all(feature = "cuda-native", target_os = "linux"))] pooled: Option<
            Arc<super::pooled::PooledRmsActor>,
        >,
    ) -> Result<Arc<Self>> {
        require_not_cancelled(cancelled)?;
        let layout = Buffer::layout(owner.rows, owner.columns);
        let physical_bytes = layout
            .iter()
            .filter(|(role, _)| {
                #[cfg(all(feature = "cuda-native", target_os = "linux"))]
                if pooled.as_ref().is_some_and(|p| {
                    Self::scratch_role(*role).is_some()
                        || p.has_workspace_outputs()
                            && matches!(role, Buffer::Embedding | Buffer::Bf16Output)
                }) {
                    return false;
                }
                let _ = role;
                true
            })
            .map(|(_, bytes)| bytes)
            .sum::<usize>();
        if service.remaining_capacity()? < physical_bytes {
            return Err(Error::Limit(
                "RMS producer capacity before allocation".into(),
            ));
        }
        let mut init = service.begin_initialization(correlation)?;
        init.retain_completion_owner(owner.clone())?;
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        if let Some(pooled) = &pooled {
            init.retain_completion_owner(pooled.clone())?;
        }
        let mut leases = Vec::with_capacity(layout.len());
        for (role, bytes) in layout {
            require_not_cancelled(cancelled)?;
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            if let Some(shared) = pooled.as_ref().zip(Self::scratch_role(role)) {
                leases.push((role, shared.0.lease(shared.1, bytes)?));
                continue;
            }
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            if let Some(pooled) = &pooled {
                let output = match role {
                    Buffer::Embedding => Some(super::pooled::RmsWorkspaceOutput::Embedding),
                    Buffer::Bf16Output => Some(super::pooled::RmsWorkspaceOutput::Normalized),
                    _ => None,
                };
                if let Some(output_role) = output
                    && let Some(lease) = pooled.workspace_output(output_role, bytes)?
                {
                    // Shared cold embedding is empty: genuine current node0
                    // fully writes it before RMS. No cold initialization grant.
                    leases.push((role, lease));
                    continue;
                }
            }
            let lease = match role {
                Buffer::GammaRaw => {
                    init.upload_bytes(&owner.gamma, BufferRole::ImmutableWeights)?
                }
                Buffer::Embedding => init.upload_bytes(&owner.embedding, BufferRole::Activation)?,
                Buffer::Bf16Output => init.allocate_empty(bytes, BufferRole::Output)?,
                _ => init.allocate_empty(bytes, BufferRole::Scratch)?,
            };
            if lease.bytes() != bytes {
                return Err(Error::State("RMS producer original physical extent".into()));
            }
            leases.push((role, lease));
        }
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        if let Some(pooled) = &pooled {
            pooled.require_storage(leases.iter().map(|(_, lease)| lease))?;
        }
        let initialized = init.complete_sync().await?;
        drop(initialized);
        let mut stages = Vec::with_capacity(Stage::ORDER.len());
        for (stage, spec) in Stage::ORDER.into_iter().zip(owner.specs.iter()) {
            let handles = Buffer::stage(stage)
                .iter()
                .map(|role| leases.iter().find(|(r, _)| r == role).unwrap().1.handle())
                .collect::<Vec<_>>();
            stages.push(spec.kernel.prepare(service.client(), spec.grid, handles)?);
        }
        let accesses = leases
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
        let outputs: Vec<_> = leases
            .iter()
            .filter(|(role, _)| !role.read_only())
            .map(|(_, lease)| lease.clone())
            .collect();
        Ok(Arc::new(Self {
            service,
            leases: leases.into(),
            stages: stages.into(),
            accesses,
            outputs: outputs.into(),
            owner,
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            pooled,
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            natural: None,
            #[cfg(all(feature = "cuda-native", target_os = "linux"))]
            text: None,
        }))
    }

    fn lease(&self, role: Buffer) -> &ResidentLease {
        &self.leases.iter().find(|(r, _)| *r == role).unwrap().1
    }
    pub(crate) fn normalized(&self) -> &ResidentLease {
        self.lease(Buffer::Bf16Output)
    }
    pub(crate) fn embedding(&self) -> &ResidentLease {
        self.lease(Buffer::Embedding)
    }
    pub(crate) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    pub(crate) fn outputs(&self) -> &[ResidentLease] {
        &self.outputs
    }
    /// Attach to the enclosing original flight before the first enqueue.
    pub(crate) fn completion_owner(self: &Arc<Self>) -> CompletionOwner {
        self.clone()
    }
    /// Caller already owns the original fair slot and retained physical flight.
    /// No device allocation/upload, stream override, sync, Ready or publication here.
    pub(crate) fn enqueue(&self) -> Result<()> {
        self.enqueue_stages(Stage::ORDER.len())
    }
    #[cfg(target_os = "linux")]
    pub(super) fn enqueue_2b_through_mean(&self) -> Result<usize> {
        if self.owner.columns != Columns::Qwen2b || self.owner._leaf.is_none() {
            return Err(Error::State(
                "2B RMS fault cannot retag ordinary source".into(),
            ));
        }
        let stages = Stage::ORDER
            .iter()
            .position(|stage| *stage == Stage::Mean)
            .ok_or_else(|| Error::State("original RMS Mean stage absent".into()))?
            + 1;
        self.enqueue_stages(stages)?;
        Ok(stages)
    }
    #[cfg(all(feature = "cuda-native", target_os = "linux"))]
    pub(super) fn enqueue_fixed_prompt_through_mean(&self) -> Result<usize> {
        if self.owner.columns != Columns::Qwen08
            || self.owner.rows != Rows::FixedPrompt
            || self.owner._leaf.is_none()
        {
            return Err(Error::State(
                "fixed_prompt RMS cut requires its sealed original source".into(),
            ));
        }
        let stages = Stage::ORDER
            .iter()
            .position(|stage| *stage == Stage::Mean)
            .ok_or_else(|| Error::State("original RMS Mean stage absent".into()))?
            + 1;
        self.enqueue_stages(stages)?;
        Ok(stages)
    }
    fn enqueue_stages(&self, stages: usize) -> Result<()> {
        self.service.ensure_usable()?;
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        let borrow = match &self.pooled {
            Some(pooled) if stages == Stage::ORDER.len() => Some(pooled.begin()?),
            Some(_) => {
                return Err(Error::Unsupported(
                    "pooled RMS requires the complete original actor".into(),
                ));
            }
            None => None,
        };
        if self.owner.source.source_identity() != self.service.source_identity() {
            return Err(Error::State("RMS producer original source owner".into()));
        }
        #[cfg(target_os = "linux")]
        if let Some(source) = &self.owner._leaf {
            source.require_source(&self.owner.source)?;
        }
        for stage in self.stages.iter().take(stages) {
            stage.launch()?;
        }
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        if let Some(borrow) = borrow {
            borrow.complete()?;
        }
        Ok(())
    }
    /// Construct before provider activation; no device transfer occurs here.
    pub(crate) fn readbacks(&self) -> Result<[ResidentReadback; 3]> {
        let gamma = self.lease(Buffer::GammaRaw);
        let embedding = self.lease(Buffer::Embedding);
        let normalized = self.normalized();
        Ok([
            gamma.readback(0, gamma.bytes())?,
            embedding.readback(0, embedding.bytes())?,
            normalized.readback(0, normalized.bytes())?,
        ])
    }
    /// Numerical reference comparison belongs to the caller's post-arithmetic
    /// completion validator; this only checks original inputs and output shape.
    pub(crate) fn validate_readback<'a>(&self, actual: &'a [Bytes]) -> Result<&'a [u8]> {
        if actual.len() != 3
            || &actual[0][..] != self.owner.gamma.as_slice()
            || actual[2].len() != self.owner.rows.count() * self.owner.columns.count() * 2
            || !finite_bf16(actual[2].as_ref())
        {
            return Err(Error::State(
                "RMS producer original inputs/output readback".into(),
            ));
        }
        #[cfg(all(feature = "cuda-native", target_os = "linux"))]
        if let Some(rows) = &self.text {
            rows.validate(&actual[1])?;
        } else if self.natural.is_some() {
            self.validate_natural_embedding(&actual[1])?;
        } else if &actual[1][..] != self.owner.embedding.as_slice() {
            return Err(Error::State(
                "RMS producer original inputs/output readback".into(),
            ));
        }
        #[cfg(not(all(feature = "cuda-native", target_os = "linux")))]
        if &actual[1][..] != self.owner.embedding.as_slice() {
            return Err(Error::State(
                "RMS producer original inputs/output readback".into(),
            ));
        }
        Ok(actual[2].as_ref())
    }
}

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
#[path = "resident/natural.rs"]
mod natural;

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
#[path = "resident/pooled.rs"]
mod pooled;

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
#[path = "resident/text.rs"]
mod text;
