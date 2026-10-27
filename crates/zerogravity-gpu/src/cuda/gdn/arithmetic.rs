//! Exact owned implementation metadata. Equality never issues an execution permit.
use super::{BindingSpec, GdnPhase, GdnStage, assets};
use std::{fmt, sync::Arc};
use zerogravity_core::source::{
    exact_text_eq as source_text_eq, exact_text_slice_eq as source_slice_eq,
};
use zerogravity_core::{Error, Result};

#[derive(Clone, Copy)]
struct CompilationSourceContract {
    // Source of the precompiled-source, header, NVRTC-option and load path.
    // External SDK/driver/compiler binaries still need independent physical proof.
    kernel_source: &'static str,
    context_source: &'static str,
    shared_header_source: &'static str,
    cuda_compiler_source: &'static str,
    cuda_server_source: &'static str,
    metadata_binding_info_source: &'static str,
    metadata_binding_info_version: &'static str,
    language: &'static str,
    hardware: &'static str,
    address_bits: u8,
    unchecked: bool,
    float_semantics: FloatSemantics,
}

impl PartialEq for CompilationSourceContract {
    fn eq(&self, other: &Self) -> bool {
        source_text_eq(self.kernel_source, other.kernel_source)
            && source_text_eq(self.context_source, other.context_source)
            && source_text_eq(self.shared_header_source, other.shared_header_source)
            && source_text_eq(self.cuda_compiler_source, other.cuda_compiler_source)
            && source_text_eq(self.cuda_server_source, other.cuda_server_source)
            && source_text_eq(
                self.metadata_binding_info_source,
                other.metadata_binding_info_source,
            )
            && source_text_eq(
                self.metadata_binding_info_version,
                other.metadata_binding_info_version,
            )
            && source_text_eq(self.language, other.language)
            && source_text_eq(self.hardware, other.hardware)
            && self.address_bits == other.address_bits
            && self.unchecked == other.unchecked
            && self.float_semantics == other.float_semantics
    }
}
impl Eq for CompilationSourceContract {}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FloatSemantics {
    // Rounding, FMA, approximate and noftz instructions are in the full body.
    OriginalInlinePtxModifiers,
}

// Exact source unit used by cubecl-runtime 0.11.0 to define the trailing
// metadata packet. The registry checksum below pins the complete crate source.
const METADATA_BINDING_INFO_SOURCE: &str = r#"#[derive(new, Debug, Default)]
pub struct MetadataBindingInfo {
    /// Scalar and metadata values
    pub data: Vec<u64>,
    /// Start of the dynamically sized portion of the metadata, relative to the entire info buffer
    pub dynamic_metadata_offset: usize,
}

impl MetadataBindingInfo {
    /// Create a new binding info for custom data, for externally compiled kernels.
    pub fn custom(data: Vec<u64>) -> Self {
        Self::new(data, 0)
    }
}
"#;
const METADATA_BINDING_INFO_VERSION: &str = concat!(
    "cubecl-runtime 0.11.0\0",
    "registry+https://github.com/rust-lang/crates.io-index\0",
    "sha256:a1c2bdd3bfe667906cd55cc8f0a956a8476124263b6838cc5d7d68d9edacfea0",
);

const COMPILATION_SOURCE: CompilationSourceContract = CompilationSourceContract {
    kernel_source: include_str!("../kernel.rs"),
    // NUL-delimited labels keep full compiler units distinct without changing
    // the existing exact-text contract/layout. Rust source units contain no NUL.
    context_source: concat!(
        "\0context.rs\0",
        include_str!("../../../vendor/cubecl-cuda/src/compute/context.rs"),
        "\0context/nvrtc.rs\0",
        include_str!("../../../vendor/cubecl-cuda/src/compute/context/nvrtc.rs"),
        "\0context/compiler_context.rs\0",
        include_str!("../../../vendor/cubecl-cuda/src/compute/context/compiler_context.rs"),
        "\0context/compiler_limits.rs\0",
        include_str!("../../../vendor/cubecl-cuda/src/compute/context/compiler_limits.rs"),
        "\0owned_command/compiler.rs\0",
        include_str!("../../../vendor/cubecl-cuda/src/owned_command/compiler.rs"),
        "\0lib.rs\0",
        include_str!("../../../vendor/cubecl-cuda/src/lib.rs"),
        "\0context/program_lifetime.rs\0",
        include_str!("../../../vendor/cubecl-cuda/src/compute/context/program_lifetime.rs"),
    ),
    shared_header_source: include_str!(
        "../../../vendor/cubecl-cuda/src/compute/context/precompiled_shared.rs"
    ),
    cuda_compiler_source: include_str!("../../../vendor/cubecl-cuda/src/compiler.rs"),
    cuda_server_source: include_str!("../../../vendor/cubecl-cuda/src/compute/server.rs"),
    metadata_binding_info_source: METADATA_BINDING_INFO_SOURCE,
    metadata_binding_info_version: METADATA_BINDING_INFO_VERSION,
    language: "cuda",
    hardware: "ptx_sm86",
    address_bits: 64,
    unchecked: true,
    float_semantics: FloatSemantics::OriginalInlinePtxModifiers,
};

// Exact selected executable sources participate in family equality.
// No qualification recipe, request, root, allocator or expected tensor here.
#[derive(Clone, Copy)]
pub(crate) struct GdnFamilyIdentity {
    text_implementation: bool,
    text_config: Option<super::GdnKernelConfig>,
    selected_sources: Option<[&'static str; 4]>,
    compile_source: CompilationSourceContract,
    selectors: &'static [&'static str],
    schedule_source: &'static str,
    bindings_source: &'static str,
    catalogue: &'static [&'static str],
}

impl PartialEq for GdnFamilyIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.text_implementation == other.text_implementation
            && self.text_config == other.text_config
            && match (self.selected_sources, other.selected_sources) {
                (Some(left), Some(right)) => source_slice_eq(&left, &right),
                (None, None) => true,
                _ => false,
            }
            && self.compile_source == other.compile_source
            && source_slice_eq(self.selectors, other.selectors)
            && source_text_eq(self.schedule_source, other.schedule_source)
            && source_text_eq(self.bindings_source, other.bindings_source)
            && source_slice_eq(self.catalogue, other.catalogue)
    }
}
impl Eq for GdnFamilyIdentity {}

impl fmt::Debug for GdnFamilyIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GdnFamilyIdentity")
            .field("hardware", &self.compile_source.hardware)
            .field("text_config", &self.text_config)
            .field(
                "source_units",
                &(self.catalogue.len() + self.selected_sources.map_or(0, |sources| sources.len())),
            )
            .finish_non_exhaustive()
    }
}
impl GdnFamilyIdentity {
    #[cfg(any(test, target_os = "linux"))]
    pub(crate) fn original2b() -> Self {
        Self {
            text_implementation: false,
            text_config: None,
            selected_sources: None,
            compile_source: COMPILATION_SOURCE,
            selectors: &[include_str!("qwen2b_assets.rs")],
            schedule_source: include_str!("resident.rs"),
            bindings_source: include_str!("../gdn.rs"),
            catalogue: ORIGINAL2B_CATALOGUE,
        }
    }
    pub(crate) fn original08() -> Self {
        Self {
            text_implementation: false,
            text_config: None,
            selected_sources: None,
            compile_source: COMPILATION_SOURCE,
            selectors: &[
                include_str!("assets.rs"),
                include_str!("fixed_prompt_assets.rs"),
            ],
            schedule_source: include_str!("resident.rs"),
            bindings_source: include_str!("../gdn.rs"),
            catalogue: ORIGINAL08_CATALOGUE,
        }
    }
    #[cfg(target_os = "linux")]
    pub(crate) fn text(config: Option<super::GdnKernelConfig>) -> Self {
        let (text_config, selected_sources, catalogue) = super::text::text_source_contract(config);
        Self {
            text_implementation: true,
            text_config,
            selected_sources,
            compile_source: COMPILATION_SOURCE,
            selectors: &[
                include_str!("text/config.rs"),
                include_str!("text/parameterized_conv.rs"),
                include_str!("text/parameterized_prefill.rs"),
                include_str!("text/parameterized_prefill_variants.rs"),
                include_str!("text/parameterized_cached.rs"),
                include_str!("text/parameterized_gated.rs"),
            ],
            schedule_source: concat!(
                include_str!("text.rs"),
                "\0workspace.rs\0",
                include_str!("text/workspace.rs"),
                "\0resident.rs\0",
                include_str!("resident.rs"),
            ),
            bindings_source: include_str!("../gdn.rs"),
            catalogue,
        }
    }
    pub(crate) fn require_same(&self, other: &Self) -> Result<()> {
        if self != other {
            return Err(Error::Unsupported(
                "different exact GDN owned code family".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) enum GdnArithmeticFamily {
    Original08,
    #[cfg(any(test, target_os = "linux"))]
    Original2b,
    #[cfg(target_os = "linux")]
    Text(Option<super::GdnKernelConfig>),
}

#[derive(Clone)]
pub(crate) struct StageArithmetic {
    family: GdnFamilyIdentity,
    phase: GdnPhase,
    stage: GdnStage,
    entry: &'static str,
    owned_source: &'static str,
    original_ptx_sha256: &'static str,
    threads: u32,
    shared: u32,
    grid: [u32; 3],
    bindings: Arc<[BindingSpec]>,
}

impl PartialEq for StageArithmetic {
    fn eq(&self, other: &Self) -> bool {
        self.family == other.family
            && self.phase == other.phase
            && self.stage == other.stage
            && source_text_eq(self.entry, other.entry)
            && source_text_eq(self.owned_source, other.owned_source)
            && source_text_eq(self.original_ptx_sha256, other.original_ptx_sha256)
            && self.threads == other.threads
            && self.shared == other.shared
            && self.grid == other.grid
            && self.bindings == other.bindings
    }
}
impl Eq for StageArithmetic {}

impl fmt::Debug for StageArithmetic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageArithmetic")
            .field("phase", &self.phase)
            .field("stage", &self.stage)
            .field("entry", &self.entry)
            .field("original_ptx_sha256", &self.original_ptx_sha256)
            .field("source_bytes", &self.owned_source.len())
            .field("grid", &self.grid)
            .finish_non_exhaustive()
    }
}
impl StageArithmetic {
    pub(super) fn for_family(
        family: GdnArithmeticFamily,
        phase: GdnPhase,
        stage: GdnStage,
        asset: &assets::Asset,
        bindings: &[BindingSpec],
    ) -> Result<Self> {
        match family {
            GdnArithmeticFamily::Original08 => Self::original08(phase, stage, asset, bindings),
            #[cfg(any(test, target_os = "linux"))]
            GdnArithmeticFamily::Original2b => Self::original2b(phase, stage, asset, bindings),
            #[cfg(target_os = "linux")]
            GdnArithmeticFamily::Text(config) => Self::checked(
                phase,
                stage,
                asset,
                bindings,
                &super::text::select(config, phase, stage)?,
                GdnFamilyIdentity::text(config),
            ),
        }
    }
    // Only the existing closed factory calls this; no caller-supplied code loader.
    pub(super) fn original08(
        phase: GdnPhase,
        stage: GdnStage,
        asset: &assets::Asset,
        bindings: &[BindingSpec],
    ) -> Result<Self> {
        Self::checked(
            phase,
            stage,
            asset,
            bindings,
            &assets::select(phase, stage)?,
            GdnFamilyIdentity::original08(),
        )
    }
    // Metadata only. Actual2B source/SSA/lease/root authority is retained by
    // ComponentCode::Qwen2b; this constructor cannot select that code path.
    #[cfg(any(test, target_os = "linux"))]
    pub(super) fn original2b(
        phase: GdnPhase,
        stage: GdnStage,
        asset: &assets::Asset,
        bindings: &[BindingSpec],
    ) -> Result<Self> {
        Self::checked(
            phase,
            stage,
            asset,
            bindings,
            &assets::select_qwen2b(phase, stage)?,
            GdnFamilyIdentity::original2b(),
        )
    }
    fn checked(
        phase: GdnPhase,
        stage: GdnStage,
        asset: &assets::Asset,
        bindings: &[BindingSpec],
        original: &assets::Asset,
        family: GdnFamilyIdentity,
    ) -> Result<Self> {
        if !source_text_eq(asset.entry, original.entry)
            || !source_text_eq(asset.source, original.source)
            || !source_text_eq(asset.ptx_sha256, original.ptx_sha256)
            || asset.threads != original.threads
            || asset.shared != original.shared
            || asset.grid != original.grid
            || !phase.stages().contains(&stage)
            || bindings.is_empty()
            || bindings.len() > 32
        {
            return Err(Error::Unsupported(
                "different closed GDN arithmetic selection".into(),
            ));
        }
        for b in bindings {
            if b.physical_bytes == 0
                || b.offset
                    .checked_add(b.physical_bytes)
                    .is_none_or(|end| end > b.role.bytes(phase))
            {
                return Err(Error::State("GDN arithmetic binding extent".into()));
            }
        }
        Ok(Self {
            family,
            phase,
            stage,
            entry: asset.entry,
            owned_source: asset.source,
            original_ptx_sha256: asset.ptx_sha256,
            threads: asset.threads,
            shared: asset.shared,
            grid: asset.grid,
            bindings: Arc::from(bindings),
        })
    }
    pub(super) fn metadata_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.bindings.len() * std::mem::size_of::<BindingSpec>() + 16
    }
    pub(super) fn require_stage(&self, phase: GdnPhase, stage: GdnStage) -> Result<()> {
        if self.phase != phase || self.stage != stage {
            return Err(Error::State("GDN arithmetic phase/stage provenance".into()));
        }
        Ok(())
    }
    pub(crate) fn family(&self) -> &GdnFamilyIdentity {
        &self.family
    }
}

// Generated from the current closed selectors; full text, never fitted values.
const ORIGINAL08_CATALOGUE: &[&str] = &[
    include_str!("conv_prefill_one.cu"),
    include_str!("conv_prefill_many.cu"),
    include_str!("conv_decode.cu"),
    include_str!("prep_one.cu"),
    include_str!("prep_two.cu"),
    include_str!("prep_64.cu"),
    include_str!("cumsum.cu"),
    include_str!("kkt.cu"),
    include_str!("solve.cu"),
    include_str!("recompute.cu"),
    include_str!("state_h.cu"),
    include_str!("output.cu"),
    include_str!("decode.cu"),
    include_str!("gated_one.cu"),
    include_str!("gated_two.cu"),
    include_str!("gated_64.cu"),
    include_str!("fixed_prompt_conv.cu"),
    include_str!("fixed_prompt_prep.cu"),
    include_str!("fixed_prompt_cumsum.cu"),
    include_str!("fixed_prompt_kkt.cu"),
    include_str!("fixed_prompt_h.cu"),
    include_str!("fixed_prompt_gated.cu"),
];

#[cfg(any(test, target_os = "linux"))]
const ORIGINAL2B_CATALOGUE: &[&str] = &[
    include_str!("qwen2b_conv_one.cu"),
    include_str!("prep_one.cu"),
    include_str!("fixed_prompt_cumsum.cu"),
    include_str!("kkt.cu"),
    include_str!("solve.cu"),
    include_str!("qwen2b_recompute.cu"),
    include_str!("state_h.cu"),
    include_str!("output.cu"),
    include_str!("gated_one.cu"),
    include_str!("qwen2b_conv_decode.cu"),
    include_str!("decode.cu"),
    include_str!("qwen2b_conv_many.cu"),
    include_str!("prep_64.cu"),
    include_str!("gated_64.cu"),
    include_str!("prep_two.cu"),
    include_str!("gated_two.cu"),
];

#[cfg(test)]
mod tests;
