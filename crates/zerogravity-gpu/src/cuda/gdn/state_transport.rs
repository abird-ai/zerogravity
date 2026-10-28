//! Exact canonical-root <-> GDN-cache bit transport. No StateRoot/Ready authority.
use crate::cuda::kernel::PinnedKernel;
use crate::resident_leases::{ResidentLease, ResidentService};
use cubecl::server::Handle;
use cubecl_runtime::kernel::BufferIOAttr;
use std::sync::Arc;
use zerogravity_core::{Error, Result};

const STATE_COPY_SOURCE: &str = include_str!("resident/state_copy.cu");
pub(crate) const CONV_ELEMENTS: usize = 6144 * 3;
pub(crate) const RECURRENT_ELEMENTS: usize = 16 * 128 * 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StateCopy {
    ConvIn,
    ConvOut,
    RecurrentIn,
    RecurrentOut,
}
impl StateCopy {
    pub(crate) fn tag(self) -> usize {
        match self {
            Self::ConvIn => 0,
            Self::ConvOut => 1,
            Self::RecurrentIn => 2,
            Self::RecurrentOut => 3,
        }
    }
    pub(super) fn elements(self) -> usize {
        if matches!(self, Self::ConvIn | Self::ConvOut) {
            CONV_ELEMENTS
        } else {
            RECURRENT_ELEMENTS
        }
    }
    fn entry(self) -> &'static str {
        match self {
            Self::ConvIn => "zg_vllm_root_conv_in",
            Self::ConvOut => "zg_vllm_root_conv_out",
            Self::RecurrentIn => "zg_vllm_root_h_in",
            Self::RecurrentOut => "zg_vllm_root_h_out",
        }
    }
    pub(super) fn source_index(self, index: usize) -> Result<usize> {
        if index >= self.elements() {
            return Err(Error::Limit("state copy index".into()));
        }
        Ok(match self {
            Self::ConvIn => (index % 6144) * 3 + index / 6144,
            Self::ConvOut => (index % 3) * 6144 + index / 3,
            Self::RecurrentIn | Self::RecurrentOut => {
                (index / 16384) * 16384 + (index % 128) * 128 + (index / 128) % 128
            }
        })
    }
    fn widths(self) -> (usize, usize) {
        match self {
            Self::ConvIn => (4, 2),
            Self::ConvOut => (2, 4),
            _ => (4, 4),
        }
    }
    fn source(self) -> Arc<str> {
        let (input, output) = match self {
            Self::ConvIn => ("unsigned int", "unsigned short"),
            Self::ConvOut => ("unsigned short", "unsigned int"),
            _ => ("unsigned int", "unsigned int"),
        };
        format!("// cubecl.cuda.dynamic_shared_bytes=0\n#define ZG_ENTRY {}\n#define ZG_COPY {}\n#define ZG_ELEMENTS {}u\n#define ZG_INPUT {input}\n#define ZG_OUTPUT {output}\n{STATE_COPY_SOURCE}", self.entry(), self.tag(), self.elements()).into()
    }
}
pub(crate) struct StateTransportDefinition {
    kind: StateCopy,
    kernel: PinnedKernel,
    grid: [u32; 3],
}
pub(crate) struct StateTransport {
    kernel: PinnedKernel,
    grid: [u32; 3],
    handles: Vec<Handle>,
}

pub(crate) fn plane_handle(cache: &ResidentLease, offset: usize, bytes: usize) -> Result<Handle> {
    let end = offset
        .checked_add(bytes)
        .filter(|&end| end <= cache.bytes())
        .ok_or_else(|| Error::State("native state cache plane range".into()))?;
    Ok(cache
        .handle()
        .offset_start(offset as u64)
        .offset_end((cache.bytes() - end) as u64))
}
impl StateTransportDefinition {
    /// Complete source/ABI/key plus cold construction peak. The generated
    /// header and entry suffix are bounded512B; no source text is rendered here.
    pub(crate) fn metadata_bound() -> usize {
        4 * (3 * (STATE_COPY_SOURCE.len() + 512)
            + PinnedKernel::KEY_METADATA_BYTES
            + 3 * std::mem::size_of::<(usize, BufferIOAttr)>()
            + std::mem::size_of::<Self>()
            + 64)
    }
    pub(crate) fn prepare(kind: StateCopy) -> Result<Self> {
        let (input_width, output_width) = kind.widths();
        if kind.source_index(kind.elements() - 1)? >= kind.elements() {
            return Err(Error::State("state transport source extent".into()));
        }
        let bindings = Arc::from([
            (kind.elements() * input_width, BufferIOAttr::ReadOnly),
            (kind.elements() * output_width, BufferIOAttr::WriteOnly),
            (4, BufferIOAttr::ReadWrite),
        ]);
        let kernel = PinnedKernel::new(kind.entry(), kind.source(), 128, bindings)?;
        let grid = [(kind.elements() as u32).div_ceil(128), 1, 1];
        if kernel.retained_metadata_bytes()? + std::mem::size_of::<Self>()
            > Self::metadata_bound() / 4
        {
            return Err(Error::Limit(
                "state transport cold source/ABI/key owner bound".into(),
            ));
        }
        Ok(Self { kind, kernel, grid })
    }
    pub(crate) fn bind(
        &self,
        service: &Arc<ResidentService>,
        kind: StateCopy,
        input: Handle,
        output: Handle,
        status: &ResidentLease,
    ) -> Result<StateTransport> {
        if kind != self.kind {
            return Err(Error::State(
                "state transport cold definition kind differs".into(),
            ));
        }
        let handles = vec![input, output, status.handle()];
        self.kernel
            .validate(service.client(), self.grid, &handles)?;
        Ok(StateTransport {
            kernel: self.kernel.clone(),
            grid: self.grid,
            handles,
        })
    }
}
impl StateTransport {
    pub(crate) fn prepare(
        service: &Arc<ResidentService>,
        kind: StateCopy,
        input: Handle,
        output: Handle,
        status: &ResidentLease,
    ) -> Result<Self> {
        StateTransportDefinition::prepare(kind)?.bind(service, kind, input, output, status)
    }
    pub(crate) fn launch(&self, service: &Arc<ResidentService>) -> Result<()> {
        self.kernel
            .launch(service.client(), self.grid, &self.handles)
    }
}

pub(crate) fn prepare_code(queue: &cubecl::cuda::OwnedCudaQueue) -> Result<usize> {
    for kind in [
        StateCopy::ConvIn,
        StateCopy::ConvOut,
        StateCopy::RecurrentIn,
        StateCopy::RecurrentOut,
    ] {
        StateTransportDefinition::prepare(kind)?
            .kernel
            .prepare_code(queue)?;
    }
    Ok(4)
}
