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

#[cfg(test)]
pub(crate) async fn qualify_roundtrip_on_nvidia(device: cubecl::Device) -> Result<()> {
    use crate::resident_leases::ResidentService;
    use zerogravity_core::{
        accelerator::SourceIdentity, device_events::BufferRole, resource_arbiter::ResourceOwner,
    };

    const MIB: usize = 1024 * 1024;
    const PATTERNS: [u32; 10] = [
        0x0000_0000,
        0x8000_0000,
        0x3d80_0000,
        0xbd80_0000,
        0x3e80_0000,
        0xbe80_0000,
        0x3f00_0000,
        0xbf00_0000,
        0x3f80_0000,
        0xbf80_0000,
    ];
    fn f32_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }
    fn status_word(raw: &[u8]) -> Result<u32> {
        let bytes: [u8; 4] = raw
            .try_into()
            .map_err(|_| Error::State("state transport status extent".into()))?;
        Ok(u32::from_le_bytes(bytes))
    }

    let source = SourceIdentity::for_new_allocation()?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source,
        ResourceOwner::new("ordinary-state-transport", "roundtrip")?,
        16 * MIB,
        (32 * MIB) as u64,
    )?;
    let client = service.client().clone();

    let conv_words = (0..CONV_ELEMENTS)
        .map(|index| PATTERNS[(index * 17 + 3) % PATTERNS.len()])
        .collect::<Vec<_>>();
    let recurrent_words = (0..RECURRENT_ELEMENTS)
        .map(|index| {
            let base = PATTERNS[(index * 19 + 7) % PATTERNS.len()];
            // Recurrent transport is exact F32 transposition, so also exercise
            // finite non-BF16 low mantissa bits.
            base | u32::try_from(index % 251).unwrap()
        })
        .collect::<Vec<_>>();

    let (conv_in, conv_native, conv_out, recurrent_in, recurrent_native, recurrent_out, status) = {
        let mut init = service.begin_initialization(0)?;
        let conv_in = init.upload_bytes(&f32_bytes(&conv_words), BufferRole::RecurrentState)?;
        let conv_native = init.allocate_empty(CONV_ELEMENTS * 2, BufferRole::Scratch)?;
        let conv_out = init.allocate_empty(CONV_ELEMENTS * 4, BufferRole::RecurrentState)?;
        let recurrent_in =
            init.upload_bytes(&f32_bytes(&recurrent_words), BufferRole::RecurrentState)?;
        let recurrent_native = init.allocate_empty(RECURRENT_ELEMENTS * 4, BufferRole::Scratch)?;
        let recurrent_out =
            init.allocate_empty(RECURRENT_ELEMENTS * 4, BufferRole::RecurrentState)?;
        let status = init.upload_zeroed(4, BufferRole::Scratch)?;
        drop(init.complete_sync().await?);
        (
            conv_in,
            conv_native,
            conv_out,
            recurrent_in,
            recurrent_native,
            recurrent_out,
            status,
        )
    };

    StateTransport::prepare(
        &service,
        StateCopy::ConvIn,
        conv_in.handle(),
        conv_native.handle(),
        &status,
    )?
    .launch(&service)?;
    StateTransport::prepare(
        &service,
        StateCopy::ConvOut,
        conv_native.handle(),
        conv_out.handle(),
        &status,
    )?
    .launch(&service)?;
    StateTransport::prepare(
        &service,
        StateCopy::RecurrentIn,
        recurrent_in.handle(),
        recurrent_native.handle(),
        &status,
    )?
    .launch(&service)?;
    StateTransport::prepare(
        &service,
        StateCopy::RecurrentOut,
        recurrent_native.handle(),
        recurrent_out.handle(),
        &status,
    )?
    .launch(&service)?;

    client
        .sync()
        .await
        .map_err(|error| Error::State(format!("state transport sync: {error}")))?;
    let flags = status_word(
        &client
            .read_one(status.handle())
            .map_err(|error| Error::State(format!("state transport status readback: {error}")))?,
    )?;
    if flags != 0 {
        return Err(Error::State(format!(
            "state transport roundtrip status nonzero: {flags:#x}"
        )));
    }
    let conv_actual = client
        .read_one(conv_out.handle())
        .map_err(|error| Error::State(format!("conv state roundtrip readback: {error}")))?;
    let recurrent_actual = client
        .read_one(recurrent_out.handle())
        .map_err(|error| Error::State(format!("recurrent state roundtrip readback: {error}")))?;
    if &conv_actual[..] != f32_bytes(&conv_words).as_slice()
        || &recurrent_actual[..] != f32_bytes(&recurrent_words).as_slice()
    {
        return Err(Error::State(
            "canonical GDN state transport roundtrip bytes differ".into(),
        ));
    }

    // ConvIn must fail closed when a canonical F32 value is not an exact
    // BF16-widened finite root.
    let bad_words = {
        let mut words = conv_words.clone();
        words[0] = 0x3f80_0001;
        words
    };
    let (bad_in, bad_native, bad_status) = {
        let mut init = service.begin_initialization(0)?;
        let input = init.upload_bytes(&f32_bytes(&bad_words), BufferRole::RecurrentState)?;
        let native = init.allocate_empty(CONV_ELEMENTS * 2, BufferRole::Scratch)?;
        let status = init.upload_zeroed(4, BufferRole::Scratch)?;
        drop(init.complete_sync().await?);
        (input, native, status)
    };
    StateTransport::prepare(
        &service,
        StateCopy::ConvIn,
        bad_in.handle(),
        bad_native.handle(),
        &bad_status,
    )?
    .launch(&service)?;
    client
        .sync()
        .await
        .map_err(|error| Error::State(format!("bad conv state sync: {error}")))?;
    let bad_flags = status_word(
        &client
            .read_one(bad_status.handle())
            .map_err(|error| Error::State(format!("bad conv status readback: {error}")))?,
    )?;
    if bad_flags & 1 == 0 {
        return Err(Error::State(
            "non-BF16 canonical conv root was not rejected".into(),
        ));
    }

    drop((
        conv_in,
        conv_native,
        conv_out,
        recurrent_in,
        recurrent_native,
        recurrent_out,
        status,
        bad_in,
        bad_native,
        bad_status,
    ));
    client
        .memory_cleanup()
        .map_err(|error| Error::State(format!("state transport cleanup: {error}")))?;
    if service.resident_bytes()? != 0 || service.model_events()? != 0 {
        return Err(Error::State(
            "state transport qualification left resident bytes/events".into(),
        ));
    }
    service.close_model_namespace()?;
    println!("ordinary-state-transport roundtrip PASS");
    Ok(())
}
