//! Exact canonical-root <-> GDN-cache bit transport. No StateRoot/Ready authority.
use crate::cuda::kernel::{PinnedKernel, PreparedPinnedLaunch};
use crate::resident_leases::{ResidentLease, ResidentService};
use cubecl::server::Handle;
use cubecl_runtime::kernel::BufferIOAttr;
use std::sync::Arc;
use zerogravity_core::{Error, Result};

const STATE_COPY_SOURCE: &str = include_str!("resident/state_copy.cu");
const CONV_B4_SOURCE: &str = include_str!("resident/state_copy_conv_b4.cu");
const GDN_STATE_IN_SOURCE: &str = include_str!("resident/state_pair_in.cu");
const GDN_STATE_OUT_CERTIFIED_SOURCE: &str = include_str!("resident/state_pair_out_certified.cu");
const RECURRENT_B8_SOURCE: &str = include_str!("resident/state_copy_recurrent_b8.cu");
const RECURRENT_OUT_CERTIFIED_SOURCE: &str =
    include_str!("resident/state_copy_recurrent_out_certified.cu");
pub(crate) const CONV_ELEMENTS: usize = 6144 * 3;
pub(crate) const RECURRENT_ELEMENTS: usize = 16 * 128 * 128;
const GATED_CERTIFICATE_ELEMENTS: usize = 2048;
const CONV_B4_ROWS: usize = 4;
const RECURRENT_B8_ROWS: usize = 8;

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
#[allow(dead_code)]
pub(crate) struct ConvB4Transport {
    launch: PreparedPinnedLaunch,
}
pub(crate) struct RecurrentB8Transport {
    launch: PreparedPinnedLaunch,
}
pub(crate) struct GdnStateIn {
    launch: PreparedPinnedLaunch,
}
pub(crate) struct GdnStateOutCertified {
    launch: PreparedPinnedLaunch,
}
#[allow(dead_code)]
pub(crate) struct RecurrentOutCertified {
    launch: PreparedPinnedLaunch,
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

#[allow(dead_code)]
impl ConvB4Transport {
    /// One exact ConvIn/ConvOut launch across four independent canonical roots.
    /// No cache-slot/session/root authority enters this physical batching layer.
    pub(crate) fn prepare(
        service: &Arc<ResidentService>,
        kind: StateCopy,
        sources: &[Handle],
        destinations: &[Handle],
        status: &ResidentLease,
    ) -> Result<Self> {
        if !matches!(kind, StateCopy::ConvIn | StateCopy::ConvOut)
            || sources.len() != CONV_B4_ROWS
            || destinations.len() != CONV_B4_ROWS
        {
            return Err(Error::Unsupported(
                "convolution B4 transport requires four convolution rows".into(),
            ));
        }
        let client = service.client();
        let (input_width, output_width) = kind.widths();
        let source_bytes = CONV_ELEMENTS
            .checked_mul(input_width)
            .ok_or_else(|| Error::Limit("convolution B4 source byte extent".into()))?;
        let destination_bytes = CONV_ELEMENTS
            .checked_mul(output_width)
            .ok_or_else(|| Error::Limit("convolution B4 destination byte extent".into()))?;
        let mut ranges = Vec::with_capacity(2 * CONV_B4_ROWS + 1);
        for (handle, bytes, role) in sources
            .iter()
            .map(|handle| (handle, source_bytes, "source"))
            .chain(
                destinations
                    .iter()
                    .map(|handle| (handle, destination_bytes, "destination")),
            )
        {
            if handle.service != client.service_id() || handle.size_in_used() != bytes as u64 {
                return Err(Error::State(format!(
                    "convolution B4 transport {role} service/extent differs"
                )));
            }
            let range = crate::resident_program::BufferRange::new(handle)?;
            if ranges
                .iter()
                .any(|prior: &crate::resident_program::BufferRange| range.overlaps(prior))
            {
                return Err(Error::State(
                    "convolution B4 transport independent planes alias".into(),
                ));
            }
            ranges.push(range);
        }
        let status_handle = status.handle();
        if status_handle.service != client.service_id() || status_handle.size_in_used() != 4 {
            return Err(Error::State(
                "convolution B4 transport status service/extent differs".into(),
            ));
        }
        let status_range = crate::resident_program::BufferRange::new(&status_handle)?;
        if ranges.iter().any(|range| status_range.overlaps(range)) {
            return Err(Error::State(
                "convolution B4 transport status aliases state".into(),
            ));
        }

        let (entry, input, output) = match kind {
            StateCopy::ConvIn => ("zg_vllm_root_conv_in_b4", "unsigned int", "unsigned short"),
            StateCopy::ConvOut => ("zg_vllm_root_conv_out_b4", "unsigned short", "unsigned int"),
            _ => unreachable!(),
        };
        let source: Arc<str> = format!(
            "{}\n#define ZG_ENTRY {}\n#define ZG_COPY {}\n#define ZG_ELEMENTS {}u\n#define ZG_INPUT {}\n#define ZG_OUTPUT {}\n{}",
            "// cubecl.cuda.dynamic_shared_bytes=0",
            entry,
            kind.tag(),
            CONV_ELEMENTS,
            input,
            output,
            CONV_B4_SOURCE,
        )
        .into();
        let mut bindings = Vec::with_capacity(9);
        bindings.extend((0..CONV_B4_ROWS).map(|_| (source_bytes, BufferIOAttr::ReadOnly)));
        bindings.extend((0..CONV_B4_ROWS).map(|_| (destination_bytes, BufferIOAttr::WriteOnly)));
        bindings.push((4, BufferIOAttr::ReadWrite));
        let kernel = PinnedKernel::new(entry, source, 128, bindings.into())?;
        let mut handles = Vec::with_capacity(9);
        handles.extend(sources.iter().cloned());
        handles.extend(destinations.iter().cloned());
        handles.push(status_handle);
        let grid = [(CONV_ELEMENTS as u32).div_ceil(128), CONV_B4_ROWS as u32, 1];
        Ok(Self {
            launch: kernel.prepare(client, grid, handles)?,
        })
    }

    pub(crate) fn launch(&self, service: &Arc<ResidentService>) -> Result<usize> {
        self.launch.launch_on(service.client())?;
        Ok(1)
    }
}

impl RecurrentB8Transport {
    /// One exact recurrent transpose launch across eight independent canonical
    /// roots. This batches only the physical row selector; each source and
    /// destination remains a separately authenticated handle.
    pub(crate) fn prepare(
        service: &Arc<ResidentService>,
        kind: StateCopy,
        sources: &[Handle],
        destinations: &[Handle],
        status: &ResidentLease,
    ) -> Result<Self> {
        if !matches!(kind, StateCopy::RecurrentIn | StateCopy::RecurrentOut)
            || sources.len() != RECURRENT_B8_ROWS
            || destinations.len() != RECURRENT_B8_ROWS
        {
            return Err(Error::Unsupported(
                "recurrent B8 transport requires eight recurrent rows".into(),
            ));
        }
        let client = service.client();
        let state_bytes = RECURRENT_ELEMENTS
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("recurrent B8 transport byte extent".into()))?;
        let mut ranges = Vec::with_capacity(2 * RECURRENT_B8_ROWS + 1);
        for (handle, role) in sources
            .iter()
            .map(|handle| (handle, "source"))
            .chain(destinations.iter().map(|handle| (handle, "destination")))
        {
            if handle.service != client.service_id() || handle.size_in_used() != state_bytes as u64
            {
                return Err(Error::State(format!(
                    "recurrent B8 transport {role} service/extent differs"
                )));
            }
            let range = crate::resident_program::BufferRange::new(handle)?;
            if ranges
                .iter()
                .any(|prior: &crate::resident_program::BufferRange| range.overlaps(prior))
            {
                return Err(Error::State(
                    "recurrent B8 transport independent planes alias".into(),
                ));
            }
            ranges.push(range);
        }
        let status_handle = status.handle();
        if status_handle.service != client.service_id() || status_handle.size_in_used() != 4 {
            return Err(Error::State(
                "recurrent B8 transport status service/extent differs".into(),
            ));
        }
        let status_range = crate::resident_program::BufferRange::new(&status_handle)?;
        if ranges.iter().any(|range| status_range.overlaps(range)) {
            return Err(Error::State(
                "recurrent B8 transport status aliases state".into(),
            ));
        }

        let entry = match kind {
            StateCopy::RecurrentIn => "zg_vllm_root_h_in_b8",
            StateCopy::RecurrentOut => "zg_vllm_root_h_out_b8",
            _ => unreachable!(),
        };
        let source: Arc<str> = format!(
            "{}
#define ZG_ENTRY {}
#define ZG_COPY {}
#define ZG_ELEMENTS {}u
{}",
            "// cubecl.cuda.dynamic_shared_bytes=0",
            entry,
            kind.tag(),
            RECURRENT_ELEMENTS,
            RECURRENT_B8_SOURCE,
        )
        .into();
        let mut bindings = Vec::with_capacity(17);
        bindings.extend((0..RECURRENT_B8_ROWS).map(|_| (state_bytes, BufferIOAttr::ReadOnly)));
        bindings.extend((0..RECURRENT_B8_ROWS).map(|_| (state_bytes, BufferIOAttr::WriteOnly)));
        bindings.push((4, BufferIOAttr::ReadWrite));
        let kernel = PinnedKernel::new(entry, source, 128, bindings.into())?;
        let mut handles = Vec::with_capacity(17);
        handles.extend(sources.iter().cloned());
        handles.extend(destinations.iter().cloned());
        handles.push(status_handle);
        let grid = [
            (RECURRENT_ELEMENTS as u32).div_ceil(128),
            RECURRENT_B8_ROWS as u32,
            1,
        ];
        Ok(Self {
            launch: kernel.prepare(client, grid, handles)?,
        })
    }

    pub(crate) fn launch(&self, service: &Arc<ResidentService>) -> Result<usize> {
        self.launch.launch_on(service.client())?;
        Ok(1)
    }
}

fn require_exact_pair_binding(
    service: &Arc<ResidentService>,
    handle: &Handle,
    bytes: usize,
    role: &str,
) -> Result<()> {
    if bytes == 0
        || handle.service != service.client().service_id()
        || handle.size_in_used() != bytes as u64
    {
        return Err(Error::State(format!(
            "paired GDN state {role} service/extent differs"
        )));
    }
    Ok(())
}

fn require_pair_nonaliasing(handles: &[&Handle]) -> Result<()> {
    let ranges = handles
        .iter()
        .map(|handle| crate::resident_program::BufferRange::new(handle))
        .collect::<Result<Vec<_>>>()?;
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].overlaps(&ranges[right]) {
                return Err(Error::State("paired GDN state bindings alias".into()));
            }
        }
    }
    Ok(())
}

impl GdnStateIn {
    /// One coalesced canonical->native import for the paired convolution and
    /// recurrent state of one GDN layer.
    pub(crate) fn prepare(
        service: &Arc<ResidentService>,
        conv_source: Handle,
        recurrent_source: Handle,
        conv_destination: Handle,
        recurrent_destination: Handle,
        status: &ResidentLease,
    ) -> Result<Self> {
        let conv_f32 = CONV_ELEMENTS
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("paired GDN convolution F32 bytes".into()))?;
        let conv_bf16 = CONV_ELEMENTS
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("paired GDN convolution BF16 bytes".into()))?;
        let recurrent_bytes = RECURRENT_ELEMENTS
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("paired GDN recurrent bytes".into()))?;
        let status_handle = status.handle();
        for (handle, bytes, role) in [
            (&conv_source, conv_f32, "conv source"),
            (&recurrent_source, recurrent_bytes, "recurrent source"),
            (&conv_destination, conv_bf16, "conv destination"),
            (
                &recurrent_destination,
                recurrent_bytes,
                "recurrent destination",
            ),
            (&status_handle, 4, "status"),
        ] {
            require_exact_pair_binding(service, handle, bytes, role)?;
        }
        require_pair_nonaliasing(&[
            &conv_source,
            &recurrent_source,
            &conv_destination,
            &recurrent_destination,
            &status_handle,
        ])?;
        let source: Arc<str> = format!(
            "{}\n#define ZG_CONV_ELEMENTS {}u\n#define ZG_RECURRENT_ELEMENTS {}u\n{}",
            "// cubecl.cuda.dynamic_shared_bytes=0",
            CONV_ELEMENTS,
            RECURRENT_ELEMENTS,
            GDN_STATE_IN_SOURCE,
        )
        .into();
        let kernel = PinnedKernel::new(
            "zg_gdn_state_in",
            source,
            128,
            Arc::from([
                (conv_f32, BufferIOAttr::ReadOnly),
                (recurrent_bytes, BufferIOAttr::ReadOnly),
                (conv_bf16, BufferIOAttr::WriteOnly),
                (recurrent_bytes, BufferIOAttr::WriteOnly),
                (4, BufferIOAttr::ReadWrite),
            ]),
        )?;
        let grid = [(RECURRENT_ELEMENTS as u32).div_ceil(128), 1, 1];
        Ok(Self {
            launch: kernel.prepare(
                service.client(),
                grid,
                vec![
                    conv_source,
                    recurrent_source,
                    conv_destination,
                    recurrent_destination,
                    status_handle,
                ],
            )?,
        })
    }

    pub(crate) fn launch(&self, service: &Arc<ResidentService>) -> Result<usize> {
        self.launch.launch_on(service.client())?;
        Ok(1)
    }
}

impl GdnStateOutCertified {
    /// One coalesced native->canonical export for paired GDN state plus
    /// finiteness certification of the three BF16 values that feed/leave the
    /// recurrent half of the layer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        service: &Arc<ResidentService>,
        conv_source: Handle,
        recurrent_source: Handle,
        conv_destination: Handle,
        recurrent_destination: Handle,
        certificate0: Handle,
        certificate1: Handle,
        certificate2: Handle,
        status: &ResidentLease,
    ) -> Result<Self> {
        let conv_bf16 = CONV_ELEMENTS
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("paired GDN convolution BF16 bytes".into()))?;
        let conv_f32 = CONV_ELEMENTS
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("paired GDN convolution F32 bytes".into()))?;
        let recurrent_bytes = RECURRENT_ELEMENTS
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("paired GDN recurrent bytes".into()))?;
        let certificate_elements = |handle: &Handle, role: &str| -> Result<usize> {
            let bytes = usize::try_from(handle.size_in_used())
                .map_err(|_| Error::Limit(format!("paired GDN {role} bytes")))?;
            if bytes == 0 || !bytes.is_multiple_of(2) {
                return Err(Error::State(format!(
                    "paired GDN {role} is not a nonempty BF16 payload"
                )));
            }
            let elements = bytes / 2;
            if elements > RECURRENT_ELEMENTS || u32::try_from(elements).is_err() {
                return Err(Error::Limit(format!("paired GDN {role} elements")));
            }
            Ok(elements)
        };
        let certificate0_elements = certificate_elements(&certificate0, "certificate0")?;
        let certificate1_elements = certificate_elements(&certificate1, "certificate1")?;
        let certificate2_elements = certificate_elements(&certificate2, "certificate2")?;
        let status_handle = status.handle();
        for (handle, bytes, role) in [
            (&conv_source, conv_bf16, "conv source"),
            (&recurrent_source, recurrent_bytes, "recurrent source"),
            (&conv_destination, conv_f32, "conv destination"),
            (
                &recurrent_destination,
                recurrent_bytes,
                "recurrent destination",
            ),
            (&certificate0, certificate0_elements * 2, "certificate0"),
            (&certificate1, certificate1_elements * 2, "certificate1"),
            (&certificate2, certificate2_elements * 2, "certificate2"),
            (&status_handle, 4, "status"),
        ] {
            require_exact_pair_binding(service, handle, bytes, role)?;
        }
        require_pair_nonaliasing(&[
            &conv_source,
            &recurrent_source,
            &conv_destination,
            &recurrent_destination,
            &certificate0,
            &certificate1,
            &certificate2,
            &status_handle,
        ])?;
        let source: Arc<str> = format!(
            "{}\n#define ZG_CONV_ELEMENTS {}u\n#define ZG_RECURRENT_ELEMENTS {}u\n#define ZG_CERTIFICATE0_ELEMENTS {}u\n#define ZG_CERTIFICATE1_ELEMENTS {}u\n#define ZG_CERTIFICATE2_ELEMENTS {}u\n{}",
            "// cubecl.cuda.dynamic_shared_bytes=0",
            CONV_ELEMENTS,
            RECURRENT_ELEMENTS,
            certificate0_elements,
            certificate1_elements,
            certificate2_elements,
            GDN_STATE_OUT_CERTIFIED_SOURCE,
        )
        .into();
        let kernel = PinnedKernel::new(
            "zg_gdn_state_out_certified",
            source,
            128,
            Arc::from([
                (conv_bf16, BufferIOAttr::ReadOnly),
                (recurrent_bytes, BufferIOAttr::ReadOnly),
                (conv_f32, BufferIOAttr::WriteOnly),
                (recurrent_bytes, BufferIOAttr::WriteOnly),
                (certificate0_elements * 2, BufferIOAttr::ReadOnly),
                (certificate1_elements * 2, BufferIOAttr::ReadOnly),
                (certificate2_elements * 2, BufferIOAttr::ReadOnly),
                (4, BufferIOAttr::ReadWrite),
            ]),
        )?;
        let grid = [(RECURRENT_ELEMENTS as u32).div_ceil(128), 1, 1];
        Ok(Self {
            launch: kernel.prepare(
                service.client(),
                grid,
                vec![
                    conv_source,
                    recurrent_source,
                    conv_destination,
                    recurrent_destination,
                    certificate0,
                    certificate1,
                    certificate2,
                    status_handle,
                ],
            )?,
        })
    }

    pub(crate) fn launch(&self, service: &Arc<ResidentService>) -> Result<usize> {
        self.launch.launch_on(service.client())?;
        Ok(1)
    }
}

#[allow(dead_code)]
impl RecurrentOutCertified {
    /// Exact canonical recurrent export plus finiteness certification of one
    /// final GDN BF16 row. This composes two physical checks only; canonical
    /// roots and publication authority remain with the caller.
    pub(crate) fn prepare(
        service: &Arc<ResidentService>,
        source: Handle,
        destination: Handle,
        certificate: Handle,
        status: &ResidentLease,
    ) -> Result<Self> {
        let client = service.client();
        let recurrent_bytes = RECURRENT_ELEMENTS
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("certified recurrent byte extent".into()))?;
        let certificate_bytes = GATED_CERTIFICATE_ELEMENTS
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("certified recurrent payload extent".into()))?;
        let status_handle = status.handle();
        for (handle, bytes, role) in [
            (&source, recurrent_bytes, "source"),
            (&destination, recurrent_bytes, "destination"),
            (&certificate, certificate_bytes, "certificate"),
            (&status_handle, 4, "status"),
        ] {
            if handle.service != client.service_id() || handle.size_in_used() != bytes as u64 {
                return Err(Error::State(format!(
                    "certified recurrent {role} service/extent differs"
                )));
            }
        }
        let ranges = [
            crate::resident_program::BufferRange::new(&source)?,
            crate::resident_program::BufferRange::new(&destination)?,
            crate::resident_program::BufferRange::new(&certificate)?,
            crate::resident_program::BufferRange::new(&status_handle)?,
        ];
        for left in 0..ranges.len() {
            for right in left + 1..ranges.len() {
                if ranges[left].overlaps(&ranges[right]) {
                    return Err(Error::State("certified recurrent bindings alias".into()));
                }
            }
        }
        let source_text: Arc<str> = format!(
            "{}\n#define ZG_COPY {}\n#define ZG_ELEMENTS {}u\n#define ZG_CERTIFY_ELEMENTS {}u\n{}",
            "// cubecl.cuda.dynamic_shared_bytes=0",
            StateCopy::RecurrentOut.tag(),
            RECURRENT_ELEMENTS,
            GATED_CERTIFICATE_ELEMENTS,
            RECURRENT_OUT_CERTIFIED_SOURCE,
        )
        .into();
        let kernel = PinnedKernel::new(
            "zg_vllm_root_h_out_certified",
            source_text,
            128,
            Arc::from([
                (recurrent_bytes, BufferIOAttr::ReadOnly),
                (recurrent_bytes, BufferIOAttr::WriteOnly),
                (certificate_bytes, BufferIOAttr::ReadOnly),
                (4, BufferIOAttr::ReadWrite),
            ]),
        )?;
        let grid = [(RECURRENT_ELEMENTS as u32).div_ceil(128), 1, 1];
        Ok(Self {
            launch: kernel.prepare(
                client,
                grid,
                vec![source, destination, certificate, status_handle],
            )?,
        })
    }

    pub(crate) fn launch(&self, service: &Arc<ResidentService>) -> Result<usize> {
        self.launch.launch_on(service.client())?;
        Ok(1)
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

#[cfg(all(test, feature = "cuda-cublas"))]
pub(crate) async fn qualify_roundtrip_on_nvidia(device: cubecl::Device) -> Result<()> {
    use zerogravity_core::device_events::BufferRole;

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
    fn bf16_bytes(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }
    fn status_word(raw: &[u8]) -> Result<u32> {
        let bytes: [u8; 4] = raw
            .try_into()
            .map_err(|_| Error::State("state transport status extent".into()))?;
        Ok(u32::from_le_bytes(bytes))
    }

    let diagnostic =
        crate::cuda::test_support::CudaDiagnostic::new_on("ordinary-state-transport", device)?;
    let service = diagnostic.service.clone();
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
    let recurrent_native_actual = client
        .read_one(recurrent_native.handle())
        .map_err(|error| Error::State(format!("recurrent native state readback: {error}")))?;
    let recurrent_actual = client
        .read_one(recurrent_out.handle())
        .map_err(|error| Error::State(format!("recurrent state roundtrip readback: {error}")))?;
    let recurrent_native_expected = (0..RECURRENT_ELEMENTS)
        .map(|index| {
            StateCopy::RecurrentIn
                .source_index(index)
                .map(|source| recurrent_words[source])
        })
        .collect::<Result<Vec<_>>>()?;
    if &conv_actual[..] != f32_bytes(&conv_words).as_slice()
        || &recurrent_native_actual[..] != f32_bytes(&recurrent_native_expected).as_slice()
        || &recurrent_actual[..] != f32_bytes(&recurrent_words).as_slice()
    {
        return Err(Error::State(
            "canonical/native GDN state transport bytes differ".into(),
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

    // B4 convolution batching must be exact against four independent scalar
    // ConvIn/ConvOut transports and the CPU canonical/native index mapping.
    let conv_b4_words = (0..CONV_B4_ROWS)
        .map(|lane| {
            let mut words = (0..CONV_ELEMENTS)
                .map(|index| PATTERNS[(index * 29 + lane * 13 + 7) % PATTERNS.len()])
                .collect::<Vec<_>>();
            if lane == 2 {
                words[17] = 0x7f80_0000;
            }
            words
        })
        .collect::<Vec<_>>();
    let (
        conv_b4_prior,
        scalar_conv_native,
        grouped_conv_native,
        scalar_conv_out,
        grouped_conv_out,
        scalar_conv_status,
        grouped_conv_status,
    ) = {
        let mut init = service.begin_initialization(0)?;
        let mut prior = Vec::with_capacity(CONV_B4_ROWS);
        let mut scalar_native = Vec::with_capacity(CONV_B4_ROWS);
        let mut grouped_native = Vec::with_capacity(CONV_B4_ROWS);
        let mut scalar_out = Vec::with_capacity(CONV_B4_ROWS);
        let mut grouped_out = Vec::with_capacity(CONV_B4_ROWS);
        for words in &conv_b4_words {
            prior.push(init.upload_bytes(&f32_bytes(words), BufferRole::RecurrentState)?);
            scalar_native.push(init.allocate_empty(CONV_ELEMENTS * 2, BufferRole::Scratch)?);
            grouped_native.push(init.allocate_empty(CONV_ELEMENTS * 2, BufferRole::Scratch)?);
            scalar_out.push(init.allocate_empty(CONV_ELEMENTS * 4, BufferRole::RecurrentState)?);
            grouped_out.push(init.allocate_empty(CONV_ELEMENTS * 4, BufferRole::RecurrentState)?);
        }
        let scalar_status = init.upload_zeroed(4, BufferRole::Scratch)?;
        let grouped_status = init.upload_zeroed(4, BufferRole::Scratch)?;
        drop(init.complete_sync().await?);
        (
            prior,
            scalar_native,
            grouped_native,
            scalar_out,
            grouped_out,
            scalar_status,
            grouped_status,
        )
    };
    for lane in 0..CONV_B4_ROWS {
        StateTransport::prepare(
            &service,
            StateCopy::ConvIn,
            conv_b4_prior[lane].handle(),
            scalar_conv_native[lane].handle(),
            &scalar_conv_status,
        )?
        .launch(&service)?;
    }
    let grouped_conv_prior = conv_b4_prior
        .iter()
        .map(ResidentLease::handle)
        .collect::<Vec<_>>();
    let grouped_conv_native_handles = grouped_conv_native
        .iter()
        .map(ResidentLease::handle)
        .collect::<Vec<_>>();
    ConvB4Transport::prepare(
        &service,
        StateCopy::ConvIn,
        &grouped_conv_prior,
        &grouped_conv_native_handles,
        &grouped_conv_status,
    )?
    .launch(&service)?;
    for lane in 0..CONV_B4_ROWS {
        StateTransport::prepare(
            &service,
            StateCopy::ConvOut,
            scalar_conv_native[lane].handle(),
            scalar_conv_out[lane].handle(),
            &scalar_conv_status,
        )?
        .launch(&service)?;
    }
    let grouped_conv_out_handles = grouped_conv_out
        .iter()
        .map(ResidentLease::handle)
        .collect::<Vec<_>>();
    ConvB4Transport::prepare(
        &service,
        StateCopy::ConvOut,
        &grouped_conv_native_handles,
        &grouped_conv_out_handles,
        &grouped_conv_status,
    )?
    .launch(&service)?;
    client
        .sync()
        .await
        .map_err(|error| Error::State(format!("convolution B4 transport sync: {error}")))?;
    let scalar_conv_flags = status_word(
        &client
            .read_one(scalar_conv_status.handle())
            .map_err(|error| Error::State(format!("scalar convolution B4 status: {error}")))?,
    )?;
    let grouped_conv_flags = status_word(
        &client
            .read_one(grouped_conv_status.handle())
            .map_err(|error| Error::State(format!("grouped convolution B4 status: {error}")))?,
    )?;
    if scalar_conv_flags != 0x03 || grouped_conv_flags != scalar_conv_flags {
        return Err(Error::State(format!(
            "convolution B4 transport status differs scalar={scalar_conv_flags:#x} grouped={grouped_conv_flags:#x}"
        )));
    }
    for lane in 0..CONV_B4_ROWS {
        let expected_native = (0..CONV_ELEMENTS)
            .map(|index| {
                StateCopy::ConvIn
                    .source_index(index)
                    .map(|source| (conv_b4_words[lane][source] >> 16) as u16)
            })
            .collect::<Result<Vec<_>>>()?;
        let scalar_native_actual = client
            .read_one(scalar_conv_native[lane].handle())
            .map_err(|error| Error::State(format!("scalar convolution B4 native: {error}")))?;
        let grouped_native_actual = client
            .read_one(grouped_conv_native[lane].handle())
            .map_err(|error| Error::State(format!("grouped convolution B4 native: {error}")))?;
        let scalar_out_actual = client
            .read_one(scalar_conv_out[lane].handle())
            .map_err(|error| Error::State(format!("scalar convolution B4 output: {error}")))?;
        let grouped_out_actual = client
            .read_one(grouped_conv_out[lane].handle())
            .map_err(|error| Error::State(format!("grouped convolution B4 output: {error}")))?;
        let expected_native = bf16_bytes(&expected_native);
        let expected_out = f32_bytes(&conv_b4_words[lane]);
        if &scalar_native_actual[..] != expected_native.as_slice()
            || grouped_native_actual != scalar_native_actual
            || &scalar_out_actual[..] != expected_out.as_slice()
            || grouped_out_actual != scalar_out_actual
        {
            return Err(Error::State(format!(
                "convolution B4 grouped/scalar/oracle bytes differ lane={lane}"
            )));
        }
    }

    // B8 recurrent batching must remain exact against both eight independent
    // production transports and the CPU source-index oracle. Include a
    // nonfinite payload so import/export status bits are compared too.
    let recurrent_b8_words = (0..RECURRENT_B8_ROWS)
        .map(|lane| {
            let mut words = (0..RECURRENT_ELEMENTS)
                .map(|index| {
                    let base = PATTERNS[(index * 23 + lane * 11 + 5) % PATTERNS.len()];
                    base | u32::try_from((index + lane * 17) % 251).unwrap()
                })
                .collect::<Vec<_>>();
            if lane == 6 {
                words[17] = 0x7f80_0042;
            }
            words
        })
        .collect::<Vec<_>>();
    let (
        recurrent_b8_prior,
        scalar_native,
        grouped_native,
        scalar_out,
        grouped_out,
        scalar_status,
        grouped_status,
    ) = {
        let mut init = service.begin_initialization(0)?;
        let mut prior = Vec::with_capacity(RECURRENT_B8_ROWS);
        let mut scalar_native = Vec::with_capacity(RECURRENT_B8_ROWS);
        let mut grouped_native = Vec::with_capacity(RECURRENT_B8_ROWS);
        let mut scalar_out = Vec::with_capacity(RECURRENT_B8_ROWS);
        let mut grouped_out = Vec::with_capacity(RECURRENT_B8_ROWS);
        for words in &recurrent_b8_words {
            prior.push(init.upload_bytes(&f32_bytes(words), BufferRole::RecurrentState)?);
            scalar_native.push(init.allocate_empty(RECURRENT_ELEMENTS * 4, BufferRole::Scratch)?);
            grouped_native.push(init.allocate_empty(RECURRENT_ELEMENTS * 4, BufferRole::Scratch)?);
            scalar_out
                .push(init.allocate_empty(RECURRENT_ELEMENTS * 4, BufferRole::RecurrentState)?);
            grouped_out
                .push(init.allocate_empty(RECURRENT_ELEMENTS * 4, BufferRole::RecurrentState)?);
        }
        let scalar_status = init.upload_zeroed(4, BufferRole::Scratch)?;
        let grouped_status = init.upload_zeroed(4, BufferRole::Scratch)?;
        drop(init.complete_sync().await?);
        (
            prior,
            scalar_native,
            grouped_native,
            scalar_out,
            grouped_out,
            scalar_status,
            grouped_status,
        )
    };

    for lane in 0..RECURRENT_B8_ROWS {
        StateTransport::prepare(
            &service,
            StateCopy::RecurrentIn,
            recurrent_b8_prior[lane].handle(),
            scalar_native[lane].handle(),
            &scalar_status,
        )?
        .launch(&service)?;
    }
    let grouped_prior = recurrent_b8_prior
        .iter()
        .map(ResidentLease::handle)
        .collect::<Vec<_>>();
    let grouped_native_handles = grouped_native
        .iter()
        .map(ResidentLease::handle)
        .collect::<Vec<_>>();
    RecurrentB8Transport::prepare(
        &service,
        StateCopy::RecurrentIn,
        &grouped_prior,
        &grouped_native_handles,
        &grouped_status,
    )?
    .launch(&service)?;

    for lane in 0..RECURRENT_B8_ROWS {
        StateTransport::prepare(
            &service,
            StateCopy::RecurrentOut,
            scalar_native[lane].handle(),
            scalar_out[lane].handle(),
            &scalar_status,
        )?
        .launch(&service)?;
    }
    let grouped_out_handles = grouped_out
        .iter()
        .map(ResidentLease::handle)
        .collect::<Vec<_>>();
    RecurrentB8Transport::prepare(
        &service,
        StateCopy::RecurrentOut,
        &grouped_native_handles,
        &grouped_out_handles,
        &grouped_status,
    )?
    .launch(&service)?;

    client
        .sync()
        .await
        .map_err(|error| Error::State(format!("recurrent B8 transport sync: {error}")))?;
    let scalar_flags = status_word(
        &client
            .read_one(scalar_status.handle())
            .map_err(|error| Error::State(format!("scalar recurrent B8 status: {error}")))?,
    )?;
    let grouped_flags = status_word(
        &client
            .read_one(grouped_status.handle())
            .map_err(|error| Error::State(format!("grouped recurrent B8 status: {error}")))?,
    )?;
    if scalar_flags != 0x0c || grouped_flags != scalar_flags {
        return Err(Error::State(format!(
            "recurrent B8 transport status differs scalar={scalar_flags:#x} grouped={grouped_flags:#x}"
        )));
    }

    for lane in 0..RECURRENT_B8_ROWS {
        let expected_native = (0..RECURRENT_ELEMENTS)
            .map(|index| {
                StateCopy::RecurrentIn
                    .source_index(index)
                    .map(|source| recurrent_b8_words[lane][source])
            })
            .collect::<Result<Vec<_>>>()?;
        let scalar_native_actual = client
            .read_one(scalar_native[lane].handle())
            .map_err(|error| Error::State(format!("scalar recurrent B8 native: {error}")))?;
        let grouped_native_actual = client
            .read_one(grouped_native[lane].handle())
            .map_err(|error| Error::State(format!("grouped recurrent B8 native: {error}")))?;
        let scalar_out_actual = client
            .read_one(scalar_out[lane].handle())
            .map_err(|error| Error::State(format!("scalar recurrent B8 output: {error}")))?;
        let grouped_out_actual = client
            .read_one(grouped_out[lane].handle())
            .map_err(|error| Error::State(format!("grouped recurrent B8 output: {error}")))?;
        let expected_native = f32_bytes(&expected_native);
        let expected_out = f32_bytes(&recurrent_b8_words[lane]);
        if &scalar_native_actual[..] != expected_native.as_slice()
            || grouped_native_actual != scalar_native_actual
            || &scalar_out_actual[..] != expected_out.as_slice()
            || grouped_out_actual != scalar_out_actual
        {
            return Err(Error::State(format!(
                "recurrent B8 grouped/scalar/oracle bytes differ lane={lane}"
            )));
        }
    }

    // Fused recurrent export + final-BF16 certification must preserve the
    // scalar recurrent bytes while OR-ing exactly the certificate bit.
    let mut certified_fixtures = Vec::new();
    for (lane, inject_certificate_nonfinite) in [(0usize, false), (6usize, true)] {
        let mut certificate_words = (0..GATED_CERTIFICATE_ELEMENTS)
            .map(|index| {
                let bits = PATTERNS[(index * 31 + lane * 7 + 3) % PATTERNS.len()];
                (bits >> 16) as u16
            })
            .collect::<Vec<_>>();
        if inject_certificate_nonfinite {
            certificate_words[29] = 0x7f80;
        }
        let (certificate, scalar_output, fused_output, scalar_status, fused_status) = {
            let mut init = service.begin_initialization(0)?;
            let certificate =
                init.upload_bytes(&bf16_bytes(&certificate_words), BufferRole::Scratch)?;
            let scalar_output =
                init.allocate_empty(RECURRENT_ELEMENTS * 4, BufferRole::RecurrentState)?;
            let fused_output =
                init.allocate_empty(RECURRENT_ELEMENTS * 4, BufferRole::RecurrentState)?;
            let scalar_status = init.upload_zeroed(4, BufferRole::Scratch)?;
            let fused_status = init.upload_zeroed(4, BufferRole::Scratch)?;
            drop(init.complete_sync().await?);
            (
                certificate,
                scalar_output,
                fused_output,
                scalar_status,
                fused_status,
            )
        };
        StateTransport::prepare(
            &service,
            StateCopy::RecurrentOut,
            grouped_native[lane].handle(),
            scalar_output.handle(),
            &scalar_status,
        )?
        .launch(&service)?;
        RecurrentOutCertified::prepare(
            &service,
            grouped_native[lane].handle(),
            fused_output.handle(),
            certificate.handle(),
            &fused_status,
        )?
        .launch(&service)?;
        client
            .sync()
            .await
            .map_err(|error| Error::State(format!("certified recurrent sync: {error}")))?;
        let scalar_flags =
            status_word(&client.read_one(scalar_status.handle()).map_err(|error| {
                Error::State(format!("scalar certified recurrent status: {error}"))
            })?)?;
        let fused_flags =
            status_word(&client.read_one(fused_status.handle()).map_err(|error| {
                Error::State(format!("fused certified recurrent status: {error}"))
            })?)?;
        let expected = scalar_flags | if inject_certificate_nonfinite { 2 } else { 0 };
        let scalar_bytes = client
            .read_one(scalar_output.handle())
            .map_err(|error| Error::State(format!("scalar certified recurrent output: {error}")))?;
        let fused_bytes = client
            .read_one(fused_output.handle())
            .map_err(|error| Error::State(format!("fused certified recurrent output: {error}")))?;
        if fused_flags != expected || fused_bytes != scalar_bytes {
            return Err(Error::State(format!(
                "certified recurrent differs lane={lane} scalar={scalar_flags:#x} fused={fused_flags:#x} expected={expected:#x}"
            )));
        }
        certified_fixtures.push((
            certificate,
            scalar_output,
            fused_output,
            scalar_status,
            fused_status,
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
        conv_b4_prior,
        scalar_conv_native,
        grouped_conv_native,
        scalar_conv_out,
        grouped_conv_out,
        scalar_conv_status,
        grouped_conv_status,
        recurrent_b8_prior,
        scalar_native,
        grouped_native,
        scalar_out,
        grouped_out,
        scalar_status,
        grouped_status,
        certified_fixtures,
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
