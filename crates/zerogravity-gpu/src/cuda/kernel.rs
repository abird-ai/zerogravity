//! Closed source kernels on the existing CUDA client; no second stream/allocator.
//! Callers retain admitted original leases through completion/quarantine.
use crate::prepared_kernel_key::PreparedKernelKey;
use cubecl::{
    client::Client,
    ir::{AddressType, ElemType, Scope, metadata::Info, settings::KernelSettings},
    prelude::{CubeCount, ExecutionMode},
    server::Handle,
};
use cubecl_runtime::{
    id::KernelId,
    kernel::{BufferIOAttr, CubeKernel, KernelDefinition, KernelMetadata, PrecompiledSource},
    server::{KernelArguments, MetadataBindingInfo},
};
use std::{
    fmt,
    hash::{Hash, Hasher},
    sync::Arc,
};
use zerogravity_core::source::exact_text_eq;
use zerogravity_core::{Error, Result};

/// Generated CUDA text contains the pinned PTX instructions, not fitted math.
/// Full text participates in cold hashing and exact equality; Debug is bounded.
struct SourceKey {
    entry: &'static str,
    source: Arc<str>,
}
impl PartialEq for SourceKey {
    fn eq(&self, other: &Self) -> bool {
        self.entry == other.entry && exact_text_eq(&self.source, &other.source)
    }
}
impl Eq for SourceKey {}
impl Hash for SourceKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.entry.hash(state);
        self.source.hash(state);
    }
}
impl fmt::Debug for SourceKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedCudaKernel")
            .field("entry", &self.entry)
            .field("source_bytes", &self.source.len())
            .finish()
    }
}

const MAX_STATIC_SCALARS: usize = 8;
const STATIC_SCALAR_ALIGNMENT: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StaticScalarType {
    U32,
    U64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct StaticScalarField {
    ty: StaticScalarType,
    offset: u8,
}

/// Closed trailing by-value ABI. Every typed value occupies one aligned u64
/// word; explicit offsets keep padding and order in the compiled cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StaticScalarSchema {
    fields: [StaticScalarField; MAX_STATIC_SCALARS],
    count: u8,
    byte_size: u8,
    alignment: u8,
}
impl StaticScalarSchema {
    pub(crate) const fn empty() -> Self {
        Self {
            fields: [StaticScalarField {
                ty: StaticScalarType::U64,
                offset: 0,
            }; MAX_STATIC_SCALARS],
            count: 0,
            byte_size: 0,
            alignment: STATIC_SCALAR_ALIGNMENT as u8,
        }
    }

    pub(crate) fn new(
        fields: &[(StaticScalarType, usize)],
        byte_size: usize,
        alignment: usize,
    ) -> Result<Self> {
        if fields.is_empty()
            || fields.len() > MAX_STATIC_SCALARS
            || alignment != STATIC_SCALAR_ALIGNMENT
            || byte_size != fields.len().saturating_mul(STATIC_SCALAR_ALIGNMENT)
            || byte_size > u8::MAX as usize
            || fields
                .iter()
                .enumerate()
                .any(|(index, (_, offset))| *offset != index * STATIC_SCALAR_ALIGNMENT)
        {
            return Err(Error::Unsupported(
                "closed CUDA static scalar ABI padding/count".into(),
            ));
        }
        let mut result = Self::empty();
        for (target, &(ty, offset)) in result.fields.iter_mut().zip(fields) {
            *target = StaticScalarField {
                ty,
                offset: offset as u8,
            };
        }
        result.count = fields.len() as u8;
        result.byte_size = byte_size as u8;
        Ok(result)
    }

    fn fields(self) -> impl ExactSizeIterator<Item = StaticScalarField> {
        self.fields.into_iter().take(self.count as usize)
    }

    #[cfg(test)]
    pub(crate) fn byte_size(self) -> usize {
        self.byte_size as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StaticScalarValue {
    U32(u64),
    U64(u64),
}

/// Immutable fixed metadata words. This is inline host data, never a device
/// buffer or dynamic metadata tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StaticScalarBlock {
    schema: StaticScalarSchema,
    words: [u64; MAX_STATIC_SCALARS],
}
impl StaticScalarBlock {
    pub(crate) const fn empty() -> Self {
        Self {
            schema: StaticScalarSchema::empty(),
            words: [0; MAX_STATIC_SCALARS],
        }
    }

    pub(crate) fn new(schema: StaticScalarSchema, values: &[StaticScalarValue]) -> Result<Self> {
        if values.len() != schema.count as usize {
            return Err(Error::State(
                "closed CUDA static scalar count differs".into(),
            ));
        }
        let mut words = [0; MAX_STATIC_SCALARS];
        for ((field, value), target) in schema.fields().zip(values).zip(words.iter_mut()) {
            *target = match (field.ty, value) {
                (StaticScalarType::U32, StaticScalarValue::U32(bits))
                    if *bits <= u32::MAX as u64 =>
                {
                    *bits
                }
                (StaticScalarType::U64, StaticScalarValue::U64(bits)) => *bits,
                (StaticScalarType::U32, StaticScalarValue::U32(_)) => {
                    return Err(Error::Limit("closed CUDA u32 scalar overflow".into()));
                }
                _ => {
                    return Err(Error::State(
                        "closed CUDA static scalar type/order differs".into(),
                    ));
                }
            };
        }
        Ok(Self { schema, words })
    }

    pub(crate) fn schema(self) -> StaticScalarSchema {
        self.schema
    }

    pub(crate) fn words(&self) -> &[u64] {
        &self.words[..self.schema.count as usize]
    }

    fn binding_info(&self) -> MetadataBindingInfo {
        let words = self.words().to_vec();
        let word_count = words.len();
        MetadataBindingInfo::new(words, word_count)
    }

    #[cfg(test)]
    pub(crate) fn packed_bytes(&self) -> Vec<u8> {
        self.words()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }
}

#[derive(Clone)]
pub(crate) struct PinnedKernel {
    key: PreparedKernelKey<SourceKey>,
    threads: u32,
    shared_bytes: usize,
    // Exact physical sizes and access, in CUDA pointer-parameter order.
    bindings: Arc<[(usize, BufferIOAttr)]>,
    static_scalars: StaticScalarBlock,
}

/// One immutable, validated launch binding. Owning stages retain the original
/// leases/flight; this capsule grants neither execution order nor completion.
#[derive(Clone)]
pub(crate) struct PreparedPinnedLaunch {
    kernel: PinnedKernel,
    client: Client,
    grid: [u32; 3],
    handles: Arc<[Handle]>,
    static_scalars: StaticScalarBlock,
}
impl PreparedPinnedLaunch {
    #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        self.kernel.prepare_code(queue)
    }
    pub(crate) fn handles(&self) -> &[Handle] {
        &self.handles
    }
    pub(crate) fn launch(&self) -> Result<()> {
        self.kernel
            .launch_bound(&self.client, self.grid, &self.handles, self.static_scalars)
    }
    pub(crate) fn launch_on(&self, client: &Client) -> Result<()> {
        if client.service_id() != self.client.service_id() {
            return Err(Error::State(
                "prepared kernel original client differs".into(),
            ));
        }
        self.kernel
            .launch_bound(client, self.grid, &self.handles, self.static_scalars)
    }
}
impl PinnedKernel {
    #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<()> {
        // SAFETY: private source/ABI factory; original actor requires installed
        // detached policy ownership. No buffers or inference are submitted.
        unsafe { queue.prepare_kernel(Box::new(self.clone())) }
            .map_err(|e| Error::State(format!("original CUDA code readiness: {e}")))
    }
    // Charged once per new(), shared by all prepared/current/driver copies.
    pub(crate) const KEY_METADATA_BYTES: usize = PreparedKernelKey::<SourceKey>::ALLOCATION_BYTES;

    /// Exact pointer-parameter count of this kernel's ABI, in binding order.
    #[cfg(test)]
    pub(crate) fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    #[cfg(test)]
    pub(crate) fn binding_sizes(&self) -> Vec<usize> {
        self.bindings.iter().map(|(bytes, _)| *bytes).collect()
    }

    /// Reject a wrong closed native kernel even when its pointer widths happen
    /// to match another family's ABI (e.g. GDN H import/export). No heap work.
    pub(crate) fn require_typed_abi(
        &self,
        entry: &str,
        expected: impl ExactSizeIterator<Item = (usize, BufferIOAttr)>,
    ) -> Result<()> {
        if self.key.entry != entry
            || self.bindings.len() != expected.len()
            || !self
                .bindings
                .iter()
                .zip(expected)
                .all(|(actual, want)| *actual == want)
        {
            return Err(Error::State("pinned CUDA source/typed ABI differs".into()));
        }
        Ok(())
    }

    /// Retained heap owners only; enclosing definitions price their inline
    /// structs separately. Source includes the exact CUDA entry suffix.
    pub(crate) fn retained_metadata_bytes(&self) -> Result<usize> {
        let arc = |payload: usize| {
            payload
                .checked_add(2 * std::mem::size_of::<usize>())
                .and_then(|n| n.checked_add(std::mem::align_of::<usize>() - 1))
                .map(|n| n & !(std::mem::align_of::<usize>() - 1))
        };
        let bindings = self
            .bindings
            .len()
            .checked_mul(std::mem::size_of::<(usize, BufferIOAttr)>());
        arc(self.key.source.len())
            .and_then(|source| bindings.and_then(arc).and_then(|b| source.checked_add(b)))
            .and_then(|n| n.checked_add(Self::KEY_METADATA_BYTES))
            .ok_or_else(|| Error::Limit("pinned source/ABI/key metadata overflow".into()))
    }

    pub(crate) fn prepare(
        &self,
        client: &Client,
        grid: [u32; 3],
        handles: Vec<Handle>,
    ) -> Result<PreparedPinnedLaunch> {
        self.validate(client, grid, &handles)?;
        Ok(PreparedPinnedLaunch {
            kernel: self.clone(),
            client: client.clone(),
            grid,
            handles: handles.into(),
            static_scalars: self.static_scalars,
        })
    }
    /// Inputs are internal generated constants. This is not a public code loader.
    pub(crate) fn new(
        entry: &'static str,
        source: Arc<str>,
        threads: u32,
        bindings: Arc<[(usize, BufferIOAttr)]>,
    ) -> Result<Self> {
        Self::new_with_static_scalars(entry, source, threads, bindings, StaticScalarBlock::empty())
    }

    pub(crate) fn new_with_static_scalars(
        entry: &'static str,
        source: Arc<str>,
        threads: u32,
        bindings: Arc<[(usize, BufferIOAttr)]>,
        static_scalars: StaticScalarBlock,
    ) -> Result<Self> {
        let shared_bytes = match source.strip_prefix("// cubecl.cuda.dynamic_shared_bytes=") {
            Some(rest) => rest
                .split_once('\n')
                .and_then(|(value, _)| value.parse::<usize>().ok())
                .filter(|bytes| *bytes <= 96 * 1024)
                .ok_or_else(|| {
                    Error::Unsupported("closed CUDA shared-memory declaration".into())
                })?,
            None => 0,
        };
        if entry.is_empty()
            || !entry
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
            || source.is_empty()
            || source.len() > 2 * 1024 * 1024
            || source.as_bytes().contains(&0)
            || !threads.is_multiple_of(32)
            || !(32..=1024).contains(&threads)
            || bindings.is_empty()
            || bindings.len() > 32
            || bindings.iter().any(|(bytes, _)| *bytes == 0)
        {
            return Err(Error::Unsupported(
                "closed vLLM CUDA source geometry".into(),
            ));
        }
        Ok(Self {
            // CubeCL's second-line CUDA cache keys on source text. Distinguish
            // entries even when a source unit contains several stored stages.
            key: PreparedKernelKey::new(SourceKey {
                entry,
                source: format!("{source}\n// zerogravity.cuda.entry={entry}\n").into(),
            }),
            threads,
            shared_bytes,
            bindings,
            static_scalars,
        })
    }

    /// Pure validation, before the caller arms its original submission.
    pub(crate) fn validate(
        &self,
        client: &Client,
        grid: [u32; 3],
        handles: &[Handle],
    ) -> Result<()> {
        let hardware = &client.properties().hardware;
        let limits = hardware.max_cube_count;
        if !crate::quantization::activation_fused::is_cuda_service(client)
            || client.properties().identity.fingerprint != "ptx_sm86"
            || handles.len() != self.bindings.len()
            || hardware.max_bindings < handles.len() as u32
            || hardware.max_units_per_cube < self.threads
            || hardware.max_cube_dim.0 < self.threads
            || hardware.max_shared_memory_size < self.shared_bytes
            || grid
                .into_iter()
                .zip([limits.0, limits.1, limits.2])
                .any(|(n, max)| n == 0 || n > max)
        {
            return Err(Error::Unsupported(
                "pinned vLLM CUDA launch geometry".into(),
            ));
        }
        for (handle, (bytes, _)) in handles.iter().zip(self.bindings.iter()) {
            if handle.service != client.service_id()
                || !handle.offset_start.unwrap_or(0).is_multiple_of(4)
                || handle.size_in_used() != *bytes as u64
            {
                return Err(Error::State(
                    "pinned vLLM CUDA original binding extent/service".into(),
                ));
            }
        }
        Ok(())
    }

    /// Original owner must already retain all handles and a physical flight.
    /// This method allocates no device storage and never declares completion.
    pub(crate) fn launch(&self, client: &Client, grid: [u32; 3], handles: &[Handle]) -> Result<()> {
        self.validate(client, grid, handles)?;
        self.launch_bound(client, grid, handles, self.static_scalars)
    }
    fn launch_bound(
        &self,
        client: &Client,
        grid: [u32; 3],
        handles: &[Handle],
        static_scalars: StaticScalarBlock,
    ) -> Result<()> {
        let args = self.launch_arguments(handles, static_scalars)?;
        client.launch(
            Box::new(self.clone()),
            CubeCount::Static(grid[0], grid[1], grid[2]),
            args,
        );
        Ok(())
    }

    /// Build the exact host packet consumed by the existing CUDA server. This
    /// proves host ordering only; compiler-reflected and physical ABI remain a
    /// native gate.
    fn launch_arguments(
        &self,
        handles: &[Handle],
        static_scalars: StaticScalarBlock,
    ) -> Result<KernelArguments> {
        if handles.len() != self.bindings.len() {
            return Err(Error::State(
                "prepared CUDA pointer argument count changed".into(),
            ));
        }
        if static_scalars.schema() != self.static_scalars.schema() {
            return Err(Error::State(
                "prepared CUDA static scalar schema changed".into(),
            ));
        }
        let mut args = KernelArguments::new();
        for (handle, (_, access)) in handles.iter().zip(self.bindings.iter()) {
            args = args.with_buffer_io(handle.clone().binding(), *access);
        }
        Ok(args.with_info(static_scalars.binding_info()))
    }

    #[cfg(test)]
    pub(crate) fn host_launch_arguments(
        &self,
        handles: &[Handle],
        static_scalars: StaticScalarBlock,
    ) -> Result<KernelArguments> {
        self.launch_arguments(handles, static_scalars)
    }
}
impl KernelMetadata for PinnedKernel {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .cube_dim((self.threads, 1, 1).into())
            .address_type(AddressType::U64)
            .mode(ExecutionMode::Unchecked)
            .info((
                self.key.clone(),
                self.bindings.clone(),
                self.static_scalars.schema(),
            ))
    }
    fn address_type(&self) -> ElemType {
        AddressType::U64.unsigned_type()
    }
}
impl CubeKernel for PinnedKernel {
    fn define(&self) -> KernelDefinition {
        let mut settings = KernelSettings::new(
            (self.threads, 1, 1).into(),
            ExecutionMode::Unchecked,
            AddressType::U64,
        );
        settings.kernel_name = self.key.entry.into();
        KernelDefinition {
            body: Scope::root(settings.clone()),
            info: Info::default(),
            settings,
        }
    }
    fn source(&self) -> Option<PrecompiledSource> {
        Some(PrecompiledSource {
            source: self.key.source.to_string(),
            entrypoint_name: self.key.entry.into(),
            lang: "cuda",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn kernel(source: &str) -> PinnedKernel {
        PinnedKernel::new(
            "fixed",
            Arc::from(source),
            32,
            Arc::from([(4, BufferIOAttr::ReadWrite)]),
        )
        .unwrap()
    }
    #[test]
    fn actual_precompiled_source_has_no_access_analysis_and_keeps_its_bounded_abi() {
        use cubecl::cuda::{CudaCompilationOptions, CudaCompiler, CudaCompilerContextPolicy};
        use cubecl_server::kernel::CompiledKernel;
        let kernel = kernel("extern \"C\" __global__ void fixed(float* p) { p[0] = 1.0f; }");
        let id = kernel.id();
        let source = kernel.source().unwrap().source;
        let compiled = CompiledKernel::compile(
            &kernel,
            kernel.define(),
            &mut CudaCompiler::default(),
            &CudaCompilationOptions::default(),
        )
        .unwrap();
        assert!(compiled.io.is_none()); // no IR analysis; original caller/fallback IO stays unchanged
        assert!(compiled.repr.is_none());
        assert_eq!(compiled.entrypoint_name, "fixed");
        assert_eq!(compiled.source, source);
        assert_eq!(kernel.id(), id);
        CudaCompilerContextPolicy::validate_compiled_metadata(
            compiled.entrypoint_name.len(),
            compiled.io.as_ref().map(Vec::len),
        )
        .unwrap();
        assert!(CudaCompilerContextPolicy::validate_compiled_metadata(257, None).is_err());
        assert!(CudaCompilerContextPolicy::validate_compiled_metadata(1, Some(33)).is_err());
        for count in [0, 33] {
            assert!(
                PinnedKernel::new(
                    "fixed",
                    Arc::from("one"),
                    32,
                    vec![(4, BufferIOAttr::ReadWrite); count].into()
                )
                .is_err()
            );
        }
        let max = PinnedKernel::new(
            "fixed",
            Arc::from("one"),
            32,
            vec![(4, BufferIOAttr::ReadWrite); 32].into(),
        )
        .unwrap();
        assert_eq!(max.bindings.len(), 32);
        assert_ne!(max.id(), id);
        assert!(max.retained_metadata_bytes().unwrap() < 4096);
    }
    #[test]
    fn cache_identity_covers_complete_source_and_physical_abi() {
        assert_eq!(kernel("one").id(), kernel("one").id());
        assert_ne!(kernel("one").id(), kernel("two").id());
        let mut other = kernel("one");
        other.threads = 64;
        assert_ne!(kernel("one").id(), other.id());
        other = kernel("one");
        other.bindings = Arc::from([(8, BufferIOAttr::ReadWrite)]);
        assert_ne!(kernel("one").id(), other.id());
        other = kernel("one");
        other.bindings = Arc::from([(4, BufferIOAttr::ReadOnly)]);
        assert_ne!(kernel("one").id(), other.id());
        let other_entry = PinnedKernel::new(
            "other",
            Arc::from("one"),
            32,
            Arc::from([(4, BufferIOAttr::ReadWrite)]),
        )
        .unwrap();
        assert_ne!(kernel("one").id(), other_entry.id());
    }
    #[test]
    fn prepared_key_retains_complete_source_with_bounded_shared_metadata() {
        let first = kernel("one");
        let equal = kernel("one");
        let cloned = first.clone();
        assert_ne!(first.key.source.as_ptr(), equal.key.source.as_ptr());
        assert_eq!(first.key, equal.key);
        assert_eq!(first.id().stable_hash(), equal.id().stable_hash());
        assert_eq!(first.id(), cloned.id());
        assert_eq!(
            first.source().unwrap().source,
            equal.source().unwrap().source
        );
        assert_eq!(
            first.key.source.as_ref(),
            "one\n// zerogravity.cuda.entry=fixed\n"
        );
        assert_eq!(format!("{:?}", first.key), format!("{:?}", *first.key));
        const { assert!(PinnedKernel::KEY_METADATA_BYTES <= 128) };
        println!(
            "prepared source key: value={} owner={} capsule={} pinned_kernel={}",
            std::mem::size_of::<SourceKey>(),
            std::mem::size_of::<PreparedKernelKey<SourceKey>>(),
            PinnedKernel::KEY_METADATA_BYTES,
            std::mem::size_of::<PinnedKernel>(),
        );
    }
    #[test]
    fn static_scalar_schema_and_words_have_separate_immutable_identities() -> Result<()> {
        let schema = StaticScalarSchema::new(
            &[
                (StaticScalarType::U32, 0),
                (StaticScalarType::U64, 8),
                (StaticScalarType::U64, 16),
            ],
            24,
            8,
        )?;
        let block = StaticScalarBlock::new(
            schema,
            &[
                StaticScalarValue::U32(5),
                StaticScalarValue::U64(8192),
                StaticScalarValue::U64(6144),
            ],
        )?;
        assert_eq!(schema.byte_size(), 24);
        assert_eq!(block.words(), [5, 8192, 6144]);
        assert_eq!(
            block.packed_bytes(),
            [
                5_u64.to_le_bytes(),
                8192_u64.to_le_bytes(),
                6144_u64.to_le_bytes(),
            ]
            .concat()
        );
        let info = block.binding_info();
        assert_eq!(info.data, [5, 8192, 6144]);
        assert_eq!(info.dynamic_metadata_offset, info.data.len());

        let scalar_kernel = |block| {
            PinnedKernel::new_with_static_scalars(
                "fixed",
                Arc::from("one"),
                32,
                Arc::from([(4, BufferIOAttr::ReadWrite)]),
                block,
            )
            .unwrap()
        };
        let changed = StaticScalarBlock::new(
            schema,
            &[
                StaticScalarValue::U32(6),
                StaticScalarValue::U64(8192),
                StaticScalarValue::U64(6144),
            ],
        )?;
        // Values identify prepared arithmetic, not compiled code/cache.
        assert_eq!(scalar_kernel(block).id(), scalar_kernel(changed).id());
        assert_ne!(block, changed);
        assert_ne!(scalar_kernel(block).id(), kernel("one").id());

        let reordered = StaticScalarSchema::new(
            &[
                (StaticScalarType::U64, 0),
                (StaticScalarType::U32, 8),
                (StaticScalarType::U64, 16),
            ],
            24,
            8,
        )?;
        let reordered_block = StaticScalarBlock::new(
            reordered,
            &[
                StaticScalarValue::U64(5),
                StaticScalarValue::U32(8192),
                StaticScalarValue::U64(6144),
            ],
        )?;
        assert_ne!(
            scalar_kernel(block).id(),
            scalar_kernel(reordered_block).id()
        );
        Ok(())
    }

    #[test]
    fn malformed_static_scalar_abi_refuses_count_type_padding_and_overflow() -> Result<()> {
        let schema = StaticScalarSchema::new(
            &[(StaticScalarType::U32, 0), (StaticScalarType::U64, 8)],
            16,
            8,
        )?;
        assert!(StaticScalarBlock::new(schema, &[StaticScalarValue::U32(1)]).is_err());
        assert!(
            StaticScalarBlock::new(
                schema,
                &[StaticScalarValue::U64(1), StaticScalarValue::U32(2)]
            )
            .is_err()
        );
        assert!(
            StaticScalarBlock::new(
                schema,
                &[
                    StaticScalarValue::U32(u32::MAX as u64 + 1),
                    StaticScalarValue::U64(2),
                ]
            )
            .is_err()
        );
        assert!(StaticScalarSchema::new(&[(StaticScalarType::U32, 0)], 16, 8).is_err());
        assert!(StaticScalarSchema::new(&[(StaticScalarType::U32, 8)], 8, 8).is_err());
        assert!(StaticScalarSchema::new(&[(StaticScalarType::U32, 0)], 8, 4).is_err());
        assert!(
            StaticScalarSchema::new(
                &[(StaticScalarType::U64, 0); MAX_STATIC_SCALARS + 1],
                usize::MAX,
                8,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn incomplete_source_and_launch_contracts_refuse_before_device_access() {
        for (entry, source, threads, bytes) in [
            ("", "one", 32, 4),
            ("bad;", "one", 32, 4),
            ("fixed", "", 32, 4),
            ("fixed", "bad\0", 32, 4),
            ("fixed", "one", 31, 4),
            ("fixed", "one", 32, 0),
        ] {
            assert!(
                PinnedKernel::new(
                    entry,
                    Arc::from(source),
                    threads,
                    Arc::from([(bytes, BufferIOAttr::ReadWrite)])
                )
                .is_err()
            );
        }
    }
}
