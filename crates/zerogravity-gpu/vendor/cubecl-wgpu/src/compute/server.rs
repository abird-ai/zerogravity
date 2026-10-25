use cubecl_core::server::{DeviceCaptures, ServerStorage};
use cubecl_server::kernel::BufferIOAttr;
use cubecl_server::kernel::DebugInformation;
use std::collections::HashMap;
use std::marker::PhantomData;

use super::graph::WgpuGraph;
use super::storage::{WgpuResource, WgpuStorage};
use crate::WgpuCompiler;
use crate::backend::ModuleSource;
use crate::schedule::{BindingsResource, ScheduleTask, ScheduledWgpuBackend};
use alloc::sync::Arc;
use cubecl_common::pool::LeasePool;
use cubecl_common::{
    bytes::Bytes,
    profile::{ProfileDuration, TimingMethod},
};
use cubecl_core::server::{BufferBinding, KernelResource};
use cubecl_core::zspace::Shape;
use cubecl_core::{
    MemoryConfiguration, WgpuCompilationOptions,
    prelude::*,
    server::{
        CopyDescriptor, IoError, KernelArguments, LaunchError, ProfileError, ProfilingToken,
        ServerCommunication, ServerError, ServerUtilities,
    },
    zspace::{Strides, strides},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::DynFut;
#[cfg(feature = "spirv")]
use cubecl_environment::persistence::Store;
use cubecl_environment::stream::StreamId;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::compiler::CompilationRecording;
#[cfg(feature = "spirv")]
use cubecl_server::compiler::{KernelCacheKey, compilation_store, store_compiled};
use cubecl_server::memory_management::{
    ManagedMemoryHandle, SharedMemoryBindings, StreamMemoryReport,
    relocation::{RelocatingStreams, RelocationReason},
};
use cubecl_server::{
    compiler::CompilationCache,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    dry_run::LaunchMode,
    id::GraphId,
    kernel::CubeKernel,
    logging::ServerLogger,
    memory_management::MemoryAllocationMode,
    server::Server,
    storage::ManagedResource,
    stream::scheduler::{SchedulerMultiStream, SchedulerMultiStreamOptions, SchedulerStrategy},
    stream::{ExecuteScope, FailureStore, StreamCapture, WriteScoped, failed_writing},
    validation::{validate_cube_dim, validate_units},
};
use wgpu::ComputePipeline;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamsTransfer {
    Immediate,
    Uniform,
}

/// Compiler kind and info used when compiling a specific kernel. Used to determine parameter passing strategies.
/// What a launch needs from a compiled kernel: the pipeline, the parameter
/// strategy, and the per-buffer IO the taint bookkeeping stages from. The IO
/// rides in the cache because on a hit nothing else of the compilation
/// survives.
pub type PipelineEntry = (
    Arc<ComputePipeline>,
    CompilerInfo,
    Option<Arc<[BufferIOAttr]>>,
    Option<Arc<NativeSource>>,
);

/// The actual compiled source retained with its pipeline. Nothing in the
/// launch path can reconstruct this after compilation, so the source-owned
/// set keeps it beside the pipeline for its whole lifetime.
pub struct NativeSource {
    pub entrypoint: String,
    pub spirv: Arc<[u32]>,
    pub bindings: Vec<Visibility>,
    pub io: Option<Arc<[BufferIOAttr]>>,
}

impl std::fmt::Debug for NativeSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeSource")
            .field("entrypoint", &self.entrypoint)
            .field("spirv_words", &self.spirv.len())
            .field("bindings", &self.bindings.len())
            .field("io", &self.io.as_ref().map(|io| io.len()))
            .finish()
    }
}

impl NativeSource {
    #[cfg(feature = "spirv")]
    pub(crate) fn from_spirv(entrypoint: &str, kernel: &cubecl_spirv::SpirvKernel) -> Self {
        Self {
            entrypoint: entrypoint.to_string(),
            spirv: kernel.assembled_module.clone().into(),
            bindings: kernel.bindings.clone(),
            io: kernel.io.clone().map(Arc::from),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerInfo {
    Vulkan { params_transfer: ParamsTransfer },
    Metal,
    WGSL,
    None,
}

/// One compilation's retained output before any driver resource exists.
/// The native catalogue admits the representation from this value and then
/// asks for the module/pipeline, so admission is never retroactive.
#[cfg(feature = "spirv")]
struct CompiledRepresentation {
    entrypoint: String,
    compiler_info: CompilerInfo,
    io: Option<Arc<[BufferIOAttr]>>,
    native_source: Option<Arc<NativeSource>>,
    source: String,
    repr: Option<crate::AutoRepresentation>,
    lang_tag: &'static str,
}

/// Wgpu compute server.
#[derive(Debug)]
pub struct WgpuServer<C: WgpuCompiler> {
    pub(crate) device: wgpu::Device,
    // A buffer that can be used to store stream id without extra allocations.
    streams_pool: Vec<StreamId>,
    /// The pipelines built so far, in front of the SPIR-V store when there is
    /// one.
    pipelines: CompilationCache<KernelId, PipelineEntry>,
    scheduler: SchedulerMultiStream<ScheduledWgpuBackend>,
    #[cfg(feature = "spirv")]
    pub(crate) spirv_cache: Option<Store<(u64, KernelCacheKey), cubecl_spirv::SpirvCacheEntry>>,
    #[cfg(feature = "spirv")]
    pub(crate) build_id: cubecl_common::hash::StableHash,
    pub compilation_options: WgpuCompilationOptions,
    pub(crate) backend: wgpu::Backend,
    pub(crate) utilities: Arc<ServerUtilities>,
    /// Reusable buffers for the cross-stream input bindings of each launch.
    shared_bindings_pool: LeasePool<SharedMemoryBindings>,
    /// Captured graphs owned by this server, keyed by the [`GraphId`] handed to
    /// the client. `end_capture` inserts, `replay` looks up, `graph_destroy`
    /// removes (dropping the [`WgpuGraph`] unpins the buffers it retained).
    graphs: HashMap<GraphId, WgpuGraph>,
    #[cfg(feature = "spirv")]
    native: crate::compute::native::NativeSets,
    /// Actor/source/compiler/device ABI seal every native hit must present.
    #[cfg(feature = "spirv")]
    native_seal: crate::compute::native::WgpuNativeSeal,
    /// A lazy pipeline was compiled outside the source-issued catalogue, so
    /// this actor is warmed and can never prepare one.
    lazy_pipeline_used: bool,
    /// A catalogue was retired; no later lazy compilation may replace it.
    #[cfg(feature = "spirv")]
    native_terminal: bool,
    preparing_native: bool,
    effective_storage_buffer_binding_size: u64,
    _compiler: PhantomData<C>,
}

impl<C: WgpuCompiler> ServerCommunication for WgpuServer<C> {}

impl<C: WgpuCompiler> WriteScoped for WgpuServer<C> {
    type Streams = SchedulerMultiStream<ScheduledWgpuBackend>;

    fn write_streams(&mut self) -> &mut Self::Streams {
        &mut self.scheduler
    }

    fn on_failure(&mut self, stream: StreamId, error: &ServerError) {
        // Measured per stream on this backend, so the scope's stream is the
        // one whose measurement a failure invalidates.
        self.scheduler.stream(&stream).profile_failure(error);
    }

    fn capturing(&mut self, stream: StreamId) -> Option<&mut StreamCapture> {
        Some(&mut self.scheduler.stream(&stream).capturing)
    }
}

impl<C: WgpuCompiler> WgpuServer<C> {
    /// Create a new server.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        compilation_options: WgpuCompilationOptions,
        device: wgpu::Device,
        queue: wgpu::Queue,
        tasks_max: usize,
        backend: wgpu::Backend,
        timing_method: TimingMethod,
        utilities: ServerUtilities,
        captures: DeviceCaptures,
        effective_storage_buffer_binding_size: u64,
    ) -> Self {
        #[cfg(feature = "spirv")]
        let adapter_info = device.adapter_info();
        let backend_scheduler = ScheduledWgpuBackend::new(
            device.clone(),
            queue.clone(),
            memory_properties,
            memory_config,
            timing_method,
            backend,
            tasks_max,
            utilities.logger.clone(),
            compilation_options.supports_vulkan_compiler,
            captures,
        );

        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        #[cfg(feature = "spirv")]
        let spirv_cache = compilation_store(
            "vulkan",
            format!("spirv_{}_{}", adapter_info.vendor, adapter_info.device),
        );

        // WGSL is compiled by the driver on every run, so without the SPIR-V
        // store there is nothing persisted for a switch to invalidate.
        #[cfg(feature = "spirv")]
        let pipelines = CompilationCache::mirroring(&spirv_cache);
        #[cfg(not(feature = "spirv"))]
        let pipelines = CompilationCache::unbound();

        #[cfg(feature = "spirv")]
        let build_id = cubecl_server::compiler::build_id_hash();
        #[cfg(feature = "spirv")]
        let native_seal = crate::compute::native::WgpuNativeSeal::new(
            utilities.service,
            utilities.properties_hash,
            build_id,
            backend,
            &compilation_options,
        );

        Self {
            compilation_options,
            streams_pool: Vec::new(),
            device,
            pipelines,
            scheduler: SchedulerMultiStream::new(
                utilities.logger.clone(),
                backend_scheduler,
                SchedulerMultiStreamOptions {
                    max_streams,
                    max_tasks: tasks_max,
                    strategy: SchedulerStrategy::Interleave,
                },
            ),
            #[cfg(feature = "spirv")]
            spirv_cache,
            #[cfg(feature = "spirv")]
            build_id,
            backend,
            utilities: Arc::new(utilities),
            shared_bindings_pool: LeasePool::with_capacity(tasks_max * max_streams as usize),
            graphs: HashMap::new(),
            #[cfg(feature = "spirv")]
            native: crate::compute::native::NativeSets::default(),
            #[cfg(feature = "spirv")]
            native_seal,
            lazy_pipeline_used: false,
            #[cfg(feature = "spirv")]
            native_terminal: false,
            preparing_native: false,
            effective_storage_buffer_binding_size,
            _compiler: PhantomData,
        }
    }

    fn prepare_bindings(
        &mut self,
        stream_id: StreamId,
        bindings: KernelArguments,
        compiler_info: CompilerInfo,
    ) -> Result<BindingsResource, IoError> {
        // Store all the resources we'll be using. This could be eliminated if
        // there was a way to tie the lifetime of the resource to the memory handle.
        let mut resources = Vec::with_capacity(bindings.resources.len());

        for resource in bindings.resources.into_iter() {
            match resource {
                KernelResource::Buffer(b) => {
                    // A recording guards the pages it was given once it seals.
                    self.scheduler
                        .stream(&stream_id)
                        .capturing
                        .touch(b.stream, b.memory.descriptor().location());
                    let stream = self.scheduler.stream(&b.stream);
                    let resource = stream.resource(b)?;
                    resources.push(resource);
                }
                KernelResource::TensorMap(_) => panic!("Tensor map not supported in wgpu"),
            }
        }

        Ok(BindingsResource {
            resources,
            info: bindings.info,
            compiler_info,
        })
    }

    /// Compile and validate one kernel exactly once. This step performs no
    /// driver module/pipeline creation, so an admission can charge the actual
    /// representation before any opaque driver resource exists.
    #[cfg(feature = "spirv")]
    fn compile_representation(
        &mut self,
        kernel: Box<dyn CubeKernel>,
        kernel_id: &KernelId,
        recording: &mut CompilationRecording,
    ) -> Result<CompiledRepresentation, LaunchError> {
        validate_cube_dim(&self.utilities.properties, kernel_id)?;
        validate_units(&self.utilities.properties, kernel_id)?;

        let definition = cubecl_core::define_kernel(&*kernel)?;
        recording.defined(&definition);

        let mut compiler = C::init(self.backend, &self.compilation_options);
        let mut compiled = compiler.compile_kernel(self, kernel, definition)?;

        if self.scheduler.logger.compilation_source_activated() {
            compiled.debug_info = Some(DebugInformation::new(
                compiler.lang_tag(),
                kernel_id.clone(),
            ));
        }
        self.scheduler.logger.log_compilation(&compiled);
        compiler.validate_ir(&compiled.repr, &self.utilities.properties)?;
        // The compiled kernel's per-buffer answer, before the repr is
        // consumed: what the write scope stages from.
        let io = compiled.io.take().map(Arc::from);
        let (compiler_info, auto_repr) = compiler.normalize_repr(compiled.repr);
        let native_source = match auto_repr.as_ref() {
            #[cfg(feature = "spirv")]
            Some(crate::AutoRepresentation::SpirV(kernel)) => Some(Arc::new(
                NativeSource::from_spirv(&compiled.entrypoint_name, kernel),
            )),
            _ => None,
        };
        Ok(CompiledRepresentation {
            entrypoint: compiled.entrypoint_name,
            compiler_info,
            io,
            native_source,
            source: compiled.source,
            repr: auto_repr,
            lang_tag: compiler.lang_tag(),
        })
    }

    /// Create the driver module and pipeline from an already-compiled and
    /// already-admitted representation. No compilation happens here.
    #[cfg(feature = "spirv")]
    fn create_compiled_pipeline(
        &self,
        compiled: &CompiledRepresentation,
        kernel_id: &KernelId,
        bindings: &KernelArguments,
        mode: ExecutionMode,
    ) -> Result<Arc<ComputePipeline>, LaunchError> {
        let repr = compiled.repr.as_ref().map(|repr| repr.as_ref());
        let module = self.create_module(
            &compiled.entrypoint,
            kernel_id.cube_dim.into(),
            ModuleSource::resolve(repr, compiled.lang_tag, &compiled.source)?,
            mode,
        )?;
        Ok(self.create_pipeline(&compiled.entrypoint, repr, module, bindings)?)
    }

    fn pipeline(
        &mut self,
        kernel: Box<dyn CubeKernel>,
        bindings: &KernelArguments,
    ) -> Result<PipelineEntry, LaunchError> {
        let kernel_id = kernel.id();
        let mode = kernel_id.mode;

        #[cfg(feature = "spirv")]
        match self.native.pipeline(&kernel_id, &self.native_seal) {
            Ok(Some(pipeline)) => return Ok(pipeline),
            Ok(None) => {}
            Err(error) => {
                return Err(LaunchError::Unknown {
                    reason: error.to_string(),
                    backtrace: BackTrace::capture(),
                });
            }
        }
        #[cfg(feature = "spirv")]
        if self.native_terminal {
            return Err(LaunchError::Unknown {
                reason: "wgpu retired native catalogue refuses later lazy compilation".into(),
                backtrace: BackTrace::capture(),
            });
        }
        #[cfg(feature = "spirv")]
        if self.native.sealed() || (!self.preparing_native && self.native.has_active()) {
            // The prepared catalogue is the only route after sealing or a
            // retained partial set. An unprepared key must refuse before any
            // lazy pipeline creation.
            return Err(LaunchError::Unknown {
                reason: "wgpu sealed native catalogue refuses unprepared kernel".into(),
                backtrace: BackTrace::capture(),
            });
        }

        if let Some(pipeline) = self.pipelines.get(&kernel_id) {
            return Ok(pipeline.clone());
        }

        let mut recording = CompilationRecording::new(&kernel_id);
        let cached = self.load_cached_pipeline(&kernel_id, bindings, mode)?;

        if let Some(Ok(pipeline)) = &cached {
            if !self.preparing_native {
                self.lazy_pipeline_used = true;
                self.pipelines.insert(kernel_id, pipeline.clone());
            }
            recording.loaded();
            return Ok(pipeline.clone());
        }

        let mut compiled = self.compile_representation(kernel, &kernel_id, &mut recording)?;
        let pipeline = self.create_compiled_pipeline(&compiled, &kernel_id, bindings, mode)?;
        if !self.preparing_native {
            self.lazy_pipeline_used = true;
            self.pipelines.insert(
                kernel_id.clone(),
                (
                    pipeline.clone(),
                    compiled.compiler_info,
                    compiled.io.clone(),
                    compiled.native_source.clone(),
                ),
            );
        }

        recording.source(&compiled.source);

        // Only a SPIR-V kernel is stored: any other build changes nothing.
        let stored = false;
        #[cfg(feature = "spirv")]
        let stored = match (cached, compiled.repr.take()) {
            (Some(Err(key)), Some(crate::AutoRepresentation::SpirV(kernel)))
                if !self.preparing_native =>
            {
                let cache = self.spirv_cache.as_mut().unwrap();
                store_compiled(
                    cache,
                    key,
                    cubecl_spirv::SpirvCacheEntry::new(compiled.entrypoint.clone(), kernel),
                )
            }
            _ => stored,
        };
        recording.compiled(stored);

        Ok((
            pipeline,
            compiled.compiler_info,
            compiled.io,
            compiled.native_source,
        ))
    }

    #[cfg(feature = "spirv")]
    pub(crate) fn native_environment(&self) -> crate::compute::native::WgpuNativeEnvironment {
        let properties = &self.utilities.properties;
        crate::compute::native::WgpuNativeEnvironment {
            service: self.utilities.service,
            name: self.utilities.name,
            adapter: properties.identity.name.clone(),
            fingerprint: properties.identity.fingerprint.clone(),
            backend: format!("{:?}", self.backend),
            compiler_options: format!("{:?}", self.compilation_options),
            supports_vulkan_compiler: self.compilation_options.supports_vulkan_compiler,
            properties_hash: self.utilities.properties_hash,
            effective_storage_buffer_binding_size: self.effective_storage_buffer_binding_size,
            max_bindings: properties.hardware.max_bindings as usize,
            max_cube_count: properties.hardware.max_cube_count.0,
            max_units_per_cube: properties.hardware.max_units_per_cube,
            max_shared_memory_size: properties.hardware.max_shared_memory_size,
            physical: properties
                .identity
                .physical
                .as_ref()
                .and_then(|physical| physical.pci_address)
                .map(|address| address.to_string()),
        }
    }

    #[cfg(feature = "spirv")]
    pub(crate) fn native_live(
        &self,
        key: u64,
        service: cubecl_common::device::ServiceId,
    ) -> Result<(), ServerError> {
        if self.utilities.service != service {
            return Err(ServerError::graph_state("wgpu native service differs"));
        }
        self.native.with_mut(key, |set| {
            if set.sealed && !set.retired() {
                set.admission().require_usable()?;
                Ok(())
            } else {
                Err(ServerError::graph_state("wgpu native set is not live"))
            }
        })
    }

    #[cfg(feature = "spirv")]
    pub(crate) fn prepare_native(
        &mut self,
        key: u64,
        kernels: Vec<Box<dyn CubeKernel>>,
        recipes: Vec<crate::compute::native::WgpuKernelRecipe>,
        admission: Arc<dyn crate::compute::native::WgpuNativeAdmission>,
    ) -> Result<crate::compute::native::WgpuNativeInventory, ServerError> {
        if kernels.is_empty() || kernels.len() > 1024 || recipes.len() != kernels.len() {
            return Err(ServerError::graph_state(
                "wgpu native catalogue extent or recipe pairing",
            ));
        }
        if self.native_terminal {
            return Err(ServerError::graph_state(
                "wgpu retired native catalogue refuses preparation",
            ));
        }
        // A catalogue may only be prepared by the cold actor that owns the
        // original source. A lazy pipeline or a live capture means this actor
        // already has aliases the catalogue does not own.
        if self.lazy_pipeline_used || !self.graphs.is_empty() {
            return Err(ServerError::graph_state(
                "wgpu native preparation requires a cold actor",
            ));
        }
        for stream in self.stream_ids() {
            if self
                .scheduler
                .stream(&stream)
                .capturing
                .device()
                .any_recording()
            {
                return Err(ServerError::graph_state(
                    "wgpu native preparation during graph capture",
                ));
            }
        }
        if self.native.has_active() {
            return Err(ServerError::graph_state(
                "wgpu native set/logical identity already present",
            ));
        }
        let environment = self.native_environment();
        if !environment.supports_vulkan_compiler || environment.backend != "Vulkan" {
            return Err(ServerError::graph_state(
                "wgpu native catalogue requires the actual Vulkan/SPIR-V compiler",
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(recipes.len());
        for (kernel, recipe) in kernels.iter().zip(recipes.iter()) {
            if kernel.id() != recipe.kernel_id || !seen.insert(recipe.kernel_id.clone()) {
                return Err(ServerError::graph_state(
                    "wgpu native recipe/kernel identity or uniqueness",
                ));
            }
        }
        if let Err(error) = admission.require_usable() {
            std::mem::forget(admission);
            return Err(error);
        }
        let preparation = match admission.reserve_preparation(kernels.len()) {
            Ok(preparation) => preparation,
            Err(error) => {
                std::mem::forget(admission);
                return Err(error);
            }
        };
        let owner = match admission.context_owner() {
            Ok(owner) => owner,
            Err(error) => {
                std::mem::forget(admission);
                return Err(error);
            }
        };
        let set = crate::compute::native::NativeSet::new(
            owner,
            preparation,
            admission.clone(),
            recipes.clone(),
            self.native_seal,
        );
        let abort_after_modules = admission.abort_after_modules();
        self.native.install(key, set)?;
        // The installed set owns partial state; an actor panic during the
        // kernel loop must still fence the original admission.
        let mut fence = crate::compute::native::FenceOnDrop::new(admission.clone());
        self.preparing_native = true;
        let result = (|| {
            let mut compiled_count = 0usize;
            for (kernel, recipe) in kernels.into_iter().zip(recipes.iter()) {
                if abort_after_modules == Some(compiled_count) {
                    return Err(ServerError::graph_state(
                        "wgpu native actual partial native preparation fault",
                    ));
                }
                let kernel_id = kernel.id();
                let mut recording = CompilationRecording::new(&kernel_id);
                let compiled = self.compile_representation(kernel, &kernel_id, &mut recording)?;
                // Actual representation, opaque driver cost and retained
                // source are charged before the driver module is created.
                let source = compiled.native_source.clone().ok_or_else(|| {
                    ServerError::graph_state("wgpu native compiled source is not SPIR-V")
                })?;
                let inventory = crate::compute::native::WgpuNativeInventory::from(source.as_ref());
                self.native.with_mut(key, |set| {
                    let admission = set.admission();
                    let owner = admission.reserve_module(inventory)?;
                    set.retain_module_owner(owner);
                    set.begin_module(recipe.clone(), source.clone(), inventory)
                })?;
                let entry = self.create_compiled_pipeline(
                    &compiled,
                    &kernel_id,
                    &KernelArguments::default(),
                    kernel_id.mode,
                )?;
                self.native.with_mut(key, |set| {
                    set.complete_module(
                        &kernel_id,
                        (
                            entry,
                            compiled.compiler_info,
                            compiled.io.clone(),
                            Some(source.clone()),
                        ),
                    )
                })?;
                compiled_count += 1;
            }
            self.native.with_mut(key, |set| {
                if set.module_count() != recipes.len() {
                    return Err(ServerError::graph_state(
                        "wgpu native catalogue module count differs",
                    ));
                }
                set.seal()
            })
        })();
        self.preparing_native = false;
        if result.is_err() {
            if self
                .native
                .with_mut(key, |set| {
                    set.admission().fence_unknown();
                    Ok(())
                })
                .is_ok()
            {
                fence.disarm();
            }
        } else {
            fence.disarm();
        }
        result
    }

    #[cfg(feature = "spirv")]
    pub(crate) fn retire_native(
        &mut self,
        key: u64,
    ) -> Result<crate::compute::native::WgpuNativeInventory, ServerError> {
        let admission = self.native.with_mut(key, |set| {
            if set.sealed && !set.retired() {
                Ok(set.admission())
            } else {
                Err(ServerError::graph_state(
                    "wgpu native retirement before a sealed catalogue",
                ))
            }
        })?;
        // Any refusal or failure after this point leaves aliases or an
        // unproven completion boundary; fence the original domain and retain
        // the set rather than crediting resources back. A panic in the actor
        // unwinds through this guard.
        let mut fence = crate::compute::native::FenceOnDrop::new(admission.clone());
        let result = (|| {
            admission.require_usable()?;
            if !self.graphs.is_empty() {
                return Err(ServerError::graph_state(
                    "wgpu native retirement with a live captured graph",
                ));
            }
            for stream in self.stream_ids() {
                if self
                    .scheduler
                    .stream(&stream)
                    .capturing
                    .device()
                    .any_recording()
                {
                    return Err(ServerError::graph_state(
                        "wgpu native retirement during graph capture",
                    ));
                }
            }
            for stream in self.stream_ids() {
                self.scheduler.execute_streams(vec![stream]);
                let stream_ref = self.scheduler.stream(&stream);
                stream_ref.flush(stream)?;
            }
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|error| ServerError::graph_state(format!("wgpu native drain: {error}")))?;
            admission.require_usable()?;
            Ok(())
        })();
        if let Err(error) = result {
            admission.fence_unknown();
            fence.disarm();
            return Err(error);
        }
        // Only one sealed set may exist on this actor. Its completed cache
        // entries are dropped with the set, never retained as a lazy cache.
        self.pipelines = CompilationCache::mirroring(&self.spirv_cache);
        let mut set = self.native.take(key)?;
        set.retire();
        self.native_terminal = true;
        fence.disarm();
        Ok(set.inventory())
    }
}

impl<C: WgpuCompiler> Server for WgpuServer<C> {
    fn logger(&self) -> Arc<ServerLogger> {
        self.scheduler.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities> {
        self.utilities.clone()
    }

    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        // TODO: Check if using a staging buffer is useful here.
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    fn initialize_memory(
        &mut self,
        memory: ManagedMemoryHandle,
        size: u64,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let mut relocating = self.scheduler.relocating(stream_id);
        relocating.relocate_when_wanted();
        let (stream, failures) = self.scheduler.stream_and_failures(&stream_id);
        let update = stream.capturing.page_update();
        let reserved = match stream.memory.reserve(size, update, failures) {
            Ok(reserved) => reserved,
            // The recording now misses whatever this memory was for, and
            // `end_capture` reports that: the handle stays unbound, and
            // whatever uses it belongs to a recording that will not seal.
            Err(err @ IoError::PageUpdateForbidden { .. }) => {
                stream.capturing.fail(err.into());
                return Ok(());
            }
            Err(err) => {
                // Nothing to allocate: the buffer carries the error instead.
                let err = ServerError::from(err);
                failures.fail_unallocated(&memory, err.clone());
                return Err(err);
            }
        };
        if let Err(err) = stream.memory.bind(reserved, memory.clone(), 0, failures) {
            let err = ServerError::from(err);
            failures.fail_unallocated(&memory, err.clone());
            return Err(err);
        }
        Ok(())
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        // A read is a host sync: it cannot be recorded, and the recorded work
        // has not executed, so there is nothing meaningful to read anyway.
        if let Err(err) = self
            .scheduler
            .stream(&stream_id)
            .reject_while_recording("read")
        {
            return Box::pin(async move { Err(err) });
        }

        // Buffers another stream wrote are only as good as the work that wrote
        // them; see `StreamPool::ensure_written`. The reader's own errors are
        // surfaced by `read_resources`' flush further down.
        if let Err(err) = self
            .scheduler
            .ensure_written(descriptors.iter().map(|d| &d.handle))
        {
            return Box::pin(async move { Err(err) });
        }

        let mut streams = vec![stream_id];
        let mut resources = Vec::with_capacity(descriptors.len());
        for desc in descriptors {
            if contiguous_strides(&desc.shape) != desc.strides {
                return Box::pin(async {
                    Err(IoError::UnsupportedStrides {
                        backtrace: BackTrace::capture(),
                    }
                    .into())
                });
            }
            if !streams.contains(&desc.handle.stream) {
                streams.push(desc.handle.stream);
            }
            let stream = self.scheduler.stream(&desc.handle.stream);
            let resource = match stream.resource(desc.handle) {
                Ok(val) => val,
                Err(err) => return Box::pin(async move { Err(err.into()) }),
            };
            resources.push((resource, desc.shape, desc.elem_size));
        }

        self.scheduler.execute_streams(streams);

        let stream = self.scheduler.stream(&stream_id);
        stream.read_resources(resources, stream_id)
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        // Writes go on the queue, not the encoder — they cannot be recorded
        // into a software graph (v1; CUDA records them as memcpy nodes).
        //
        // Rejected lazily. When the caller is the stream recording the
        // capture, the refusal dooms its `end_capture` rather than handing
        // back a graph missing an operation. When it is a neighbour sharing
        // the pooled stream, the write was never going into anyone's graph and
        // the taint on its own destinations is the whole report — dooming a
        // capture on it would charge one stream's window to another.
        {
            let recording = self
                .scheduler
                .stream(&stream_id)
                .reject_while_recording("write");
            if let Err(err) = recording {
                // Nothing is copied, so every destination this call was given is
                // left as it was — taint them, or a read of one on another
                // logical stream finds no failure to fail on and copies stale
                // bytes.
                self.scheduler.taint(
                    err.clone(),
                    descriptors.iter().map(|(desc, _)| &desc.handle),
                );
                // The owner's own write dooms its capture: the recording is
                // missing that operation and must not seal. A neighbour's
                // refusal is not the capture's failure — the taint on its
                // destinations is the whole report.
                let stream = self.scheduler.stream(&stream_id);
                if stream.capturing.owner() == Some(stream_id) {
                    stream.capturing.fail(err);
                }
                return;
            }
        }
        for (desc, data) in descriptors {
            // Each copy runs in its own scope over its destination: the write
            // that lands fills it, which is what releases an earlier
            // failure's hold on it — a caller recovers by writing from the
            // host as much as by relaunching — and a failure leaves it as it
            // was, which is what a later read of it has to fail on. The scope
            // queues failures on the caller's stream, the one that flushes
            // them, even though the resource is resolved on the stream that
            // owns the handle.
            let mut written = self.write_set();
            written.push(desc.handle.clone());
            ExecuteScope::over(self, stream_id, written).execute(|server| {
                if contiguous_strides(&desc.shape) != desc.strides {
                    return Err(ServerError::Io(IoError::UnsupportedStrides {
                        backtrace: BackTrace::capture(),
                    }));
                }

                // The write is registered on the caller, so name the
                // stream that owns the handle as an argument: its queued
                // work has to land before this write overwrites the same
                // memory.
                let owner = desc.handle.stream;
                let handle = desc.handle.clone();
                let stream = server.scheduler.stream(&owner);
                let resource = stream.resource(desc.handle).map_err(ServerError::Io)?;
                let task = ScheduleTask::Write {
                    data,
                    buffer: resource,
                    handle,
                };

                server.scheduler.register(stream_id, task, &[owner]);
                Ok(())
            });
        }
    }

    fn check(
        &mut self,
        handles: Vec<BufferBinding>,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        self.scheduler.ensure_written(handles.iter())
    }

    unsafe fn launch(
        &mut self,
        kernel: Box<dyn CubeKernel>,
        count: CubeCount,
        args: KernelArguments,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) {
        // Compilation comes first — memoized, so a launch after the first
        // pays a map lookup — because the write scope stages what the
        // compiled kernel says it writes. A kernel that fails to compile has
        // no IR and no compiled answer, so the caller's declared IO decides:
        // only the declared outputs are left carrying the failure, never the
        // buffers the kernel was only going to read — tainting those would
        // refuse every later launch that shares them, an autotune sweep
        // above all.
        //
        // A dry run stages none either way. It was never going to write, so a
        // failure in it leaves nothing stale, and tainting its buffers would
        // fail unrelated reads of memory the run deliberately left alone.
        let kernel_id = kernel.id();
        let (pipeline, compiler_info, io, _native_source) = match self.pipeline(kernel, &args) {
            Ok(val) => val,
            Err(err) => {
                let error = ServerError::Launch(err);
                self.scheduler.stream(&stream_id).profile_failure(&error);
                if !launch_mode.is_skipped() {
                    let mut written = self.write_set();
                    written.extend(args.buffers_written(None).cloned());
                    failed_writing(self, stream_id, written, error);
                }
                return;
            }
        };
        if launch_mode.is_skipped() {
            return;
        }

        // Skip, do not taint: a launch whose input cannot be trusted does not
        // run. Running it is not merely wasted device time — a buffer holding
        // garbage can be read as a dynamic cube count or as gather indices,
        // scattering into memory that carried no failure at all. The outputs
        // take the failure that stopped the launch, exactly as a failed
        // launch's would, so a read downstream fails on the root cause.
        //
        // Except while this stream records a graph: skipping would seal a
        // recording missing an operation, and the replay contract has the
        // caller write fresh inputs before each replay — clearing the very
        // claim that would explain the hole. A doomed capture is refused at
        // `end_capture` instead.
        //
        // The scope claims what the launch writes until the body proves the
        // work enqueued, so a failure — or a panic — anywhere in it leaves a
        // read of those buffers failing on the error rather than copying
        // bytes nothing wrote.
        let mut written = self.write_set();
        written.extend(args.buffers_written(io.as_deref()).cloned());
        // A dynamic count travels outside `resources`, so `buffers_read`
        // never names it — yet the indirect dispatch reads it as its grid
        // dimensions, which is exactly the garbage-as-cube-count read the
        // skip exists to prevent.
        let count_read = match &count {
            CubeCount::Dynamic(binding) => Some(binding),
            CubeCount::Static(..) => None,
        };
        ExecuteScope::launching(
            self,
            kernel_id,
            stream_id,
            args.buffers_read(io.as_deref()).chain(count_read),
            written,
        )
        .execute(|server| {
            server.streams_pool.clear();
            // Reuse a pooled buffer to avoid allocating on every launch; it returns to the pool
            // automatically when the guard drops.
            let mut shared_inputs = server.shared_bindings_pool.acquire();
            // Pin the memory of every input that lives on another stream (released in `WgpuStream::flush`).
            args.resources.iter().for_each(|resource| match resource {
                KernelResource::Buffer(b) => {
                    server.streams_pool.push(b.stream);
                    if b.stream != stream_id {
                        shared_inputs.push(b.memory.clone());
                    }
                }
                KernelResource::TensorMap(_) => {
                    panic!("Tensor maps not supported in WGPU")
                }
            });

            let resources = server
                .prepare_bindings(stream_id, args, compiler_info)
                .map_err(ServerError::Io)?;

            let task = ScheduleTask::Execute {
                pipeline,
                count,
                resources,
                shared_inputs,
            };

            server
                .scheduler
                .register(stream_id, task, &server.streams_pool);
            Ok(())
        });
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);

        let stream = self.scheduler.stream(&stream_id);

        stream.flush(stream_id)
    }

    /// Returns the total time of GPU work this sync completes.
    fn sync(
        &mut self,
        handles: Vec<BufferBinding>,
        stream_id: StreamId,
    ) -> DynFut<Result<(), ServerError>> {
        if let Err(err) = self
            .scheduler
            .stream(&stream_id)
            .reject_while_recording("sync")
        {
            return Box::pin(async move { Err(err) });
        }
        // The claim check a read would have made, without the read; claims
        // are set at enqueue time, so they are already in place.
        if let Err(err) = self.scheduler.ensure_written(handles.iter()) {
            return Box::pin(async move { Err(err) });
        }
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.sync(stream_id)
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        // Recorded launches do not execute, so a profile of the window would
        // measure nothing.
        self.scheduler
            .stream(&stream_id)
            .reject_while_recording("start_profile")?;
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.start_profile(stream_id)
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.end_profile(token, stream_id)
    }

    fn abandon_profile(&mut self, stream_id: StreamId, token: ProfilingToken) {
        // No `execute_streams`: the default reaches this through `end_profile`,
        // which has to flush what it is about to measure. An abandon measures
        // nothing, so it leaves the stream's queued work where it found it.
        self.scheduler.stream(&stream_id).abandon_profile(token);
    }

    fn memory_report(&mut self, stream_id: StreamId) -> Option<StreamMemoryReport> {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        Some(StreamMemoryReport {
            stream: stream_id,
            pools: stream.memory.memory_report(),
            auxiliary: stream.auxiliary.report(),
        })
    }

    fn stream_ids(&self) -> Vec<StreamId> {
        self.scheduler.stream_ids().collect()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.scheduler.execute_streams(vec![stream_id]);
        self.scheduler
            .relocating(stream_id)
            .refuse_while_recording()?;
        let stream = self.scheduler.stream(&stream_id);
        // The info cache's buffers are live slices in the uniforms pool; an
        // explicit cleanup exists to leave the pools empty, so every entry not
        // pinned by a live graph goes too (entries are recreated on their next
        // miss).
        stream.info_cache.clear_unpinned();
        self.scheduler.relocating(stream_id).reclaim()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);
        stream.memory.mode(mode);
    }

    fn graph_prepare(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        // Drain queued tasks first so pre-capture work is not attributed to
        // the capture window.
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        // The non-`NoCapture` state isolates this stream in the scheduler
        // (see `requires_isolation`) until the capture ends.
        stream.capturing.prepare(stream_id)?;

        // The pages the recording touches are guarded for the graph's life,
        // and a guarded page is never relocated: empty every stream's outdated
        // pools now, since the recording may touch memory any stream owns.
        self.scheduler
            .relocating(stream_id)
            .relocate_device(RelocationReason::Capture);
        Ok(())
    }

    fn begin_capture(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        // Materialize the warmup work queued in the scheduler before the
        // recording window opens.
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        stream.capturing.begin()?;

        // Submit the warmup work and surface its failure now, so a warmup
        // failure is reported here — where the diagnostic points at the cause
        // — instead of dooming `end_capture` later.
        let stream = self.scheduler.stream(&stream_id);
        if let Err(err) = stream.flush(stream_id) {
            // The capture never opened: return to `NoCapture`, so a failed
            // `start_capture` leaves the stream fully usable and
            // re-capturable.
            stream.info_cache.capture_discard();
            stream.capturing.abort();
            return Err(err);
        }
        Ok(())
    }

    fn end_capture(&mut self, stream_id: StreamId) -> Result<GraphId, ServerError> {
        // Materialize the recorded launches still queued in the scheduler.
        self.scheduler.execute_streams(vec![stream_id]);
        let stream = self.scheduler.stream(&stream_id);

        // The capture is over even on the failure path below, so an error here
        // doesn't leave the stream stuck in capture/persistent state — and it
        // is over for a caller that does not own the window too, since that is
        // a window nobody is coming back to close. Only its owner gets a graph
        // out of it, and the errors raised inside belong to that owner rather
        // than to whoever happens to be flushing.
        let outcome = match stream.capturing.end(stream_id) {
            Ok(outcome) => outcome,
            Err(err) => {
                // A capture prepared but never opened still holds the stream,
                // and a `graph_prepare` retry is refused while it does.
                // Closing is the only call the caller has left — a warmup that
                // failed never reaches `start_capture` — so a close from
                // `Prepare` unwinds it, as `begin_capture` does when the
                // warmup flush fails, instead of leaving it held forever.
                if stream.capturing.is_active() {
                    stream.info_cache.capture_discard();
                    stream.capturing.abort();
                }
                return Err(err);
            }
        };
        let recording = stream.take_recording();
        // The memory the recorded launches write. A graph that seals answers
        // for it on a failed replay; one that does not is answered for here,
        // since those launches never ran and now never will.
        let written = stream.capturing.take_recorded();
        let touched = stream.capturing.take_touched();

        // A failure raised during the window — a rejected write, a failed or
        // skipped launch — means the recording is missing an operation:
        // reject the capture rather than hand back a graph that silently
        // skips work. The window is doomed from the moment one lands, so what
        // is read here arose inside it and nowhere else.
        let doomed = stream.capturing.take_failure().map(|reason| {
            ServerError::graph_state(format!(
                "an operation inside the capture window failed, so the recording is missing \
                 an operation and cannot seal: {reason}"
            ))
        });
        let discarded = match outcome.is_abandoned() {
            true => Some(outcome.abandoned_error(stream_id, doomed)),
            false => doomed,
        };
        if let Some(err) = discarded {
            stream.info_cache.capture_discard();
            // The recording is thrown away, so the launches in it never run:
            // every buffer they were given is left as it was. The caller gets
            // the error below; the taint is what makes a read of one of those
            // buffers fail on some other stream, which heard nothing.
            self.scheduler.taint(err.clone(), written.iter());
            return Err(err);
        }

        let id = GraphId::new();
        // Seal the info-cache entries this capture pinned under the graph's
        // id, so `graph_destroy` can release them later.
        stream.info_cache.capture_commit(id);
        // Guard every page the recording touched, on whichever stream owns
        // it, so nothing on them moves or is reused for the graph's lifetime.
        let guards = touched
            .iter()
            .filter_map(|page| {
                self.scheduler
                    .stream(&page.stream)
                    .memory
                    .guard(page.location)
            })
            .collect();
        self.graphs.insert(
            id,
            WgpuGraph {
                tasks: recording.tasks,
                _guards: guards,
                _uniforms: recording.uniform_pins,
                _shared: recording.shared,
                written,
            },
        );
        Ok(id)
    }

    fn replay(&mut self, graph: GraphId, stream_id: StreamId) -> Result<(), ServerError> {
        // Order the replay after previously queued work on this stream.
        self.scheduler.execute_streams(vec![stream_id]);

        // A use-after-free in the caller's own code, with the caller standing
        // right there, so it is returned. Nothing to taint either — the
        // graph is gone, and with it the record of which buffers its launches
        // would have written.
        let Some(wgpu_graph) = self.graphs.get(&graph) else {
            return Err(ServerError::graph_state(
                "replay was given an unknown or already-destroyed graph",
            ));
        };

        // A replay writes the buffers its recorded launches were given, so it
        // takes the same scope over that write set and settles it: a failed
        // enqueue leaves them carrying the failure, and the next replay that
        // lands releases the claim. Without the settle one transient failure
        // would leave the graph's buffers unreadable forever — the graph
        // retains their handles, so none of the shedding paths can ever fire
        // for them, and the graph itself is the only thing that writes them.
        let recorded = wgpu_graph.written.clone();
        let mut written = self.write_set();
        written.extend(recorded);
        ExecuteScope::over(self, stream_id, written)
            .execute(|server| {
                server
                    .scheduler
                    .stream(&stream_id)
                    .reject_while_recording("replay")?;
                let wgpu_graph = server
                    .graphs
                    .get(&graph)
                    .expect("checked above; nothing in the scope removes graphs");
                let stream = server.scheduler.stream(&stream_id);
                stream.replay_graph(wgpu_graph);
                Ok(())
            })
            .into_result()
    }

    fn graph_destroy(&mut self, graph: GraphId, stream_id: StreamId) {
        // No-op for an unknown id (e.g. a double release). The graph is held
        // until the end of this function, so its pins outlive the flush below.
        let Some(wgpu_graph) = self.graphs.remove(&graph) else {
            return;
        };
        let stream = self.scheduler.stream(&stream_id);
        // Submit any replay still sitting in the encoder before the pins drop:
        // a `queue.write_buffer` onto a reclaimed slice runs at the *next*
        // submit, ahead of everything already in the encoder, so it would reach
        // the GPU before the still-unsubmitted replay that reads it. The `Write`
        // path flushes on its own account, so this covers the writes that do
        // not — the uniform uploads in `create_uniform`/`info_uniform`. Once the
        // replay is submitted, queue ordering makes releasing the slices safe
        // with no host sync, unlike CUDA.
        stream.submit();
        // Release the info-cache entries this graph pinned; entries no other
        // live graph still pins are dropped, freeing their buffers.
        stream.info_cache.graph_release(graph);
        drop(wgpu_graph);
    }
}

pub(crate) fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = strides![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl<C: WgpuCompiler> ServerStorage for WgpuServer<C> {
    type Storage = WgpuStorage;

    fn get_resource(
        &mut self,
        binding: BufferBinding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<WgpuResource>, ServerError> {
        // The same claim check a read makes: a buffer a failed launch never
        // filled reports the failure rather than handing back a pointer to
        // whatever was there before.
        self.scheduler.ensure_written([&binding].into_iter())?;
        let mut streams = vec![stream_id];
        if binding.stream != stream_id {
            streams.push(binding.stream);
        }
        self.scheduler.execute_streams(streams);
        let stream = self.scheduler.stream(&binding.stream);
        Ok(stream.memory.managed_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        )?)
    }
}
