//! Original failure tracking over host BytesStorage. No CUDA actor or launch.
use super::*;
use cubecl_common::device::{DeviceId, ServiceId};
use cubecl_server::logging::ServerLogger;
use cubecl_server::memory_management::{
    ErrorGraph, FailureId, MemoryManagement, MemoryManagementOptions, PageUpdate,
};
use cubecl_server::storage::BytesStorage;
use cubecl_server::stream::{Failures, ScopedOutcome, StreamFactory, StreamMemory, StreamPool};
use std::cell::Cell;

struct HostStream {
    memory: MemoryManagement<BytesStorage>,
}
impl std::fmt::Debug for HostStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostStream").finish()
    }
}
impl StreamMemory for HostStream {
    fn failure(&self, binding: &BufferBinding) -> Option<FailureId> {
        self.memory.failure(&binding.memory, binding.range())
    }
    fn taint(&mut self, binding: &BufferBinding, failure: FailureId, graph: &mut ErrorGraph) {
        self.memory
            .taint(&binding.memory, binding.range(), failure, graph)
    }
    fn written(&mut self, binding: &BufferBinding, graph: &mut ErrorGraph) {
        self.memory.written(&binding.memory, binding.range(), graph)
    }
}
struct HostFactory {
    properties: MemoryDeviceProperties,
    logger: Arc<ServerLogger>,
}
impl StreamFactory for HostFactory {
    type Stream = HostStream;

    fn create(&mut self) -> Result<HostStream, ServerError> {
        Ok(HostStream {
            memory: MemoryManagement::from_configuration(
                BytesStorage::default(),
                &self.properties,
                MemoryConfiguration::ExclusivePages,
                self.logger.clone(),
                MemoryManagementOptions::new("cubin failure tracking test"),
            ),
        })
    }
}
struct HostTracking {
    pool: StreamPool<HostFactory>,
    failures: Failures,
}
impl FailureStore for HostTracking {
    type Factory = HostFactory;

    fn split(&mut self) -> (&mut StreamPool<HostFactory>, &mut Failures) {
        (&mut self.pool, &mut self.failures)
    }
    fn parts(&self) -> (&StreamPool<HostFactory>, &Failures) {
        (&self.pool, &self.failures)
    }
}
impl WriteScoped for HostTracking {
    type Streams = Self;

    fn write_streams(&mut self) -> &mut Self {
        self
    }
}
impl HostTracking {
    fn new() -> Self {
        let logger = Arc::new(ServerLogger::default());
        Self {
            pool: StreamPool::new(
                HostFactory {
                    properties: MemoryDeviceProperties::new(16 * 1024, 16),
                    logger: logger.clone(),
                },
                1,
                0,
            ),
            failures: Failures::new(logger),
        }
    }

    fn buffer(&mut self) -> Handle {
        let stream = StreamId::current();
        let handle = Handle::new(ServiceId::of::<Self>(DeviceId::new(0, 0)), stream, 64);
        let memory = &mut self.pool.get_mut(&stream).memory;
        let reserved = memory
            .reserve(64, PageUpdate::Allow, self.failures.graph_mut())
            .unwrap();
        memory
            .bind(
                reserved,
                handle.memory.clone(),
                0,
                self.failures.graph_mut(),
            )
            .unwrap();
        handle
    }
}

#[test]
fn read_write_binding_checks_prior_contents_before_its_own_write_claim() {
    let mut host = HostTracking::new();
    let handle = host.buffer();
    let binding = handle.binding();
    let ran = Cell::new(false);
    let outcome = launch_scope(
        &mut host,
        StreamId::current(),
        7,
        [&binding].into_iter(),
        vec![binding.clone()],
    )
    .execute(|_| {
        ran.set(true);
        Ok::<_, ServerError>(())
    });
    assert!(matches!(outcome, ScopedOutcome::Executed(())));
    assert!(ran.get());
    host.ensure_written([&binding].into_iter()).unwrap();
    assert!(host.failures.graph().is_empty());
}

#[test]
fn failed_prior_read_skips_and_propagates_to_every_output_without_new_failure() {
    let mut host = HostTracking::new();
    let handles = [host.buffer(), host.buffer()];
    let rw = handles[0].clone().binding();
    let output = handles[1].clone().binding();
    let _ = ExecuteScope::over(&mut host, StreamId::current(), vec![rw.clone()])
        .execute(|_| Err::<(), _>(failure("original producer failed")));
    let failures = host.failures.graph().len();
    let ran = Cell::new(false);
    let outcome = launch_scope(
        &mut host,
        StreamId::current(),
        7,
        [&rw].into_iter(),
        vec![rw.clone(), output.clone()],
    )
    .execute(|_| {
        ran.set(true);
        Ok::<_, ServerError>(())
    });
    assert!(matches!(outcome, ScopedOutcome::Skipped));
    assert!(!ran.get());
    assert_eq!(host.failures.graph().len(), failures);
    for binding in [&rw, &output] {
        let error = host.ensure_written([binding].into_iter()).unwrap_err();
        assert!(error.to_string().contains("original producer failed"));
    }
}

#[test]
fn argument_resource_or_launch_failure_taints_writes_and_preserves_read_only_input() {
    for reason in ["argument refusal", "resource refusal", "launch refusal"] {
        let mut host = HostTracking::new();
        let handles = [host.buffer(), host.buffer(), host.buffer()];
        let input = handles[0].clone().binding();
        let rw = handles[1].clone().binding();
        let output = handles[2].clone().binding();
        let outcome = launch_scope(
            &mut host,
            StreamId::current(),
            9,
            [&input, &rw].into_iter(),
            vec![rw.clone(), output.clone()],
        )
        .execute(|_| Err::<(), _>(failure(reason)));
        assert!(matches!(outcome, ScopedOutcome::Failed(_)));
        host.ensure_written([&input].into_iter()).unwrap();
        for binding in [&rw, &output] {
            let error = host.ensure_written([binding].into_iter()).unwrap_err();
            assert!(error.to_string().contains(reason));
        }
    }
}

#[test]
fn panic_inside_launch_scope_leaves_read_write_output_claimed() {
    let mut host = HostTracking::new();
    let handle = host.buffer();
    let binding = handle.binding();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        launch_scope(
            &mut host,
            StreamId::current(),
            11,
            [&binding].into_iter(),
            vec![binding.clone()],
        )
        .execute(|_| -> Result<(), ServerError> { panic!("original scope unwind") });
    }));
    assert!(outcome.is_err());
    assert!(host.ensure_written([&binding].into_iter()).is_err());
    assert_eq!(host.failures.graph().len(), 1);
}

#[test]
fn write_prefix_has_no_prior_dependency_but_read_write_still_skips() {
    use crate::owned_command::CubinIo;
    for io in [CubinIo::Write, CubinIo::ReadWrite] {
        let mut host = HostTracking::new();
        let input = host.buffer().binding();
        let workspace = host.buffer().binding();
        let output = host.buffer().binding();
        let _ = ExecuteScope::over(&mut host, StreamId::current(), vec![workspace.clone()])
            .execute(|_| Err::<(), _>(failure("old workspace producer failed")));
        let bindings = [input.clone(), workspace.clone(), output.clone()];
        let access = [CubinIo::Read, io, CubinIo::Write];
        let checkpoints = Cell::new(0);
        let outcome = launch_scope(
            &mut host,
            StreamId::current(),
            13,
            bindings
                .iter()
                .zip(access)
                .filter(|(_, io)| io.reads())
                .map(|(b, _)| b),
            vec![workspace.clone(), output.clone()],
        )
        .execute(|_| {
            // Original scope only: no native clear or launch in this host test.
            checkpoints.set(1); // Prefix.
            checkpoints.set(2); // Kernel may read its own newly written bytes.
            Ok::<_, ServerError>(())
        });
        host.ensure_written([&input].into_iter()).unwrap();
        if io == CubinIo::Write {
            assert!(matches!(outcome, ScopedOutcome::Executed(())));
            assert_eq!(checkpoints.get(), 2);
            host.ensure_written([&workspace, &output].into_iter())
                .unwrap();
        } else {
            assert!(matches!(outcome, ScopedOutcome::Skipped));
            assert_eq!(checkpoints.get(), 0);
            for binding in [&workspace, &output] {
                let error = host.ensure_written([binding].into_iter()).unwrap_err();
                assert!(error.to_string().contains("old workspace producer failed"));
            }
        }
    }
}

#[test]
fn prefix_or_post_prefix_launch_failure_claims_workspace_and_output() {
    for failed_stage in [1, 2] {
        let mut host = HostTracking::new();
        let input = host.buffer().binding();
        let workspace = host.buffer().binding();
        let output = host.buffer().binding();
        let prefix = binding_region(&workspace, 0..32).unwrap();
        let tail = binding_region(&workspace, 32..64).unwrap();
        let _ = ExecuteScope::over(&mut host, StreamId::current(), vec![tail.clone()])
            .execute(|_| Err::<(), _>(failure("untouched tail failed")));
        let checkpoints = Cell::new(0);
        let outcome = launch_scope(
            &mut host,
            StreamId::current(),
            17,
            [&input].into_iter(),
            vec![prefix.clone(), output.clone()],
        )
        .execute(|_| {
            // Checkpoints exercise the actual original write scope, not CUDA.
            checkpoints.set(1);
            if failed_stage == 1 {
                return Err(failure("workspace prefix failed"));
            }
            checkpoints.set(2);
            Err::<(), _>(failure("kernel failed after workspace prefix"))
        });
        assert!(matches!(outcome, ScopedOutcome::Failed(_)));
        assert_eq!(checkpoints.get(), failed_stage);
        host.ensure_written([&input].into_iter()).unwrap();
        for binding in [&prefix, &output] {
            let error = host.ensure_written([binding].into_iter()).unwrap_err();
            assert!(error.to_string().contains(if failed_stage == 1 {
                "workspace prefix failed"
            } else {
                "kernel failed after workspace prefix"
            }));
        }
        let error = host.ensure_written([&tail].into_iter()).unwrap_err();
        assert!(error.to_string().contains("untouched tail failed"));
    }
}

#[test]
fn validated_prefix_ignores_only_covered_reads_and_preserves_tail_failure() {
    use crate::owned_command::{
        CubinBinding, CubinIo, CubinLaunchAttributes, CubinParameter, CubinPointerFixup,
        CubinWorkspaceClear,
    };
    let manifest = CubinKernelManifest {
        entry: "compound_scope_test".into(),
        binary_sm: 80,
        grid: [1, 1, 1],
        block: [32, 1, 1],
        dynamic_shared_bytes: 0,
        attributes: CubinLaunchAttributes::None,
        bindings: [CubinIo::Read, CubinIo::ReadWrite, CubinIo::Write]
            .into_iter()
            .map(|io| CubinBinding {
                bytes: 64,
                alignment: 16,
                io,
            })
            .collect(),
        parameters: vec![CubinParameter {
            offset: 0,
            template: vec![0; 24],
        }],
        fixups: [
            (0, 0, CubinIo::Read, 64),
            (8, 1, CubinIo::ReadWrite, 32),
            (16, 2, CubinIo::Write, 64),
        ]
        .into_iter()
        .map(|(field, binding, io, extent)| CubinPointerFixup {
            parameter: 0,
            byte_offset: field,
            binding,
            binding_offset: 0,
            extent,
            alignment: 16,
            io,
        })
        .collect(),
    };
    let clear = CubinWorkspaceClear::zero32(1).unwrap();
    clear.validate(&manifest).unwrap();
    for failed_input in [false, true] {
        let mut host = HostTracking::new();
        let input = host.buffer().binding();
        let workspace = host.buffer().binding();
        let output = host.buffer().binding();
        let prefix = binding_region(&workspace, 0..32).unwrap();
        let tail = binding_region(&workspace, 32..64).unwrap();
        for (binding, reason) in [(&prefix, "old prefix failed"), (&tail, "old tail failed")] {
            let _ = ExecuteScope::over(&mut host, StreamId::current(), vec![binding.clone()])
                .execute(|_| Err::<(), _>(failure(reason)));
        }
        if failed_input {
            let _ = ExecuteScope::over(&mut host, StreamId::current(), vec![input.clone()])
                .execute(|_| Err::<(), _>(failure("original input failed")));
        }
        let native = [input.clone(), workspace.clone(), output.clone()];
        let mut written = vec![output.clone()];
        for range in clear.write_regions(&manifest).unwrap() {
            written.push(binding_region(&workspace, range).unwrap());
        }
        let ran = Cell::new(false);
        let outcome = launch_scope(
            &mut host,
            StreamId::current(),
            19,
            native
                .iter()
                .zip(&manifest.bindings)
                .enumerate()
                .filter(|(i, (_, b))| b.io.reads() && *i != clear.binding())
                .map(|(_, (b, _))| b),
            written,
        )
        .execute(|_| {
            ran.set(true);
            Ok::<_, ServerError>(())
        });
        assert_eq!(ran.get(), !failed_input);
        if failed_input {
            assert!(matches!(outcome, ScopedOutcome::Skipped));
            for binding in [&prefix, &output] {
                let error = host.ensure_written([binding].into_iter()).unwrap_err();
                assert!(error.to_string().contains("original input failed"));
            }
        } else {
            assert!(matches!(outcome, ScopedOutcome::Executed(())));
            host.ensure_written([&prefix, &output].into_iter()).unwrap();
        }
        let error = host.ensure_written([&tail].into_iter()).unwrap_err();
        assert!(error.to_string().contains("old tail failed"));
    }
}

#[test]
fn workspace_claim_ranges_cannot_escape_the_original_view() {
    let mut host = HostTracking::new();
    let binding = host.buffer().binding();
    for range in [0..0, 32..16, 0..65, u64::MAX..u64::MAX] {
        assert!(binding_region(&binding, range).is_err());
    }
    let mut shifted = binding.clone();
    shifted.offset_start = Some(16);
    shifted.offset_end = Some(16);
    assert_eq!(binding_region(&shifted, 0..32).unwrap().range(), 16..48);
    assert!(binding_region(&shifted, 0..33).is_err());
    shifted.offset_start = Some(u64::MAX);
    assert!(binding_region(&shifted, 1..2).is_err());
}

#[test]
fn attention_copy_checks_source_before_destination_taint_and_preserves_failure() {
    let mut host = HostTracking::new();
    let source = host.buffer().binding();
    let destination = host.buffer().binding();
    let _ = ExecuteScope::over(&mut host, StreamId::current(), vec![source.clone()])
        .execute(|_| Err::<(), _>(failure("actual copy source producer failed")));
    let count = host.failures.graph().len();
    let ran = Cell::new(false);
    let outcome = launch_scope(
        &mut host,
        StreamId::current(),
        39,
        std::iter::once(&source),
        vec![destination.clone()],
    )
    .execute(|_| {
        ran.set(true);
        Ok::<_, ServerError>(())
    });
    assert!(matches!(outcome, ScopedOutcome::Skipped));
    assert!(!ran.get());
    assert_eq!(host.failures.graph().len(), count);
    for binding in [&source, &destination] {
        assert!(
            host.ensure_written(std::iter::once(binding))
                .unwrap_err()
                .to_string()
                .contains("actual copy source producer failed")
        );
    }
    let mut host = HostTracking::new();
    let source = host.buffer().binding();
    let destination = host.buffer().binding();
    let outcome = launch_scope(
        &mut host,
        StreamId::current(),
        40,
        std::iter::once(&source),
        vec![destination.clone()],
    )
    .execute(|_| Err::<(), _>(failure("actual copy enqueue failed")));
    assert!(matches!(outcome, ScopedOutcome::Failed(_)));
    host.ensure_written(std::iter::once(&source)).unwrap();
    assert!(
        host.ensure_written(std::iter::once(&destination))
            .unwrap_err()
            .to_string()
            .contains("actual copy enqueue failed")
    );
}
