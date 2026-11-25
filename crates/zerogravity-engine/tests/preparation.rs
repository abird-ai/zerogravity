//! Source-free owner-routing controls, not numerical execution or Ready evidence.
use std::{
    future::Future,
    pin::pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Waker},
};
use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        AttentionGeometry, ForwardPlan, InvocationInput, InvocationPhase, KvDomain,
        MathematicalBackend, MatrixAxes, NumericalContract, ParameterRef, ParameterSlot,
        PlanBuilder, SourceAttentionMask, SourceOperatorKind, SourceOperatorTemplate,
        SourceParameter, SourcePlanGeometry, SourceQkRotaryGeometry, SourceRowSelection,
        SourceTensor, StateSlot, TensorView, TextCallGeometry, ValueStorageFormat,
    },
    encoding::{ScalarDType, StoredTensorEncoding},
    identity::SourceIdentity,
    model::{ModelDefinition, ModelProgram, ModelRegion, ProgramRequest},
    provider::{
        CompiledProvider, ExecutionMode, ObservationSupport, PreparedExecutable, PreparedRegion,
        PreparedTextProgram, ProviderBufferMode, ProviderCompletionMode, ProviderKind,
        ProviderRequirements, RegionExecutionProvider, SelectedMode, StateBridge, StateStorage,
    },
    semantic::{InvocationPorts, RegionStatePort, StateLayout, StateVersion},
    session_root::SessionAuthority,
    source::{SourceTensorRef, TensorSource},
    task::TaskKind,
};
use zerogravity_engine::prepare::{prepare, prepare_region};

#[derive(Clone, Copy)]
enum Recording {
    Original,
    WrongSource,
    WrongContext,
    MissingLogits,
}
struct Definition {
    source: SourceIdentity,
    declared_source: Option<SourceIdentity>,
    drops: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
    recording: Recording,
}
impl Drop for Definition {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
impl TensorSource for Definition {
    fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        let (kind, shape, packed_bytes): (_, &[u64], &[u8]) = match name {
            "weights" => (ScalarDType::F32, &[2, 2], &[0; 16]),
            "embedding" | "head" => (ScalarDType::Bf16, &[17, 16], &[0; 17 * 16 * 2]),
            "projection" => (ScalarDType::Bf16, &[16, 16], &[0; 16 * 16 * 2]),
            "packed_qkv" => (ScalarDType::Bf16, &[64, 16], &[0; 64 * 16 * 2]),
            "q_norm" | "k_norm" => (ScalarDType::Bf16, &[16], &[0; 16 * 2]),
            _ => return Err(Error::Malformed("unknown tensor".into())),
        };
        Ok(SourceTensorRef {
            kind: StoredTensorEncoding::Scalar(kind),
            shape,
            packed_bytes,
        })
    }
}
impl ModelDefinition for Definition {
    fn source(&self) -> SourceIdentity {
        self.declared_source.unwrap_or(self.source)
    }
    fn tasks(&self) -> &[TaskKind] {
        &[TaskKind::Language]
    }
    fn program(&self, request: ProgramRequest) -> Result<ForwardPlan> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let source = match self.recording {
            Recording::WrongSource => SourceIdentity::for_new_allocation()?,
            _ => self.source,
        };
        let context =
            request.context + usize::from(matches!(self.recording, Recording::WrongContext));
        let mut builder =
            PlanBuilder::new(source, context, vec![StateSlot::Kv(KvDomain(0))], false)?;
        let value = builder.embed(
            SourceTensor {
                source,
                name: "weights".into(),
                kind: StoredTensorEncoding::Scalar(ScalarDType::F32),
                axes: MatrixAxes::InputOutput,
                columns: 2,
                rows: 2,
            },
            ParameterRef(ParameterSlot::Token),
        )?;
        builder.append_kv(KvDomain(0), &value, &value)?;
        builder.advance_frontier()?;
        builder.finish(
            (request.logits && !matches!(self.recording, Recording::MissingLogits))
                .then_some(value),
        )
    }
}
fn definition(recording: Recording) -> Arc<Definition> {
    Arc::new(Definition {
        source: SourceIdentity::for_new_allocation().unwrap(),
        declared_source: None,
        drops: Arc::new(AtomicUsize::new(0)),
        calls: Arc::new(AtomicUsize::new(0)),
        recording,
    })
}
fn request() -> ProgramRequest {
    ProgramRequest {
        task: TaskKind::Language,
        context: 4,
        logits: true,
        observations: false,
    }
}
fn ready<T>(future: impl Future<Output = T>) -> T {
    let mut future = pin!(future);
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(result) => result,
        Poll::Pending => panic!("source-free immediate control unexpectedly pending"),
    }
}

// No execution method. The only assertions concern source/program ownership.
struct OwnerProbe {
    source: Arc<dyn TensorSource>,
    plan: Arc<ForwardPlan>,
    selection: SelectedMode,
    region: Option<PreparedRegion>,
    text: Option<PreparedTextProgram>,
}
impl PreparedExecutable for OwnerProbe {
    type Source = Arc<dyn TensorSource>;
    async fn source(&self) -> Result<Arc<dyn TensorSource>> {
        Ok(self.source.clone())
    }
    async fn program(&self) -> Result<Arc<ForwardPlan>> {
        Ok(self.plan.clone())
    }
    fn selected(&self) -> Result<SelectedMode> {
        Ok(self.selection)
    }
    fn prepared_region(&self) -> Option<&PreparedRegion> {
        self.region.as_ref()
    }
    fn prepared_text(&self) -> Option<&PreparedTextProgram> {
        self.text.as_ref()
    }
}
#[derive(Clone, Copy)]
enum Preparation {
    Original,
    CopiedProgram,
    WrongSource,
    SameIdDifferentOwner,
    MissingTextContract,
    WrongTextProvider,
    WrongTextMode,
    Refused,
}
struct ProviderProbe {
    mode: Preparation,
    yield_once: bool,
}
impl RegionExecutionProvider for ProviderProbe {
    type Executable = OwnerProbe;
    fn selected(&self) -> SelectedMode {
        SelectedMode {
            provider: ProviderKind::GraphOnly,
            mode: ExecutionMode::Graph,
        }
    }
    async fn prepare(&self, program: ModelProgram) -> Result<OwnerProbe> {
        let mut pending = self.yield_once;
        std::future::poll_fn(|cx| {
            if std::mem::take(&mut pending) {
                cx.waker().wake_by_ref();
                Poll::Pending
            } else {
                Poll::Ready(())
            }
        })
        .await;
        if matches!(self.mode, Preparation::Refused) {
            return Err(Error::Unsupported("probe has no executor".into()));
        }
        let (mut source, mut plan) = program.into_parts();
        if matches!(self.mode, Preparation::CopiedProgram) {
            plan = Arc::new(plan.as_ref().clone());
        }
        if matches!(
            self.mode,
            Preparation::WrongSource | Preparation::SameIdDifferentOwner
        ) {
            let mut replacement = definition(Recording::Original);
            if matches!(self.mode, Preparation::SameIdDifferentOwner) {
                Arc::get_mut(&mut replacement).unwrap().source = source.source_identity();
            }
            source = replacement;
        }
        let mut selection = self.selected();
        if matches!(self.mode, Preparation::WrongTextProvider) {
            selection.provider = ProviderKind::Cuda;
        }
        if matches!(self.mode, Preparation::WrongTextMode) {
            selection.mode = ExecutionMode::Native;
        }
        let text = if plan.source_plan_geometry().is_some()
            && !matches!(self.mode, Preparation::MissingTextContract)
        {
            Some(PreparedTextProgram::new(
                ModelProgram::from_recorded(source.clone(), plan.clone())?,
                selection,
                2..=8,
                ObservationSupport::None,
            )?)
        } else {
            None
        };
        Ok(OwnerProbe {
            source,
            plan,
            selection,
            region: None,
            text,
        })
    }
}

#[test]
fn unsupported_task_zero_context_and_unbound_source_never_record() {
    let mut source = definition(Recording::Original);
    for invalid in [
        ProgramRequest {
            context: 0,
            ..request()
        },
        ProgramRequest {
            task: TaskKind::Decision,
            ..request()
        },
    ] {
        assert!(ModelProgram::record(source.clone(), invalid).is_err());
    }
    assert_eq!(source.calls.load(Ordering::SeqCst), 0);
    Arc::get_mut(&mut source).unwrap().declared_source =
        Some(SourceIdentity::for_new_allocation().unwrap());
    assert!(matches!(
        ModelProgram::record(source.clone(), request()),
        Err(Error::State(_))
    ));
    assert_eq!(source.calls.load(Ordering::SeqCst), 0);
    Arc::get_mut(&mut source).unwrap().declared_source = None;
    Arc::get_mut(&mut source).unwrap().source = SourceIdentity::unbound();
    assert!(matches!(
        ModelProgram::record(source.clone(), request()),
        Err(Error::State(_))
    ));
    assert_eq!(source.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn foreign_recording_source_context_and_output_refuse_without_owner_escape() {
    for recording in [
        Recording::WrongSource,
        Recording::WrongContext,
        Recording::MissingLogits,
    ] {
        let source = definition(recording);
        let drops = source.drops.clone();
        assert!(matches!(
            ModelProgram::record(source, request()),
            Err(Error::State(_))
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn original_source_and_program_survive_preparation_and_final_reader() {
    let source = definition(Recording::Original);
    let drops = source.drops.clone();
    let original: Arc<dyn TensorSource> = source.clone();
    let program = ModelProgram::record(source, request()).unwrap();
    assert!(Arc::ptr_eq(&original, program.source()));
    let plan = program.program().clone();
    // The other ingress path must retain the same source and original plan,
    // without recording again. A foreign source cannot bind this recording.
    let rebound = ModelProgram::from_recorded(original.clone(), plan.clone()).unwrap();
    assert!(Arc::ptr_eq(rebound.program(), &plan));
    assert!(Arc::ptr_eq(rebound.source(), &original));
    assert!(
        ModelProgram::from_recorded(
            definition(Recording::Original) as Arc<dyn TensorSource>,
            plan.clone()
        )
        .is_err()
    );
    drop((original, program));
    let program = rebound;
    let provider = ProviderProbe {
        mode: Preparation::Original,
        yield_once: true,
    };
    let mut pending = pin!(prepare(program, &provider));
    let mut cx = Context::from_waker(Waker::noop());
    assert!(pending.as_mut().poll(&mut cx).is_pending());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let Poll::Ready(result) = pending.as_mut().poll(&mut cx) else {
        panic!("owner probe still pending")
    };
    let bundle = result.unwrap();
    assert!(Arc::ptr_eq(&plan, &ready(bundle.program()).unwrap()));
    assert_eq!(
        ready(bundle.source()).unwrap().source_identity(),
        plan.source()
    );
    let owner = bundle.into_prepared();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(owner);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn copied_program_foreign_source_and_provider_refusal_retire_original_owner() {
    for mode in [
        Preparation::CopiedProgram,
        Preparation::WrongSource,
        Preparation::SameIdDifferentOwner,
        Preparation::Refused,
    ] {
        let source = definition(Recording::Original);
        let drops = source.drops.clone();
        let program = ModelProgram::record(source, request()).unwrap();
        let provider = ProviderProbe {
            mode,
            yield_once: false,
        };
        assert!(ready(prepare(program, &provider)).is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

// Synthetic encoded source and shape fixture; no physical execution or SDK claim.
fn region_program(source: Arc<Definition>) -> ModelProgram {
    let identity = source.source;
    let geometry = SourcePlanGeometry::new(identity, 17, 512, 64).unwrap();
    let slot = StateSlot::Kv(KvDomain(0));
    let mut builder = PlanBuilder::new_source_template(geometry, vec![slot]).unwrap();
    let matrix = |name: &str, rows| SourceTensor {
        source: identity,
        name: name.into(),
        kind: StoredTensorEncoding::Scalar(ScalarDType::Bf16),
        axes: MatrixAxes::OutputInput,
        columns: 16,
        rows,
    };
    let template = |kind, rows, matrices| {
        SourceOperatorTemplate::new(
            geometry,
            rows,
            kind,
            NumericalContract::CudaSm86Bf16ReferenceV1,
            matrices,
            vec![],
        )
        .unwrap()
    };
    let raw = builder
        .record_source_template(
            "embedding",
            &template(
                SourceOperatorKind::EmbeddingRows,
                SourceRowSelection::AllActive,
                vec![matrix("embedding", 17)],
            ),
            &[],
        )
        .unwrap()[0];
    let projected = builder
        .record_source_template(
            "projection",
            &template(
                SourceOperatorKind::Projection { selected_row: None },
                SourceRowSelection::AllActive,
                vec![matrix("projection", 16)],
            ),
            &[raw],
        )
        .unwrap()[0];
    let activated = builder
        .record_source_template(
            "sigmoid",
            &template(
                SourceOperatorKind::Sigmoid { width: 16 },
                SourceRowSelection::AllActive,
                vec![],
            ),
            &[projected],
        )
        .unwrap()[0];
    let product = builder
        .record_source_template(
            "multiply",
            &template(
                SourceOperatorKind::Multiply { width: 16 },
                SourceRowSelection::AllActive,
                vec![],
            ),
            &[projected, activated],
        )
        .unwrap()[0];
    let packed = builder
        .record_source_template(
            "packed_qkv",
            &template(
                SourceOperatorKind::Projection { selected_row: None },
                SourceRowSelection::AllActive,
                vec![matrix("packed_qkv", 64)],
            ),
            &[product],
        )
        .unwrap()[0];
    let gamma = |name: &str| SourceParameter {
        source: identity,
        name: name.into(),
        kind: StoredTensorEncoding::Scalar(ScalarDType::Bf16),
        shape: vec![16],
        bytes: 32,
    };
    let qk = SourceOperatorTemplate::new(
        geometry,
        SourceRowSelection::AllActive,
        SourceOperatorKind::QkRmsRotaryGated(SourceQkRotaryGeometry {
            attention: AttentionGeometry {
                width: 16,
                query_heads: 1,
                kv_heads: 1,
            },
            rotary_dimensions: 16,
            theta: 10000.0,
            epsilon: 1e-6,
        }),
        NumericalContract::CudaSm86Bf16ReferenceV1,
        vec![],
        vec![gamma("q_norm"), gamma("k_norm")],
    )
    .unwrap();
    let [query, key, _gate] = builder
        .record_source_template("rotary", &qk, &[packed])
        .unwrap()
        .try_into()
        .unwrap();
    builder
        .record_source_template(
            "append",
            &template(
                SourceOperatorKind::KvAppend {
                    domain: KvDomain(0),
                    key_width: 16,
                    packed_value: TensorView {
                        offset: 48,
                        rows: 64,
                        width: 16,
                        row_stride: 64,
                    },
                },
                SourceRowSelection::AllActive,
                vec![],
            ),
            &[key, packed],
        )
        .unwrap();
    let attended = builder
        .record_source_template(
            "attention",
            &template(
                SourceOperatorKind::Attention {
                    domain: KvDomain(0),
                    geometry: AttentionGeometry {
                        width: 16,
                        query_heads: 1,
                        kv_heads: 1,
                    },
                    scale: 0.25,
                    mask: SourceAttentionMask::Causal,
                },
                SourceRowSelection::AllActive,
                vec![],
            ),
            &[query],
        )
        .unwrap()[0];
    let logits = builder
        .record_source_template(
            "head",
            &template(
                SourceOperatorKind::Projection { selected_row: None },
                SourceRowSelection::LastActive,
                vec![matrix("head", 17)],
            ),
            &[attended],
        )
        .unwrap()[0];
    ModelProgram::from_recorded(
        source as Arc<dyn TensorSource>,
        Arc::new(builder.finish_source_template(logits).unwrap()),
    )
    .unwrap()
}

#[test]
fn closed_region_ports_and_active_invocation_keep_original_semantics() {
    let program = region_program(definition(Recording::Original));
    let plan = program.program();
    let region = program.closed_region(1..3).unwrap();
    let region = region.region();
    assert!(Arc::ptr_eq(region.program(), plan));
    assert_eq!(region.inputs().len(), 1);
    assert_eq!(region.inputs()[0].value, plan.nodes()[0].outputs[0]);
    assert_eq!(
        region.outputs().iter().map(|p| p.value).collect::<Vec<_>>(),
        vec![plan.nodes()[1].outputs[0], plan.nodes()[2].outputs[0]]
    );
    let head = program.closed_region(8..9).unwrap();
    let bind = |plan: &Arc<ForwardPlan>, rows, emit| {
        plan.bind_invocation(
            InvocationInput::new(
                TextCallGeometry::new(plan.source(), 17, 512).unwrap(),
                0,
                vec![1; rows].into(),
                emit,
            )
            .unwrap(),
        )
        .unwrap()
    };
    for rows in [1, 7, 64] {
        let invocation = bind(plan, rows, true);
        let binding = region
            .bind(invocation.call(InvocationPhase::Prefill).unwrap())
            .unwrap();
        let inputs = binding.inputs().collect::<Vec<_>>();
        let outputs = binding.outputs().collect::<Vec<_>>();
        assert_eq!(inputs[0].layout.rows, rows);
        assert_eq!(inputs[0].layout.row_stride, 16);
        assert_eq!(outputs[0].layout.elements().unwrap(), rows * 16);
        binding.require_bindings(&inputs, &outputs).unwrap();
        let mut wrong = outputs.clone();
        wrong[0].layout.row_stride += 1;
        assert!(binding.require_bindings(&inputs, &wrong).is_err());
        let head_binding = head
            .region()
            .bind(invocation.call(InvocationPhase::Prefill).unwrap())
            .unwrap();
        assert_eq!(head_binding.inputs().next().unwrap().layout.rows, rows);
        assert_eq!(head_binding.outputs().next().unwrap().layout.rows, 1);
        assert_eq!(head_binding.outputs().next().unwrap().layout.width, 17);
    }
    let state_only = bind(plan, 2, false);
    assert!(
        region
            .bind(state_only.call(InvocationPhase::Prefill).unwrap())
            .is_ok()
    );
    assert!(
        head.region()
            .bind(state_only.call(InvocationPhase::Prefill).unwrap())
            .is_err()
    );
    let selected = SelectedMode {
        provider: ProviderKind::Cuda,
        mode: ExecutionMode::Native,
    };
    let head = PreparedRegion::new(
        head,
        selected,
        ProviderRequirements::original_queue(0, 0),
        1..=64,
        &[ValueStorageFormat::Bf16],
        &[ValueStorageFormat::F32],
    )
    .unwrap();
    let call = bind(plan, 7, true);
    let bound = head
        .bind(call.call(InvocationPhase::Prefill).unwrap())
        .unwrap();
    assert_eq!(bound.input(0).unwrap().byte_len().unwrap(), 7 * 16 * 2);
    assert_eq!(bound.output(0).unwrap().rows(), 1);
    assert_eq!(bound.output(0).unwrap().byte_len().unwrap(), 17 * 4);
    assert!(
        head.bind(state_only.call(InvocationPhase::Prefill).unwrap())
            .is_err()
    );
    for rows in [0..=8, 1..=65, std::ops::RangeInclusive::new(9, 8)] {
        assert!(
            PreparedRegion::new(
                program.closed_region(1..3).unwrap(),
                selected,
                ProviderRequirements::original_queue(0, 0),
                rows,
                &[ValueStorageFormat::Bf16],
                &[ValueStorageFormat::Bf16; 2]
            )
            .is_err()
        );
    }
    assert!(
        PreparedRegion::new(
            program.closed_region(1..3).unwrap(),
            selected,
            ProviderRequirements::original_queue(0, 0),
            1..=8,
            &[ValueStorageFormat::Bf16],
            &[ValueStorageFormat::Bf16]
        )
        .is_err()
    );
    let copy = Arc::new(plan.as_ref().clone());
    assert!(
        region
            .bind(bind(&copy, 2, true).call(InvocationPhase::Prefill).unwrap())
            .is_err()
    );
    for range in [0..0, 3..9, 8..10] {
        assert!(program.closed_region(range).is_err());
    }
}

#[test]
fn stateful_region_ports_and_bridge_keep_provisional_versions_on_one_attempt() {
    let program = region_program(definition(Recording::Original));
    let plan = program.program();
    let slot = StateSlot::Kv(KvDomain(0));
    let prior = StateVersion { slot, ordinal: 0 };
    let candidate = StateVersion { slot, ordinal: 1 };
    let layout = StateLayout::Kv {
        slab: 16,
        capacity: 512,
    };
    let port = |version| RegionStatePort { version, layout };
    let domains = plan.source_state_domains().unwrap();
    assert_eq!(domains.len(), 1);
    assert_eq!(domains[0].writer, 6);
    assert_eq!(domains[0].layout, layout);
    assert_eq!(domains[0].effect.read, prior);
    assert_eq!(domains[0].effect.write, candidate);

    let append = program.closed_region(6..7).unwrap();
    assert!(append.region().outputs().is_empty());
    assert_eq!(append.region().state_inputs(), [port(prior)]);
    assert_eq!(append.region().state_outputs(), [port(candidate)]);
    let attention = program.closed_region(7..8).unwrap();
    assert_eq!(attention.region().state_inputs(), [port(candidate)]);
    assert!(attention.region().state_outputs().is_empty());
    let both = program.closed_region(6..8).unwrap();
    assert_eq!(both.region().state_inputs(), [port(prior)]);
    assert_eq!(both.region().state_outputs(), [port(candidate)]);
    let state = StateStorage::new(slot, layout, ValueStorageFormat::Bf16).unwrap();
    let widened = StateStorage::new(slot, layout, ValueStorageFormat::F32).unwrap();
    assert_eq!(state.bytes().unwrap(), 2 * 16 * 512 * 2);
    assert_eq!(widened.bytes().unwrap(), 2 * state.bytes().unwrap());
    let selected = SelectedMode {
        provider: ProviderKind::Cuda,
        mode: ExecutionMode::Native,
    };
    let requirements = ProviderRequirements::original_queue(0, 0);
    let prepare = |range,
                   storage: &[StateStorage],
                   inputs: &[ValueStorageFormat],
                   outputs: &[ValueStorageFormat]| {
        PreparedRegion::with_state(
            program.closed_region(range).unwrap(),
            selected,
            requirements,
            1..=64,
            inputs,
            outputs,
            storage,
        )
    };
    assert!(
        PreparedRegion::new(
            program.closed_region(6..7).unwrap(),
            selected,
            requirements,
            1..=64,
            &[ValueStorageFormat::Bf16; 2],
            &[]
        )
        .is_err()
    );
    assert!(prepare(6..7, &[state, state], &[ValueStorageFormat::Bf16; 2], &[]).is_err());
    let wrong = StateStorage::new(
        slot,
        StateLayout::Kv {
            slab: 32,
            capacity: 256,
        },
        ValueStorageFormat::Bf16,
    )
    .unwrap();
    assert_eq!(wrong.bytes().unwrap(), state.bytes().unwrap());
    assert!(prepare(6..7, &[wrong], &[ValueStorageFormat::Bf16; 2], &[]).is_err());
    let append = prepare(6..7, &[state], &[ValueStorageFormat::Bf16; 2], &[]).unwrap();
    let attention = prepare(
        7..8,
        &[state],
        &[ValueStorageFormat::Bf16],
        &[ValueStorageFormat::Bf16],
    )
    .unwrap();
    let different = prepare(
        7..8,
        &[widened],
        &[ValueStorageFormat::Bf16],
        &[ValueStorageFormat::Bf16],
    )
    .unwrap();

    let invocation = plan
        .bind_invocation(
            InvocationInput::new(
                TextCallGeometry::new(plan.source(), 17, 512).unwrap(),
                0,
                vec![1; 7].into(),
                false,
            )
            .unwrap(),
        )
        .unwrap();
    let call = invocation.call(InvocationPhase::Prefill).unwrap();
    let owner = SessionAuthority::new(plan.source()).unwrap();
    let current = owner.binding();
    let attempt = owner.begin_advance(7).unwrap();
    let replacement = owner.begin_validated_replacement(7).unwrap();
    assert!(append.bind_state(call, current, &replacement).is_err());
    let foreign = SessionAuthority::new(plan.source())
        .unwrap()
        .begin_advance(7)
        .unwrap();
    assert!(append.bind_state(call, current, &foreign).is_err());
    let write = append.bind_state(call, current, &attempt).unwrap();
    let read = attention.bind_state(call, current, &attempt).unwrap();
    assert_eq!(write.prior(), current);
    assert_eq!(write.frontier(), 7);
    read.require_input_from(&write, candidate, StateBridge::OriginalLayout)
        .unwrap();
    assert!(
        read.require_input_from(&write, prior, StateBridge::OriginalLayout)
            .is_err()
    );
    assert!(
        read.require_input_from(&write, candidate, StateBridge::ConvertAtCommittedFrontier)
            .is_err()
    );
    assert!(
        write
            .require_input_from(&read, candidate, StateBridge::OriginalLayout)
            .is_err()
    );
    assert!(
        different
            .bind_state(call, current, &attempt)
            .unwrap()
            .require_input_from(&write, candidate, StateBridge::OriginalLayout)
            .is_err()
    );
    let other_attempt = owner.begin_advance(7).unwrap();
    assert!(
        attention
            .bind_state(call, current, &other_attempt)
            .unwrap()
            .require_input_from(&write, candidate, StateBridge::OriginalLayout)
            .is_err()
    );
    let other_call = invocation.clone();
    assert!(
        attention
            .bind_state(
                other_call.call(InvocationPhase::Prefill).unwrap(),
                current,
                &attempt
            )
            .unwrap()
            .require_input_from(&write, candidate, StateBridge::OriginalLayout)
            .is_err()
    );
    assert_eq!(
        owner.binding(),
        current,
        "region binding cannot publish state"
    );
    let mut owner = owner;
    owner.prepare_publish(attempt).unwrap().commit();
    let current = owner.binding();
    let cached = plan
        .bind_invocation(
            InvocationInput::new(
                TextCallGeometry::new(plan.source(), 17, 512).unwrap(),
                7,
                vec![1].into(),
                true,
            )
            .unwrap(),
        )
        .unwrap();
    let attempt = owner.begin_advance(1).unwrap();
    let prefill = cached.call(InvocationPhase::Prefill).unwrap();
    let decode = cached.call(InvocationPhase::Decode).unwrap();
    let write = append.bind_state(prefill, current, &attempt).unwrap();
    let read = attention.bind_state(decode, current, &attempt).unwrap();
    assert!(
        read.require_input_from(&write, candidate, StateBridge::OriginalLayout)
            .is_err()
    );
}

#[test]
fn original_text_controls_observation_sites_and_whole_preparation() {
    let program = region_program(definition(Recording::Original));
    let plan = program.program().clone();
    let embedding = program.closed_region(0..1).unwrap();
    let rotary = program.closed_region(5..6).unwrap();
    let head = program.closed_region(8..9).unwrap();
    let bind = |plan: &Arc<ForwardPlan>, base, rows, emit| {
        plan.bind_invocation(
            InvocationInput::new(
                TextCallGeometry::new(plan.source(), 17, 512).unwrap(),
                base,
                vec![1; rows].into(),
                emit,
            )
            .unwrap(),
        )
        .unwrap()
    };
    let invocation = bind(&plan, 13, 7, true);
    let call = invocation.call(InvocationPhase::Prefill).unwrap();
    let tokens = embedding.region().bind(call).unwrap();
    assert!(tokens.inputs().next().is_none());
    assert_eq!(tokens.token_ids(), Some(invocation.input().ids()));
    assert_eq!(tokens.positions(), None);
    let positions = rotary.region().bind(call).unwrap();
    assert_eq!(positions.positions(), Some(13..20));
    assert_eq!(positions.token_ids(), None);
    assert_eq!(positions.phase(), InvocationPhase::Prefill);
    let observed = positions.observations().collect::<Vec<_>>();
    assert_eq!(observed.len(), 3); // Includes internal/unconsumed rotary gate.
    assert_eq!(positions.outputs().count(), 2);
    assert_eq!(observed[2].1.value, plan.nodes()[5].outputs[2]);
    assert_eq!(observed[2].1.layout.elements().unwrap(), 7 * 16);
    let head_obs = head
        .region()
        .bind(call)
        .unwrap()
        .observations()
        .collect::<Vec<_>>();
    assert_eq!(head_obs[0].1.layout.rows, 1);
    assert_eq!(head_obs[0].1.layout.width, 17);
    assert_eq!(
        plan.source_observation_ports(5)
            .unwrap()
            .map(|p| p.unwrap().bind(7).unwrap())
            .collect::<Vec<_>>(),
        observed
            .iter()
            .map(|(_, binding)| *binding)
            .collect::<Vec<_>>(),
    );
    assert!(
        plan.source_observation_ports(5)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .bind(65)
            .is_err()
    );

    let selected = SelectedMode {
        provider: ProviderKind::GraphOnly,
        mode: ExecutionMode::Graph,
    };
    let prepared_embedding = PreparedRegion::new(
        embedding,
        selected,
        ProviderRequirements::original_queue(0, 0),
        1..=64,
        &[],
        &[ValueStorageFormat::Bf16],
    )
    .unwrap();
    assert!(prepared_embedding.bind(call).is_ok());
    assert!(prepared_embedding.bind_observed(call).is_err());
    let prepared_embedding = prepared_embedding.with_observations(ObservationSupport::AtProducer);
    assert!(prepared_embedding.bind_observed(call).is_ok());

    let provider = ProviderProbe {
        mode: Preparation::Original,
        yield_once: false,
    };
    let bundle = ready(prepare(program, &provider)).unwrap();
    let prepared = bundle.prepared_text().unwrap();
    assert_eq!(
        prepared.invocation_ports(),
        InvocationPorts {
            token_ids: true,
            positions: true
        }
    );
    assert_eq!(prepared.selected(), selected);
    assert_eq!(prepared.supported_rows(), 2..=8);
    assert!(
        prepared
            .bind(&invocation, InvocationPhase::Prefill, false)
            .unwrap()
            .same_call(call)
    );
    assert!(
        prepared
            .bind(&invocation, InvocationPhase::Prefill, true)
            .is_err()
    );
    let state_only = bind(&plan, 13, 7, false);
    let call = prepared
        .bind(&state_only, InvocationPhase::Prefill, false)
        .unwrap();
    assert_eq!(
        call.invocation().active_nodes().len(),
        plan.state_nodes().len()
    );
    assert!(
        prepared
            .bind(&bind(&plan, 13, 9, true), InvocationPhase::Prefill, false)
            .is_err()
    );
    assert!(
        prepared
            .bind(
                &bind(&Arc::new(plan.as_ref().clone()), 13, 7, true),
                InvocationPhase::Prefill,
                false
            )
            .is_err()
    );
    assert!(
        prepared
            .bind(&invocation, InvocationPhase::Decode, false)
            .is_err()
    );
    for mode in [
        Preparation::MissingTextContract,
        Preparation::WrongTextProvider,
        Preparation::WrongTextMode,
    ] {
        let program = region_program(definition(Recording::Original));
        assert!(
            ready(prepare(
                program,
                &ProviderProbe {
                    mode,
                    yield_once: false
                }
            ))
            .is_err()
        );
    }
    assert!(
        InvocationPhase::Prefill
            .require_source_profile(
                NumericalContract::HipRocBlasBf16MixedV2.program_profile(),
                13,
                1,
            )
            .is_err()
    );
}

struct RegionProbe {
    inner: ProviderProbe,
    selection: Option<SelectedMode>,
}
const COMPILED_PROBE: SelectedMode = SelectedMode {
    provider: ProviderKind::GraphOnly,
    mode: ExecutionMode::Compiled(CompiledProvider::Iree),
};
impl RegionExecutionProvider for RegionProbe {
    type Executable = OwnerProbe;
    fn selected(&self) -> SelectedMode {
        COMPILED_PROBE
    }
    async fn prepare(&self, program: ModelProgram) -> Result<OwnerProbe> {
        self.inner.prepare(program).await
    }
    async fn prepare_region(&self, region: ModelRegion) -> Result<OwnerProbe> {
        // Compiler-local lowering consumes the original node and typed encoded
        // operands directly. Nothing is exported or given execution authority.
        for op in region.operators() {
            assert!(std::ptr::eq(
                op.node(),
                &region.program().nodes()[op.index()]
            ));
            for bound in op.matrices() {
                let (descriptor, binding) = bound?;
                assert_eq!(
                    binding.packed_bytes.len() as u64,
                    descriptor.packed_bytes()?
                );
            }
            for bound in op.parameters() {
                let (descriptor, binding) = bound?;
                assert_eq!(binding.shape, descriptor.shape);
                assert_eq!(binding.packed_bytes.len(), descriptor.bytes);
            }
        }
        let program =
            ModelProgram::from_recorded(region.source().clone(), region.program().clone())?;
        let mut owner = self.inner.prepare(program).await?;
        owner.text = None;
        owner.selection = self.selection.unwrap_or(COMPILED_PROBE);
        owner.region = self
            .selection
            .map(|selected| {
                let inputs = vec![ValueStorageFormat::Bf16; region.region().inputs().len()];
                let outputs = (0..region.region().outputs().len())
                    .map(|i| {
                        if i == 1 {
                            ValueStorageFormat::F32
                        } else {
                            ValueStorageFormat::Bf16
                        }
                    })
                    .collect::<Vec<_>>();
                PreparedRegion::new(
                    region,
                    selected,
                    ProviderRequirements::original_queue(256, 1024),
                    2..=8,
                    &inputs,
                    &outputs,
                )
            })
            .transpose()?;
        Ok(owner)
    }
}

#[test]
fn partial_provider_preparation_retains_exact_source_plan_and_scope() {
    for (mode, selection, accepted) in [
        (Preparation::Original, Some(COMPILED_PROBE), true),
        (Preparation::Original, None, false),
        (
            Preparation::Original,
            Some(SelectedMode {
                provider: ProviderKind::Cuda,
                ..COMPILED_PROBE
            }),
            false,
        ),
        (
            Preparation::Original,
            Some(SelectedMode {
                mode: ExecutionMode::Graph,
                ..COMPILED_PROBE
            }),
            false,
        ),
        (
            Preparation::Original,
            Some(SelectedMode {
                mode: ExecutionMode::Compiled(CompiledProvider::OnnxRuntime),
                ..COMPILED_PROBE
            }),
            false,
        ),
        (Preparation::CopiedProgram, Some(COMPILED_PROBE), false),
        (
            Preparation::SameIdDifferentOwner,
            Some(COMPILED_PROBE),
            false,
        ),
        (Preparation::Refused, Some(COMPILED_PROBE), false),
    ] {
        let source = definition(Recording::Original);
        let drops = source.drops.clone();
        let program = region_program(source);
        let plan = program.program().clone();
        let region = program.closed_region(1..3).unwrap();
        drop(program);
        let provider = RegionProbe {
            inner: ProviderProbe {
                mode,
                yield_once: false,
            },
            selection,
        };
        let result = ready(prepare_region(region, &provider));
        if accepted {
            let bundle = result.unwrap();
            assert_eq!(bundle.region_nodes(), Some(1..3));
            assert!(Arc::ptr_eq(&ready(bundle.program()).unwrap(), &plan));
            let prepared = bundle.prepared_region().unwrap();
            assert_eq!(bundle.selected().unwrap(), COMPILED_PROBE);
            assert_eq!(prepared.selected(), COMPILED_PROBE);
            let requirements = prepared.requirements();
            assert_eq!(requirements.device_bytes().unwrap(), 1280);
            requirements.require_original_queue().unwrap();
            requirements.require_admitted(256, 1024).unwrap();
            assert!(requirements.require_admitted(0, 1024).is_err());
            assert!(requirements.require_admitted(256, 0).is_err());
            for bridge in [
                ProviderRequirements {
                    buffers: ProviderBufferMode::Private,
                    ..requirements
                },
                ProviderRequirements {
                    completion: ProviderCompletionMode::ExternalFence,
                    ..requirements
                },
            ] {
                assert!(matches!(
                    bridge.require_original_queue(),
                    Err(Error::Unsupported(_))
                ));
            }
            assert!(
                ProviderRequirements::original_queue(usize::MAX, 1)
                    .device_bytes()
                    .is_err()
            );
            assert_eq!(prepared.supported_rows(), 2..=8);
            assert_eq!(
                prepared.capacity().input(0).unwrap().byte_len().unwrap(),
                8 * 16 * 2
            );
            let bind = |plan: &Arc<ForwardPlan>, rows| {
                plan.bind_invocation(
                    InvocationInput::new(
                        TextCallGeometry::new(plan.source(), 17, 512).unwrap(),
                        0,
                        vec![1; rows].into(),
                        true,
                    )
                    .unwrap(),
                )
                .unwrap()
            };
            let invocation = bind(&plan, 7);
            let bound = prepared
                .bind(invocation.call(InvocationPhase::Prefill).unwrap())
                .unwrap();
            assert_eq!(bound.input(0).unwrap().row_stride_bytes(), 16 * 2);
            assert_eq!(bound.output(0).unwrap().byte_len().unwrap(), 7 * 16 * 2);
            assert_eq!(bound.output(1).unwrap().byte_len().unwrap(), 7 * 16 * 4);
            assert_eq!(
                bound.outputs().nth(1).unwrap().unwrap().0.format(),
                ValueStorageFormat::F32
            );
            assert!(matches!(
                prepared.bind(bind(&plan, 1).call(InvocationPhase::Prefill).unwrap()),
                Err(Error::Unsupported(_))
            ));
            assert!(matches!(
                prepared.bind(bind(&plan, 9).call(InvocationPhase::Prefill).unwrap()),
                Err(Error::Unsupported(_))
            ));
            assert!(matches!(
                prepared.bind(
                    bind(&Arc::new(plan.as_ref().clone()), 7)
                        .call(InvocationPhase::Prefill)
                        .unwrap()
                ),
                Err(Error::State(_))
            ));
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            drop(bundle);
        } else {
            assert!(result.is_err());
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
    let program = region_program(definition(Recording::Original));
    let region = program.closed_region(1..3).unwrap();
    let whole_only = ProviderProbe {
        mode: Preparation::Original,
        yield_once: false,
    };
    assert!(matches!(
        ready(prepare_region(region, &whole_only)),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn compiled_region_borrows_original_learned_operands_and_late_call() {
    let source = definition(Recording::Original);
    let drops = source.drops.clone();
    let program = region_program(source);
    let plan = program.program().clone();
    let region = program.closed_region(4..6).unwrap();
    let mut operators = region.operators();
    let projection = operators.next().unwrap();
    assert_eq!(projection.index(), 4);
    let (descriptor, bound) = projection.matrices().next().unwrap().unwrap();
    assert_eq!(descriptor.name, "packed_qkv");
    assert_eq!(bound.shape, [64, 16]);
    assert!(projection.parameters().next().is_none());
    let rotary = operators.next().unwrap();
    assert_eq!(rotary.index(), 5);
    assert!(rotary.matrices().next().is_none());
    let names = rotary
        .parameters()
        .map(|bound| {
            let (descriptor, bound) = bound.unwrap();
            assert_eq!(bound.shape, [16]);
            assert_eq!(bound.packed_bytes.len(), 32);
            descriptor.name.as_str()
        })
        .collect::<Vec<_>>();
    assert_eq!(names, ["q_norm", "k_norm"]);
    assert!(operators.next().is_none());
    drop(operators);
    drop(program);

    let provider = RegionProbe {
        inner: ProviderProbe {
            mode: Preparation::Original,
            yield_once: false,
        },
        selection: Some(COMPILED_PROBE),
    };
    let bundle = ready(prepare_region(region, &provider)).unwrap();
    let invocation = plan
        .bind_invocation(
            InvocationInput::new(
                TextCallGeometry::new(plan.source(), 17, 512).unwrap(),
                13,
                vec![1; 7].into(),
                true,
            )
            .unwrap(),
        )
        .unwrap();
    let call = invocation.call(InvocationPhase::Prefill).unwrap();
    let prepared = bundle.prepared_region().unwrap();
    let (logical, physical) = prepared.bind_call(call, false).unwrap();
    assert_eq!(logical.positions(), Some(13..20));
    assert_eq!(logical.token_ids(), None);
    assert_eq!(physical.input(0).unwrap().byte_len().unwrap(), 7 * 16 * 2);
    assert!(prepared.bind_call(call, true).is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(bundle);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
