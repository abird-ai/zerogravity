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
        ForwardPlan, InvocationInput, KvDomain, MathematicalBackend, MatrixAxes, NumericalContract,
        ParameterRef, ParameterSlot, PlanBuilder, SourceOperatorKind, SourceOperatorTemplate,
        SourcePlanGeometry, SourceRowSelection, SourceTensor, StateSlot, TextCallGeometry,
    },
    encoding::{ScalarDType, StoredTensorEncoding},
    identity::SourceIdentity,
    model::{ModelDefinition, ModelProgram, ModelRegion, ProgramRequest},
    provider::{PreparedExecutable, ProviderKind, RegionExecutionProvider},
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
        if name != "weights" {
            return Err(Error::Malformed("unknown tensor".into()));
        }
        Ok(SourceTensorRef {
            kind: StoredTensorEncoding::Scalar(ScalarDType::F32),
            shape: &[2, 2],
            packed_bytes: &[0; 16],
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
    scope: Option<std::ops::Range<usize>>,
}
impl PreparedExecutable for OwnerProbe {
    type Source = Arc<dyn TensorSource>;
    async fn source(&self) -> Result<Arc<dyn TensorSource>> {
        Ok(self.source.clone())
    }
    async fn program(&self) -> Result<Arc<ForwardPlan>> {
        Ok(self.plan.clone())
    }
    fn region_nodes(&self) -> Option<std::ops::Range<usize>> {
        self.scope.clone()
    }
}
#[derive(Clone, Copy)]
enum Preparation {
    Original,
    CopiedProgram,
    WrongSource,
    SameIdDifferentOwner,
    Refused,
}
struct ProviderProbe {
    mode: Preparation,
    yield_once: bool,
}
impl RegionExecutionProvider for ProviderProbe {
    type Executable = OwnerProbe;
    fn kind(&self) -> ProviderKind {
        ProviderKind::GraphOnly
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
        Ok(OwnerProbe {
            source,
            plan,
            scope: None,
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

// Shape fixture only: no physical source bytes or compiler success claim.
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
    builder
        .record_source_template(
            "carry",
            &template(
                SourceOperatorKind::StateCarry { slot },
                SourceRowSelection::AllActive,
                vec![],
            ),
            &[],
        )
        .unwrap();
    let logits = builder
        .record_source_template(
            "head",
            &template(
                SourceOperatorKind::Projection { selected_row: None },
                SourceRowSelection::LastActive,
                vec![matrix("head", 17)],
            ),
            &[product],
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
    let head = program.closed_region(5..6).unwrap();
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
        let binding = region.bind(&invocation).unwrap();
        let inputs = binding.inputs().collect::<Vec<_>>();
        let outputs = binding.outputs().collect::<Vec<_>>();
        assert_eq!(inputs[0].layout.rows, rows);
        assert_eq!(inputs[0].layout.row_stride, 16);
        assert_eq!(outputs[0].layout.elements().unwrap(), rows * 16);
        binding.require_bindings(&inputs, &outputs).unwrap();
        let mut wrong = outputs.clone();
        wrong[0].layout.row_stride += 1;
        assert!(binding.require_bindings(&inputs, &wrong).is_err());
        let head_binding = head.region().bind(&invocation).unwrap();
        assert_eq!(head_binding.inputs().next().unwrap().layout.rows, rows);
        assert_eq!(head_binding.outputs().next().unwrap().layout.rows, 1);
        assert_eq!(head_binding.outputs().next().unwrap().layout.width, 17);
    }
    let state_only = bind(plan, 2, false);
    assert!(region.bind(&state_only).is_ok());
    assert!(head.region().bind(&state_only).is_err());
    let copy = Arc::new(plan.as_ref().clone());
    assert!(region.bind(&bind(&copy, 2, true)).is_err());
    for range in [0..0, 0..1, 4..5, 3..6, 5..7] {
        assert!(program.closed_region(range).is_err());
    }
}

struct RegionProbe {
    inner: ProviderProbe,
    wrong_scope: bool,
}
impl RegionExecutionProvider for RegionProbe {
    type Executable = OwnerProbe;
    fn kind(&self) -> ProviderKind {
        ProviderKind::GraphOnly
    }
    async fn prepare(&self, program: ModelProgram) -> Result<OwnerProbe> {
        self.inner.prepare(program).await
    }
    async fn prepare_region(&self, region: ModelRegion) -> Result<OwnerProbe> {
        let scope = region.region().range();
        let program =
            ModelProgram::from_recorded(region.source().clone(), region.program().clone())?;
        let mut owner = self.inner.prepare(program).await?;
        owner.scope = if self.wrong_scope { None } else { Some(scope) };
        Ok(owner)
    }
}

#[test]
fn partial_provider_preparation_retains_exact_source_plan_and_scope() {
    for (mode, wrong_scope, accepted) in [
        (Preparation::Original, false, true),
        (Preparation::Original, true, false),
        (Preparation::CopiedProgram, false, false),
        (Preparation::SameIdDifferentOwner, false, false),
        (Preparation::Refused, false, false),
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
            wrong_scope,
        };
        let result = ready(prepare_region(region, &provider));
        if accepted {
            let bundle = result.unwrap();
            assert_eq!(bundle.region_nodes(), Some(1..3));
            assert!(Arc::ptr_eq(&ready(bundle.program()).unwrap(), &plan));
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
