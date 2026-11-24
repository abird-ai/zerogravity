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
        ForwardPlan, KvDomain, MathematicalBackend, MatrixAxes, ParameterRef, ParameterSlot,
        PlanBuilder, SourceTensor, StateSlot,
    },
    encoding::{ScalarDType, StoredTensorEncoding},
    identity::SourceIdentity,
    model::{ModelDefinition, ModelProgram, ProgramRequest},
    provider::{PreparedExecutable, ProviderKind, RegionExecutionProvider},
    source::{SourceTensorRef, TensorSource},
    task::TaskKind,
};
use zerogravity_engine::prepare::prepare;

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
}
impl PreparedExecutable for OwnerProbe {
    type Source = Arc<dyn TensorSource>;
    async fn source(&self) -> Result<Arc<dyn TensorSource>> {
        Ok(self.source.clone())
    }
    async fn program(&self) -> Result<Arc<ForwardPlan>> {
        Ok(self.plan.clone())
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
        Ok(OwnerProbe { source, plan })
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
