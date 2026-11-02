//! Genuine CPU diagnostic SDK + original source admission, not trained-model proof.
use super::{fixture, reexec};
use crate::{
    diagnostics::cpu_text::{self, AuditSink, Diagnostics, Observation, TextOptions},
    pipeline::ModelInstance,
    prepare::selection::Selection,
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner, TransferAdmission},
};
use std::{cell::Cell, io::Write, rc::Rc};

struct Audit {
    progress: Rc<Cell<usize>>,
    opened_at: Option<usize>,
    bytes: Vec<u8>,
}
impl AuditSink for Audit {
    fn open(&mut self) -> crate::Result<&mut dyn Write> {
        assert!(self.opened_at.is_none(), "audit opened more than once");
        self.opened_at = Some(self.progress.get());
        Ok(&mut self.bytes)
    }
}

#[test]
#[ignore = "requires ZG_APPROVED_TOKENIZER; real sparse GGUF CPU execution"]
fn source_bound_cpu_diagnostic_and_planning_retirement() {
    const TEST: &str = "builtins::text::tests::serving_population::direct::source_bound_cpu_diagnostic_and_planning_retirement";
    const DONE: &str = "DIRECT_SDK_CHILD_OK";
    if reexec("ZG_DIRECT_SDK_CHILD", TEST, DONE) {
        return;
    }
    let root = fixture("direct-sdk");
    let source = ModelInstance {
        id: "fixture".into(),
        target: String::new(),
        source: Some(root.join("model.gguf").to_string_lossy().into_owned()),
        model_config: None,
        tokenizer: None,
        snapshot: None,
        native_artifacts: None,
    };
    let mut wrong = source.clone();
    wrong.target = "not-the-source-target".into();
    assert!(Selection::resolve(&wrong, 8, None).is_err());
    assert!(Selection::resolve(&source, 0, None).is_err());

    let arbiter = ResourceArbiter::process_shared();
    let host_key = PhysicalDeviceKey::new("host", "system-ram").unwrap();
    assert!(arbiter.snapshot(&host_key).is_err());
    use crate::run_policy::{Backend, Entrypoint, Headroom, Origin, Speculation};
    let request = crate::run::config::resolve(
        None,
        crate::run::config::RunOverrides {
            model: source.source.clone(),
            entrypoint: Some(Entrypoint::Model(source.id.clone())),
            task: crate::pipelines::TaskDoc {
                context: Some(8),
                max_tokens: Some(2),
                raw: Some(true),
                ..Default::default()
            },
            backend: Some(Backend::Cpu),
            headroom: Some(Headroom::Bytes(0)),
            numerical: Some("source".into()),
            speculation: Some(Speculation::Off),
            diagnostics: crate::pipelines::DiagnosticDoc {
                dry_run: Some(true),
                ..Default::default()
            },
            ..Default::default()
        },
        &root,
        Origin::Caller,
    )
    .unwrap();
    assert_eq!(request.provenance["task.context"], Origin::Caller);
    assert!(arbiter.snapshot(&host_key).is_err());
    let inventory = crate::devices::Inventory {
        observed_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis(),
        devices: vec![crate::devices::Observation {
            id: "cpu:0".into(),
            class: "host-memory".into(),
            vendor: "host".into(),
            ceiling: Some(32 * 1024 * 1024 * 1024),
            available: Some(32 * 1024 * 1024 * 1024),
            backends: vec!["cpu".into()],
            exclusion: None,
            provenance: "planning-only observation control".into(),
            cuda_ordinal: None,
            cuda_cc: None,
        }],
    };
    let inspection = crate::run::cpu::Plan::new(request, inventory).unwrap();
    inspection.require_proposal().unwrap();
    assert!(inspection.dry_run());
    assert_eq!(inspection.decision()["reservation"], "none");
    assert_eq!(inspection.decision()["selected"], "cpu:0");
    assert_eq!(inspection.decision()["source_verified"], false);
    assert_eq!(
        inspection.decision()["model_execution_qualified_by_metadata"],
        false
    );
    assert!(matches!(inspection.prepare(), Err(crate::Error::State(_))));
    assert!(
        arbiter.snapshot(&host_key).is_err(),
        "cold inspection registered execution credit"
    );
    let refused_key = PhysicalDeviceKey::new("host", "direct-sdk-refusal").unwrap();
    arbiter.register(refused_key.clone(), 1).unwrap();
    let mut admission = TransferAdmission {
        arbiter: arbiter.clone(),
        resource: refused_key.clone(),
        owner: ResourceOwner::new("fixture", "direct-sdk").unwrap(),
    };
    assert!(matches!(
        Selection::resolve(&source, 8, None)
            .unwrap()
            .prepare(&admission),
        Err(crate::Error::Limit(_))
    ));
    assert_eq!(arbiter.snapshot(&refused_key).unwrap().reserved, 0);

    let key = PhysicalDeviceKey::new("host", "system-ram").unwrap();
    arbiter
        .register(key.clone(), 8 * 1024 * 1024 * 1024)
        .unwrap();
    admission.resource = key.clone();
    let too_small = Selection::resolve(&source, 8, None)
        .unwrap()
        .prepare(&admission)
        .unwrap();
    assert!(matches!(
        cpu_text::generate(
            too_small,
            TextOptions {
                prompt: "!".into(),
                context: 9,
                max_tokens: 1,
                raw: true,
                prefill_chunk: 1,
                teacher_ids: None,
            },
            Diagnostics::default(),
            |_| panic!("execution exceeded admitted context")
        ),
        Err(crate::Error::Limit(_))
    ));
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    let selected = Selection::resolve(&source, 8, None).unwrap();
    let planned = selected.planning_bytes();
    let prepared = selected.prepare(&admission).unwrap();
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, planned);
    {
        use crate::models::{ModelProgram, ProgramRequest};
        use std::sync::Arc;
        // Real family recording, not GPU preparation or an execution claim.
        let model = prepared.prepared.model().clone();
        let original: Arc<dyn zerogravity_core::source::TensorSource> = model.clone();
        let request = ProgramRequest {
            task: crate::task::TaskKind::Language,
            context: 8,
            logits: true,
            observations: true,
        };
        assert!(matches!(
            ModelProgram::record(
                model.clone(),
                ProgramRequest {
                    task: crate::task::TaskKind::Decision,
                    ..request
                }
            ),
            Err(crate::Error::Unsupported(_))
        ));
        assert!(matches!(
            ModelProgram::record(
                model.clone(),
                ProgramRequest {
                    context: usize::MAX,
                    ..request
                }
            ),
            Err(crate::Error::Limit(_))
        ));
        let program = ModelProgram::record(model, request).unwrap();
        assert!(Arc::ptr_eq(&original, program.source()));
        assert_eq!(program.program().source(), original.source_identity());
        assert_eq!(program.program().prefix_capacity(), 8);
        assert!(program.program().logits().is_some());
        assert_eq!(program.program().required_domains().len(), 24);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, planned);
    }
    let progress = Rc::new(Cell::new(0));
    let mut layers = Audit {
        progress: progress.clone(),
        opened_at: None,
        bytes: Vec::new(),
    };
    let mut logits = Audit {
        progress: progress.clone(),
        opened_at: None,
        bytes: Vec::new(),
    };
    let mut top_steps = Vec::new();
    let output = cpu_text::generate(
        prepared,
        TextOptions {
            prompt: "!\n!".into(),
            context: 8,
            max_tokens: 2,
            raw: true,
            prefill_chunk: 1,
            teacher_ids: Some(vec![1, 0]),
        },
        Diagnostics {
            logits: Some(&mut logits),
            layers: Some(&mut layers),
            top_logits: true,
        },
        |event| {
            assert_eq!(
                arbiter.snapshot(&key)?.reserved,
                planned,
                "source/control credit released during execution"
            );
            match event {
                Observation::Prefill { done, total } => {
                    assert_eq!(total, 3);
                    progress.set(done);
                }
                Observation::Logits { step, values } => {
                    assert_eq!(values.len(), 8);
                    top_steps.push(step);
                }
            }
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(output.generated(), [1, 0]);
    assert_eq!(output.text(), "\"!");
    assert_eq!(output.consumed(), 4);
    assert_eq!(layers.opened_at, Some(0));
    assert_eq!(logits.opened_at, Some(3));
    assert_eq!(top_steps, [0, 1]);
    assert!(!layers.bytes.is_empty());
    for line in String::from_utf8(layers.bytes).unwrap().lines() {
        serde_json::from_str::<serde_json::Value>(line).unwrap();
    }
    let rows: Vec<serde_json::Value> = String::from_utf8(logits.bytes)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    for (step, row) in rows.iter().enumerate() {
        assert_eq!(row["stage"], "decode");
        assert_eq!(row["step"], step);
        assert_eq!(row["prefill_tokens"], 3);
        assert_eq!(row["prefill_chunk"], 1);
        assert_eq!(row["position"], 3 + step);
        assert_eq!(row["selected_token"], 0);
        assert_eq!(row["committed_token"], [1, 0][step]);
        assert!(row["state_bytes"].as_u64().unwrap() > 0);
        assert_eq!(row["top"].as_array().unwrap().len(), 20);
    }
    assert_ne!(rows[0]["root_id"], rows[1]["root_id"]);
    assert_eq!(
        arbiter.snapshot(&key).unwrap().reserved,
        planned,
        "returned payload lost its original credit"
    );
    drop(output);
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    assert!(!arbiter.snapshot(&key).unwrap().fenced);
    std::fs::remove_dir_all(root).unwrap();
    println!("{DONE}");
}
