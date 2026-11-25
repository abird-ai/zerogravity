//! Physical SDK chain compared with ordinary sequential requests on the same
//! loaded models. No new model fixtures or reference/capability grants.
use std::{num::NonZeroUsize, time::Duration};
use zerogravity::{
    Runtime,
    pipelines::{
        Config, LanguageBinding, ModelInstance, Pipeline, PipelineEdge, PipelineNode,
        PreparedLanguagePipeline, ValueKind,
    },
    run::{OutputPoll, TaskTarget, TextControls, TextTask},
    runtime::text::{Request, ResponseEvent},
};

pub fn graph(models: Vec<ModelInstance>) -> Config {
    let steps: Vec<_> = models
        .iter()
        .enumerate()
        .map(|(index, model)| PipelineNode {
            id: format!("stage{index}"),
            kind: "llm".into(),
            model: Some(model.id.clone()),
        })
        .collect();
    let mut links = Vec::new();
    let mut from = "input.value".to_string();
    for stage in &steps {
        let endpoint = format!("{}.text", stage.id);
        links.push(PipelineEdge {
            from,
            to: endpoint.clone(),
        });
        from = endpoint;
    }
    links.push(PipelineEdge {
        from,
        to: "output.value".into(),
    });
    Config {
        models,
        pipelines: vec![Pipeline {
            name: "chain".into(),
            input: ValueKind::Text,
            output: ValueKind::Text,
            realtime: false,
            chat_context: None,
            steps,
            links,
        }],
        routes: vec![],
    }
}

#[allow(dead_code)]
pub fn compare(
    runtime: &Runtime,
    target: &str,
    aliases: &[&str],
    prompt: &str,
    limit: usize,
) -> PreparedLanguagePipeline {
    let owners: Vec<_> = aliases
        .iter()
        .map(|alias| runtime.models().resolve(alias).unwrap())
        .collect();
    let models: Vec<_> = aliases
        .iter()
        .enumerate()
        .map(|(index, _)| ModelInstance {
            id: format!("model{index}"),
            target: target.into(),
            source: None,
            model_config: None,
            tokenizer: None,
            snapshot: None,
            native_artifacts: None,
        })
        .collect();
    let graph = graph(models);
    let bindings: Vec<_> = graph
        .models
        .iter()
        .zip(aliases)
        .zip(&owners)
        .map(|((model, alias), owner)| LanguageBinding {
            model: &model.id,
            alias,
            owner,
        })
        .collect();
    let plan = runtime
        .prepare_pipeline(&graph, "chain", &bindings)
        .unwrap();
    compare_prepared(runtime, &plan, prompt, limit);
    plan
}

pub fn compare_prepared(
    runtime: &Runtime,
    plan: &PreparedLanguagePipeline,
    prompt: &str,
    limit: usize,
) {
    let mut input = prompt.to_owned();
    let mut expected = None;
    for stage in plan.stages() {
        assert!(
            !input.is_empty(),
            "physical chain needs nonempty intermediate text"
        );
        let events = runtime
            .request_text(Request {
                model: stage.alias().into(),
                input,
                chat: false,
                max_tokens: limit,
                stream: false,
                token_metrics: None,
            })
            .unwrap();
        let ResponseEvent::Done {
            output,
            prompt_tokens,
            completion_tokens,
            finish_reason,
        } = events
            .events
            .recv_timeout(Duration::from_secs(120))
            .unwrap()
        else {
            panic!("direct physical chain request failed");
        };
        expected = Some((
            output.clone(),
            prompt_tokens,
            completion_tokens,
            finish_reason,
        ));
        input = output;
    }
    let task = TextTask {
        target: TaskTarget::Pipeline(plan),
        input: prompt,
        controls: TextControls {
            observe_tokens: true,
            ..TextControls::new(NonZeroUsize::new(limit).unwrap())
        },
    };
    let selected = runtime.inspect_task(task).unwrap();
    assert_eq!(selected.pipeline.as_deref(), Some(plan.name()));
    assert_eq!(selected.stages.len(), plan.stages().len());
    assert_eq!(selected.controls, task.controls);
    for (actual, stage) in selected.stages.iter().zip(plan.stages()) {
        assert_eq!(actual.alias, stage.alias());
        assert_eq!(actual.registration_id, stage.registration_id());
        let owner = selected
            .models
            .iter()
            .find(|owner| owner.registration_id == actual.registration_id)
            .unwrap();
        assert_eq!(owner.source, stage.source());
        let execution = owner
            .capabilities
            .as_ref()
            .unwrap()
            .details
            .execution
            .as_ref()
            .unwrap();
        if execution.selected.provider == zerogravity::provider::ProviderKind::Cpu {
            assert!(execution.program.is_none());
        } else {
            let recorded = execution.program.expect("actual prepared GPU program");
            assert_eq!(recorded.source, owner.source);
            assert!(recorded.node_count > 0 && recorded.context_capacity > 0);
            assert!(
                execution.region.is_none(),
                "whole model is not a selected partial region"
            );
            if execution.text_calls.is_some() {
                assert!(recorded.source_template.is_some());
            }
        }
    }
    let mut stream = runtime.submit_task(task).unwrap();
    let mut deltas = String::new();
    let mut terminal = None;
    let mut observed = 0;
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    loop {
        assert!(std::time::Instant::now() < deadline);
        let event = match stream.poll_next(Duration::from_millis(100)).unwrap() {
            OutputPoll::Pending => continue,
            OutputPoll::Finished => break,
            OutputPoll::Event(event) => event,
        };
        assert!(terminal.is_none());
        match event {
            ResponseEvent::TokenText { text, token } => {
                assert_eq!(token.index, observed);
                observed += 1;
                deltas.push_str(&text);
            }
            ResponseEvent::Done {
                output,
                prompt_tokens,
                completion_tokens,
                finish_reason,
            } => terminal = Some((output, prompt_tokens, completion_tokens, finish_reason)),
            ResponseEvent::Error(error) => panic!("physical pipeline: {error}"),
            _ => panic!("unexpected pipeline event"),
        }
    }
    assert_eq!(terminal, expected);
    let (output, _, completion_tokens, _) = terminal.unwrap();
    assert_eq!(deltas, output);
    assert_eq!(
        observed, completion_tokens,
        "only final stage token observations"
    );
    println!(
        "LANGUAGE_PIPELINE_SDK_PASS stages={} selected_owner_count={} observed_final_tokens={}",
        plan.stages().len(),
        selected.models.len(),
        observed
    );
}
