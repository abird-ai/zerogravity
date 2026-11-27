#![cfg(all(
    feature = "cpu",
    feature = "text",
    feature = "cache",
    feature = "serve"
))]

use serde_json::{Value, json};
use std::{path::Path, time::Duration};
use zerogravity::{
    Runtime,
    models::ModelLoadRequest,
    runtime::{
        TextConfig,
        text::{ChatMessage, ChatRole, ResponseEvent, SemanticRequest, TextInput},
    },
    serving::TextService,
};

#[test]
#[ignore = "requires existing approved Qwen 0.8B GGUF and installed OpenAI SDK"]
fn actual_realtime_text_through_openai_sdk() {
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/models/qwen35-0.8b/model.gguf"
    ));
    let runtime = Runtime::builder().build().unwrap();
    runtime
        .models()
        .load(ModelLoadRequest::CpuText {
            declared: vec![
                ("primary".into(), path.into()),
                ("alias".into(), path.into()),
            ],
            config: TextConfig {
                max_context: 128,
                max_output: 8,
                host_mib: 8192,
                ..Default::default()
            },
        })
        .unwrap();
    let registration = runtime.realtime_text_registration("alias").unwrap();
    assert_eq!(
        registration,
        runtime.realtime_text_registration("primary").unwrap()
    );
    assert!(runtime.realtime_text_registration("missing").is_err());
    let mut turns = vec![ChatMessage {
        role: ChatRole::System,
        text: "Answer briefly.".into(),
    }];
    let mut expected = Vec::<Value>::new();
    for input in ["Say hello.", "Say goodbye."] {
        turns.push(ChatMessage {
            role: ChatRole::User,
            text: input.into(),
        });
        let request = |stream| SemanticRequest {
            model: "alias".into(),
            input: TextInput::Conversation(turns.clone()),
            max_tokens: 8,
            stream,
            token_metrics: None,
        };
        assert_eq!(
            runtime
                .request_realtime_text_admitted(request(true), registration + 1, vec![], None)
                .err()
                .unwrap()
                .code,
            "model_changed"
        );
        let receipt = runtime.request_semantic(request(false)).unwrap();
        let ResponseEvent::Done {
            output,
            prompt_tokens,
            cached_prompt_tokens,
            completion_tokens,
            finish_reason,
        } = receipt
            .events
            .recv_timeout(Duration::from_secs(120))
            .unwrap()
        else {
            panic!("original model terminal")
        };
        expected.push(json!({"input":input,"text":output,"status":if finish_reason=="stop" {"completed"} else {"incomplete"},
            "input_tokens":prompt_tokens,"output_tokens":completion_tokens,"cached_tokens":cached_prompt_tokens}));
        if finish_reason == "stop" {
            turns.push(ChatMessage {
                role: ChatRole::Assistant,
                text: output,
            });
        }
        drop(receipt);
    }
    let server = runtime
        .server()
        .bind("127.0.0.1:0".parse().unwrap())
        .api_key(Some("secret".into()))
        .build()
        .unwrap()
        .start()
        .unwrap();
    let python = std::env::var_os("ZG_TEST_OPENAI_PYTHON").expect("installed SDK interpreter");
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/realtime_sdk.py");
    let status = std::process::Command::new(python)
        .arg("-B")
        .arg(script)
        .arg(server.local_addr().to_string())
        .arg(serde_json::to_string(&expected).unwrap())
        .env(
            "PYTHONPATH",
            std::env::var_os("ZG_TEST_OPENAI_PYTHONPATH").unwrap_or_default(),
        )
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .status()
        .unwrap();
    assert!(status.success());
    server.shutdown().unwrap();
    for model in runtime.models().snapshot().unwrap() {
        runtime
            .models()
            .unload_registration(&model.aliases[0], model.registration_id)
            .unwrap();
    }
    assert!(runtime.models().snapshot().unwrap().is_empty());
    runtime.shutdown().unwrap();
}
