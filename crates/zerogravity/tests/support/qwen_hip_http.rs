//! Public loopback HTTP on the same runtime/worker; supervised private netns only.
#[path = "qwen_hip_http/progress.rs"]
mod progress;

use super::{population::Harness, *};
use serde_json::json;
use std::{collections::BTreeSet, net::TcpStream, sync::Barrier};

#[path = "qwen_source_http.rs"]
mod protocol;
use protocol::*;

#[test]
#[ignore = "approved gfx1151/source, private loopback netns, fresh release process"]
fn public_source_http_population_json_sse_disconnect_and_replay() {
    let label = std::env::var("ZG_QWEN_HIP_SDK_MODEL").unwrap();
    let clients: usize = std::env::var("ZG_QWEN_HIP_CLIENTS")
        .unwrap()
        .parse()
        .unwrap();
    let mode = std::env::var("ZG_QWEN_HIP_POPULATION").unwrap();
    assert!([1, 4, 8, 16, 32, 64].contains(&clients));
    assert!(matches!(mode.as_str(), "complete" | "disconnect"));
    // The launch wrapper verifies namespace ownership before enabling only lo.
    assert_eq!(std::env::var("ZG_PRIVATE_LOOPBACK").as_deref(), Ok("1"));
    let h = Harness::open(&label, clients.min(8), 514, 3, clients + 1);
    let (prompt, expected) = h.canonical();
    let expected_text = h.tokenizer.decode(&expected, true).unwrap();
    let server = h
        .runtime
        .server()
        .bind("127.0.0.1:0".parse().unwrap())
        .limits(zerogravity::serving::ingress::IngressLimits {
            connections: clients,
            body_bytes: 64 * 1024,
            output_bytes: 64 * 1024,
        })
        .enable_per_request_metrics(true)
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    refuse_chat(address);
    let barrier = Barrier::new(clients);
    let outcomes = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for i in 0..clients {
            let disconnect = mode == "disconnect" && i % 3 == 0;
            let stream = disconnect || i % 2 != 0;
            let body = json!({"model":if i%2==0 {"source"} else {"alias"},"prompt":prompt,"max_tokens":3,"stream":stream,"return_token_ids":true});
            let (barrier, expected, text) = (&barrier, &expected, &expected_text);
            handles.push(scope.spawn(move || {
                barrier.wait();
                completion(
                    address,
                    body,
                    expected,
                    text,
                    &format!("population-{i}"),
                    disconnect,
                    47,
                )
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    let mut responses = BTreeSet::new();
    let mut sessions = BTreeSet::new();
    let mut disconnects = 0;
    for outcome in outcomes {
        if let Some((response, session)) = outcome {
            assert!(responses.insert(response));
            assert!(sessions.insert(session));
        } else {
            disconnects += 1;
        }
    }
    assert_eq!(responses.len() + disconnects, clients);
    let metrics = server.metrics();
    server.shutdown().unwrap();
    assert!(TcpStream::connect(address).is_err());
    assert!(!h.runtime.status().closed);
    let replay = complete(
        h.runtime.request_text(request("alias", &prompt)).unwrap(),
        47,
        h.tokenizer.as_ref(),
    );
    assert_eq!(replay.ids, expected);
    assert!(sessions.insert(replay.roots[0].producer_root.unwrap().session_id));
    let admission = h.admission();
    assert_eq!(admission["failed_gpu_quanta"], 0);
    assert_eq!(admission["gpu_fused_batches"], 0);
    println!(
        "SOURCE_HTTP_OUTCOME model={label} mode={mode} offered={clients} completed_readers={} disconnect_requested={disconnects} active_limit={} accepted_http_requests={} fused_cohorts=false original_ids=true sdk_replay=true full_float_comparisons=0 admission={admission}",
        responses.len(),
        clients.min(8),
        metrics.accepted_requests
    );
    h.retire();
    println!("SOURCE_HTTP_PASS model={label} mode={mode} offered={clients}");
}

#[test]
#[ignore = "approved original reference, gfx1151, private loopback, fresh release process"]
fn public_source_http_completion_disconnect_and_replay() {
    if cfg!(debug_assertions) {
        panic!("release physical target required");
    }
    assert_eq!(std::env::var("ZG_PRIVATE_LOOPBACK").as_deref(), Ok("1"));
    let label = std::env::var("ZG_QWEN_HIP_SDK_MODEL").unwrap();
    let h = Harness::open(&label, 1, 514, 3, 2);
    let (prompt, expected) = h.canonical();
    let text = h.tokenizer.decode(&expected, true).unwrap();
    let server = h
        .runtime
        .server()
        .bind("127.0.0.1:0".parse().unwrap())
        .limits(zerogravity::serving::ingress::IngressLimits {
            connections: 2,
            body_bytes: 64 * 1024,
            output_bytes: 64 * 1024,
        })
        .enable_per_request_metrics(true)
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    refuse_chat(address);
    let mut responses = BTreeSet::new();
    let mut sessions = BTreeSet::new();
    for (id, alias, streaming, disconnect) in [
        ("json", "source", false, false),
        ("sse", "alias", true, false),
        ("disconnect", "source", true, true),
    ] {
        let body = json!({"model":alias,"prompt":prompt,"max_tokens":3,"stream":streaming,"return_token_ids":true});
        let outcome = completion(address, body, &expected, &text, id, disconnect, 47);
        if disconnect {
            assert!(outcome.is_none());
        } else {
            let (response, session) = outcome.unwrap();
            assert!(responses.insert(response));
            assert!(sessions.insert(session));
        }
    }
    assert_eq!(responses.len(), 2);
    server.shutdown().unwrap();
    assert!(TcpStream::connect(address).is_err());
    assert!(!h.runtime.status().closed);
    let replay = complete(
        h.runtime.request_text(request("alias", &prompt)).unwrap(),
        47,
        h.tokenizer.as_ref(),
    );
    assert_eq!(replay.ids, expected);
    assert!(sessions.insert(replay.roots[0].producer_root.unwrap().session_id));
    let admission = h.admission();
    assert_eq!(admission["failed_gpu_quanta"], 0);
    assert_eq!(admission["queued_requests"], 0);
    assert_eq!(admission["gpu_fused_batches"], 0);
    println!(
        "SOURCE_HTTP_SEQUENTIAL_OUTCOME model={label} concurrent_clients=1 active_limit=1 completed_readers=2 disconnect_requested=1 json=true sse=true original_ids=true sdk_replay=true fused_cohorts=false full_float_comparisons=0 admission={admission}"
    );
    h.retire();
    println!("SOURCE_HTTP_SEQUENTIAL_PASS model={label}");
}

#[test]
#[ignore = "approved populated original, gfx1151, private loopback, fresh release process"]
fn public_source_populated_http_json_sse_disconnect_and_replay() {
    let label = std::env::var("ZG_QWEN_HIP_SDK_MODEL").unwrap();
    let capacity: usize = std::env::var("ZG_QWEN_CONTEXT").unwrap().parse().unwrap();
    let reference = super::contexts::ContextReference::open(&label);
    let case = reference.case(capacity);
    let mode = std::env::var("ZG_QWEN_HIP_POPULATION").unwrap();
    assert!(matches!(mode.as_str(), "complete" | "disconnect"));
    assert_eq!(std::env::var("ZG_PRIVATE_LOOPBACK").as_deref(), Ok("1"));
    let h = Harness::open(&label, 2, capacity, 3, 3);
    let (prompt, expected) = reference.trajectory(case, capacity, h.tokenizer.as_ref());
    let expected_text = h.tokenizer.decode(&expected, true).unwrap();
    let server = h
        .runtime
        .server()
        .bind("127.0.0.1:0".parse().unwrap())
        .limits(zerogravity::serving::ingress::IngressLimits {
            connections: 2,
            body_bytes: 128 * 1024,
            output_bytes: 64 * 1024,
        })
        .enable_per_request_metrics(true)
        .build()
        .unwrap()
        .start()
        .unwrap();
    let address = server.local_addr();
    // Long transport framing here; SDK/context and population cover concurrency.
    let outcomes = (0..2).map(|i| {
        let body = json!({"model":if i == 0 {"source"} else {"alias"},"prompt":prompt,"max_tokens":3,"stream":i==0,"return_token_ids":true});
        assert!(serde_json::to_vec(&body).unwrap().len() <= 128*1024);
        completion(address,body,&expected,&expected_text,&format!("context-{i}"),mode == "disconnect" && i == 0,capacity-2)
    }).collect::<Vec<_>>();
    let mut sessions = BTreeSet::new();
    let mut responses = BTreeSet::new();
    let mut disconnects = 0;
    for outcome in outcomes {
        if let Some((response, session)) = outcome {
            assert!(responses.insert(response));
            assert!(sessions.insert(session));
        } else {
            disconnects += 1;
        }
    }
    assert_eq!(disconnects, usize::from(mode == "disconnect"));
    assert_eq!(sessions.len() + disconnects, 2);
    server.shutdown().unwrap();
    assert!(TcpStream::connect(address).is_err());
    assert!(!h.runtime.status().closed);
    let replay = complete(
        h.runtime.request_text(request("source", &prompt)).unwrap(),
        capacity - 2,
        h.tokenizer.as_ref(),
    );
    assert_eq!(replay.ids, expected);
    assert!(sessions.insert(replay.roots[0].producer_root.unwrap().session_id));
    let admission = h.admission();
    assert_eq!(admission["failed_gpu_quanta"], 0);
    assert_eq!(admission["queued_requests"], 0);
    assert_eq!(admission["gpu_fused_batches"], 0);
    println!(
        "SOURCE_CONTEXT_HTTP_OUTCOME model={label} context={capacity} mode={mode} offered=2 concurrent_clients=1 completed_readers={} disconnect_requested={disconnects} prompt_tokens={} original_ids=true sdk_replay=true fused_cohorts=false full_float_comparisons=0 admission={admission}",
        responses.len(),
        capacity - 2
    );
    drop(reference);
    h.retire();
    println!("SOURCE_CONTEXT_HTTP_PASS model={label} context={capacity} mode={mode}");
}
