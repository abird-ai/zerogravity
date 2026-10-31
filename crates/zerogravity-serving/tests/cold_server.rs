use zerogravity_engine::Runtime;
use zerogravity_serving::{Error, Server, ingress::IngressLimits, openai::Endpoint};

#[test]
fn cold_builder_reports_configuration_and_start_refuses() {
    let runtime = Runtime::builder().build().unwrap();
    let address = "127.0.0.1:8080".parse().unwrap();
    let server = Server::builder(runtime.clone())
        .bind(address)
        .build()
        .unwrap();
    assert_eq!(server.address(), address);
    assert_eq!(server.metrics().accepted_requests, 0);
    assert!(matches!(server.start(), Err(Error::Unsupported(_))));
    assert!(!runtime.status().closed);
    // Cold configuration is not a live listener or an actual bound-address receipt.
    let server = Server::builder(runtime.clone()).build().unwrap();
    runtime.shutdown().unwrap();
    assert!(matches!(server.start(), Err(Error::State(_))));
    assert!(Server::builder(runtime).build().is_err());
}

#[test]
fn external_bind_and_unimplemented_endpoints_refuse() {
    let runtime = Runtime::builder().build().unwrap();
    for address in ["0.0.0.0:8080", "[::]:8080", "192.0.2.1:8080"] {
        assert!(matches!(
            Server::builder(runtime.clone())
                .bind(address.parse().unwrap())
                .build(),
            Err(Error::Unsupported(_))
        ));
    }
    assert!(
        Server::builder(runtime)
            .bind("[::1]:8080".parse().unwrap())
            .build()
            .is_ok()
    );
    for endpoint in [
        Endpoint::Health,
        Endpoint::Models,
        Endpoint::Completions,
        Endpoint::ChatCompletions,
        Endpoint::Embeddings,
        Endpoint::Responses,
    ] {
        assert!(matches!(
            endpoint.require_implemented(),
            Err(Error::Unsupported(_))
        ));
    }
    assert!(matches!(
        zerogravity_serving::responses::require_implemented(),
        Err(Error::Unsupported(_))
    ));
    assert!(
        zerogravity_serving::realtime::RealtimeTransport::WebSocket
            .require_implemented()
            .is_err()
    );
}

#[test]
fn ingress_refuses_zero_and_overflow_before_any_server_exists() {
    for limits in [
        IngressLimits {
            connections: 0,
            body_bytes: 1,
            output_bytes: 1,
        },
        IngressLimits {
            connections: 1,
            body_bytes: 0,
            output_bytes: 1,
        },
        IngressLimits {
            connections: 1,
            body_bytes: 1,
            output_bytes: 0,
        },
        IngressLimits {
            connections: usize::MAX,
            body_bytes: 2,
            output_bytes: 1,
        },
        IngressLimits {
            connections: usize::MAX,
            body_bytes: 1,
            output_bytes: 2,
        },
    ] {
        assert!(matches!(limits.validate(), Err(Error::Limit(_))));
    }
}
