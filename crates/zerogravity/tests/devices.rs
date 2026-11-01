//! Facade-only device policy tests. All observations are synthetic; no live
//! discovery, driver initialization, device-node inspection or model source IO.
use zerogravity::{
    Error, Runtime,
    devices::{self, Inventory, Observation},
    pipeline::Config,
    run_policy::{
        Backend, DeviceRef, Devices, Diagnostics, Entrypoint, Headroom, Request, Speculation, Task,
    },
};

fn observation(id: &str, vendor: &str) -> Observation {
    Observation {
        id: id.into(),
        class: if vendor == "cpu" {
            "host-memory"
        } else {
            "gpu"
        }
        .into(),
        vendor: vendor.into(),
        ceiling: Some(8 * 1024u64.pow(3)),
        available: None,
        backends: vec![],
        exclusion: Some("unqualified source/runtime binding".into()),
        provenance: "pure metadata fixture".into(),
        cuda_ordinal: None,
        cuda_cc: None,
    }
}
fn inventory() -> Inventory {
    Inventory {
        devices: vec![
            observation("pci:0000:01:00.0", "amd"),
            observation("pci:0000:02:00.0", "nvidia"),
            observation("cpu:0", "cpu"),
        ],
        observed_unix_ms: 0,
    }
}
fn request(backend: Backend, selectors: &[&str]) -> Request {
    Request {
        graph: Config {
            models: vec![],
            pipelines: vec![],
            routes: vec![],
        },
        entrypoint: Entrypoint::Model("not-opened".into()),
        task: Task::default(),
        backend,
        devices: if selectors.is_empty() {
            Devices::Auto
        } else {
            Devices::Ordered(
                selectors
                    .iter()
                    .map(|s| DeviceRef::parse(s).unwrap())
                    .collect(),
            )
        },
        headroom: Headroom::Default,
        numerical: None,
        speculation: Speculation::Off,
        draft_device: None,
        diagnostics: Diagnostics::default(),
        provenance: Default::default(),
        vendor_filter: None,
    }
}

#[test]
fn full_physical_identity_and_selector_syntax_keep_existing_refusals() {
    assert_eq!(
        devices::normalize_pci("00000000:01:02.3").unwrap(),
        "0000:01:02.3"
    );
    for s in ["01:02.3", "0000:01:20.0", "0000:01:02.8"] {
        assert!(devices::normalize_pci(s).is_err(), "{s}");
    }
    for node in [
        "/dev/dri/renderD128",
        "/dev/dri/card0",
        "/dev/dri/by-path/pci-0000:01:00.0-render",
        "/dev/nvidia0",
    ] {
        let reference = DeviceRef::parse(node).unwrap();
        assert_eq!(reference.host, "local");
        assert_eq!(reference.selector, node);
        assert_eq!(
            Devices::parse(&[node.into()]).unwrap(),
            Devices::Ordered(vec![reference.clone()])
        );
        assert!(reference.label().ends_with(node));
    }
    for bad in [
        "/dev/",
        "/dev/dri/../kfd",
        "/dev/dri/./renderD128",
        "/dev/dri//renderD128",
        "/dev/dri/renderD128/",
        "/tmp/renderD128",
    ] {
        assert!(DeviceRef::parse(bad).is_err(), "{bad}");
    }
    assert!(Devices::parse(&["cpu".into(), "/dev/dri/renderD128".into()]).is_err());
    assert!(Devices::parse(&["/dev/dri/renderD128".into(), "cpu".into()]).is_ok());
}

#[test]
fn every_explicit_member_must_match_the_forced_backend() {
    let request = request(Backend::Cuda, &["gpu:0", "gpu:1"]);
    assert!(
        matches!(devices::allowed(&request, &inventory()), Err(Error::Malformed(m)) if m == "explicit device contradicts forced backend; every allowlist member must be compatible")
    );
    let cpu = request_with_devices(Backend::Cpu, vec![DeviceRef::parse("gpu:0").unwrap()]);
    assert!(devices::allowed(&cpu, &inventory()).is_err());
}
fn request_with_devices(backend: Backend, selected: Vec<DeviceRef>) -> Request {
    let mut request = request(backend, &[]);
    request.devices = Devices::Ordered(selected);
    request
}

#[test]
fn aliases_deduplicate_without_losing_capacity_holds_or_binding_metadata() {
    let mut inventory = inventory();
    inventory.devices[1].cuda_ordinal = Some(7);
    inventory.devices[1].cuda_cc = Some((8, 6));
    inventory.devices[1].backends = vec!["cuda:physical-binding-observed".into()];
    let request = request(Backend::Auto, &["gpu:1", "pci:00000000:02:00.0", "cpu:0"]);
    let selected = devices::allowed(&request, &inventory).unwrap();
    assert_eq!(selected.len(), 2);
    assert_eq!(
        serde_json::to_value(&selected[0]).unwrap(),
        serde_json::to_value(&inventory.devices[1]).unwrap()
    );
    assert_eq!(selected[1].id, "cpu:0");
    // Unknown availability and an exclusion remain visible, never "eligible".
    assert!(selected[0].available.is_none());
    assert_eq!(
        selected[0].exclusion.as_deref(),
        Some("unqualified source/runtime binding")
    );
}

#[test]
fn cpu_is_terminal_and_remote_or_missing_devices_never_fall_back() {
    let request = request(Backend::Auto, &["cpu:0", "gpu:1"]);
    assert!(
        matches!(devices::allowed(&request, &inventory()), Err(Error::Malformed(m)) if m == "CPU must be terminal after physical alias resolution")
    );
    let missing = request_with_devices(Backend::Auto, vec![DeviceRef::parse("gpu:9").unwrap()]);
    assert!(matches!(
        devices::allowed(&missing, &inventory()),
        Err(Error::Unsupported(_))
    ));
    let remote = request_with_devices(
        Backend::Auto,
        vec![DeviceRef {
            host: "other-host".into(),
            selector: "gpu:0".into(),
        }],
    );
    assert!(
        matches!(devices::allowed(&remote, &inventory()), Err(Error::Unsupported(m)) if m == "remote compute transport is not qualified")
    );
}

#[test]
fn automatic_backend_filtering_does_not_grant_execution_or_hide_unknown_capacity() {
    for (backend, expected) in [
        (Backend::Cpu, vec!["cpu:0"]),
        (Backend::Cuda, vec!["pci:0000:02:00.0"]),
        (Backend::Rocm, vec!["pci:0000:01:00.0"]),
        (
            Backend::Vulkan,
            vec!["pci:0000:01:00.0", "pci:0000:02:00.0"],
        ),
    ] {
        let selected = devices::allowed(&request(backend, &[]), &inventory()).unwrap();
        assert_eq!(
            selected.iter().map(|d| d.id.as_str()).collect::<Vec<_>>(),
            expected
        );
        assert!(
            selected
                .iter()
                .all(|d| d.available.is_none() && d.exclusion.is_some())
        );
    }
    let mut explicit = request(Backend::Auto, &["gpu:1"]);
    explicit.vendor_filter = Some("amd".into());
    assert!(
        matches!(devices::allowed(&explicit, &inventory()), Err(Error::Malformed(m)) if m == "explicit device contradicts alias vendor filter")
    );
    explicit.devices = Devices::Auto;
    let selected = devices::allowed(&explicit, &inventory()).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].vendor, "amd");
}

#[test]
fn runtime_handle_checks_open_before_inventory_or_selector_io() {
    let runtime = Runtime::builder().build().unwrap();
    let policy = request(Backend::Auto, &["gpu:1"]);
    let inventory = inventory();
    let selected = runtime.devices().allowed(&policy, &inventory).unwrap();
    assert_eq!(selected[0].id, "pci:0000:02:00.0");
    runtime.shutdown().unwrap();
    // No call to live inventory on an open runtime: shutdown must refuse first.
    assert!(matches!(
        runtime.devices().inventory(),
        Err(Error::State(_))
    ));
    assert!(matches!(
        runtime.devices().allowed(&policy, &inventory),
        Err(Error::State(_))
    ));
    #[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
    assert!(matches!(
        runtime.devices().host_available(0, &Headroom::Bytes(0)),
        Err(Error::State(_))
    ));
}
