//! Pure merges use the same map entry, overlay and ordering as live discovery.
use super::super::*;
use crate::{
    Error,
    devices::allowed,
    pipeline::Config,
    run_policy::{
        Backend, DeviceRef, Devices, Diagnostics, Entrypoint, Headroom, Request, Speculation, Task,
    },
};

fn selected(inventory: &Inventory, selectors: &[&str]) -> Result<Vec<Observation>> {
    let request = Request {
        graph: Config {
            models: vec![],
            pipelines: vec![],
            routes: vec![],
        },
        entrypoint: Entrypoint::Model("not-opened".into()),
        task: Task::default(),
        backend: Backend::Auto,
        devices: Devices::Ordered(
            selectors
                .iter()
                .map(|s| DeviceRef::parse(s).unwrap())
                .collect(),
        ),
        headroom: Headroom::Default,
        numerical: None,
        speculation: Speculation::Off,
        draft_device: None,
        diagnostics: Diagnostics::default(),
        provenance: Default::default(),
        vendor_filter: None,
    };
    allowed(&request, inventory)
}

fn host_snapshot() -> Observation {
    Observation {
        id: "cpu:0".into(),
        class: "host-memory".into(),
        vendor: "cpu".into(),
        ceiling: None,
        available: None,
        backends: vec![],
        exclusion: Some("current host availability unavailable".into()),
        provenance: "synthetic snapshot".into(),
        cuda_ordinal: None,
        cuda_cc: None,
    }
}

#[test]
fn drm_and_cuda_share_one_normalized_physical_row_and_stable_index_aliases() {
    for reversed in [false, true] {
        let mut cards = BTreeMap::new();
        let mut drm = [
            ("00000000:02:00.0", "nvidia"),
            ("0000:01:00.0", "amd"),
            ("0000:03:00.0", "nvidia"),
        ];
        if reversed {
            drm.reverse();
        }
        for (pci, vendor) in drm {
            record_drm(&mut cards, pci, vendor, Some(12_000), Some(1000)).unwrap();
        }
        let before = sorted_cards(cards.clone())
            .into_iter()
            .map(|d| d.id)
            .collect::<Vec<_>>();
        // Driver ordinals deliberately differ from physical inventory aliases.
        for (pci, ordinal) in [("0000:02:00.0", 7), ("00000000:03:00.0", 1)] {
            let row = physical_card(&mut cards, pci, "nvidia").unwrap();
            record_cuda(row, 8000, ordinal, (8, 6), Some((9000, 6000)), false);
        }
        assert_eq!(cards.len(), 3, "one row per full PCI identity");
        assert!(cards.iter().all(|(key, row)| key == &row.id));
        let mut devices = sorted_cards(cards);
        assert_eq!(
            devices.iter().map(|d| d.id.clone()).collect::<Vec<_>>(),
            before
        );
        assert_eq!(
            before,
            ["pci:0000:02:00.0", "pci:0000:03:00.0", "pci:0000:01:00.0"]
        );
        assert_eq!(devices[0].ceiling, Some(8000));
        assert_eq!(devices[0].available, Some(6000));
        assert_eq!(devices[0].cuda_ordinal, Some(7));
        assert_eq!(devices[0].cuda_cc, Some((8, 6)));
        assert_eq!(devices[0].exclusion, None);
        assert_eq!(
            devices[0]
                .backends
                .iter()
                .map(String::as_str)
                .filter(|b| b.starts_with("cuda"))
                .collect::<Vec<_>>(),
            vec!["cuda:physical-binding-observed"]
        );
        if cfg!(feature = "vulkan") {
            assert!(
                devices[0]
                    .backends
                    .iter()
                    .any(|b| b == "vulkan:binding-unverified")
            );
        }
        assert!(
            devices[2].exclusion.is_some(),
            "DRM-only AMD stays unqualified"
        );
        devices.push(host_snapshot());
        let inventory = Inventory {
            devices,
            observed_unix_ms: 0,
        };
        for (index, pci) in [
            (0, "pci:0000:02:00.0"),
            (1, "pci:0000:03:00.0"),
            (2, "pci:0000:01:00.0"),
        ] {
            let alias = format!("gpu:{index}");
            let rows = selected(&inventory, &[&alias, pci, "cpu:0"]).unwrap();
            assert_eq!(rows.len(), 2, "index and PCI aliases deduplicate");
            assert_eq!(rows[0].id, pci);
            assert_eq!(rows[1].id, "cpu:0");
        }
        assert!(matches!(
            selected(&inventory, &["gpu:3"]),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            selected(&inventory, &["pci:0000:09:00.0"]),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            selected(&inventory, &["cpu:0", "gpu:0"]),
            Err(Error::Malformed(_))
        ));
    }
}

#[test]
fn merged_rows_keep_nvml_unknown_sm86_and_partition_refusals() {
    let mut cards = BTreeMap::new();
    record_drm(
        &mut cards,
        "0000:02:00.0",
        "nvidia",
        Some(12_000),
        Some(1000),
    )
    .unwrap();
    let row = physical_card(&mut cards, "00000000:02:00.0", "nvidia").unwrap();
    record_cuda(row, 8000, 7, (8, 6), None, false);
    assert_eq!(row.ceiling, Some(8000));
    assert_eq!(
        row.available, None,
        "never borrow sysfs free bytes for CUDA"
    );
    assert_eq!(
        row.exclusion.as_deref(),
        Some("NVML current free/capacity binding unavailable")
    );
    record_cuda(row, 8000, 7, (8, 9), Some((7000, 6000)), false);
    assert_eq!(row.ceiling, Some(7000));
    assert_eq!(
        row.exclusion.as_deref(),
        Some("finite Qwen CUDA route requires qualified sm86 profile")
    );
    record_cuda(row, 8000, 7, (8, 6), Some((7000, 6000)), false);
    assert!(row.exclusion.is_none());
    record_cuda(row, 8000, 8, (8, 6), Some((7000, 6000)), true);
    assert_eq!(row.cuda_ordinal, None);
    assert_eq!(row.available, None);
    assert_eq!(
        row.exclusion.as_deref(),
        Some("multiple CUDA views share one PCI identity (MIG/partition ambiguous)")
    );
    // Further duplicate CUDA views cannot restore an unambiguous binding.
    record_cuda(row, 8000, 9, (8, 6), Some((7000, 6000)), true);
    assert_eq!(row.cuda_ordinal, None);
    assert_eq!(row.available, None);
    assert_eq!(cards.len(), 1);
}

#[test]
fn missing_or_malformed_pci_never_becomes_an_ordinal_or_another_card() {
    let mut cards = BTreeMap::new();
    for pci in [
        "",
        "unknown",
        "gpu:0",
        "02:00.0",
        "0000:02:20.0",
        "0000:02:00.8",
    ] {
        assert!(record_drm(&mut cards, pci, "nvidia", Some(1), Some(0)).is_err());
        assert!(physical_card(&mut cards, pci, "nvidia").is_err());
        assert!(cards.is_empty(), "{pci}");
    }
    // CUDA-only observation is still one real full PCI identity, not gpu:N.
    let row = physical_card(&mut cards, "00000000:05:00.0", "nvidia").unwrap();
    record_cuda(row, 8000, 0, (8, 6), None, false);
    assert_eq!(row.id, "pci:0000:05:00.0");
    assert_eq!(row.available, None);
    let inventory = Inventory {
        devices: sorted_cards(cards),
        observed_unix_ms: 0,
    };
    assert_eq!(
        selected(&inventory, &["gpu:0"]).unwrap()[0].id,
        "pci:0000:05:00.0"
    );
    assert!(matches!(
        selected(&inventory, &["pci:0000:00:00.0"]),
        Err(Error::Unsupported(_))
    ));
}
