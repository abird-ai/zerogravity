//! Device presentation; observations and selector policy belong to the SDK.
#[cfg(all(test, target_os = "linux"))]
pub(super) use zerogravity::devices::device_node_identity;
pub(super) use zerogravity::devices::{Inventory, Observation, discover};
#[cfg(test)]
pub(super) use zerogravity::devices::{allowed, normalize_pci};

/// Human inventory, not an execution-capability claim. No source is inspected.
pub(super) fn display(device: &Observation) -> String {
    let memory = |bytes: Option<u64>| {
        bytes.map_or_else(
            || "unknown".into(),
            |n| format!("{} MiB", n / (1024 * 1024)),
        )
    };
    let backend = if device.backends.is_empty() {
        "none".into()
    } else {
        device.backends.join(",")
    };
    let status = device.exclusion.as_deref().unwrap_or("inventory only");
    format!(
        "{}  {} {}  total={} available={}  backends={}  ({status})",
        device.id,
        device.vendor,
        device.class,
        memory(device.ceiling),
        memory(device.available),
        backend,
    )
}
