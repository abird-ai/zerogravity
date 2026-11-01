//! Device presentation; observations and selector policy belong to the SDK.
#[cfg(all(test, target_os = "linux"))]
pub(super) use zerogravity::devices::device_node_identity;
#[cfg(any(
    all(feature = "cpu", feature = "model-qwen35", feature = "text"),
    all(feature = "qwen35-cuda-qualification", target_os = "linux")
))]
pub(super) use zerogravity::devices::fresh;
#[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
pub(super) use zerogravity::devices::host_available;
#[cfg(test)]
pub(super) use zerogravity::devices::normalize_pci;
pub(super) use zerogravity::devices::{Inventory, Observation, allowed, discover};

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
