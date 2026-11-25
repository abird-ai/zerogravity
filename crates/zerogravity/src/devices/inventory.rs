//! OS/driver metadata only. No Client, model, primary context or ledger mutation.
use std::{collections::BTreeMap, io::Read, path::Path};

use serde::Serialize;

use super::{bad, now_ms};
use crate::Result;

/// Physical observation, never a source-qualified execution capability.
#[derive(Clone, Debug, Serialize)]
pub struct Observation {
    pub id: String,
    pub class: String,
    pub vendor: String,
    pub ceiling: Option<u64>,
    pub available: Option<u64>,
    pub backends: Vec<String>,
    pub exclusion: Option<String>,
    pub provenance: String,
    pub cuda_ordinal: Option<usize>,
    pub cuda_cc: Option<(i32, i32)>,
}

#[derive(Debug, Serialize)]
pub struct Inventory {
    pub devices: Vec<Observation>,
    pub observed_unix_ms: u128,
}

pub(super) fn read(path: &Path, limit: u64) -> Option<String> {
    let f = std::fs::File::open(path).ok()?;
    if !f.metadata().ok()?.is_file() {
        return None;
    }
    let mut s = String::new();
    f.take(limit + 1).read_to_string(&mut s).ok()?;
    (s.len() as u64 <= limit).then_some(s)
}

pub fn normalize_pci(s: &str) -> Result<String> {
    let (domain, tail) = s
        .split_once(':')
        .ok_or_else(|| bad("PCI identity requires complete domain:bus:device.function"))?;
    let (bus, tail) = tail
        .split_once(':')
        .ok_or_else(|| bad("PCI identity requires complete domain"))?;
    let (device, function) = tail
        .split_once('.')
        .ok_or_else(|| bad("PCI function required"))?;
    if !(domain.len() == 4 || domain.len() == 8)
        || bus.len() != 2
        || device.len() != 2
        || function.len() != 1
    {
        return Err(bad("invalid full PCI identity width"));
    }
    let hex =
        |s: &str| u32::from_str_radix(s, 16).map_err(|_| bad("invalid hexadecimal PCI identity"));
    let domain = hex(domain)?;
    let bus = hex(bus)?;
    let device = hex(device)?;
    let function = hex(function)?;
    if device > 31 || function > 7 {
        return Err(bad("PCI device/function out of range"));
    }
    Ok(format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}"))
}

pub(super) fn host() -> Observation {
    let observed = zerogravity_core::resource_arbiter::HostMemoryObservation::observe();
    Observation {
        id: "cpu:0".into(),
        class: "host-memory".into(),
        vendor: "cpu".into(),
        ceiling: observed.ceiling,
        available: observed.available,
        backends: if cfg!(feature = "cpu") {
            vec!["cpu".into()]
        } else {
            vec![]
        },
        exclusion: observed
            .available
            .is_none()
            .then(|| "current host availability unavailable".into()),
        provenance: observed.provenance,
        cuda_ordinal: None,
        cuda_cc: None,
    }
}

/// Both discovery sources use the same normalized full PCI key. Never invent
/// an ordinal-based identity when a physical binding is missing or malformed.
fn physical_card<'a>(
    cards: &'a mut BTreeMap<String, Observation>,
    pci: &str,
    vendor: &str,
) -> Result<&'a mut Observation> {
    let id = format!("pci:{}", normalize_pci(pci)?);
    Ok(cards.entry(id.clone()).or_insert_with(|| Observation {
        id,
        class: "gpu".into(),
        vendor: vendor.into(),
        ceiling: None,
        available: None,
        backends: Vec::new(),
        exclusion: None,
        provenance: String::new(),
        cuda_ordinal: None,
        cuda_cc: None,
    }))
}

fn record_drm(
    cards: &mut BTreeMap<String, Observation>,
    pci: &str,
    vendor: &str,
    total: Option<u64>,
    used: Option<u64>,
) -> Result<()> {
    let d = physical_card(cards, pci, vendor)?;
    d.vendor = vendor.into();
    d.cuda_ordinal = None;
    d.cuda_cc = None;
    d.ceiling = total;
    d.available = zerogravity_core::placement::LocalVramObservation { total, used }.available();
    d.backends.clear();
    if vendor == "nvidia" && cfg!(feature = "cuda") {
        d.backends.push("cuda:binding-unverified".into());
    }
    if vendor == "amd" && cfg!(feature = "hip") {
        d.backends.push("rocm:binding-unverified".into());
    }
    if cfg!(feature = "vulkan") {
        d.backends.push("vulkan:binding-unverified".into());
    }
    d.exclusion = Some("source-qualified cold/compiler cost and runtime physical binding unavailable in generic run".into());
    d.provenance =
        "DRM full PCI identity; local VRAM total/used only; host overlap unresolved, GTT is not another pool".into();
    Ok(())
}

/// CUDA overlays the DRM row only after exact PCI/UUID verification of NVML
/// memory. `memory` is (total, free); None must not reuse sysfs availability.
#[cfg(any(test, all(feature = "cuda", target_os = "linux")))]
fn record_cuda(
    d: &mut Observation,
    total_bytes: u64,
    ordinal: usize,
    compute_capability: (i32, i32),
    memory: Option<(u64, u64)>,
    duplicate: bool,
) {
    if duplicate {
        d.cuda_ordinal = None;
        d.available = None;
        d.exclusion =
            Some("multiple CUDA views share one PCI identity (MIG/partition ambiguous)".into());
        return;
    }
    d.ceiling = Some(memory.map_or(total_bytes, |(total, _)| total_bytes.min(total)));
    d.available = memory.map(|(_, free)| free.min(total_bytes));
    d.cuda_ordinal = Some(ordinal);
    d.cuda_cc = Some(compute_capability);
    d.backends.retain(|b| !b.starts_with("cuda"));
    d.backends.push("cuda:physical-binding-observed".into());
    d.provenance="CUDA cuDevice metadata without primary context +NVML exact UUID+full BDF/current free, capacity intersection; independent immutable source eligibility required".into();
    d.exclusion = if compute_capability != (8, 6) {
        Some("finite Qwen CUDA route requires qualified sm86 profile".into())
    } else if d.available.is_none() {
        Some("NVML current free/capacity binding unavailable".into())
    } else {
        None
    };
}

fn sorted_cards(cards: BTreeMap<String, Observation>) -> Vec<Observation> {
    let mut devices = cards.into_values().collect::<Vec<_>>();
    devices.sort_by_key(|d| {
        (
            match d.vendor.as_str() {
                "nvidia" => 0,
                "amd" => 1,
                _ => 2,
            },
            d.id.clone(),
        )
    });
    devices
}

/// Inspect current OS/driver metadata. No source is opened, and an observed
/// backend or free-byte count never authorizes execution or replaces admission.
pub fn discover() -> Result<Inventory> {
    let observed_unix_ms = now_ms()?;
    let mut cards = BTreeMap::new();
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        for entry in entries.take(256).flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if !name
                .strip_prefix("card")
                .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
            {
                continue;
            }
            let Ok(device) = std::fs::canonicalize(entry.path().join("device")) else {
                continue;
            };
            let Some(id) = device
                .file_name()
                .and_then(|s| s.to_str())
                .and_then(|s| normalize_pci(s).ok())
            else {
                continue;
            };
            let vendor = match read(&device.join("vendor"), 128).as_deref().map(str::trim) {
                Some("0x10de") => "nvidia",
                Some("0x1002") => "amd",
                _ => "other",
            };
            let memory = zerogravity_core::placement::LocalVramObservation::observe(&device);
            record_drm(&mut cards, &id, vendor, memory.total, memory.used)?;
        }
    }
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    if let Ok(observations) = crate::gpu::observe_cuda_devices() {
        let nvml = nvml_wrapper::Nvml::init().ok();
        let mut seen = std::collections::BTreeSet::new();
        for view in observations {
            let pci = normalize_pci(&view.pci_bus_id)?;
            let id = format!("pci:{pci}");
            let duplicate = !seen.insert(id.clone());
            let memory = nvml
                .as_ref()
                .and_then(|n| n.device_by_pci_bus_id(view.pci_bus_id.as_str()).ok())
                .and_then(|d| {
                    let pci = d.pci_info().ok()?;
                    if normalize_pci(&pci.bus_id).ok()? != normalize_pci(&view.pci_bus_id).ok()? {
                        return None;
                    }
                    let uuid = d.uuid().ok()?;
                    let bytes = uuid.strip_prefix("GPU-")?.replace('-', "");
                    if bytes.len() != 32 || !bytes.bytes().all(|b| b.is_ascii_hexdigit()) {
                        return None;
                    }
                    for (i, expected) in view.uuid.iter().enumerate() {
                        if u8::from_str_radix(&bytes[i * 2..i * 2 + 2], 16).ok()? != *expected {
                            return None;
                        }
                    }
                    let memory = d.memory_info().ok()?;
                    (memory.free <= memory.total).then_some(memory)
                });
            let d = physical_card(&mut cards, &pci, "nvidia")?;
            record_cuda(
                d,
                view.total_bytes,
                view.ordinal,
                view.compute_capability,
                memory.map(|m| (m.total, m.free)),
                duplicate,
            );
        }
    }
    let mut devices = sorted_cards(cards);
    devices.push(host());
    Ok(Inventory {
        devices,
        observed_unix_ms,
    })
}

#[cfg(test)]
mod tests;
