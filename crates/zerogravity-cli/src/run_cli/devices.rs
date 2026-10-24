//! No Client, model, driver context or ledger mutation in inventory/ranking.
use super::*;
use serde::Serialize;
use std::io::Read;
#[derive(Clone, Debug, Serialize)]
pub(super) struct Observation {
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
pub(super) struct Inventory {
    pub devices: Vec<Observation>,
    pub observed_unix_ms: u128,
}
fn read(path: &Path, limit: u64) -> Option<String> {
    let f = std::fs::File::open(path).ok()?;
    if !f.metadata().ok()?.is_file() {
        return None;
    }
    let mut s = String::new();
    f.take(limit + 1).read_to_string(&mut s).ok()?;
    (s.len() as u64 <= limit).then_some(s)
}
fn number(path: &Path) -> Option<u64> {
    read(path, 128)?.trim().parse().ok()
}
pub(super) fn normalize_pci(s: &str) -> Result<String> {
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
fn host() -> Observation {
    let memory = read(Path::new("/proc/meminfo"), 64 * 1024).unwrap_or_default();
    let field = |name: &str| {
        memory
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|l| l.split_whitespace().next())
            .and_then(|s| s.parse::<u64>().ok())
            .and_then(|n| n.checked_mul(1024))
    };
    let mut total = field("MemTotal:");
    let mut available = field("MemAvailable:");
    let mut provenance = "/proc/meminfo MemAvailable".to_string();
    if let Some(cgroups) = read(Path::new("/proc/self/cgroup"), 64 * 1024) {
        if let Some(relative) = cgroups.lines().find_map(|l| l.strip_prefix("0::")) {
            let p = Path::new(relative);
            if p.components().all(|c| {
                matches!(
                    c,
                    std::path::Component::RootDir | std::path::Component::Normal(_)
                )
            }) {
                // Every ancestor's cgroup ceiling applies to this process.
                let mut group = Path::new("/sys/fs/cgroup").join(p.strip_prefix("/").unwrap_or(p));
                loop {
                    let max_text = read(&group.join("memory.max"), 128);
                    let max = max_text
                        .as_deref()
                        .and_then(|s| s.trim().parse::<u64>().ok());
                    let used = number(&group.join("memory.current"));
                    match max_text.as_deref() {
                        Some(s) if s.trim() == "max" => {
                            if used.is_none() {
                                available = None;
                            }
                        }
                        Some(s) if s.trim().parse::<u64>().is_ok() => {
                            if used.is_none() {
                                available = None;
                            }
                        }
                        None if group == Path::new("/sys/fs/cgroup") => {}
                        None => {
                            let enabled = group
                                .parent()
                                .and_then(|p| read(&p.join("cgroup.subtree_control"), 4096));
                            if enabled
                                .as_deref()
                                .is_none_or(|s| s.split_whitespace().any(|c| c == "memory"))
                            {
                                available = None;
                            }
                        }
                        _ => available = None,
                    }
                    if let Some(max) = max {
                        total = total.map(|n| n.min(max));
                        available = match (available, used) {
                            (Some(a), Some(u)) => max.checked_sub(u).map(|r| a.min(r)),
                            _ => None,
                        };
                        provenance.push_str("; cgroup-v2 memory.max/current");
                    }
                    if group == Path::new("/sys/fs/cgroup") || !group.pop() {
                        break;
                    }
                }
            } else {
                available = None;
            }
        } else {
            available = None;
        }
    } else {
        available = None;
    }
    Observation {
        id: "cpu:0".into(),
        class: "host-memory".into(),
        vendor: "cpu".into(),
        ceiling: total,
        available,
        backends: if cfg!(feature = "cpu") {
            vec!["cpu".into()]
        } else {
            vec![]
        },
        exclusion: available
            .is_none()
            .then(|| "current host availability unavailable".into()),
        provenance,
        cuda_ordinal: None,
        cuda_cc: None,
    }
}
pub(super) fn discover() -> Result<Inventory> {
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
            let total = number(&device.join("mem_info_vram_total"));
            let used = number(&device.join("mem_info_vram_used"));
            let available = total.zip(used).and_then(|(t, u)| t.checked_sub(u));
            let mut backends = Vec::new();
            if vendor == "nvidia" && cfg!(feature = "cuda") {
                backends.push("cuda:binding-unverified".into());
            }
            if vendor == "amd" && cfg!(feature = "hip") {
                backends.push("rocm:binding-unverified".into());
            }
            if cfg!(feature = "vulkan") {
                backends.push("vulkan:binding-unverified".into());
            }
            cards.insert(id.clone(),Observation{id:format!("pci:{id}"),class:"gpu".into(),vendor:vendor.into(),ceiling:total,available,backends,exclusion:Some("source-qualified cold/compiler cost and runtime physical binding unavailable in generic run".into()),provenance:"DRM full PCI identity; sysfs VRAM total/used if exposed; GTT is not another pool".into(),cuda_ordinal:None,cuda_cc:None});
        }
    }
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    if let Ok(observations) = zerogravity::gpu::observe_cuda_devices() {
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
            let d = cards.entry(id.clone()).or_insert_with(|| Observation {
                id,
                class: "gpu".into(),
                vendor: "nvidia".into(),
                ceiling: None,
                available: None,
                backends: Vec::new(),
                exclusion: None,
                provenance: String::new(),
                cuda_ordinal: None,
                cuda_cc: None,
            });
            if duplicate {
                d.cuda_ordinal = None;
                d.available = None;
                d.exclusion = Some(
                    "multiple CUDA views share one PCI identity (MIG/partition ambiguous)".into(),
                );
                continue;
            }
            d.ceiling = Some(
                memory
                    .as_ref()
                    .map_or(view.total_bytes, |m| view.total_bytes.min(m.total)),
            );
            d.available = memory.map(|m| m.free.min(view.total_bytes));
            d.cuda_ordinal = Some(view.ordinal);
            d.cuda_cc = Some(view.compute_capability);
            d.backends.retain(|b| !b.starts_with("cuda"));
            d.backends.push("cuda:physical-binding-observed".into());
            d.provenance="CUDA cuDevice metadata without primary context +NVML exact UUID+full BDF/current free, capacity intersection; independent immutable source eligibility required".into();
            d.exclusion = if view.compute_capability != (8, 6) {
                Some("finite short-HF route requires qualified sm86 profile".into())
            } else if d.available.is_none() {
                Some("NVML current free/capacity binding unavailable".into())
            } else {
                None
            };
        }
    }
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
    devices.push(host());
    Ok(Inventory {
        devices,
        observed_unix_ms,
    })
}
pub(super) fn allowed(request: &Request, inventory: &Inventory) -> Result<Vec<Observation>> {
    let mut result = Vec::new();
    let gpu = inventory
        .devices
        .iter()
        .filter(|d| d.class == "gpu")
        .collect::<Vec<_>>();
    let selected = match &request.devices {
        Devices::Auto => inventory.devices.iter().collect(),
        Devices::Ordered(devices) => {
            let mut list = Vec::new();
            for d in devices {
                if !matches!(d.host.as_str(), "local" | "localhost") {
                    return Err(Error::Unsupported(
                        "remote compute transport is not qualified".into(),
                    ));
                }
                let found = if let Some(n) = d.selector.strip_prefix("gpu:") {
                    n.parse::<usize>().ok().and_then(|n| gpu.get(n).copied())
                } else if let Some(pci) = d.selector.strip_prefix("pci:") {
                    let id = format!("pci:{}", normalize_pci(pci)?);
                    inventory.devices.iter().find(|d| d.id == id)
                } else {
                    inventory.devices.iter().find(|item| item.id == d.selector)
                }
                .ok_or_else(|| {
                    Error::Unsupported(format!("explicit device {} is unavailable", d.label()))
                })?;
                if !list.iter().any(|old: &&Observation| old.id == found.id) {
                    list.push(found);
                }
            }
            list
        }
    };
    if let Some(i) = selected.iter().position(|d| d.class == "host-memory")
        && i + 1 != selected.len()
    {
        return Err(bad("CPU must be terminal after physical alias resolution"));
    }
    for device in selected {
        let compatible = match request.backend {
            Backend::Auto => true,
            Backend::Cpu => device.class == "host-memory",
            Backend::Cuda => device.class == "gpu" && device.vendor == "nvidia",
            Backend::Rocm => device.class == "gpu" && device.vendor == "amd",
            Backend::Vulkan => device.class == "gpu",
        };
        if !compatible {
            if matches!(request.devices, Devices::Ordered(_)) {
                return Err(bad(
                    "explicit device contradicts forced backend; every allowlist member must be compatible",
                ));
            }
            continue;
        }
        if request.backend == Backend::Cpu && device.class != "host-memory" {
            if matches!(request.devices, Devices::Ordered(_)) {
                return Err(bad("CPU backend contradicts explicit GPU device"));
            }
            continue;
        }
        if !matches!(request.backend, Backend::Auto | Backend::Cpu) && device.class == "host-memory"
        {
            continue;
        }
        if let Some(v) = &request.vendor_filter
            && device.vendor != *v
        {
            if matches!(request.devices, Devices::Ordered(_)) {
                return Err(bad("explicit device contradicts alias vendor filter"));
            }
            continue;
        }
        result.push(device.clone());
    }
    Ok(result)
}

fn now_ms() -> Result<u128> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| bad("system clock before epoch"))?
        .as_millis())
}
#[cfg(any(
    all(feature = "cpu", feature = "model-qwen35", feature = "text"),
    all(
        any(feature = "short-hf-run", feature = "short-hf-serving"),
        target_os = "linux"
    )
))]
pub(super) fn fresh(inventory: &Inventory) -> Result<()> {
    if now_ms()?
        .checked_sub(inventory.observed_unix_ms)
        .is_none_or(|age| age > 5000)
    {
        return Err(Error::State(
            "current resource observation expired; no startup effect admitted".into(),
        ));
    }
    Ok(())
}
