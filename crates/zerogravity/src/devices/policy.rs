//! Ordered selector resolution and current-observation refusal policy.
use super::{Inventory, Observation, bad, device_node_identity, normalize_pci, now_ms};
use crate::run_policy::{Backend, DeviceRef, Devices, Request};
use crate::{Error, Result};

/// Resolve one observed physical identity, without backend or model admission.
pub fn observe_selected<'a>(
    inventory: &'a Inventory,
    device: &DeviceRef,
) -> Result<&'a Observation> {
    if !matches!(device.host.as_str(), "local" | "localhost") {
        return Err(Error::Unsupported(
            "remote compute transport is not qualified".into(),
        ));
    }
    let found = if let Some(n) = device.selector.strip_prefix("gpu:") {
        n.parse::<usize>()
            .ok()
            .and_then(|n| inventory.devices.iter().filter(|d| d.class == "gpu").nth(n))
    } else if let Some(pci) = device.selector.strip_prefix("pci:") {
        let id = format!("pci:{}", normalize_pci(pci)?);
        inventory.devices.iter().find(|d| d.id == id)
    } else if device.selector.starts_with("/dev/") {
        let id = device_node_identity(&device.selector)?;
        inventory.devices.iter().find(|d| d.id == id)
    } else {
        inventory.devices.iter().find(|d| d.id == device.selector)
    };
    found.ok_or_else(|| {
        Error::Unsupported(format!("explicit device {} is unavailable", device.label()))
    })
}

/// Select observations in caller order, preserving exclusions and unknown
/// capacity. This is not a source/profile fit check or execution eligibility.
/// `/dev` selectors perform read-only node/sysfs/proc identity inspection.
pub fn allowed(request: &Request, inventory: &Inventory) -> Result<Vec<Observation>> {
    let mut result = Vec::new();
    let selected = match &request.devices {
        Devices::Auto => inventory.devices.iter().collect(),
        Devices::Ordered(devices) => {
            let mut list = Vec::new();
            for d in devices {
                let found = observe_selected(inventory, d)?;
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

/// Host-only current availability/headroom check for the Qwen CUDA HOST policy.
/// Reuses the same `/proc`/cgroup observation; no GPU discovery or second pool.
#[cfg(all(feature = "qwen35-cuda-qualification", target_os = "linux"))]
pub fn host_available(host_bytes: u64, headroom: &crate::run_policy::Headroom) -> Result<()> {
    let backing = crate::resource_arbiter::ResourceArbiter::process_shared().host_backing()?;
    check_host_available(&super::inventory::host(), host_bytes, headroom, backing)
}

#[cfg(any(test, all(feature = "qwen35-cuda-qualification", target_os = "linux")))]
fn check_host_available(
    host: &Observation,
    host_bytes: u64,
    headroom: &crate::run_policy::Headroom,
    backing: crate::resource_arbiter::HostBackingSnapshot,
) -> Result<()> {
    let ceiling = host
        .ceiling
        .ok_or_else(|| Error::State("HOST ceiling unknown".into()))?;
    let available = host
        .available
        .ok_or_else(|| Error::State("HOST current availability unavailable".into()))?;
    let margin = headroom.bytes(ceiling, true)?;
    let usable = backing.conservative_available(ceiling, available, margin);
    if host_bytes > usable {
        return Err(Error::resource(
            "host memory",
            "cold host fit after headroom and protected commitments",
            host_bytes,
            usable,
        ));
    }
    Ok(())
}

/// Refuse stale/future observations before a caller's startup effects. This
/// check grants no admission and does not refresh or probe hardware itself.
pub fn fresh(inventory: &Inventory) -> Result<()> {
    fresh_at(inventory, now_ms()?)
}

fn fresh_at(inventory: &Inventory, now: u128) -> Result<()> {
    if now
        .checked_sub(inventory.observed_unix_ms)
        .is_none_or(|age| age > 5000)
    {
        return Err(Error::State(
            "current resource observation expired; no startup effect admitted".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_policy::Headroom;

    #[test]
    fn observation_freshness_keeps_the_original_five_second_boundary() {
        let inventory = Inventory {
            devices: vec![],
            observed_unix_ms: 10_000,
        };
        assert!(fresh_at(&inventory, 10_000).is_ok());
        assert!(fresh_at(&inventory, 15_000).is_ok());
        for now in [9999, 15_001, u128::MAX] {
            assert!(
                matches!(fresh_at(&inventory, now), Err(Error::State(m)) if m == "current resource observation expired; no startup effect admitted")
            );
        }
    }

    #[test]
    fn host_precheck_keeps_unknown_capacity_headroom_and_fit_refusals() {
        let mut host = Observation {
            id: "cpu:0".into(),
            class: "host-memory".into(),
            vendor: "cpu".into(),
            ceiling: None,
            available: None,
            backends: vec![],
            exclusion: None,
            provenance: "synthetic host snapshot".into(),
            cuda_ordinal: None,
            cuda_cc: None,
        };
        let headroom = Headroom::Bytes(20);
        let backing = crate::resource_arbiter::HostBackingSnapshot {
            ceiling: None,
            observed_ceiling: None,
            reserved: 0,
        };
        assert!(
            matches!(check_host_available(&host, 1, &headroom, backing), Err(Error::State(m)) if m == "HOST ceiling unknown")
        );
        host.ceiling = Some(100);
        assert!(
            matches!(check_host_available(&host, 1, &headroom, backing), Err(Error::State(m)) if m == "HOST current availability unavailable")
        );
        host.available = Some(19);
        assert!(
            matches!(check_host_available(&host, 1, &headroom, backing), Err(Error::Resource(r)) if r.requested == 1 && r.available == 0)
        );
        host.available = Some(50);
        assert!(check_host_available(&host, 30, &headroom, backing).is_ok());
        assert!(
            matches!(check_host_available(&host, 31, &headroom, backing), Err(Error::Resource(r)) if r.requested == 31 && r.available == 30)
        );
        assert!(check_host_available(&host, 40, &Headroom::Percent(10), backing).is_ok());
        assert_eq!(
            Headroom::Percent(99).bytes(u64::MAX, false).unwrap(),
            u64::MAX / 100 * 99 + (u64::MAX % 100) * 99 / 100
        );
        assert!(Headroom::parse("100%").is_err());
        assert!(Headroom::parse("18446744073709551615GiB").is_err());
    }
}
