//! Advisory, diagnostic-only OS driver memory telemetry.
//!
//! These counters are snapshots of a whole physical device (including other
//! processes and driver pools), NOT exact per-model allocation measurements,
//! not a resource-admission oracle and not a guarantee of free VRAM/GTT.
//! Production memory safety stays under the conservative ResourceArbiter.
use std::{fs, process::Command};
use zerogravity_core::resource_arbiter::PhysicalDeviceKey;

/// Allocator observations, not a free-memory or admission authority.
pub(crate) fn allocator_report(client: &cubecl::client::Client, stage: &str) {
    use cubecl_runtime::memory_management::MemoryScope;
    let report = client.memory_report(MemoryScope::Device);
    for stream in report.streams {
        let usage = stream.usage();
        println!(
            "P12_COLD_ALLOCATOR stage={stage} stream={:?} live_allocations={} live_bytes={} padding_bytes={} reserved_bytes={} pools={}",
            stream.stream,
            usage.number_allocs,
            usage.bytes_in_use,
            usage.bytes_padding,
            usage.bytes_reserved,
            stream.pools.dynamic.len() + 2,
        );
    }
}

#[derive(Clone, Copy)]
pub struct DriverMemory {
    pub used_vram_bytes: u64,
    pub used_gtt_bytes: Option<u64>,
}

fn read_u64(path: &str) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse::<u64>().ok()
}

pub fn read(key: &PhysicalDeviceKey) -> Option<DriverMemory> {
    if key.backend != "pci" {
        return None;
    }
    let pci = key.device.as_str();
    let vendor = fs::read_to_string(format!("/sys/bus/pci/devices/{pci}/vendor")).ok()?;
    if vendor.trim() == "0x1002" {
        return Some(DriverMemory {
            used_vram_bytes: read_u64(&format!("/sys/bus/pci/devices/{pci}/mem_info_vram_used"))?,
            used_gtt_bytes: read_u64(&format!("/sys/bus/pci/devices/{pci}/mem_info_gtt_used")),
        });
    }
    if vendor.trim() == "0x10de" {
        // NVIDIA's externally reported device-wide used memory is rounded to
        // whole MiB. Match the physical PCI bus suffix, not the first GPU.
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=pci.bus_id,memory.used",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let list = String::from_utf8(output.stdout).ok()?;
        for line in list.lines() {
            let (reported_pci, used_mib) = line.split_once(',')?;
            // NVIDIA reports eight-digit PCI domains, while CubeCL reports
            // a four-digit domain. The bus:device.function tail is stable.
            let bus_device_function = pci.split_once(':')?.1;
            if !reported_pci.trim().ends_with(bus_device_function) {
                continue;
            }
            let value = used_mib.trim().parse::<u64>().ok()?;
            return Some(DriverMemory {
                used_vram_bytes: value.checked_mul(1024 * 1024)?,
                used_gtt_bytes: None,
            });
        }
    }
    None
}

pub fn report(key: &PhysicalDeviceKey, stage: &str, baseline: Option<DriverMemory>) {
    match read(key) {
        Some(current) => {
            let vram_delta = baseline.map(|before| {
                i128::from(current.used_vram_bytes) - i128::from(before.used_vram_bytes)
            });
            let gtt_delta = baseline
                .zip(current.used_gtt_bytes)
                .and_then(|(before, used)| {
                    before
                        .used_gtt_bytes
                        .map(|start| i128::from(used) - i128::from(start))
                });
            println!(
                "P07_GPU_DRIVER_TELEMETRY card={key:?} stage={stage} device_used_vram_bytes={} vram_delta_from_baseline_bytes={vram_delta:?} device_used_gtt_bytes={:?} gtt_delta_from_baseline_bytes={gtt_delta:?} scope=device_wide_advisory_not_model_allocation_or_free_headroom",
                current.used_vram_bytes, current.used_gtt_bytes
            );
        }
        None => println!(
            "P07_GPU_DRIVER_TELEMETRY card={key:?} stage={stage} status=unavailable_not_guessed"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn non_physical_device_cannot_report_driver_memory() {
        assert!(read(&PhysicalDeviceKey::new("test", "0").unwrap()).is_none());
    }
}
