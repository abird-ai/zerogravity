//! Four independent placement requests. None is an admission or transfer receipt.
use crate::physical::PhysicalDeviceKey;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlacementTarget {
    Physical(PhysicalDeviceKey),
    /// Future authenticated IO/peer placement, never implicit remote enrollment.
    Storage,
    Remote,
}
#[derive(Clone, Debug)]
pub struct PlacementRequest {
    pub compute: PlacementTarget,
    pub parameters: PlacementTarget,
    pub mutable_state: PlacementTarget,
    pub reusable_cache: PlacementTarget,
}
#[derive(Clone, Debug)]
pub struct TransferRequest {
    pub from: PlacementTarget,
    pub to: PlacementTarget,
    pub bytes: u64,
}
mod tensor_layout;
pub use tensor_layout::PreparedTensorLayout;

/// Device-local VRAM aperture, not a claim of independent physical backing.
/// On UMA this may be a carved-out part of system memory. Do not add this
/// capacity to OS-visible RAM or GTT without actual backing identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LocalVramObservation {
    pub total: Option<u64>,
    pub used: Option<u64>,
}
impl LocalVramObservation {
    pub fn available(self) -> Option<u64> {
        self.total
            .zip(self.used)
            .map(|(total, used)| total.saturating_sub(used))
    }

    /// Same Linux DRM device-local counters for inventory and native cold fit.
    pub fn observe(device: &std::path::Path) -> Self {
        use std::io::Read;
        let number = |name: &str| {
            let file = std::fs::File::open(device.join(name)).ok()?;
            if !file.metadata().ok()?.is_file() {
                return None;
            }
            let mut text = String::new();
            file.take(129).read_to_string(&mut text).ok()?;
            if text.len() > 128 {
                return None;
            }
            text.trim().parse::<u64>().ok()
        };
        Self {
            total: number("mem_info_vram_total").filter(|n| *n != 0),
            used: number("mem_info_vram_used"),
        }
    }

    /// An actual driver observation and local aperture are intersecting
    /// constraints. An integrated adapter without local-aperture evidence
    /// cannot qualify a device-local fit from addressability alone.
    pub fn intersect_driver(
        self,
        total: u64,
        available: u64,
        integrated: bool,
    ) -> crate::Result<(u64, u64)> {
        if total == 0 || available > total {
            return Err(crate::Error::State(
                "invalid device memory observation".into(),
            ));
        }
        if integrated && self.total.zip(self.available()).is_none() {
            return Err(crate::Error::Unsupported(
                "integrated device-local VRAM capacity/current usage unavailable; host/GTT overlap unresolved".into(),
            ));
        }
        let ceiling = self.total.map_or(total, |n| n.min(total));
        let free = self
            .available()
            .map_or(available, |n| n.min(available))
            .min(ceiling);
        Ok((ceiling, free))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_vram_intersects_driver_limits_without_adding_host_or_gtt() {
        let local = LocalVramObservation {
            total: Some(80),
            used: Some(20),
        };
        assert_eq!(local.intersect_driver(120, 100, true).unwrap(), (80, 60));
        assert_eq!(local.intersect_driver(70, 50, true).unwrap(), (70, 50));
        assert!(
            LocalVramObservation::default()
                .intersect_driver(120, 100, true)
                .is_err()
        );
        assert_eq!(
            LocalVramObservation::default()
                .intersect_driver(120, 100, false)
                .unwrap(),
            (120, 100)
        );
        assert!(local.intersect_driver(120, 121, false).is_err());
    }
}
