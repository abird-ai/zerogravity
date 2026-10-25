//! Exact Linux device-node -> existing physical PCI identity.
//! Only read-only metadata/sysfs/proc inspection; never open the GPU node.
use super::*;
use std::os::unix::fs::{FileTypeExt, MetadataExt};

const DRM_MAJOR: u64 = 226;
const NVIDIA_MAJOR: u64 = 195;

fn major(rdev: u64) -> u64 {
    ((rdev >> 8) & 0xfff) | ((rdev >> 32) & !0xfff)
}
fn minor(rdev: u64) -> u64 {
    (rdev & 0xff) | ((rdev >> 12) & !0xff)
}

fn numbered(name: &str, prefix: &str) -> Option<u64> {
    let n = name.strip_prefix(prefix)?;
    if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    n.parse().ok()
}

/// Bind the original character-device rdev to the kernel's DRM class identity,
/// then to the same full PCI function as ordinary inventory discovery.
fn drm_identity(
    class_root: &Path,
    devices_root: &Path,
    name: &str,
    major: u64,
    minor: u64,
) -> Result<String> {
    if major != DRM_MAJOR
        || (numbered(name, "renderD").is_none() && numbered(name, "card").is_none())
    {
        return Err(Error::Unsupported(
            "device node is not a DRM GPU card/render node".into(),
        ));
    }
    let class = class_root.join(name);
    let reported = read(&class.join("dev"), 128).ok_or_else(|| {
        Error::Unsupported("DRM character-device sysfs identity unavailable".into())
    })?;
    if reported.trim() != format!("{major}:{minor}") {
        return Err(Error::State(
            "DRM device node and sysfs major/minor differ".into(),
        ));
    }
    let physical = std::fs::canonicalize(class.join("device"))
        .map_err(|_| Error::Unsupported("DRM physical PCI binding unavailable".into()))?;
    if !physical.starts_with(devices_root) {
        return Err(Error::State(
            "DRM device sysfs binding is outside physical devices".into(),
        ));
    }
    let bdf = physical
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Error::Unsupported("DRM device has no complete PCI identity".into()))?;
    Ok(format!("pci:{}", normalize_pci(bdf)?))
}

/// NVIDIA /dev/nvidiaN is an instance-specific control/compute node. Its
/// numeric minor is NOT assumed equal to CUDA ordinal, DRM card number or BDF.
/// Verify the kernel's per-GPU Device Minor and Bus Location records.
fn nvidia_identity(gpus_root: &Path, wanted_minor: u64) -> Result<String> {
    let entries = std::fs::read_dir(gpus_root).map_err(|_| {
        Error::Unsupported("NVIDIA per-GPU proc identity unavailable; use PCI selection".into())
    })?;
    let mut found = None;
    for (i, item) in entries.enumerate() {
        if i >= 128 {
            return Err(Error::Limit(
                "NVIDIA GPU identity inventory exceeds 128 entries".into(),
            ));
        }
        let item = item.map_err(|_| Error::State("NVIDIA identity directory unreadable".into()))?;
        let Some(directory_bdf) = item
            .file_name()
            .to_str()
            .and_then(|s| normalize_pci(s).ok())
        else {
            continue;
        };
        let text = read(&item.path().join("information"), 4096).ok_or_else(|| {
            Error::Unsupported("NVIDIA GPU minor/bus information unavailable".into())
        })?;
        let field = |key: &str| {
            text.lines()
                .filter_map(|l| l.split_once(':'))
                .find_map(|(name, value)| (name.trim() == key).then_some(value.trim()))
        };
        let Some(index) = field("Device Minor").and_then(|s| s.parse::<u64>().ok()) else {
            return Err(Error::State("NVIDIA GPU has no valid device minor".into()));
        };
        let Some(bus) = field("Bus Location").and_then(|s| normalize_pci(s).ok()) else {
            return Err(Error::State(
                "NVIDIA GPU has no valid full PCI bus location".into(),
            ));
        };
        if bus != directory_bdf {
            return Err(Error::State(
                "NVIDIA proc bus identity differs from directory".into(),
            ));
        }
        if index == wanted_minor && found.replace(bus).is_some() {
            return Err(Error::State(
                "NVIDIA minor matches multiple physical GPU records".into(),
            ));
        }
    }
    found.map(|bdf| format!("pci:{bdf}")).ok_or_else(|| {
        Error::Unsupported("NVIDIA device minor has no unique physical GPU record".into())
    })
}

pub(super) fn resolve(path: &str) -> Result<String> {
    let original = Path::new(path);
    // DeviceRef already enforces an absolute bounded lexical /dev path.
    if !path.starts_with("/dev/") {
        return Err(Error::Malformed(
            "expected an absolute /dev device node".into(),
        ));
    }
    let canonical = std::fs::canonicalize(original)
        .map_err(|_| Error::Unsupported("requested /dev device node is unavailable".into()))?;
    if !canonical.starts_with("/dev/") {
        return Err(Error::Unsupported(
            "/dev device node resolved outside /dev".into(),
        ));
    }
    let metadata = std::fs::metadata(original)
        .map_err(|_| Error::Unsupported("requested device node cannot be inspected".into()))?;
    let target = std::fs::metadata(&canonical)
        .map_err(|_| Error::Unsupported("resolved device node cannot be inspected".into()))?;
    if !metadata.file_type().is_char_device() || !target.file_type().is_char_device() {
        return Err(Error::Unsupported(
            "device selector must resolve to a character device".into(),
        ));
    }
    if metadata.rdev() != target.rdev() || metadata.dev() != target.dev() {
        return Err(Error::State(
            "device node identity changed during resolution".into(),
        ));
    }
    let major = major(metadata.rdev());
    let minor = minor(metadata.rdev());
    let name = canonical
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Error::Malformed("device node requires a UTF-8 name".into()))?;
    if canonical.parent() == Some(Path::new("/dev/dri")) {
        return drm_identity(
            Path::new("/sys/class/drm"),
            Path::new("/sys/devices"),
            name,
            major,
            minor,
        );
    }
    if canonical.parent() == Some(Path::new("/dev")) && numbered(name, "nvidia") == Some(minor) {
        if major != NVIDIA_MAJOR {
            return Err(Error::Unsupported(
                "NVIDIA character-device major differs".into(),
            ));
        }
        return nvidia_identity(Path::new("/proc/driver/nvidia/gpus"), minor);
    }
    Err(Error::Unsupported(
        "/dev path is not a unique GPU: use a DRM card/render node or /dev/nvidiaN; /dev/kfd is shared".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    struct Fixture(std::path::PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn fixture(tag: &str) -> Fixture {
        let root =
            std::env::temp_dir().join(format!("zg-device-identity-{}-{tag}", std::process::id()));
        assert!(!root.exists(), "independent identity fixture");
        std::fs::create_dir_all(&root).unwrap();
        Fixture(root)
    }

    #[test]
    fn drm_identity_checks_major_minor_and_full_physical_bus() {
        let f = fixture("drm");
        let class = f.0.join("class/drm/renderD128");
        let pci = f.0.join("devices/pci0000:00/0000:05:00.0");
        std::fs::create_dir_all(&class).unwrap();
        std::fs::create_dir_all(&pci).unwrap();
        std::fs::write(class.join("dev"), "226:128\n").unwrap();
        symlink(&pci, class.join("device")).unwrap();
        let class_root = f.0.join("class/drm");
        let devices_root = f.0.join("devices");
        assert_eq!(
            drm_identity(&class_root, &devices_root, "renderD128", 226, 128).unwrap(),
            "pci:0000:05:00.0"
        );
        for (name, major, minor) in [
            ("renderD128", 226, 129),
            ("renderD128", 195, 128),
            ("card0", 226, 128),
            ("renderD128", 226, 127),
        ] {
            assert!(drm_identity(&class_root, &devices_root, name, major, minor).is_err());
        }
    }

    #[test]
    fn nvidia_minor_is_not_a_cuda_ordinal_and_requires_unique_verified_bus() {
        let f = fixture("nvidia");
        for (bus, minor) in [("0000:02:00.0", 7), ("0000:01:00.0", 0)] {
            let gpu = f.0.join(bus);
            std::fs::create_dir(&gpu).unwrap();
            std::fs::write(
                gpu.join("information"),
                format!("Model: GPU\nBus Location: {bus}\nDevice Minor: {minor}\n"),
            )
            .unwrap();
        }
        assert_eq!(nvidia_identity(&f.0, 0).unwrap(), "pci:0000:01:00.0");
        assert_eq!(nvidia_identity(&f.0, 7).unwrap(), "pci:0000:02:00.0");
        assert!(nvidia_identity(&f.0, 1).is_err());
        let duplicate = f.0.join("0000:03:00.0");
        std::fs::create_dir(&duplicate).unwrap();
        std::fs::write(
            duplicate.join("information"),
            "Bus Location: 0000:03:00.0\nDevice Minor: 0\n",
        )
        .unwrap();
        assert!(nvidia_identity(&f.0, 0).is_err());
    }

    #[test]
    fn read_only_live_gpu_nodes_resolve_to_the_same_inventory_identity() {
        let inventory = discover().unwrap();
        let mut checked = 0;
        for node in [
            "/dev/dri/renderD128",
            "/dev/dri/renderD129",
            "/dev/dri/card0",
            "/dev/dri/card1",
            "/dev/nvidia0",
        ] {
            if !Path::new(node).exists() {
                continue;
            }
            let id = resolve(node).unwrap();
            assert!(
                inventory.devices.iter().any(|v| v.id == id),
                "{node} = {id}"
            );
            checked += 1;
        }
        assert!(checked <= 5);
        for invalid in [
            "/dev/null",
            "/dev/zero",
            "/dev/kfd",
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
        ] {
            if Path::new(invalid).exists() {
                assert!(resolve(invalid).is_err(), "{invalid} is not a unique GPU");
            }
        }
    }
}
