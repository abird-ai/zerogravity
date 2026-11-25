//! Bounded proc/sysfs reads; no cgroup writes or capacity guessed from free RAM.
use super::HostMemoryObservation;
use std::{
    io::Read,
    path::{Component, Path},
};

fn read(path: &Path, limit: u64) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    if !file.metadata().ok()?.is_file() {
        return None;
    }
    let mut bytes = String::new();
    file.take(limit + 1).read_to_string(&mut bytes).ok()?;
    (bytes.len() as u64 <= limit).then_some(bytes)
}

fn complete_cgroup_mount(mountinfo: &str) -> bool {
    // Do not guess a hidden ancestor or a different mount's hierarchy.
    mountinfo
        .lines()
        .filter_map(|line| line.split_once(" - "))
        .any(|(mount, fs)| {
            let mut fields = mount.split_whitespace();
            fields.nth(3) == Some("/")
                && fields.next() == Some("/sys/fs/cgroup")
                && fs.split_whitespace().next() == Some("cgroup2")
        })
}

pub(super) fn observe() -> HostMemoryObservation {
    let root = Path::new("/sys/fs/cgroup");
    let mounted = read(Path::new("/proc/self/mountinfo"), 1024 * 1024)
        .is_some_and(|s| complete_cgroup_mount(&s));
    if !mounted {
        return HostMemoryObservation {
            ceiling: None,
            available: None,
            provenance: "complete visible cgroup-v2 mount unavailable".into(),
        };
    }
    host_from(
        Path::new("/proc/meminfo"),
        Path::new("/proc/self/cgroup"),
        root,
    )
}

fn absent(path: &Path) -> bool {
    matches!(std::fs::symlink_metadata(path), Err(error) if error.kind() == std::io::ErrorKind::NotFound)
}

fn host_from(meminfo: &Path, membership: &Path, root: &Path) -> HostMemoryObservation {
    let memory = read(meminfo, 64 * 1024).unwrap_or_default();
    let field = |name: &str| {
        let mut fields = memory.lines().filter_map(|l| l.strip_prefix(name));
        let mut words = fields.next()?.split_whitespace();
        let value = words.next()?.parse::<u64>().ok()?.checked_mul(1024)?;
        (words.next() == Some("kB") && words.next().is_none() && fields.next().is_none())
            .then_some(value)
    };
    let mut total = field("MemTotal:").filter(|n| *n != 0);
    let mut available = field("MemAvailable:")
        .zip(total)
        .and_then(|(a, t)| (a <= t).then_some(a));
    let mut provenance = "/proc/meminfo MemTotal/MemAvailable".to_string();
    let relative = read(membership, 64 * 1024).and_then(|groups| {
        let mut unified = groups.lines().filter_map(|line| line.strip_prefix("0::"));
        let path = unified.next()?.to_owned();
        // v1 memory controller constraints are not represented by v2 files.
        if unified.next().is_some()
            || groups.lines().any(|line| {
                line.split(':')
                    .nth(1)
                    .is_some_and(|s| s.split(',').any(|c| c == "memory"))
            })
        {
            return None;
        }
        Some(path)
    });
    let valid = relative.as_deref().map(Path::new).filter(|p| {
        p.is_absolute()
            && p.components()
                .all(|c| matches!(c, Component::RootDir | Component::Normal(_)))
    });
    let Some(path) = valid else {
        return HostMemoryObservation {
            ceiling: None,
            available: None,
            provenance: format!("{provenance}; cgroup membership unavailable"),
        };
    };
    let mut group = root.join(path.strip_prefix("/").expect("absolute path"));
    loop {
        if !group.is_dir() {
            total = None;
            available = None;
            break;
        }
        let max_path = group.join("memory.max");
        match read(&max_path, 128).as_deref().map(str::trim) {
            Some("max") => {
                if read(&group.join("memory.current"), 128)
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .is_none()
                {
                    available = None;
                }
            }
            Some(text) => match text.parse::<u64>() {
                Ok(max) => {
                    total = total.map(|n| n.min(max));
                    let used = read(&group.join("memory.current"), 128)
                        .and_then(|s| s.trim().parse::<u64>().ok());
                    available = available
                        .zip(used)
                        .map(|(a, u)| a.min(max.saturating_sub(u)));
                    provenance.push_str("; cgroup-v2 memory.max/current");
                }
                Err(_) => {
                    total = None;
                    available = None;
                }
            },
            None if group == root && absent(&max_path) => {}
            None if absent(&max_path)
                && group
                    .parent()
                    .and_then(|p| read(&p.join("cgroup.subtree_control"), 4096))
                    .is_some_and(|s| !s.split_whitespace().any(|c| c == "memory")) => {}
            None => {
                total = None;
                available = None;
            }
        }
        if group == root || !group.pop() {
            break;
        }
    }
    if total.is_none() || available.is_none() {
        provenance.push_str("; effective capacity/current availability incomplete");
    }
    HostMemoryObservation {
        ceiling: total,
        available: total.zip(available).map(|(t, a)| a.min(t)),
        provenance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "zg-sdk-host-observation-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            std::fs::create_dir(&root).unwrap();
            let fixture = Self(root);
            fixture.write("meminfo", "MemTotal: 1024 kB\nMemAvailable: 900 kB\n");
            fixture.write("membership", "0::/parent/leaf\n");
            fixture.group("parent", "819200", Some("102400"));
            fixture.group("parent/leaf", "921600", Some("204800"));
            fixture
        }
        fn write(&self, name: &str, contents: &str) {
            let path = self.0.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents).unwrap();
        }
        fn group(&self, name: &str, max: &str, current: Option<&str>) {
            self.write(&format!("cgroup/{name}/memory.max"), max);
            let current_path = format!("cgroup/{name}/memory.current");
            if let Some(current) = current {
                self.write(&current_path, current);
            } else {
                let _ = std::fs::remove_file(self.0.join(current_path));
            }
        }
        fn observe(&self) -> HostMemoryObservation {
            host_from(
                &self.0.join("meminfo"),
                &self.0.join("membership"),
                &self.0.join("cgroup"),
            )
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn host_intersects_memavailable_and_every_ancestor_without_a_new_pool() {
        let f = Fixture::new();
        let host = f.observe();
        assert_eq!(host.ceiling, Some(800 * 1024));
        assert_eq!(host.available, Some(700 * 1024));
        assert_eq!(
            host.provenance
                .matches("cgroup-v2 memory.max/current")
                .count(),
            2
        );
        f.group("parent", "614400", Some("512000"));
        let constrained = f.observe();
        assert_eq!(constrained.ceiling, Some(600 * 1024));
        assert_eq!(constrained.available, Some(100 * 1024));
    }

    #[test]
    fn unknown_current_malformed_limits_and_overuse_never_become_free_memory() {
        let f = Fixture::new();
        for (max, current) in [("max", None), ("921600", None), ("bad", Some("0"))] {
            f.group("parent/leaf", max, current);
            let host = f.observe();
            assert_eq!(host.available, None, "{max} / {current:?}");
        }
        f.group("parent/leaf", "1", Some("2"));
        assert_eq!(f.observe().available, Some(0));
    }

    #[test]
    fn missing_or_unsafe_cgroup_membership_refuses_current_availability() {
        let f = Fixture::new();
        for membership in [
            "",
            "1:memory:/legacy\n",
            "0::/../outside\n",
            "0::/parent/../../outside\n",
        ] {
            f.write("membership", membership);
            assert_eq!(f.observe().available, None, "{membership}");
        }
        std::fs::remove_file(f.0.join("membership")).unwrap();
        assert_eq!(f.observe().available, None);
        f.write("membership", &"x".repeat(64 * 1024 + 1));
        assert_eq!(f.observe().available, None);
    }

    #[test]
    fn missing_controller_files_require_explicitly_disabled_parent_controller() {
        let f = Fixture::new();
        for name in ["memory.max", "memory.current"] {
            std::fs::remove_file(f.0.join("cgroup/parent/leaf").join(name)).unwrap();
        }
        assert_eq!(f.observe().available, None);
        f.write("cgroup/parent/cgroup.subtree_control", "cpu memory\n");
        assert_eq!(f.observe().available, None);
        f.write("cgroup/parent/cgroup.subtree_control", "cpu\n");
        assert_eq!(f.observe().available, Some(700 * 1024));
    }

    #[test]
    fn bounded_host_reads_and_missing_meminfo_fields_stay_unknown() {
        let f = Fixture::new();
        f.write("bounded", "1234");
        assert_eq!(read(&f.0.join("bounded"), 4).as_deref(), Some("1234"));
        assert!(read(&f.0.join("bounded"), 3).is_none());
        assert!(read(&f.0, 1024).is_none());
        f.write("meminfo", "MemTotal: 1024 kB\n");
        assert_eq!(f.observe().available, None);
        f.write(
            "meminfo",
            "MemTotal: 18446744073709551615 kB\nMemAvailable: 900 kB\n",
        );
        assert_eq!(f.observe().ceiling, None);
        f.write("meminfo", &"x".repeat(64 * 1024 + 1));
        let unknown = f.observe();
        assert_eq!(unknown.ceiling, None);
        assert_eq!(unknown.available, None);
    }

    #[test]
    fn only_complete_visible_v2_mount_is_accepted() {
        assert!(complete_cgroup_mount(
            "1 2 0:3 / /sys/fs/cgroup ro - cgroup2 cgroup2 rw"
        ));
        assert!(!complete_cgroup_mount(
            "1 2 0:3 /hidden /sys/fs/cgroup ro - cgroup2 cgroup2 rw"
        ));
        assert!(!complete_cgroup_mount(
            "1 2 0:3 / /sys/fs/cgroup ro - cgroup cgroup rw"
        ));
    }
}
