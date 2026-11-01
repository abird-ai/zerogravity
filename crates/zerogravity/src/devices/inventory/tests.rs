use super::*;
mod pci;
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
    fn observe(&self) -> Observation {
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
    assert_eq!(host.id, "cpu:0");
    assert_eq!(host.class, "host-memory");
    assert_eq!(host.ceiling, Some(800 * 1024));
    assert_eq!(host.available, Some(700 * 1024));
    assert_eq!(host.exclusion, None);
    assert_eq!(
        host.provenance
            .matches("cgroup-v2 memory.max/current")
            .count(),
        2
    );
    assert_eq!(
        host.backends,
        if cfg!(feature = "cpu") {
            vec!["cpu"]
        } else {
            vec![]
        }
    );
    assert!(host.cuda_ordinal.is_none() && host.cuda_cc.is_none());
    f.group("parent", "614400", Some("512000"));
    let constrained = f.observe();
    assert_eq!(constrained.ceiling, Some(600 * 1024));
    assert_eq!(constrained.available, Some(100 * 1024));
}

#[test]
fn unknown_current_malformed_limits_and_overuse_never_become_free_memory() {
    let f = Fixture::new();
    for (max, current) in [
        ("max", None),
        ("921600", None),
        ("bad", Some("0")),
        ("1", Some("2")),
    ] {
        f.group("parent/leaf", max, current);
        let host = f.observe();
        assert_eq!(host.available, None, "{max} / {current:?}");
        assert_eq!(
            host.exclusion.as_deref(),
            Some("current host availability unavailable")
        );
    }
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
