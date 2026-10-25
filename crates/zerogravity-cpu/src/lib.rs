pub use zerogravity_artifacts::{artifact, quant};
pub use zerogravity_core::{Error, Result, accelerator};
pub mod math;
pub mod original_hf;
pub mod reference_text;
pub mod weights;

/// One live Rayon projection's two `Injector` blocks. Charged by the serving
/// active-session owner, not as a new caller cap.
#[cfg(feature = "cache")]
pub const RAYON_PROJECTION_INJECTOR_BYTES: u64 = 2 * 1_520;

/// `Arc<CachePadded<Inner<JobRef>>>` payload: front/back/buffer atomics and
/// the cache-line padding of the pinned `crossbeam-deque`.
#[cfg(feature = "cache")]
const WORKER_INNER_ARC: u64 = 384;
/// Inline `Cell<Buffer<JobRef>>` copy held by each `Worker`.
#[cfg(feature = "cache")]
const WORKER_BUFFER: u64 = 16;
/// `JobRef` is a data pointer plus an execute function pointer.
#[cfg(feature = "cache")]
const JOB_REF: u64 = 16;
/// One worker deque: the `Arc` inner allocation, the inline buffer copy and
/// the initial `MIN_CAP=64` job buffer.
#[cfg(feature = "cache")]
const WORKER_QUEUE: u64 = WORKER_INNER_ARC + WORKER_BUFFER + 64 * JOB_REF;
/// One `Injector` block: next pointer plus 63 slots of `Slot<JobRef>`
/// (16 B task + 8 B state).
#[cfg(feature = "cache")]
const INJECTOR_BLOCK: u64 = 8 + 63 * (JOB_REF + 8);
/// Inline `Worker<T>` in the broadcast vector: `Arc`, `Buffer` cell, flavor.
#[cfg(feature = "cache")]
const WORKER: u64 = 32;
/// Inline `Stealer<T>` stored in `ThreadInfo` and `WorkerThread`.
#[cfg(feature = "cache")]
const STEALER: u64 = 16;
/// Epoch cache line owned by each worker.
#[cfg(feature = "cache")]
const EPOCH_LOCAL: u64 = 2_304;
/// Global epoch plus its pinned sentinel node.
#[cfg(feature = "cache")]
const EPOCH_GLOBAL: u64 = 2_712;
/// `Vec<CachePadded<WorkerSleepState>>` element: `crossbeam_utils` pads the
/// mutex+condvar pair to the x86_64 128 B cache line.
#[cfg(feature = "cache")]
const SLEEP_STATE: u64 = align128(LOCK_LATCH);
/// `Mutex<bool>` + `Condvar` of one `LockLatch`.
#[cfg(feature = "cache")]
const LOCK_LATCH: u64 = std::mem::size_of::<(std::sync::Mutex<bool>, std::sync::Condvar)>() as u64;
/// `OnceLatch` wraps one `AtomicUsize` core latch.
#[cfg(feature = "cache")]
const ONCE_LATCH: u64 = std::mem::size_of::<std::sync::atomic::AtomicUsize>() as u64;

#[cfg(feature = "cache")]
const fn align8(value: u64) -> u64 {
    value.div_ceil(8) * 8
}

#[cfg(feature = "cache")]
const fn align128(value: u64) -> u64 {
    value.div_ceil(128) * 128
}

/// `ThreadInfo`: two `LockLatch`es, one `OnceLatch` and the worker stealer.
#[cfg(feature = "cache")]
const THREAD_INFO: u64 = align8(2 * LOCK_LATCH + ONCE_LATCH + STEALER);

/// Registry fields excluding separately allocated vectors/blocks: each field's
/// width is separately `CachePadded`-rounded to the x86_64 128 B line. The
/// `thread_infos` and broadcast `Vec` headers, `Sleep{Vec,AtomicCounters}`,
/// injected-jobs `Injector`, three optional handler fat pointers and the
/// terminate counter are all that remain.
#[cfg(feature = "cache")]
const REGISTRY_FIELDS: u64 = align128(24)
    + align128(32)
    + 256
    + align128(std::mem::size_of::<std::sync::Mutex<Vec<u8>>>() as u64)
    + 3 * align128(std::mem::size_of::<Option<Box<dyn Fn() + Send + Sync>>>() as u64)
    + align128(std::mem::size_of::<std::sync::atomic::AtomicUsize>() as u64);

#[cfg(feature = "cache")]
fn rayon_arithmetic() -> Error {
    Error::Limit("global Rayon inventory overflow".into())
}

#[cfg(feature = "cache")]
fn cadd(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or_else(rayon_arithmetic)
}

#[cfg(feature = "cache")]
fn cmul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or_else(rayon_arithmetic)
}

#[cfg(feature = "cache")]
fn csum(parts: &[u64]) -> Result<u64> {
    parts
        .iter()
        .try_fold(0u64, |total, &part| cadd(total, part))
}

#[cfg(feature = "cache")]
const fn min_elements(size: u64) -> u64 {
    if size == 1 {
        8
    } else if size <= 1024 {
        4
    } else {
        1
    }
}

/// `R_t(n)`: retained `RawVec` backing with doubling and the minimum
/// non-zero capacity floor.
#[cfg(feature = "cache")]
fn reserve_bytes(size: u64, n: u64) -> Result<u64> {
    if n == 0 {
        return Ok(0);
    }
    let elements = (n as u128)
        .checked_mul(2)
        .ok_or_else(rayon_arithmetic)?
        .max(min_elements(size) as u128);
    let bytes = elements
        .checked_mul(size as u128)
        .ok_or_else(rayon_arithmetic)?;
    u64::try_from(bytes).map_err(|_| rayon_arithmetic())
}

/// `G_t(n)=2*R_t(n)`: old and new backing overlap.
#[cfg(feature = "cache")]
fn regrow_bytes(size: u64, n: u64) -> Result<u64> {
    cmul(2, reserve_bytes(size, n)?)
}

/// `Arc_a(x)=align_a(align_a(16)+x)` for the registry allocation.
#[cfg(feature = "cache")]
fn arc_header(payload: u64) -> u64 {
    align128(align128(16) + payload)
}

/// Typed process-retained inventory of the one global Rayon pool. Requested
/// thread stacks, guards, TLS, libc/glibc caches and the OS mapping remain
/// external unquantified headroom; this subtotal is application-owned only.
#[cfg(feature = "cache")]
fn rayon_inventory_bytes(threads: usize) -> Result<u64> {
    let p = u64::try_from(threads).map_err(|_| rayon_arithmetic())?;
    if p == 0 {
        return Err(Error::Limit("global Rayon thread count is zero".into()));
    }
    let two_p = cmul(2, p)?;
    // Two deques per worker: the worker deque and the broadcast deque.
    let queues = cmul(two_p, WORKER_QUEUE)?;
    let thread_infos = reserve_bytes(THREAD_INFO, p)?;
    let sleep_states = reserve_bytes(SLEEP_STATE, p)?;
    let broadcasts = reserve_bytes(WORKER, p)?;
    // Initialization coexistence: `workers`/`broadcast_stealers`/`stealers`
    // retain their old backing while the registry vectors collect/unzip.
    let init_workers = regrow_bytes(WORKER, p)?;
    let init_stealers = cmul(2, regrow_bytes(STEALER, p)?)?;
    // One JobFifo block per worker; the global injected-jobs injector owns
    // its own block. `(2P+2)` blocks are the permanent delayed-stealer slack.
    let job_fifos = cmul(p, INJECTOR_BLOCK)?;
    let global_injector = INJECTOR_BLOCK;
    let injector_slack = cmul(cadd(two_p, 2)?, INJECTOR_BLOCK)?;
    let epoch = cadd(cmul(p, EPOCH_LOCAL)?, EPOCH_GLOBAL)?;
    let registry = arc_header(REGISTRY_FIELDS);
    // Pinned std thread-control heap rows: the default `move || thread.run()`
    // closure captures one `ThreadBuilder` by value, and the pool names no
    // thread. The initializer's attempt flag is one allocation, not P.
    let control = cmul(
        p,
        zerogravity_core::resource_arbiter::rust_thread_control_bytes(
            std::alloc::Layout::new::<rayon::ThreadBuilder>(),
            None,
        )?,
    )?;
    let attempt_flag = zerogravity_core::resource_arbiter::thread_attempt_flag_bytes()?;
    csum(&[
        queues,
        thread_infos,
        sleep_states,
        broadcasts,
        init_workers,
        init_stealers,
        job_fifos,
        global_injector,
        injector_slack,
        epoch,
        registry,
        control,
        attempt_flag,
    ])
}

#[cfg(feature = "cache")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RayonState {
    /// No project-owned initialization record; an external pool may exist.
    Unknown,
    Ready {
        threads: usize,
    },
    Failed,
    /// A global pool was completed outside the project owner, so its geometry
    /// is never claimed. Terminal: generic callers keep Rayon's normal
    /// behavior, bounded serving refuses the missing provenance.
    Untracked,
}

#[cfg(feature = "cache")]
struct RayonRecord {
    state: RayonState,
    lease: Option<zerogravity_core::resource_arbiter::GlobalResourceLease>,
}

#[cfg(feature = "cache")]
static GLOBAL_RAYON: std::sync::Mutex<RayonRecord> = std::sync::Mutex::new(RayonRecord {
    state: RayonState::Unknown,
    lease: None,
});

/// Pinned rayon-core 1.13.0 `get_num_threads` resolution, given the raw
/// environment values and the available-parallelism fallback: a positive
/// `RAYON_NUM_THREADS` wins, `0` means available parallelism immediately,
/// missing/invalid values consult the deprecated positive `RAYON_RS_NUM_CPUS`,
/// otherwise available parallelism. rayon-core's registry then clamps with
/// `max_num_threads()`.
#[cfg(feature = "cache")]
fn rayon_thread_count_from(
    modern: Option<&str>,
    legacy: Option<&str>,
    available: impl Fn() -> usize,
) -> usize {
    let parsed = |value: Option<&str>| value.and_then(|value| value.parse::<usize>().ok());
    match parsed(modern) {
        Some(0) => available(),
        Some(count) => count,
        None => match parsed(legacy) {
            Some(count) if count > 0 => count,
            _ => available(),
        },
    }
    .min(rayon::max_num_threads())
}

#[cfg(feature = "cache")]
fn rayon_thread_count() -> usize {
    let modern = std::env::var("RAYON_NUM_THREADS").ok();
    let legacy = std::env::var("RAYON_RS_NUM_CPUS").ok();
    rayon_thread_count_from(modern.as_deref(), legacy.as_deref(), || {
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
    })
}

/// Mirror rayon-core's private `DefaultSpawn`: preserve the builder's
/// requested name/stack (both absent by default) and std's cached/OS default
/// semantics. No explicit stack value is invented here.
#[cfg(feature = "cache")]
fn spawn_default(thread: rayon::ThreadBuilder) -> std::io::Result<()> {
    let mut builder = std::thread::Builder::new();
    if let Some(name) = thread.name() {
        builder = builder.name(name.to_owned());
    }
    if let Some(stack_size) = thread.stack_size() {
        builder = builder.stack_size(stack_size);
    }
    builder.spawn(move || thread.run())?;
    Ok(())
}

/// Private spawn seam: production passes [`spawn_default`] itself, so the
/// real thread body is unchanged. Tests may inject a bounded failure.
#[cfg(feature = "cache")]
type SpawnThread = fn(rayon::ThreadBuilder) -> std::io::Result<()>;

/// Only seam: one real global build with the production geometry and the
/// attempt flag set immediately before each actual spawn.
#[cfg(feature = "cache")]
fn build_global(
    threads: usize,
    attempted: std::sync::Arc<std::sync::atomic::AtomicBool>,
    spawn: SpawnThread,
) -> std::result::Result<(), rayon::ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .spawn_handler(move |thread| {
            attempted.store(true, std::sync::atomic::Ordering::SeqCst);
            spawn(thread)
        })
        .build_global()
}

/// Build the global pool with the original default geometry, recording one
/// Ready/Failed outcome for a later bounded-serving admission. Generic callers
/// keep Rayon's normal initialization result and never force a local pool.
#[cfg(feature = "cache")]
pub(crate) fn record_generic_rayon() {
    if rayon::current_thread_index().is_some() {
        return;
    }
    let Ok(mut record) = GLOBAL_RAYON.lock() else {
        return;
    };
    if record.state != RayonState::Unknown {
        return;
    }
    let threads = rayon_thread_count();
    let attempted = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    match build_global(threads, attempted.clone(), spawn_default) {
        Ok(()) => record.state = RayonState::Ready { threads },
        Err(_) if attempted.load(std::sync::atomic::Ordering::SeqCst) => {
            record.state = RayonState::Failed;
        }
        // A pre-existing Once completed elsewhere: keep generic behavior and
        // record the terminal Untracked provenance instead of retrying the
        // build or claiming another owner's geometry.
        Err(_) => record.state = RayonState::Untracked,
    }
}

#[cfg(not(feature = "cache"))]
pub(crate) fn record_generic_rayon() {}

#[cfg(feature = "cache")]
fn admit_rayon_inventory(
    admission: &zerogravity_core::resource_arbiter::TransferAdmission,
    threads: usize,
) -> Result<zerogravity_core::resource_arbiter::GlobalResourceLease> {
    let worker_bytes = rayon_inventory_bytes(threads)?;
    // The global reservation owns two allocation records. Its inventory comes
    // from the values `admit` consumes: the typed rows, the cloned key payload
    // and this fresh owner's spare capacity.
    let owner = zerogravity_core::resource_arbiter::ResourceOwner::new(
        &admission.owner.model,
        &admission.owner.session,
    )?;
    let inventory = zerogravity_core::resource_arbiter::lease_actual_inventory_bytes(
        &admission.resource,
        &owner,
        2,
    )?;
    let allocations = [
        zerogravity_core::resource_arbiter::AllocationRequest {
            kind: zerogravity_core::resource_arbiter::AllocationClass::Scratch,
            bytes: worker_bytes,
        },
        zerogravity_core::resource_arbiter::AllocationRequest {
            kind: zerogravity_core::resource_arbiter::AllocationClass::Other,
            bytes: inventory,
        },
    ];
    admission
        .arbiter
        .admit(&admission.resource, owner, &allocations)
}

/// Admit the one process-lifetime global Rayon worker storage before its
/// threads exist. A later caller reuses that owner; it does not spawn a second
/// global pool or silently re-charge a different host domain. A generic-first
/// Ready pool attaches its exact recorded geometry on the first serving call.
#[cfg(feature = "cache")]
pub fn admit_global_rayon(
    admission: &zerogravity_core::resource_arbiter::TransferAdmission,
) -> Result<()> {
    admit_global_rayon_with(admission, spawn_default)
}

/// The public admission above over the private spawn seam; production passes
/// the original [`spawn_default`] function item.
#[cfg(feature = "cache")]
fn admit_global_rayon_with(
    admission: &zerogravity_core::resource_arbiter::TransferAdmission,
    spawn: SpawnThread,
) -> Result<()> {
    let mut record = GLOBAL_RAYON
        .lock()
        .map_err(|_| Error::State("global Rayon admission mutex poisoned".into()))?;
    // The process-retained runtime thread-control root, in this original
    // arbiter/key domain, before any bounded pool thread is built or attached.
    zerogravity_core::resource_arbiter::admit_thread_control_root(
        &admission.arbiter,
        &admission.resource,
        zerogravity_core::resource_arbiter::ResourceOwner::new(
            &admission.owner.model,
            &admission.owner.session,
        )?,
    )?;
    match record.state {
        RayonState::Ready { threads } => {
            if let Some(lease) = record.lease.as_ref() {
                // A process-lifetime pool cannot move domains. Refuse a
                // different arbiter/key rather than silently inheriting
                // another host's charge.
                lease.bytes_in_domain(&admission.arbiter, &admission.resource)?;
                return Ok(());
            }
            let lease = admit_rayon_inventory(admission, threads)?;
            record.lease = Some(lease);
            return Ok(());
        }
        RayonState::Failed => {
            return Err(Error::State(
                "global Rayon initialization previously failed; bounded serving cannot claim its provenance"
                    .into(),
            ));
        }
        RayonState::Untracked => {
            return Err(Error::State(
                "global Rayon pool was initialized outside the project owner; bounded serving cannot claim its provenance"
                    .into(),
            ));
        }
        RayonState::Unknown => {}
    }

    let threads = rayon_thread_count();
    // Publish the typed credit before thread creation. A partial failed spawn
    // keeps this process-retained credit forever.
    let lease = admit_rayon_inventory(admission, threads)?;
    record.lease = Some(lease);
    let attempted = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    match build_global(threads, attempted.clone(), spawn) {
        Ok(()) => {
            record.state = RayonState::Ready { threads };
            Ok(())
        }
        Err(error) if attempted.load(std::sync::atomic::Ordering::SeqCst) => {
            record.state = RayonState::Failed;
            Err(Error::State(format!(
                "global Rayon worker spawn failed after admission: {error}"
            )))
        }
        Err(error) => {
            // The Once was completed outside the project owner before our
            // spawn handler ran. No thread used this credit, but the pool is
            // now terminal and cannot be re-owned: keep the retained charge and
            // record the terminal Untracked provenance, then report it.
            record.state = RayonState::Untracked;
            Err(Error::State(format!(
                "global Rayon pool was initialized outside the project owner: {error}"
            )))
        }
    }
}

#[cfg(all(test, feature = "cache"))]
mod tests {
    use super::*;

    #[test]
    fn rayon_thread_count_matches_pinned_rayon_core_resolution() {
        let available = || 12;
        // Positive modern wins over legacy.
        assert_eq!(rayon_thread_count_from(Some("3"), Some("5"), available), 3);
        // Modern zero means available parallelism immediately.
        assert_eq!(rayon_thread_count_from(Some("0"), Some("5"), available), 12);
        // Missing/invalid modern consult the deprecated positive legacy value.
        assert_eq!(rayon_thread_count_from(None, Some("4"), available), 4);
        assert_eq!(
            rayon_thread_count_from(Some("bogus"), Some("4"), available),
            4
        );
        assert_eq!(rayon_thread_count_from(Some(""), None, available), 12);
        // Legacy zero/absent falls back to available parallelism.
        assert_eq!(rayon_thread_count_from(None, Some("0"), available), 12);
        assert_eq!(rayon_thread_count_from(None, None, available), 12);
        // The registry clamp is preserved.
        assert_eq!(
            rayon_thread_count_from(Some("9999999"), None, available),
            rayon::max_num_threads()
        );
    }

    /// The global `Once` cannot be re-owned inside one process, so the foreign
    /// initialization and the bounded admission run in a re-executed child.
    #[test]
    fn bounded_admission_records_untracked_after_a_foreign_global_pool() {
        const CHILD: &str = "ZG_RAYON_FOREIGN_CHILD";
        if std::env::var(CHILD).is_err() {
            let exe = std::env::current_exe().unwrap();
            let output = std::process::Command::new(exe)
                .args([
                    "--exact",
                    "tests::bounded_admission_records_untracked_after_a_foreign_global_pool",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(output.status.success(), "child failed: {stdout}\n{stderr}");
            assert!(stdout.contains("FOREIGN_CHILD_OK"), "child: {stdout}");
            return;
        }
        // Foreign owner: rayon's own default spawn handler completes the Once.
        let _ = rayon::current_num_threads();
        let arbiter = zerogravity_core::resource_arbiter::ResourceArbiter::new();
        let key =
            zerogravity_core::resource_arbiter::PhysicalDeviceKey::new("host", "foreign").unwrap();
        arbiter.register(key.clone(), 64 * 1024 * 1024).unwrap();
        let admission = zerogravity_core::resource_arbiter::TransferAdmission {
            arbiter: arbiter.clone(),
            resource: key.clone(),
            owner: zerogravity_core::resource_arbiter::ResourceOwner::new("foreign", "rayon")
                .unwrap(),
        };
        let error = admit_global_rayon(&admission).unwrap_err().to_string();
        assert!(
            error.contains("initialized outside the project owner"),
            "{error}"
        );
        // Terminal provenance, retained credit and the actual domain charge.
        let record = GLOBAL_RAYON.lock().unwrap();
        assert!(matches!(record.state, RayonState::Untracked));
        assert!(record.lease.is_some());
        drop(record);
        let reserved = arbiter.snapshot(&key).unwrap().reserved;
        assert!(reserved > 0);
        // Reuse never rebuilds or re-charges; it reports the same provenance.
        let again = admit_global_rayon(&admission).unwrap_err().to_string();
        assert!(
            again.contains("bounded serving cannot claim its provenance"),
            "{again}"
        );
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, reserved);
        println!("FOREIGN_CHILD_OK");
    }

    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[test]
    fn rayon_inventory_geometry_is_checked_and_grows_with_threads() {
        assert_eq!(WORKER_QUEUE, 1_424);
        assert_eq!(INJECTOR_BLOCK, 1_520);
        assert_eq!(EPOCH_LOCAL, 2_304);
        let one = rayon_inventory_bytes(1).unwrap();
        let two = rayon_inventory_bytes(2).unwrap();
        assert!(one > 0);
        assert!(two > one);
        assert_eq!(one % 8, 0);
        assert_eq!(two % 8, 0);
        assert!(rayon_inventory_bytes(0).is_err());
    }
}

#[cfg(all(test, feature = "cache"))]
mod production_tests {
    use super::*;
    use std::io::Write as _;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, mpsc};
    use std::time::Duration;
    use zerogravity_core::resource_arbiter::{
        PhysicalDeviceKey, ResourceArbiter, ResourceOwner, TransferAdmission,
    };

    const COLUMNS: usize = 8;
    const ROWS: usize = 4;

    fn string(out: &mut Vec<u8>, text: &str) {
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }

    /// Tiny synthetic genuine GGUF v3: one F32 matrix, alignment 32.
    fn tiny_gguf(path: &std::path::Path, name: &str, values: &[f32]) {
        let mut header = b"GGUF".to_vec();
        header.extend_from_slice(&3u32.to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        header.extend_from_slice(&1u64.to_le_bytes());
        string(&mut header, "general.architecture");
        header.extend_from_slice(&8u32.to_le_bytes());
        string(&mut header, "test");
        string(&mut header, name);
        header.extend_from_slice(&2u32.to_le_bytes());
        header.extend_from_slice(&(COLUMNS as u64).to_le_bytes());
        header.extend_from_slice(&(ROWS as u64).to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes());
        while !header.len().is_multiple_of(32) {
            header.push(0);
        }
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&header).unwrap();
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        file.sync_all().unwrap();
    }

    fn fixture(tag: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("zg-cpu-{tag}-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let values = (0..(COLUMNS * ROWS)).map(|v| v as f32).collect::<Vec<_>>();
        tiny_gguf(&root.join("w.gguf"), "w", &values);
        root
    }

    fn input() -> Vec<f32> {
        (1..=COLUMNS).map(|value| value as f32).collect()
    }

    /// Independent naive reference, not the crate's dot_row.
    fn expected() -> Vec<f32> {
        let activation = input();
        (0..ROWS)
            .map(|row| {
                (0..COLUMNS)
                    .map(|column| (row * COLUMNS + column) as f32 * activation[column])
                    .sum()
            })
            .collect()
    }

    fn admission(name: &str) -> (ResourceArbiter, PhysicalDeviceKey, TransferAdmission) {
        let arbiter = ResourceArbiter::new();
        let key = PhysicalDeviceKey::new("host", name).unwrap();
        arbiter.register(key.clone(), 512 * 1024 * 1024).unwrap();
        let admission = TransferAdmission {
            arbiter: arbiter.clone(),
            resource: key.clone(),
            owner: ResourceOwner::new("cpu-projection", "rayon").unwrap(),
        };
        (arbiter, key, admission)
    }

    fn reexec(marker: &str, test: &str, done: &str) -> bool {
        reexec_with(marker, test, done, &[])
    }

    fn reexec_with(marker: &str, test: &str, done: &str, envs: &[(&str, &str)]) -> bool {
        if std::env::var_os(marker).is_some() {
            return false;
        }
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", test, "--nocapture"])
            .env(marker, "1");
        for (key, value) in envs {
            command.env(key, value);
        }
        let output = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "child failed: {stdout}\n{stderr}");
        assert!(stdout.contains(done), "child marker missing: {stdout}");
        true
    }

    #[test]
    fn generic_projection_precedes_the_bounded_attachment() {
        let test = "production_tests::generic_projection_precedes_the_bounded_attachment";
        if reexec("ZG_RAYON_PROJECTION_CHILD", test, "PROJECTION_CHILD_OK") {
            return;
        }
        let root = fixture("projection");
        let source = weights::CpuWeights::open(&root.join("w.gguf")).unwrap();
        // The real generic source projection runs FIRST, before any admission.
        let out = weights::matvec_with_source(&source, "w", &input(), None).unwrap();
        assert_eq!(out, expected());
        {
            let record = GLOBAL_RAYON.lock().unwrap();
            assert!(matches!(
                record.state,
                RayonState::Ready { threads } if threads == rayon_thread_count()
            ));
            assert!(record.lease.is_none(), "generic caller holds no credit");
        }
        // Bounded serving attaches its typed credit to the ready geometry.
        let (arbiter, key, admission) = admission("projection-first");
        admit_global_rayon(&admission).unwrap();
        let reserved = arbiter.snapshot(&key).unwrap().reserved;
        assert!(reserved > 0);
        {
            let record = GLOBAL_RAYON.lock().unwrap();
            assert!(record.lease.is_some(), "retained lease after attach");
        }
        // Repeated attach observes the same owner without a new charge.
        admit_global_rayon(&admission).unwrap();
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, reserved);
        std::fs::remove_dir_all(root).unwrap();
        println!("PROJECTION_CHILD_OK");
    }

    #[test]
    fn local_pool_install_keeps_the_projection_off_the_global_pool() {
        let test = "production_tests::local_pool_install_keeps_the_projection_off_the_global_pool";
        if reexec("ZG_RAYON_LOCAL_CHILD", test, "LOCAL_CHILD_OK") {
            return;
        }
        let root = fixture("local");
        let source = weights::CpuWeights::open(&root.join("w.gguf")).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let (out, index, threads) = pool.install(|| {
            let out = weights::matvec_with_source(&source, "w", &input(), None).unwrap();
            (
                out,
                rayon::current_thread_index(),
                rayon::current_num_threads(),
            )
        });
        assert_eq!(pool.current_num_threads(), 2);
        assert!(index.is_some(), "projection ran on a pool worker");
        assert_eq!(threads, 2, "existing local pool geometry");
        assert_eq!(out, expected());
        {
            let record = GLOBAL_RAYON.lock().unwrap();
            assert!(
                matches!(record.state, RayonState::Unknown),
                "local install must not force the default global pool"
            );
            assert!(record.lease.is_none());
        }
        drop(pool);
        std::fs::remove_dir_all(root).unwrap();
        println!("LOCAL_CHILD_OK");
    }

    static SPAWN_CALLS: AtomicUsize = AtomicUsize::new(0);
    static SPAWN_HANDLES: Mutex<Vec<std::thread::JoinHandle<()>>> = Mutex::new(Vec::new());
    /// Reserved bytes observed at the moment of the first real spawn.
    static SPAWN_RESERVED: AtomicUsize = AtomicUsize::new(0);
    static SPAWN_DOMAIN: Mutex<Option<(ResourceArbiter, PhysicalDeviceKey)>> = Mutex::new(None);

    /// First spawn is the real production body; the second is refused.
    fn second_spawn_fails(thread: rayon::ThreadBuilder) -> std::io::Result<()> {
        if SPAWN_CALLS.fetch_add(1, Ordering::SeqCst) > 0 {
            return Err(std::io::Error::other("injected spawn failure"));
        }
        if let Some((arbiter, key)) = SPAWN_DOMAIN.lock().unwrap().as_ref() {
            SPAWN_RESERVED.store(
                arbiter.snapshot(key).unwrap().reserved as usize,
                Ordering::SeqCst,
            );
        }
        let mut builder = std::thread::Builder::new().name("zg-rayon-witness".into());
        if let Some(stack_size) = thread.stack_size() {
            builder = builder.stack_size(stack_size);
        }
        SPAWN_HANDLES
            .lock()
            .unwrap()
            .push(builder.spawn(move || thread.run())?);
        Ok(())
    }

    #[test]
    fn failed_second_spawn_keeps_the_published_credit_terminal() {
        let test = "production_tests::failed_second_spawn_keeps_the_published_credit_terminal";
        if reexec_with(
            "ZG_RAYON_FAILED_SPAWN_CHILD",
            test,
            "FAILED_SPAWN_CHILD_OK",
            &[("RAYON_NUM_THREADS", "2")],
        ) {
            return;
        }
        let (arbiter, key, admission) = admission("failed-spawn");
        *SPAWN_DOMAIN.lock().unwrap() = Some((arbiter.clone(), key.clone()));
        let error = admit_global_rayon_with(&admission, second_spawn_fails)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("worker spawn failed after admission"),
            "{error}"
        );
        assert_eq!(SPAWN_CALLS.load(Ordering::SeqCst), 2);
        // The typed credit was already reserved when the first real spawn ran.
        assert!(SPAWN_RESERVED.load(Ordering::SeqCst) > 0);
        // The first worker really ran its production body and then exited.
        let handle = SPAWN_HANDLES.lock().unwrap().pop().expect("first worker");
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = sender.send(handle.join().is_ok());
        });
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(20)),
            Ok(true),
            "spawned worker thread did not exit"
        );
        {
            let record = GLOBAL_RAYON.lock().unwrap();
            assert!(matches!(record.state, RayonState::Failed));
            assert!(record.lease.is_some(), "published credit is retained");
            assert!(record.lease.as_ref().unwrap().reserved_bytes().unwrap() > 0);
        }
        let reserved = arbiter.snapshot(&key).unwrap().reserved;
        assert!(reserved > 0);
        // Terminal reuse neither spawns nor charges again.
        let again = admit_global_rayon_with(&admission, second_spawn_fails)
            .unwrap_err()
            .to_string();
        assert!(again.contains("previously failed"), "{again}");
        assert_eq!(SPAWN_CALLS.load(Ordering::SeqCst), 2);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, reserved);
        println!("FAILED_SPAWN_CHILD_OK");
    }
}
