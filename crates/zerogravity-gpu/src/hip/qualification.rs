//! Owned finite original-HIP-cache probe; no trained-model/numerical permission.
use crate::physical_service::NativeContextBudget;
use crate::resident_leases::{ForeignProviderCharge, NativeCacheBudget, ResidentService};
use cubecl::hip::{HipNativeAdmission, HipNativeInventory, NativeRetention, PreparedHipNative};
use cubecl::server::ServerError;
use cubecl::{Device, client::Client, prelude::*};
use cubecl_server::kernel::CubeKernel;
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use zerogravity_core::{Error, Result};

const GIB: u64 = 1024 * 1024 * 1024;
const POLICY: &str = "x2-gfx1151-native-four-v1-qualification-only";
const DEVICE_CAP: u64 = 16 * GIB;
const PROVIDER_CAP: usize = (8 * GIB) as usize;
const CONTEXT_DEVICE: u64 = 2 * GIB;
const CONTEXT_HOST: u64 = GIB;
const HOST_CAP: u64 = 16 * GIB;
const COMPILER_HOST: u64 = 4 * GIB;
const MODULE_OPAQUE: usize = GIB as usize;
const MAX_KNOWN: usize = 256 * 1024 * 1024;
const WORDS: usize = 1024;
const KERNELS: usize = 4;
const CYCLES: usize = 4;

fn server(error: Error) -> ServerError {
    ServerError::graph_state(error.to_string())
}
fn host(bytes: u64, role: &str) -> Result<GlobalResourceLease> {
    ResourceArbiter::process_shared().admit(
        &PhysicalDeviceKey::new("host", "system-ram")?,
        ResourceOwner::new(
            if role == "persistent-context-host" {
                "backend/hip"
            } else {
                "native-owner-fixture"
            },
            role,
        )?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes,
        }],
    )
}
struct Fixture {
    identity: SourceIdentity,
    _host: GlobalResourceLease,
}
struct PersistentHost(Arc<GlobalResourceLease>);
impl Drop for PersistentHost {
    fn drop(&mut self) {
        std::mem::forget(self.0.clone());
    }
}
struct ContextOwner {
    device: Arc<NativeContextBudget>,
    _host: Arc<PersistentHost>,
}
impl Drop for ContextOwner {
    fn drop(&mut self) {
        // The initial policy has no established HIP/RTC cache destruction
        // boundary. Failed attachment must preserve host baseline as well.
        std::mem::forget(self._host.clone());
    }
}
struct HostOwner {
    _host: GlobalResourceLease,
    _source: Arc<Fixture>,
}
struct ModuleOwner {
    _device: Arc<ForeignProviderCharge>,
    _host: GlobalResourceLease,
    _source: Arc<Fixture>,
}
#[derive(Default)]
struct Totals {
    modules: usize,
    code: usize,
    host: usize,
}
struct Admission {
    budget: Arc<NativeCacheBudget>,
    context: Arc<ContextOwner>,
    fixture: Arc<Fixture>,
    physical: PhysicalDeviceKey,
    service: cubecl_common::device::ServiceId,
    totals: Mutex<Totals>,
}
impl HipNativeAdmission for Admission {
    fn context_owner(&self) -> std::result::Result<NativeRetention, ServerError> {
        self.context
            .device
            .require(&self.physical, self.service, POLICY)
            .map_err(server)?;
        Ok(self.context.clone())
    }
    fn require_usable(&self) -> std::result::Result<(), ServerError> {
        self.context
            .device
            .require(&self.physical, self.service, POLICY)
            .map_err(server)?;
        self.budget.require_usable().map_err(server)
    }
    fn fence_unknown(&self) {
        self.context.device.fence_unknown();
        self.budget.fence_unknown();
    }
    fn reserve_preparation(
        &self,
        kernels: usize,
    ) -> std::result::Result<NativeRetention, ServerError> {
        self.require_usable()?;
        if kernels != KERNELS {
            return Err(ServerError::graph_state("finite native four-kernel policy"));
        }
        Ok(Arc::new(HostOwner {
            _host: host(COMPILER_HOST, "compiler").map_err(server)?,
            _source: self.fixture.clone(),
        }))
    }
    fn reserve_module(
        &self,
        artifact: HipNativeInventory,
        code: &[i8],
    ) -> std::result::Result<NativeRetention, ServerError> {
        self.require_usable()?;
        let mut total = self
            .totals
            .lock()
            .map_err(|_| ServerError::graph_state("native policy totals poisoned"))?;
        let modules = total
            .modules
            .checked_add(artifact.modules)
            .filter(|n| *n <= KERNELS)
            .ok_or_else(|| ServerError::graph_state("native policy module bound"))?;
        let load_bytes = total
            .code
            .checked_add(artifact.code_object_load_bytes)
            .filter(|n| *n <= MAX_KNOWN)
            .ok_or_else(|| ServerError::graph_state("native policy known-code bound"))?;
        let host_bytes = total
            .host
            .checked_add(artifact.retained_host_bytes)
            .filter(|n| *n <= 2 * MAX_KNOWN)
            .ok_or_else(|| ServerError::graph_state("native policy retained-host bound"))?;
        let device_bytes = artifact
            .code_object_load_bytes
            .checked_add(if total.modules == 0 { MODULE_OPAQUE } else { 0 })
            .ok_or_else(|| ServerError::graph_state("native policy module allowance overflow"))?;
        let device = self.budget.reserve(device_bytes).map_err(server)?;
        let retained =
            host(artifact.retained_host_bytes as u64, "actual-module-binary").map_err(server)?;
        *total = Totals {
            modules,
            code: load_bytes,
            host: host_bytes,
        };
        let mut hash = Sha256::new();
        let mut scratch = [0u8; 65536];
        for chunk in code.chunks(scratch.len()) {
            for (output, input) in scratch.iter_mut().zip(chunk) {
                *output = *input as u8;
            }
            hash.update(&scratch[..chunk.len()]);
        }
        eprintln!(
            "NATIVE_ACTUAL_CODE module={modules} bytes={} sha256={:x} load_segment_bytes={} retained_host_bytes={}",
            code.len(),
            hash.finalize(),
            artifact.code_object_load_bytes,
            artifact.retained_host_bytes
        );

        Ok(Arc::new(ModuleOwner {
            _device: device,
            _host: retained,
            _source: self.fixture.clone(),
        }))
    }
}

#[cube(launch_unchecked)]
fn probe(output: &mut [u32], #[comptime] salt: u32) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        output[i] = u32::cast_from(i) * 17u32 + salt;
    }
}
fn kernels(client: &Client) -> Vec<Box<dyn CubeKernel>> {
    (0..KERNELS as u32)
        .map(|salt| {
            let kernel = probe::Probe::new(
                KernelSettings::new(
                    CubeDim::new_1d(32).into(),
                    ExecutionMode::Unchecked,
                    AddressType::U32,
                ),
                client.properties_shared(),
                client.target_properties_shared(),
                BufferCompilationArg { inplace: None },
                salt,
            );
            eprintln!("NATIVE_KERNEL salt={salt} id={:?}", kernel.id());
            Box::new(kernel) as Box<dyn CubeKernel>
        })
        .collect()
}
fn observe(stage: &str, service: &Arc<ResidentService>) -> Result<()> {
    let physical = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
    let host = ResourceArbiter::process_shared()
        .snapshot(&PhysicalDeviceKey::new("host", "system-ram")?)?;
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let rss = status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .unwrap_or("VmRSS: unavailable");
    let hwm = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .unwrap_or("VmHWM: unavailable");
    eprintln!(
        "NATIVE_OBSERVATION stage={stage} physical={} host={} managed_plus_module={} context={} {rss} {hwm}",
        physical.reserved,
        host.reserved,
        service.resident_bytes()?,
        CONTEXT_DEVICE
    );
    Ok(())
}

async fn run(unknown: bool, sealed_miss: bool) -> Result<()> {
    use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
    let mut config = CubeClRuntimeConfig::default();
    config.streaming.max_streams = 1;
    config.streaming.policy = cubecl_runtime::config::streaming::StreamPolicy::Single;
    // Default compilation cache=false; exact set recompiles in each cycle.
    // Programmatic cold config prevents external env/file policy substitution.
    if !CubeClRuntimeConfig::try_set(config.clone()) {
        return Err(Error::State(
            "native qualifier needs untouched runtime configuration".into(),
        ));
    }
    eprintln!("NATIVE_RUNTIME_CONFIG {config:?}");
    let arbiter = ResourceArbiter::process_shared();
    let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
    arbiter.register_matching(host_key.clone(), HOST_CAP)?;
    let fixture = Arc::new(Fixture {
        identity: SourceIdentity::for_new_allocation()?,
        _host: host(1024 * 1024, "source")?,
    });
    let context_host = Arc::new(PersistentHost(Arc::new(host(
        CONTEXT_HOST,
        "persistent-context-host",
    )?)));
    let (service, context_device) = ResidentService::new_with_native_context(
        Device::Hip(cubecl::device::AmdDevice::new(0)),
        fixture.identity,
        ResourceOwner::new("native-owner-fixture", "original-service")?,
        PROVIDER_CAP,
        DEVICE_CAP,
        Arc::from(POLICY),
        CONTEXT_DEVICE,
    )?;
    if service.physical_key() != PhysicalDeviceKey::new("pci", "0000:c6:00.0")? {
        return Err(Error::State("native qualification exact x2 PCI".into()));
    }
    let environment = PreparedHipNative::environment(service.client())
        .map_err(|e| Error::State(format!("native environment: {e}")))?;
    if !environment.fingerprint.starts_with("hip-kernel_gfx1151") {
        return Err(Error::State("native qualification gfx1151 only".into()));
    }
    eprintln!("NATIVE_ENVIRONMENT {environment:?}");
    let context = Arc::new(ContextOwner {
        device: context_device,
        _host: context_host,
    });
    assert_eq!(service.capacity(), PROVIDER_CAP - CONTEXT_DEVICE as usize);
    let budget = service.native_cache_budget()?;
    observe("charged-bootstrap", &service)?;
    for cycle in 0..if unknown || sealed_miss { 1 } else { CYCLES } {
        let admission = Arc::new(Admission {
            budget: budget.clone(),
            context: context.clone(),
            fixture: fixture.clone(),
            physical: service.physical_key(),
            service: service.client().service_id(),
            totals: Mutex::new(Totals::default()),
        });
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: (WORDS * 4) as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        // SAFETY: complete exact four generated specializations, actual original
        // host/native owners, one actor/default stream and owned fair slot.
        let native = unsafe {
            PreparedHipNative::prepare(
                service.client(),
                kernels(service.client()),
                admission.clone(),
            )
        }
        .map_err(|e| Error::State(format!("actual native prepare: {e}")))?;
        assert_eq!(native.inventory().modules, KERNELS);
        eprintln!("NATIVE_INVENTORY cycle={cycle} {:?}", native.inventory());
        let retained_bytes = service.resident_bytes()?;
        // SAFETY: same authenticated admission/catalogue/actor. Policy must refuse
        // this second live set before preparation changes the first set.
        let second = unsafe {
            PreparedHipNative::prepare(
                service.client(),
                kernels(service.client()),
                admission.clone(),
            )
        };
        let refusal = match second {
            Err(error) => error.to_string(),
            Ok(_) => {
                budget.fence_unknown();
                return Err(Error::State(
                    "second active HIP native set unexpectedly accepted".into(),
                ));
            }
        };
        if !refusal.contains("HIP native preparation needs a cold/unowned cache") {
            return Err(Error::State(format!(
                "unexpected second-set refusal: {refusal}"
            )));
        }
        native
            .require_live(service.client())
            .map_err(|e| Error::State(format!("first native set after second refusal: {e}")))?;
        if retained_bytes != service.resident_bytes()? {
            budget.fence_unknown();
            return Err(Error::State(
                "second active native refusal changed original model charge".into(),
            ));
        }
        eprintln!(
            "NATIVE_SECOND_ACTIVE_REFUSED cycle={cycle} first_live=true retained_bytes={retained_bytes}"
        );
        let maps = std::fs::read_to_string("/proc/self/maps")
            .map_err(|e| Error::State(format!("native runtime maps: {e}")))?;
        let paths = maps
            .lines()
            .filter_map(|line| line.split_whitespace().last())
            .filter(|path| {
                path.starts_with("/nix/store/")
                    && [
                        "libamdhip",
                        "libhiprtc",
                        "libamd_comgr",
                        "libhsa-runtime",
                        "libLLVM",
                        "libc.so",
                        "libstdc++",
                    ]
                    .iter()
                    .any(|name| path.contains(name))
            })
            .collect::<std::collections::BTreeSet<_>>();
        for path in paths {
            eprintln!("NATIVE_RUNTIME_MAPPING cycle={cycle} path={path}");
        }

        observe("compiled-loaded", &service)?;
        let mut initialization = service.begin_initialization(0)?;
        initialization.retain_completion_owner(native.clone())?;
        let output = initialization.upload_bytes(&vec![0; WORDS * 4], BufferRole::Output)?;
        drop(initialization.complete_sync().await?);
        for launch in 0..if sealed_miss {
            KERNELS + 1
        } else {
            2 * KERNELS
        } {
            let salt = if sealed_miss {
                launch as u32
            } else {
                (launch % KERNELS) as u32
            };
            native
                .require_live(service.client())
                .map_err(|e| Error::State(format!("native set: {e}")))?;
            let mut flight = service.submit(0, &[(output.clone(), DeviceAccess::Write)])?;
            flight.retain_completion_owner(native.clone())?;
            // SAFETY: original paired lease has WORDS u32, same actor and exact
            // prepared variant. The final deliberate miss exercises refusal.
            unsafe {
                probe::launch_unchecked(
                    service.client(),
                    CubeCount::Static((WORDS / 32) as u32, 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(output.handle(), WORDS),
                    salt,
                );
            }
            if unknown {
                drop(flight);
                drop(slot);
                assert!(service.ensure_usable().is_err());
                assert!(native.require_live(service.client()).is_err());
                assert!(native.retire().is_err());
                assert_eq!(
                    arbiter.snapshot(&service.physical_key())?.reserved,
                    PROVIDER_CAP as u64
                );
                observe("actual-launch-unknown-retained", &service)?;
                return Ok(());
            }
            let result = flight
                .complete_with_readback(
                    std::slice::from_ref(&output),
                    &[output.readback(0, WORDS * 4)?],
                    |raw| {
                        if raw.len() != 1 || raw[0].len() != WORDS * 4 {
                            return Err(Error::State("native actual read extent".into()));
                        }
                        for (i, bytes) in raw[0].chunks_exact(4).enumerate() {
                            if u32::from_le_bytes(bytes.try_into().unwrap()) != i as u32 * 17 + salt
                            {
                                return Err(Error::State(format!("native actual output word{i}")));
                            }
                        }
                        Ok(())
                    },
                )
                .await;
            if sealed_miss && salt == KERNELS as u32 {
                assert!(result.is_err());
                assert!(service.ensure_usable().is_err());
                assert_eq!(native.inventory().modules, KERNELS);
                eprintln!("NATIVE_INVENTORY cycle={cycle} {:?}", native.inventory());
                let maps = std::fs::read_to_string("/proc/self/maps")
                    .map_err(|e| Error::State(format!("native runtime maps: {e}")))?;
                let paths = maps
                    .lines()
                    .filter_map(|line| line.split_whitespace().last())
                    .filter(|path| {
                        path.starts_with("/nix/store/")
                            && [
                                "libamdhip",
                                "libhiprtc",
                                "libamd_comgr",
                                "libhsa-runtime",
                                "libLLVM",
                                "libc.so",
                                "libstdc++",
                            ]
                            .iter()
                            .any(|name| path.contains(name))
                    })
                    .collect::<std::collections::BTreeSet<_>>();
                for path in paths {
                    eprintln!("NATIVE_RUNTIME_MAPPING cycle={cycle} path={path}");
                }

                drop(slot);
                observe("sealed-miss-refused-original-flight-retained", &service)?;
                return Ok(());
            }
            drop(result?);
            observe(
                if launch == 0 {
                    "first-launch"
                } else {
                    "warmed-launch"
                },
                &service,
            )?;
        }
        drop(output);
        native
            .retire()
            .map_err(|e| Error::State(format!("actual native unload: {e}")))?;
        assert!(native.require_live(service.client()).is_err());
        drop((native, admission));
        assert_eq!(service.resident_bytes()?, 0);
        slot.finish()?;
        eprintln!("NATIVE_CYCLE known_retired={cycle}");
        observe("modules-unloaded-context-retained", &service)?;
    }
    let physical = service.physical_key();
    drop((budget, service, fixture, context));
    let physical = arbiter.snapshot(&physical)?;
    let host = arbiter.snapshot(&host_key)?;
    assert_eq!(physical.reserved, CONTEXT_DEVICE);
    assert_eq!(host.reserved, CONTEXT_HOST);
    assert_eq!(
        physical.active_models,
        std::collections::BTreeSet::from(["backend/hip".into()])
    );
    eprintln!(
        "NATIVE_FINAL model_bytes=0 context_device={} context_host={} process_retirement_required=true",
        physical.reserved, host.reserved
    );
    Ok(())
}

#[test]
#[ignore = "owned remote gfx1151 process; original cache and fixed policy only"]
fn actual_native_prepared_lifecycle() {
    futures_lite::future::block_on(run(false, false)).unwrap();
}
#[test]
#[ignore = "owned remote gfx1151 process; actual launch unknown retains/fences"]
fn actual_native_prepared_unknown() {
    futures_lite::future::block_on(run(true, false)).unwrap();
}
#[test]
#[ignore = "owned remote gfx1151 process; sealed missing variant must not compile"]
fn actual_native_prepared_sealed_miss() {
    futures_lite::future::block_on(run(false, true)).unwrap();
}
