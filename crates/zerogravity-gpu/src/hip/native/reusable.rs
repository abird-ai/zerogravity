//! Source-neutral, process-lifetime HIP context. A model owns only its
//! scoped native set and fresh physical reservation, not the original actor.
use super::*;
use crate::resident_leases::{ResidentPhysical, ResidentSource};
use std::collections::BTreeMap;
use std::sync::OnceLock;

struct Owner {
    client: Client,
    context: Arc<Context>,
    device_cap: u64,
    // Keep the original fair queue between models; queue_for returns this
    // same owner for every admitted physical envelope.
    _queue: Arc<crate::physical_queue::PhysicalQueue>,
}

fn owners() -> &'static Mutex<BTreeMap<PhysicalDeviceKey, Owner>> {
    static OWNERS: OnceLock<Mutex<BTreeMap<PhysicalDeviceKey, Owner>>> = OnceLock::new();
    OWNERS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub(super) fn begin(
    device: Device,
    source: &VerifiedTensorSource,
    source_owner: crate::physical_quarantine::CompletionOwner,
    profile: SingleSourceProfile,
) -> Result<ColdNativePreparation> {
    source.require_process_host_workspace(SOURCE_HOST_BYTES)?;
    let Device::Hip(selected) = &device else {
        return Err(Error::Unsupported(
            "original HIP source requires HIP device".into(),
        ));
    };
    let observation = cubecl::hip::observe_device(selected.clone())
        .ok_or_else(|| Error::Unsupported("HIP physical device observation unavailable".into()))?;
    let key = crate::physical_service::physical_key(Some(&observation.physical))?;
    let mut owners = owners()
        .lock()
        .map_err(|_| Error::State("HIP context registry poisoned".into()))?;
    let (physical, context) = if let Some(owner) = owners.get(&key) {
        // The source policy can change its model budget, never its live
        // actor's ABI or physical identity. Check before allocating anything.
        owner
            .context
            .device
            .require(&key, owner.client.service_id(), profile.policy())?;
        let cap = profile
            .provider_cap()
            .checked_sub(CONTEXT_DEVICE as usize)
            .ok_or_else(|| Error::Limit("HIP model envelope excludes persistent context".into()))?;
        // Reuse uses the stable original device ceiling, not the fluctuating
        // free-memory-derived model cap. The arbiter also counts the retained
        // context slice and every other admitted provider.
        let physical = ResidentPhysical::reuse_hip_context(
            device,
            &owner.client,
            cap,
            owner.device_cap,
            owner.context.device.clone(),
            profile.policy(),
        )?;
        (physical, owner.context.clone())
    } else {
        let (physical, context) = bootstrap_context(device, source, profile)?;
        // SAFETY: the just-admitted original cold actor has a single physical
        // stream. Every model will attach its own authenticated logical view.
        if let Err(error) = unsafe { RegisteredHipNative::select_scoped(physical.client()) } {
            context.device.fence_unknown();
            return Err(native(error));
        }
        owners.insert(
            key,
            Owner {
                client: physical.client().clone(),
                context: context.clone(),
                device_cap: profile.device_cap(),
                _queue: crate::physical_queue::queue_for(physical.physical_key())?,
            },
        );
        (physical, context)
    };
    drop(owners);
    // SAFETY: same original admitted actor and context; the new source view
    // is installed before any allocations, compilation or execution.
    let scoped = unsafe { RegisteredHipNative::register(physical.client()) }.map_err(native)?;
    let service = physical.attach_native_model(
        ResidentSource::Windows(source.clone()),
        ResourceOwner::new("source-bf16", profile.policy())?,
        physical.available_model_capacity()?,
        &scoped,
    )?;
    Ok(ColdNativePreparation {
        service,
        context,
        source: source.clone(),
        source_owner,
        scoped: Some(scoped),
    })
}
