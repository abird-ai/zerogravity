//! Physical cancellation of numerical-state upload; no model codec.
use super::*;

#[cfg(feature = "hardware-abort-probe")]
#[allow(clippy::too_many_arguments)]
pub async fn abort_upload(
    device: Device,
    source: zerogravity_core::accelerator::SourceIdentity,
    resident: Arc<ResidentModel>,
    mut session: ResidentSession,
    prior: ResidentSnapshot,
    expected: Vec<(StateSlot, Vec<f32>)>,
    stale_ready: PreparedInference,
    position: usize,
    domains: &[(StateSlot, Vec<f32>)],
    device_cap: u64,
) -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    let seen = AtomicBool::new(false);
    let binding = session.binding();
    let uploads = resident.service.uploaded_value_bytes();
    let client = device.client();
    let mut future =
        Box::pin(session.import_state_domains_inner(source, position, domains, Some(&seen)));
    loop {
        if futures_lite::future::poll_once(future.as_mut())
            .await
            .is_some()
        {
            return Err(Error::State("upload abort completed unexpectedly".into()));
        }
        if seen.load(Ordering::Acquire) {
            break;
        }
        futures_lite::future::yield_now().await;
    }
    drop(future);
    let handles = resident
        .states
        .iter()
        .map(|r| match r.layout {
            StateLayout::Vector { .. } => 1,
            StateLayout::Kv { .. } => 2,
        })
        .sum::<usize>();
    let quarantined = crate::physical_quarantine::quarantined_handles_for(&client)?;
    if resident.service.uploaded_value_bytes() - uploads != resident.state_bytes as u64
        || quarantined != handles
        || session.binding() != binding
        || resident.service.ensure_usable().is_ok()
        || session.commit(stale_ready).is_ok()
    {
        return Err(Error::State(
            "checkpoint upload cancellation lost root/handles/fence".into(),
        ));
    }
    if ResidentService::new(
        device,
        source,
        ResourceOwner::new("other-model", "checkpoint-abort")?,
        1024 * 1024,
        device_cap,
    )
    .is_ok()
    {
        return Err(Error::State(
            "checkpoint abort allowed another model".into(),
        ));
    }
    // Diagnostic quiescence/readback ONLY. Never clear fences or report Ready.
    client
        .sync()
        .await
        .map_err(|e| Error::State(format!("checkpoint abort sync: {e}")))?;
    for (domain, (slot, values)) in prior.domains.iter().zip(&expected) {
        if domain.slot != *slot {
            return Err(Error::State("checkpoint abort domain order".into()));
        }
        let read = client
            .read_async(domain.leases.iter().map(ResidentLease::handle).collect())
            .await
            .map_err(|e| Error::State(format!("checkpoint abort readback: {e}")))?;
        if read.len() != domain.leases.len()
            || read
                .iter()
                .zip(&domain.leases)
                .any(|(b, l)| b.len() != l.bytes())
        {
            return Err(Error::State(
                "checkpoint abort short physical readback".into(),
            ));
        }
        let mut flat = Vec::new();
        for (data, lease) in read.iter().zip(&domain.leases) {
            let words = words(data)?;
            let active = if matches!(domain.buffers, StateBuffers::Kv { .. }) {
                lease.bytes() / 4 / prior.capacity * binding.frontier() as usize
            } else {
                words.len()
            };
            if words[active..].iter().any(|x| x.to_bits() != 0) {
                return Err(Error::State(
                    "checkpoint abort prior KV tail changed".into(),
                ));
            }
            flat.extend_from_slice(&words[..active]);
        }
        exact(values, &flat, "checkpoint abort prior physical bits")?;
    }
    let key = resident.service.physical_key();
    let weak = Arc::downgrade(resident.service());
    let cap = resident.service.capacity() as u64;
    drop(prior);
    drop(session);
    drop(resident);
    let kept = weak
        .upgrade()
        .ok_or_else(|| Error::State("checkpoint quarantine lost service".into()))?;
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if budget.reserved != cap || !budget.fenced || kept.physical_remaining()? != device_cap - cap {
        return Err(Error::State(
            "checkpoint quarantine lost full envelope".into(),
        ));
    }
    println!(
        "H21_CHECKPOINT_UPLOAD_ABORT_PASS actual_h2d_submitted=true frontier={} root_unchanged=true prior_physical_bits=exact canonical_kv_tail=exact original_handles={handles} quarantined_handles={quarantined} stale_ready=refused other_model=refused retained_envelope_bytes={cap} trusted_recovery=false",
        binding.frontier()
    );
    Ok(())
}
