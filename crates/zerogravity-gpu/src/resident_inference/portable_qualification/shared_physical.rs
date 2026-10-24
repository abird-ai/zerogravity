//! Actual two-source managed owner/lifetime probe; original Q8/Conv32 arithmetic.
use super::*;
const MODEL: usize = PROVIDER / 2;

async fn load_pair(
    backend: Backend,
) -> Result<(
    Arc<ResidentPhysical>,
    Arc<Source>,
    Arc<ResidentModel>,
    Arc<Source>,
    Arc<ResidentModel>,
)> {
    let device = backend.device();
    backend.validate(&device)?;
    let physical = ResidentPhysical::new(device, PROVIDER, DEVICE)?;
    let a = Source::new()?;
    let b = Source::new_with_salt(5)?;
    if a.identity == b.identity || a.sha256 == b.sha256 {
        return Err(Error::State(
            "shared probe needs different actual source owners/content".into(),
        ));
    }
    if preflight_profile(b.as_ref(), &a.plan()?).is_ok()
        || preflight_profile(a.as_ref(), &b.plan()?).is_ok()
    {
        return Err(Error::State(
            "different actual sources accepted each other's recorded source".into(),
        ));
    }
    let model_a = ResidentModel::load_shared(physical.clone(), a.clone(), a.plan()?, MODEL).await?;
    let model_b = ResidentModel::load_shared(physical.clone(), b.clone(), b.plan()?, MODEL).await?;
    if !Arc::ptr_eq(&model_a.physical_owner(), &physical)
        || !Arc::ptr_eq(&model_b.physical_owner(), &physical)
        || model_a.service.client().service_id() != model_b.service.client().service_id()
        || !Arc::ptr_eq(model_a.service.queue(), model_b.service.queue())
        || model_a.service.generation() == model_b.service.generation()
        || model_a.service.generation() == physical.generation()
        || model_b.service.generation() == physical.generation()
        || !Arc::ptr_eq(
            &model_a.service.budget().retained_envelope(),
            &model_b.service.budget().retained_envelope(),
        )
        || physical.available_model_capacity()? != 0
        || physical.resident_bytes()?
            != model_a.service.resident_bytes()? + model_b.service.resident_bytes()?
    {
        return Err(Error::State(
            "two-source original Client/envelope/quota/generation join".into(),
        ));
    }
    let snapshot = ResourceArbiter::process_shared().snapshot(&model_a.service.physical_key())?;
    if snapshot.reserved != PROVIDER as u64
        || snapshot.reservations != 1
        || snapshot.limit != DEVICE
        || snapshot.fenced
    {
        return Err(Error::State(
            "shared owner minted a second global admission".into(),
        ));
    }
    let bytes = physical.resident_bytes()?;
    if physical
        .attach_model(
            ResidentSource::Tensor(a.clone()),
            ResourceOwner::new("shared-probe", "over-combined-cap")?,
            1,
        )
        .is_ok()
        || physical.resident_bytes()? != bytes
        || physical.available_model_capacity()? != 0
    {
        return Err(Error::State(
            "combined model quota refused with effects".into(),
        ));
    }
    model_a.service.ensure_usable()?;
    model_b.service.ensure_usable()?;
    eprintln!(
        "SHARED_PHYSICAL_SOURCES backend={backend:?} a={} b={} original_provider={PROVIDER} model_quotas={MODEL}+{MODEL} actual_sources=true synthetic=true learned=false",
        a.sha256, b.sha256
    );
    Ok((physical, a, model_a, b, model_b))
}
async fn healthy(backend: Backend) -> Result<()> {
    let (physical, source_a, model_a, source_b, model_b) = load_pair(backend).await?;
    let host_key = source_a.host_key.clone();
    let key = model_a.service.physical_key();
    let source_weak = Arc::downgrade(&source_a);
    let model_weak = Arc::downgrade(&model_a);
    let service_weak = Arc::downgrade(&model_a.service);
    let old_generation = model_a.service.generation();
    let old_identity = source_a.identity;
    let old_hash = source_a.sha256.clone();
    let generation = physical.generation();
    let native_reader = model_a.service.native_cache_budget()?;
    native_reader.require_service(&model_a.service)?;
    if native_reader.require_service(&model_b.service).is_ok() {
        return Err(Error::State(
            "same-Client native source/admission confusion accepted".into(),
        ));
    }
    let mut a = model_a.session().await?;
    let mut b = model_b.session().await?;
    let mut oracle_a = Oracle::new(source_a.clone())?;
    let mut oracle_b = Oracle::new(source_b.clone())?;
    let (ready, next) = checked_prepare(&a, &oracle_a, &[1, 7]).await?;
    oracle_a = next;
    a.commit(ready)?;
    let (ready, next) = checked_prepare(&b, &oracle_b, &[3, 9]).await?;
    oracle_b = next;
    b.commit(ready)?;
    let snapshot = a.snapshot();
    let foreign = snapshot.domains[0].leases[0].clone();
    let peer_lease = b.domains[0].leases[0].clone();
    if foreign.id() == peer_lease.id()
        || foreign.allocation_generation() == peer_lease.allocation_generation()
        || model_b
            .service
            .submit(0, &[(foreign.clone(), DeviceAccess::Read)])
            .is_ok()
        || b.restore(&snapshot).is_ok()
    {
        return Err(Error::State(
            "shared source/lease/root namespace confusion accepted".into(),
        ));
    }
    drop((foreign, peer_lease));
    let before_a = a.binding();
    let before_b = b.binding();
    let (foreign_ready, _) = checked_prepare(&a, &oracle_a, &[11]).await?;
    if b.commit(foreign_ready).is_ok() || a.binding() != before_a || b.binding() != before_b {
        return Err(Error::State(
            "shared source Ready crossed logical authority".into(),
        ));
    }
    let (delayed, next_a) = checked_prepare(&a, &oracle_a, &[11]).await?;
    let (ready, next) = checked_prepare(&b, &oracle_b, &[13]).await?;
    oracle_b = next;
    b.commit(ready)?;
    queue_cancel(&model_b)?;
    model_a.service.ensure_usable()?;
    let expected_history = oracle_a.history.clone();
    drop((a, model_a, source_a, oracle_a, next_a));
    if model_weak.strong_count() != 0
        || service_weak.strong_count() == 0
        || source_weak.strong_count() == 0
        || physical.available_model_capacity()? != 0
    {
        return Err(Error::State(
            "delayed snapshot/Ready lost original source/quota".into(),
        ));
    }
    exact_words(
        &expected_history,
        &snapshot.read_domains().await?[0].1,
        "shared retained source snapshot",
    )?;
    model_b.abort_after_launch.store(1, Ordering::Release);
    let mut pending = Box::pin(b.prepare(&[17], true));
    let active_handles = std::future::poll_fn(|cx| match pending.as_mut().poll(cx) {
        Poll::Ready(_) => Poll::Ready(Err(Error::State(
            "shared delayed flight completed before owned launch pause".into(),
        ))),
        Poll::Pending => {
            let count = model_b.launch_seen.load(Ordering::Acquire);
            if count == 0 {
                Poll::Pending
            } else {
                Poll::Ready(Ok(count))
            }
        }
    })
    .await?;
    if active_handles == 0 || model_b.service.pending_events()? == 0 {
        return Err(Error::State(
            "shared delayed source-B flight lacks actual event/owners".into(),
        ));
    }
    drop((snapshot, delayed));
    if service_weak.strong_count() != 0
        || source_weak.strong_count() == 0
        || physical.available_model_capacity()? != 0
        || physical.resident_bytes()? != model_b.service.resident_bytes()?
    {
        return Err(Error::State(
            "detached native reader retained actor or lost authentic source quota".into(),
        ));
    }
    native_reader.require_usable()?;
    eprintln!(
        "SHARED_RETIREMENT backend={backend:?} phase=last_native_reader model_a={} service_a={} source_a={} quota_available={} resident={} source_b_resident={} host_reserved={}",
        model_weak.strong_count(),
        service_weak.strong_count(),
        source_weak.strong_count(),
        physical.available_model_capacity()?,
        physical.resident_bytes()?,
        model_b.service.resident_bytes()?,
        ResourceArbiter::process_shared()
            .snapshot(&host_key)?
            .reserved
    );
    drop(native_reader);
    eprintln!(
        "SHARED_RETIREMENT backend={backend:?} phase=source_a_retired model_a={} service_a={} source_a={} quota_available={} resident={} source_b_resident={} host_reserved={} sibling_pending={}",
        model_weak.strong_count(),
        service_weak.strong_count(),
        source_weak.strong_count(),
        physical.available_model_capacity()?,
        physical.resident_bytes()?,
        model_b.service.resident_bytes()?,
        ResourceArbiter::process_shared()
            .snapshot(&host_key)?
            .reserved,
        model_b.service.pending_events()?
    );
    if source_weak.strong_count() != 0
        || physical.available_model_capacity()? != MODEL
        || ResourceArbiter::process_shared()
            .snapshot(&host_key)?
            .reserved
            != SOURCE_HOST
    {
        return Err(Error::State(
            "source quota/host release did not follow final actual reader".into(),
        ));
    }
    // Quota retirement above occurred while B's original physical work was live.
    model_b.launch_seen.store(0, Ordering::Release);
    let ready = pending.await?;
    oracle_b
        .advance(&[17])?
        .compare(&ready.readback, b.binding().frontier() as usize)?;
    b.commit(ready)?;
    let source_reload = Source::new()?;
    if source_reload.identity == old_identity || source_reload.sha256 != old_hash {
        return Err(Error::State(
            "shared reload authentic source identity/content".into(),
        ));
    }
    let reload = ResidentModel::load_shared(
        physical.clone(),
        source_reload.clone(),
        source_reload.plan()?,
        MODEL,
    )
    .await?;
    if physical.generation() != generation
        || reload.service.generation() == old_generation
        || !Arc::ptr_eq(&reload.physical_owner(), &physical)
    {
        return Err(Error::State(
            "reload replaced physical owner or reused source namespace".into(),
        ));
    }
    let mut session = reload.session().await?;
    let mut oracle = Oracle::new(source_reload.clone())?;
    let (ready, next) = checked_prepare(&session, &oracle, &[1, 7]).await?;
    oracle = next;
    session.commit(ready)?;
    let (ready, next_b) = checked_prepare(&b, &oracle_b, &[19]).await?;
    b.commit(ready)?;
    drop((
        session,
        reload,
        source_reload,
        oracle,
        b,
        model_b,
        source_b,
        oracle_b,
        next_b,
    ));
    eprintln!(
        "SHARED_RETIREMENT backend={backend:?} phase=all_sources_retired source_a={} quota_available={} resident={} host_reserved={}",
        source_weak.strong_count(),
        physical.available_model_capacity()?,
        physical.resident_bytes()?,
        ResourceArbiter::process_shared()
            .snapshot(&host_key)?
            .reserved
    );
    if physical.resident_bytes()? != 0
        || physical.available_model_capacity()? != PROVIDER
        || ResourceArbiter::process_shared()
            .snapshot(&host_key)?
            .reserved
            != 0
    {
        return Err(Error::State(
            "shared source retirement aggregate/model/host final disposition".into(),
        ));
    }
    let reserved = ResourceArbiter::process_shared().snapshot(&key)?;
    if reserved.reserved != PROVIDER as u64 || reserved.fenced {
        return Err(Error::State(
            "live original physical envelope disappeared".into(),
        ));
    }
    drop(physical);
    if ResourceArbiter::process_shared().snapshot(&key)?.reserved != 0 {
        return Err(Error::State(
            "shared physical retirement retained managed envelope".into(),
        ));
    }
    eprintln!(
        "SHARED_PHYSICAL_HEALTHY_PASS backend={backend:?} actual_two_sources=true one_client=true one_envelope=true different_content=true source_confusion_refused=true combined_cap_refused=true delayed_ready_snapshot_native_reader=true independent_close_reload=true learned=false perf=false"
    );
    Ok(())
}
async fn unknown(backend: Backend) -> Result<()> {
    let (physical, source_a, model_a, source_b, model_b) = load_pair(backend).await?;
    let key = model_a.service.physical_key();
    let host_key = source_a.host_key.clone();
    let source_weak = Arc::downgrade(&source_a);
    let peer_source_weak = Arc::downgrade(&source_b);
    let service_weak = Arc::downgrade(&model_a.service);
    let peer_service_weak = Arc::downgrade(&model_b.service);
    let mut a = model_a.session().await?;
    let mut b = model_b.session().await?;
    let mut oracle_a = Oracle::new(source_a.clone())?;
    let mut oracle_b = Oracle::new(source_b.clone())?;
    let (ready, next) = checked_prepare(&a, &oracle_a, &[1, 7]).await?;
    oracle_a = next;
    a.commit(ready)?;
    let (ready, next) = checked_prepare(&b, &oracle_b, &[3, 9]).await?;
    oracle_b = next;
    b.commit(ready)?;
    let before = a.binding();
    let peer_before = b.binding();
    let peer_bytes = model_b.service.resident_bytes()?;
    let peer_uploads = model_b.service.uploaded_value_bytes();
    let count = crate::physical_quarantine::quarantined_handles_for(model_a.service.client())?;
    let retained = drop_after_cached_launch(&model_a, &a).await?;
    let queue = model_b.service.queue().stats()?;
    if a.binding() != before
        || b.binding() != peer_before
        || model_a.service.ensure_usable().is_ok()
        || model_b.service.ensure_usable().is_ok()
        || model_a.service.pending_events()? == 0
        || b.prepare(&[11], true).await.is_ok()
        || model_b.service.resident_bytes()? != peer_bytes
        || model_b.service.uploaded_value_bytes() != peer_uploads
        || model_b.service.queue().stats()? != queue
        || physical
            .attach_model(
                ResidentSource::Tensor(source_b.clone()),
                ResourceOwner::new("shared-probe", "after-unknown")?,
                1,
            )
            .is_ok()
        || crate::physical_quarantine::quarantined_handles_for(model_a.service.client())?
            < count + retained
    {
        return Err(Error::State(
            "unknown flight escaped shared sibling/source publication fence".into(),
        ));
    }
    drop((
        a, b, model_a, model_b, source_a, source_b, oracle_a, oracle_b,
    ));
    eprintln!(
        "SHARED_RETIREMENT backend={backend:?} phase=unknown_after_fixture_drop source_a={} source_b={} service_a={} service_b={} quota_available={} resident={} host_reserved={}",
        source_weak.strong_count(),
        peer_source_weak.strong_count(),
        service_weak.strong_count(),
        peer_service_weak.strong_count(),
        physical.available_model_capacity()?,
        physical.resident_bytes()?,
        ResourceArbiter::process_shared()
            .snapshot(&host_key)?
            .reserved
    );
    if source_weak.strong_count() == 0
        || peer_source_weak.strong_count() == 0
        || service_weak.strong_count() == 0
        || peer_service_weak.strong_count() == 0
        || physical.available_model_capacity()? != 0
        || physical.resident_bytes()? == 0
        || ResourceArbiter::process_shared()
            .snapshot(&host_key)?
            .reserved
            != 2 * SOURCE_HOST
    {
        return Err(Error::State(
            "unknown shared owner lost authentic source/host/lease/quota retention".into(),
        ));
    }
    let snapshot = ResourceArbiter::process_shared().snapshot(&key)?;
    if !snapshot.fenced || snapshot.reserved != PROVIDER as u64 || snapshot.reservations != 1 {
        return Err(Error::State(
            "unknown shared physical original envelope disposition".into(),
        ));
    }
    drop(physical);
    if service_weak.strong_count() == 0
        || source_weak.strong_count() == 0
        || peer_source_weak.strong_count() == 0
    {
        return Err(Error::State(
            "unknown process boundary lost actual retained owners".into(),
        ));
    }
    eprintln!(
        "SHARED_PHYSICAL_UNKNOWN_PASS backend={backend:?} actual_retained_handles={retained} sibling_fenced=true roots_unchanged=true original_envelope_retained={PROVIDER} authentic_sources_retained=2 host_retained={} native_reader_policy=unchanged learned=false perf=false; process retirement required",
        2 * SOURCE_HOST
    );
    Ok(())
}
#[test]
#[ignore = "pvl-x2 gfx1151 actual shared HIP owner; one owned isolated process"]
fn physical_hip_shared_owner_healthy() -> Result<()> {
    futures_lite::future::block_on(healthy(Backend::Hip))
}
#[test]
#[ignore = "pvl-x2 gfx1151 actual shared HIP unknown; ONE fresh owned process"]
fn physical_hip_shared_owner_unknown() -> Result<()> {
    futures_lite::future::block_on(unknown(Backend::Hip))
}
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "pvl-x2 c6:00.0 actual shared Vulkan owner; one owned isolated process"]
fn physical_vulkan_shared_owner_healthy() -> Result<()> {
    futures_lite::future::block_on(healthy(Backend::Vulkan))
}
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "pvl-x2 c6:00.0 actual shared Vulkan unknown; ONE fresh owned process"]
fn physical_vulkan_shared_owner_unknown() -> Result<()> {
    futures_lite::future::block_on(unknown(Backend::Vulkan))
}
