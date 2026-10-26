//! Shared .8B/2B root checkout: fresh logical leases, unchanged domain binding.
use super::*;
#[cfg(any(
    feature = "cuda-native",
    feature = "gemma-bf16-qualification",
    feature = "qwen-source-qualification"
))]
use crate::resident_leases::backing_pool::ResidentPlaneLayout;
use crate::resident_leases::backing_pool::ResidentPool;
use std::sync::atomic::{AtomicBool, Ordering};

pub(super) struct PooledStates {
    pub pool: Arc<ResidentPool>,
    #[cfg(all(feature = "qualification", target_os = "linux"))]
    pub source: Option<zerogravity_artifacts::source_weights::VerifiedTensorSource>,
}
fn check_cancel(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("resident root checkout cancelled".into()));
    }
    Ok(())
}
pub(crate) async fn allocate_pooled_state<F>(
    model: &ResidentModel,
    pool: &Arc<ResidentPool>,
    zero: bool,
    cancelled: Option<&AtomicBool>,
    retain: F,
) -> Result<Vec<Domain>>
where
    F: FnOnce(&mut crate::resident_leases::ResidentInitialization<'_>) -> Result<()>,
{
    check_cancel(cancelled)?;
    model.service.ensure_usable()?;
    // Nonblocking checkout occurs BEFORE any fair slot/workspace is acquired.
    let leases = pool.checkout()?;
    let mut next = leases.iter();
    let mut domains = Vec::with_capacity(model.states.len());
    for requirement in &model.states {
        domains.push(make_domain(requirement, &mut |elements| {
            let lease = next
                .next()
                .ok_or_else(|| Error::State("resident root pool plane missing".into()))?;
            if elements.checked_mul(4) != Some(lease.bytes())
                || lease.role() != BufferRole::RecurrentState
                || lease.is_initialized()
                || !Arc::ptr_eq(lease.service(), &model.service)
            {
                return Err(Error::State(
                    "resident root pool exact current plane".into(),
                ));
            }
            Ok(lease.clone())
        })?);
    }
    if next.next().is_some() {
        return Err(Error::State("resident root pool extra plane".into()));
    }
    model.validate_domains(&domains, false)?;
    if zero {
        let _allocation = model.allocation.lock().await;
        let ticket = model.service.queue().submit(
            model.service.budget(),
            model.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::RecurrentState,
                bytes: model.state_bytes as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        check_cancel(cancelled)?;
        slot.begin_submission()?;
        let mut init = model.service.begin_initialization(0)?;
        retain(&mut init)?;
        for lease in &leases {
            init.upload_zeroed_into(lease)?;
        }
        drop(init.complete_sync().await?);
        slot.finish()?;
    }
    check_cancel(cancelled)?;
    model.validate_domains(&domains, zero)?;
    Ok(domains)
}

impl ResidentModel {
    #[cfg(any(
        feature = "cuda-native",
        feature = "gemma-bf16-qualification",
        feature = "qwen-source-qualification"
    ))]
    pub(crate) fn root_pool_layout(&self) -> Result<Vec<ResidentPlaneLayout>> {
        let alignment = usize::try_from(self.service.client().properties().memory.alignment)
            .map_err(|_| Error::Limit("resident root allocator alignment".into()))?;
        let mut layout = Vec::new();
        let mut total = 0usize;
        for requirement in &self.states {
            let (count, elements) = requirement.layout.planes()?;
            let bytes = elements
                .checked_mul(4)
                .filter(|n| *n != 0)
                .ok_or_else(|| Error::Limit("resident root plane extent".into()))?;
            total = count
                .checked_mul(bytes)
                .and_then(|n| total.checked_add(n))
                .ok_or_else(|| Error::Limit("resident root layout sum".into()))?;
            layout.extend((0..count).map(|_| ResidentPlaneLayout {
                bytes,
                role: BufferRole::RecurrentState,
                alignment,
            }));
        }
        if total != self.state_bytes {
            return Err(Error::State(
                "resident root source layout payload differs".into(),
            ));
        }
        Ok(layout)
    }
    #[cfg(all(
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification"
        ),
        target_os = "linux"
    ))]
    pub(crate) fn install_window_root_pool(
        &self,
        source: zerogravity_artifacts::source_weights::VerifiedTensorSource,
        roots: usize,
    ) -> Result<()> {
        if source.source_identity() != self.service.source_identity()
            || self.state_pool.get().is_some()
        {
            return Err(Error::State(
                "resident root pool source/reinitialization".into(),
            ));
        }
        let layout = self.root_pool_layout()?;
        let metadata = source
            .reserve_workspace(ResidentPool::metadata_bytes(layout.len(), roots)?)?
            .into_host_workspace()?;
        let pool = ResidentPool::prepare(self.service.clone(), metadata, &layout, roots)?;
        self.state_pool
            .set(PooledStates {
                pool,
                source: Some(source),
            })
            .map_err(|_| Error::State("resident root pool rebound".into()))
    }
}
