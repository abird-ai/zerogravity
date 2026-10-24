//! Exact whole-allocation reuse; original leases/events still own retirement.
// Current normal constructors live behind the finite native consumer feature.
// Keep shared ownership available to other providers and source formats.
#![cfg_attr(not(feature = "vllm-cuda-qualification"), allow(dead_code))]
use super::*;
use std::sync::Weak;
use zerogravity_core::device_events::DeviceBackingId;
use zerogravity_core::resource_arbiter::HostWorkspace;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResidentPlaneLayout {
    pub(crate) bytes: usize,
    pub(crate) role: BufferRole,
    pub(crate) alignment: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResidentViewLayout {
    pub(crate) plane: usize,
    pub(crate) offset: usize,
    pub(crate) bytes: usize,
    pub(crate) role: BufferRole,
    pub(crate) alignment: usize,
}

fn bytes_total(layout: &[ResidentPlaneLayout]) -> Result<usize> {
    if layout.is_empty() {
        return Err(Error::Limit("resident backing empty layout".into()));
    }
    layout.iter().try_fold(0usize, |sum, p| {
        if p.bytes == 0 || !p.alignment.is_power_of_two() {
            return Err(Error::Limit("resident backing extent/alignment".into()));
        }
        sum.checked_add(p.bytes)
            .ok_or_else(|| Error::Limit("resident backing layout overflow".into()))
    })
}
fn metadata_bytes(planes: usize, sets: usize, views: usize) -> Result<usize> {
    planes
        .checked_mul(sets)
        .and_then(|n| n.checked_add(views))
        .and_then(|n| n.checked_mul(1024))
        .and_then(|n| sets.checked_mul(1024).and_then(|s| n.checked_add(s)))
        .and_then(|n| n.checked_add(64 * 1024))
        .filter(|&n| n <= 32 * 1024 * 1024)
        .ok_or_else(|| Error::Limit("resident backing metadata bound".into()))
}
fn require_layout(service: &ResidentService, layout: &[ResidentPlaneLayout]) -> Result<usize> {
    service.ensure_usable()?;
    let bytes = bytes_total(layout)?;
    let memory = &service.client.properties().memory;
    for p in layout {
        if p.bytes as u64 > memory.max_page_size
            || memory.alignment == 0
            || !(p.bytes as u64).is_multiple_of(memory.alignment)
            || !memory.alignment.is_multiple_of(p.alignment as u64)
        {
            return Err(Error::Unsupported(
                "resident backing page/alignment capability".into(),
            ));
        }
    }
    if bytes > service.cap {
        return Err(Error::Limit(
            "resident backing exceeds provider envelope".into(),
        ));
    }
    Ok(bytes)
}
fn require_metadata(
    service: &ResidentService,
    metadata: HostWorkspace,
    bytes: usize,
) -> Result<Arc<HostWorkspace>> {
    if metadata.source_identity() != service.source_identity() || metadata.bytes() != bytes {
        return Err(Error::State(
            "resident backing exact host metadata charge/source differs".into(),
        ));
    }
    Ok(Arc::new(metadata))
}

/// One original managed Handle and one physical ledger charge, even when idle.
pub(super) struct BackingPlane {
    handle: Option<Handle>,
    id: DeviceBackingId,
    layout: ResidentPlaneLayout,
    _metadata: Arc<HostWorkspace>,
    service: Arc<ResidentService>,
}
struct PendingBacking {
    service: Arc<ResidentService>,
    id: DeviceBackingId,
    armed: bool,
}
impl Drop for PendingBacking {
    fn drop(&mut self) {
        if self.armed
            && self
                .service
                .ledger
                .lock()
                .ok()
                .is_none_or(|mut l| l.release_backing(self.id).is_err())
        {
            self.service.fence_and_retain(&[]);
        }
    }
}
impl BackingPlane {
    fn allocate(
        service: &Arc<ResidentService>,
        layout: ResidentPlaneLayout,
        metadata: &Arc<HostWorkspace>,
    ) -> Result<Arc<Self>> {
        service.ensure_usable()?;
        let id = service
            .ledger
            .lock()
            .map_err(|_| Error::State("resident backing ledger poisoned".into()))?
            .admit_backing_for(service.generation, layout.bytes, layout.role)?;
        let mut pending = PendingBacking {
            service: service.clone(),
            id,
            armed: true,
        };
        let handle = service.client.empty(layout.bytes);
        // Original allocator views may have nonzero start/tail and shared pages.
        let used = handle
            .size()
            .checked_sub(handle.offset_start.unwrap_or(0))
            .and_then(|n| n.checked_sub(handle.offset_end.unwrap_or(0)));
        if used != Some(layout.bytes as u64) {
            return Err(Error::State(
                "resident backing original exact allocator extent".into(),
            ));
        }
        let result = Arc::new(Self {
            handle: Some(handle),
            id,
            layout,
            _metadata: metadata.clone(),
            service: service.clone(),
        });
        pending.armed = false;
        Ok(result)
    }
    fn view(
        self: &Arc<Self>,
        view: ResidentViewLayout,
        pool_use: Option<Arc<PoolUse>>,
        workspace: Option<WorkspaceMembership>,
    ) -> Result<ResidentLease> {
        self.service.ensure_usable()?;
        require_view(self.layout, view)?;
        let allocation_generation = issue(&self.service.next_allocation)?;
        let original = self.handle.as_ref().expect("live backing handle");
        let mut handle = original.clone();
        handle.offset_start = Some(
            original
                .offset_start
                .unwrap_or(0)
                .checked_add(view.offset as u64)
                .ok_or_else(|| Error::Limit("resident backing start overflow".into()))?,
        );
        let tail = self.layout.bytes - view.offset - view.bytes;
        handle.offset_end = Some(
            original
                .offset_end
                .unwrap_or(0)
                .checked_add(tail as u64)
                .ok_or_else(|| Error::Limit("resident backing tail overflow".into()))?,
        );
        if handle.size_in_used() != view.bytes as u64 {
            return Err(Error::State("resident backing exact view extent".into()));
        }
        let id = self
            .service
            .ledger
            .lock()
            .map_err(|_| Error::State("resident backing ledger poisoned".into()))?
            .admit_backing_view(self.id, view.offset, view.bytes, view.role)?;
        let mut pending = PendingAdmission {
            service: self.service.clone(),
            id,
            armed: true,
        };
        let lease = ResidentLease(Arc::new(LeaseInner {
            handle: Some(handle),
            _backing: Some(self.clone()),
            pool_use,
            workspace,
            id,
            bytes: view.bytes,
            role: view.role,
            allocation_generation,
            initialized: AtomicBool::new(false),
            workspace_written_epoch: AtomicU64::new(0),
            quarantined: AtomicBool::new(false),
            service: self.service.clone(),
        }));
        pending.armed = false;
        Ok(lease)
    }
}
impl Drop for BackingPlane {
    fn drop(&mut self) {
        let released = self.service.ensure_usable().is_ok()
            && self.service.ledger.lock().ok().is_some_and(|mut ledger| {
                if ledger.require_idle_backing(self.id).is_err() {
                    return false;
                }
                // Keep local admission locked until the real allocator owner drops.
                drop(self.handle.take());
                ledger.release_backing(self.id).is_ok()
            });
        if !released {
            self.service.fence_and_retain(&[]);
            if let Some(handle) = self.handle.take() {
                crate::physical_quarantine::quarantine_uncertain(
                    self.service.client().clone(),
                    vec![handle],
                );
            }
        }
    }
}
fn require_view(plane: ResidentPlaneLayout, view: ResidentViewLayout) -> Result<()> {
    if view.bytes == 0
        || view.role != plane.role
        || !view.alignment.is_power_of_two()
        || !plane.alignment.is_multiple_of(view.alignment)
        || !view.offset.is_multiple_of(view.alignment)
        || view
            .offset
            .checked_add(view.bytes)
            .is_none_or(|end| end > plane.bytes)
    {
        return Err(Error::State(
            "resident backing exact view role/range/alignment".into(),
        ));
    }
    Ok(())
}

type PlaneSet = Vec<Arc<BackingPlane>>;
struct PoolState {
    created: usize,
    idle: Vec<PlaneSet>,
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ResidentPoolSnapshot {
    pub(crate) allocated_sets: usize,
    pub(crate) idle_sets: usize,
    pub(crate) backing_bytes: usize,
}
/// Source-neutral bounded cohort; checkout never waits for another use to retire.
pub(crate) struct ResidentPool {
    state: Mutex<PoolState>,
    layout: Box<[ResidentPlaneLayout]>,
    max_sets: usize,
    bytes_per_set: usize,
    metadata: Arc<HostWorkspace>,
    service: Arc<ResidentService>,
}
pub(super) struct PoolUse {
    planes: Option<PlaneSet>,
    reusable: AtomicBool,
    pool: Arc<ResidentPool>,
}
impl PoolUse {
    pub(super) fn poison(&self) {
        self.reusable.store(false, Ordering::Release);
    }
}
impl Drop for PoolUse {
    fn drop(&mut self) {
        let Some(planes) = self.planes.take() else {
            return;
        };
        if self.reusable.load(Ordering::Acquire) && self.pool.service.ensure_usable().is_ok() {
            let idle = self.pool.service.ledger.lock().ok().is_some_and(|ledger| {
                planes
                    .iter()
                    .all(|p| ledger.require_idle_backing(p.id).is_ok())
            });
            if idle {
                match self.pool.state.lock() {
                    Ok(mut state) => {
                        state.idle.push(planes);
                        return;
                    }
                    Err(_) => self.pool.service.fence_and_retain(&[]),
                }
            } else {
                self.pool.service.fence_and_retain(&[]);
            }
        }
        // Uncertain backing drops through the original quarantine path.
        drop(planes);
    }
}
impl ResidentPool {
    pub(crate) fn metadata_bytes(planes: usize, max_sets: usize) -> Result<usize> {
        metadata_bytes(planes, max_sets, 0)
    }
    pub(crate) fn prepare(
        service: Arc<ResidentService>,
        metadata: HostWorkspace,
        layout: &[ResidentPlaneLayout],
        max_sets: usize,
    ) -> Result<Arc<Self>> {
        let bytes_per_set = require_layout(&service, layout)?;
        if max_sets == 0
            || bytes_per_set
                .checked_mul(max_sets)
                .is_none_or(|n| n > service.cap)
        {
            return Err(Error::Limit(
                "resident pool complete layout/max sets exceeds envelope".into(),
            ));
        }
        let metadata = require_metadata(
            &service,
            metadata,
            Self::metadata_bytes(layout.len(), max_sets)?,
        )?;
        Ok(Arc::new(Self {
            state: Mutex::new(PoolState {
                created: 0,
                idle: Vec::with_capacity(max_sets),
            }),
            layout: layout.into(),
            max_sets,
            bytes_per_set,
            metadata,
            service,
        }))
    }
    /// Snapshot/Pending/Ready/flight clones all retain the same checked-out use.
    /// No initialization flag or old logical identity is transferred on reuse.
    pub(crate) fn checkout(self: &Arc<Self>) -> Result<Vec<ResidentLease>> {
        self.service.ensure_usable()?;
        let planes = {
            let mut state = match self.state.try_lock() {
                Ok(state) => state,
                Err(std::sync::TryLockError::WouldBlock) => {
                    return Err(Error::Limit("resident pool busy".into()));
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    self.service.fence_and_retain(&[]);
                    return Err(Error::State("resident pool poisoned".into()));
                }
            };
            if let Some(planes) = state.idle.pop() {
                planes
            } else {
                if state.created == self.max_sets {
                    return Err(Error::Limit("resident pool exhausted".into()));
                }
                let mut planes = Vec::with_capacity(self.layout.len());
                for &layout in &self.layout {
                    planes.push(BackingPlane::allocate(
                        &self.service,
                        layout,
                        &self.metadata,
                    )?);
                }
                state.created += 1;
                planes
            }
        };
        let usage = Arc::new(PoolUse {
            planes: Some(planes),
            reusable: AtomicBool::new(true),
            pool: self.clone(),
        });
        let mut leases = Vec::with_capacity(self.layout.len());
        for (plane, &layout) in usage
            .planes
            .as_ref()
            .expect("live pool use")
            .iter()
            .zip(&self.layout)
        {
            leases.push(plane.view(
                ResidentViewLayout {
                    plane: 0,
                    offset: 0,
                    bytes: layout.bytes,
                    role: layout.role,
                    alignment: layout.alignment,
                },
                Some(usage.clone()),
                None,
            )?);
        }
        Ok(leases)
    }
    pub(crate) fn snapshot(&self) -> Result<ResidentPoolSnapshot> {
        let state = self
            .state
            .lock()
            .map_err(|_| Error::State("resident pool poisoned".into()))?;
        Ok(ResidentPoolSnapshot {
            allocated_sets: state.created,
            idle_sets: state.idle.len(),
            backing_bytes: state.created * self.bytes_per_set,
        })
    }
}

struct WorkspaceGate {
    active: AtomicU64,
    next_geometry: AtomicU64,
    next_run: AtomicU64,
    run: Mutex<Weak<ResidentWorkspaceRun>>,
    remaining_views: std::sync::atomic::AtomicUsize,
}
impl WorkspaceGate {
    fn acquire(&self, geometry: u64) -> Result<u64> {
        let epoch = issue(&self.next_run)?;
        self.active
            .compare_exchange(0, geometry, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::Limit("resident workspace is already active".into()))?;
        Ok(epoch)
    }
    fn release(&self, geometry: u64) -> Result<()> {
        self.active
            .compare_exchange(geometry, 0, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("resident workspace activation changed".into()))?;
        Ok(())
    }
}
#[derive(Clone)]
pub(super) struct WorkspaceMembership {
    gate: Arc<WorkspaceGate>,
    geometry: u64,
    // A cloned lease may escape its geometry wrapper; keep view metadata charged.
    _admission: Arc<WorkspaceViewAdmission>,
}
impl WorkspaceMembership {
    pub(super) fn require_active(&self) -> Result<()> {
        if self.gate.active.load(Ordering::Acquire) != self.geometry {
            return Err(Error::State(
                "resident workspace geometry is inactive".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn key(&self) -> usize {
        Arc::as_ptr(&self.gate) as usize
    }
    pub(super) fn require_run(&self, run: &ResidentWorkspaceRun) -> Result<()> {
        if run.membership.geometry != self.geometry
            || !Arc::ptr_eq(&run.membership.gate, &self.gate)
        {
            return Err(Error::State(
                "resident workspace active owner differs".into(),
            ));
        }
        self.require_active()
    }
    pub(super) fn active_run(&self) -> Result<Arc<ResidentWorkspaceRun>> {
        self.require_active()?;
        let run = self
            .gate
            .run
            .lock()
            .map_err(|_| Error::State("resident workspace run poisoned".into()))?
            .upgrade()
            .ok_or_else(|| Error::State("resident workspace active owner absent".into()))?;
        self.require_run(&run)?;
        Ok(run)
    }
}
/// Retain in the SAME original armed flight before dispatch; unknown retains it.
pub(crate) struct ResidentWorkspaceRun {
    membership: WorkspaceMembership,
    _geometry: Arc<ResidentWorkspaceGeometry>,
    epoch: u64,
}
impl ResidentWorkspaceRun {
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }
}
impl Drop for ResidentWorkspaceRun {
    fn drop(&mut self) {
        if self
            .membership
            .gate
            .release(self.membership.geometry)
            .is_err()
        {
            self._geometry.service.fence_and_retain(&[]);
        }
    }
}
pub(crate) struct ResidentWorkspaceGeometry {
    // Views drop before their metadata/gate and original service owner.
    leases: Vec<ResidentLease>,
    membership: WorkspaceMembership,
    _metadata: Arc<HostWorkspace>,
    service: Arc<ResidentService>,
}
struct WorkspaceViewAdmission {
    gate: Arc<WorkspaceGate>,
    views: usize,
}
impl WorkspaceViewAdmission {
    fn reserve(gate: &Arc<WorkspaceGate>, views: usize) -> Result<Arc<Self>> {
        if views == 0 {
            return Err(Error::Limit("resident workspace empty views".into()));
        }
        gate.remaining_views
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_sub(views)
            })
            .map_err(|_| Error::Limit("resident workspace total retained views exceeded".into()))?;
        Ok(Arc::new(Self {
            gate: gate.clone(),
            views,
        }))
    }
}
impl Drop for WorkspaceViewAdmission {
    fn drop(&mut self) {
        self.gate
            .remaining_views
            .fetch_add(self.views, Ordering::AcqRel);
    }
}
impl ResidentWorkspaceGeometry {
    pub(crate) fn leases(&self) -> &[ResidentLease] {
        &self.leases
    }
    pub(crate) fn try_activate(self: &Arc<Self>) -> Result<Arc<ResidentWorkspaceRun>> {
        self.service.ensure_usable()?;
        let epoch = self.membership.gate.acquire(self.membership.geometry)?;
        let run = Arc::new(ResidentWorkspaceRun {
            membership: self.membership.clone(),
            _geometry: self.clone(),
            epoch,
        });
        *self
            .membership
            .gate
            .run
            .lock()
            .map_err(|_| Error::State("resident workspace run poisoned".into()))? =
            Arc::downgrade(&run);
        Ok(run)
    }
}
/// Charged cells with fixed genuine exact views. This does NOT authorize cropped
/// native packets: each lowerer must support these actual original Handle ranges.
pub(crate) struct ResidentWorkspace {
    planes: PlaneSet,
    gate: Arc<WorkspaceGate>,
    metadata: Arc<HostWorkspace>,
    service: Arc<ResidentService>,
}
impl ResidentWorkspace {
    pub(crate) fn metadata_bytes(planes: usize, max_views: usize) -> Result<usize> {
        metadata_bytes(planes, 1, max_views)
    }
    pub(crate) fn prepare(
        service: Arc<ResidentService>,
        metadata: HostWorkspace,
        layout: &[ResidentPlaneLayout],
        max_views: usize,
    ) -> Result<Arc<Self>> {
        require_layout(&service, layout)?;
        if max_views == 0 {
            return Err(Error::Limit(
                "resident workspace empty view capacity".into(),
            ));
        }
        if layout.iter().any(|p| {
            matches!(
                p.role,
                BufferRole::ImmutableWeights | BufferRole::RecurrentState
            )
        }) {
            return Err(Error::State(
                "resident workspace excludes immutable/escaped state".into(),
            ));
        }
        let metadata = require_metadata(
            &service,
            metadata,
            Self::metadata_bytes(layout.len(), max_views)?,
        )?;
        let mut planes = Vec::with_capacity(layout.len());
        for &p in layout {
            planes.push(BackingPlane::allocate(&service, p, &metadata)?);
        }
        Ok(Arc::new(Self {
            planes,
            gate: Arc::new(WorkspaceGate {
                active: AtomicU64::new(0),
                next_geometry: AtomicU64::new(1),
                next_run: AtomicU64::new(1),
                run: Mutex::new(Weak::new()),
                remaining_views: std::sync::atomic::AtomicUsize::new(max_views),
            }),
            metadata,
            service,
        }))
    }
    pub(crate) fn geometry(
        self: &Arc<Self>,
        layout: &[ResidentViewLayout],
    ) -> Result<Arc<ResidentWorkspaceGeometry>> {
        self.service.ensure_usable()?;
        if layout.is_empty() {
            return Err(Error::Limit("resident workspace empty geometry".into()));
        }
        // Bound persistent view metadata before allocation; final capacity below
        // is reserved at prepare and accounted independently from physical bytes.
        let needed = metadata_bytes(self.planes.len(), 1, layout.len())?;
        if needed > self.metadata.bytes() {
            return Err(Error::Limit(
                "resident workspace geometry metadata exceeded".into(),
            ));
        }
        let admission = WorkspaceViewAdmission::reserve(&self.gate, layout.len())?;
        for (i, &view) in layout.iter().enumerate() {
            let plane = self
                .planes
                .get(view.plane)
                .ok_or_else(|| Error::State("resident workspace foreign plane".into()))?;
            require_view(plane.layout, view)?;
            for previous in &layout[..i] {
                if previous.plane == view.plane
                    && previous.offset < view.offset + view.bytes
                    && view.offset < previous.offset + previous.bytes
                {
                    return Err(Error::State(
                        "resident workspace geometry overlaps itself".into(),
                    ));
                }
            }
        }
        let membership = WorkspaceMembership {
            gate: self.gate.clone(),
            geometry: issue(&self.gate.next_geometry)?,
            _admission: admission,
        };
        let mut leases = Vec::with_capacity(layout.len());
        for &view in layout {
            leases.push(self.planes[view.plane].view(view, None, Some(membership.clone()))?);
        }
        Ok(Arc::new(ResidentWorkspaceGeometry {
            leases,
            membership,
            _metadata: self.metadata.clone(),
            service: self.service.clone(),
        }))
    }
}

#[cfg(test)]
mod tests;

impl ResidentLease {
    /// Actual enqueue-time bridge; inactive cold views cannot mint a binding.
    /// Old whole allocations keep the original canonical native handle contract.
    #[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
    pub(crate) fn cubin_binding(&self) -> Result<cubecl::cuda::OwnedCudaBinding> {
        let service = self.service();
        service.ensure_usable()?;
        service.check_lease(self)?;
        let Some(workspace) = &self.0.workspace else {
            return Ok(cubecl::cuda::OwnedCudaBinding::whole(self.handle()));
        };
        let backing = self
            .0
            ._backing
            .as_ref()
            .ok_or_else(|| Error::State("resident workspace original backing absent".into()))?;
        let run = workspace.active_run()?;
        // SAFETY: private exact view constructor retains the actual original
        // allocator owner. The selected geometry is active on the same service,
        // and original submission automatically retains this SAME run owner.
        unsafe {
            service.pinned_cuda_queue()?.admit_workspace_binding(
                backing
                    .handle
                    .as_ref()
                    .expect("live backing handle")
                    .clone(),
                self.handle(),
                cubecl::cuda::CubinWorkspaceView {
                    backing_bytes: backing.layout.bytes as u64,
                    bytes: self.bytes() as u64,
                    alignment: backing.layout.alignment as u64,
                },
                run,
            )
        }
        .map_err(|e| Error::State(format!("resident native workspace binding: {e}")))
    }
}
