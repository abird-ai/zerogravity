//! Cold isolated B2/B4/B8 CUDA graph owner/replay substrate.
//! No Ready/root permit or production selection lives here; fused preparation
//! still owns publication, and external capture/replay entrypoints remain
//! qualification-only.
#![cfg_attr(
    not(feature = "qualification"),
    allow(
        dead_code,
        reason = "production compiles the graph ownership substrate before any serving selector is admitted"
    )
)]
use super::*;
use crate::physical_quarantine::CompletionOwner;
use crate::resident_leases::ForeignProviderCharge;
use crate::resident_program::{BatchBodyReceipt, BatchGraphProfile, BatchGraphReplay};
use cubecl::client::Graph;
use std::sync::{
    Mutex as HostMutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
#[cfg(feature = "qualification")]
pub(super) mod qualification;
#[cfg(feature = "qualification")]
pub(super) mod wider;

const GRAPH_CONTROL_RESERVE_BYTES: usize = 64 * 1024;
const GRAPH_DISPATCH_RESERVE_BYTES: usize = 4 * 1024;
const GRAPH_DISPATCH_BOUND: usize = 65_536;

fn graph_policy_bytes(dispatches: usize, access_capacity: usize) -> Result<usize> {
    if dispatches == 0 || dispatches > GRAPH_DISPATCH_BOUND || access_capacity > 65_536 {
        return Err(Error::Limit("graph resource admission geometry".into()));
    }
    let metadata = std::mem::size_of::<CapturedResidentGraph>()
        .checked_add(std::mem::size_of::<GraphCaptureReceipt>())
        .and_then(|n| {
            access_capacity
                .checked_mul(std::mem::size_of::<(ResidentLease, DeviceAccess)>())
                .and_then(|accesses| n.checked_add(accesses))
        })
        .ok_or_else(|| Error::Limit("graph owned metadata byte overflow".into()))?;
    let dispatch_bytes = dispatches
        .checked_mul(GRAPH_DISPATCH_RESERVE_BYTES)
        .ok_or_else(|| Error::Limit("graph dispatch reserve overflow".into()))?;
    GRAPH_CONTROL_RESERVE_BYTES
        .checked_add(dispatch_bytes)
        .and_then(|n| n.checked_add(metadata))
        .ok_or_else(|| Error::Limit("graph resource reserve overflow".into()))
}

#[derive(Clone)]
struct GraphResourceAdmission {
    charges: Vec<Arc<ForeignProviderCharge>>,
    dispatch_slots: usize,
    bytes: usize,
}
impl GraphResourceAdmission {
    fn initial(
        service: &Arc<ResidentService>,
        dispatches: usize,
        access_capacity: usize,
    ) -> Result<Self> {
        let bytes = graph_policy_bytes(dispatches, access_capacity)?;
        let charge = service.reserve_foreign_provider(bytes)?;
        charge.require_service(service)?;
        if charge.bytes() != bytes {
            return Err(Error::State("graph initial resource charge differs".into()));
        }
        Ok(Self {
            charges: vec![charge],
            dispatch_slots: dispatches,
            bytes,
        })
    }

    fn ensure_dispatches(
        &mut self,
        service: &Arc<ResidentService>,
        dispatches: usize,
    ) -> Result<()> {
        if dispatches == 0 || dispatches > GRAPH_DISPATCH_BOUND {
            return Err(Error::Limit("graph dispatch reserve bound".into()));
        }
        if dispatches <= self.dispatch_slots {
            return Ok(());
        }
        let extra = dispatches
            .checked_sub(self.dispatch_slots)
            .and_then(|n| n.checked_mul(GRAPH_DISPATCH_RESERVE_BYTES))
            .ok_or_else(|| Error::Limit("graph additional dispatch reserve overflow".into()))?;
        let charge = service.reserve_foreign_provider(extra)?;
        charge.require_service(service)?;
        if charge.bytes() != extra {
            return Err(Error::State(
                "graph additional resource charge differs".into(),
            ));
        }
        self.charges.push(charge);
        self.dispatch_slots = dispatches;
        self.bytes = self
            .bytes
            .checked_add(extra)
            .ok_or_else(|| Error::Limit("graph total reserve overflow".into()))?;
        Ok(())
    }

    fn require_service(&self, service: &Arc<ResidentService>) -> Result<()> {
        if self.bytes == 0 || self.charges.is_empty() {
            return Err(Error::State("graph resource admission absent".into()));
        }
        for charge in &self.charges {
            charge.require_service(service)?;
        }
        Ok(())
    }

    fn bytes(&self) -> usize {
        self.bytes
    }
}

/// A source launch-site count, not a driver graph-node enumeration or timing.
#[derive(Clone, Copy, Debug)]
pub struct GraphCaptureReceipt {
    pub warm_dispatches: [usize; 2],
    pub captured_kernels: usize,
    pub projection_dots: usize,
    pub body_nodes: usize,
    pub retained_lease_bytes: usize,
    pub owned_lease_vector_bytes: usize,
    /// Conservative policy reserve, not measured CUDA driver memory.
    pub resource_charge_bytes: usize,
}

pub struct CapturedResidentGraph {
    service: Arc<ResidentService>,
    profile: BatchGraphProfile,
    stream: u64,
    body: BatchBodyReceipt,
    graph: HostMutex<Option<Graph>>,
    /// Exact captured internal storage and its event authority. Replays must
    /// prove these same allocations/accesses are retained by the current armed
    /// ResidentSubmission before the backend graph may enqueue.
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    status: ResidentLease,
    resource: GraphResourceAdmission,
    _band_pin: Arc<()>,
    retired: AtomicBool,
    replay_enqueues: AtomicUsize,
    receipt: GraphCaptureReceipt,
}

/// No Drop: process quarantine must never invoke driver destruction/sync.
struct QuarantinedGraph {
    _graph: Option<Graph>,
    _leases: Vec<ResidentLease>,
    _service: Arc<ResidentService>,
    _resource: GraphResourceAdmission,
    _band_pin: Arc<()>,
}
impl Drop for CapturedResidentGraph {
    fn drop(&mut self) {
        if !self.retired.load(Ordering::Acquire) {
            let graph = self
                .graph
                .get_mut()
                .unwrap_or_else(|p| p.into_inner())
                .take();
            let leases = self
                .accesses
                .iter()
                .map(|(lease, _)| lease.clone())
                .collect::<Vec<_>>();
            let owner: CompletionOwner = Arc::new(QuarantinedGraph {
                _graph: graph,
                _leases: leases.clone(),
                _service: Arc::clone(&self.service),
                _resource: self.resource.clone(),
                _band_pin: Arc::clone(&self._band_pin),
            });
            self.service.fence_and_retain_owners(&leases, vec![owner]);
        }
    }
}

impl CapturedResidentGraph {
    pub fn output(&self) -> ResidentOutput {
        self.profile.output
    }

    pub fn validate(
        &self,
        model: &ResidentModel,
        count: usize,
        positions: &[usize],
        readout: Readout,
    ) -> Result<()> {
        self.service.ensure_usable()?;
        self.resource.require_service(&self.service)?;
        let queue = self.service.pinned_cuda_queue()?;
        if !Arc::ptr_eq(&self.service, &model.service)
            || self.retired.load(Ordering::Acquire)
            || self.service.client().service_id() != queue.service_id()
            || self.stream != queue.logical_stream_id().value
            || !matches!(
                (self.profile.output, readout),
                (ResidentOutput::StateOnly, Readout::None)
                    | (ResidentOutput::Logits, Readout::Logits | Readout::Greedy)
            )
            || count != self.profile.rows
            || positions.len() != count
            || positions.iter().any(|&p| p >= 256)
        {
            return Err(Error::State(
                "foreign/retired/incompatible resident graph".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn validate_program(
        &self,
        program: &BatchProgram,
        parameters: &[RuntimeParameters],
    ) -> Result<()> {
        if program.graph_profile(parameters, self.profile.output)? != self.profile
            || !program.graph_pinned()
            || self
                .graph
                .lock()
                .map_err(|_| Error::State("resident graph lock poisoned".into()))?
                .is_none()
        {
            return Err(Error::State("stale resident graph workspace key".into()));
        }
        Ok(())
    }

    /// A graph replay is authorized only by the same armed resident event that
    /// retains every exact internal allocation captured by the graph, with at
    /// least the access authority recorded at capture. This prevents a valid
    /// executable from being paired with replacement storage or a sibling
    /// physical event merely because the batch geometry still matches.
    pub(super) fn require_flight(
        &self,
        flight: &crate::resident_leases::ResidentSubmission<'_>,
    ) -> Result<()> {
        self.service.ensure_usable()?;
        self.resource.require_service(&self.service)?;
        if self.accesses.is_empty() {
            return Err(Error::State(
                "resident graph captured no storage authority".into(),
            ));
        }
        flight.require_accesses(self.accesses.iter().map(|(lease, access)| (lease, *access)))
    }

    pub(super) fn launch(
        &self,
        program: &mut BatchProgram,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
    ) -> Result<crate::resident_program::ResidentSubmission> {
        let graph = self
            .graph
            .lock()
            .map_err(|_| Error::State("resident graph lock poisoned".into()))?;
        let graph = graph
            .as_ref()
            .ok_or_else(|| Error::State("resident graph retired".into()))?;
        program.launch_graph(
            self.service.client(),
            parameters,
            bindings,
            BatchGraphReplay {
                profile: self.profile,
                graph,
                body: self.body,
                accepted_enqueues: &self.replay_enqueues,
            },
        )
    }
    pub fn receipt(&self) -> GraphCaptureReceipt {
        self.receipt
    }
    pub fn replay_enqueues(&self) -> usize {
        self.replay_enqueues.load(Ordering::Acquire)
    }

    /// Consumes the caller's sole graph handle. Both terminal barriers use the
    /// original physical queue and initialized Read leases. No graph destroy
    /// is posted until all prior replays have a trusted terminal readback.
    pub async fn retire(self: Arc<Self>, root: u64) -> Result<()> {
        if Arc::strong_count(&self) != 1 || self.retired.load(Ordering::Acquire) {
            return Err(Error::State(
                "resident graph retirement requires exclusive live owner".into(),
            ));
        }
        self.terminal_barrier(root, false).await?;
        self.terminal_barrier(root, true).await?;
        self.retired.store(true, Ordering::Release);
        Ok(())
    }

    async fn terminal_barrier(self: &Arc<Self>, root: u64, destroy: bool) -> Result<()> {
        self.resource.require_service(&self.service)?;
        let ticket = self.service.queue().submit(
            self.service.budget(),
            self.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: self.resource.bytes() as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        self.service.ensure_usable()?;
        let accesses = self
            .accesses
            .iter()
            .map(|(lease, _)| (lease.clone(), DeviceAccess::Read))
            .collect::<Vec<_>>();
        slot.begin_submission()?;
        let mut flight = self.service.submit(root, &accesses)?;
        let owner: CompletionOwner = self.clone();
        flight.retain_completion_owner(owner)?;
        if destroy {
            // Barrier one completed before this method was entered. Preserve
            // leases, band pin and flight until actor graph destruction retires.
            let graph = self
                .graph
                .lock()
                .map_err(|_| Error::State("resident graph lock poisoned".into()))?
                .take()
                .ok_or_else(|| Error::State("resident graph already destroyed".into()))?;
            drop(graph);
        }
        flight
            .complete_read_only_readback(&[self.status.readback(0, 4)?], |bytes| {
                if bytes.len() != 1 || !all_zero(&bytes[0]) {
                    return Err(Error::State("graph terminal status refused".into()));
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        Ok(())
    }
}

/// Owns only a window opened after a cold no-other-capture check. Every failure
/// closes that same pinned window; a failed window is never a direct fallback.
struct CaptureWindow {
    service: Arc<ResidentService>,
    leases: Vec<ResidentLease>,
    band_pin: Arc<()>,
    resource: Option<GraphResourceAdmission>,
    active: bool,
}
impl CaptureWindow {
    fn prepare(
        service: &Arc<ResidentService>,
        leases: Vec<ResidentLease>,
        band_pin: Arc<()>,
        resource: GraphResourceAdmission,
    ) -> Result<Self> {
        if service.client().is_capturing() {
            return Err(Error::State("capture already active".into()));
        }
        let window = Self {
            service: service.clone(),
            leases,
            band_pin,
            resource: Some(resource),
            active: true,
        };
        // Construct the owner first so a prepare failure/unwind also closes
        // the window under the caller's exclusive cold-capture contract.
        window
            .service
            .client()
            .graph_prepare()
            .map_err(|e| Error::State(format!("graph prepare: {e}")))?;
        Ok(window)
    }
    fn ensure_dispatches(&mut self, dispatches: usize) -> Result<()> {
        self.resource
            .as_mut()
            .ok_or_else(|| Error::State("graph capture resource owner absent".into()))?
            .ensure_dispatches(&self.service, dispatches)
    }

    fn take_resource(&mut self) -> Result<GraphResourceAdmission> {
        self.resource
            .take()
            .ok_or_else(|| Error::State("graph capture resource owner already moved".into()))
    }

    fn resource_bytes(&self) -> Result<usize> {
        self.resource
            .as_ref()
            .map(GraphResourceAdmission::bytes)
            .ok_or_else(|| Error::State("graph capture resource owner absent".into()))
    }

    fn covers_dispatches(&self, dispatches: usize) -> bool {
        self.resource
            .as_ref()
            .is_some_and(|resource| dispatches <= resource.dispatch_slots)
    }

    fn start(&self) -> Result<()> {
        self.service
            .client()
            .start_capture()
            .map_err(|e| Error::State(format!("graph start: {e}")))
    }
    fn stop(&mut self) -> Result<Graph> {
        self.active = false; // Vendor end_capture closes even on refusal.
        self.service
            .client()
            .stop_capture()
            .map_err(|e| Error::State(format!("graph stop: {e}")))
    }
}
impl Drop for CaptureWindow {
    fn drop(&mut self) {
        if self.active {
            let graph = self.service.client().stop_capture().ok();
            let resource = self
                .resource
                .take()
                .expect("active capture window retains resource admission");
            let owner: CompletionOwner = Arc::new(QuarantinedGraph {
                _graph: graph,
                _leases: self.leases.clone(),
                _service: self.service.clone(),
                _resource: resource,
                _band_pin: self.band_pin.clone(),
            });
            self.service
                .fence_and_retain_owners(&self.leases, vec![owner]);
        }
    }
}

impl ResidentModel {
    /// Caller MUST establish process-wide cold same-card isolation. The local
    /// allocation/queue/workspace order cannot exclude another model's Client.
    #[cfg(feature = "qualification")]
    pub async fn capture_logits_b2(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
    ) -> Result<Arc<CapturedResidentGraph>> {
        if rows.len() != 2 {
            return Err(Error::Unsupported(
                "initial graph wrapper remains B2".into(),
            ));
        }
        self.capture_body_internal(rows, ResidentOutput::Logits)
            .await
    }

    #[cfg(feature = "qualification")]
    pub async fn capture_body(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
        output: ResidentOutput,
    ) -> Result<Arc<CapturedResidentGraph>> {
        self.capture_body_internal(rows, output).await
    }

    async fn capture_body_internal(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
        output: ResidentOutput,
    ) -> Result<Arc<CapturedResidentGraph>> {
        self.service.pinned_cuda_queue()?;
        if !matches!(rows.len(), 2 | 4 | 8)
            || self.capacity != 512
            || rows.iter().any(|(s, _)| s.binding().frontier() >= 256)
        {
            return Err(Error::Unsupported(
                "graph prototype B2/B4/B8/cap512/padded256".into(),
            ));
        }
        {
            let workspace = self.workspace.lock().await;
            if workspace.program.plan().nodes().len() > 4096 {
                return Err(Error::Limit("graph original plan node bound".into()));
            }
            if workspace
                .batch
                .as_ref()
                .is_some_and(|b| b.program.graph_pinned())
                || self.service.client().is_capturing()
            {
                return Err(Error::State(
                    "one cold capture per live batch workspace".into(),
                ));
            }
        }
        let before = rows.iter().map(|(s, _)| s.binding()).collect::<Vec<_>>();
        let parameters = rows
            .iter()
            .map(|(s, token)| {
                let position = usize::try_from(s.binding().frontier())
                    .map_err(|_| Error::Limit("graph frontier".into()))?;
                Ok(RuntimeParameters {
                    token: *token,
                    position,
                    valid_tokens: position + 1,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // First warm admits the original native workspace and completes private
        // output without publishing. Second warm fills prepared info keys.
        let first = match output {
            ResidentOutput::StateOnly => self.prepare_state_fused_cohort(rows).await?,
            ResidentOutput::Logits => self.prepare_logits_fused_cohort(rows).await?,
        };
        let first_count = first.dispatches();
        drop(first);
        let (accesses, band_pin, profile) = {
            let workspace = self.workspace.lock().await;
            let batch = workspace
                .batch
                .as_ref()
                .ok_or_else(|| Error::State("graph warm batch missing".into()))?;
            if batch.program.graph_pinned() {
                return Err(Error::State("one graph per live batch workspace".into()));
            }
            let profile = batch.program.graph_profile(&parameters, output)?;
            let handles = batch.program.execution_handles(output);
            let accesses = workspace
                .leases
                .iter()
                .chain(&batch.leases)
                .filter(|(lease, _)| {
                    handles
                        .iter()
                        .any(|handle| same_handle(handle, &lease.handle()))
                })
                .cloned()
                .collect::<Vec<_>>();
            (accesses, batch.program.graph_pin(), profile)
        };
        if accesses.is_empty() || accesses.len() > 65_536 || profile.body_nodes > 4096 {
            return Err(Error::Limit("graph lease/access metadata bound".into()));
        }
        let leases = accesses
            .iter()
            .map(|(lease, _)| lease.clone())
            .collect::<Vec<_>>();
        let resource = GraphResourceAdmission::initial(
            &self.service,
            first_count.max(profile.body_nodes),
            accesses.capacity(),
        )?;
        let mut window = CaptureWindow::prepare(&self.service, leases.clone(), band_pin, resource)?;
        let second = match output {
            ResidentOutput::StateOnly => self.prepare_state_fused_cohort(rows).await?,
            ResidentOutput::Logits => self.prepare_logits_fused_cohort(rows).await?,
        };
        let second_count = second.dispatches();
        drop(second);
        window.ensure_dispatches(second_count)?;
        if rows
            .iter()
            .map(|(s, _)| s.binding())
            .ne(before.iter().copied())
        {
            return Err(Error::State("graph private warm changed root".into()));
        }
        let allocation = self.allocation.lock().await;
        let ticket = self.service.queue().submit(
            self.service.budget(),
            self.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: window.resource_bytes()? as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        let workspace = self.workspace.lock().await;
        let batch = workspace
            .batch
            .as_ref()
            .ok_or_else(|| Error::State("graph capture batch missing".into()))?;
        if batch.program.graph_profile(&parameters, output)? != profile {
            return Err(Error::State("graph profile changed after warm".into()));
        }
        let handles = batch.program.execution_handles(output);
        if handles
            .iter()
            .any(|h| !leases.iter().any(|l| same_handle(h, &l.handle())))
        {
            return Err(Error::State(
                "graph capture workspace changed after warm".into(),
            ));
        }
        let status = lease_for(
            &workspace
                .leases
                .iter()
                .chain(&batch.leases)
                .cloned()
                .collect::<Vec<_>>(),
            &batch.program.status(),
        )?
        .clone();
        let reads = leases
            .iter()
            .cloned()
            .map(|l| (l, DeviceAccess::Read))
            .collect::<Vec<_>>();
        slot.begin_submission()?;
        let mut flight = self.service.submit(before[0].root(), &reads)?;
        drop(allocation);
        // No await/host transfers between start and stop. Only the hot body
        // and original certifications record; vendor refuses device allocations.
        window.start()?;
        let body_result = batch
            .program
            .launch_body(self.service.client(), &parameters, output);
        let graph_result = window.stop();
        let graph = graph_result?;
        let body = match body_result {
            Ok(body)
                if body.dispatches > 0
                    && body.dispatches <= GRAPH_DISPATCH_BOUND
                    && body.fused_dispatches > 0
                    && window.covers_dispatches(body.dispatches) =>
            {
                body
            }
            error => {
                let resource = window.take_resource()?;
                let owner: CompletionOwner = Arc::new(QuarantinedGraph {
                    _graph: Some(graph),
                    _leases: leases.clone(),
                    _service: self.service.clone(),
                    _resource: resource,
                    _band_pin: batch.program.graph_pin(),
                });
                flight.retain_completion_owner(owner)?;
                return Err(error
                    .err()
                    .unwrap_or_else(|| Error::State("graph body dispatch/resource bound".into())));
            }
        };
        let retained_lease_bytes = accesses.iter().map(|(lease, _)| lease.bytes()).sum();
        let owned_lease_vector_bytes =
            accesses.capacity() * std::mem::size_of::<(ResidentLease, DeviceAccess)>();
        let resource = window.take_resource()?;
        let resource_charge_bytes = resource.bytes();
        let graph = Arc::new(CapturedResidentGraph {
            service: self.service.clone(),
            profile,
            stream: self.service.pinned_cuda_queue()?.logical_stream_id().value,
            body,
            graph: HostMutex::new(Some(graph)),
            accesses,
            status: status.clone(),
            resource,
            _band_pin: batch.program.graph_pin(),
            retired: AtomicBool::new(false),
            replay_enqueues: AtomicUsize::new(0),
            receipt: GraphCaptureReceipt {
                warm_dispatches: [first_count, second_count],
                captured_kernels: body.dispatches,
                projection_dots: body.fused_dispatches,
                body_nodes: profile.body_nodes,
                retained_lease_bytes,
                owned_lease_vector_bytes,
                resource_charge_bytes,
            },
        });
        let owner: CompletionOwner = graph.clone();
        flight.retain_completion_owner(owner)?;
        flight
            .complete_read_only_readback(&[status.readback(0, 4)?], |data| {
                if data.len() != 1 || !all_zero(&data[0]) {
                    return Err(Error::State("graph capture terminal status".into()));
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        Ok(graph)
    }

    #[cfg(feature = "qualification")]
    pub async fn prepare_graph_logits(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
        graph: &Arc<CapturedResidentGraph>,
    ) -> Result<PreparedResidentCohort> {
        self.prepare_graph_readout_internal(rows, graph, Readout::Logits)
            .await
    }

    #[cfg(feature = "qualification")]
    pub async fn prepare_graph_readout(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
        graph: &Arc<CapturedResidentGraph>,
        readout: Readout,
    ) -> Result<PreparedResidentCohort> {
        self.prepare_graph_readout_internal(rows, graph, readout)
            .await
    }

    async fn prepare_graph_readout_internal(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
        graph: &Arc<CapturedResidentGraph>,
        readout: Readout,
    ) -> Result<PreparedResidentCohort> {
        self.prepare_fused_execution(
            rows,
            None,
            readout,
            super::fused::FusedExecution::Graph(graph.clone()),
        )
        .await?
        .ok_or_else(|| Error::Limit("graph cohort admission exhausted".into()))
    }
}

#[cfg(test)]
mod resource_policy_tests {
    use super::*;

    #[test]
    fn graph_resource_policy_is_monotonic_and_bounded() {
        let base = graph_policy_bytes(1, 1).unwrap();
        let wider = graph_policy_bytes(8, 1).unwrap();
        let more_owners = graph_policy_bytes(8, 64).unwrap();
        assert!(base >= GRAPH_CONTROL_RESERVE_BYTES + GRAPH_DISPATCH_RESERVE_BYTES);
        assert_eq!(
            wider - base,
            7 * GRAPH_DISPATCH_RESERVE_BYTES,
            "dispatch policy is one conservative page per source launch slot"
        );
        assert_eq!(
            more_owners - wider,
            63 * std::mem::size_of::<(ResidentLease, DeviceAccess)>(),
            "owned lease/access-vector metadata is charged exactly"
        );
        assert!(graph_policy_bytes(0, 1).is_err());
        assert!(graph_policy_bytes(GRAPH_DISPATCH_BOUND + 1, 1).is_err());
        assert!(graph_policy_bytes(1, 65_537).is_err());
    }
}

#[cfg(feature = "qualification")]
impl CapturedResidentGraph {
    pub fn qualification_generation(&self) -> u64 {
        self.profile.generation
    }
    pub fn qualification_set_generation(&mut self, generation: u64) {
        self.profile.generation = generation;
    }
    pub fn qualification_rows(&self) -> usize {
        self.profile.rows
    }
    pub fn qualification_resource_witness(&self) -> (usize, usize, Vec<usize>) {
        (
            self.resource.bytes,
            self.resource.charges.len(),
            self.resource
                .charges
                .iter()
                .map(Arc::strong_count)
                .collect(),
        )
    }
}
