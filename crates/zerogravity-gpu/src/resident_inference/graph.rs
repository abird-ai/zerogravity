//! Qualifier-only, cold isolated B2/B4/B8 typed original node body.
//! No Ready/root permit lives in this module; fused preparation still owns it.
use super::*;
use crate::physical_quarantine::CompletionOwner;
use crate::resident_program::{BatchBodyReceipt, BatchGraphProfile, BatchGraphReplay};
use cubecl::client::Graph;
use std::sync::{
    Mutex as HostMutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
mod qualification;
mod wider;
pub(crate) use qualification::qualify;
#[cfg(feature = "hardware-abort-probe")]
pub(crate) use qualification::qualify_abort;
pub(crate) use wider::qualify as qualify_wide;
#[cfg(feature = "hardware-abort-probe")]
pub(crate) use wider::qualify_abort as qualify_wide_abort;

/// A source launch-site count, not a driver graph-node enumeration or timing.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GraphCaptureReceipt {
    pub warm_dispatches: [usize; 2],
    pub captured_kernels: usize,
    pub projection_dots: usize,
    pub body_nodes: usize,
    pub retained_lease_bytes: usize,
    pub owned_lease_vector_bytes: usize,
}

pub(super) struct CapturedResidentGraph {
    service: Arc<ResidentService>,
    profile: BatchGraphProfile,
    stream: u64,
    body: BatchBodyReceipt,
    graph: HostMutex<Option<Graph>>,
    leases: Vec<ResidentLease>,
    status: ResidentLease,
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
            let owner: CompletionOwner = Arc::new(QuarantinedGraph {
                _graph: graph,
                _leases: self.leases.clone(),
                _service: Arc::clone(&self.service),
                _band_pin: Arc::clone(&self._band_pin),
            });
            self.service
                .fence_and_retain_owners(&self.leases, vec![owner]);
        }
    }
}

impl CapturedResidentGraph {
    pub(super) fn output(&self) -> ResidentOutput {
        self.profile.output
    }

    pub(super) fn validate(
        &self,
        model: &ResidentModel,
        count: usize,
        positions: &[usize],
        readout: Readout,
    ) -> Result<()> {
        self.service.ensure_usable()?;
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
    pub(super) fn receipt(&self) -> GraphCaptureReceipt {
        self.receipt
    }
    pub(super) fn replay_enqueues(&self) -> usize {
        self.replay_enqueues.load(Ordering::Acquire)
    }

    /// Consumes the caller's sole graph handle. Both terminal barriers use the
    /// original physical queue and initialized Read leases. No graph destroy
    /// is posted until all prior replays have a trusted terminal readback.
    pub(super) async fn retire(self: Arc<Self>, root: u64) -> Result<()> {
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
        let ticket = self.service.queue().submit(
            self.service.budget(),
            self.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 4,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        self.service.ensure_usable()?;
        let accesses = self
            .leases
            .iter()
            .cloned()
            .map(|l| (l, DeviceAccess::Read))
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
    active: bool,
}
impl CaptureWindow {
    fn prepare(
        service: &Arc<ResidentService>,
        leases: Vec<ResidentLease>,
        band_pin: Arc<()>,
    ) -> Result<Self> {
        if service.client().is_capturing() {
            return Err(Error::State("capture already active".into()));
        }
        let window = Self {
            service: service.clone(),
            leases,
            band_pin,
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
            let owner: CompletionOwner = Arc::new(QuarantinedGraph {
                _graph: graph,
                _leases: self.leases.clone(),
                _service: self.service.clone(),
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
    pub(super) async fn capture_logits_b2(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
    ) -> Result<Arc<CapturedResidentGraph>> {
        if rows.len() != 2 {
            return Err(Error::Unsupported(
                "initial graph wrapper remains B2".into(),
            ));
        }
        self.capture_body(rows, ResidentOutput::Logits).await
    }

    pub(super) async fn capture_body(
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
        // First warm admits the original native workspace and completes private
        // output without publishing. Second warm fills prepared info keys.
        let first = match output {
            ResidentOutput::StateOnly => self.prepare_state_fused_cohort(rows).await?,
            ResidentOutput::Logits => self.prepare_logits_fused_cohort(rows).await?,
        };
        let first_count = first.dispatches();
        drop(first);
        let (leases, band_pin) = {
            let workspace = self.workspace.lock().await;
            let batch = workspace
                .batch
                .as_ref()
                .ok_or_else(|| Error::State("graph warm batch missing".into()))?;
            if batch.program.graph_pinned() {
                return Err(Error::State("one graph per live batch workspace".into()));
            }
            let handles = batch.program.execution_handles(output);
            let leases = workspace
                .leases
                .iter()
                .chain(&batch.leases)
                .filter(|(l, _)| handles.iter().any(|h| same_handle(h, &l.handle())))
                .map(|(l, _)| l.clone())
                .collect::<Vec<_>>();
            (leases, batch.program.graph_pin())
        };
        if leases.is_empty() || leases.len() > 65_536 {
            return Err(Error::Limit("graph lease metadata bound".into()));
        }
        let mut window = CaptureWindow::prepare(&self.service, leases.clone(), band_pin)?;
        let second = match output {
            ResidentOutput::StateOnly => self.prepare_state_fused_cohort(rows).await?,
            ResidentOutput::Logits => self.prepare_logits_fused_cohort(rows).await?,
        };
        let second_count = second.dispatches();
        drop(second);
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
                kind: AllocationClass::Scratch,
                bytes: 4,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        let workspace = self.workspace.lock().await;
        let batch = workspace
            .batch
            .as_ref()
            .ok_or_else(|| Error::State("graph capture batch missing".into()))?;
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
        let profile = batch.program.graph_profile(&parameters, output)?;
        if profile.body_nodes > 4096 {
            return Err(Error::Limit("graph node-prefix bound".into()));
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
                    && body.dispatches <= 65_536
                    && body.fused_dispatches > 0 =>
            {
                body
            }
            error => {
                let owner: CompletionOwner = Arc::new(QuarantinedGraph {
                    _graph: Some(graph),
                    _leases: leases.clone(),
                    _service: self.service.clone(),
                    _band_pin: batch.program.graph_pin(),
                });
                flight.retain_completion_owner(owner)?;
                return Err(error
                    .err()
                    .unwrap_or_else(|| Error::State("graph body dispatch bound".into())));
            }
        };
        let retained_lease_bytes = leases.iter().map(ResidentLease::bytes).sum();
        let owned_lease_vector_bytes = leases.capacity() * std::mem::size_of::<ResidentLease>();
        let graph = Arc::new(CapturedResidentGraph {
            service: self.service.clone(),
            profile,
            stream: self.service.pinned_cuda_queue()?.logical_stream_id().value,
            body,
            graph: HostMutex::new(Some(graph)),
            leases,
            status: status.clone(),
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

    pub(super) async fn prepare_graph_logits(
        self: &Arc<Self>,
        rows: &[(&ResidentSession, u32)],
        graph: &Arc<CapturedResidentGraph>,
    ) -> Result<PreparedResidentCohort> {
        self.prepare_graph_readout(rows, graph, Readout::Logits)
            .await
    }

    pub(super) async fn prepare_graph_readout(
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
