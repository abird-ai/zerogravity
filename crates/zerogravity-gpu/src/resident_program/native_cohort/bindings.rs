//! Every logical row keeps its own source, root, phase and StateSlot namespace.
use super::super::*;
use crate::resident_inference::native_fragment::{
    PrefixPhaseEnqueue,
    short_hf::{ShortHfCallInput, ShortHfCallState},
};
use crate::resident_leases::ResidentService;
use zerogravity_core::accelerator::program::RecordedVllm08Model;

pub(super) struct CohortRowBinding {
    input: Arc<ShortHfCallInput>,
    state: Arc<ShortHfCallState>,
    runtime: RuntimeParameters,
    bindings: Vec<StateBinding>,
}

/// Created from all actual permits before node0. No first-row runtime/state
/// argument can stand in for this closed population.
pub(super) struct CohortBindings {
    rows: Box<[CohortRowBinding]>,
}
impl CohortBindings {
    pub(super) fn bind(
        service: &Arc<ResidentService>,
        states: &[Arc<ShortHfCallState>],
        permits: &[PrefixPhaseEnqueue<'_>],
        owned: &OwnedRanges,
    ) -> Result<Self> {
        if !matches!(states.len(), 4 | 8) || states.len() != permits.len() {
            return Err(Error::State(
                "cohort exact original4/8 row population".into(),
            ));
        }
        service.ensure_usable()?;
        let mut rows = Vec::with_capacity(states.len());
        for (state, permit) in states.iter().zip(permits) {
            let input = state.input().clone();
            input.require_permit(permit, service)?;
            state.require_current(permit.session())?;
            state.require_submission_claimed()?;
            input
                .scope()
                .require_record(input.source(), input.record())?;
            let runtime = RuntimeParameters {
                token: input.ids()[0],
                position: input.base(),
                valid_tokens: input.rows(),
            };
            if !matches!((input.rows(), input.base()), (47, 0) | (1, 47..=53))
                || input.workload().prompt_rows() != 47
                || rows.first().is_some_and(|first: &CohortRowBinding| {
                    first.runtime.valid_tokens != runtime.valid_tokens
                })
            {
                return Err(Error::Unsupported(
                    "cohort canonical47 original phase/row geometry".into(),
                ));
            }
            let bindings = state.runtime_bindings(permit.session())?;
            rows.push(CohortRowBinding {
                input,
                state: state.clone(),
                runtime,
                bindings,
            });
        }
        for row in rows.iter().skip(1) {
            require_same_semantics(rows[0].input.record(), row.input.record())?;
        }
        let result = Self {
            rows: rows.into_boxed_slice(),
        };
        result.require_current(service, permits, owned)?;
        Ok(result)
    }

    /// All ranges and source/current-root authorities are proved before ANY
    /// clear/copy/kernel. Candidate writes cannot alias any other row's prior,
    /// candidate or workspace, including shared-allocation subviews.
    pub(super) fn require_current(
        &self,
        service: &Arc<ResidentService>,
        permits: &[PrefixPhaseEnqueue<'_>],
        owned: &OwnedRanges,
    ) -> Result<()> {
        if self.rows.len() != permits.len() || owned.service != service.client().service_id() {
            return Err(Error::State(
                "cohort original runtime population/service".into(),
            ));
        }
        service.ensure_usable()?;
        let mut roots = BTreeSet::new();
        let mut priors = Vec::with_capacity(48 * self.rows.len());
        let mut candidates = Vec::with_capacity(48 * self.rows.len());
        for (row, permit) in self.rows.iter().zip(permits) {
            row.input.require_permit(permit, service)?;
            row.state.require_current(permit.session())?;
            row.state.require_submission_claimed()?;
            // Original root bindings include source, session, frontier and root.
            if !roots.insert(permit.session().binding().session()) {
                return Err(Error::State("cohort duplicate original session".into()));
            }
            if row.runtime.position != row.input.base()
                || row.runtime.valid_tokens != row.input.rows()
                || row.input.ids().first().copied() != Some(row.runtime.token)
                || !Arc::ptr_eq(row.state.input(), &row.input)
            {
                return Err(Error::State(
                    "cohort actual row runtime/source differs".into(),
                ));
            }
            for state in row.state.states() {
                state.require_enqueue(service, permit)?;
            }
            let requirements = super::super::native::state_requirements(
                row.input.source().source(),
                row.input.record().plan(),
            )?;
            let mut slots = BTreeMap::new();
            for binding in &row.bindings {
                if slots.insert(binding.slot, binding).is_some() {
                    return Err(Error::State("cohort duplicate state in one row".into()));
                }
            }
            if slots.len() != requirements.len() || slots.len() != 42 {
                return Err(Error::State("cohort all42 domains in every row".into()));
            }
            for requirement in requirements {
                let binding = slots
                    .get(&requirement.slot)
                    .ok_or_else(|| Error::State("cohort missing actual row domain".into()))?;
                for (state, ranges) in [
                    (&binding.prior, &mut priors),
                    (&binding.candidate, &mut candidates),
                ] {
                    match (requirement.layout, state) {
                        (StateLayout::Vector { elements }, StateBuffers::Vector(value)) => {
                            ranges.push(value.check(service.client(), elements)?);
                        }
                        (StateLayout::Kv { slab, capacity }, StateBuffers::Kv { keys, values }) => {
                            let elements = slab
                                .checked_mul(capacity)
                                .ok_or_else(|| Error::Limit("cohort KV extent".into()))?;
                            if binding.prior_tokens != row.runtime.position {
                                return Err(Error::State("cohort KV current frontier".into()));
                            }
                            ranges.push(keys.check(service.client(), elements)?);
                            ranges.push(values.check(service.client(), elements)?);
                        }
                        _ => return Err(Error::State("cohort typed state layout".into())),
                    }
                }
            }
        }
        if priors.len() != 48 * self.rows.len() || candidates.len() != priors.len() {
            return Err(Error::State("cohort actual48 planes per row".into()));
        }
        for (index, prior) in priors.iter().enumerate() {
            if owned.overlaps(prior) || priors[..index].iter().any(|other| prior.overlaps(other)) {
                return Err(Error::State(
                    "cohort prior aliases sibling prior or body workspace".into(),
                ));
            }
        }
        for (index, candidate) in candidates.iter().enumerate() {
            if owned.overlaps(candidate)
                || priors
                    .iter()
                    .chain(&candidates[..index])
                    .any(|other| candidate.overlaps(other))
            {
                return Err(Error::State(
                    "cohort proposal aliases any prior/proposal/workspace".into(),
                ));
            }
        }
        Ok(())
    }

    /// Current authorities at every shared node; full range proof is cold per
    /// launch. Original borrowed permits prevent restore/publication meanwhile.
    pub(super) fn require_enqueue(
        &self,
        service: &Arc<ResidentService>,
        permits: &[PrefixPhaseEnqueue<'_>],
    ) -> Result<()> {
        service.ensure_usable()?;
        if permits.len() != self.rows.len() {
            return Err(Error::State("cohort enqueue exact row permits".into()));
        }
        for (row, permit) in self.rows.iter().zip(permits) {
            row.input.require_permit(permit, service)?;
        }
        Ok(())
    }

    pub(super) fn inputs(&self) -> Vec<Arc<ShortHfCallInput>> {
        self.rows.iter().map(|row| row.input.clone()).collect()
    }
    pub(super) fn states_at(
        &self,
        layer: usize,
    ) -> Result<Vec<Arc<crate::resident_inference::native_fragment::model_state::BoundModelState>>>
    {
        self.rows
            .iter()
            .map(|row| {
                let mut states = row
                    .state
                    .states()
                    .iter()
                    .filter(|state| state.layer().index() == layer);
                let state = states
                    .next()
                    .ok_or_else(|| Error::State("cohort original layer state absent".into()))?;
                if states.next().is_some() {
                    return Err(Error::State("cohort repeated row/layer state".into()));
                }
                Ok(state.clone())
            })
            .collect()
    }
    pub(super) fn effects(
        &self,
        node: usize,
    ) -> Result<Vec<zerogravity_core::accelerator::program::StateEffect>> {
        self.rows
            .iter()
            .map(|row| {
                row.input
                    .record()
                    .plan()
                    .nodes()
                    .get(node)
                    .and_then(|node| node.state)
                    .ok_or_else(|| {
                        Error::State("cohort original per-row state effect absent".into())
                    })
            })
            .collect()
    }
    pub(super) fn state_accesses(
        &self,
    ) -> Vec<(
        crate::resident_leases::ResidentLease,
        zerogravity_core::device_events::DeviceAccess,
    )> {
        self.rows
            .iter()
            .flat_map(|row| row.state.states())
            .flat_map(|state| state.leases())
            .collect()
    }
    pub(super) fn source(
        &self,
    ) -> &Arc<zerogravity_model_qwen35::original_binding::OriginalWindowModelSource> {
        self.rows[0].input.source().source()
    }
    pub(super) fn width(&self) -> usize {
        self.rows.len()
    }
    pub(super) fn records(&self) -> Vec<Arc<RecordedVllm08Model>> {
        self.rows
            .iter()
            .map(|row| row.input.record().clone())
            .collect()
    }
    pub(super) fn record(&self, row: usize) -> Result<&RecordedVllm08Model> {
        self.rows
            .get(row)
            .map(|row| row.input.record().as_ref())
            .ok_or_else(|| Error::State("cohort original row record absent".into()))
    }

    /// Grouped terminal proposals remain bound to each real row's actual plan;
    /// duplicate StateSlots across requests are never flattened into one map.
    pub(super) fn proposals(&self) -> Result<Vec<Vec<(StateVersion, StateBuffers)>>> {
        self.rows
            .iter()
            .map(|row| {
                let mut proposals = Vec::with_capacity(42);
                for version in row.input.record().plan().proposed_domains() {
                    let binding = row
                        .bindings
                        .iter()
                        .find(|binding| binding.slot == version.slot)
                        .ok_or_else(|| {
                            Error::State("cohort actual terminal row domain absent".into())
                        })?;
                    proposals.push((*version, binding.candidate.clone()));
                }
                if proposals.len() != 42 {
                    return Err(Error::State("cohort terminal all42 domains per row".into()));
                }
                Ok(proposals)
            })
            .collect()
    }

    pub(super) fn bindings(&self) -> impl Iterator<Item = &[StateBinding]> {
        self.rows.iter().map(|row| row.bindings.as_slice())
    }
}

// Correctness bootstrap for independently recorded real invocations. Prepared
// recipe identity may replace this immutable comparison after peer integration;
// neither route shares a currently bound record or a row's provenance.
fn require_same_semantics(a: &RecordedVllm08Model, b: &RecordedVllm08Model) -> Result<()> {
    let (ap, bp) = (a.plan(), b.plan());
    if a.call().source() != b.call().source()
        || a.call().model_profile() != b.call().model_profile()
        || ap.numerical_profile() != bp.numerical_profile()
        || ap.values() != bp.values()
        || ap.precisions() != bp.precisions()
        || ap.required_domains() != bp.required_domains()
        || ap.proposed_domains() != bp.proposed_domains()
        || a.normalized() != b.normalized()
        || a.logits() != b.logits()
        || a.boundaries() != b.boundaries()
        || ap.nodes().len() != bp.nodes().len()
    {
        return Err(Error::State(
            "cohort source recipe/SSA/state geometry differs".into(),
        ));
    }
    if a.shares_prepared_semantics(b) {
        return Ok(());
    }
    for (index, (x, y)) in ap.nodes().iter().zip(bp.nodes()).enumerate() {
        if x.inputs != y.inputs || x.outputs != y.outputs || x.state != y.state {
            return Err(Error::State(
                "cohort shared node changes SSA/state effects".into(),
            ));
        }
        match (&x.operation, &y.operation) {
            (
                PlanOperation::TypedVllmBf16Embedding { descriptor: x },
                PlanOperation::TypedVllmBf16Embedding { descriptor: y },
            ) if index == 0 => {
                x.validate()?;
                y.validate()?;
                if x.tensor() != y.tensor() {
                    return Err(Error::State(
                        "cohort original embedding source differs".into(),
                    ));
                }
            }
            (
                PlanOperation::TypedVllm08 {
                    descriptor: x,
                    state: sx,
                },
                PlanOperation::TypedVllm08 {
                    descriptor: y,
                    state: sy,
                },
            ) => {
                x.validate()?;
                y.validate()?;
                if x.call() != a.call()
                    || y.call() != b.call()
                    || x.kind() != y.kind()
                    || x.layer() != y.layer()
                    || x.matrices() != y.matrices()
                    || x.parameters() != y.parameters()
                    || x.input_elements() != y.input_elements()
                    || x.output_elements() != y.output_elements()
                    || x.selected_row() != y.selected_row()
                    || sx != sy
                {
                    return Err(Error::State(
                        "cohort original source operator roles differ".into(),
                    ));
                }
            }
            _ => return Err(Error::State("cohort foreign mathematical operation".into())),
        }
    }
    Ok(())
}
