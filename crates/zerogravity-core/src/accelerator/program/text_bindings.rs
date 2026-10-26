//! Immutable text-plan source metadata; no family, backend or completion permit.
use super::*;
use crate::resource_arbiter::HostWorkspace;
use crate::session_root::RootAttempt;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub enum SourceNorm {
    Direct(SourceParameter),
    Unit { width: usize },
}

/// Structural costs supplied by the source owner, checked again by lowerers.
/// These counts do not authenticate a model or admit physical storage.
#[derive(Clone, Debug)]
pub struct TextPlanInventory {
    pub unique_source_matrices: usize,
    pub full_source_matrix_bytes: usize,
    pub retained_source_parameter_bytes: usize,
    pub recorded_value_bytes: usize,
    pub kv_root_bytes: usize,
    pub six_root_payload_bytes: usize,
    pub source_page_bytes: usize,
    pub source_page_host_staging_bytes: usize,
    pub observation_frames: usize,
    pub largest_observation_bytes: usize,
    pub observation_host_upper_bytes: usize,
    pub known_host_upper_bytes: usize,
    pub projection_pages_with_head: usize,
    pub projection_pages_state_only: usize,
}
impl TextPlanInventory {
    pub fn host_with_lowerers(&self, actual_additional_bytes: usize) -> Result<usize> {
        add(self.known_host_upper_bytes, actual_additional_bytes)
    }
}

/// One completed producer coordinate; no late read or completion authority.
#[derive(Clone, Debug)]
pub struct TextObservation {
    pub node: usize,
    pub value: ValueId,
    pub layer: usize,
    pub stage: &'static str,
    shape: ValueShape,
    pub before_frontier: bool,
    pub allows_attention_mask: bool,
    frame_limit: usize,
}
impl TextObservation {
    pub fn for_plan(plan: &ForwardPlan, frame_limit: usize) -> Result<Vec<Self>> {
        observations(plan, frame_limit)
    }
    pub fn bytes(&self, row: &Bf16TextRow<'_>) -> Result<usize> {
        let elements = match self.shape {
            ValueShape::CausalRows { heads, capacity } => {
                if row.valid_tokens() > capacity {
                    return Err(Error::State("text observation row exceeds capacity".into()));
                }
                times(heads, row.valid_tokens())?
            }
            shape => shape.capacity_elements()?,
        };
        let bytes = times(elements, 4)?;
        if bytes == 0 || bytes > self.frame_limit {
            return Err(Error::Limit(
                "text producer frame exceeds source page".into(),
            ));
        }
        Ok(bytes)
    }
}

/// Explicit source-owned call/readout policy; not a backend support grant.
#[derive(Clone, Debug)]
pub struct TextCallPolicy {
    pub cached_rows: usize,
    pub end_tokens: Arc<[u32]>,
}

pub struct TextPlanBindings {
    plan: Arc<ForwardPlan>,
    call: Bf16TextCall,
    norms: BTreeMap<usize, SourceNorm>,
    scalars: BTreeMap<usize, SourceParameter>,
    inventory: TextPlanInventory,
    policy: TextCallPolicy,
    observations: Vec<TextObservation>,
}
impl TextPlanBindings {
    /// Structural metadata only. The model authenticates roles, raw parameter
    /// words, ties and finite numerical/source eligibility before construction.
    pub fn new(
        plan: ForwardPlan,
        call: Bf16TextCall,
        norms: BTreeMap<usize, SourceNorm>,
        scalars: BTreeMap<usize, SourceParameter>,
        inventory: TextPlanInventory,
        policy: TextCallPolicy,
        metadata: HostWorkspace,
    ) -> Result<Self> {
        if plan.source() != call.geometry().source()
            || plan.prefix_capacity() != call.geometry().capacity()
            || metadata.source_identity() != plan.source()
            || metadata.bytes() == 0
            || inventory.source_page_bytes == 0
            || inventory.known_host_upper_bytes < metadata.bytes()
        {
            return Err(Error::State(
                "text binding source/call/metadata differs".into(),
            ));
        }
        if policy.cached_rows == 0
            || policy
                .end_tokens
                .iter()
                .any(|id| *id as usize >= call.geometry().vocabulary())
            || policy
                .end_tokens
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != policy.end_tokens.len()
        {
            return Err(Error::State(
                "text end-token policy outside source vocabulary".into(),
            ));
        }
        let expected_norms = plan
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                matches!(node.operation, PlanOperation::Rms { .. }).then_some(index)
            })
            .collect::<BTreeSet<_>>();
        if norms.keys().copied().collect::<BTreeSet<_>>() != expected_norms {
            return Err(Error::State(
                "text binding normalization coverage differs".into(),
            ));
        }
        for (&index, norm) in &norms {
            let PlanOperation::Rms { gamma, .. } = plan.nodes()[index].operation else {
                unreachable!("validated normalization coverage")
            };
            let vector = plan
                .constants()
                .get(gamma.0 as usize)
                .ok_or_else(|| Error::State("text normalization constant absent".into()))?;
            if vector.source != plan.source() || vector.values.is_empty() {
                return Err(Error::State(
                    "text normalization constant source/extent".into(),
                ));
            }
            match norm {
                SourceNorm::Direct(parameter) => {
                    validate_parameter(plan.source(), parameter)?;
                    if parameter.name != vector.name
                        || parameter.shape != [vector.values.len() as u64]
                    {
                        return Err(Error::State("text direct gamma descriptor differs".into()));
                    }
                }
                SourceNorm::Unit { width } => {
                    if *width != vector.values.len()
                        || vector
                            .values
                            .iter()
                            .any(|v| v.to_bits() != 1.0f32.to_bits())
                    {
                        return Err(Error::State("text unit gamma constant differs".into()));
                    }
                }
            }
        }
        for (&index, parameter) in &scalars {
            validate_parameter(plan.source(), parameter)?;
            if parameter.shape != [1]
                || !plan.nodes().get(index).is_some_and(|node| {
                    matches!(
                        node.operation,
                        PlanOperation::TextElementwise(TextElementwise::Scale(_))
                    )
                })
            {
                return Err(Error::State("text scalar descriptor/node differs".into()));
            }
        }
        let observations = observations(&plan, inventory.source_page_bytes)?;
        let mut largest = 0;
        for observation in &observations {
            for index in 0..call.rows() {
                largest = largest.max(observation.bytes(&call.row(index)?)?);
            }
        }
        let value_bytes = plan.values().iter().try_fold(0usize, |sum, shape| {
            add(sum, times(shape.capacity_elements()?, 4)?)
        })?;
        if observations.len() != inventory.observation_frames
            || largest != inventory.largest_observation_bytes
            || value_bytes != inventory.recorded_value_bytes
        {
            return Err(Error::State(
                "text binding structural inventory differs".into(),
            ));
        }
        Ok(Self {
            plan: Arc::new(plan.retain_prepared_metadata(metadata)?),
            call,
            norms,
            scalars,
            inventory,
            policy,
            observations,
        })
    }
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub fn shared_plan(&self) -> &Arc<ForwardPlan> {
        &self.plan
    }
    pub fn call(&self) -> &Bf16TextCall {
        &self.call
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.plan.source()
    }
    pub fn inventory(&self) -> &TextPlanInventory {
        &self.inventory
    }
    pub fn end_tokens(&self) -> &[u32] {
        &self.policy.end_tokens
    }
    pub fn bind_call(&self, attempt: RootAttempt, ids: Arc<[u32]>) -> Result<Bf16TextCall> {
        if attempt.base().frontier() != 0 && ids.len() > self.policy.cached_rows {
            return Err(Error::Unsupported(
                "text call exceeds source cached-row policy".into(),
            ));
        }
        Bf16TextCall::bind(self.call.geometry(), attempt, ids)
    }
    pub fn normalization_source(&self, node: usize) -> Result<&SourceNorm> {
        self.norms
            .get(&node)
            .ok_or_else(|| Error::State("text normalization node absent".into()))
    }
    pub fn scalar_source(&self, node: usize) -> Option<&SourceParameter> {
        self.scalars.get(&node)
    }
    pub fn observations(&self) -> &[TextObservation] {
        &self.observations
    }
    pub fn observations_for_row(&self, index: usize) -> Result<Vec<TextObservation>> {
        let row = self.call.row(index)?;
        Ok(self
            .observations
            .iter()
            .filter(|o| o.before_frontier || row.needs_logits())
            .cloned()
            .collect())
    }
    pub fn projection_geometry_for_row(
        &self,
        node: usize,
        row: &Bf16TextRow<'_>,
    ) -> Result<Bf16TextProjectionGeometry> {
        if node >= self.plan.state_nodes().len() && !row.needs_logits() {
            return Err(Error::State(
                "text state-only row has no output projection".into(),
            ));
        }
        let Some(PlanNode {
            operation: PlanOperation::Project { tensor },
            ..
        }) = self.plan.nodes().get(node)
        else {
            return Err(Error::State("text projection source node absent".into()));
        };
        row.projection_geometry(tensor)
    }
}

fn observations(plan: &ForwardPlan, frame_limit: usize) -> Result<Vec<TextObservation>> {
    let masked = plan
        .nodes()
        .iter()
        .filter(|n| matches!(n.operation, PlanOperation::TextWindowQk { .. }))
        .flat_map(|n| n.outputs.iter().map(|value| value.0))
        .collect::<BTreeSet<_>>();
    let mut result = Vec::new();
    for (index, node) in plan.nodes().iter().enumerate() {
        if let PlanOperation::Observe { layer, stage } = node.operation {
            let [value] = node.inputs.as_slice() else {
                return Err(Error::State("text observer exact producer arity".into()));
            };
            let shape = *plan
                .values()
                .get(value.0 as usize)
                .ok_or_else(|| Error::State("text observer producer extent absent".into()))?;
            result.push(TextObservation {
                node: index,
                value: *value,
                layer,
                stage,
                shape,
                before_frontier: index < plan.state_nodes().len(),
                allows_attention_mask: masked.contains(&value.0),
                frame_limit,
            });
        }
    }
    Ok(result)
}
fn validate_parameter(source: SourceIdentity, parameter: &SourceParameter) -> Result<()> {
    let width = match parameter.kind {
        QuantKind::Bf16 => 2,
        QuantKind::F32 => 4,
        _ => return Err(Error::Unsupported("text source parameter codec".into())),
    };
    let count = parameter.shape.iter().try_fold(1usize, |n, width| {
        let width =
            usize::try_from(*width).map_err(|_| Error::Limit("text parameter width".into()))?;
        times(n, width)
    })?;
    if parameter.source != source
        || parameter.name.is_empty()
        || parameter.shape.is_empty()
        || count == 0
        || parameter.bytes != times(count, width)?
    {
        return Err(Error::State(
            "text source parameter shape/bytes/owner".into(),
        ));
    }
    Ok(())
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| Error::Limit("text binding resource sum overflow".into()))
}
fn times(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Limit("text binding resource product overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    };
    use crate::session_root::SessionAuthority;

    fn metadata(source: SourceIdentity) -> HostWorkspace {
        let arbiter = ResourceArbiter::new();
        let host = PhysicalDeviceKey::new("host", "text-bindings-fixture").unwrap();
        arbiter.register(host.clone(), 8192).unwrap();
        arbiter
            .admit(
                &host,
                ResourceOwner::new("fixture", "metadata").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 4096,
                }],
            )
            .unwrap()
            .into_host_workspace(source, Arc::new(()))
            .unwrap()
    }
    fn fixture() -> (
        ForwardPlan,
        Bf16TextCall,
        TextPlanInventory,
        SessionAuthority,
    ) {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let authority = SessionAuthority::new(source).unwrap();
        let call = Bf16TextCall::bind(
            TextCallGeometry::new(source, 4, 8).unwrap(),
            authority.begin_advance(1).unwrap(),
            Arc::from([1]),
        )
        .unwrap();
        let mut builder = PlanBuilder::new_with_profile(
            source,
            8,
            vec![StateSlot::Kv(KvDomain(0))],
            true,
            ProgramNumericalProfile::AnalyticF32TextV1,
        )
        .unwrap();
        let value = builder
            .embed(
                SourceTensor {
                    source,
                    name: "fixture.embedding".into(),
                    kind: QuantKind::F32,
                    axes: MatrixAxes::OutputInput,
                    columns: 1,
                    rows: 4,
                },
                PlanBuilder::inputs().token,
            )
            .unwrap();
        builder.observe(0, "embedding", &value).unwrap();
        builder.append_kv(KvDomain(0), &value, &value).unwrap();
        builder.advance_frontier().unwrap();
        let plan = builder.finish(Some(value)).unwrap();
        let recorded_value_bytes = plan
            .values()
            .iter()
            .map(|v| v.capacity_elements().unwrap() * 4)
            .sum();
        let inventory = TextPlanInventory {
            unique_source_matrices: 1,
            full_source_matrix_bytes: 16,
            retained_source_parameter_bytes: 0,
            recorded_value_bytes,
            kv_root_bytes: 64,
            six_root_payload_bytes: 384,
            source_page_bytes: 1024,
            source_page_host_staging_bytes: 3072,
            observation_frames: 1,
            largest_observation_bytes: 4,
            observation_host_upper_bytes: 12,
            known_host_upper_bytes: 4096,
            projection_pages_with_head: 0,
            projection_pages_state_only: 0,
        };
        (plan, call, inventory, authority)
    }
    fn policy() -> TextCallPolicy {
        TextCallPolicy {
            cached_rows: 1,
            end_tokens: Arc::from([3]),
        }
    }
    #[test]
    fn shared_graph_retains_source_charge_after_the_adapter_is_dropped() {
        let (plan, call, inventory, authority) = fixture();
        assert_eq!(plan.prepared_metadata_bytes(), None);
        let before = authority.binding();
        let arbiter = ResourceArbiter::new();
        let host = PhysicalDeviceKey::new("host", "shared-text-plan").unwrap();
        arbiter.register(host.clone(), 8192).unwrap();
        let owner = Arc::new(());
        let witness = Arc::downgrade(&owner);
        let charge = arbiter
            .admit(
                &host,
                ResourceOwner::new("fixture", "metadata").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 4096,
                }],
            )
            .unwrap()
            .into_host_workspace(plan.source(), owner)
            .unwrap();
        let bindings = TextPlanBindings::new(
            plan,
            call,
            BTreeMap::new(),
            BTreeMap::new(),
            inventory,
            policy(),
            charge,
        )
        .unwrap();
        let graph = Arc::clone(bindings.shared_plan());
        drop(bindings);
        assert_eq!(graph.prepared_metadata_bytes(), Some(4096));
        assert_eq!(arbiter.snapshot(&host).unwrap().reserved, 4096);
        assert!(witness.upgrade().is_some());
        assert_eq!(authority.binding(), before);
        drop(graph);
        assert_eq!(arbiter.snapshot(&host).unwrap().reserved, 0);
        assert!(witness.upgrade().is_none());
    }
    #[test]
    fn source_and_structural_inventory_cannot_retag_the_retained_charge() {
        let (plan, call, inventory, _) = fixture();
        let other = SourceIdentity::for_new_allocation().unwrap();
        assert!(
            TextPlanBindings::new(
                plan,
                call,
                BTreeMap::new(),
                BTreeMap::new(),
                inventory,
                policy(),
                metadata(other)
            )
            .is_err()
        );
        let (plan, call, mut inventory, _) = fixture();
        let source = plan.source();
        inventory.recorded_value_bytes += 4;
        assert!(
            TextPlanBindings::new(
                plan,
                call,
                BTreeMap::new(),
                BTreeMap::new(),
                inventory,
                policy(),
                metadata(source)
            )
            .is_err()
        );
    }
    #[test]
    fn source_policy_preserves_cached_row_limit_and_vocabulary() {
        let (plan, call, inventory, mut authority) = fixture();
        let source = plan.source();
        let bindings = TextPlanBindings::new(
            plan,
            call,
            BTreeMap::new(),
            BTreeMap::new(),
            inventory,
            policy(),
            metadata(source),
        )
        .unwrap();
        let first = authority.begin_advance(1).unwrap();
        authority.prepare_publish(first).unwrap().commit();
        assert!(
            bindings
                .bind_call(authority.begin_advance(2).unwrap(), Arc::from([1, 2]))
                .is_err()
        );
        assert!(
            bindings
                .bind_call(authority.begin_advance(1).unwrap(), Arc::from([4]))
                .is_err()
        );
        assert!(
            bindings
                .bind_call(authority.begin_advance(1).unwrap(), Arc::from([2]))
                .is_ok()
        );
        assert_eq!(bindings.end_tokens(), [3]);
    }
    #[test]
    fn duplicate_or_foreign_end_tokens_are_not_source_policy() {
        for tokens in [Arc::<[u32]>::from([3, 3]), Arc::<[u32]>::from([3, 4])] {
            let (plan, call, inventory, _) = fixture();
            let source = plan.source();
            assert!(
                TextPlanBindings::new(
                    plan,
                    call,
                    BTreeMap::new(),
                    BTreeMap::new(),
                    inventory,
                    TextCallPolicy {
                        cached_rows: 1,
                        end_tokens: tokens
                    },
                    metadata(source)
                )
                .is_err()
            );
        }
    }
}
