//! Immutable one-row source plan bindings; no family or execution authority.
use super::*;
use crate::resource_arbiter::HostWorkspace;
use crate::session_root::RootAttempt;

/// A recorded mathematical step and its source-owned call geometry. The source
/// factory authenticates its family recipe; the backend joins actual storage.
/// This data neither checks the current root nor permits state publication.
pub struct TextStepBindings {
    plan: Arc<ForwardPlan>,
    geometry: TextCallGeometry,
    position: usize,
}
impl TextStepBindings {
    pub fn new(
        plan: ForwardPlan,
        geometry: TextCallGeometry,
        position: usize,
        metadata: HostWorkspace,
    ) -> Result<Self> {
        let logits = plan
            .logits()
            .ok_or_else(|| Error::State("text step complete vocabulary producer absent".into()))?;
        let output = plan
            .values()
            .get(logits.0 as usize)
            .ok_or_else(|| Error::State("text step vocabulary producer outside plan".into()))?;
        if plan.source() != geometry.source()
            || plan.prefix_capacity() != geometry.capacity()
            || position
                .checked_add(1)
                .is_none_or(|end| end > geometry.capacity())
            || plan.is_operator_slice()
            || plan.frontier_delta() != 1
            || output.capacity_elements()? != geometry.vocabulary()
            || metadata.source_identity() != plan.source()
            || metadata.bytes() == 0
        {
            return Err(Error::State(
                "text step original source/geometry/metadata differs".into(),
            ));
        }
        Ok(Self {
            plan: Arc::new(plan.retain_prepared_metadata(metadata)?),
            geometry,
            position,
        })
    }
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub fn shared_plan(&self) -> &Arc<ForwardPlan> {
        &self.plan
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.geometry.source()
    }
    pub fn geometry(&self) -> TextCallGeometry {
        self.geometry
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn capacity(&self) -> usize {
        self.geometry.capacity()
    }
    /// Preserve the genuine attempt and one actual ID; current-root/completion
    /// checks remain with the consuming session and physical step.
    pub fn bind_call(&self, attempt: RootAttempt, id: u32) -> Result<Bf16TextCall> {
        if attempt.base().frontier() != self.position as u64 {
            return Err(Error::State(
                "text step phase/actual root attempt differs".into(),
            ));
        }
        Bf16TextCall::bind(self.geometry, attempt, Arc::from([id]))
    }
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
        let host = PhysicalDeviceKey::new("host", "text-step-fixture").unwrap();
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
    fn plan(source: SourceIdentity) -> ForwardPlan {
        let mut builder = PlanBuilder::new_with_profile(
            source,
            2,
            vec![StateSlot::Kv(KvDomain(0))],
            true,
            ProgramNumericalProfile::AnalyticF32TextV1,
        )
        .unwrap();
        let output = builder
            .embed(
                SourceTensor {
                    source,
                    name: "fixture.embedding".into(),
                    kind: QuantKind::F32,
                    axes: MatrixAxes::OutputInput,
                    columns: 4,
                    rows: 4,
                },
                PlanBuilder::inputs().token,
            )
            .unwrap();
        builder.append_kv(KvDomain(0), &output, &output).unwrap();
        builder.advance_frontier().unwrap();
        builder.finish(Some(output)).unwrap()
    }
    #[test]
    fn source_capacity_vocabulary_position_and_charge_cannot_be_retagged() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let other = SourceIdentity::for_new_allocation().unwrap();
        for (geometry, position, charge) in [
            (TextCallGeometry::new(other, 4, 2).unwrap(), 0, source),
            (TextCallGeometry::new(source, 4, 3).unwrap(), 0, source),
            (TextCallGeometry::new(source, 5, 2).unwrap(), 0, source),
            (TextCallGeometry::new(source, 4, 2).unwrap(), 2, source),
            (TextCallGeometry::new(source, 4, 2).unwrap(), 0, other),
        ] {
            assert!(
                TextStepBindings::new(plan(source), geometry, position, metadata(charge)).is_err()
            );
        }
    }
    #[test]
    fn actual_one_row_attempt_keeps_source_position_ids_and_original_root() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let initial = TextStepBindings::new(
            plan(source),
            TextCallGeometry::new(source, 4, 2).unwrap(),
            0,
            metadata(source),
        )
        .unwrap();
        let cached = TextStepBindings::new(
            plan(source),
            TextCallGeometry::new(source, 4, 2).unwrap(),
            1,
            metadata(source),
        )
        .unwrap();
        let mut authority = SessionAuthority::new(source).unwrap();
        let before = authority.binding();
        let attempt = authority.begin_advance(1).unwrap();
        let call = initial.bind_call(attempt.clone(), 3).unwrap();
        call.require_current(before).unwrap();
        assert_eq!(call.ids(), [3]);
        assert_eq!(authority.binding(), before);
        assert!(
            initial
                .bind_call(authority.begin_advance(2).unwrap(), 1)
                .is_err()
        );
        assert!(initial.bind_call(attempt.clone(), 4).is_err());
        assert!(cached.bind_call(attempt.clone(), 1).is_err());
        let foreign = SessionAuthority::new(SourceIdentity::for_new_allocation().unwrap()).unwrap();
        assert!(
            initial
                .bind_call(foreign.begin_advance(1).unwrap(), 1)
                .is_err()
        );
        authority.prepare_publish(attempt).unwrap().commit();
        assert!(
            initial
                .bind_call(authority.begin_advance(1).unwrap(), 1)
                .is_err()
        );
        let call = cached
            .bind_call(authority.begin_advance(1).unwrap(), 2)
            .unwrap();
        call.require_current(authority.binding()).unwrap();
        assert_eq!(call.row(0).unwrap().position(), 1);
        assert_eq!(call.geometry().capacity(), 2);
    }
}
