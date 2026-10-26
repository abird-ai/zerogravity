//! Exact fixed prompt policy; original teacher issuer, no prompt retagging.
use super::*;
use crate::plan::{QwenInputRows, QwenPrefixLayer};

const PROMPT_IDS: [u32; 47] = [
    248045, 846, 198, 22791, 411, 1103, 314, 24959, 11, 18101, 539, 73982, 11, 364, 1599, 777,
    3665, 25, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220,
    22, 11, 220, 23, 11, 248046, 198, 248045, 74455, 198,
];
impl OriginalWindowModelSource {
    pub fn validate_fixed_prompt_request(ids: &[u32], max_tokens: usize) -> Result<()> {
        if ids != PROMPT_IDS || !(1..=8).contains(&max_tokens) {
            return Err(Error::Unsupported(
                "fixed prompt exact47 IDs and1..8 output tokens only".into(),
            ));
        }
        Ok(())
    }
    /// Model-authenticated metadata; computed feedback and roots remain engine-owned.
    pub fn bind_fixed_prompt_call(
        &self,
        base: usize,
        ids: &[u32],
    ) -> Result<crate::plan::QwenSourceCall> {
        self.require_small_model_source()?;
        let teachers = if ids.len() == 1 {
            QwenInputRows::new(self.source_identity(), base, 512, ids)?
        } else {
            QwenInputRows::from_span(TeacherSpan::new(
                self.source_identity(),
                base,
                512,
                248320,
                ids,
            )?)?
        };
        let mode = if base == 0 {
            SourceCallMode::Prefill
        } else {
            SourceCallMode::Decode
        };
        let request = QwenGdnCall::from_span(
            QwenPrefixLayer::new(self.source_identity(), 0)?,
            mode,
            teachers,
        )?;
        CallKind::of(&request)?;
        crate::plan::QwenSourceCall::from_span(request)
    }
    pub fn prepare_fixed_prompt_call(&self, base: usize, ids: &[u32]) -> Result<PreparedPlan> {
        let call = self.bind_fixed_prompt_call(base, ids)?;
        self.prepare_fixed_prompt_program(self.record_small_model(call)?)
    }
    pub fn fixed_prompt_ids() -> &'static [u32; 47] {
        &PROMPT_IDS
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CallKind {
    Prefill,
    Decode { base: usize },
}
impl CallKind {
    fn of(request: &QwenGdnCall) -> Result<Self> {
        request.validate()?;
        if request.layer().index() != 0
            || request.teachers().prompt_span().is_some()
            || request.teachers().capacity() != 512
            || request.reference_phase().is_ok()
        {
            return Err(Error::Unsupported(
                "fixed prompt original source span only".into(),
            ));
        }
        match (request.mode(), request.teachers().base(), request.rows()) {
            (SourceCallMode::Prefill, 0, 47) if request.teachers().ids() == PROMPT_IDS => {
                Ok(Self::Prefill)
            }
            (SourceCallMode::Decode, base @ 47..=53, 1) => Ok(Self::Decode { base }),
            _ => Err(Error::Unsupported(
                "fixed prompt prefill47/decode47..53 source span".into(),
            )),
        }
    }
}
pub(super) fn prepare(
    source: &OriginalWindowModelSource,
    record: QwenPlan,
) -> Result<PreparedPlan> {
    if record.call().model_profile() != QwenSourceProfile::Qwen08
        || source.source().source_bytes() != 1_746_942_600
        || source.source().staging_chunk_bytes() != 1024 * 1024
        || source.owner().config().rms_norm_eps() != 1e-6
    {
        return Err(Error::Unsupported(
            "fixed prompt original .8B source/config/staging".into(),
        ));
    }
    require_reference_record(source, &record)?;
    CallKind::of(record.call().request())?;
    let metadata = source.source().reserve_workspace(METADATA_BYTES)?;
    require_embedding(source, record.plan())?;
    require_coverage(&record, |operator| source.validate_operator(operator))?;
    require_state_layout(source, record.plan())?;
    let pairs = gdn::for_record(source, &record, None)?;
    prepare_whole(source.source().clone(), record, metadata, pairs)
}
fn require_coverage(
    record: &QwenPlan,
    mut validate_member: impl FnMut(&QwenOperator) -> Result<()>,
) -> Result<()> {
    let mut rms = BTreeSet::new();
    let mut gdn = BTreeSet::new();
    let mut projections = 0usize;
    let mut matrices = BTreeSet::new();
    let mut parameters = BTreeSet::new();
    for (index, node) in record.plan().nodes().iter().enumerate() {
        let Some(descriptor) = record.operator(index)? else {
            continue;
        };
        if descriptor.call() != record.call() {
            return Err(Error::State(
                "fixed prompt foreign source/teacher issuer".into(),
            ));
        }
        validate_member(&descriptor)?;
        matrices.extend(descriptor.matrices().iter().map(|m| m.name.clone()));
        parameters.extend(descriptor.parameters().iter().map(|p| p.name.clone()));
        match descriptor.kind() {
            QwenOperation::InputRms | QwenOperation::ResidualRms(_) => {
                rms.insert(index);
            }
            QwenOperation::GdnConv => {
                let layer = descriptor
                    .layer()
                    .ok_or_else(|| Error::State("fixed prompt convolution layer absent".into()))?
                    .index();
                let next = record
                    .plan()
                    .nodes()
                    .get(index + 1)
                    .ok_or_else(|| Error::State("fixed prompt recurrence absent".into()))?;
                let recurrent = record
                    .operator(index + 1)?
                    .ok_or_else(|| Error::State("fixed prompt typed recurrence absent".into()))?;
                let ([qkvz, ba], [activated, z], [ra, rz, rb], [_gated]) = (
                    node.inputs.as_slice(),
                    node.outputs.as_slice(),
                    next.inputs.as_slice(),
                    next.outputs.as_slice(),
                ) else {
                    return Err(Error::State("fixed prompt actual GDN arities".into()));
                };
                let _ = qkvz;
                if recurrent.kind() != QwenOperation::GdnRecurrent
                    || recurrent.call() != record.call()
                    || recurrent.layer() != descriptor.layer()
                    || activated != ra
                    || z != rz
                    || ba != rb
                    || !gdn.insert(layer)
                {
                    return Err(Error::State(
                        "fixed prompt original GDN source/SSA pair".into(),
                    ));
                }
            }
            QwenOperation::Projection(p) if p != QwenProjection::Head => projections += 1,
            _ => {}
        }
    }
    if rms.len() != 49
        || gdn.len() != 18
        || projections != 114
        || matrices.len() != 187
        || parameters.len() != 133
        || !matrices.is_disjoint(&parameters)
        || !rms.contains(&1)
        || gdn.iter().copied().ne((0..24).filter(|i| i % 4 != 3))
    {
        return Err(Error::State(
            "fixed prompt all24 source member/operator coverage".into(),
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{PromptCall, PromptSpan, PromptWorkload};
    use zerogravity_core::accelerator::SourceIdentity;
    use zerogravity_core::accelerator::program::TeacherSpan;
    fn request(
        source: SourceIdentity,
        mode: SourceCallMode,
        base: usize,
        capacity: usize,
        ids: &[u32],
        layer: usize,
    ) -> Result<QwenGdnCall> {
        let teachers = if ids.len() == 1 {
            QwenInputRows::new(source, base, capacity, ids)?
        } else {
            QwenInputRows::from_span(TeacherSpan::new(source, base, capacity, 248320, ids)?)?
        };
        QwenGdnCall::from_span(QwenPrefixLayer::new(source, layer)?, mode, teachers)
    }
    #[test]
    fn original_fixed_prompt_and_cached_positions_keep_the_actual_issuer() -> Result<()> {
        let source = SourceIdentity::for_new_allocation()?;
        let prefill = request(source, SourceCallMode::Prefill, 0, 512, &PROMPT_IDS, 0)?;
        let actual = prefill.source_call()?;
        assert_eq!(CallKind::of(&prefill)?, CallKind::Prefill);
        assert_eq!(prefill.source_call()?, actual);
        for base in 47..=53 {
            let call = request(source, SourceCallMode::Decode, base, 512, &[353], 0)?;
            assert_eq!(CallKind::of(&call)?, CallKind::Decode { base });
        }
        Ok(())
    }
    #[test]
    fn refuses_retagged_prompt_reference_wrong_ids_mode_base_capacity_and_layer() -> Result<()> {
        let source = SourceIdentity::for_new_allocation()?;
        assert!(QwenInputRows::new(source, 0, 512, &PROMPT_IDS).is_err());
        assert!(request(source, SourceCallMode::Decode, 0, 512, &[353], 0).is_err());
        let mut wrong = PROMPT_IDS;
        wrong[19] = 17;
        for call in [
            request(source, SourceCallMode::Prefill, 0, 512, &wrong, 0)?,
            request(source, SourceCallMode::Prefill, 1, 512, &PROMPT_IDS, 0)?,
            request(source, SourceCallMode::Prefill, 0, 511, &PROMPT_IDS, 0)?,
            request(source, SourceCallMode::Prefill, 0, 512, &PROMPT_IDS, 1)?,
            request(source, SourceCallMode::Prefill, 47, 512, &[353], 0)?,
            request(source, SourceCallMode::Decode, 46, 512, &[353], 0)?,
            request(source, SourceCallMode::Decode, 54, 512, &[353], 0)?,
            request(source, SourceCallMode::Prefill, 0, 512, &[353; 64], 0)?,
        ] {
            assert!(CallKind::of(&call).is_err());
        }
        let tagged = PromptCall::new(
            QwenSourceProfile::Qwen08,
            PromptSpan::prefill(source, PromptWorkload::from_prompt_tokens(47)?, &PROMPT_IDS)?,
        )?;
        assert!(CallKind::of(tagged.request()).is_err());
        Ok(())
    }
}
