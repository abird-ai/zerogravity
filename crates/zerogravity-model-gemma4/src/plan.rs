//! Finite Gemma source recipes over the shared mathematical plan.
//! Geometry metadata grants neither authenticated source nor native eligibility.
use crate::{Error, Result};
use std::sync::Arc;
use zerogravity_core::accelerator::{SourceIdentity, program::*};
use zerogravity_core::session_root::RootAttempt;

pub const BF16_TEXT_CAPACITY: usize = 64;
pub const BF16_TEXT_VOCABULARY: usize = 262144;

/// Existing E2B candidate limits; numerical profile identity remains unchanged.
pub fn bf16_text_contract() -> Bf16TextContract {
    Bf16TextContract {
        capacity: BF16_TEXT_CAPACITY,
        vocabulary: BF16_TEXT_VOCABULARY,
        required_domains: (0..15).map(|i| StateSlot::Kv(KvDomain(i))).collect(),
        constant_widths: vec![256, 512, 1536],
        projection_columns: vec![256, 1536, 2048, 4096, 6144, 12288],
        rms: [(256, 1), (512, 1), (1536, 1), (256, 8), (512, 8), (256, 35)]
            .into_iter()
            .map(|(width, groups)| Bf16TextRmsContract {
                width,
                groups,
                epsilon: 1e-6,
            })
            .collect(),
        rotary: vec![
            TextRotary {
                head_width: 256,
                rotated_pairs: 128,
                theta: 10000.0,
            },
            TextRotary {
                head_width: 512,
                rotated_pairs: 64,
                theta: 1000000.0,
            },
        ],
        attention: vec![
            Bf16TextAttentionContract {
                width: 256,
                query_heads: 8,
                kv_heads: 1,
                window: Some(512),
            },
            Bf16TextAttentionContract {
                width: 512,
                query_heads: 8,
                kv_heads: 1,
                window: None,
            },
        ],
    }
}
pub(crate) fn bf16_text_builder(
    source: SourceIdentity,
    capacity: usize,
    required: Vec<StateSlot>,
    capture: bool,
) -> Result<PlanBuilder> {
    PlanBuilder::new_bf16_text(source, capacity, required, capture, bf16_text_contract())
}
/// Bind the original attempt after applying this family's finite call policy.
pub fn bind_bf16_text_call(
    geometry: TextCallGeometry,
    attempt: RootAttempt,
    ids: Arc<[u32]>,
) -> Result<Bf16TextCall> {
    if geometry.vocabulary() != BF16_TEXT_VOCABULARY
        || geometry.capacity() > BF16_TEXT_CAPACITY
        || attempt.base().frontier() != 0 && ids.len() != 1
    {
        return Err(Error::Unsupported(
            "Gemma BF16 finite vocabulary/capacity/cached row".into(),
        ));
    }
    Bf16TextCall::bind(geometry, attempt, ids)
}

/// Numeric zero ties and lowest original ID, with the source vocabulary bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16TextGreedy;
impl Bf16TextGreedy {
    pub fn prefer(candidate: (u32, u32), prior: (u32, u32)) -> Result<bool> {
        zerogravity_core::accelerator::program::Bf16TextGreedy::prefer(
            BF16_TEXT_VOCABULARY,
            candidate,
            prior,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::quant::QuantKind;
    use zerogravity_core::session_root::SessionAuthority;

    fn builder(source: SourceIdentity) -> PlanBuilder {
        bf16_text_builder(source, 64, bf16_text_contract().required_domains, true).unwrap()
    }
    fn embedding(source: SourceIdentity) -> SourceTensor {
        SourceTensor {
            source,
            name: "actual.embedding".into(),
            kind: QuantKind::Bf16,
            axes: MatrixAxes::OutputInput,
            columns: 1536,
            rows: BF16_TEXT_VOCABULARY,
        }
    }
    #[test]
    fn finite_e2b_policy_keeps_original_domains_and_operator_geometries() {
        let contract = bf16_text_contract();
        contract.validate().unwrap();
        assert_eq!(contract.capacity, 64);
        assert_eq!(contract.vocabulary, 262144);
        assert_eq!(
            contract.required_domains,
            (0..15)
                .map(|i| StateSlot::Kv(KvDomain(i)))
                .collect::<Vec<_>>()
        );
        assert_eq!(contract.constant_widths, [256, 512, 1536]);
        assert_eq!(
            contract.projection_columns,
            [256, 1536, 2048, 4096, 6144, 12288]
        );
        let source = SourceIdentity::for_new_allocation().unwrap();
        for (capacity, domains) in [
            (65, contract.required_domains.clone()),
            (64, contract.required_domains[..14].to_vec()),
            (64, (1..16).map(|i| StateSlot::Kv(KvDomain(i))).collect()),
        ] {
            assert!(bf16_text_builder(source, capacity, domains, true).is_err());
        }
        let mut b = builder(source);
        let input = b
            .embed(embedding(source), PlanBuilder::inputs().token)
            .unwrap();
        let gamma = vec![0.0; 1536];
        let normalized = b
            .grouped_rms(
                "actual.norm",
                &input,
                SourceVector {
                    source,
                    name: "actual.norm.weight",
                    values: &gamma,
                },
                1,
                1e-6,
            )
            .unwrap();
        assert_eq!(normalized.precision(), ActivationPrecision::Bf16WidenedF32);
        assert!(
            b.grouped_rms(
                "wrong.eps",
                &input,
                SourceVector {
                    source,
                    name: "actual.norm.weight",
                    values: &gamma
                },
                1,
                1e-5
            )
            .is_err()
        );
        assert!(
            b.project(
                SourceTensor {
                    columns: 1536,
                    rows: 262145,
                    ..embedding(source)
                },
                &input
            )
            .is_err()
        );
        assert!(
            OrdinaryRmsContract::TorchCpuBf16Avx2
                .validate(
                    &SourceParameter {
                        source,
                        name: "norm".into(),
                        kind: QuantKind::Bf16,
                        shape: vec![1536],
                        bytes: 3072,
                    },
                    1536
                )
                .is_err()
        );
    }
    #[test]
    fn source_call_keeps_cached_one_and_original_attempt_without_publication() {
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut authority = SessionAuthority::new(source).unwrap();
        let geometry = TextCallGeometry::new(source, BF16_TEXT_VOCABULARY, 64).unwrap();
        let before = authority.binding();
        let initial = authority.begin_advance(4).unwrap();
        let call =
            bind_bf16_text_call(geometry, initial.clone(), Arc::from([4906, 2915, 886, 563]))
                .unwrap();
        assert_eq!(
            call.row(3).unwrap().pv_geometry(512).unwrap().contraction,
            4
        );
        assert_eq!(authority.binding(), before);
        authority.prepare_publish(initial).unwrap().commit();
        assert!(
            bind_bf16_text_call(
                geometry,
                authority.begin_advance(2).unwrap(),
                Arc::from([1, 2])
            )
            .is_err()
        );
        assert!(
            bind_bf16_text_call(
                geometry,
                authority.begin_advance(1).unwrap(),
                Arc::from([1])
            )
            .is_ok()
        );
        assert!(
            bind_bf16_text_call(
                TextCallGeometry::new(source, BF16_TEXT_VOCABULARY, 65).unwrap(),
                authority.begin_advance(1).unwrap(),
                Arc::from([1])
            )
            .is_err()
        );
        assert!(
            bind_bf16_text_call(
                TextCallGeometry::new(source, 17, 64).unwrap(),
                authority.begin_advance(1).unwrap(),
                Arc::from([1])
            )
            .is_err()
        );
        assert!(
            bind_bf16_text_call(
                geometry,
                authority.begin_advance(1).unwrap(),
                Arc::from([262144])
            )
            .is_err()
        );
    }
    #[test]
    fn source_argmax_keeps_exact_vocabulary_signed_zeros_and_nonfinite_refusals() {
        assert!(!Bf16TextGreedy::prefer((4, 0), (3, 0x80000000)).unwrap());
        assert!(Bf16TextGreedy::prefer((3, 0x80000000), (4, 0)).unwrap());
        assert!(Bf16TextGreedy::prefer((262143, 0x3f800000), (1, 0)).unwrap());
        assert!(Bf16TextGreedy::prefer((262144, 0x3f800000), (1, 0)).is_err());
        assert!(Bf16TextGreedy::prefer((0, 0x7fc00000), (1, 0)).is_err());
        assert!(Bf16TextGreedy::prefer((0, 1), (1, 0)).is_err());
    }
}
