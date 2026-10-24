//! Guard-only tests; no device/client/lease or numerical execution claim.
use super::*;
fn fixture(case: VocabularyHeadCase, scope: Scope) -> VocabularyHeadFixture {
    let (selected, final_norm, roles) = match scope {
        Scope::CapturedSelected => (vec![0; 2048], None, vec![VocabularyHeadOutput::Vocabulary]),
        Scope::ComputedFinalNorm => (
            Vec::new(),
            Some(FinalNormInputs {
                branch: vec![0; case.phase_rows() * 2048],
                residual: vec![0; case.phase_rows() * 2048],
            }),
            vec![
                VocabularyHeadOutput::SumF32,
                VocabularyHeadOutput::Normalized,
                VocabularyHeadOutput::Residual,
                VocabularyHeadOutput::Selected,
                VocabularyHeadOutput::Vocabulary,
            ],
        ),
    };
    VocabularyHeadFixture {
        case,
        selected,
        final_norm,
        expected: roles
            .into_iter()
            .map(|output| VocabularyHeadExpected {
                output,
                bytes: vec![0; output.bytes(case)],
            })
            .collect(),
    }
}
#[test]
fn true_phase_and_selected_row_are_preserved_for_all_five_calls() {
    assert_eq!(
        VocabularyHeadCase::ALL.map(|c| c.phase_rows()),
        [1, 1, 64, 2, 1]
    );
    assert_eq!(
        VocabularyHeadCase::ALL.map(|c| c.selected_row() * 2048),
        [0, 0, 129_024, 2048, 0]
    );
    for case in VocabularyHeadCase::ALL {
        validate_record(
            &fixture(case, Scope::ComputedFinalNorm),
            case,
            Scope::ComputedFinalNorm,
        )
        .unwrap();
    }
}
#[test]
fn captured_inputs_cannot_enter_the_computed_scope() {
    let mut captured = fixture(VocabularyHeadCase::Initial11, Scope::CapturedSelected);
    validate_record(&captured, captured.case, Scope::CapturedSelected).unwrap();
    assert!(validate_record(&captured, captured.case, Scope::ComputedFinalNorm).is_err());
    captured.final_norm = Some(FinalNormInputs {
        branch: vec![0; 2048],
        residual: vec![0; 2048],
    });
    assert!(validate_record(&captured, captured.case, Scope::CapturedSelected).is_err());
}
#[test]
fn bulk_phase_cannot_be_retagged_as_one_row() {
    let mut actual = fixture(VocabularyHeadCase::InitialFirst64, Scope::ComputedFinalNorm);
    assert!(
        validate_record(
            &actual,
            VocabularyHeadCase::Initial11,
            Scope::ComputedFinalNorm
        )
        .is_err()
    );
    actual.case = VocabularyHeadCase::Initial11;
    assert!(validate_record(&actual, actual.case, Scope::ComputedFinalNorm).is_err());
}
#[test]
fn independent_sum_residual_selected_and_full_vocabulary_are_mandatory() {
    for missing in 0..5 {
        let mut f = fixture(
            VocabularyHeadCase::Continued65And66,
            Scope::ComputedFinalNorm,
        );
        f.expected.remove(missing);
        assert!(validate_record(&f, f.case, Scope::ComputedFinalNorm).is_err());
    }
    let mut f = fixture(VocabularyHeadCase::Initial11, Scope::ComputedFinalNorm);
    f.expected[0].output = VocabularyHeadOutput::Residual;
    assert!(validate_record(&f, f.case, Scope::ComputedFinalNorm).is_err());
}
#[test]
fn wrong_dtype_extents_nonfinite_input_and_nonfinite_expectations_refuse() {
    let mut f = fixture(VocabularyHeadCase::Initial11, Scope::CapturedSelected);
    f.selected[0..2].copy_from_slice(&0x7fc1u16.to_le_bytes());
    assert!(validate_record(&f, f.case, Scope::CapturedSelected).is_err());
    let mut f = fixture(VocabularyHeadCase::Initial11, Scope::ComputedFinalNorm);
    f.expected[0].bytes[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(validate_record(&f, f.case, Scope::ComputedFinalNorm).is_err());
    let mut f = fixture(VocabularyHeadCase::InitialFirst64, Scope::ComputedFinalNorm);
    f.final_norm.as_mut().unwrap().branch.truncate(2048);
    assert!(validate_record(&f, f.case, Scope::ComputedFinalNorm).is_err());
}
