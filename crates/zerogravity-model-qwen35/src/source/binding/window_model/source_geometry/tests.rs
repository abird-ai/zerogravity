//! Descriptor checks only; no original source seal or execution authority.
use super::*;

#[test]
fn parameter_storage_is_exact_not_inferred_from_geometry() {
    let source = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
    for heads in [16, 48] {
        for dtype in [ScalarDType::Bf16, ScalarDType::F32] {
            let kind = StoredTensorEncoding::Scalar(dtype);
            let width = if dtype == ScalarDType::Bf16 { 2 } else { 4 };
            for (role, elements) in [
                (QwenParameterRole::GdnA(0), heads),
                (QwenParameterRole::GdnNorm(0), 128),
            ] {
                let descriptor = QwenParameterDescriptor {
                    source,
                    name: "original-parameter".into(),
                    kind,
                    shape: vec![elements],
                    bytes: elements as usize * width,
                };
                assert!(require_parameter_descriptor(&descriptor, role, kind, &[elements]).is_ok());
                let other = if dtype == ScalarDType::Bf16 {
                    ScalarDType::F32
                } else {
                    ScalarDType::Bf16
                };
                let mut wrong = descriptor.clone();
                wrong.kind = StoredTensorEncoding::Scalar(other);
                wrong.bytes = elements as usize * if other == ScalarDType::Bf16 { 2 } else { 4 };
                assert!(require_parameter_descriptor(&wrong, role, kind, &[elements]).is_err());
                let mut wrong = descriptor.clone();
                wrong.shape = vec![elements + 1];
                wrong.bytes += width;
                assert!(require_parameter_descriptor(&wrong, role, kind, &[elements]).is_err());
                let mut wrong = descriptor;
                wrong.bytes -= 1;
                assert!(require_parameter_descriptor(&wrong, role, kind, &[elements]).is_err());
            }
        }
    }
}
