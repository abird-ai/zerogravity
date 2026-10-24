//! Pinned ATen CPU BF16 PV accumulator; no transposed GEMV/QK or native authority.
// PyTorch cf30153c4c131c8164ee7798e5022d810682e2cb, LinearAlgebra.cpp and
// cpu/BlasKernel.cpp. BSD notice: transcendentals/softmax/PYTORCH-LICENSE.txt.
use crate::{Error, Result, accelerator::SourceIdentity};

#[derive(Clone, Copy, Debug)]
pub struct Bf16PvGeometry {
    /// Full current-call query extent, not the number of already computed tokens.
    pub query_rows: usize,
    /// Original BMM contraction including masked future columns in this call.
    pub contraction: usize,
    pub output_width: usize,
}

/// One output before its original BF16 store. Inputs are actual finite BF16
/// active-prefix probabilities/V, not expected rows. Absent future terms have
/// source softmax probability+0; no cache row or source operand is fabricated.
pub fn bf16_pv_f32_accumulator(
    source: SourceIdentity,
    probabilities: &[f32],
    values: &[f32],
    geometry: Bf16PvGeometry,
) -> Result<f32> {
    if !source.is_bound() {
        return Err(Error::State("BF16 PV actual source".into()));
    }
    if !(1..=64).contains(&geometry.contraction)
        || !(1..=geometry.contraction).contains(&geometry.query_rows)
        || !matches!(geometry.output_width, 256 | 512)
        || probabilities.is_empty()
        || probabilities.len() != values.len()
        || probabilities.len() > geometry.contraction
    {
        return Err(Error::Unsupported("BF16 PV finite source geometry".into()));
    }
    if probabilities
        .iter()
        .any(|p| !p.is_finite() || p.to_bits() & 0xffff != 0 || !(0.0..=1.0).contains(p))
        || values
            .iter()
            .any(|v| !v.is_finite() || v.to_bits() & 0xffff != 0)
    {
        return Err(Error::Malformed("BF16 PV finite source words".into()));
    }
    let product = |i| {
        if i < probabilities.len() {
            // Source opmath<float> product; no GEMV FMA/reassociation is introduced.
            values[i] * probabilities[i]
        } else {
            0.0 // Source masked future terms; +0 partials make their zero signs immaterial.
        }
    };
    let result = if geometry.query_rows * geometry.contraction * geometry.output_width < 400 {
        // LinearAlgebra::baddbmm_cpu_kernel's small-BMM path.
        let mut sum = 0.0f32;
        for i in 0..geometry.contraction {
            sum += product(i);
        }
        sum
    } else {
        // Low-precision gemm_notrans_: four FP32 partials, then tail in0, then1/2/3.
        let mut partial = [0.0f32; 4];
        let vector_end = geometry.contraction / 4 * 4;
        for start in (0..vector_end).step_by(4) {
            for (lane, sum) in partial.iter_mut().enumerate() {
                *sum += product(start + lane);
            }
        }
        for i in vector_end..geometry.contraction {
            partial[0] += product(i);
        }
        for lane in 1..4 {
            partial[0] += partial[lane];
        }
        partial[0]
    };
    if !result.is_finite() {
        return Err(Error::Malformed("BF16 PV FP32 accumulator overflow".into()));
    }
    // The existing producer performs the single source BF16 result store.
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }
    fn geometry(contraction: usize, width: usize) -> Bf16PvGeometry {
        Bf16PvGeometry {
            query_rows: 1,
            contraction,
            output_width: width,
        }
    }
    #[test]
    fn four_partials_and_original_full_call_contraction_are_distinct_from_serial_gemv() {
        let probabilities = [0.5, 0.125, 0.125, 0.125, 0.0625, 0.0625];
        let values = [
            2.0,
            2.0f32.powi(-21),
            -8.0,
            2.0f32.powi(-21),
            2.0f32.powi(-20),
            2.0f32.powi(-20),
        ];
        let result =
            bf16_pv_f32_accumulator(source(), &probabilities, &values, geometry(6, 512)).unwrap();
        assert_eq!(result.to_bits(), 2.0f32.powi(-24).to_bits());
        let serial = probabilities
            .iter()
            .zip(values)
            .fold(0.0, |sum, (p, v)| sum + p * v);
        assert_eq!(serial.to_bits(), (3.0 * 2.0f32.powi(-24)).to_bits());
        // Same active6 inputs inside a genuine8-row prefill have a different tail assignment.
        let shape = Bf16PvGeometry {
            query_rows: 8,
            contraction: 8,
            output_width: 512,
        };
        assert_eq!(
            bf16_pv_f32_accumulator(source(), &probabilities, &values, shape)
                .unwrap()
                .to_bits(),
            serial.to_bits()
        );
    }
    #[test]
    fn retired_actual_six_term_operand_fixture_matches_source_pv_bf16_word() {
        // Test-only actual softmax/currentV + genuine priorV from healthy v2; never runtime operands.
        let p = [
            0x383b0000, 0x3d170000, 0x3e5a0000, 0x3e770000, 0x3e110000, 0x3ebc0000,
        ]
        .map(f32::from_bits);
        let v = [
            0xbed00000, 0xbf150000, 0x3e4b0000, 0xbf0a0000, 0x3f0c0000, 0x3e600000,
        ]
        .map(f32::from_bits);
        let result = bf16_pv_f32_accumulator(source(), &p, &v, geometry(6, 512)).unwrap();
        assert_eq!(
            super::super::norm::round_bf16(result).unwrap().to_bits(),
            0x3d460000
        );
        let serial = p.iter().zip(v).fold(0.0, |sum, (p, v)| sum + p * v);
        assert_eq!(
            super::super::norm::round_bf16(serial).unwrap().to_bits(),
            0x3d470000
        );
    }
    #[test]
    fn all_finite_context_lengths_and_both_widths_keep_exact_constant_columns() {
        for width in [256, 512] {
            for n in 1..=64 {
                let p = vec![0.125; n];
                let v = vec![-0.5; n];
                assert_eq!(
                    bf16_pv_f32_accumulator(source(), &p, &v, geometry(n, width)).unwrap(),
                    -(n as f32) / 16.0
                );
                let shape = Bf16PvGeometry {
                    query_rows: n,
                    contraction: n,
                    output_width: width,
                };
                assert_eq!(
                    bf16_pv_f32_accumulator(source(), &p, &v, shape).unwrap(),
                    -(n as f32) / 16.0
                );
            }
        }
    }
    #[test]
    fn malformed_precision_extent_probability_and_nonfinite_refuse() {
        for shape in [
            geometry(0, 256),
            geometry(65, 512),
            geometry(1, 128),
            Bf16PvGeometry {
                query_rows: 0,
                contraction: 1,
                output_width: 256,
            },
            Bf16PvGeometry {
                query_rows: 2,
                contraction: 1,
                output_width: 256,
            },
        ] {
            assert!(bf16_pv_f32_accumulator(source(), &[1.0], &[1.0], shape).is_err());
        }
        for p in [f32::NAN, -1.0, 2.0, f32::from_bits(0x3f000001)] {
            assert!(bf16_pv_f32_accumulator(source(), &[p], &[1.0], geometry(1, 256)).is_err());
        }
        assert!(
            bf16_pv_f32_accumulator(source(), &[1.0], &[f32::INFINITY], geometry(1, 512)).is_err()
        );
        assert!(bf16_pv_f32_accumulator(source(), &[1.0, 0.0], &[1.0], geometry(2, 512)).is_err());
        assert!(
            bf16_pv_f32_accumulator(source(), &[1.0, 0.0], &[1.0, 1.0], geometry(1, 512)).is_err()
        );
        assert_eq!(
            bf16_pv_f32_accumulator(source(), &[0.0], &[-0.0], geometry(1, 256))
                .unwrap()
                .to_bits(),
            0
        );
    }
}
