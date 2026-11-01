use super::dequantize;
use crate::Error;
use zerogravity_core::encoding::{QuantScheme, ScalarDType, StoredTensorEncoding};

#[test]
fn scalar_decode_preserves_special_value_bits() {
    let f32_words = [0u32, 0x8000_0000, 1, 0x7f80_0000, 0xff80_0000, 0x7fc1_2345];
    let f32_bytes: Vec<_> = f32_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let decoded = dequantize(
        StoredTensorEncoding::Scalar(ScalarDType::F32),
        &f32_bytes,
        f32_words.len(),
    )
    .unwrap();
    assert_eq!(
        decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        f32_words
    );

    let bf16_words = [0u16, 0x8000, 1, 0x7f80, 0xff80, 0x7fc1, 0xff81];
    let bf16_bytes: Vec<_> = bf16_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let decoded = dequantize(
        StoredTensorEncoding::Scalar(ScalarDType::Bf16),
        &bf16_bytes,
        bf16_words.len(),
    )
    .unwrap();
    assert_eq!(
        decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        bf16_words.map(|word| u32::from(word) << 16)
    );

    let f16_words = [0u16, 0x8000, 1, 0x7c00, 0xfc00, 0x7e01, 0xfc01];
    let f16_bytes: Vec<_> = f16_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let decoded = dequantize(
        StoredTensorEncoding::Scalar(ScalarDType::F16),
        &f16_bytes,
        f16_words.len(),
    )
    .unwrap();
    assert_eq!(
        decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [
            0,
            0x8000_0000,
            0x3380_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_2000,
            0xff80_2000
        ]
    );
}

#[test]
fn q4_1_offset_and_q6_k_signed_scales_keep_source_lane_order() {
    let mut q4 = [0xf2; 20];
    q4[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5
    q4[2..4].copy_from_slice(&0xbc00u16.to_le_bytes()); // -1
    let decoded = dequantize(StoredTensorEncoding::Quantized(QuantScheme::Q4_1), &q4, 32).unwrap();
    assert_eq!(&decoded[..16], &[0.0; 16]);
    assert_eq!(&decoded[16..], &[6.5; 16]);

    let mut q6 = [0u8; 210];
    q6[..128].fill(0x21);
    q6[128..192].fill(0xe4);
    for half in 0..2 {
        q6[192 + half * 8..200 + half * 8].copy_from_slice(&[1, 255, 2, 254, 3, 253, 4, 252]);
    }
    q6[208..].copy_from_slice(&0x3c00u16.to_le_bytes()); // 1
    let decoded = dequantize(StoredTensorEncoding::Quantized(QuantScheme::Q6K), &q6, 256).unwrap();
    let expected = [-31.0, 31.0, -30.0, 30.0, 6.0, -6.0, 72.0, -72.0];
    for half in decoded.chunks_exact(128) {
        for (lane, expected) in half.chunks_exact(16).zip(expected) {
            assert_eq!(lane, &[expected; 16]);
        }
    }
}

#[test]
fn every_codec_rejects_truncated_and_excess_bytes() {
    for encoding in [
        StoredTensorEncoding::Scalar(ScalarDType::F32),
        StoredTensorEncoding::Scalar(ScalarDType::F16),
        StoredTensorEncoding::Scalar(ScalarDType::Bf16),
        StoredTensorEncoding::Quantized(QuantScheme::Q4_0),
        StoredTensorEncoding::Quantized(QuantScheme::Q4_1),
        StoredTensorEncoding::Quantized(QuantScheme::Q4K),
        StoredTensorEncoding::Quantized(QuantScheme::Q6K),
        StoredTensorEncoding::Quantized(QuantScheme::Q8_0),
    ] {
        let (elements, bytes) = encoding.block();
        for length in [0, bytes - 1, bytes + 1] {
            assert!(matches!(
                dequantize(encoding, &vec![0; length as usize], elements as usize),
                Err(Error::Malformed(_))
            ));
        }
        assert_eq!(
            dequantize(encoding, &vec![0; bytes as usize], elements as usize)
                .unwrap()
                .len(),
            elements as usize
        );
        assert!(dequantize(encoding, &[], 0).unwrap().is_empty());
    }
}

#[test]
fn every_quantized_scale_and_offset_rejects_nonfinite_values() {
    for (scheme, offsets) in [
        (QuantScheme::Q4_0, &[0][..]),
        (QuantScheme::Q4_1, &[0, 2][..]),
        (QuantScheme::Q4K, &[0, 2][..]),
        (QuantScheme::Q6K, &[208][..]),
        (QuantScheme::Q8_0, &[0][..]),
    ] {
        let encoding = StoredTensorEncoding::Quantized(scheme);
        let (elements, bytes) = encoding.block();
        for &offset in offsets {
            for bits in [0x7c00u16, 0xfc00, 0x7e01] {
                let mut data = vec![0; bytes as usize];
                data[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
                assert!(matches!(
                    dequantize(encoding, &data, elements as usize),
                    Err(Error::Malformed(_))
                ));
            }
        }
    }
}
