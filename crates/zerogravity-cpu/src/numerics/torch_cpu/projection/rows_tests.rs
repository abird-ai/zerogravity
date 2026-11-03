use super::*;
use crate::accelerator::SourceIdentity;
use zerogravity_core::{
    encoding::{ScalarDType, StoredTensorEncoding},
    source::SourceTensorRef,
};

fn source() -> SourceIdentity {
    SourceIdentity::for_new_allocation().unwrap()
}
fn tensor<'a>(shape: &'a [u64], bytes: &'a [u8]) -> SourceTensorRef<'a> {
    SourceTensorRef {
        kind: StoredTensorEncoding::Scalar(ScalarDType::Bf16),
        shape,
        packed_bytes: bytes,
    }
}
#[test]
fn complete_rows_keep_output_order_and_bf16_precision() {
    let weights = [1.0f32, 2., 3., 4., 5., 6.]
        .into_iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        project_bf16_rows(
            source(),
            tensor(&[2, 3], &weights),
            &[1., 2., 3., 4., 5., 6.],
            2
        )
        .unwrap(),
        [14., 32., 32., 77.]
    );
    for rows in [0, 65, usize::MAX] {
        assert!(project_bf16_rows(source(), tensor(&[2, 3], &weights), &[1.; 6], rows).is_err());
    }
    for input in [
        vec![],
        vec![1.; 5],
        vec![f32::INFINITY; 6],
        vec![f32::from_bits(0x3f800001); 6],
    ] {
        assert!(project_bf16_rows(source(), tensor(&[2, 3], &weights), &input, 2).is_err());
    }
    let mut malformed = weights.clone();
    malformed[..2].copy_from_slice(&0x7f80u16.to_le_bytes());
    assert!(project_bf16_rows(source(), tensor(&[2, 3], &malformed), &[1.; 6], 2).is_err());
    assert!(project_bf16_rows(source(), tensor(&[2, 3], &weights[..10]), &[1.; 6], 2).is_err());
}

#[test]
#[ignore = "requires the independently captured pinned Torch CPU projection archive"]
fn original_linear_rows_are_exact() {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let path = std::env::var("ZG_TORCH_CPU_LINEAR_REFERENCE").expect("original reference path");
    let file = std::fs::File::open(path).unwrap();
    assert_eq!(file.metadata().unwrap().len(), 7_135_232);
    let mut bytes = Vec::new();
    file.take(7_135_233).read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes.len(), 7_135_232);
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        "87060268f5fd936e17218ca33d8a1860a366e408f52ddf7e64c0537989c0611d"
    );
    let mut at = 0;
    let mut take = |n: usize| {
        let value = &bytes[at..at + n];
        at += n;
        value
    };
    assert_eq!(take(8), b"ZGLIN001");
    assert_eq!(u64::from_le_bytes(take(8).try_into().unwrap()), 49);
    let mut words = 0;
    for case in 0..49 {
        let [rows, outputs, columns] = std::array::from_fn(|_| {
            usize::try_from(u64::from_le_bytes(take(8).try_into().unwrap())).unwrap()
        });
        assert!((1..=64).contains(&rows) && outputs <= 33 && columns <= 6144);
        let decode = |raw: &[u8]| {
            raw.chunks_exact(2)
                .map(|v| f32::from_bits(u32::from(u16::from_le_bytes(v.try_into().unwrap())) << 16))
                .collect::<Vec<_>>()
        };
        let input = decode(take(rows * columns * 2));
        let weight = take(outputs * columns * 2);
        let expected = decode(take(rows * outputs * 2));
        let shape = [outputs as u64, columns as u64];
        let actual = project_bf16_rows(source(), tensor(&shape, weight), &input, rows).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (word, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "case={case} word={word}");
        }
        words += actual.len();
    }
    assert_eq!(at, bytes.len());
    eprintln!("SOURCE_LINEAR_CPU_EXACT cases=49 words={words} trained_model=false gpu=false");
}
