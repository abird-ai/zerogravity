use super::*;

struct Words<'a>(&'a [u8]);

impl<'a> Words<'a> {
    fn take(&mut self, count: usize) -> &'a [u8] {
        let (value, rest) = self.0.split_at(count);
        self.0 = rest;
        value
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    fn blob(&mut self) -> &'a [u8] {
        let count = self.u32() as usize;
        self.take(count)
    }
}

fn source() -> SourceIdentity {
    SourceIdentity::for_new_allocation().unwrap()
}

fn input_words(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16))
        .collect()
}

fn synthetic(width: usize, rows: usize, kind: QuantKind, mode: u32) -> (Vec<f32>, Vec<u8>) {
    let input = (0..rows)
        .flat_map(|row| {
            (0..width).map(move |index| {
                let word = match mode {
                    1 => {
                        if (index + row) % 2 == 0 {
                            0
                        } else {
                            0x8000
                        }
                    }
                    2 => {
                        if index == 0 {
                            0x4300
                        } else if index == 1 || index == 513.min(width - 1) {
                            0x3d00
                        } else {
                            0
                        }
                    }
                    3 => {
                        let magnitude = 0x3a00 + ((index * 73 + row * 17) % 1024);
                        magnitude | if (index + row) % 3 == 0 { 0 } else { 0x8000 }
                    }
                    4 => {
                        let magnitude = if index % 37 == 0 { 0x4200 } else { 0x3a00 };
                        magnitude | if (index + row) % 3 == 0 { 0 } else { 0x8000 }
                    }
                    _ => panic!("golden pattern"),
                };
                f32::from_bits((word as u32) << 16)
            })
        })
        .collect();
    // Raw public-oracle input recipe; no runtime gamma adjustment here.
    let weight = (0..width)
        .flat_map(|index| {
            let word = [0xbf80_0000u32, 0xbf00_0000, 0, 0x3e80_0000, 0x3f80_0347][index % 5];
            if kind == QuantKind::Bf16 {
                ((word >> 16) as u16).to_le_bytes().to_vec()
            } else {
                word.to_le_bytes().to_vec()
            }
        })
        .collect();
    (input, weight)
}

fn stages(result: &OrdinaryRms) -> [&[f32]; 11] {
    [
        result.input_f32(),
        result.squares_f32(),
        result.mean_f32(),
        result.denominator_f32(),
        result.inverse_f32(),
        result.normalized_f32(),
        result.weight_f32(),
        result.runtime_gamma_f32(),
        result.weighted_f32(),
        result.output_bf16(),
        result.sums_f32(),
    ]
}

// Genuine pinned original source36F32 .8B Q-norm head0 for tokens11,353,
// plus32 original public forward fixtures at256/1024/2048/5120, both raw
// weight dtypes and4 input patterns. Full actual122×10 stages replay stays
// in the ignored run. Expected words come directly from authentic outputs;
// source-free input recipes above never compute their expected arithmetic.
// Fixture SHAe73c1ad593f9927388e66d40037ca687bdc398169f72f9c4ddd31a7e639c7583.
#[test]
fn pinned_actual_and_public_goldens_cover_all_stages_and_cascade_widths() {
    let mut data = Words(include_bytes!("golden.bin"));
    assert_eq!(data.take(8), b"ORMST001");
    let cases = data.u32();
    assert_eq!(cases, 34);
    let source = source();
    for _ in 0..cases {
        let name = std::str::from_utf8(data.blob()).unwrap();
        let width = data.u32() as usize;
        let rows = data.u32() as usize;
        let kind = match data.u32() {
            0 => QuantKind::Bf16,
            1 => QuantKind::F32,
            _ => panic!("golden weight kind"),
        };
        let mode = data.u32();
        let (input, weight) = if mode == 0 {
            (input_words(data.blob()), data.blob().to_vec())
        } else {
            synthetic(width, rows, kind, mode)
        };
        let original_weight = weight.clone();
        let shape = [width as u64];
        let result = ordinary_rms(
            source,
            SourceTensorRef {
                kind,
                shape: &shape,
                packed_bytes: &weight,
            },
            &input,
            width,
        )
        .unwrap();
        assert_eq!(result.source_identity(), source);
        assert_eq!(result.width(), width);
        assert_eq!(result.rows(), rows);
        for (field, values) in stages(&result).into_iter().enumerate() {
            let samples = data.u32();
            for _ in 0..samples {
                let index = data.u32() as usize;
                let expected = data.u32();
                assert_eq!(
                    values[index].to_bits(),
                    expected,
                    "{name} stage{field} word{index}"
                );
            }
        }
        assert_eq!(weight, original_weight, "raw learned bytes changed");
    }
    assert!(data.0.is_empty());
}

#[test]
fn malformed_source_shape_dtype_and_values_refuse_without_source_mutation() {
    let source = source();
    let width = 256;
    let input = vec![1.0; width];
    let weight = vec![0u8; width * 2];
    let shape = [width as u64];
    let raw = SourceTensorRef {
        kind: QuantKind::Bf16,
        shape: &shape,
        packed_bytes: &weight,
    };
    assert!(ordinary_rms(SourceIdentity::unbound(), raw, &input, width).is_err());
    for width in [0, 1, 128, 512, 4096, usize::MAX] {
        assert!(matches!(
            ordinary_rms(source, raw, &input, width),
            Err(Error::Unsupported(_))
        ));
    }
    for values in [&[][..], &input[..255]] {
        assert!(ordinary_rms(source, raw, values, width).is_err());
    }
    for shape in [&[][..], &[1, 256][..], &[255][..], &[u64::MAX][..]] {
        assert!(ordinary_rms(source, SourceTensorRef { shape, ..raw }, &input, width).is_err());
    }
    for bytes in [&weight[..511], &[0u8; 513][..]] {
        assert!(
            ordinary_rms(
                source,
                SourceTensorRef {
                    packed_bytes: bytes,
                    ..raw
                },
                &input,
                width
            )
            .is_err()
        );
    }
    for kind in [QuantKind::F16, QuantKind::Q4_0, QuantKind::Q8_0] {
        assert!(matches!(
            ordinary_rms(source, SourceTensorRef { kind, ..raw }, &input, width),
            Err(Error::Unsupported(_))
        ));
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0001] {
        let mut values = input.clone();
        values[127] = invalid;
        assert!(ordinary_rms(source, raw, &values, width).is_err());
    }
    let too_many = vec![0.0; MAX_ELEMENTS + width];
    assert!(matches!(
        ordinary_rms(source, raw, &too_many, width),
        Err(Error::Limit(_))
    ));
    for word in [0x7fc0u16, 0x7f80, 0xff80] {
        let mut bytes = weight.clone();
        bytes[2..4].copy_from_slice(&word.to_le_bytes());
        assert!(
            ordinary_rms(
                source,
                SourceTensorRef {
                    packed_bytes: &bytes,
                    ..raw
                },
                &input,
                width
            )
            .is_err()
        );
    }
    for word in [0x7fc0_0000u32, 0x7f80_0000, 0xff80_0000] {
        let mut bytes = vec![0u8; width * 4];
        bytes[4..8].copy_from_slice(&word.to_le_bytes());
        let f32_weight = SourceTensorRef {
            kind: QuantKind::F32,
            packed_bytes: &bytes,
            ..raw
        };
        assert!(ordinary_rms(source, f32_weight, &input, width).is_err());
    }
    let large = vec![f32::from_bits(0x7f7f_0000); width];
    assert!(ordinary_rms(source, raw, &large, width).is_err());
    assert!(weight.iter().all(|value| *value == 0));
}
