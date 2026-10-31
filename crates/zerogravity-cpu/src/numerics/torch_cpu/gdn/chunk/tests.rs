use super::*;

mod core;

fn source() -> SourceIdentity {
    SourceIdentity::for_new_allocation().unwrap()
}

#[test]
fn source_solver_retains_product_rounding_before_rhs_subtraction() {
    // Original19-case archive, first RHS of17 active rows, head0/column18.
    let mut a = vec![0.0; ORDER * ORDER];
    let mut b = vec![0.0; ORDER * WIDTH];
    a[ORDER] = f32::from_bits(0xbd4e08a7);
    b[18] = f32::from_bits(0xbf52ee00);
    b[WIDTH + 18] = f32::from_bits(0x3d29aa00);
    let x = solve_unit_lower(source(), &a, &b).unwrap();
    assert_eq!(x[WIDTH + 18].to_bits(), 0xb7c52800);
    assert_ne!(
        (-a[ORDER]).mul_add(b[18], b[WIDTH + 18]).to_bits(),
        x[WIDTH + 18].to_bits()
    );
}

#[test]
fn source_solver_preserves_four_lane_tree_and_independent_columns() {
    // Original same solve, head0/column1; source outputs, not a Rust oracle.
    let packed = [
        0xbd4e08a7, 0xbcb56981, 0x3ce6918d, 0xbd14e46e, 0x3d1b1dcd, 0x3d8aaf50, 0x3cdd976a,
        0x3d06615a, 0xbd37b2a6, 0xbd0506f4,
    ];
    let input = [0x3d285400, 0x3ddb4600, 0x3fa7ac00, 0x3cdc7800, 0xbd11b100];
    let expected = [0x3d285400, 0x3ddf81ca, 0x3fa7652b, 0xbd83a7fb, 0x3c858c16];
    let mut a = vec![0.0; ORDER * ORDER];
    let mut at = 0;
    for row in 0..5 {
        for column in 0..row {
            a[row * ORDER + column] = f32::from_bits(packed[at]);
            at += 1;
        }
    }
    let mut b = vec![0.0; ORDER * WIDTH];
    for (row, value) in input.into_iter().enumerate() {
        b[row * WIDTH + 1] = f32::from_bits(value);
    }
    let x = solve_unit_lower(source(), &a, &b).unwrap();
    for (row, value) in expected.into_iter().enumerate() {
        assert_eq!(x[row * WIDTH + 1].to_bits(), value);
    }
    let sequential = (0..4).fold(0.0, |sum, k| sum + a[4 * ORDER + k] * x[k * WIDTH + 1]);
    assert_ne!((b[4 * WIDTH + 1] - sequential).to_bits(), expected[4]);
    for (i, value) in x.iter().enumerate() {
        if i % WIDTH != 1 {
            assert_eq!(value.to_bits(), 0);
        }
    }
    assert_eq!(x.capacity(), b.len());
}

#[test]
fn unit_diagonal_upper_values_and_systems_do_not_mix() {
    let mut a = vec![0.0; 2 * ORDER * ORDER];
    let mut b = vec![0.0f32; 2 * ORDER * WIDTH];
    for matrix in a.chunks_exact_mut(ORDER * ORDER) {
        for row in 0..ORDER {
            for column in row..ORDER {
                matrix[row * ORDER + column] = 123.0;
            }
        }
    }
    for (i, value) in b.iter_mut().enumerate() {
        *value = if i < ORDER * WIDTH { -0.0 } else { 3.5 };
    }
    let before = b.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let x = solve_unit_lower(source(), &a, &b).unwrap();
    assert_eq!(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), before);
    assert_eq!(b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), before);
}

#[test]
fn source_solver_refuses_invalid_extents_nonfinite_and_overflow() {
    let source = source();
    let a = vec![0.0; ORDER * ORDER];
    let b = vec![0.0; ORDER * WIDTH];
    assert!(solve_unit_lower(SourceIdentity::unbound(), &a, &b).is_err());
    for (lhs, rhs) in [
        (&[][..], &[][..]),
        (&a[..a.len() - 1], &b),
        (&a, &b[..b.len() - 1]),
    ] {
        assert!(solve_unit_lower(source, lhs, rhs).is_err());
    }
    assert!(solve_unit_lower(source, &vec![0.0; (MAX_SYSTEMS + 1) * ORDER * ORDER], &[]).is_err());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut malformed = a.clone();
        malformed[0] = bad;
        assert!(solve_unit_lower(source, &malformed, &b).is_err());
        let mut malformed = b.clone();
        malformed[0] = bad;
        assert!(solve_unit_lower(source, &a, &malformed).is_err());
    }
    let mut a = a;
    let mut b = b;
    a[ORDER] = f32::MAX;
    b[0] = f32::MAX;
    assert!(solve_unit_lower(source, &a, &b).is_err());
    assert_eq!(a[ORDER], f32::MAX);
    assert_eq!(b[0], f32::MAX);
}

#[test]
#[ignore = "requires independently captured original Torch GDN solver archive"]
fn original_triangular_archive_matches_every_word() {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    const BYTES: u64 = 70_779_348;
    let path = std::env::var("ZG_TORCH_CPU_GDN_TRIANGULAR_REFERENCE").unwrap();
    let mut file = std::fs::File::open(path).unwrap();
    assert_eq!(file.metadata().unwrap().len(), BYTES);
    let mut digest = Sha256::new();
    let mut read = 0u64;
    let mut take = |size: usize| {
        assert!(size <= 48 * ORDER * WIDTH * 4);
        read = read.checked_add(size as u64).unwrap();
        assert!(read <= BYTES);
        let mut bytes = vec![0u8; size];
        file.read_exact(&mut bytes).unwrap();
        digest.update(&bytes);
        bytes
    };
    assert_eq!(take(8), b"ZGTS0001");
    assert_eq!(u32::from_le_bytes(take(4).try_into().unwrap()), 38);
    let mut words = 0;
    let source = source();
    for case in 0..38 {
        let dimensions = std::array::from_fn::<_, 3, _>(|_| {
            u32::from_le_bytes(take(4).try_into().unwrap()) as usize
        });
        let [systems, order, width] = dimensions;
        assert!(matches!(systems, 16 | 48));
        assert_eq!((order, width), (ORDER, WIDTH));
        let mut values = |n: usize| {
            take(n * 4)
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let a = values(systems * order * order);
        let b = values(systems * order * width);
        let expected = values(systems * order * width);
        let actual = solve_unit_lower(source, &a, &b).unwrap();
        for (word, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "case{case} word{word}"
            );
        }
        words += actual.len();
    }
    assert_eq!(read, BYTES);
    assert_eq!(file.read(&mut [0u8; 1]).unwrap(), 0);
    assert_eq!(
        format!("{:x}", digest.finalize()),
        "96148b96d17dd67056f6dea8195e859ce3eccae77fa4b95e0957fe77692b01d0"
    );
    assert_eq!(words, 7_077_888);
    println!("ORIGINAL_GDN_TRIANGULAR_PASS solves=38 words={words} exact=true");
}
