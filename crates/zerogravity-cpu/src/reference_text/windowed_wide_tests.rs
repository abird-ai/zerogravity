//! Actual tiny BF16 files exercise the dense31B Down row width, not its weights.
use super::*;

fn source(fixture: &Fixture, columns: usize) -> WindowReferenceTextSource {
    let count = 2 * columns * 2;
    let header = serde_json::to_vec(
        &serde_json::json!({"wide":{"dtype":"BF16","shape":[2,columns],"data_offsets":[0,count]}}),
    )
    .unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    for row in 0..2 {
        for col in 0..columns {
            let value = if row == 0 && (col == 0 || col == columns - 1) {
                1.0f32
            } else if row == 1 && col == 17000 {
                2.0
            } else {
                0.0
            };
            bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
    }
    let basename = "wide.safetensors";
    let path = fixture.path.join(basename);
    fs::write(&path, &bytes).unwrap();
    let shard = ExpectedSourceShard::new(
        basename.into(),
        Artifact::from_file(File::open(path).unwrap(), Path::new(basename)).unwrap(),
        bytes.len() as u64,
        format!("{:x}", Sha256::digest(&bytes)),
    )
    .unwrap();
    let index = serde_json::to_vec(
        &serde_json::json!({"metadata":{"total_size":count},"weight_map":{"wide":basename}}),
    )
    .unwrap();
    let plan = SourceWeights::prepare_verified_windows(
        &index,
        &format!("{:x}", Sha256::digest(&index)),
        vec![shard],
        WindowOpenLimits {
            max_source_bytes: 1024 * 1024,
            max_metadata_bytes: 16 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 1,
            scan_chunk_bytes: 65536,
        },
        &fixture.admission,
    )
    .unwrap();
    WindowReferenceTextSource::new(plan.open().unwrap())
}

#[test]
fn genuine21504_column_windows_compute_tail_words_refuse_pressure_and_retain_source() {
    let fixture = Fixture::new();
    let source = source(&fixture, 21504);
    let descriptor = source.matrix("wide").unwrap();
    let baseline = fixture.reserved();
    let mut input = vec![0.0; 21504];
    input[0] = 3.0;
    input[17000] = 7.0;
    input[21503] = 5.0;
    assert_eq!(
        source.project(&descriptor, &input, None).unwrap(),
        [8.0, 14.0]
    );
    assert_eq!(fixture.reserved(), baseline);
    let cancel = AtomicBool::new(true);
    assert!(source.project(&descriptor, &input, Some(&cancel)).is_err());
    let first = source.source().reserve_workspace(32 * 1024 * 1024).unwrap();
    let remaining = fixture
        .admission
        .arbiter
        .snapshot(&fixture.admission.resource)
        .unwrap()
        .remaining as usize;
    let rest = source.source().reserve_workspace(remaining).unwrap();
    assert!(matches!(
        source.project(&descriptor, &input, None),
        Err(Error::Limit(_))
    ));
    drop(rest);
    drop(first);
    assert_eq!(fixture.reserved(), baseline);
    assert_eq!(
        source.project(&descriptor, &input, None).unwrap(),
        [8.0, 14.0]
    );
    let matrix = source
        .source()
        .matrix("wide", MatrixAxes::OutputInput)
        .unwrap();
    let retained = source.source().read_rows(&matrix, 1..2).unwrap();
    drop(matrix);
    drop(source);
    assert!(fixture.reserved() > 0);
    assert_eq!(retained.as_slice().len(), 43008);
    assert_eq!(
        &retained.as_slice()[34000..34002],
        &2.0f32.to_bits().to_le_bytes()[2..]
    );
    drop(retained);
    assert_eq!(fixture.reserved(), 0);
}

#[test]
fn matrix_width_above_dense31_bound_still_refuses_before_math() {
    let fixture = Fixture::new();
    let source = source(&fixture, 21505);
    assert!(matches!(source.matrix("wide"), Err(Error::Unsupported(_))));
    drop(source);
    assert_eq!(fixture.reserved(), 0);
}
