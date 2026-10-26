//! Synthetic authenticated files exercise transport/identity, not trained27B.
use super::*;
use crate::source::projection::output_first_projection_with_profile;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use zerogravity_artifacts::artifact::Artifact;
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_artifacts::source_weights::{
    ExpectedSourceShard, SourceOpenLimits, SourceWeights, VerifiedMatrix, VerifiedTensorSource,
    WindowOpenLimits,
};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
use zerogravity_core::source::TensorSource;

const NAME: &str = "original.weight";
const FILE: &str = "model.safetensors";
const BYTES: &[u8] = include_bytes!("fixture.safetensors");
const INDEX: &[u8] = include_bytes!("fixture.index.json");
const FILE_SHA: &str = "973fca86c0585f6b11d0b686c051bad06b896f13b2b3d10ef0299eea4510f033";
const INDEX_SHA: &str = "42e2f4619abb5431bccce39b473b23eaa90679c9947d26c9d439308c78a86149";
const META: u64 = 16 * 1024 * 1024;

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let directory = std::env::temp_dir().join(format!(
            "zg-cpu-window-projection-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let fixture = Self(directory);
        fs::write(fixture.0.join(FILE), BYTES).unwrap();
        fixture
    }
    fn shards(&self) -> Vec<ExpectedSourceShard> {
        vec![
            ExpectedSourceShard::new(
                FILE.into(),
                Artifact::from_file(File::open(self.0.join(FILE)).unwrap(), Path::new(FILE))
                    .unwrap(),
                BYTES.len() as u64,
                FILE_SHA.into(),
            )
            .unwrap(),
        ]
    }
    fn open(&self, host: &TransferAdmission) -> VerifiedTensorSource {
        self.open_with_chunk(host, 64)
    }
    fn open_with_chunk(
        &self,
        host: &TransferAdmission,
        scan_chunk_bytes: usize,
    ) -> VerifiedTensorSource {
        SourceWeights::prepare_verified_windows(
            INDEX,
            INDEX_SHA,
            self.shards(),
            WindowOpenLimits {
                max_source_bytes: 1024 * 1024,
                max_metadata_bytes: META,
                max_shards: 1,
                max_tensors: 6,
                scan_chunk_bytes,
            },
            host,
        )
        .unwrap()
        .open()
        .unwrap()
    }
    fn complete_reference(&self, host: &TransferAdmission, name: &str, input: &[f32]) -> Vec<f32> {
        let full = SourceWeights::prepare_verified_shards(
            INDEX,
            INDEX_SHA,
            self.shards(),
            SourceOpenLimits {
                max_owned_bytes: 1024 * 1024,
                max_metadata_bytes: META,
                max_shards: 1,
                max_tensors: 6,
            },
            host,
        )
        .unwrap()
        .materialize()
        .unwrap();
        // The real complete source supplies this full reference; no fabricated
        // reduced header or missing-payload TensorSource is constructed.
        output_first_projection_with_profile(
            full.source_identity(),
            name,
            TensorSource::tensor(&full, name).unwrap(),
            input,
            None,
            SourceProjectionProfile::TorchCpuBf16Avx2,
        )
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn host() -> TransferAdmission {
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("host", "synthetic-cpu-window-projection").unwrap();
    arbiter
        .register(resource.clone(), 64 * 1024 * 1024)
        .unwrap();
    TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("synthetic-original", "window-projection").unwrap(),
    }
}
fn reserved(host: &TransferAdmission) -> u64 {
    host.arbiter.snapshot(&host.resource).unwrap().reserved
}
fn descriptor(matrix: &VerifiedMatrix) -> SourceTensor {
    SourceTensor {
        source: matrix.source_identity(),
        name: matrix.tensor().name.clone(),
        kind: matrix.tensor().kind,
        axes: matrix.axes(),
        columns: matrix.columns(),
        rows: matrix.rows(),
    }
}
fn input() -> Vec<f32> {
    let mut values = vec![0.5; 65];
    values[0] = 1.0;
    values[1] = f32::from_bits(0x3a80_0000); // Exact2^-10, not a fitted output.
    values[64] = 1.5;
    values
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn row_partitions_match_complete_kernel_and_retain_full_unrounded_identity() {
    let fixture = Fixture::new();
    let host = host();
    let source = fixture.open(&host);
    let matrix = source.matrix(NAME, MatrixAxes::OutputInput).unwrap();
    let expected = descriptor(&matrix);
    let input = input();
    let full = fixture.complete_reference(&host, NAME, &input);
    assert_eq!(reserved(&host), META);
    let mut assembled = Vec::new();
    for rows in [0..1, 1..3, 3..4] {
        let window = source.read_rows(&matrix, rows.clone()).unwrap();
        let charge = META + window.charged_staging_bytes() as u64;
        assert_eq!(reserved(&host), charge);
        let projected =
            project_bf16_window(&expected, rows.clone(), &window, &input, None, None).unwrap();
        assert_eq!(bits(projected.accumulations()), bits(&full[rows.clone()]));
        let identity = projected.identity();
        assert_eq!(identity.descriptor(), &expected);
        assert_eq!(identity.descriptor().rows, 4);
        assert_eq!(identity.original_shape(), [4, 65]);
        assert_eq!(identity.original_byte_offset(), matrix.tensor().byte_offset);
        assert_eq!(identity.original_byte_len(), 4 * 65 * 2);
        assert_eq!(identity.row_range(), rows.clone());
        assert_eq!(
            identity.source_offset(),
            matrix.tensor().byte_offset + rows.start as u64 * 65 * 2
        );
        assert_eq!(identity.window_bytes(), rows.len() * 65 * 2);
        assert_eq!(
            identity.profile(),
            SourceProjectionProfile::TorchCpuBf16Avx2
        );
        assert_eq!(reserved(&host), charge);
        let (identity, values) = projected.into_parts();
        assert_eq!(identity.descriptor().source, expected.source);
        assembled.extend(values);
        drop(window);
        assert_eq!(reserved(&host), META);
    }
    assert_eq!(bits(&assembled), bits(&full));
    // Row1 is exactly1+2^-10. The wrapper must not round this to BF16.
    assert_eq!(assembled[1].to_bits(), 0x3f80_2000);
    drop((matrix, source));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn original_width5120_uses_all_columns_without_padding_or_truncation() {
    let fixture = Fixture::new();
    let host = host();
    let source = fixture.open_with_chunk(&host, 1024);
    let matrix = source
        .matrix("wide.weight", MatrixAxes::OutputInput)
        .unwrap();
    let expected = descriptor(&matrix);
    let mut input = vec![0.0; 5120];
    input[0] = 1.0;
    input[1] = f32::from_bits(0x3a80_0000);
    input[5119] = -1.0;
    let full = fixture.complete_reference(&host, "wide.weight", &input);
    let window = source.read_rows(&matrix, 1..3).unwrap();
    let projected = project_bf16_window(&expected, 1..3, &window, &input, None, None).unwrap();
    assert_eq!(bits(projected.accumulations()), bits(&full[1..3]));
    assert_eq!(bits(projected.accumulations()), [0x3a80_0000, 0xbf80_0000]);
    assert_eq!(projected.identity().original_shape(), [3, 5120]);
    assert!(project_bf16_window(&expected, 1..3, &window, &input[..5119], None, None).is_err());
    drop((window, matrix, source));
    assert_eq!(reserved(&host), 0);
    // Private result metadata does not pretend to retain a source/staging lease.
    assert_eq!(projected.identity().descriptor(), &expected);
}

#[test]
fn sealed_window_keeps_original_owner_and_stage_until_synchronous_math_finishes() {
    let fixture = Fixture::new();
    let host = host();
    let source = fixture.open(&host);
    let matrix = source.matrix(NAME, MatrixAxes::OutputInput).unwrap();
    let expected = descriptor(&matrix);
    let window = source.read_rows(&matrix, 0..2).unwrap();
    let charge = META + window.charged_staging_bytes() as u64;
    drop((matrix, source));
    assert_eq!(reserved(&host), charge);
    let output = project_bf16_window(&expected, 0..2, &window, &input(), None, None).unwrap();
    assert_eq!(bits(output.accumulations()), [0x3f80_0000, 0x3f80_2000]);
    assert_eq!(reserved(&host), charge);
    drop(window);
    assert_eq!(reserved(&host), 0);
    assert_eq!(output.identity().descriptor().rows, 4);
}

#[test]
fn source_name_axes_dtype_full_geometry_and_exact_requested_range_refuse() {
    let fixture = Fixture::new();
    let host = host();
    let source = fixture.open(&host);
    let foreign = fixture.open(&host);
    let matrix = source.matrix(NAME, MatrixAxes::OutputInput).unwrap();
    let foreign_matrix = foreign.matrix(NAME, MatrixAxes::OutputInput).unwrap();
    let expected = descriptor(&matrix);
    let window = source.read_rows(&matrix, 1..3).unwrap();
    let mut invalid = Vec::new();
    let mut changed = expected.clone();
    changed.source = crate::accelerator::SourceIdentity::unbound();
    invalid.push(changed);
    invalid.push(descriptor(&foreign_matrix));
    let mut changed = expected.clone();
    changed.name = "other.weight".into();
    invalid.push(changed);
    let mut changed = expected.clone();
    changed.name.clear();
    invalid.push(changed);
    let mut changed = expected.clone();
    changed.axes = MatrixAxes::InputOutput;
    invalid.push(changed);
    let mut changed = expected.clone();
    changed.kind = QuantKind::F32;
    invalid.push(changed);
    for rows in [0, 2, 10240, 1_000_001, usize::MAX] {
        let mut changed = expected.clone();
        changed.rows = rows;
        invalid.push(changed);
    }
    for columns in [0, 64, 5120, usize::MAX] {
        let mut changed = expected.clone();
        changed.columns = columns;
        invalid.push(changed);
    }
    for changed in invalid {
        assert!(project_bf16_window(&changed, 1..3, &window, &input(), None, None).is_err());
    }
    for (start, end) in [
        (0, 2),
        (1, 2),
        (1, 1),
        (2, 1),
        (1, 5),
        (usize::MAX, usize::MAX),
    ] {
        assert!(project_bf16_window(&expected, start..end, &window, &input(), None, None).is_err());
    }
    assert!(
        source
            .matrix("vector.weight", MatrixAxes::OutputInput)
            .is_err()
    );
    let float_matrix = source
        .matrix("float.weight", MatrixAxes::OutputInput)
        .unwrap();
    let float_window = source.read_rows(&float_matrix, 0..1).unwrap();
    assert!(matches!(
        project_bf16_window(
            &descriptor(&float_matrix),
            0..1,
            &float_window,
            &input(),
            None,
            None
        ),
        Err(Error::Unsupported(_))
    ));
    drop((
        window,
        matrix,
        source,
        foreign_matrix,
        foreign,
        float_window,
        float_matrix,
    ));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn nonfinite_weights_input_profile_and_output_fail_without_exposing_partial_rows() {
    let fixture = Fixture::new();
    let host = host();
    let source = fixture.open(&host);
    let matrix = source.matrix(NAME, MatrixAxes::OutputInput).unwrap();
    let expected = descriptor(&matrix);
    let window = source.read_rows(&matrix, 0..2).unwrap();
    for invalid in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x3f80_0001),
    ] {
        let mut values = input();
        values[64] = invalid;
        assert!(matches!(
            project_bf16_window(&expected, 0..2, &window, &values, None, None),
            Err(Error::Malformed(_))
        ));
    }
    let mut overflowing = input();
    overflowing[..2].fill(f32::from_bits(0x7f7f_0000));
    assert!(matches!(
        project_bf16_window(&expected, 0..2, &window, &overflowing, None, None),
        Err(Error::Malformed(message)) if message.contains("nonfinite original window projection output")
    ));
    let nonfinite_matrix = source
        .matrix("nonfinite.weight", MatrixAxes::OutputInput)
        .unwrap();
    for row in 0..3 {
        let nonfinite_window = source.read_rows(&nonfinite_matrix, row..row + 1).unwrap();
        assert!(matches!(
            project_bf16_window(&descriptor(&nonfinite_matrix), row..row + 1, &nonfinite_window, &input(), None, None),
            Err(Error::Malformed(message)) if message.contains("nonfinite BF16 source weights")
        ));
    }
    drop((window, matrix, nonfinite_matrix, source));
    assert_eq!(reserved(&host), 0);
}

struct UnawareProvider(AtomicBool);
impl MatVecKernel for UnawareProvider {
    fn try_execute(
        &self,
        _: crate::accelerator::MatVecRegion<'_>,
    ) -> Result<crate::accelerator::RegionExecution> {
        self.0.store(true, Ordering::Release);
        Err(Error::State("unaware provider must not be invoked".into()))
    }
}

#[test]
fn unaware_provider_and_pre_cancel_refuse_without_callback_or_stage_release() {
    let fixture = Fixture::new();
    let host = host();
    let source = fixture.open(&host);
    let matrix = source.matrix(NAME, MatrixAxes::OutputInput).unwrap();
    let expected = descriptor(&matrix);
    let window = source.read_rows(&matrix, 0..2).unwrap();
    let charge = META + window.charged_staging_bytes() as u64;
    let provider = UnawareProvider(AtomicBool::new(false));
    assert!(matches!(
        project_bf16_window(&expected, 0..2, &window, &input(), Some(&provider), None),
        Err(Error::Unsupported(_))
    ));
    assert!(!provider.0.load(Ordering::Acquire));
    assert_eq!(reserved(&host), charge);
    let cancelled = AtomicBool::new(true);
    assert!(matches!(
        project_bf16_window(&expected, 0..2, &window, &input(), None, Some(&cancelled)),
        Err(Error::State(message)) if message.contains("cancelled")
    ));
    assert_eq!(reserved(&host), charge);
    cancelled.store(false, Ordering::Release);
    let accepted =
        project_bf16_window(&expected, 0..2, &window, &input(), None, Some(&cancelled)).unwrap();
    cancelled.store(true, Ordering::Release);
    assert!(
        project_bf16_window(&expected, 0..2, &window, &input(), None, Some(&cancelled)).is_err()
    );
    assert_eq!(bits(accepted.accumulations()), [0x3f80_0000, 0x3f80_2000]);
    assert_eq!(reserved(&host), charge);
    drop((window, matrix, source));
    assert_eq!(reserved(&host), 0);
}
