//! Tiny synthetic original-file windows; never a trained model fixture.
use super::*;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use zerogravity_artifacts::{
    artifact::Artifact,
    cache::TransferAdmission,
    source_weights::{ExpectedSourceShard, SourceWeights, WindowOpenLimits},
};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
#[path = "windowed_wide_tests.rs"]
mod wide;

struct Fixture {
    path: PathBuf,
    admission: TransferAdmission,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let path = std::env::temp_dir().join(format!(
            "zg-text-windows-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        let arbiter = ResourceArbiter::new();
        let resource = PhysicalDeviceKey::new("host", path.to_str().unwrap()).unwrap();
        arbiter
            .register(resource.clone(), 64 * 1024 * 1024)
            .unwrap();
        Self {
            path,
            admission: TransferAdmission {
                arbiter,
                resource,
                owner: ResourceOwner::new("text-reference", "tiny-source-fixture").unwrap(),
            },
        }
    }
    fn source(&self) -> WindowReferenceTextSource {
        let header=serde_json::to_vec(&serde_json::json!({"matrix":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]},"norm":{"dtype":"F32","shape":[4],"data_offsets":[16,32]},"experts":{"dtype":"BF16","shape":[2,2,4],"data_offsets":[32,64]}})).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        for value in 1..=8 {
            bytes.extend(
                ((value as f32).to_bits() >> 16).to_le_bytes()[..2]
                    .iter()
                    .copied(),
            );
        }
        bytes.extend([1.0f32; 4].into_iter().flat_map(f32::to_le_bytes));
        for value in 101..=116 {
            bytes.extend_from_slice(&((value as f32).to_bits() >> 16).to_le_bytes()[..2]);
        }
        let basename = "model.safetensors";
        let file = self.path.join(basename);
        fs::write(&file, &bytes).unwrap();
        let artifact = Artifact::from_file(File::open(file).unwrap(), Path::new(basename)).unwrap();
        let shard = ExpectedSourceShard::new(
            basename.into(),
            artifact,
            bytes.len() as u64,
            format!("{:x}", Sha256::digest(&bytes)),
        )
        .unwrap();
        let index=serde_json::to_vec(&serde_json::json!({"metadata":{"total_size":64},"weight_map":{"matrix":basename,"norm":basename,"experts":basename}})).unwrap();
        let plan = SourceWeights::prepare_verified_windows(
            &index,
            &format!("{:x}", Sha256::digest(&index)),
            vec![shard],
            WindowOpenLimits {
                max_source_bytes: 1024 * 1024,
                max_metadata_bytes: 16 * 1024 * 1024,
                max_shards: 1,
                max_tensors: 3,
                scan_chunk_bytes: 64,
            },
            &self.admission,
        )
        .unwrap();
        WindowReferenceTextSource::new(plan.open().unwrap())
    }
    fn reserved(&self) -> u64 {
        self.admission
            .arbiter
            .snapshot(&self.admission.resource)
            .unwrap()
            .reserved
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[test]
fn actual_verified_rows_parameters_and_shared_projection_retire_all_admissions() {
    let fixture = Fixture::new();
    let source = fixture.source();
    let baseline = fixture.reserved();
    assert!(baseline > 0);
    let descriptor = source.matrix("matrix").unwrap();
    assert_eq!(
        source.embed(&descriptor, 1, None).unwrap(),
        [5., 6., 7., 8.]
    );
    assert_eq!(
        source
            .project(&descriptor, &[1., 0., -1., 2.], None)
            .unwrap(),
        [6., 14.]
    );
    let parameter = source.parameter("norm", None).unwrap();
    assert_eq!(parameter.values, [1.; 4]);
    assert_eq!(parameter.descriptor.source, source.source_identity());
    assert!(fixture.reserved() > baseline);
    drop(parameter);
    assert_eq!(fixture.reserved(), baseline);
    let mut state = ReferenceTextState::new(&source, &[KvDomain(0)], 2).unwrap();
    let value = {
        let mut math = ReferenceTextMath::new(&source, &mut state, None).unwrap();
        math.embed(descriptor, 0).unwrap()
    };
    drop(state);
    drop(source);
    assert!(fixture.reserved() > 0);
    assert_eq!(value.as_slice().unwrap(), [1., 2., 3., 4.]);
    drop(value);
    assert_eq!(fixture.reserved(), 0);
}

#[test]
fn wrong_issuer_transposed_metadata_and_cancellation_refuse_before_partial_output() {
    let fixture = Fixture::new();
    let source = fixture.source();
    let good = source.matrix("matrix").unwrap();
    let mut wrong = good.clone();
    wrong.source = SourceIdentity::for_new_allocation().unwrap();
    assert!(source.project(&wrong, &[1.; 4], None).is_err());
    let mut wrong = good.clone();
    std::mem::swap(&mut wrong.rows, &mut wrong.columns);
    assert!(source.embed(&wrong, 0, None).is_err());
    let flag = AtomicBool::new(true);
    assert!(source.project(&good, &[1.; 4], Some(&flag)).is_err());
    assert!(source.parameter("norm", Some(&flag)).is_err());
    drop(source);
    assert_eq!(fixture.reserved(), 0);
}

#[test]
fn verified_rank3_expert_plane_uses_true_source_offset_and_refuses_foreign_or_oob() {
    let fixture = Fixture::new();
    let source = fixture.source();
    let baseline = fixture.reserved();
    assert!(source.matrix("experts").is_err());
    let expert = source.expert_tensor("experts").unwrap();
    assert_eq!((expert.experts, expert.rows, expert.columns), (2, 2, 4));
    assert_eq!(
        source
            .project_expert(&expert, 1, &[1., 0., -1., 2.], None)
            .unwrap(),
        [222., 230.]
    );
    assert_eq!(
        source
            .project_expert(&expert, 0, &[1., 0., -1., 2.], None)
            .unwrap(),
        [206., 214.]
    );
    assert!(source.project_expert(&expert, 2, &[1.; 4], None).is_err());
    let mut wrong = expert.clone();
    wrong.source = SourceIdentity::for_new_allocation().unwrap();
    assert!(source.project_expert(&wrong, 0, &[1.; 4], None).is_err());
    wrong = expert.clone();
    wrong.rows = 1;
    assert!(source.project_expert(&wrong, 0, &[1.; 4], None).is_err());
    let flag = AtomicBool::new(true);
    assert!(
        source
            .project_expert(&expert, 0, &[1.; 4], Some(&flag))
            .is_err()
    );
    assert_eq!(fixture.reserved(), baseline);
    drop(source);
    assert_eq!(fixture.reserved(), 0);
}

#[test]
fn expert_projection_refuses_real_exhausted_host_then_recovers_without_source_reload() {
    let fixture = Fixture::new();
    let source = fixture.source();
    let expert = source.expert_tensor("experts").unwrap();
    let baseline = fixture.reserved();
    let first = source.source().reserve_workspace(32 * 1024 * 1024).unwrap();
    let rest = source
        .source()
        .reserve_workspace((64 * 1024 * 1024 - baseline - 32 * 1024 * 1024) as usize)
        .unwrap();
    assert!(matches!(
        source.project_expert(&expert, 1, &[1.; 4], None),
        Err(Error::Limit(_))
    ));
    drop(rest);
    drop(first);
    assert_eq!(fixture.reserved(), baseline);
    assert_eq!(
        source
            .project_expert(&expert, 1, &[1., 0., -1., 2.], None)
            .unwrap(),
        [222., 230.]
    );
    let plane = source.source().expert_matrix("experts", 1).unwrap();
    let retained = source.source().read_rows(&plane, 1..2).unwrap();
    drop(source);
    assert!(fixture.reserved() > 0);
    assert_eq!(retained.row_range(), 1..2);
    assert_eq!(retained.as_slice().len(), 8);
    drop(plane);
    drop(retained);
    assert_eq!(fixture.reserved(), 0);
}

#[test]
fn candidate_casts_have_separate_value_and_kv_precision_owners() {
    let fixture = Fixture::new();
    let source = fixture.source();
    let mut candidate = ReferenceTextState::new_with_arithmetic(
        &source,
        &[KvDomain(0)],
        2,
        ReferenceTextArithmetic::Bf16EagerCandidate,
    )
    .unwrap();
    struct Position(std::cell::Cell<Option<usize>>);
    impl ReferenceTextObserver for Position {
        fn output(&self, event: ReferenceTextObservation<'_>) -> Result<()> {
            if event.operation == ReferenceTextOperation::Boundary {
                self.0.set(Some(event.position));
            }
            Ok(())
        }
    }
    let observed = Position(std::cell::Cell::new(None));
    let output = {
        let mut math = ReferenceTextMath::new_observed(
            &source,
            &mut candidate,
            None,
            ReferenceTextArithmetic::Bf16EagerCandidate,
            Some(&observed),
        )
        .unwrap();
        let raw = math.embed(source.matrix("matrix").unwrap(), 0).unwrap();
        let scaled = math
            .text_elementwise(
                "buffer scalar",
                TextElementwise::ScaleBf16Scalar(1536.0f32.sqrt()),
                &[&raw],
            )
            .unwrap();
        assert_eq!(scaled.as_slice().unwrap(), [39.25, 78.5, 118.0, 157.0]);
        assert_eq!(math.precision(&scaled), ActivationPrecision::Bf16WidenedF32);
        math.append_kv(KvDomain(0), &scaled, &scaled).unwrap();
        math.advance_frontier().unwrap();
        math.observe(usize::MAX, "final_norm", &scaled).unwrap();
        assert_eq!(observed.0.get(), Some(0)); // A final producer is still token0 after advancing KV.
        scaled
    };
    assert_eq!(
        candidate.domain_row(KvDomain(0), 0).unwrap().0,
        output.as_slice().unwrap()
    );
    let mut analytic = ReferenceTextState::new(&source, &[KvDomain(0)], 2).unwrap();
    let mut math = ReferenceTextMath::new(&source, &mut analytic, None).unwrap();
    assert!(
        math.project(source.matrix("matrix").unwrap(), &output)
            .is_err()
    );
    assert!(math.contiguous(&output).is_err());
    assert!(ReferenceTextMath::new(&source, &mut candidate, None).is_err());
    assert!(
        ReferenceTextState::new_with_arithmetic(
            &source,
            &[KvDomain(0)],
            65,
            ReferenceTextArithmetic::Bf16EagerCandidate
        )
        .is_err()
    );
    drop(output);
    drop(candidate);
    drop(analytic);
    drop(source);
    assert_eq!(fixture.reserved(), 0);
}

#[test]
fn candidate_pv_requires_one_actual_call_binding_and_preserves_cancellation_and_owners() {
    let fixture = Fixture::new();
    let source = fixture.source();
    let mut state = ReferenceTextState::new_with_arithmetic(
        &source,
        &[KvDomain(0)],
        8,
        ReferenceTextArithmetic::Bf16EagerCandidate,
    )
    .unwrap();
    let cancelled = AtomicBool::new(false);
    {
        let mut math = ReferenceTextMath::new_observed(
            &source,
            &mut state,
            Some(&cancelled),
            ReferenceTextArithmetic::Bf16EagerCandidate,
            None,
        )
        .unwrap();
        let v = math.produce(256, false, || Ok(vec![1.0; 256])).unwrap();
        let p = math.produce(1, false, || Ok(vec![1.0])).unwrap();
        math.append_kv(KvDomain(0), &v, &v).unwrap();
        let g = AttentionGeometry {
            width: 256,
            query_heads: 1,
            kv_heads: 1,
        };
        assert!(
            math.attention_value("missing source call", KvDomain(0), &p, 1, g)
                .is_err()
        );
        assert!(math.bind_bf16_attention_call(2, 1).is_err());
        math.bind_bf16_attention_call(1, 1).unwrap();
        assert!(math.bind_bf16_attention_call(1, 1).is_err());
        let actual = math
            .attention_value("actual source PV", KvDomain(0), &p, 1, g)
            .unwrap();
        assert_eq!(actual.as_slice().unwrap(), vec![1.0; 256]);
        cancelled.store(true, Ordering::Release);
        assert!(
            math.attention_value("cancelled PV", KvDomain(0), &p, 1, g)
                .is_err()
        );
    }
    drop(state);
    let mut analytic = ReferenceTextState::new(&source, &[KvDomain(0)], 8).unwrap();
    {
        let mut math = ReferenceTextMath::new(&source, &mut analytic, None).unwrap();
        assert!(math.bind_bf16_attention_call(1, 1).is_err());
    }
    drop(analytic);
    drop(source);
    assert_eq!(fixture.reserved(), 0);
}
