use super::*;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::{quant::QuantKind, session_root::SessionAuthority};
use zerogravity_cpu::reference_text::{
    ReferenceTextMath, ReferenceTextSource, ReferenceTextState, TextAdmission,
};
#[path = "dense31_tests.rs"]
mod dense31;

struct Fixture {
    geometry: Geometry,
    source: SourceIdentity,
    matrices: BTreeMap<Gemma4MatrixRole, SourceTensor>,
    parameters: BTreeMap<Gemma4ParameterRole, (String, Vec<f32>)>,
    rows: BTreeMap<String, Vec<f32>>,
    experts: BTreeMap<Gemma4ExpertRole, TextExpertTensor>,
    fail_head: AtomicBool,
    cancel_in_ple: AtomicBool,
    cancel_in_expert: AtomicBool,
    cancel_in_dense: AtomicBool,
}
impl Fixture {
    fn geometry(g: Geometry, materialize: bool) -> Self {
        use Gemma4MatrixRole as M;
        use Gemma4ParameterRole as P;
        let source = SourceIdentity::for_new_allocation().unwrap();
        let mut matrices = BTreeMap::new();
        let mut rows = BTreeMap::new();
        for (role, name) in matrix_names(&g) {
            let (n, k) = match role {
                M::Embedding | M::Output => (g.vocab, g.hidden),
                M::PerLayerEmbedding => (g.vocab, g.layers.len() * g.ple),
                M::PerLayerModelProjection => (g.layers.len() * g.ple, g.hidden),
                M::Query(i) => (g.layers[i].query_heads * g.layers[i].head_dim, g.hidden),
                M::Key(i) | M::Value(i) => (g.layers[i].kv_heads * g.layers[i].head_dim, g.hidden),
                M::AttentionOutput(i) => (g.hidden, g.layers[i].query_heads * g.layers[i].head_dim),
                M::FfnGate(i) | M::FfnUp(i) => (g.layers[i].feed_forward_width, g.hidden),
                M::FfnDown(i) => (g.hidden, g.layers[i].feed_forward_width),
                M::Router(_) => (g.moe.as_ref().unwrap().experts(), g.hidden),
                M::PerLayerGate(_) => (g.ple, g.hidden),
                M::PerLayerProjection(_) => (g.hidden, g.ple),
            };
            matrices.insert(
                role,
                SourceTensor {
                    source,
                    name: name.clone(),
                    kind: QuantKind::F32,
                    axes: MatrixAxes::OutputInput,
                    columns: k,
                    rows: n,
                },
            );
            if materialize {
                rows.entry(name).or_insert_with(|| vec![0.0; n * k]);
            }
        }
        let mut experts = BTreeMap::new();
        if let Some(moe) = &g.moe {
            for (role, name) in expert_names(&g) {
                let (n, k) = match role {
                    Gemma4ExpertRole::GateUp(_) => (2 * moe.intermediate_size(), g.hidden),
                    Gemma4ExpertRole::Down(_) => (g.hidden, moe.intermediate_size()),
                };
                experts.insert(
                    role,
                    TextExpertTensor {
                        source,
                        name: name.clone(),
                        kind: QuantKind::F32,
                        experts: moe.experts(),
                        rows: n,
                        columns: k,
                    },
                );
                if materialize {
                    rows.insert(name, vec![0.0; moe.experts() * n * k]);
                }
            }
        }
        let parameters = parameter_names(&g)
            .into_iter()
            .map(|(role, name)| {
                let width = match role {
                    P::PerLayerProjectionNorm => g.ple,
                    P::QueryNorm(i) | P::KeyNorm(i) => g.layers[i].head_dim,
                    P::LayerScalar(_) => 1,
                    P::ExpertScale(_) => g.moe.as_ref().unwrap().experts(),
                    _ => g.hidden,
                };
                (role, (name, vec![1.0; width]))
            })
            .collect();
        Self {
            geometry: g,
            source,
            matrices,
            parameters,
            rows,
            experts,
            fail_head: AtomicBool::new(false),
            cancel_in_ple: AtomicBool::new(false),
            cancel_in_expert: AtomicBool::new(false),
            cancel_in_dense: AtomicBool::new(false),
        }
    }
    fn tiny() -> Self {
        let mut layers = Vec::new();
        for i in 0..3 {
            layers.push(Gemma4LayerSpec {
                attention: if i == 1 {
                    crate::AttentionKind::Full
                } else {
                    crate::AttentionKind::Sliding
                },
                head_dim: 4,
                query_heads: 1,
                kv_heads: 1,
                kv_source_layer: if i == 2 { 0 } else { i },
                feed_forward_width: if i == 2 { 8 } else { 4 },
                rotary: crate::RotaryKind::Proportional,
                rope_rotate: 2,
                rope_theta: 100.0,
                key_equals_value: false,
            });
        }
        let mut result = Self::geometry(
            Geometry {
                hidden: 4,
                vocab: 3,
                ple: 2,
                max_position: 8,
                window: 2,
                epsilon: 1e-6,
                softcap: 3.0,
                layers,
                moe: None,
            },
            true,
        );
        result.rows.insert(
            "model.language_model.embed_tokens.weight".into(),
            vec![1., 2., 3., 4., -2., 1., 0.5, -1., 0.5, -0.25, 1., 2.],
        );
        result
    }
}
impl Weights for Fixture {
    fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    fn geometry(&self) -> &Geometry {
        &self.geometry
    }
    fn matrix(&self, role: Gemma4MatrixRole) -> Result<SourceTensor> {
        self.matrices
            .get(&role)
            .cloned()
            .ok_or_else(|| Error::State("fixture absent role".into()))
    }
    fn expert(&self, role: Gemma4ExpertRole) -> Result<TextExpertTensor> {
        self.experts
            .get(&role)
            .cloned()
            .ok_or_else(|| Error::State("fixture absent expert role".into()))
    }
    fn parameter(&self, role: Gemma4ParameterRole) -> Result<SourceVector<'_>> {
        let (name, values) = self
            .parameters
            .get(&role)
            .ok_or_else(|| Error::State("fixture absent gamma".into()))?;
        Ok(SourceVector {
            source: self.source,
            name,
            values,
        })
    }
}
impl ReferenceTextSource for Fixture {
    fn project_expert(
        &self,
        tensor: &TextExpertTensor,
        expert: usize,
        input: &[f32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        if self.cancel_in_expert.load(Ordering::Acquire) {
            cancelled
                .ok_or_else(|| Error::State("fixture missing cancellation flag".into()))?
                .store(true, Ordering::Release);
        }
        if cancelled.is_some_and(|x| x.load(Ordering::Acquire)) {
            return Err(Error::State("reference text cancelled".into()));
        }
        if tensor.source != self.source
            || expert >= tensor.experts
            || !self.experts.values().any(|t| t == tensor)
        {
            return Err(Error::State("fixture expert source/plane".into()));
        }
        let data = self
            .rows
            .get(&tensor.name)
            .ok_or_else(|| Error::State("fixture expert data missing".into()))?;
        let size = tensor.rows * tensor.columns;
        zerogravity_cpu::math::matvec(
            &data[expert * size..(expert + 1) * size],
            tensor.rows,
            tensor.columns,
            input,
        )
    }

    fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    fn reserve_output(&self, bytes: usize) -> Result<Option<TextAdmission>> {
        if bytes > 1024 * 1024 {
            return Err(Error::Limit("tiny source fixture output cap".into()));
        }
        Ok(None)
    }
    fn embed(
        &self,
        tensor: &SourceTensor,
        token: usize,
        _: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        if tensor.source != self.source || !self.matrices.values().any(|t| t == tensor) {
            return Err(Error::State("fixture actual matrix issuer".into()));
        }
        let data = self
            .rows
            .get(&tensor.name)
            .ok_or_else(|| Error::State("fixture no payload".into()))?;
        Ok(data[token * tensor.columns..(token + 1) * tensor.columns].to_vec())
    }
    fn project(
        &self,
        tensor: &SourceTensor,
        input: &[f32],
        cancelled: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        if tensor.name == "model.language_model.embed_tokens.weight"
            && self.fail_head.load(Ordering::Acquire)
        {
            return Err(Error::State(
                "deliberate final reference validation failure".into(),
            ));
        }
        if tensor.source != self.source || !self.matrices.values().any(|t| t == tensor) {
            return Err(Error::State("fixture actual matrix issuer".into()));
        }
        if self.cancel_in_ple.load(Ordering::Acquire)
            && tensor
                .name
                .ends_with("layers.0.per_layer_projection.weight")
        {
            cancelled
                .ok_or_else(|| Error::State("fixture cancellation flag missing".into()))?
                .store(true, Ordering::Release);
        }
        if self.cancel_in_dense.load(Ordering::Acquire)
            && tensor.name.ends_with("layers.59.mlp.down_proj.weight")
        {
            cancelled
                .ok_or_else(|| Error::State("fixture cancellation flag missing".into()))?
                .store(true, Ordering::Release);
        }
        zerogravity_cpu::math::matvec(
            self.rows
                .get(&tensor.name)
                .ok_or_else(|| Error::State("fixture no payload".into()))?,
            tensor.rows,
            tensor.columns,
            input,
        )
    }
}

#[test]
fn full_official_e2b_records_one_schedule_and_fifteen_real_kv_owners() {
    let config = Gemma4Config::from_json(include_bytes!(
        "../../tests/fixtures/official-e2b-it-config.json"
    ))
    .unwrap();
    let fixture = Fixture::geometry(Geometry::from_config(&config).unwrap(), false);
    let domains = fixture.geometry.domains();
    assert_eq!(domains, (0..15).map(KvDomain).collect::<Vec<_>>());
    assert!(fixture.matrix(Gemma4MatrixRole::Key(15)).is_err());
    assert!(fixture.parameter(Gemma4ParameterRole::KeyNorm(34)).is_err());
    let mut builder = PlanBuilder::new_with_profile(
        fixture.source,
        8,
        domains.into_iter().map(StateSlot::Kv).collect(),
        true,
        ProgramNumericalProfile::AnalyticF32TextV1,
    )
    .unwrap();
    let output = forward(&fixture, &mut builder, PlanBuilder::inputs(), true).unwrap();
    let plan = builder.finish(output).unwrap();
    assert_eq!(
        plan.numerical_profile(),
        ProgramNumericalProfile::AnalyticF32TextV1
    );
    assert_eq!(
        plan.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::AppendKv))
            .count(),
        15
    );
    assert_eq!(
        plan.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::TextWindowQk { .. }))
            .count(),
        35
    );
    assert_eq!(plan.nodes().iter().filter(|n|matches!(n.operation,PlanOperation::TextElementwise(TextElementwise::TanhSoftcap(x)) if x==30.0)).count(),1);
    assert!(plan.nodes().iter().any(|n|matches!(&n.operation,PlanOperation::Project{tensor} if tensor.name.ends_with("layers.34.mlp.down_proj.weight")&&tensor.columns==12288)));
}

#[test]
fn complete_small_text_pipeline_matches_closed_form_full_vocabulary_and_shared_cache() {
    let fixture = Fixture::tiny();
    let mut state = ReferenceTextState::new(&fixture, &fixture.geometry.domains(), 8).unwrap();
    let mut authority = SessionAuthority::new(fixture.source).unwrap();
    let logits = original::evaluate(
        &fixture,
        &fixture,
        &mut state,
        &mut authority,
        &[0, 1],
        true,
        None,
    )
    .unwrap()
    .unwrap();
    // Every branch is zero and all layer scalars are one: only scaled embedding,
    // final RMS and the tied vocabulary projection remain in the independent oracle.
    let x = [-4.0f32, 2.0, 1.0, -2.0];
    let mean = x.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / 4.0;
    let normal = x.map(|x| (f64::from(x) / (mean + 1e-6f64).sqrt()) as f32);
    let expected = fixture.rows["model.language_model.embed_tokens.weight"]
        .chunks_exact(4)
        .map(|row| {
            let score = row
                .iter()
                .zip(normal)
                .map(|(w, x)| f64::from(*w) * f64::from(x))
                .sum::<f64>() as f32;
            (score / 3.0).tanh() * 3.0
        })
        .collect::<Vec<_>>();
    for (actual, wanted) in logits.as_slice().unwrap().iter().zip(expected) {
        assert!((actual - wanted).abs() < 2e-6);
    }
    assert_eq!(state.frontier(), 2);
    assert_eq!(state.domain_tokens(KvDomain(0)).unwrap(), 2);
    assert_eq!(state.domain_tokens(KvDomain(1)).unwrap(), 2);
    assert!(state.domain_tokens(KvDomain(2)).is_err());
    assert_eq!(authority.binding().frontier(), 2);
    assert_eq!(authority.binding().root(), 1);
}

#[test]
fn token_by_token_and_single_prefill_share_math_but_publish_distinct_transactions() {
    let mut fixture = Fixture::tiny();
    // Nonzero attention, FFN and PLE branches exercise cache sharing at each step.
    for (name, values) in &mut fixture.rows {
        if name != "model.language_model.embed_tokens.weight" {
            for (i, value) in values.iter_mut().enumerate() {
                *value = ((i % 11) as f32 - 5.0) * 0.03;
            }
        }
    }
    let mut one = ReferenceTextState::new(&fixture, &fixture.geometry.domains(), 8).unwrap();
    let mut batch = one.fork(&fixture).unwrap();
    let mut a = SessionAuthority::new(fixture.source).unwrap();
    let mut b = SessionAuthority::new(fixture.source).unwrap();
    original::evaluate(&fixture, &fixture, &mut one, &mut a, &[0], false, None).unwrap();
    let one_out = original::evaluate(&fixture, &fixture, &mut one, &mut a, &[1, 2], true, None)
        .unwrap()
        .unwrap();
    let batch_out = original::evaluate(
        &fixture,
        &fixture,
        &mut batch,
        &mut b,
        &[0, 1, 2],
        true,
        None,
    )
    .unwrap()
    .unwrap();
    assert_eq!(one_out.as_slice().unwrap(), batch_out.as_slice().unwrap());
    assert_eq!(one.frontier(), batch.frontier());
    assert_eq!(a.binding().root(), 2);
    assert_eq!(b.binding().root(), 1);
}

#[test]
fn cancellation_or_final_head_failure_keeps_original_state_and_root() {
    let fixture = Fixture::tiny();
    let mut state = ReferenceTextState::new(&fixture, &fixture.geometry.domains(), 8).unwrap();
    let mut authority = SessionAuthority::new(fixture.source).unwrap();
    let original = authority.binding();
    let flag = AtomicBool::new(true);
    assert!(
        original::evaluate(
            &fixture,
            &fixture,
            &mut state,
            &mut authority,
            &[0],
            true,
            Some(&flag)
        )
        .is_err()
    );
    flag.store(false, Ordering::Release);
    fixture.cancel_in_ple.store(true, Ordering::Release);
    assert!(
        original::evaluate(
            &fixture,
            &fixture,
            &mut state,
            &mut authority,
            &[0],
            true,
            Some(&flag)
        )
        .is_err()
    );
    assert!(flag.load(Ordering::Acquire));
    fixture.cancel_in_ple.store(false, Ordering::Release);
    fixture.fail_head.store(true, Ordering::Release);
    assert!(
        original::evaluate(
            &fixture,
            &fixture,
            &mut state,
            &mut authority,
            &[0, 1],
            true,
            None
        )
        .is_err()
    );
    assert_eq!(state.frontier(), 0);
    assert_eq!(state.domain_tokens(KvDomain(0)).unwrap(), 0);
    assert_eq!(authority.binding(), original);
}

#[test]
fn proportional_rotary_pairs_use_full_head_denominator_and_split_half_partners() {
    let mut geometry = Fixture::tiny().geometry;
    geometry.hidden = 8;
    for layer in &mut geometry.layers {
        layer.head_dim = 8;
        layer.rope_rotate = 4;
    }
    let mut fixture = Fixture::geometry(geometry, true);
    fixture
        .rows
        .get_mut("model.language_model.embed_tokens.weight")
        .unwrap()[..8]
        .copy_from_slice(&[1., 2., 3., 4., 5., 6., 7., 8.]);
    let mut state = ReferenceTextState::new(&fixture, &fixture.geometry.domains(), 8).unwrap();
    let mut authority = SessionAuthority::new(fixture.source).unwrap();
    original::evaluate(
        &fixture,
        &fixture,
        &mut state,
        &mut authority,
        &[0],
        false,
        None,
    )
    .unwrap();
    let mut math = ReferenceTextMath::new(&fixture, &mut state, None).unwrap();
    let x = math
        .embed(fixture.matrix(Gemma4MatrixRole::Embedding).unwrap(), 0)
        .unwrap();
    let rotated = math
        .text_rotary(
            "rotary",
            &x,
            1,
            TextRotary {
                head_width: 8,
                rotated_pairs: 2,
                theta: 100.,
            },
        )
        .unwrap();
    let values = rotated.as_slice().unwrap();
    let raw = x.as_slice().unwrap();
    for i in [2, 3, 6, 7] {
        assert_eq!(values[i], raw[i]);
    }
    for (pair, angle) in [(0, 1.0f64), (1, 100.0f64.powf(-0.25))] {
        let expected = f64::from(raw[pair]) * angle.cos() - f64::from(raw[pair + 4]) * angle.sin();
        assert!((f64::from(values[pair]) - expected).abs() < 1e-6);
    }
    assert!(
        math.text_rotary(
            "rotary",
            &x,
            0,
            TextRotary {
                head_width: 8,
                rotated_pairs: 2,
                theta: 100.
            }
        )
        .is_err()
    );
    assert!(
        math.text_rotary(
            "rotary",
            &x,
            1,
            TextRotary {
                head_width: 8,
                rotated_pairs: 5,
                theta: 100.
            }
        )
        .is_err()
    );
}

#[test]
fn unit_attention_scale_and_sliding_mask_use_the_same_actual_kv_owner() {
    let mut fixture = Fixture::tiny();
    fixture
        .rows
        .get_mut("model.language_model.embed_tokens.weight")
        .unwrap()
        .copy_from_slice(&[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]);
    let mut state = ReferenceTextState::new(&fixture, &fixture.geometry.domains(), 8).unwrap();
    {
        let mut math = ReferenceTextMath::new(&fixture, &mut state, None).unwrap();
        let k = math
            .embed(fixture.matrix(Gemma4MatrixRole::Embedding).unwrap(), 0)
            .unwrap();
        let v = math
            .embed(fixture.matrix(Gemma4MatrixRole::Embedding).unwrap(), 1)
            .unwrap();
        for domain in fixture.geometry.domains() {
            math.append_kv(domain, &k, &v).unwrap();
        }
        math.advance_frontier().unwrap();
    }
    let mut math = ReferenceTextMath::new(&fixture, &mut state, None).unwrap();
    let k = math
        .embed(fixture.matrix(Gemma4MatrixRole::Embedding).unwrap(), 1)
        .unwrap();
    let v = math
        .embed(fixture.matrix(Gemma4MatrixRole::Embedding).unwrap(), 2)
        .unwrap();
    for domain in fixture.geometry.domains() {
        math.append_kv(domain, &k, &v).unwrap();
    }
    let q = math
        .embed(fixture.matrix(Gemma4MatrixRole::Embedding).unwrap(), 0)
        .unwrap();
    let g = AttentionGeometry {
        width: 4,
        query_heads: 1,
        kv_heads: 1,
    };
    let qk = math
        .text_window_qk("full", KvDomain(0), &q, 2, g, None)
        .unwrap();
    let p = math
        .attention_softmax("probabilities", &qk.scores, 1, 2)
        .unwrap();
    let out = math
        .attention_value("value", KvDomain(0), &p, 2, g)
        .unwrap();
    let weight = std::f64::consts::E / (1.0 + std::f64::consts::E);
    assert!((f64::from(out.as_slice().unwrap()[1]) - weight).abs() < 1e-6);
    assert!((f64::from(out.as_slice().unwrap()[2]) - (1.0 - weight)).abs() < 1e-6);
    let qk = math
        .text_window_qk("sliding", KvDomain(0), &q, 2, g, Some(1))
        .unwrap();
    let p = math
        .attention_softmax("probabilities", &qk.scores, 1, 2)
        .unwrap();
    let out = math
        .attention_value("value", KvDomain(0), &p, 2, g)
        .unwrap();
    assert_eq!(out.as_slice().unwrap(), [0., 0., 1., 0.]);
}

#[test]
fn analytic_ops_are_not_an_implicit_legacy_or_torch_profile_fallback() {
    let fixture = Fixture::tiny();
    for profile in [
        ProgramNumericalProfile::GgufSourceV1,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1,
    ] {
        let mut builder = PlanBuilder::new_operator_slice(fixture.source, profile).unwrap();
        let input = builder.embed(
            fixture.matrix(Gemma4MatrixRole::Embedding).unwrap(),
            PlanBuilder::inputs().token,
        );
        if let Ok(input) = input {
            assert!(
                builder
                    .text_elementwise("GELU", TextElementwise::GeluTanh, &[&input])
                    .is_err()
            );
        }
    }
}

#[test]
fn bf16_buffer_scalar_is_not_recorded_under_the_analytic_native_profile() {
    let fixture = Fixture::tiny();
    let mut builder =
        PlanBuilder::new_operator_slice(fixture.source, ProgramNumericalProfile::AnalyticF32TextV1)
            .unwrap();
    let input = builder
        .embed(
            fixture.matrix(Gemma4MatrixRole::Embedding).unwrap(),
            PlanBuilder::inputs().token,
        )
        .unwrap();
    assert!(
        builder
            .text_elementwise(
                "BF16 source buffer",
                TextElementwise::ScaleBf16Scalar(39.191837),
                &[&input]
            )
            .is_err()
    );
    assert!(
        builder
            .text_elementwise(
                "original analytic scale",
                TextElementwise::Scale(2.0),
                &[&input]
            )
            .is_ok()
    );
}

fn tiny_sparse() -> Fixture {
    let mut g = Fixture::tiny().geometry;
    g.ple = 0;
    g.moe = Some(crate::moe::MoeGeometry::new(4, 3, 2, 2).unwrap());
    for (i, l) in g.layers.iter_mut().enumerate() {
        l.kv_source_layer = i;
        l.key_equals_value = i == 1;
    }
    let mut f = Fixture::geometry(g, true);
    f.rows.insert(
        "model.language_model.embed_tokens.weight".into(),
        vec![1., 2., 3., 4., -2., 1., 0.5, -1., 0.5, -0.25, 1., 2.],
    );
    f
}
#[test]
fn full_26b_plan_records_thirty_sparse_branches_and_true_global_key_equals_value() {
    let config = Gemma4Config::from_json(include_bytes!(
        "../../tests/fixtures/official-26b-it-config.json"
    ))
    .unwrap();
    let fixture = Fixture::geometry(Geometry::from_config(&config).unwrap(), false);
    assert_eq!(fixture.geometry.domains().len(), 30);
    assert!(fixture.matrix(Gemma4MatrixRole::Value(5)).is_err());
    assert!(fixture.matrix(Gemma4MatrixRole::Value(4)).is_ok());
    let mut builder = PlanBuilder::new_with_profile(
        fixture.source,
        8,
        fixture
            .geometry
            .domains()
            .into_iter()
            .map(StateSlot::Kv)
            .collect(),
        true,
        ProgramNumericalProfile::AnalyticF32TextV1,
    )
    .unwrap();
    let logits = forward(&fixture, &mut builder, PlanBuilder::inputs(), true).unwrap();
    let plan = builder.finish(logits).unwrap();
    assert_eq!(
        plan.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::TextMoe { .. }))
            .count(),
        30
    );
    assert_eq!(
        plan.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::AppendKv))
            .count(),
        30
    );
    assert!(!plan.nodes().iter().any(|n| n.name.contains(".ple.")));
    let moe = plan
        .nodes()
        .iter()
        .find_map(|n| {
            if let PlanOperation::TextMoe { contract, .. } = &n.operation {
                Some(contract)
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(
        (
            moe.selected,
            moe.gate_up.experts,
            moe.gate_up.rows,
            moe.gate_up.columns
        ),
        (8, 128, 1408, 2816)
    );
    assert_eq!((moe.down.rows, moe.down.columns), (2816, 704));
}
#[test]
fn sparse_nonzero_full_pipeline_prefill_continuation_and_independent_roots_agree() {
    let mut fixture = tiny_sparse();
    for (name, row) in &mut fixture.rows {
        if !name.ends_with("embed_tokens.weight") {
            for (i, x) in row.iter_mut().enumerate() {
                *x = ((i % 13) as f32 - 6.) * 0.02;
            }
        }
    }
    let mut a = ReferenceTextState::new(&fixture, &fixture.geometry.domains(), 8).unwrap();
    let mut b = a.fork(&fixture).unwrap();
    let mut sa = SessionAuthority::new(fixture.source).unwrap();
    let mut sb = SessionAuthority::new(fixture.source).unwrap();
    original::evaluate(&fixture, &fixture, &mut a, &mut sa, &[0], false, None).unwrap();
    let x = original::evaluate(&fixture, &fixture, &mut a, &mut sa, &[1, 2], true, None)
        .unwrap()
        .unwrap();
    let y = original::evaluate(&fixture, &fixture, &mut b, &mut sb, &[0, 1, 2], true, None)
        .unwrap()
        .unwrap();
    assert_eq!(x.as_slice().unwrap(), y.as_slice().unwrap());
    assert_eq!(a.frontier(), 3);
    assert_eq!(sa.binding().root(), 2);
    assert_eq!(sb.binding().root(), 1);
    for i in 0..3 {
        assert_eq!(a.domain_tokens(KvDomain(i)).unwrap(), 3);
    }
}
#[test]
fn sparse_route_scale_and_expert_plane_order_match_independent_closed_form() {
    let mut fixture = tiny_sparse();
    // Uniform router => original expert IDs0,1, each weight1/2. Only true selected planes are read.
    let gate = fixture.expert(Gemma4ExpertRole::GateUp(0)).unwrap();
    let down = fixture.expert(Gemma4ExpertRole::Down(0)).unwrap();
    let mut gu = vec![0.0; 3 * 4 * 4];
    let mut d = vec![0.0; 3 * 4 * 2];
    for e in 0..3 {
        gu[e * 16] = 1.;
        gu[e * 16 + 8] = 2.;
        d[e * 8] = (e + 1) as f32;
    }
    fixture.rows.insert(gate.name.clone(), gu);
    fixture.rows.insert(down.name.clone(), d);
    fixture
        .parameters
        .get_mut(&Gemma4ParameterRole::ExpertScale(0))
        .unwrap()
        .1 = vec![2., 3., 100.];
    let mut state = ReferenceTextState::new(&fixture, &fixture.geometry.domains(), 8).unwrap();
    let mut math = ReferenceTextMath::new(&fixture, &mut state, None).unwrap();
    let input = math
        .embed(fixture.matrix(Gemma4MatrixRole::Embedding).unwrap(), 0)
        .unwrap();
    let c = TextMoeContract {
        router: fixture.matrix(Gemma4MatrixRole::Router(0)).unwrap(),
        gate_up: gate,
        down,
        selected: 2,
        epsilon: 1e-6,
    };
    let result = math
        .text_moe(
            "MoE",
            &c,
            TextMoeInputs {
                normalized: &input,
                router_input: &input,
                router_scale: fixture
                    .parameter(Gemma4ParameterRole::RouterScale(0))
                    .unwrap(),
                expert_scale: fixture
                    .parameter(Gemma4ParameterRole::ExpertScale(0))
                    .unwrap(),
            },
        )
        .unwrap();
    let a = (2.0f64 / std::f64::consts::PI).sqrt();
    let gelu = 0.5 * (1.0 + (a * (1.0 + 0.044715)).tanh());
    let expected = (gelu * 2.0) * (0.5 * 2.0 + 2.0 * 0.5 * 3.0);
    assert!((f64::from(result.as_slice().unwrap()[0]) - expected).abs() < 2e-6);
    assert_eq!(&result.as_slice().unwrap()[1..], &[0.; 3]);
}
#[test]
fn sparse_failure_and_cancellation_never_publish_partial_state() {
    let fixture = tiny_sparse();
    let mut state = ReferenceTextState::new(&fixture, &fixture.geometry.domains(), 8).unwrap();
    let mut session = SessionAuthority::new(fixture.source).unwrap();
    let before = session.binding();
    fixture.fail_head.store(true, Ordering::Release);
    assert!(
        original::evaluate(
            &fixture,
            &fixture,
            &mut state,
            &mut session,
            &[0, 1],
            true,
            None
        )
        .is_err()
    );
    assert_eq!(session.binding(), before);
    assert_eq!(state.frontier(), 0);
    fixture.fail_head.store(false, Ordering::Release);
    let flag = AtomicBool::new(false);
    fixture.cancel_in_expert.store(true, Ordering::Release);
    assert!(
        original::evaluate(
            &fixture,
            &fixture,
            &mut state,
            &mut session,
            &[0],
            true,
            Some(&flag)
        )
        .is_err()
    );
    assert!(flag.load(Ordering::Acquire));
    assert_eq!(session.binding(), before);
    assert_eq!(state.frontier(), 0);
    fixture.cancel_in_expert.store(false, Ordering::Release);
    let flag = AtomicBool::new(true);
    assert!(
        original::evaluate(
            &fixture,
            &fixture,
            &mut state,
            &mut session,
            &[0],
            true,
            Some(&flag)
        )
        .is_err()
    );
    assert_eq!(session.binding(), before);
}

#[test]
fn sparse_contract_refuses_foreign_source_bad_shapes_and_legacy_plan_profile() {
    let fixture = tiny_sparse();
    let c = TextMoeContract {
        router: fixture.matrix(Gemma4MatrixRole::Router(0)).unwrap(),
        gate_up: fixture.expert(Gemma4ExpertRole::GateUp(0)).unwrap(),
        down: fixture.expert(Gemma4ExpertRole::Down(0)).unwrap(),
        selected: 2,
        epsilon: 1e-6,
    };
    let router = fixture
        .parameter(Gemma4ParameterRole::RouterScale(0))
        .unwrap();
    let expert = fixture
        .parameter(Gemma4ParameterRole::ExpertScale(0))
        .unwrap();
    for mutation in 0..5 {
        let mut wrong = c.clone();
        match mutation {
            0 => wrong.router.source = SourceIdentity::for_new_allocation().unwrap(),
            1 => wrong.selected = 4,
            2 => wrong.gate_up.rows += 1,
            3 => wrong.down.columns += 1,
            _ => wrong.epsilon = f32::NAN,
        };
        assert!(wrong.validate(fixture.source, 4, router, expert).is_err());
    }
    let mut builder =
        PlanBuilder::new_operator_slice(fixture.source, ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let x = builder
        .embed(
            fixture.matrix(Gemma4MatrixRole::Embedding).unwrap(),
            PlanBuilder::inputs().token,
        )
        .unwrap();
    assert!(
        builder
            .text_moe(
                "refused",
                &c,
                TextMoeInputs {
                    normalized: &x,
                    router_input: &x,
                    router_scale: router,
                    expert_scale: expert
                }
            )
            .is_err()
    );
}

#[test]
fn dense_sparse_and_combined_norms_have_independent_source_order_oracle() {
    let mut g = tiny_sparse().geometry;
    g.layers.truncate(1);
    let mut f = Fixture::geometry(g, true);
    f.rows.insert(
        "model.language_model.embed_tokens.weight".into(),
        vec![1., 2., 3., 4., -2., 1., 0.5, -1., 0.5, -0.25, 1., 2.],
    );
    for role in [
        Gemma4MatrixRole::FfnGate(0),
        Gemma4MatrixRole::FfnUp(0),
        Gemma4MatrixRole::FfnDown(0),
    ] {
        let descriptor = f.matrix(role).unwrap();
        f.rows.get_mut(&descriptor.name).unwrap()[0] = 1.;
    }
    let gu = f.expert(Gemma4ExpertRole::GateUp(0)).unwrap();
    let down = f.expert(Gemma4ExpertRole::Down(0)).unwrap();
    for e in 0..3 {
        let data = f.rows.get_mut(&gu.name).unwrap();
        data[e * 16] = 1.;
        data[e * 16 + 8] = 1.;
        f.rows.get_mut(&down.name).unwrap()[e * 8 + 2] = 2.;
    }
    for (role, values) in [
        (Gemma4ParameterRole::PreFfnNorm(0), [2., 1., 1., 1.]),
        (Gemma4ParameterRole::MoePreFfnNorm(0), [1., 2., 1., 1.]),
        (Gemma4ParameterRole::DensePostFfnNorm(0), [3., 1., 1., 1.]),
        (Gemma4ParameterRole::MoePostFfnNorm(0), [1., 4., 1., 1.]),
        (Gemma4ParameterRole::PostFfnNorm(0), [1., 1., 5., 1.]),
    ] {
        f.parameters.get_mut(&role).unwrap().1 = values.to_vec();
    }
    let mut state = ReferenceTextState::new(&f, &f.geometry.domains(), 8).unwrap();
    let mut session = SessionAuthority::new(f.source).unwrap();
    let output = original::evaluate(&f, &f, &mut state, &mut session, &[0], true, None)
        .unwrap()
        .unwrap();
    let x = [2.0f64, 4., 6., 8.];
    let inv = (30.0f64 + 1e-6).sqrt().recip();
    let gelu = |v: f64| {
        0.5 * v * (1. + ((2. / std::f64::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh())
    };
    let d = gelu(4. * inv) * (4. * inv);
    let d = 3. * d / (d * d / 4. + 1e-6).sqrt();
    let s = gelu(2. * inv) * (2. * inv);
    let s = 8. * s / (s * s + 1e-6).sqrt();
    let common = ((d * d + s * s) / 4. + 1e-6).sqrt();
    let y = [x[0] + d / common, x[1] + s / common, x[2], x[3]];
    let norm = (y.iter().map(|v| v * v).sum::<f64>() / 4. + 1e-6).sqrt();
    for (row, actual) in f.rows["model.language_model.embed_tokens.weight"]
        .chunks_exact(4)
        .zip(output.as_slice().unwrap())
    {
        let score = row
            .iter()
            .zip(y)
            .map(|(w, v)| f64::from(*w) * v / norm)
            .sum::<f64>();
        let expected = 3. * (score / 3.).tanh();
        assert!((f64::from(*actual) - expected).abs() < 3e-6);
    }
    assert_eq!(session.binding().frontier(), 1);
}

#[test]
fn completed_diagnostic_rejection_and_late_cancel_preserve_original_root_and_rows() {
    let fixture = Fixture::tiny();
    let mut state = ReferenceTextState::new(&fixture, &fixture.geometry.domains(), 8).unwrap();
    let mut root = SessionAuthority::new(fixture.source).unwrap();
    original::evaluate(&fixture, &fixture, &mut state, &mut root, &[0], false, None).unwrap();
    let before = root.binding();
    let saved = state.domain_row(KvDomain(0), 0).unwrap().0.to_vec();
    let flag = AtomicBool::new(false);
    for cancel in [false, true] {
        let mut called = 0;
        let failed = original::evaluate_with(
            &fixture,
            &fixture,
            &mut state,
            &mut root,
            &[1, 2],
            true,
            Some(&flag),
            |completed| {
                called += 1;
                assert_eq!(completed.root_before(), before);
                assert_eq!(completed.frontier(), 3);
                assert_eq!(completed.source_identity(), fixture.source);
                assert_eq!(completed.domain_tokens(KvDomain(0))?, 3);
                assert_eq!(completed.kv_row(KvDomain(0), 0)?.0, saved);
                assert!(completed.kv_row(KvDomain(99), 0).is_err());
                assert!(completed.kv_row(KvDomain(0), 3).is_err());
                assert_eq!(completed.logits()?.unwrap().len(), 3);
                if cancel {
                    flag.store(true, Ordering::Release);
                    Ok(())
                } else {
                    Err(Error::State("expected-only deliberate rejection".into()))
                }
            },
        );
        assert!(failed.is_err());
        assert_eq!(called, 1);
        assert_eq!(root.binding(), before);
        assert_eq!(state.frontier(), 1);
        assert_eq!(state.domain_row(KvDomain(0), 0).unwrap().0, saved);
        assert!(state.domain_row(KvDomain(0), 1).is_err());
        flag.store(false, Ordering::Release);
    }
    let mut calls = 0;
    original::evaluate_with(
        &fixture,
        &fixture,
        &mut state,
        &mut root,
        &[1],
        true,
        None,
        |_| {
            calls += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(root.binding().frontier(), 2);
    assert_eq!(root.binding().root(), before.root() + 1);
}

#[test]
fn producer_rejection_and_observed_cancellation_leave_actual_root_unpublished() {
    use std::cell::Cell;
    use zerogravity_cpu::reference_text::{
        ReferenceTextArithmetic, ReferenceTextObservation, ReferenceTextObserver,
        ReferenceTextOperation,
    };
    struct Observer<'a> {
        cancel: &'a AtomicBool,
        seen: Cell<usize>,
        cancel_only: bool,
    }
    impl ReferenceTextObserver for Observer<'_> {
        fn output(&self, event: ReferenceTextObservation<'_>) -> Result<()> {
            if event.operation == ReferenceTextOperation::Boundary && event.label == "block.output"
            {
                assert_eq!(event.layer, Some(0));
                assert_eq!(event.position, 0);
                self.seen.set(self.seen.get() + 1);
                if self.cancel_only {
                    self.cancel.store(true, Ordering::Release);
                } else {
                    return Err(Error::State("fixture producer rejected".into()));
                }
            }
            Ok(())
        }
    }
    let fixture = Fixture::tiny();
    for cancel_only in [false, true] {
        let cancelled = AtomicBool::new(false);
        let observer = Observer {
            cancel: &cancelled,
            seen: Cell::new(0),
            cancel_only,
        };
        let mut state = ReferenceTextState::new(&fixture, &fixture.geometry.domains(), 4).unwrap();
        let mut authority = SessionAuthority::new(fixture.source).unwrap();
        let before = authority.binding();
        assert!(
            original::evaluate_configured(
                &fixture,
                &fixture,
                &mut state,
                &mut authority,
                &[0],
                true,
                original::EvaluationOptions {
                    cancelled: Some(&cancelled),
                    observer: Some(&observer),
                    arithmetic: ReferenceTextArithmetic::AnalyticF32
                },
                |_| panic!("producer rejection must occur before completed proposal callback")
            )
            .is_err()
        );
        assert_eq!(observer.seen.get(), 1);
        assert_eq!(authority.binding(), before);
        assert_eq!(state.frontier(), 0);
        for domain in fixture.geometry.domains() {
            assert_eq!(state.domain_tokens(domain).unwrap(), 0);
        }
        cancelled.store(false, Ordering::Release);
        original::evaluate(
            &fixture,
            &fixture,
            &mut state,
            &mut authority,
            &[0],
            true,
            Some(&cancelled),
        )
        .unwrap();
        assert_eq!(authority.binding().root(), before.root() + 1);
    }
}

#[test]
fn completed_rejection_retry_keeps_independent_session_and_all_shared_kv_owners() {
    let mut fixture = Fixture::tiny();
    for (name, values) in &mut fixture.rows {
        if name != "model.language_model.embed_tokens.weight" {
            for (i, value) in values.iter_mut().enumerate() {
                *value = ((i % 7) as f32 - 3.0) * 0.025;
            }
        }
    }
    let mut a = ReferenceTextState::new(&fixture, &fixture.geometry.domains(), 8).unwrap();
    let mut b = ReferenceTextState::new(&fixture, &fixture.geometry.domains(), 8).unwrap();
    let mut ra = SessionAuthority::new(fixture.source).unwrap();
    let mut rb = SessionAuthority::new(fixture.source).unwrap();
    for (state, root) in [(&mut a, &mut ra), (&mut b, &mut rb)] {
        original::evaluate(&fixture, &fixture, state, root, &[0], true, None).unwrap();
    }
    assert_ne!(ra.binding().session(), rb.binding().session());
    let snapshot = |state: &ReferenceTextState| {
        fixture
            .geometry
            .domains()
            .into_iter()
            .map(|domain| {
                (0..state.frontier())
                    .map(|token| {
                        let (k, v) = state.domain_row(domain, token).unwrap();
                        (
                            k.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let (before_a, before_b, rows_a, rows_b) =
        (ra.binding(), rb.binding(), snapshot(&a), snapshot(&b));
    for cancel in [false, true] {
        let flag = AtomicBool::new(false);
        let mut callbacks = 0;
        let result = original::evaluate_with(
            &fixture,
            &fixture,
            &mut a,
            &mut ra,
            &[1],
            true,
            Some(&flag),
            |actual| {
                callbacks += 1;
                assert_eq!(actual.root_before(), before_a);
                assert_eq!(actual.frontier(), 2);
                assert_eq!(actual.logits()?.unwrap().len(), 3);
                if cancel {
                    flag.store(true, Ordering::Release);
                    Ok(())
                } else {
                    Err(Error::State("deliberate independent-root rejection".into()))
                }
            },
        );
        assert!(result.is_err());
        assert_eq!(callbacks, 1);
        assert_eq!((ra.binding(), rb.binding()), (before_a, before_b));
        assert_eq!(
            (snapshot(&a), snapshot(&b)),
            (rows_a.clone(), rows_b.clone())
        );
    }
    let retry = original::evaluate(&fixture, &fixture, &mut a, &mut ra, &[1], true, None)
        .unwrap()
        .unwrap();
    assert_eq!(rb.binding(), before_b);
    assert_eq!(snapshot(&b), rows_b);
    let survivor = original::evaluate(&fixture, &fixture, &mut b, &mut rb, &[1], true, None)
        .unwrap()
        .unwrap();
    assert_eq!(retry.as_slice().unwrap(), survivor.as_slice().unwrap());
    assert_eq!(snapshot(&a), snapshot(&b));
    assert_eq!(ra.binding().root(), before_a.root() + 1);
    assert_eq!(rb.binding().root(), before_b.root() + 1);
}
