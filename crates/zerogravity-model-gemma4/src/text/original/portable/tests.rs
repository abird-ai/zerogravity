use super::*;
use zerogravity_core::session_root::SessionAuthority;

/// Header/config metadata only; no learned payload, source owner or GPU authority.
struct Metadata {
    geometry: Geometry,
    source: SourceIdentity,
    matrices: BTreeMap<Gemma4MatrixRole, SourceTensor>,
    parameters: BTreeMap<Gemma4ParameterRole, (String, Vec<f32>)>,
}
impl Metadata {
    fn e2b() -> Self {
        let config = Gemma4Config::from_json(include_bytes!(
            "../../../../tests/fixtures/official-e2b-it-config.json"
        ))
        .unwrap();
        let geometry = Geometry::from_config(&config).unwrap();
        let specs = config.tensor_specs().unwrap();
        let source = SourceIdentity::for_new_allocation().unwrap();
        let matrices = matrix_names(&geometry)
            .into_iter()
            .map(|(role, name)| {
                let s = specs.iter().find(|s| s.name == name).unwrap();
                (
                    role,
                    SourceTensor {
                        source,
                        name,
                        kind: QuantKind::Bf16,
                        axes: MatrixAxes::OutputInput,
                        rows: s.shape[0],
                        columns: s.shape[1],
                    },
                )
            })
            .collect();
        let parameters = parameter_names(&geometry)
            .into_iter()
            .map(|(role, name)| {
                let s = specs.iter().find(|s| s.name == name).unwrap();
                (role, (name, vec![1.0; s.shape[0]]))
            })
            .collect();
        Self {
            geometry,
            source,
            matrices,
            parameters,
        }
    }
}
impl Weights for Metadata {
    fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    fn geometry(&self) -> &Geometry {
        &self.geometry
    }
    fn activation_precision(&self) -> ActivationPrecision {
        ActivationPrecision::Bf16WidenedF32
    }
    fn matrix(&self, role: Gemma4MatrixRole) -> Result<SourceTensor> {
        self.matrices
            .get(&role)
            .cloned()
            .ok_or_else(|| Error::State("missing metadata role".into()))
    }
    fn parameter(&self, role: Gemma4ParameterRole) -> Result<SourceVector<'_>> {
        let (name, values) = self
            .parameters
            .get(&role)
            .ok_or_else(|| Error::State("missing metadata parameter".into()))?;
        Ok(SourceVector {
            source: self.source,
            name,
            values,
        })
    }
}
#[test]
fn complete35_uses_same_forward_all15_states_and_explicit_bf16_contract() {
    let f = Metadata::e2b();
    let p = record(&f).unwrap();
    assert_eq!(
        p.numerical_profile(),
        ProgramNumericalProfile::DirectGammaBf16TextCandidateV1
    );
    assert_eq!(p.frontier_delta(), 1);
    assert_eq!(
        p.required_domains(),
        &(0..15)
            .map(|i| StateSlot::Kv(KvDomain(i)))
            .collect::<Vec<_>>()
    );
    assert!(p.proposed_domains().iter().all(|s| s.ordinal == 1));
    assert_eq!(
        p.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::AppendKv))
            .count(),
        15
    );
    assert_eq!(
        p.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::TextWindowQk { .. }))
            .count(),
        35
    );
    assert_eq!(
        p.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::Rms { .. }))
            .count(),
        242
    );
    assert_eq!(
        p.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::Project { .. }))
            .count(),
        277
    );
    assert_eq!(
        p.nodes()
            .iter()
            .filter(|n| matches!(n.operation, PlanOperation::Embed { .. }))
            .count(),
        2
    );
    assert!(
        p.precisions()
            .iter()
            .all(|v| *v == ActivationPrecision::Bf16WidenedF32)
    );
    assert_eq!(
        p.values()[p.logits().unwrap().0 as usize],
        ValueShape::Vector(262144)
    );
    let source_bytes = f
        .matrices
        .values()
        .map(|m| (m.name.clone(), m.rows * m.columns * 2))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(source_bytes.len(), 278);
    assert_eq!(source_bytes.values().sum::<usize>(), 9_256_566_784);
    assert_eq!(
        f.parameters
            .values()
            .map(|(_, p)| p.len() * 2)
            .sum::<usize>(),
        571_974
    );
    assert!(p.nodes().iter().any(|n| matches!(
        n.operation,
        PlanOperation::TextElementwise(TextElementwise::ScaleBf16Scalar(_))
    )));
    assert!(!p.nodes().iter().any(|n| matches!(
        n.operation,
        PlanOperation::TypedOrdinaryRms { .. }
            | PlanOperation::SourceOperator { .. }
            | PlanOperation::SourceOperatorTemplate { .. }
            | PlanOperation::TextMoe { .. }
    )));
}
#[test]
fn observations_have_one_bounded_frame_and_actual_row_extents() {
    let f = Metadata::e2b();
    let p = record(&f).unwrap();
    let authority = SessionAuthority::new(f.source).unwrap();
    let call = crate::plan::bind_bf16_text_call(
        TextCallGeometry::new(f.source, f.geometry().vocab, 64).unwrap(),
        authority.begin_advance(4).unwrap(),
        Arc::from([4906, 2915, 886, 563]),
    )
    .unwrap();
    let frames = TextObservation::for_plan(&p, PAGE).unwrap();
    assert_eq!(
        frames.iter().filter(|f| f.stage == "block.output").count(),
        35
    );
    assert_eq!(
        frames.iter().filter(|f| f.allows_attention_mask).count(),
        35
    );
    for index in 0..call.rows() {
        let row = call.row(index).unwrap();
        for frame in &frames {
            assert!(frame.bytes(&row).unwrap() <= PAGE);
            if frame.stage == "attention.qk" {
                assert_eq!(frame.bytes(&row).unwrap(), 8 * (index + 1) * 4);
            }
        }
    }
    assert_eq!(
        frames
            .iter()
            .map(|f| f.bytes(&call.row(3).unwrap()).unwrap())
            .max(),
        Some(PAGE)
    );
    assert!(
        frames
            .iter()
            .any(|f| f.stage == "logits" && !f.before_frontier)
    );
    assert_eq!(authority.binding().frontier(), 0);
}
#[test]
fn foreign_source_and_non_bf16_parameters_cannot_record_candidate() {
    let mut f = Metadata::e2b();
    f.matrices
        .get_mut(&Gemma4MatrixRole::Query(0))
        .unwrap()
        .source = SourceIdentity::for_new_allocation().unwrap();
    assert!(record(&f).is_err());
    let mut f = Metadata::e2b();
    f.parameters
        .get_mut(&Gemma4ParameterRole::InputNorm(0))
        .unwrap()
        .1[0] = f32::from_bits(0x3f800001);
    assert!(record(&f).is_err());
}

#[test]
fn resident_projection_selection_is_complete_tied_once_and_ple_row_remains_bounded() {
    let f = Metadata::e2b();
    let p = record(&f).unwrap();
    let projections = p
        .nodes()
        .iter()
        .filter_map(|n| match &n.operation {
            PlanOperation::Project { tensor } => {
                Some((tensor.name.clone(), tensor.rows * tensor.columns * 2))
            }
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(projections.len(), 277);
    assert_eq!(projections.values().sum::<usize>(), 4_558_946_304);
    let embedding = &f.matrices[&Gemma4MatrixRole::Embedding];
    assert_eq!(projections[&embedding.name], 805_306_368);
    let ple = &f.matrices[&Gemma4MatrixRole::PerLayerEmbedding];
    assert!(!projections.contains_key(&ple.name));
    assert_eq!(ple.columns * 2, 17920);
    assert_eq!(
        projections.values().sum::<usize>() + ple.rows * ple.columns * 2,
        9_256_566_784
    );
    for n in p.nodes() {
        if let PlanOperation::Project { tensor } = &n.operation {
            assert!(tensor.columns.is_multiple_of(64));
            assert!(tensor.rows <= 262144);
            assert!(tensor.columns <= 12288);
            assert!(tensor.rows * tensor.columns / 2 <= i32::MAX as usize);
        }
    }
}
