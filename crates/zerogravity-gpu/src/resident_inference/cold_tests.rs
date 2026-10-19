//! Host-only refusal gates at the public constructor's exact cold boundary.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use zerogravity_core::{
    accelerator::{
        SourceIdentity,
        program::{
            AttentionGeometry, ConvDomain, KvDomain, MathematicalBackend, MatrixAxes, ParameterRef,
            PlanBuilder, SourceTensor, SourceVector, TensorView,
        },
    },
    quant::QuantKind,
    source::SourceTensorRef,
};

struct Source {
    identity: SourceIdentity,
    reads: AtomicUsize,
    shape: [u64; 2],
    packed: Vec<u8>,
    secondary: Option<(QuantKind, Vec<u8>)>,
}
impl Source {
    fn new() -> Self {
        Self {
            identity: SourceIdentity::for_new_allocation().unwrap(),
            reads: AtomicUsize::new(0),
            shape: [32, 32],
            packed: vec![0; 32 * 34],
            secondary: None,
        }
    }
}
impl TensorSource for Source {
    fn source_identity(&self) -> SourceIdentity {
        self.identity
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if name == "projection-as-embedding"
            && let Some((kind, bytes)) = &self.secondary
        {
            return Ok(SourceTensorRef {
                kind: *kind,
                shape: &self.shape,
                packed_bytes: bytes,
            });
        }
        Ok(SourceTensorRef {
            kind: QuantKind::Q8_0,
            shape: &self.shape,
            packed_bytes: &self.packed,
        })
    }
}
fn plan(
    source: SourceIdentity,
    capacity: usize,
    rows: usize,
    token: ParameterSlot,
    logits: bool,
) -> ForwardPlan {
    let mut builder = PlanBuilder::new(
        source,
        capacity,
        vec![StateSlot::Conv(ConvDomain(0))],
        false,
    )
    .unwrap();
    let input = builder
        .embed(
            SourceTensor {
                source,
                name: "embedding".into(),
                kind: QuantKind::Q8_0,
                axes: MatrixAxes::InputOutput,
                columns: 32,
                rows,
            },
            ParameterRef(token),
        )
        .unwrap();
    let output = builder
        .conv_step(
            "conv",
            ConvDomain(0),
            &input,
            SourceVector {
                source,
                name: "conv.weight",
                values: &[0.0; 128],
            },
        )
        .unwrap();
    builder.advance_frontier().unwrap();
    builder.finish(logits.then_some(output)).unwrap()
}

#[test]
fn public_profile_requires_exact_token_vector_geometry() {
    assert_eq!(
        profile_vocabulary(Some(ValueShape::Vector(32))).unwrap(),
        32
    );
    for shape in [
        None,
        Some(ValueShape::Vector(0)),
        Some(ValueShape::CausalRows {
            heads: 1,
            capacity: 32,
        }),
    ] {
        assert!(profile_vocabulary(shape).is_err());
    }
    if usize::BITS > 32 {
        assert!(profile_vocabulary(Some(ValueShape::Vector(u32::MAX as usize + 1))).is_err());
    }
    let source = Source::new();
    for bad in [
        plan(source.identity, 256, 33, ParameterSlot::Token, true),
        plan(source.identity, 256, 32, ParameterSlot::Position, true),
        plan(source.identity, 256, 32, ParameterSlot::Token, false),
    ] {
        assert!(preflight_profile(&source, &bad).is_err());
    }
    // Invalid profile geometry cannot even inspect source tensors, acquire a
    // device service, enqueue work or call an allocation callback.
    assert_eq!(source.reads.load(Ordering::Relaxed), 0);
}

#[test]
fn cold_source_capacity_and_packed_shape_guards_precede_device_ownership() {
    let mut source = Source::new();
    let valid = plan(source.identity, 512, 32, ParameterSlot::Token, true);
    let (capacity, vocab, budget) = preflight_profile(&source, &valid).unwrap();
    assert_eq!((capacity, vocab), (512, 32));
    assert_eq!(budget.states.len(), 1);
    source.reads.store(0, Ordering::Relaxed);
    let foreign = SourceIdentity::for_new_allocation().unwrap();
    assert!(
        preflight_profile(&source, &plan(foreign, 256, 32, ParameterSlot::Token, true)).is_err()
    );
    for capacity in [1, 255, 257, 768] {
        assert!(
            preflight_profile(
                &source,
                &plan(source.identity, capacity, 32, ParameterSlot::Token, true)
            )
            .is_err()
        );
    }
    assert_eq!(source.reads.load(Ordering::Relaxed), 0);
    source.shape = [31, 32];
    assert!(preflight_profile(&source, &valid).is_err());
    source.shape = [32, 32];
    source.packed.pop();
    assert!(preflight_profile(&source, &valid).is_err());
}

struct OutputFirstSource {
    identity: SourceIdentity,
    kind: QuantKind,
    embedding: Vec<u8>,
    projection: Vec<u8>,
}
impl TensorSource for OutputFirstSource {
    fn source_identity(&self) -> SourceIdentity {
        self.identity
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        match name {
            "embedding" => Ok(SourceTensorRef {
                kind: QuantKind::Q8_0,
                shape: &[32, 64],
                packed_bytes: &self.embedding,
            }),
            "output-first" => Ok(SourceTensorRef {
                kind: self.kind,
                shape: &[64, 32],
                packed_bytes: &self.projection,
            }),
            _ => Err(Error::Malformed("unknown axes fixture tensor".into())),
        }
    }
}
fn output_first_plan(source: &OutputFirstSource, axes: MatrixAxes) -> ForwardPlan {
    let mut builder = PlanBuilder::new(
        source.identity,
        256,
        vec![StateSlot::Conv(ConvDomain(0))],
        false,
    )
    .unwrap();
    let embedding =
        SourceTensor::from_source(source, "embedding", MatrixAxes::InputOutput).unwrap();
    let mut projection =
        SourceTensor::from_source(source, "output-first", MatrixAxes::OutputInput).unwrap();
    projection.axes = axes;
    let input = builder
        .embed(embedding, ParameterRef(ParameterSlot::Token))
        .unwrap();
    let hidden = builder
        .conv_step(
            "conv",
            ConvDomain(0),
            &input,
            SourceVector {
                source: source.identity,
                name: "conv.weight",
                values: &[0.0; 128],
            },
        )
        .unwrap();
    let output = builder.project(projection, &hidden).unwrap();
    builder.advance_frontier().unwrap();
    builder.finish(Some(output)).unwrap()
}

#[test]
fn cold_output_first_descriptor_keeps_raw_shape_and_existing_codec_gates() {
    let mut source = OutputFirstSource {
        identity: SourceIdentity::for_new_allocation().unwrap(),
        kind: QuantKind::F32,
        embedding: vec![0; 64 * 34],
        projection: vec![0; 64 * 32 * 4],
    };
    let valid = output_first_plan(&source, MatrixAxes::OutputInput);
    assert_eq!(preflight_profile(&source, &valid).unwrap().1, 64);
    let wrong = output_first_plan(&source, MatrixAxes::InputOutput);
    assert!(preflight_profile(&source, &wrong).is_err());
    // Axis metadata alone cannot enable an unimplemented native source codec.
    source.kind = QuantKind::Bf16;
    source.projection = vec![0; 64 * 32 * 2];
    let unsupported = output_first_plan(&source, MatrixAxes::OutputInput);
    assert!(matches!(
        preflight_profile(&source, &unsupported),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn raw_device_provider_guards_never_initialize_a_runtime() {
    use cubecl::device::{AmdDevice, CpuDevice, CudaDevice, MetalDevice, WgpuBackend, WgpuDevice};

    // Construct values directly, without Device's hardware-enumerating selectors.
    for (device, enabled) in [
        (Device::Cuda(CudaDevice::new(0)), cfg!(feature = "cuda")),
        (Device::Hip(AmdDevice::new(0)), cfg!(feature = "hip")),
        (
            Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::Vulkan)),
            cfg!(feature = "vulkan"),
        ),
    ] {
        match (validate_provider(&device), enabled) {
            (Ok(()), true) | (Err(Error::Unsupported(_)), false) => {}
            (result, _) => panic!("unexpected provider guard result: {result:?}"),
        }
        if !enabled {
            assert_public_load_refuses(device);
        }
    }
    for device in [
        Device::Cpu(CpuDevice),
        Device::Metal(MetalDevice::DefaultDevice),
        Device::Wgpu(WgpuDevice::new(Default::default())),
        Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::Metal)),
        Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::Dx12)),
        Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::Gl)),
        Device::Wgpu(WgpuDevice::new(Default::default()).on(WgpuBackend::WebGpu)),
    ] {
        assert!(matches!(
            validate_provider(&device),
            Err(Error::Unsupported(_))
        ));
        assert_public_load_refuses(device);
    }
}

fn assert_public_load_refuses(device: Device) {
    let source = Arc::new(Source::new());
    let plan = plan(source.identity, 256, 32, ParameterSlot::Token, true);
    let result = futures_lite::future::block_on(ResidentModel::load(
        device.clone(),
        source.clone(),
        plan,
        1024 * 1024,
        1024 * 1024,
    ));
    assert!(matches!(result, Err(Error::Unsupported(_))));
    assert_eq!(source.reads.load(Ordering::Relaxed), 0);
    assert!(matches!(
        crate::resident_provider::CubeResidentProvider::new(
            device,
            ["embedding".into()],
            32,
            1024 * 1024,
            1024 * 1024,
            ResourceOwner {
                model: "cold-refusal".into(),
                session: "no-runtime".into(),
            },
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn projection_storage_reuse_cannot_bypass_embedding_codec_validation() {
    for kind in [QuantKind::Q4_0, QuantKind::Q4_1, QuantKind::F32] {
        let mut source = Source::new();
        source.secondary = Some((kind, vec![0; kind.byte_len(32 * 32).unwrap() as usize]));
        let mut builder = PlanBuilder::new(
            source.identity,
            256,
            vec![StateSlot::Conv(ConvDomain(0))],
            false,
        )
        .unwrap();
        let original = builder
            .embed(
                SourceTensor {
                    source: source.identity,
                    name: "embedding".into(),
                    kind: QuantKind::Q8_0,
                    axes: MatrixAxes::InputOutput,
                    columns: 32,
                    rows: 32,
                },
                ParameterRef(ParameterSlot::Token),
            )
            .unwrap();
        let shared = SourceTensor {
            source: source.identity,
            name: "projection-as-embedding".into(),
            kind,
            axes: MatrixAxes::InputOutput,
            columns: 32,
            rows: 32,
        };
        // Both operations are valid engine operations. GPU embedding has a
        // narrower codec contract even when Project already owns this source.
        builder.project(shared.clone(), &original).unwrap();
        let embedded = builder
            .embed(shared, ParameterRef(ParameterSlot::Token))
            .unwrap();
        let output = builder
            .conv_step(
                "conv",
                ConvDomain(0),
                &embedded,
                SourceVector {
                    source: source.identity,
                    name: "conv.weight",
                    values: &[0.0; 128],
                },
            )
            .unwrap();
        builder.advance_frontier().unwrap();
        let plan = builder.finish(Some(output)).unwrap();
        assert_eq!(validate_profile_plan(&plan).unwrap(), (256, 32));
        let error = preflight_profile(&source, &plan).err().unwrap();
        assert!(
            matches!(error, Error::Unsupported(message) if message == "source embedding GPU codec is Q8_0/Q6_K only")
        );
        // The fixture reached real source validation of the preceding nodes;
        // refusal belongs to Embed support rather than initial profile shape.
        assert!(source.reads.load(Ordering::Relaxed) >= 2);
    }
}

#[test]
fn every_attention_state_reference_matches_the_appended_kv_slab() {
    let mut source = Source::new();
    source.shape = [256, 32];
    source.packed = vec![0; 32 * 8 * 34];
    for (bad_qk, bad_value) in [(false, false), (true, false), (false, true)] {
        let mut builder = PlanBuilder::new(
            source.identity,
            256,
            vec![StateSlot::Kv(KvDomain(0))],
            false,
        )
        .unwrap();
        let wide = builder
            .embed(
                SourceTensor {
                    source: source.identity,
                    name: "embedding".into(),
                    kind: QuantKind::Q8_0,
                    axes: MatrixAxes::InputOutput,
                    columns: 256,
                    rows: 32,
                },
                ParameterRef(ParameterSlot::Token),
            )
            .unwrap();
        let slab = builder
            .view(
                &wide,
                TensorView {
                    offset: 0,
                    rows: 1,
                    width: 32,
                    row_stride: 32,
                },
            )
            .unwrap();
        builder.append_kv(KvDomain(0), &slab, &slab).unwrap();
        let geometry = |wide| AttentionGeometry {
            query_heads: 1,
            kv_heads: 1,
            width: if wide { 256 } else { 32 },
        };
        let qk = builder
            .attention_qk(
                "qk",
                KvDomain(0),
                if bad_qk { &wide } else { &slab },
                ParameterRef(ParameterSlot::ValidTokens),
                geometry(bad_qk),
                1.0,
            )
            .unwrap();
        builder
            .attention_value(
                "value",
                KvDomain(0),
                &qk.scores,
                ParameterRef(ParameterSlot::ValidTokens),
                geometry(bad_value),
            )
            .unwrap();
        builder.advance_frontier().unwrap();
        let plan = builder.finish(Some(slab)).unwrap();
        assert_eq!(validate_profile_plan(&plan).unwrap(), (256, 32));
        let result = preflight_profile(&source, &plan);
        if bad_qk || bad_value {
            assert!(
                matches!(result, Err(Error::State(message)) if message == "resident attention KV slab differs from geometry")
            );
        } else {
            let (_, _, budget) = result.unwrap();
            assert_eq!(budget.states.len(), 1);
            assert!(matches!(
                budget.states[0].layout,
                StateLayout::Kv {
                    slab: 32,
                    capacity: 256
                }
            ));
        }
    }
}
