use super::*;

fn geometry() -> Geometry {
    Geometry::qwen35_08()
}

fn program() -> Program {
    Program::new(geometry()).unwrap()
}

#[test]
fn complete_family_has_source_derived_entries_and_no_capture_coordinates() -> Result<()> {
    let definitions = program().definitions(Step::new(64, 32_704, 32_768)?, 1e-6)?;
    assert_eq!(definitions.len(), 13);
    assert_eq!(
        definitions
            .iter()
            .map(Definition::primitive)
            .collect::<Vec<_>>(),
        [
            Primitive::Positions,
            Primitive::Rope,
            Primitive::Embedding,
            Primitive::Rms,
            Primitive::ResidualRms,
            Primitive::SiluUp,
            Primitive::Sigmoid,
            Primitive::Multiply,
            Primitive::Residual,
            Primitive::KvAppend,
            Primitive::QueryHeadRms,
            Primitive::KeyHeadRms,
            Primitive::Greedy,
        ]
    );
    for definition in &definitions {
        assert!(
            definition.source().starts_with(
                "// cubecl.cuda.dynamic_shared_bytes=0\n#define ZG_HIDDEN_WIDTH 1024\n"
            )
        );
        assert!(definition.source().contains(definition.entry()));
        assert_eq!(definition.threads(), 256);
        assert!(!definition.bindings().is_empty());
        assert!(
            definition
                .bindings()
                .iter()
                .all(|binding| binding.bytes > 0)
        );
    }
    assert_eq!(definitions[0].scalar_words(), [64, 32_704, 32_768]);
    assert_eq!(definitions[1].scalar_words(), [64, 32_768]);
    assert_eq!(
        definitions[3].scalar_words(),
        [64, 1e-6f32.to_bits() as u64]
    );
    assert_eq!(definitions[5].scalar_words(), [64, 0, 3584, 7168]);
    assert_eq!(definitions[6].scalar_words(), [64, 256, 512, 5120]);
    assert_eq!(definitions[7].scalar_words(), [64]);
    Ok(())
}

#[test]
fn canonical_buffer_schemas_are_exact() -> Result<()> {
    use Access::{Read as R, ReadWrite as RW, Write as W};
    use BufferRole::*;
    let p = program();
    let step = Step::new(2, 7, 32)?;
    let plan = p.memory_plan(2, 32)?;
    let expected = |items: &[(BufferRole, usize, Access)]| {
        items
            .iter()
            .map(|&(role, bytes, access)| BufferBinding {
                role,
                bytes,
                alignment: 4,
                access,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        p.rope(step)?.bindings(),
        expected(&[
            (Query, plan.query_batch_bytes, RW),
            (Key, plan.kv_batch_bytes, RW),
            (Positions, plan.positions_bytes, R),
            (InverseFrequency, plan.inverse_frequency_bytes, R),
            (Status, 4, RW),
        ])
    );
    assert_eq!(
        p.embedding(2)?.bindings(),
        expected(&[
            (TokenIds, plan.token_ids_bytes, R),
            (EmbeddingTable, plan.embedding_table_bytes, R),
            (Output, plan.hidden_batch_bytes, W),
            (Status, 4, RW),
        ])
    );
    assert_eq!(
        p.residual_rms(2, 1e-6)?.bindings(),
        expected(&[
            (Input, plan.hidden_batch_bytes, R),
            (Residual, plan.hidden_batch_bytes, RW),
            (RawGamma, 1024 * 2, R),
            (Output, plan.hidden_batch_bytes, W),
            (Status, 4, RW),
        ])
    );
    assert_eq!(
        p.kv_append(step)?.bindings(),
        expected(&[
            (Key, plan.kv_batch_bytes, R),
            (PackedAttention, 2 * 5120 * 2, R),
            (KeyCache, plan.key_cache_bytes, RW),
            (ValueCache, plan.value_cache_bytes, RW),
            (Status, 4, RW),
        ])
    );
    assert_eq!(p.kv_append(step)?.scalar_words(), [2, 7, 32, 4608, 5120]);
    assert_eq!(
        p.silu_up(2)?.bindings(),
        expected(&[
            (PackedGateUp, 2 * 2 * 3584 * 2, R),
            (Output, plan.intermediate_batch_bytes, W),
            (Status, 4, RW),
        ])
    );
    assert_eq!(
        p.sigmoid(2)?.bindings(),
        expected(&[
            (PackedAttention, 2 * 5120 * 2, R),
            (Output, plan.query_batch_bytes, W),
            (Status, 4, RW),
        ])
    );
    assert_eq!(
        p.unqualified_online_fp32_attention(step)?.bindings(),
        expected(&[
            (Query, plan.query_batch_bytes, R),
            (KeyCache, plan.key_cache_bytes, R),
            (ValueCache, plan.value_cache_bytes, R),
            (Output, plan.query_batch_bytes, W),
            (Status, 4, RW),
        ])
    );
    assert_eq!(
        p.query_head_rms(2, 1e-6)?.bindings(),
        expected(&[
            (PackedAttention, 2 * 5120 * 2, R),
            (QueryRawGamma, 512, R),
            (QueryNormalized, plan.query_batch_bytes, W),
            (Status, 4, RW),
        ])
    );
    assert_eq!(
        p.key_head_rms(2, 1e-6)?.bindings(),
        expected(&[
            (PackedAttention, 2 * 5120 * 2, R),
            (KeyRawGamma, 512, R),
            (KeyNormalized, plan.kv_batch_bytes, W),
            (Status, 4, RW),
        ])
    );
    assert_eq!(
        p.greedy(2)?.bindings(),
        expected(&[
            (Logits, plan.logits_bytes, R),
            (TokenIds, 8, W),
            (Maxima, 8, W),
            (Status, 4, RW),
        ])
    );
    Ok(())
}

#[test]
fn memory_plan_covers_both_sm86_qwen_geometries_without_allocating() -> Result<()> {
    assert_eq!(Geometry::qwen35_08().validate()?, geometry());
    let two_b = Geometry::qwen35_2b().validate()?;
    assert_eq!(two_b.hidden_width, 2048);
    assert_eq!(two_b.intermediate_width, 6144);
    assert_eq!(two_b.query_width()?, 2048);
    assert_eq!(two_b.kv_width()?, 512);
    let two_b_plan = Program::new(two_b)?.memory_plan(64, 32_768)?;
    assert_eq!(two_b_plan.hidden_batch_bytes, 64 * 2048 * 2);
    assert_eq!(two_b_plan.intermediate_batch_bytes, 64 * 6144 * 2);
    assert_eq!(two_b_plan.embedding_table_bytes, 248_320 * 2048 * 2);
    assert_eq!(two_b_plan.key_cache_bytes, 32_768 * 2 * 256 * 2);

    let plan = program().memory_plan(64, 32_768)?;
    assert_eq!(plan.positions_bytes, 256);
    assert_eq!(plan.hidden_batch_bytes, 64 * 1024 * 2);
    assert_eq!(plan.intermediate_batch_bytes, 64 * 3584 * 2);
    assert_eq!(plan.query_batch_bytes, 64 * 8 * 256 * 2);
    assert_eq!(plan.kv_batch_bytes, 64 * 2 * 256 * 2);
    assert_eq!(plan.embedding_table_bytes, 248_320 * 1024 * 2);
    assert_eq!(plan.inverse_frequency_bytes, 32 * 4);
    assert_eq!(plan.key_cache_bytes, 32_768 * 2 * 256 * 2);
    assert_eq!(plan.value_cache_bytes, plan.key_cache_bytes);
    assert_eq!(plan.logits_bytes, 64 * 248_320 * 4);
    assert_eq!(plan.kv_bytes_for_layers(6)?, 6 * 2 * plan.key_cache_bytes);
    assert_eq!(plan.max_activation_bytes, plan.logits_bytes);
    assert_eq!(plan.status_bytes, 4);
    Ok(())
}

#[test]
fn rows_intervals_geometry_epsilon_and_layer_overflow_refuse() -> Result<()> {
    for (rows, base, capacity) in [
        (0, 0, 1),
        (65, 0, 65),
        (1, 0, 0),
        (2, 31, 32),
        (1, 32_768, 32_768),
        (1, 0, 32_769),
        (usize::MAX, 1, 1),
    ] {
        assert!(Step::new(rows, base, capacity).is_err());
    }
    assert_eq!(Step::new(1, 0, 1)?.base, 0);
    assert_eq!(Step::new(64, 32_704, 32_768)?.rows, 64);
    assert_eq!(Step::new(64, 32_704, 32_768)?.capacity, 32_768);

    let p = program();
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(p.rms(1, epsilon).is_err());
    }
    let mut bad = geometry();
    bad.head_width = 128;
    assert!(Program::new(bad).is_err());
    bad = geometry();
    bad.query_heads = 7;
    bad.kv_heads = 2;
    assert!(Program::new(bad).is_err());
    bad = geometry();
    bad.rotary_width = 63;
    assert!(Program::new(bad).is_err());
    assert!(
        p.memory_plan(64, 32_768)?
            .kv_bytes_for_layers(usize::MAX)
            .is_err()
    );
    assert!(require_strided_view(2, 5120, 8, 512, 256, 256, 2 * 5120, "valid").is_ok());
    assert!(require_strided_view(2, 5120, 8, 512, 256, 257, 2 * 5120, "wide").is_err());
    assert!(require_strided_view(2, usize::MAX, 8, 512, 256, 256, usize::MAX, "overflow").is_err());
    Ok(())
}

#[test]
fn launch_geometry_is_runtime_derived_not_enumerated() -> Result<()> {
    let p = program();
    for rows in 1..=64 {
        let step = Step::new(rows, 19, 19 + rows)?;
        assert_eq!(p.positions(step)?.grid(), [1, 1, 1]);
        assert_eq!(p.rms(rows, 1e-6)?.grid(), [rows as u32, 1, 1]);
        assert_eq!(
            p.unqualified_online_fp32_attention(step)?.grid(),
            [(rows * 8) as u32, 1, 1]
        );
        assert_eq!(
            p.query_head_rms(rows, 1e-6)?.grid(),
            [(rows * 8) as u32, 1, 1]
        );
        assert_eq!(
            p.key_head_rms(rows, 1e-6)?.grid(),
            [(rows * 2) as u32, 1, 1]
        );
        assert_eq!(p.greedy(rows)?.grid(), [rows as u32, 1, 1]);
    }
    Ok(())
}

#[test]
fn selectable_contract_is_raw_gamma_headwise_and_excludes_unqualified_attention() -> Result<()> {
    let p = program();
    let definitions = p.definitions(Step::new(2, 7, 32)?, 1e-6)?;
    assert!(
        definitions
            .iter()
            .all(|definition| definition.primitive() != Primitive::UnqualifiedOnlineFp32Attention)
    );
    let source = definitions[0].source();
    assert!(source.contains("const float gamma = zg_load(raw_gamma + c);"));
    assert!(source.contains("ZG_DEFINE_HEAD_RMS(zg_ot_query_head_rms_v1, ZG_QUERY_HEADS)"));
    assert!(source.contains("ZG_DEFINE_HEAD_RMS(zg_ot_key_head_rms_v1, ZG_KV_HEADS)"));
    assert!(source.contains("Q/K zero-centered RMS over packed Q/G/K/V projection rows"));
    assert!(source.contains("head * input_head_stride + lane"));
    assert!(source.contains("head * head_stride + gate_offset + lane"));
    assert!(source.contains("UNQUALIFIED causal online-FP32 softmax diagnostic"));
    assert!(source.contains("void zg_ot_sigmoid_v1"));
    assert!(source.contains("void zg_ot_multiply_v1"));
    assert!(source.contains("const float sigmoid = 1.0f / (1.0f + expf(-gate));"));
    assert!(source.contains("const float y = input_value * rounded_sigmoid;"));
    assert!(!source.contains("input_value / (1.0f + expf(-gate))"));
    assert_eq!(
        p.unqualified_online_fp32_attention(Step::new(2, 7, 32)?)?
            .primitive(),
        Primitive::UnqualifiedOnlineFp32Attention
    );
    Ok(())
}

#[test]
fn packed_plan_layouts_are_exact_for_both_profiles_and_every_runtime_row() -> Result<()> {
    for geometry in [Geometry::qwen35_08(), Geometry::qwen35_2b()] {
        let p = Program::new(geometry)?;
        for rows in 1..=64 {
            let step = Step::new(rows, 31, 31 + rows)?;
            let packed_attention_bytes = rows * 5120 * 2;
            let q = p.query_head_rms(rows, 1e-6)?;
            let k = p.key_head_rms(rows, 1e-6)?;
            let kv = p.kv_append(step)?;
            let gate = p.sigmoid(rows)?;
            let ffn = p.silu_up(rows)?;
            assert_eq!(q.bindings()[0].role, BufferRole::PackedAttention);
            assert_eq!(q.bindings()[0].bytes, packed_attention_bytes);
            assert_eq!(
                q.scalar_words(),
                [rows as u64, 0, 5120, 512, 1e-6f32.to_bits() as u64]
            );
            assert_eq!(k.bindings()[0].bytes, packed_attention_bytes);
            assert_eq!(
                k.scalar_words(),
                [rows as u64, 4096, 5120, 256, 1e-6f32.to_bits() as u64]
            );
            assert_eq!(kv.bindings()[1].bytes, packed_attention_bytes);
            assert_eq!(
                kv.scalar_words(),
                [rows as u64, 31, (31 + rows) as u64, 4608, 5120]
            );
            assert_eq!(gate.bindings()[0].bytes, packed_attention_bytes);
            assert_eq!(gate.scalar_words(), [rows as u64, 256, 512, 5120]);
            assert_eq!(
                ffn.bindings()[0].bytes,
                rows * 2 * geometry.intermediate_width * 2
            );
            assert_eq!(
                ffn.scalar_words(),
                [
                    rows as u64,
                    0,
                    geometry.intermediate_width as u64,
                    (2 * geometry.intermediate_width) as u64,
                ]
            );
        }
    }
    Ok(())
}

#[test]
fn kernel_source_flags_nonfinite_gate_embedding_and_all_arithmetic_inputs() {
    let source = program().source;
    assert!(source.contains("if (!isfinite(weight)) zg_bad(status, 2u);"));
    assert!(source.contains("if (!isfinite(gate) || !isfinite(sigmoid)) zg_bad(status, 2u);"));
    assert!(source.contains("!isfinite(input_value) || !isfinite(residual_value)"));
    assert!(source.contains("!isfinite(key_value) || !isfinite(value_value)"));
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(!value.is_finite());
    }
}

#[test]
fn host_greedy_keeps_lowest_numeric_tie_and_flags_every_nonfinite() -> Result<()> {
    assert_eq!(
        greedy_host_semantics(&[-0.0, 0.0, -1.0])?,
        GreedyHostOutcome {
            token: 0,
            maximum_bits: (-0.0f32).to_bits(),
            status: 0,
        }
    );
    assert_eq!(greedy_host_semantics(&[0.0, -0.0])?.maximum_bits, 0);
    assert_eq!(greedy_host_semantics(&[1.0, 2.0, 2.0])?.token, 1);
    assert_eq!(
        greedy_host_semantics(&[f32::NAN, 3.0, f32::INFINITY])?,
        GreedyHostOutcome {
            token: 1,
            maximum_bits: 3.0f32.to_bits(),
            status: STATUS_NONFINITE | STATUS_NAN_LOGIT,
        }
    );
    assert_eq!(
        greedy_host_semantics(&[f32::NEG_INFINITY])?,
        GreedyHostOutcome {
            token: 0,
            maximum_bits: f32::NEG_INFINITY.to_bits(),
            status: STATUS_NONFINITE,
        }
    );
    assert!(greedy_host_semantics(&[]).is_err());
    Ok(())
}

#[test]
fn role_access_service_extent_alignment_and_alias_policy_are_checked() -> Result<()> {
    use cubecl_common::device::DeviceId;
    use cubecl_core::environment::stream::StreamId;

    let service = ServiceId::of::<()>(DeviceId::new(0, 0));
    let foreign = ServiceId::of::<u8>(DeviceId::new(0, 0));
    let stream = StreamId::current();
    let definition = program().residual_rms(2, 1e-6)?;
    let fresh = || {
        definition
            .bindings()
            .iter()
            .map(|expected| {
                LaunchBinding::new(
                    expected.role,
                    expected.access,
                    Handle::new(service, stream, expected.bytes as u64),
                )
            })
            .collect::<Vec<_>>()
    };
    definition.validate_bindings(service, &fresh())?;

    let mut bad = fresh();
    bad[0].role = BufferRole::Gate;
    assert!(definition.validate_bindings(service, &bad).is_err());
    let mut bad = fresh();
    bad[0].access = Access::Write;
    assert!(definition.validate_bindings(service, &bad).is_err());
    let mut bad = fresh();
    bad[0].handle.service = foreign;
    assert!(definition.validate_bindings(service, &bad).is_err());
    let mut bad = fresh();
    bad[0].handle = Handle::new(service, stream, bad[0].handle.size_in_used() + 4);
    assert!(definition.validate_bindings(service, &bad).is_err());
    let mut bad = fresh();
    bad[0].handle = Handle::new(service, stream, bad[0].handle.size_in_used() + 2).offset_start(2);
    assert!(definition.validate_bindings(service, &bad).is_err());

    // Residual and input have equal extents but no permitted alias.
    let mut bad = fresh();
    bad[1].handle = bad[0].handle.clone();
    assert!(definition.validate_bindings(service, &bad).is_err());

    // Input/output is an explicit exact in-place alias.
    let mut allowed = fresh();
    allowed[3].handle = allowed[0].handle.clone();
    definition.validate_bindings(service, &allowed)?;

    // Even an allowed role pair may not partially overlap.
    let bytes = definition.bindings()[0].bytes as u64;
    let base = Handle::new(service, stream, bytes + 4);
    let mut partial = fresh();
    partial[0].handle = base.clone().offset_end(4);
    partial[3].handle = base.offset_start(4);
    assert!(definition.validate_bindings(service, &partial).is_err());
    Ok(())
}

#[test]
fn status_clear_submit_fence_read_order_and_publication_gate_are_mandatory() {
    let mut wrong = StatusProtocol::new();
    assert!(wrong.submitted().is_err());
    assert!(wrong.fenced_read(0).is_err());

    let mut protocol = StatusProtocol::new();
    protocol.cleared().unwrap();
    assert!(protocol.cleared().is_err());
    protocol.submitted().unwrap();
    let accepted = protocol.fenced_read(0).unwrap();
    accepted.require_accepted().unwrap();
    assert!(protocol.fenced_read(0).is_err());

    for status in [
        STATUS_OUT_OF_RANGE,
        STATUS_NONFINITE,
        STATUS_NAN_LOGIT,
        u32::MAX,
    ] {
        assert!(LaunchCompletion { status }.require_accepted().is_err());
    }
}

#[test]
fn status_owner_identity_requires_same_exact_four_byte_range() -> Result<()> {
    use cubecl_common::device::DeviceId;
    use cubecl_core::environment::stream::StreamId;

    let service = ServiceId::of::<()>(DeviceId::new(0, 0));
    let stream = StreamId::current();
    let status = Handle::new(service, stream, 4);
    let binding = LaunchBinding::new(BufferRole::Status, Access::ReadWrite, status.clone());
    require_same_status(&status, &binding)?;
    assert!(
        require_same_status(
            &status,
            &LaunchBinding::new(
                BufferRole::Status,
                Access::ReadWrite,
                Handle::new(service, stream, 4),
            ),
        )
        .is_err()
    );
    let base = Handle::new(service, stream, 8);
    assert!(
        require_same_status(
            &base.clone().offset_end(4),
            &LaunchBinding::new(BufferRole::Status, Access::ReadWrite, base.offset_start(4),),
        )
        .is_err()
    );
    Ok(())
}

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
#[test]
#[ignore = "SM86 CUDA compiler/driver reflection diagnostic; compile and load only"]
fn all_ordinary_text_driver_formals_compile_and_load() -> Result<()> {
    use cubecl::cuda::{CudaRuntime, OwnedCudaQueue};
    use cubecl_runtime::runtime::Runtime;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    let client = CudaRuntime::client(&Default::default());
    if client.properties().identity.fingerprint != "ptx_sm86" {
        return Err(Error::Unsupported(
            "ordinary text ABI diagnostic requires SM86".into(),
        ));
    }
    let (_client, queue) = unsafe { OwnedCudaQueue::pin_cold(client) }
        .map_err(|error| Error::State(format!("pin ordinary text ABI queue: {error}")))?;
    let adopted = Arc::new(AtomicBool::new(false));
    unsafe { queue.install_compiler_context(Arc::new(()), adopted.clone()) }
        .map_err(|error| Error::State(format!("admit ordinary text compiler: {error}")))?;
    if !adopted.load(Ordering::Acquire) {
        return Err(Error::State(
            "ordinary text compiler owner was not adopted".into(),
        ));
    }
    assert_eq!(compile_reflection_diagnostic(&queue)?, 28);
    Ok(())
}
