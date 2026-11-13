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
    assert_eq!(definitions.len(), 14);
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
            Primitive::SigmoidMultiply,
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
    assert_eq!(definitions[7].scalar_words(), [64, 256, 512, 5120]);
    assert_eq!(definitions[8].scalar_words(), [64]);
    Ok(())
}

#[test]
fn legacy_rope_checked_contract_is_permanently_refused() -> Result<()> {
    let error = program()
        .contract(PlanPrimitive::Rope, Step::new(1, 1, 512)?)
        .unwrap_err()
        .to_string();
    assert!(error.contains("qualification-only"));
    assert!(error.contains("source-exact rotary"));
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
        p.sigmoid_multiply(2)?.bindings(),
        expected(&[
            (PackedAttention, 2 * 5120 * 2, R),
            (Input, plan.query_batch_bytes, R),
            (Gate, plan.query_batch_bytes, W),
            (Output, plan.query_batch_bytes, W),
            (Status, 4, RW),
        ])
    );
    let fused_contract = p.contract(PlanPrimitive::SigmoidMultiply, step)?;
    assert_eq!(fused_contract.primitive, PrimitiveKind::SigmoidMultiply);
    assert_eq!(fused_contract.entry, "zg_ot_sigmoid_multiply_v1");
    assert_eq!(
        fused_contract.bindings.as_ref(),
        p.sigmoid_multiply(2)?.bindings()
    );
    assert_eq!(fused_contract.scalar_words.as_ref(), &[2, 256, 512, 5120]);
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
fn bounded_row_definitions_keep_one_physical_abi_across_ragged_active_rows() -> Result<()> {
    let p = program();
    let short = RowExtent::bounded(47, 64)?;
    let full = RowExtent::bounded(64, 64)?;
    let pairs = vec![
        (p.embedding_bounded(short)?, p.embedding_bounded(full)?),
        (p.rms_bounded(short, 1e-6)?, p.rms_bounded(full, 1e-6)?),
        (
            p.residual_rms_bounded(short, 1e-6)?,
            p.residual_rms_bounded(full, 1e-6)?,
        ),
        (p.silu_up_bounded(short)?, p.silu_up_bounded(full)?),
        (p.sigmoid_bounded(short)?, p.sigmoid_bounded(full)?),
        (
            p.sigmoid_multiply_bounded(short)?,
            p.sigmoid_multiply_bounded(full)?,
        ),
        (p.multiply_bounded(short)?, p.multiply_bounded(full)?),
        (p.residual_bounded(short)?, p.residual_bounded(full)?),
        (
            p.kv_append_bounded(Step::new(47, 7, 512)?, short)?,
            p.kv_append_bounded(Step::new(64, 7, 512)?, full)?,
        ),
        (
            p.query_head_rms_bounded(short, 1e-6)?,
            p.query_head_rms_bounded(full, 1e-6)?,
        ),
        (
            p.key_head_rms_bounded(short, 1e-6)?,
            p.key_head_rms_bounded(full, 1e-6)?,
        ),
    ];
    for (short, full) in pairs {
        assert_eq!(short.entry(), full.entry());
        assert_eq!(short.source(), full.source());
        assert_eq!(short.bindings(), full.bindings());
        assert_eq!(short.kernel.binding_sizes(), full.kernel.binding_sizes());
        assert_ne!(short.scalar_words(), full.scalar_words());
    }
    Ok(())
}

#[test]
fn stateless_rows_can_cross_causal_bound_without_widening_source_steps() -> Result<()> {
    assert!(Step::new(65, 0, 1024).is_err());
    assert!(RowExtent::bounded(65, 65).is_err());
    assert!(StatelessRowExtent::bounded(0, 1).is_err());
    assert!(StatelessRowExtent::bounded(2, 1).is_err());

    let short = StatelessRowExtent::bounded(404, 513)?;
    let full = StatelessRowExtent::bounded(513, 513)?;
    assert_eq!((short.active(), short.bound()), (404, 513));
    assert_eq!((full.active(), full.bound()), (513, 513));
    #[cfg(target_pointer_width = "64")]
    assert!(StatelessRowExtent::bounded(u32::MAX as usize + 1, u32::MAX as usize + 1).is_err());

    let p = program();
    let short = p.silu_up_stateless(short)?;
    let full = p.silu_up_stateless(full)?;
    assert_eq!(short.entry(), full.entry());
    assert_eq!(short.source(), full.source());
    assert_eq!(short.bindings(), full.bindings());
    assert_eq!(short.kernel.binding_sizes(), full.kernel.binding_sizes());
    assert_ne!(short.scalar_words(), full.scalar_words());
    Ok(())
}

#[test]
fn stateless_rms_separates_physical_rows_from_exact_reduction_class() -> Result<()> {
    let p = program();
    assert_eq!(p.rms_reduction_class(1)?.lanes(), 256);
    assert_eq!(p.rms_reduction_class(3)?.lanes(), 256);
    assert_eq!(p.rms_reduction_class(4)?.lanes(), 128);
    assert_eq!(p.rms_reduction_class(8)?.lanes(), 64);
    assert_eq!(p.rms_reduction_class(16)?.lanes(), 32);
    let reduction = p.rms_reduction_class(47)?;
    for rows in 48..=64 {
        assert_eq!(p.rms_reduction_class(rows)?, reduction);
    }

    let physical = StatelessRowExtent::bounded(404, 513)?;
    let rms = p.rms_stateless(physical, reduction, 1e-6)?;
    let residual = p.residual_rms_stateless(physical, reduction, 1e-6)?;
    assert_eq!(rms.primitive(), Primitive::RmsStateless);
    assert_eq!(residual.primitive(), Primitive::ResidualRmsStateless);
    assert_eq!(rms.scalar_words(), [404, 32, 1e-6f32.to_bits() as u64]);
    assert_eq!(residual.scalar_words(), [404, 32, 1e-6f32.to_bits() as u64]);
    assert_eq!(rms.grid(), [404, 1, 1]);
    assert_eq!(residual.grid(), [404, 1, 1]);

    let two_b = Program::new(Geometry::qwen35_2b())?;
    assert_eq!(two_b.rms_reduction_class(1)?.lanes(), 512);
    assert_eq!(two_b.rms_reduction_class(2)?.lanes(), 256);
    assert_eq!(two_b.rms_reduction_class(47)?.lanes(), 32);
    assert!(
        p.rms_stateless(physical, two_b.rms_reduction_class(47)?, 1e-6)
            .is_err()
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
    assert!(source.contains("head * input_head_stride;"));
    assert!(source.contains("zg_load(input + base + lane)"));
    assert!(source.contains("head * head_stride + gate_offset + lane"));
    assert!(source.contains("UNQUALIFIED causal online-FP32 softmax diagnostic"));
    assert!(source.contains("void zg_ot_sigmoid_v1"));
    assert!(source.contains("void zg_ot_sigmoid_multiply_v1"));
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
    for entry in [
        "void zg_ot_rms_v1",
        "void zg_ot_residual_rms_v1",
        "void zg_ot_rms_stateless_v1",
        "void zg_ot_residual_rms_stateless_v1",
    ] {
        assert!(source.contains(entry));
    }
    // Four vector RMS kernels plus the shared head-RMS macro all fail closed
    // on nonfinite/zero inverse normalization.
    assert_eq!(source.matches("!isfinite(mean)").count(), 5);
    assert_eq!(source.matches("inverse == 0.0f").count(), 5);
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

    // Read/read overlap is still an alias and is not implicitly permitted.
    let input_bytes = definition.bindings()[0].bytes as u64;
    let gamma_bytes = definition.bindings()[2].bytes as u64;
    let base = Handle::new(service, stream, input_bytes + gamma_bytes - 4);
    let mut read_read = fresh();
    read_read[0].handle = base.clone().offset_end(gamma_bytes - 4);
    read_read[2].handle = base.offset_start(input_bytes - 4);
    assert!(definition.validate_bindings(service, &read_read).is_err());

    // Write/write overlap is also rejected unless the exact role pair is listed.
    let greedy = program().greedy(2)?;
    let mut write_write = greedy
        .bindings()
        .iter()
        .map(|expected| {
            LaunchBinding::new(
                expected.role,
                expected.access,
                Handle::new(service, stream, expected.bytes as u64),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(greedy.bindings()[1].access, Access::Write);
    assert_eq!(greedy.bindings()[2].access, Access::Write);
    assert_eq!(greedy.bindings()[1].bytes, greedy.bindings()[2].bytes);
    write_write[2].handle = write_write[1].handle.clone();
    assert!(greedy.validate_bindings(service, &write_write).is_err());
    Ok(())
}

#[test]
fn owner_gated_status_binding_uses_the_same_exact_abi_without_minting_completion() -> Result<()> {
    use cubecl_common::device::DeviceId;
    use cubecl_core::environment::stream::StreamId;

    let service = ServiceId::of::<()>(DeviceId::new(0, 0));
    let stream = StreamId::current();
    let definition = program().sigmoid(2)?;
    let mut bindings = definition
        .bindings()
        .iter()
        .filter(|binding| binding.role != BufferRole::Status)
        .map(|expected| {
            LaunchBinding::new(
                expected.role,
                expected.access,
                Handle::new(service, stream, expected.bytes as u64),
            )
        })
        .collect::<Vec<_>>();
    let status = LaunchBinding::new(
        BufferRole::Status,
        Access::ReadWrite,
        Handle::new(service, stream, 4),
    );
    let (validated, retained) =
        definition.validated_owner_bindings(service, bindings.clone(), status.clone())?;
    assert_eq!(validated.len(), definition.bindings().len());
    assert!(same_range(&retained, &status.handle)?);

    // Caller cannot smuggle its own status among data operands.
    bindings.push(status.clone());
    assert!(
        definition
            .validated_owner_bindings(service, bindings, status.clone())
            .is_err()
    );

    // The owner status is still the exact four-byte RW range, not merely a
    // same-size or overlapping allocation.
    let wrong = LaunchBinding::new(
        BufferRole::Status,
        Access::ReadWrite,
        Handle::new(service, stream, 4),
    );
    definition.validated_owner_bindings(
        service,
        validated[..validated.len() - 1].to_vec(),
        wrong.clone(),
    )?;
    assert!(require_same_status(&retained, &wrong).is_err());
    let wrong_access = LaunchBinding::new(BufferRole::Status, Access::Read, status.handle.clone());
    assert!(
        definition
            .validated_owner_bindings(
                service,
                validated[..validated.len() - 1].to_vec(),
                wrong_access,
            )
            .is_err()
    );
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

#[cfg(all(
    feature = "qualification",
    feature = "cuda-cublas",
    target_os = "linux"
))]
#[test]
#[ignore = "physical SM86 stateless SiLU-vs-causal-chunks qualification"]
fn physical_stateless_silu_rows_match_causal_chunks_exactly() -> Result<()> {
    futures_lite::future::block_on(async {
        let diagnostic = crate::cuda::test_support::CudaDiagnostic::new("stateless-silu-rows")?;
        let service = diagnostic.service.clone();
        let client = service.client();
        let program = program();
        let intermediate = program.geometry().intermediate_width;
        let packed_width = intermediate * 2;
        let bound_rows = 513usize;
        let packed_bytes = bound_rows * packed_width * 2;
        let output_bytes = bound_rows * intermediate * 2;

        let word = |i: usize, seed: usize| -> u16 {
            match (i + seed) % 8 {
                0 => 0x3f80, // 1.0
                1 => 0xbf00, // -0.5
                2 => 0x3e80, // 0.25
                3 => 0x4000, // 2.0
                4 => 0xbe00, // -0.125
                5 => 0x3f00, // 0.5
                6 => 0xc000, // -2.0
                _ => 0x0000,
            }
        };
        let bytes_from_words = |words: &[u16]| {
            words
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let view = |handle: &Handle, total: usize, offset: usize, bytes: usize| {
            handle
                .clone()
                .offset_start(offset as u64)
                .offset_end((total - offset - bytes) as u64)
        };
        let status_binding =
            |handle: Handle| LaunchBinding::new(BufferRole::Status, Access::ReadWrite, handle);

        for active_rows in [65usize, 404, 512, 513] {
            let mut packed_words = (0..bound_rows * packed_width)
                .map(|i| word(i, active_rows))
                .collect::<Vec<_>>();
            packed_words[(active_rows / 2) * packed_width + 7] = 0x7f80; // +inf gate.
            let packed = client.create_from_slice(&bytes_from_words(&packed_words));
            let zeros = vec![0u8; output_bytes];
            let sequential_output = client.create_from_slice(&zeros);
            let wide_output = client.create_from_slice(&zeros);
            let sequential_status = client.create_from_slice(&0u32.to_ne_bytes());
            let wide_status = client.create_from_slice(&0u32.to_ne_bytes());

            let mut row = 0usize;
            while row < active_rows {
                let rows = (active_rows - row).min(MAX_CAUSAL_ROWS);
                let packed_offset = row * packed_width * 2;
                let packed_chunk_bytes = rows * packed_width * 2;
                let output_offset = row * intermediate * 2;
                let output_chunk_bytes = rows * intermediate * 2;
                let definition = program.silu_up(rows)?;
                let prepared = definition.prepare_owned(
                    client,
                    vec![
                        LaunchBinding::new(
                            BufferRole::PackedGateUp,
                            Access::Read,
                            view(&packed, packed_bytes, packed_offset, packed_chunk_bytes),
                        ),
                        LaunchBinding::new(
                            BufferRole::Output,
                            Access::Write,
                            view(
                                &sequential_output,
                                output_bytes,
                                output_offset,
                                output_chunk_bytes,
                            ),
                        ),
                    ],
                    status_binding(sequential_status.clone()),
                )?;
                prepared.enqueue_owned(client, &sequential_status)?;
                row += rows;
            }

            let definition =
                program.silu_up_stateless(StatelessRowExtent::bounded(active_rows, bound_rows)?)?;
            let prepared = definition.prepare_owned(
                client,
                vec![
                    LaunchBinding::new(BufferRole::PackedGateUp, Access::Read, packed.clone()),
                    LaunchBinding::new(BufferRole::Output, Access::Write, wide_output.clone()),
                ],
                status_binding(wide_status.clone()),
            )?;
            prepared.enqueue_owned(client, &wide_status)?;

            client
                .sync()
                .await
                .map_err(|error| Error::State(format!("stateless SiLU sync: {error}")))?;
            let sequential = client
                .read_one(sequential_output)
                .map_err(|error| Error::State(format!("stateless SiLU sequential: {error}")))?;
            let wide = client
                .read_one(wide_output)
                .map_err(|error| Error::State(format!("stateless SiLU wide: {error}")))?;
            let sequential_status = client.read_one(sequential_status).map_err(|error| {
                Error::State(format!("stateless SiLU sequential status: {error}"))
            })?;
            let wide_status = client
                .read_one(wide_status)
                .map_err(|error| Error::State(format!("stateless SiLU wide status: {error}")))?;
            if sequential != wide || sequential_status != wide_status {
                return Err(Error::State(format!(
                    "stateless SiLU physical mismatch active_rows={active_rows}"
                )));
            }
            if &wide_status[..] != 2u32.to_ne_bytes().as_slice() {
                return Err(Error::State(format!(
                    "stateless SiLU nonfinite status active_rows={active_rows}: {wide_status:?}"
                )));
            }
            println!(
                "STATELESS_SILU_PASS active_rows={active_rows} bound_rows={bound_rows} exact=true status=2"
            );
        }

        client
            .memory_cleanup()
            .map_err(|error| Error::State(format!("stateless SiLU cleanup: {error}")))?;
        service.close_model_namespace()?;
        Ok(())
    })
}

#[cfg(all(
    feature = "qualification",
    feature = "cuda-cublas",
    target_os = "linux"
))]
#[test]
#[ignore = "physical SM86 stateless RMS-vs-ragged-causal qualification"]
fn physical_stateless_rms_matches_ragged_causal_calls_exactly() -> Result<()> {
    futures_lite::future::block_on(async {
        let diagnostic = crate::cuda::test_support::CudaDiagnostic::new("stateless-rms-rows")?;
        let service = diagnostic.service.clone();
        let client = service.client();
        let program = program();
        let hidden = program.geometry().hidden_width;
        let lane_rows = [47usize, 48, 49, 50, 51, 52, 53, 54];
        let active_rows = lane_rows.iter().sum::<usize>();
        let bound_rows = 513usize;
        let epsilon = 1e-6f32;
        let reduction = program.rms_reduction_class(lane_rows[0])?;
        if lane_rows
            .iter()
            .any(|&rows| program.rms_reduction_class(rows).ok() != Some(reduction))
        {
            return Err(Error::State(
                "ragged RMS fixture reduction classes differ".into(),
            ));
        }

        let word = |i: usize, seed: usize| -> u16 {
            match (i + seed) % 8 {
                0 => 0x3f80, // 1.0
                1 => 0xbf00, // -0.5
                2 => 0x3e80, // 0.25
                3 => 0x4000, // 2.0
                4 => 0xbe00, // -0.125
                5 => 0x3f00, // 0.5
                6 => 0xc000, // -2.0
                _ => 0x0000,
            }
        };
        let bytes_from_words = |words: &[u16]| {
            words
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let view = |handle: &Handle, total: usize, offset: usize, bytes: usize| {
            handle
                .clone()
                .offset_start(offset as u64)
                .offset_end((total - offset - bytes) as u64)
        };
        let status_binding =
            |handle: Handle| LaunchBinding::new(BufferRole::Status, Access::ReadWrite, handle);

        let total_bytes = bound_rows * hidden * 2;
        let active_bytes = active_rows * hidden * 2;
        let mut input_words = (0..bound_rows * hidden)
            .map(|i| word(i, 3))
            .collect::<Vec<_>>();
        input_words[(active_rows / 2) * hidden + 7] = 0x7f80; // +inf.
        let residual_words = (0..bound_rows * hidden)
            .map(|i| word(i, 11))
            .collect::<Vec<_>>();
        let gamma_words = (0..hidden).map(|i| word(i, 19)).collect::<Vec<_>>();
        let input = client.create_from_slice(&bytes_from_words(&input_words));
        let gamma = client.create_from_slice(&bytes_from_words(&gamma_words));
        let residual_bytes = bytes_from_words(&residual_words);
        let output_seed = vec![0x5au8; total_bytes];

        let sequential_rms = client.create_from_slice(&output_seed);
        let wide_rms = client.create_from_slice(&output_seed);
        let sequential_residual = client.create_from_slice(&residual_bytes);
        let wide_residual = client.create_from_slice(&residual_bytes);
        let sequential_residual_output = client.create_from_slice(&output_seed);
        let wide_residual_output = client.create_from_slice(&output_seed);
        let sequential_rms_status = client.create_from_slice(&0u32.to_ne_bytes());
        let wide_rms_status = client.create_from_slice(&0u32.to_ne_bytes());
        let sequential_residual_status = client.create_from_slice(&0u32.to_ne_bytes());
        let wide_residual_status = client.create_from_slice(&0u32.to_ne_bytes());

        let mut row = 0usize;
        for rows in lane_rows {
            let offset = row * hidden * 2;
            let bytes = rows * hidden * 2;
            let rms = program.rms(rows, epsilon)?;
            rms.prepare_owned(
                client,
                vec![
                    LaunchBinding::new(
                        BufferRole::Input,
                        Access::Read,
                        view(&input, total_bytes, offset, bytes),
                    ),
                    LaunchBinding::new(BufferRole::RawGamma, Access::Read, gamma.clone()),
                    LaunchBinding::new(
                        BufferRole::Output,
                        Access::Write,
                        view(&sequential_rms, total_bytes, offset, bytes),
                    ),
                ],
                status_binding(sequential_rms_status.clone()),
            )?
            .enqueue_owned(client, &sequential_rms_status)?;

            let residual = program.residual_rms(rows, epsilon)?;
            residual
                .prepare_owned(
                    client,
                    vec![
                        LaunchBinding::new(
                            BufferRole::Input,
                            Access::Read,
                            view(&input, total_bytes, offset, bytes),
                        ),
                        LaunchBinding::new(
                            BufferRole::Residual,
                            Access::ReadWrite,
                            view(&sequential_residual, total_bytes, offset, bytes),
                        ),
                        LaunchBinding::new(BufferRole::RawGamma, Access::Read, gamma.clone()),
                        LaunchBinding::new(
                            BufferRole::Output,
                            Access::Write,
                            view(&sequential_residual_output, total_bytes, offset, bytes),
                        ),
                    ],
                    status_binding(sequential_residual_status.clone()),
                )?
                .enqueue_owned(client, &sequential_residual_status)?;
            row += rows;
        }
        if row != active_rows {
            return Err(Error::State("ragged RMS fixture row coverage".into()));
        }

        let physical = StatelessRowExtent::bounded(active_rows, bound_rows)?;
        program
            .rms_stateless(physical, reduction, epsilon)?
            .prepare_owned(
                client,
                vec![
                    LaunchBinding::new(BufferRole::Input, Access::Read, input.clone()),
                    LaunchBinding::new(BufferRole::RawGamma, Access::Read, gamma.clone()),
                    LaunchBinding::new(BufferRole::Output, Access::Write, wide_rms.clone()),
                ],
                status_binding(wide_rms_status.clone()),
            )?
            .enqueue_owned(client, &wide_rms_status)?;
        program
            .residual_rms_stateless(physical, reduction, epsilon)?
            .prepare_owned(
                client,
                vec![
                    LaunchBinding::new(BufferRole::Input, Access::Read, input.clone()),
                    LaunchBinding::new(
                        BufferRole::Residual,
                        Access::ReadWrite,
                        wide_residual.clone(),
                    ),
                    LaunchBinding::new(BufferRole::RawGamma, Access::Read, gamma.clone()),
                    LaunchBinding::new(
                        BufferRole::Output,
                        Access::Write,
                        wide_residual_output.clone(),
                    ),
                ],
                status_binding(wide_residual_status.clone()),
            )?
            .enqueue_owned(client, &wide_residual_status)?;

        client
            .sync()
            .await
            .map_err(|error| Error::State(format!("stateless RMS sync: {error}")))?;
        let sequential_rms = client
            .read_one(sequential_rms)
            .map_err(|error| Error::State(format!("causal RMS read: {error}")))?;
        let wide_rms = client
            .read_one(wide_rms)
            .map_err(|error| Error::State(format!("stateless RMS read: {error}")))?;
        let sequential_residual = client
            .read_one(sequential_residual)
            .map_err(|error| Error::State(format!("causal residual read: {error}")))?;
        let wide_residual = client
            .read_one(wide_residual)
            .map_err(|error| Error::State(format!("stateless residual read: {error}")))?;
        let sequential_residual_output = client
            .read_one(sequential_residual_output)
            .map_err(|error| Error::State(format!("causal residual RMS read: {error}")))?;
        let wide_residual_output = client
            .read_one(wide_residual_output)
            .map_err(|error| Error::State(format!("stateless residual RMS read: {error}")))?;
        let sequential_rms_status = client
            .read_one(sequential_rms_status)
            .map_err(|error| Error::State(format!("causal RMS status: {error}")))?;
        let wide_rms_status = client
            .read_one(wide_rms_status)
            .map_err(|error| Error::State(format!("stateless RMS status: {error}")))?;
        let sequential_residual_status = client
            .read_one(sequential_residual_status)
            .map_err(|error| Error::State(format!("causal residual RMS status: {error}")))?;
        let wide_residual_status = client
            .read_one(wide_residual_status)
            .map_err(|error| Error::State(format!("stateless residual RMS status: {error}")))?;

        if sequential_rms != wide_rms
            || sequential_residual != wide_residual
            || sequential_residual_output != wide_residual_output
            || sequential_rms_status != wide_rms_status
            || sequential_residual_status != wide_residual_status
        {
            return Err(Error::State(
                "stateless RMS differs from ragged causal calls".into(),
            ));
        }
        if &wide_rms_status[..] != 2u32.to_ne_bytes().as_slice()
            || &wide_residual_status[..] != 2u32.to_ne_bytes().as_slice()
            || wide_rms[active_bytes..] != output_seed[active_bytes..]
            || wide_residual_output[active_bytes..] != output_seed[active_bytes..]
            || wide_residual[active_bytes..] != residual_bytes[active_bytes..]
        {
            return Err(Error::State(
                "stateless RMS status/tail preservation".into(),
            ));
        }
        println!(
            "STATELESS_RMS_PASS lanes=47..54 active_rows={active_rows} bound_rows={bound_rows} reduction_width={} exact=true status=2",
            reduction.lanes()
        );

        // Exercise the width>physical-thread fold independently on the 2B
        // geometry: one-row causal RMS uses 512 logical reduction lanes.
        let two_b = Program::new(Geometry::qwen35_2b())?;
        let hidden = two_b.geometry().hidden_width;
        let active_rows = 65usize;
        let total_bytes = active_rows * hidden * 2;
        let mut input_words = (0..active_rows * hidden)
            .map(|i| word(i, 29))
            .collect::<Vec<_>>();
        input_words[31 * hidden + 5] = 0x7f80;
        let gamma_words = (0..hidden).map(|i| word(i, 37)).collect::<Vec<_>>();
        let input = client.create_from_slice(&bytes_from_words(&input_words));
        let gamma = client.create_from_slice(&bytes_from_words(&gamma_words));
        let sequential = client.create_from_slice(&vec![0u8; total_bytes]);
        let wide = client.create_from_slice(&vec![0u8; total_bytes]);
        let sequential_status = client.create_from_slice(&0u32.to_ne_bytes());
        let wide_status = client.create_from_slice(&0u32.to_ne_bytes());
        for row in 0..active_rows {
            let offset = row * hidden * 2;
            two_b
                .rms(1, epsilon)?
                .prepare_owned(
                    client,
                    vec![
                        LaunchBinding::new(
                            BufferRole::Input,
                            Access::Read,
                            view(&input, total_bytes, offset, hidden * 2),
                        ),
                        LaunchBinding::new(BufferRole::RawGamma, Access::Read, gamma.clone()),
                        LaunchBinding::new(
                            BufferRole::Output,
                            Access::Write,
                            view(&sequential, total_bytes, offset, hidden * 2),
                        ),
                    ],
                    status_binding(sequential_status.clone()),
                )?
                .enqueue_owned(client, &sequential_status)?;
        }
        let reduction = two_b.rms_reduction_class(1)?;
        two_b
            .rms_stateless(
                StatelessRowExtent::bounded(active_rows, active_rows)?,
                reduction,
                epsilon,
            )?
            .prepare_owned(
                client,
                vec![
                    LaunchBinding::new(BufferRole::Input, Access::Read, input),
                    LaunchBinding::new(BufferRole::RawGamma, Access::Read, gamma),
                    LaunchBinding::new(BufferRole::Output, Access::Write, wide.clone()),
                ],
                status_binding(wide_status.clone()),
            )?
            .enqueue_owned(client, &wide_status)?;
        client
            .sync()
            .await
            .map_err(|error| Error::State(format!("stateless 2B RMS sync: {error}")))?;
        let sequential = client
            .read_one(sequential)
            .map_err(|error| Error::State(format!("causal 2B RMS read: {error}")))?;
        let wide = client
            .read_one(wide)
            .map_err(|error| Error::State(format!("stateless 2B RMS read: {error}")))?;
        let sequential_status = client
            .read_one(sequential_status)
            .map_err(|error| Error::State(format!("causal 2B RMS status: {error}")))?;
        let wide_status = client
            .read_one(wide_status)
            .map_err(|error| Error::State(format!("stateless 2B RMS status: {error}")))?;
        if sequential != wide
            || sequential_status != wide_status
            || &wide_status[..] != 2u32.to_ne_bytes().as_slice()
        {
            return Err(Error::State(
                "stateless 2B one-row RMS reduction differs".into(),
            ));
        }
        println!(
            "STATELESS_RMS_PASS geometry=2B source_rows=1 physical_rows={active_rows} reduction_width={} exact=true status=2",
            reduction.lanes()
        );

        client
            .memory_cleanup()
            .map_err(|error| Error::State(format!("stateless RMS cleanup: {error}")))?;
        service.close_model_namespace()?;
        Ok(())
    })
}

#[cfg(all(
    feature = "qualification",
    feature = "cuda-cublas",
    target_os = "linux"
))]
#[test]
#[ignore = "physical SM86 stateless Q/K head-RMS-vs-ragged-causal qualification"]
fn physical_stateless_qk_rms_matches_ragged_causal_calls_exactly() -> Result<()> {
    futures_lite::future::block_on(async {
        let diagnostic = crate::cuda::test_support::CudaDiagnostic::new("stateless-qk-rms")?;
        let service = diagnostic.service.clone();
        let client = service.client();
        let program = program();
        let query_width = program.geometry().query_width()?;
        let key_width = program.geometry().kv_width()?;
        let head_width = program.geometry().head_width;
        let packed_width = program.attention_row_stride()?;
        let lane_rows = [47usize, 48, 49, 50, 51, 52, 53, 54];
        let active_rows = lane_rows.iter().sum::<usize>();
        let bound_rows = 513usize;
        let epsilon = 1e-6f32;

        let word = |i: usize, seed: usize| -> u16 {
            match (i + seed) % 8 {
                0 => 0x3f80,
                1 => 0xbf00,
                2 => 0x3e80,
                3 => 0x4000,
                4 => 0xbe00,
                5 => 0x3f00,
                6 => 0xc000,
                _ => 0x0000,
            }
        };
        let bytes_from_words = |words: &[u16]| {
            words
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let view = |handle: &Handle, total: usize, offset: usize, bytes: usize| {
            handle
                .clone()
                .offset_start(offset as u64)
                .offset_end((total - offset - bytes) as u64)
        };
        let status_binding =
            |handle: Handle| LaunchBinding::new(BufferRole::Status, Access::ReadWrite, handle);

        let packed_bytes = bound_rows * packed_width * 2;
        let query_bytes = bound_rows * query_width * 2;
        let key_bytes = bound_rows * key_width * 2;
        let active_query_bytes = active_rows * query_width * 2;
        let active_key_bytes = active_rows * key_width * 2;
        let packed_words = (0..bound_rows * packed_width)
            .map(|i| word(i, 5))
            .collect::<Vec<_>>();
        let mut query_gamma_words = (0..head_width).map(|i| word(i, 11)).collect::<Vec<_>>();
        query_gamma_words[3] = 0x7f80; // +inf drives the shared status bit.
        let key_gamma_words = (0..head_width).map(|i| word(i, 17)).collect::<Vec<_>>();

        let packed = client.create_from_slice(&bytes_from_words(&packed_words));
        let query_gamma = client.create_from_slice(&bytes_from_words(&query_gamma_words));
        let key_gamma = client.create_from_slice(&bytes_from_words(&key_gamma_words));
        let query_seed = vec![0x5au8; query_bytes];
        let key_seed = vec![0xa5u8; key_bytes];
        let sequential_query = client.create_from_slice(&query_seed);
        let wide_query = client.create_from_slice(&query_seed);
        let dual_query = client.create_from_slice(&query_seed);
        let sequential_key = client.create_from_slice(&key_seed);
        let wide_key = client.create_from_slice(&key_seed);
        let dual_key = client.create_from_slice(&key_seed);
        let sequential_status = client.create_from_slice(&0u32.to_ne_bytes());
        let wide_status = client.create_from_slice(&0u32.to_ne_bytes());
        let dual_status = client.create_from_slice(&0u32.to_ne_bytes());

        let mut row = 0usize;
        for rows in lane_rows {
            let packed_offset = row * packed_width * 2;
            let packed_extent = rows * packed_width * 2;
            let query_offset = row * query_width * 2;
            let query_extent = rows * query_width * 2;
            let key_offset = row * key_width * 2;
            let key_extent = rows * key_width * 2;

            program
                .query_head_rms(rows, epsilon)?
                .prepare_owned(
                    client,
                    vec![
                        LaunchBinding::new(
                            BufferRole::PackedAttention,
                            Access::Read,
                            view(&packed, packed_bytes, packed_offset, packed_extent),
                        ),
                        LaunchBinding::new(
                            BufferRole::QueryRawGamma,
                            Access::Read,
                            query_gamma.clone(),
                        ),
                        LaunchBinding::new(
                            BufferRole::QueryNormalized,
                            Access::Write,
                            view(&sequential_query, query_bytes, query_offset, query_extent),
                        ),
                    ],
                    status_binding(sequential_status.clone()),
                )?
                .enqueue_owned(client, &sequential_status)?;
            program
                .key_head_rms(rows, epsilon)?
                .prepare_owned(
                    client,
                    vec![
                        LaunchBinding::new(
                            BufferRole::PackedAttention,
                            Access::Read,
                            view(&packed, packed_bytes, packed_offset, packed_extent),
                        ),
                        LaunchBinding::new(
                            BufferRole::KeyRawGamma,
                            Access::Read,
                            key_gamma.clone(),
                        ),
                        LaunchBinding::new(
                            BufferRole::KeyNormalized,
                            Access::Write,
                            view(&sequential_key, key_bytes, key_offset, key_extent),
                        ),
                    ],
                    status_binding(sequential_status.clone()),
                )?
                .enqueue_owned(client, &sequential_status)?;
            row += rows;
        }
        if row != active_rows {
            return Err(Error::State("ragged Q/K RMS fixture row coverage".into()));
        }

        let physical = StatelessRowExtent::bounded(active_rows, bound_rows)?;
        program
            .query_head_rms_stateless(physical, epsilon)?
            .prepare_owned(
                client,
                vec![
                    LaunchBinding::new(BufferRole::PackedAttention, Access::Read, packed.clone()),
                    LaunchBinding::new(
                        BufferRole::QueryRawGamma,
                        Access::Read,
                        query_gamma.clone(),
                    ),
                    LaunchBinding::new(
                        BufferRole::QueryNormalized,
                        Access::Write,
                        wide_query.clone(),
                    ),
                ],
                status_binding(wide_status.clone()),
            )?
            .enqueue_owned(client, &wide_status)?;
        program
            .key_head_rms_stateless(physical, epsilon)?
            .prepare_owned(
                client,
                vec![
                    LaunchBinding::new(BufferRole::PackedAttention, Access::Read, packed.clone()),
                    LaunchBinding::new(BufferRole::KeyRawGamma, Access::Read, key_gamma.clone()),
                    LaunchBinding::new(BufferRole::KeyNormalized, Access::Write, wide_key.clone()),
                ],
                status_binding(wide_status.clone()),
            )?
            .enqueue_owned(client, &wide_status)?;
        program
            .query_key_head_rms_stateless(physical, epsilon)?
            .prepare_owned(
                client,
                vec![
                    LaunchBinding::new(BufferRole::PackedAttention, Access::Read, packed.clone()),
                    LaunchBinding::new(
                        BufferRole::QueryRawGamma,
                        Access::Read,
                        query_gamma.clone(),
                    ),
                    LaunchBinding::new(BufferRole::KeyRawGamma, Access::Read, key_gamma.clone()),
                    LaunchBinding::new(
                        BufferRole::QueryNormalized,
                        Access::Write,
                        dual_query.clone(),
                    ),
                    LaunchBinding::new(BufferRole::KeyNormalized, Access::Write, dual_key.clone()),
                ],
                status_binding(dual_status.clone()),
            )?
            .enqueue_owned(client, &dual_status)?;

        client
            .sync()
            .await
            .map_err(|error| Error::State(format!("stateless Q/K RMS sync: {error}")))?;
        let sequential_query = client
            .read_one(sequential_query)
            .map_err(|error| Error::State(format!("causal Q RMS read: {error}")))?;
        let wide_query = client
            .read_one(wide_query)
            .map_err(|error| Error::State(format!("stateless Q RMS read: {error}")))?;
        let dual_query = client
            .read_one(dual_query)
            .map_err(|error| Error::State(format!("dual Q RMS read: {error}")))?;
        let sequential_key = client
            .read_one(sequential_key)
            .map_err(|error| Error::State(format!("causal K RMS read: {error}")))?;
        let wide_key = client
            .read_one(wide_key)
            .map_err(|error| Error::State(format!("stateless K RMS read: {error}")))?;
        let dual_key = client
            .read_one(dual_key)
            .map_err(|error| Error::State(format!("dual K RMS read: {error}")))?;
        let sequential_status = client
            .read_one(sequential_status)
            .map_err(|error| Error::State(format!("causal Q/K RMS status: {error}")))?;
        let wide_status = client
            .read_one(wide_status)
            .map_err(|error| Error::State(format!("stateless Q/K RMS status: {error}")))?;
        let dual_status = client
            .read_one(dual_status)
            .map_err(|error| Error::State(format!("dual Q/K RMS status: {error}")))?;

        if sequential_query != wide_query
            || dual_query != wide_query
            || sequential_key != wide_key
            || dual_key != wide_key
            || sequential_status != wide_status
            || dual_status != wide_status
            || &wide_status[..] != 2u32.to_ne_bytes().as_slice()
            || wide_query[active_query_bytes..] != query_seed[active_query_bytes..]
            || dual_query[active_query_bytes..] != query_seed[active_query_bytes..]
            || wide_key[active_key_bytes..] != key_seed[active_key_bytes..]
            || dual_key[active_key_bytes..] != key_seed[active_key_bytes..]
        {
            return Err(Error::State(
                "stateless/dual Q/K RMS differs from ragged causal calls".into(),
            ));
        }
        println!(
            "STATELESS_QK_RMS_PASS lanes=47..54 active_rows={active_rows} bound_rows={bound_rows} dual=true exact=true status=2"
        );

        client
            .memory_cleanup()
            .map_err(|error| Error::State(format!("stateless Q/K RMS cleanup: {error}")))?;
        service.close_model_namespace()?;
        Ok(())
    })
}

#[cfg(all(
    feature = "qualification",
    feature = "cuda-cublas",
    target_os = "linux"
))]
#[test]
#[ignore = "physical SM86 stateless sigmoid/multiply-vs-ragged-causal qualification"]
fn physical_stateless_sigmoid_multiply_matches_ragged_causal_calls_exactly() -> Result<()> {
    futures_lite::future::block_on(async {
        let diagnostic =
            crate::cuda::test_support::CudaDiagnostic::new("stateless-gate-elementwise")?;
        let service = diagnostic.service.clone();
        let client = service.client();
        let program = program();
        let width = program.geometry().query_width()?;
        let row_stride = program.attention_row_stride()?;
        let lane_rows = [47usize, 48, 49, 50, 51, 52, 53, 54];
        let active_rows = lane_rows.iter().sum::<usize>();
        let bound_rows = 513usize;
        let packed_bytes = bound_rows * row_stride * 2;
        let value_bytes = bound_rows * width * 2;

        let word = |i: usize, seed: usize| -> u16 {
            match (i + seed) % 8 {
                0 => 0x3f80,
                1 => 0xbf00,
                2 => 0x3e80,
                3 => 0x4000,
                4 => 0xbe00,
                5 => 0x3f00,
                6 => 0xc000,
                _ => 0x0000,
            }
        };
        let bytes_from_words = |words: &[u16]| {
            words
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let view = |handle: &Handle, total: usize, offset: usize, bytes: usize| {
            handle
                .clone()
                .offset_start(offset as u64)
                .offset_end((total - offset - bytes) as u64)
        };
        let status_binding =
            |handle: Handle| LaunchBinding::new(BufferRole::Status, Access::ReadWrite, handle);

        let mut packed_words = (0..bound_rows * row_stride)
            .map(|i| word(i, 5))
            .collect::<Vec<_>>();
        packed_words[(active_rows / 2) * row_stride + program.geometry().head_width] = 0x7f80;
        let input_words = (0..bound_rows * width)
            .map(|i| word(i, 13))
            .collect::<Vec<_>>();
        let packed = client.create_from_slice(&bytes_from_words(&packed_words));
        let input = client.create_from_slice(&bytes_from_words(&input_words));
        let seed = vec![0x5au8; value_bytes];
        let sequential_gate = client.create_from_slice(&seed);
        let wide_gate = client.create_from_slice(&seed);
        let sequential_output = client.create_from_slice(&seed);
        let wide_output = client.create_from_slice(&seed);
        let sequential_status = client.create_from_slice(&0u32.to_ne_bytes());
        let wide_status = client.create_from_slice(&0u32.to_ne_bytes());

        let mut row = 0usize;
        for rows in lane_rows {
            let packed_offset = row * row_stride * 2;
            let packed_extent = rows * row_stride * 2;
            let value_offset = row * width * 2;
            let value_extent = rows * width * 2;
            program
                .sigmoid(rows)?
                .prepare_owned(
                    client,
                    vec![
                        LaunchBinding::new(
                            BufferRole::PackedAttention,
                            Access::Read,
                            view(&packed, packed_bytes, packed_offset, packed_extent),
                        ),
                        LaunchBinding::new(
                            BufferRole::Output,
                            Access::Write,
                            view(&sequential_gate, value_bytes, value_offset, value_extent),
                        ),
                    ],
                    status_binding(sequential_status.clone()),
                )?
                .enqueue_owned(client, &sequential_status)?;
            program
                .multiply(rows)?
                .prepare_owned(
                    client,
                    vec![
                        LaunchBinding::new(
                            BufferRole::Input,
                            Access::Read,
                            view(&input, value_bytes, value_offset, value_extent),
                        ),
                        LaunchBinding::new(
                            BufferRole::Gate,
                            Access::Read,
                            view(&sequential_gate, value_bytes, value_offset, value_extent),
                        ),
                        LaunchBinding::new(
                            BufferRole::Output,
                            Access::Write,
                            view(&sequential_output, value_bytes, value_offset, value_extent),
                        ),
                    ],
                    status_binding(sequential_status.clone()),
                )?
                .enqueue_owned(client, &sequential_status)?;
            row += rows;
        }
        if row != active_rows {
            return Err(Error::State(
                "ragged stateless gate fixture row coverage".into(),
            ));
        }

        let physical = StatelessRowExtent::bounded(active_rows, bound_rows)?;
        program
            .sigmoid_stateless(physical)?
            .prepare_owned(
                client,
                vec![
                    LaunchBinding::new(BufferRole::PackedAttention, Access::Read, packed.clone()),
                    LaunchBinding::new(BufferRole::Output, Access::Write, wide_gate.clone()),
                ],
                status_binding(wide_status.clone()),
            )?
            .enqueue_owned(client, &wide_status)?;
        program
            .multiply_stateless(physical)?
            .prepare_owned(
                client,
                vec![
                    LaunchBinding::new(BufferRole::Input, Access::Read, input.clone()),
                    LaunchBinding::new(BufferRole::Gate, Access::Read, wide_gate.clone()),
                    LaunchBinding::new(BufferRole::Output, Access::Write, wide_output.clone()),
                ],
                status_binding(wide_status.clone()),
            )?
            .enqueue_owned(client, &wide_status)?;

        client
            .sync()
            .await
            .map_err(|error| Error::State(format!("stateless sigmoid/multiply sync: {error}")))?;
        let sequential_gate = client
            .read_one(sequential_gate)
            .map_err(|error| Error::State(format!("causal sigmoid gate read: {error}")))?;
        let wide_gate = client
            .read_one(wide_gate)
            .map_err(|error| Error::State(format!("stateless sigmoid gate read: {error}")))?;
        let sequential_output = client
            .read_one(sequential_output)
            .map_err(|error| Error::State(format!("causal multiply output read: {error}")))?;
        let wide_output = client
            .read_one(wide_output)
            .map_err(|error| Error::State(format!("stateless multiply output read: {error}")))?;
        let sequential_status = client
            .read_one(sequential_status)
            .map_err(|error| Error::State(format!("causal gate status read: {error}")))?;
        let wide_status = client
            .read_one(wide_status)
            .map_err(|error| Error::State(format!("stateless gate status read: {error}")))?;
        let active_bytes = active_rows * width * 2;
        if sequential_gate != wide_gate
            || sequential_output != wide_output
            || sequential_status != wide_status
            || wide_gate[active_bytes..] != seed[active_bytes..]
            || wide_output[active_bytes..] != seed[active_bytes..]
            || &wide_status[..] != 2u32.to_ne_bytes().as_slice()
        {
            return Err(Error::State(
                "stateless sigmoid/multiply differs from ragged causal calls".into(),
            ));
        }
        println!(
            "STATELESS_GATE_PASS lanes=47..54 active_rows={active_rows} bound_rows={bound_rows} exact=true status=2"
        );

        client.memory_cleanup().map_err(|error| {
            Error::State(format!("stateless sigmoid/multiply cleanup: {error}"))
        })?;
        service.close_model_namespace()?;
        Ok(())
    })
}

#[cfg(all(
    feature = "qualification",
    feature = "cuda-cublas",
    target_os = "linux"
))]
#[test]
#[ignore = "physical SM86 sigmoid+multiply fused-vs-sequential qualification"]
fn physical_sigmoid_multiply_matches_sequential_exactly() -> Result<()> {
    futures_lite::future::block_on(async {
        let diagnostic = crate::cuda::test_support::CudaDiagnostic::new("sigmoid-multiply")?;
        let service = diagnostic.service.clone();
        let client = service.client();
        let program = program();
        let geometry = program.geometry();
        let query_width = geometry.query_width()?;
        let packed_stride = 2 * (geometry.query_width()? + geometry.kv_width()?);
        let gate_offset = geometry.head_width;
        let head_stride = 2 * geometry.head_width;

        let bytes_from_words = |words: &[u16]| {
            words
                .iter()
                .flat_map(|word| word.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        let word = |i: usize, seed: usize| -> u16 {
            match (i + seed) % 8 {
                0 => 0x3f80, // 1.0
                1 => 0xbf00, // -0.5
                2 => 0x3e80, // 0.25
                3 => 0x4000, // 2.0
                4 => 0xbe00, // -0.125
                5 => 0x3f00, // 0.5
                6 => 0xc000, // -2.0
                _ => 0x0000,
            }
        };

        for rows in [1usize, 2, 8, 17, 64] {
            let packed_elements = rows * packed_stride;
            let mut packed_words = (0..packed_elements)
                .map(|i| word(i, rows))
                .collect::<Vec<_>>();
            let input_words = (0..rows * query_width)
                .map(|i| word(i, rows + 3))
                .collect::<Vec<_>>();
            if rows == 17 {
                let poison = 3 * packed_stride + 2 * head_stride + gate_offset + 7;
                packed_words[poison] = 0x7f80; // +inf: both paths must flag nonfinite.
            }

            let packed = client.create_from_slice(&bytes_from_words(&packed_words));
            let input = client.create_from_slice(&bytes_from_words(&input_words));
            let value_bytes = rows * query_width * 2;
            let zeros = vec![0u8; value_bytes];
            let sequential_gate = client.create_from_slice(&zeros);
            let sequential_output = client.create_from_slice(&zeros);
            let fused_gate = client.create_from_slice(&zeros);
            let fused_output = client.create_from_slice(&zeros);
            let sequential_status = client.create_from_slice(&0u32.to_ne_bytes());
            let fused_status = client.create_from_slice(&0u32.to_ne_bytes());

            let status_binding =
                |handle: Handle| LaunchBinding::new(BufferRole::Status, Access::ReadWrite, handle);
            let sigmoid = program.sigmoid(rows)?;
            let prepared = sigmoid.prepare_owned(
                client,
                vec![
                    LaunchBinding::new(BufferRole::PackedAttention, Access::Read, packed.clone()),
                    LaunchBinding::new(BufferRole::Output, Access::Write, sequential_gate.clone()),
                ],
                status_binding(sequential_status.clone()),
            )?;
            prepared.enqueue_owned(client, &sequential_status)?;

            let multiply = program.multiply(rows)?;
            let prepared = multiply.prepare_owned(
                client,
                vec![
                    LaunchBinding::new(BufferRole::Input, Access::Read, input.clone()),
                    LaunchBinding::new(BufferRole::Gate, Access::Read, sequential_gate.clone()),
                    LaunchBinding::new(
                        BufferRole::Output,
                        Access::Write,
                        sequential_output.clone(),
                    ),
                ],
                status_binding(sequential_status.clone()),
            )?;
            prepared.enqueue_owned(client, &sequential_status)?;

            let fused = program.sigmoid_multiply(rows)?;
            let prepared = fused.prepare_owned(
                client,
                vec![
                    LaunchBinding::new(BufferRole::PackedAttention, Access::Read, packed.clone()),
                    LaunchBinding::new(BufferRole::Input, Access::Read, input.clone()),
                    LaunchBinding::new(BufferRole::Gate, Access::Write, fused_gate.clone()),
                    LaunchBinding::new(BufferRole::Output, Access::Write, fused_output.clone()),
                ],
                status_binding(fused_status.clone()),
            )?;
            prepared.enqueue_owned(client, &fused_status)?;

            client
                .sync()
                .await
                .map_err(|error| Error::State(format!("sigmoid-multiply sync: {error}")))?;
            let read = |handle: Handle, role: &str| {
                client
                    .read_one(handle)
                    .map_err(|error| Error::State(format!("sigmoid-multiply {role}: {error}")))
            };
            let sequential_gate_bytes = read(sequential_gate, "sequential gate")?;
            let fused_gate_bytes = read(fused_gate, "fused gate")?;
            let sequential_output_bytes = read(sequential_output, "sequential output")?;
            let fused_output_bytes = read(fused_output, "fused output")?;
            let sequential_status_bytes = read(sequential_status, "sequential status")?;
            let fused_status_bytes = read(fused_status, "fused status")?;
            if sequential_gate_bytes != fused_gate_bytes
                || sequential_output_bytes != fused_output_bytes
                || sequential_status_bytes != fused_status_bytes
            {
                return Err(Error::State(format!(
                    "sigmoid-multiply physical mismatch rows={rows}"
                )));
            }
            let expected_status = if rows == 17 { 2u32 } else { 0 };
            if &sequential_status_bytes[..] != expected_status.to_ne_bytes().as_slice() {
                return Err(Error::State(format!(
                    "sigmoid-multiply status rows={rows}: {sequential_status_bytes:?}"
                )));
            }
            println!(
                "SIGMOID_MULTIPLY_PASS rows={rows} exact_gate=true exact_output=true status={expected_status}"
            );
        }

        client
            .memory_cleanup()
            .map_err(|error| Error::State(format!("sigmoid-multiply cleanup: {error}")))?;
        service.close_model_namespace()?;
        Ok(())
    })
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
    assert_eq!(compile_reflection_diagnostic(&queue)?, 30);
    Ok(())
}
