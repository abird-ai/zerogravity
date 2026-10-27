use super::*;

fn geometry() -> Geometry {
    Geometry {
        hidden_width: 1024,
        intermediate_width: 3584,
        vocabulary: 248_320,
        query_heads: 8,
        kv_heads: 2,
        head_width: 256,
        rotary_width: 64,
    }
}

fn program() -> Program {
    Program::new(geometry()).unwrap()
}

#[test]
fn complete_family_has_source_derived_entries_and_no_capture_coordinates() -> Result<()> {
    let definitions = program().definitions(Step::new(64, 32_704, 32_768)?, 1e-6)?;
    assert_eq!(definitions.len(), 11);
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
            Primitive::SigmoidGate,
            Primitive::Residual,
            Primitive::KvAppend,
            Primitive::Attention,
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
            (Gamma, 1024 * 2, R),
            (Output, plan.hidden_batch_bytes, W),
            (Status, 4, RW),
        ])
    );
    assert_eq!(
        p.kv_append(step)?.bindings(),
        expected(&[
            (Key, plan.kv_batch_bytes, R),
            (Value, plan.kv_batch_bytes, R),
            (KeyCache, plan.key_cache_bytes, RW),
            (ValueCache, plan.value_cache_bytes, RW),
            (Status, 4, RW),
        ])
    );
    assert_eq!(
        p.attention(step)?.bindings(),
        expected(&[
            (Query, plan.query_batch_bytes, R),
            (KeyCache, plan.key_cache_bytes, R),
            (ValueCache, plan.value_cache_bytes, R),
            (Output, plan.query_batch_bytes, W),
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
fn memory_plan_covers_sm86_qwen_extents_without_allocating() -> Result<()> {
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
    Ok(())
}

#[test]
fn launch_geometry_is_runtime_derived_not_enumerated() -> Result<()> {
    let p = program();
    for rows in 1..=64 {
        let step = Step::new(rows, 19, 19 + rows)?;
        assert_eq!(p.positions(step)?.grid(), [1, 1, 1]);
        assert_eq!(p.rms(rows, 1e-6)?.grid(), [rows as u32, 1, 1]);
        assert_eq!(p.attention(step)?.grid(), [(rows * 8) as u32, 1, 1]);
        assert_eq!(p.greedy(rows)?.grid(), [rows as u32, 1, 1]);
    }
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "SM86 CUDA compiler/driver ABI diagnostic; compile and load only"]
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
    for definition in program().definitions(Step::new(64, 32_704, 32_768)?, 1e-6)? {
        definition.prepare_code(&queue)?;
    }
    Ok(())
}
