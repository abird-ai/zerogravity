//! Explicit physical/source qualification entry point. No production frontend.
use crate::*;

use cubecl::{Device, device::WgpuDeviceKind, prelude::*};

#[cube(launch_unchecked)]
fn multiply_by_two(input: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 2.0;
    }
}

pub fn run() {
    let mut args = std::env::args().skip(1);
    let Some(selected) = args.next() else {
        print_help();
        return;
    };
    if matches!(selected.as_str(), "--help" | "-h") {
        print_help();
        return;
    }
    if selected == "--version" {
        println!("zerogravity-gpu {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    #[cfg(feature = "hardware-abort-probe")]
    let invalid_quant_mode = selected.strip_prefix("model-invalid-device-quant-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_mode = selected.strip_prefix("model-abort-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_norm_mode = selected.strip_prefix("model-abort-norm-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_grouped_mode = selected.strip_prefix("model-abort-grouped-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_device_quant_mode = selected.strip_prefix("model-abort-device-quant-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_embedding_mode = selected.strip_prefix("model-abort-embedding-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_residual_mode = selected.strip_prefix("model-abort-residual-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_attention_gate_mode = selected.strip_prefix("model-abort-attention-gate-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_ffn_mode = selected.strip_prefix("model-abort-ffn-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_value_mode = selected.strip_prefix("model-abort-value-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_softmax_mode = selected.strip_prefix("model-abort-softmax-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_qk_mode = selected.strip_prefix("model-abort-qk-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_rope_mode = selected.strip_prefix("model-abort-rope-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_coefficient_mode = selected.strip_prefix("model-abort-coefficients-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_silu_mode = selected.strip_prefix("model-abort-silu-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_gate_mode = selected.strip_prefix("model-abort-gate-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_l2_mode = selected.strip_prefix("model-abort-l2-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_recurrent_mode = selected.strip_prefix("model-abort-recurrent-");
    #[cfg(feature = "hardware-abort-probe")]
    let abort_conv_mode = selected.strip_prefix("model-abort-conv-");
    let resident_cohort_mode = selected.strip_prefix("resident-cohort-");
    let resident_fused_mode = selected.strip_prefix("resident-fused-");
    let resident_fused_abort_mode = selected.strip_prefix("resident-fused-abort-");
    let resident_fused_boundary_mode = selected.strip_prefix("resident-fused-boundary-");
    let resident_fused_greedy_mode = selected.strip_prefix("resident-fused-greedy-");
    let resident_fused_greedy_abort_mode = selected.strip_prefix("resident-fused-greedy-abort-");
    let resident_fused_mode = resident_fused_greedy_abort_mode
        .or(resident_fused_greedy_mode)
        .or(resident_fused_boundary_mode)
        .or(resident_fused_abort_mode)
        .or(resident_fused_mode);
    let resident_run_mode = selected.strip_prefix("run-");
    let checkpoint_abort_mode = selected.strip_prefix("resident-checkpoint-abort-");
    let checkpoint_mode =
        checkpoint_abort_mode.or_else(|| selected.strip_prefix("resident-checkpoint-"));
    let resident_abort_mode = selected.strip_prefix("resident-abort-");
    let resident_context_abort_mode = selected.strip_prefix("resident-context-abort-");
    let resident_context_mode =
        resident_context_abort_mode.or_else(|| selected.strip_prefix("resident-context-"));
    let resident_graph_mode = selected
        .strip_prefix("resident-qualify-")
        .or(resident_abort_mode)
        .or(resident_context_mode)
        .or(checkpoint_mode)
        .or(resident_cohort_mode);
    let resident_graph_mode = resident_graph_mode.or(resident_fused_mode);
    let quant_stress_mode = selected.strip_prefix("quant-stress-");
    let device_quant_mode = selected.strip_prefix("model-device-quant-");
    let operators_mode = selected.strip_prefix("model-operators-");
    let elementwise_stress_mode = selected.strip_prefix("elementwise-stress-");
    let value_stress_mode = selected.strip_prefix("value-stress-");
    let value_mode = selected.strip_prefix("model-value-");
    let softmax_stress_mode = selected.strip_prefix("softmax-stress-");
    let softmax_mode = selected.strip_prefix("model-softmax-");
    let qk_stress_mode = selected.strip_prefix("qk-stress-");
    let qk_mode = selected.strip_prefix("model-qk-");
    let rope_stress_mode = selected.strip_prefix("rope-stress-");
    let rope_mode = selected.strip_prefix("model-rope-");
    let coefficient_stress_mode = selected.strip_prefix("coefficient-stress-");
    let coefficient_mode = selected.strip_prefix("model-coefficients-");
    let silu_stress_mode = selected.strip_prefix("silu-stress-");
    let silu_mode = selected.strip_prefix("model-silu-");
    let l2_mode = selected.strip_prefix("model-l2-");
    let recurrent_layer_mode = selected.strip_prefix("model-recurrent-layer-");
    let recurrent_mode = selected
        .strip_prefix("model-recurrent-")
        .or(l2_mode)
        .or(silu_mode)
        .or(coefficient_mode)
        .or(rope_mode)
        .or(qk_mode)
        .or(softmax_mode)
        .or(value_mode)
        .or(operators_mode)
        .or(device_quant_mode);
    let resident_mode = selected.strip_prefix("model-resident-");
    let conv_multi_mode = selected.strip_prefix("model-conv-multi-");
    let conv_mode = selected.strip_prefix("model-conv-");
    let heads_multi_mode = selected.strip_prefix("model-heads-multi-");
    let heads_mode = selected.strip_prefix("model-heads-");
    let layer_ops_mode = selected.strip_prefix("model-layer-ops-");
    let ops_mode = selected.strip_prefix("model-ops-");
    let layer_mode = selected.strip_prefix("model-layer-");
    let gemv_mode = selected.strip_prefix("model-gemv-");
    let sessions_mode = selected.strip_prefix("model-sessions-");
    let benchmark_mode = selected.strip_prefix("model-bench-");
    let batch_projection_mode = selected.strip_prefix("batch-projection-");
    let batch_q8_mode = selected.strip_prefix("batch-q8-projection-");
    let batch_elementwise_mode = selected.strip_prefix("batch-elementwise-");
    let batch_recurrent_mode = selected.strip_prefix("batch-recurrent-");
    let batch_kv_mode = selected.strip_prefix("batch-kv-");
    let batch_attention_mode = selected.strip_prefix("batch-attention-");
    let native_workers = selected.starts_with("model-workers-native-");
    let workers_mode = selected
        .strip_prefix("model-workers-native-")
        .or_else(|| selected.strip_prefix("model-workers-"));
    let (name, q4_mode, model_mode, mixed_mode) = if let Some(backend) = batch_attention_mode {
        (backend, false, false, false)
    } else if let Some(backend) = batch_kv_mode {
        (backend, false, false, false)
    } else if let Some(backend) = batch_recurrent_mode {
        (backend, false, false, false)
    } else if let Some(backend) = batch_elementwise_mode {
        (backend, false, false, false)
    } else if let Some(backend) = batch_projection_mode {
        (backend, false, false, false)
    } else if let Some(backend) = batch_q8_mode {
        (backend, false, false, false)
    } else if let Some(backend) = resident_run_mode
        .or(resident_graph_mode)
        .or(resident_mode)
        .or(silu_stress_mode)
        .or(coefficient_stress_mode)
        .or(rope_stress_mode)
        .or(qk_stress_mode)
        .or(softmax_stress_mode)
        .or(value_stress_mode)
        .or(elementwise_stress_mode)
        .or(quant_stress_mode)
    {
        (backend, false, false, false)
    } else if let Some(backend) = recurrent_layer_mode
        .or(recurrent_mode)
        .or(conv_multi_mode)
        .or(conv_mode)
        .or(heads_multi_mode)
        .or(heads_mode)
        .or(layer_ops_mode)
        .or(ops_mode)
        .or(layer_mode)
        .or(gemv_mode)
    {
        (backend, false, false, false)
    } else if let Some(backend) = workers_mode {
        (backend, false, false, false)
    } else if let Some(backend) = sessions_mode {
        (backend, false, false, false)
    } else if let Some(backend) = benchmark_mode {
        (backend, false, false, false)
    } else if let Some(backend) = {
        #[cfg(feature = "hardware-abort-probe")]
        {
            invalid_quant_mode
                .or(abort_recurrent_mode)
                .or(abort_l2_mode)
                .or(abort_coefficient_mode)
                .or(abort_rope_mode)
                .or(abort_qk_mode)
                .or(abort_device_quant_mode)
                .or(abort_embedding_mode)
                .or(abort_residual_mode)
                .or(abort_attention_gate_mode)
                .or(abort_ffn_mode)
                .or(abort_value_mode)
                .or(abort_softmax_mode)
                .or(abort_silu_mode)
                .or(abort_gate_mode)
                .or(abort_conv_mode)
                .or(abort_grouped_mode)
                .or(abort_norm_mode)
                .or(abort_mode)
        }
        #[cfg(not(feature = "hardware-abort-probe"))]
        {
            None
        }
    } {
        (backend, false, false, false)
    } else if let Some(backend) = selected.strip_prefix("model-mixed-") {
        (backend, false, false, true)
    } else if let Some(backend) = selected.strip_prefix("model-q4-") {
        (backend, false, true, false)
    } else {
        match selected.strip_prefix("q4-") {
            Some(backend) => (backend, true, false, false),
            None => (selected.as_str(), false, false, false),
        }
    };
    if resident_run_mode.is_some() {
        panic!("GPU text generation moved to zerogravity run-{name}; use the production CLI");
    }
    let device = match name {
        "cuda" => Device::cuda(0).expect("CUDA hardware/runtime not available"),
        "rocm" => Device::rocm(0).expect("AMD ROCm hardware/runtime not available"),
        "vulkan-amd" | "amd" => Device::vulkan(WgpuDeviceKind::IntegratedGpu(0))
            .expect("AMD Vulkan adapter not available"),
        "vulkan-nvidia" | "nvidia" => Device::vulkan(WgpuDeviceKind::DiscreteGpu(0))
            .expect("NVIDIA Vulkan adapter not available"),
        _ => panic!("unknown backend: {name}; choose cuda, rocm, vulkan-amd or vulkan-nvidia"),
    };
    if batch_attention_mode.is_some() {
        batch_attention::qualify(device)
            .unwrap_or_else(|error| panic!("P12 physical B>1 attention failed: {error}"));
        return;
    }
    if batch_kv_mode.is_some() {
        batch_kv::qualify(device)
            .unwrap_or_else(|error| panic!("P12 physical B>1 causal KV failed: {error}"));
        return;
    }
    if batch_recurrent_mode.is_some() {
        batch_recurrent::qualify(device)
            .unwrap_or_else(|error| panic!("P12 physical B>1 recurrent update failed: {error}"));
        return;
    }
    if batch_elementwise_mode.is_some() {
        batch_elementwise::qualify(device)
            .unwrap_or_else(|error| panic!("P12 physical B>1 source elementwise failed: {error}"));
        return;
    }
    if batch_projection_mode.is_some() {
        batch_projection::qualify(device)
            .unwrap_or_else(|error| panic!("P12 physical B>1 F32 projection failed: {error}"));
        return;
    }
    if batch_q8_mode.is_some() {
        batch_quant_projection::qualify(device).unwrap_or_else(|error| {
            panic!("P12 physical B>1 original Q8 projection failed: {error}")
        });
        return;
    }
    if resident_graph_mode.is_some() {
        let path = args
            .next()
            .expect("resident qualification requires original GGUF");
        let context = resident_context_mode.map(|_| {
            (
                args.next()
                    .expect("resident context requires UTF8 prompt file"),
                args.next()
                    .expect("resident context requires independent raw IDs file"),
            )
        });
        let checkpoint = checkpoint_mode.map(|_| {
            (
                args.next()
                    .expect("checkpoint qualification requires state file path"),
                args.next()
                    .expect("checkpoint requires independently verified model SHA256"),
                args.next()
                    .expect("checkpoint requires independently verified token SHA256"),
            )
        });
        let provider_mib: usize = args
            .next()
            .map(|s| s.parse().expect("provider MiB"))
            .unwrap_or(2048);
        let device_mib: u64 = args
            .next()
            .map(|s| s.parse().expect("physical operator MiB"))
            .unwrap_or(4096);
        let provider = provider_mib
            .checked_mul(1024 * 1024)
            .expect("provider cap overflow");
        let cap = device_mib
            .checked_mul(1024 * 1024)
            .expect("device cap overflow");
        if resident_cohort_mode.is_some() || resident_fused_mode.is_some() {
            let rows = args
                .next()
                .map(|v| {
                    v.parse::<usize>()
                        .expect("resident cohort rows must be 2,4,8")
                })
                .unwrap_or(2);
            let qualification = async {
                if resident_fused_greedy_abort_mode.is_some() {
                    #[cfg(feature = "hardware-abort-probe")]
                    return resident_inference::qualify_fused_greedy_abort(
                        device,
                        std::path::Path::new(&path),
                        provider,
                        cap,
                    )
                    .await;
                    #[cfg(not(feature = "hardware-abort-probe"))]
                    return Err(zerogravity_core::Error::Unsupported(
                        "requires hardware-abort-probe".into(),
                    ));
                } else if resident_fused_greedy_mode.is_some() {
                    resident_inference::qualify_fused_greedy(
                        device,
                        std::path::Path::new(&path),
                        provider,
                        cap,
                        rows,
                    )
                    .await
                } else if resident_fused_boundary_mode.is_some() {
                    #[cfg(feature = "portable-checkpoint")]
                    return resident_inference::qualify_fused_boundary(
                        device,
                        std::path::Path::new(&path),
                        provider,
                        cap,
                    )
                    .await;
                    #[cfg(not(feature = "portable-checkpoint"))]
                    return Err(zerogravity_core::Error::Unsupported(
                        "requires portable-checkpoint".into(),
                    ));
                } else if resident_fused_abort_mode.is_some() {
                    #[cfg(feature = "hardware-abort-probe")]
                    return resident_inference::qualify_fused_abort(
                        device,
                        std::path::Path::new(&path),
                        provider,
                        cap,
                    )
                    .await;
                    #[cfg(not(feature = "hardware-abort-probe"))]
                    return Err(zerogravity_core::Error::Unsupported(
                        "requires hardware-abort-probe".into(),
                    ));
                } else if resident_fused_mode.is_some() {
                    resident_inference::qualify_fused(
                        device,
                        std::path::Path::new(&path),
                        provider,
                        cap,
                        rows,
                    )
                    .await
                } else {
                    resident_inference::qualify_cohort(
                        device,
                        std::path::Path::new(&path),
                        provider,
                        cap,
                        rows,
                    )
                    .await
                }
            };
            futures_lite::future::block_on(qualification).unwrap_or_else(|e| {
                panic!("resident cohort state publication qualification failed: {e}")
            });
            return;
        }
        if let Some((file, model_sha, tokens_sha)) = checkpoint {
            #[cfg(feature = "portable-checkpoint")]
            futures_lite::future::block_on(resident_inference::qualify_checkpoint(
                device,
                std::path::Path::new(&path),
                std::path::Path::new(&file),
                (&model_sha, &tokens_sha),
                provider,
                cap,
                checkpoint_abort_mode.is_some(),
            ))
            .unwrap_or_else(|e| panic!("resident checkpoint qualification failed: {e}"));
            #[cfg(not(feature = "portable-checkpoint"))]
            {
                let _ = (file, model_sha, tokens_sha);
                panic!("resident checkpoint qualification requires portable-checkpoint feature");
            }
            #[cfg(feature = "portable-checkpoint")]
            return;
        }
        #[cfg(feature = "hardware-abort-probe")]
        if resident_context_abort_mode.is_some() {
            let (prompt, ids) = context.as_ref().expect("context abort input binding");
            futures_lite::future::block_on(resident_inference::qualify_context_abort(
                device,
                std::path::Path::new(&path),
                std::path::Path::new(prompt),
                std::path::Path::new(ids),
                provider,
                cap,
            ))
            .unwrap_or_else(|e| panic!("resident context abort qualification failed: {e}"));
            return;
        }
        #[cfg(not(feature = "hardware-abort-probe"))]
        assert!(
            resident_context_abort_mode.is_none(),
            "resident context abort requires hardware-abort-probe feature"
        );
        if let Some((prompt, ids)) = context {
            futures_lite::future::block_on(resident_inference::qualify_context(
                device,
                std::path::Path::new(&path),
                std::path::Path::new(&prompt),
                std::path::Path::new(&ids),
                provider,
                cap,
            ))
            .unwrap_or_else(|e| panic!("resident context qualification failed: {e}"));
            return;
        }
        #[cfg(feature = "hardware-abort-probe")]
        if resident_abort_mode.is_some() {
            futures_lite::future::block_on(resident_inference::qualify_abort(
                device,
                std::path::Path::new(&path),
                provider,
                cap,
            ))
            .unwrap_or_else(|e| panic!("resident abort qualification failed: {e}"));
            return;
        }
        #[cfg(not(feature = "hardware-abort-probe"))]
        assert!(
            resident_abort_mode.is_none(),
            "resident abort requires hardware-abort-probe feature"
        );
        futures_lite::future::block_on(resident_inference::qualify(
            device,
            std::path::Path::new(&path),
            provider,
            cap,
        ))
        .unwrap_or_else(|e| panic!("resident graph qualification failed: {e}"));
        return;
    }
    #[cfg(feature = "hardware-abort-probe")]
    if invalid_quant_mode.is_some() {
        let model = args.next().expect("invalid quant diagnostic requires GGUF");
        physical_abort_probe::run_invalid_quant(device, std::path::Path::new(&model))
            .unwrap_or_else(|e| panic!("physical quant status check failed: {e}"));
        return;
    }
    #[cfg(feature = "hardware-abort-probe")]
    if abort_mode.is_some() {
        let model = args
            .next()
            .expect("model-abort-<backend> requires source GGUF");
        if abort_device_quant_mode.is_some() {
            physical_abort_probe::run_device_quant(device, std::path::Path::new(&model))
                .unwrap_or_else(|e| panic!("physical device quant abort failed: {e}"));
            return;
        }
        if abort_embedding_mode.is_some()
            || abort_residual_mode.is_some()
            || abort_attention_gate_mode.is_some()
            || abort_ffn_mode.is_some()
        {
            let kind = if abort_embedding_mode.is_some() {
                "embedding"
            } else if abort_residual_mode.is_some() {
                "residual"
            } else if abort_attention_gate_mode.is_some() {
                "attention-gate"
            } else {
                "ffn"
            };
            physical_abort_probe::run_remaining(device, std::path::Path::new(&model), kind)
                .unwrap_or_else(|e| panic!("physical remaining abort failed: {e}"));
            return;
        }
        if abort_value_mode.is_some() {
            physical_abort_probe::run_value(device, std::path::Path::new(&model))
                .unwrap_or_else(|e| panic!("physical value abort failed: {e}"));
            return;
        }
        if abort_softmax_mode.is_some() {
            physical_abort_probe::run_softmax(device, std::path::Path::new(&model))
                .unwrap_or_else(|e| panic!("physical softmax abort failed: {e}"));
            return;
        }
        if abort_qk_mode.is_some() {
            physical_abort_probe::run_qk(device, std::path::Path::new(&model))
                .unwrap_or_else(|e| panic!("physical QK abort failed: {e}"));
            return;
        }
        if abort_rope_mode.is_some() {
            physical_abort_probe::run_rope(device, std::path::Path::new(&model))
                .unwrap_or_else(|e| panic!("physical RoPE abort failed: {e}"));
            return;
        }
        if abort_coefficient_mode.is_some() {
            physical_abort_probe::run_coefficients(device, std::path::Path::new(&model))
                .unwrap_or_else(|e| panic!("physical coefficient abort failed: {e}"));
            return;
        }
        if abort_silu_mode.is_some() || abort_gate_mode.is_some() {
            physical_abort_probe::run_silu(
                device,
                std::path::Path::new(&model),
                abort_gate_mode.is_some(),
            )
            .unwrap_or_else(|e| panic!("physical SiLU abort failed: {e}"));
            return;
        }
        if abort_l2_mode.is_some() {
            physical_abort_probe::run_l2(device, std::path::Path::new(&model))
                .unwrap_or_else(|e| panic!("physical GDN L2 readback-abort proof failed: {e}"));
        } else if abort_recurrent_mode.is_some() {
            physical_abort_probe::run_recurrent(device, std::path::Path::new(&model))
                .unwrap_or_else(|e| panic!("physical recurrent readback-abort proof failed: {e}"));
        } else if abort_conv_mode.is_some() {
            physical_abort_probe::run_conv(device, std::path::Path::new(&model)).unwrap_or_else(
                |e| panic!("physical GDN convolution readback-abort proof failed: {e}"),
            );
        } else if abort_grouped_mode.is_some() {
            physical_abort_probe::run_grouped_norm(device, std::path::Path::new(&model))
                .unwrap_or_else(|e| {
                    panic!("physical grouped-RMS readback-abort proof failed: {e}")
                });
        } else if abort_norm_mode.is_some() {
            physical_abort_probe::run_norm(device, std::path::Path::new(&model))
                .unwrap_or_else(|e| panic!("physical RMSNorm readback-abort proof failed: {e}"));
        } else {
            physical_abort_probe::run(device, std::path::Path::new(&model))
                .unwrap_or_else(|e| panic!("physical readback-abort proof failed: {e}"));
        }
        return;
    }
    if benchmark_mode.is_some() {
        let model = args
            .next()
            .expect("model-bench-<backend> requires original Qwen GGUF");
        model_bench::run(device, std::path::Path::new(&model)).unwrap_or_else(|error| {
            panic!("source-audited physical GPU benchmark failed: {error}")
        });
        return;
    }
    if recurrent_mode.is_some()
        || recurrent_layer_mode.is_some()
        || layer_mode.is_some()
        || gemv_mode.is_some()
        || layer_ops_mode.is_some()
        || ops_mode.is_some()
        || heads_mode.is_some()
        || heads_multi_mode.is_some()
        || conv_multi_mode.is_some()
        || conv_mode.is_some()
    {
        let model = args
            .next()
            .expect("model-layer/gemv/ops-<backend> requires original Qwen GGUF");
        let blocks = if recurrent_layer_mode.is_some()
            || layer_ops_mode.is_some()
            || (layer_mode.is_some() && ops_mode.is_none())
        {
            1
        } else {
            24
        };
        let include_output = recurrent_mode.is_some()
            || recurrent_layer_mode.is_some()
            || conv_multi_mode.is_some()
            || conv_mode.is_some()
            || heads_multi_mode.is_some()
            || heads_mode.is_some()
            || gemv_mode.is_some()
            || ops_mode.is_some();
        let include_norms = recurrent_mode.is_some()
            || recurrent_layer_mode.is_some()
            || conv_multi_mode.is_some()
            || conv_mode.is_some()
            || heads_multi_mode.is_some()
            || heads_mode.is_some()
            || layer_ops_mode.is_some()
            || ops_mode.is_some();
        model_layer::run(
            device,
            std::path::Path::new(&model),
            model_layer::SweepConfig {
                blocks,
                output_head: include_output,
                residual_norms: include_norms,
                grouped_heads: recurrent_mode.is_some()
                    || recurrent_layer_mode.is_some()
                    || heads_mode.is_some()
                    || heads_multi_mode.is_some()
                    || conv_multi_mode.is_some()
                    || conv_mode.is_some(),
                tokens: if recurrent_mode.is_some() || recurrent_layer_mode.is_some() {
                    4
                } else if heads_multi_mode.is_some() || conv_multi_mode.is_some() {
                    2
                } else {
                    1
                },
                gdn_recurrent: recurrent_mode.is_some() || recurrent_layer_mode.is_some(),
                gdn_convolution: recurrent_mode.is_some()
                    || recurrent_layer_mode.is_some()
                    || conv_multi_mode.is_some()
                    || conv_mode.is_some(),
                gdn_l2: l2_mode.is_some()
                    || silu_mode.is_some()
                    || coefficient_mode.is_some()
                    || rope_mode.is_some()
                    || qk_mode.is_some()
                    || (softmax_mode.is_some()
                        || (value_mode.is_some()
                            || (operators_mode.is_some() || device_quant_mode.is_some()))),
                gdn_silu: silu_mode.is_some()
                    || coefficient_mode.is_some()
                    || rope_mode.is_some()
                    || qk_mode.is_some()
                    || (softmax_mode.is_some()
                        || (value_mode.is_some()
                            || (operators_mode.is_some() || device_quant_mode.is_some()))),
                gdn_coefficients: coefficient_mode.is_some()
                    || rope_mode.is_some()
                    || qk_mode.is_some()
                    || (softmax_mode.is_some()
                        || (value_mode.is_some()
                            || (operators_mode.is_some() || device_quant_mode.is_some()))),
                rope_kv: rope_mode.is_some()
                    || qk_mode.is_some()
                    || (softmax_mode.is_some()
                        || (value_mode.is_some()
                            || (operators_mode.is_some() || device_quant_mode.is_some()))),
                attention_qk: qk_mode.is_some()
                    || (softmax_mode.is_some()
                        || (value_mode.is_some()
                            || (operators_mode.is_some() || device_quant_mode.is_some()))),
                attention_softmax: (softmax_mode.is_some()
                    || (value_mode.is_some()
                        || (operators_mode.is_some() || device_quant_mode.is_some()))),
                attention_value: (value_mode.is_some()
                    || (operators_mode.is_some() || device_quant_mode.is_some())),
                remaining_operators: (operators_mode.is_some() || device_quant_mode.is_some()),
                device_activation: device_quant_mode.is_some(),
            },
        )
        .unwrap_or_else(|error| panic!("source-strict GPU model matrix placement failed: {error}"));
        return;
    }
    if quant_stress_mode.is_some() {
        let model = args.next().expect("quant stress needs source GGUF");
        activation_stress::run(device, std::path::Path::new(&model))
            .unwrap_or_else(|e| panic!("quant stress failed: {e}"));
        return;
    }
    if elementwise_stress_mode.is_some() {
        let model = args
            .next()
            .expect("elementwise stress requires source GGUF");
        operator_stress::run(device, std::path::Path::new(&model))
            .unwrap_or_else(|e| panic!("elementwise stress failed: {e}"));
        return;
    }
    if value_stress_mode.is_some() {
        let model = args.next().expect("value stress requires existing GGUF");
        model_layer::value_stress(device, std::path::Path::new(&model))
            .unwrap_or_else(|e| panic!("value stress failed: {e}"));
        return;
    }
    if softmax_stress_mode.is_some() {
        let model = args.next().expect("softmax stress requires existing GGUF");
        model_layer::softmax_stress(device, std::path::Path::new(&model))
            .unwrap_or_else(|e| panic!("softmax stress failed: {e}"));
        return;
    }
    if qk_stress_mode.is_some() {
        let model = args.next().expect("qk-stress requires source GGUF");
        model_layer::qk_stress(device, std::path::Path::new(&model))
            .unwrap_or_else(|e| panic!("physical QK stress failed: {e}"));
        return;
    }
    if rope_stress_mode.is_some() {
        let model = args.next().expect("rope-stress requires source GGUF");
        model_layer::rope_stress(device, std::path::Path::new(&model))
            .unwrap_or_else(|e| panic!("RoPE stress failed: {e}"));
        return;
    }
    if coefficient_stress_mode.is_some() {
        let model = args
            .next()
            .expect("coefficient-stress requires source GGUF");
        model_layer::coefficient_stress(device, std::path::Path::new(&model))
            .unwrap_or_else(|e| panic!("coefficient arithmetic stress failed: {e}"));
        return;
    }
    if silu_stress_mode.is_some() {
        let model = args.next().expect("silu-stress requires source GGUF");
        model_layer::silu_stress(device, std::path::Path::new(&model))
            .unwrap_or_else(|e| panic!("SiLU arithmetic stress failed: {e}"));
        return;
    }
    if sessions_mode.is_some() {
        let model = args
            .next()
            .expect("model-sessions-<backend> requires a local original Qwen GGUF");
        model_sessions::run(device, std::path::Path::new(&model))
            .unwrap_or_else(|e| panic!("physical shared-model sessions failed: {e}"));
        return;
    }
    if workers_mode.is_some() {
        let model_08 = args
            .next()
            .expect("model-workers-<backend> needs Qwen3.5 0.8B GGUF");
        let model_2b = args
            .next()
            .expect("model-workers-<backend> needs Qwen3.5 2B GGUF");
        model_workers::run(
            device,
            std::path::Path::new(&model_08),
            std::path::Path::new(&model_2b),
            native_workers,
        )
        .unwrap_or_else(|error| panic!("P07-G2b physical dual-model worker failed: {error}"));
        return;
    }
    if resident_mode.is_some() {
        let model = args
            .next()
            .expect("model-resident-<backend> requires GGUF path");
        let tensor = args
            .next()
            .expect("model-resident-<backend> requires one source tensor name");
        let prompt = args.next().unwrap_or_else(|| "Hello, world!".to_owned());
        model_resident::run(device, std::path::Path::new(&model), &tensor, &prompt).unwrap_or_else(
            |error| panic!("P07-C resident source GPU model parity failed: {error}"),
        );
        return;
    }
    if mixed_mode {
        let model = args
            .next()
            .expect("model-mixed-<backend> requires GGUF source path");
        let tensor_name = args
            .next()
            .expect("model-mixed-<backend> requires exactly one GGUF tensor name");
        let prompt = args.next().unwrap_or_else(|| "Hello, world!".to_owned());
        model_mixed::run(device, std::path::Path::new(&model), &tensor_name, &prompt)
            .unwrap_or_else(|error| {
                panic!("P07-B source GPU mixed region model parity failed: {error}")
            });
        return;
    }
    if model_mode {
        let model = args
            .next()
            .expect("model-q4-<backend> requires source GGUF path");
        let prompt = args.next().unwrap_or_else(|| "Hello, world!".to_owned());
        model_q4::run(device, std::path::Path::new(&model), &prompt)
            .unwrap_or_else(|error| panic!("P07-A source GPU matvec model parity failed: {error}"));
        return;
    }
    if q4_mode {
        let fixture = args
            .next()
            .expect("q4-<backend> requires a source-verified fixture path");
        q4::run(&device, std::path::Path::new(&fixture))
            .unwrap_or_else(|error| panic!("Qwen Q4_0 device qualification failed: {error}"));
        return;
    }
    let client = device.client();
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let src = client.create_from_slice(f32::as_bytes(&input));
    let dst = client.empty(input.len() * std::mem::size_of::<f32>());
    // SAFETY: the launch dimensions cover exactly four in-bounds f32 elements;
    // our kernel additionally checks the dynamic array length per lane.
    unsafe {
        multiply_by_two::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            BufferArg::from_raw_parts(src, input.len()),
            BufferArg::from_raw_parts(dst.clone(), input.len()),
        );
    }
    let bytes = client.read_one(dst).expect("GPU kernel execution failed");
    let values = f32::from_bytes(&bytes);
    assert_eq!(
        values,
        &[2.0, 4.0, 6.0, 8.0],
        "physical GPU output differs from the checked CPU oracle"
    );
    println!("PASS {name}: {} {values:?}", client.name());
}

fn print_help() {
    println!(
        "zerogravity-gpu-probe — explicit GPU qualification tool\n\nUse resident-qualify-BACKEND MODEL for source/physical qualification.\nUse resident-context-BACKEND MODEL PROMPT IDS for genuine512 qualification.\nBACKEND: vulkan-amd, vulkan-nvidia, cuda or rocm when enabled.\nGPU text generation is provided by the zerogravity CLI run-BACKEND command.\nFault and checkpoint commands require their explicit diagnostic features."
    );
}
