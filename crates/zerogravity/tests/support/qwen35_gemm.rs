//! Fresh authenticated reference operands; independent driver-only GEMM diagnostics.
use super::*;
use sha2::{Digest, Sha256};
use std::{fs::OpenOptions, io::Write};
use zerogravity::accelerator::program::{PlanOperation, SourceOperatorKind};

#[test]
#[ignore = "fresh isolated release process; capture first-layer reference GEMM operands"]
fn capture_first_layer_projections() -> Result<()> {
    capture_layer_projections(false)
}

#[test]
#[ignore = "fresh isolated release process; capture natural-feedback cached GEMM operands"]
fn capture_cached_layer_projections() -> Result<()> {
    capture_layer_projections(true)
}

#[test]
#[ignore = "fresh isolated release process; capture first attention projection"]
fn capture_first_attention_projection() -> Result<()> {
    capture_projections(false, true)
}

#[test]
#[ignore = "fresh isolated release process; capture cached attention projection"]
fn capture_cached_attention_projection() -> Result<()> {
    capture_projections(true, true)
}

fn capture_layer_projections(cached: bool) -> Result<()> {
    capture_projections(cached, false)
}

fn capture_projections(cached: bool, attention: bool) -> Result<()> {
    ensure(!cfg!(debug_assertions), "release physical diagnostic only")?;
    let directory = env("ZG_QWEN35_MODEL_DIR")?;
    let native = env("ZG_QWEN35_CUDA_KERNEL_DIR")?;
    let destination = env("ZG_BF16_GEMM_REFERENCE")?;
    block_on(async {
        let host = qwen35_cuda::host_admission_mib(
            qwen35_cuda::HOST_MIB_MAX,
            ResourceOwner::new("qwen35-gemm-reference", "first-layer-diagnostic")?,
        )?;
        let device =
            zerogravity::gpu::Device::cuda(0).map_err(|e| fail(format!("CUDA device: {e}")))?;
        let model = qwen35_cuda::qualification::load(
            device,
            &directory,
            &native,
            &host,
            qwen35_cuda::PROVIDER_MIB * MIB,
            None,
        )
        .await?;
        let run = async {
            let prompt =
                zerogravity::qwen35::source::binding::OriginalWindowModelSource::fixed_prompt_ids();
            let mut session = model.session(None).await?;
            let mut digest = None;
            if cached {
                session.infer_prompt(prompt.as_slice(), None).await?;
            }
            let capture = |readback: zerogravity::gpu::cuda::native::TextReadback<'_>| {
                let mut file = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&destination)
                    .map_err(|e| fail(format!("create reference GEMM fixture: {e}")))?;
                let mut hash = Sha256::new();
                let mut write = |bytes: &[u8]| -> Result<()> {
                    file.write_all(bytes)
                        .map_err(|e| fail(format!("write GEMM fixture: {e}")))?;
                    hash.update(bytes);
                    Ok(())
                };
                write(b"ZGGM0001")?;
                let count = if attention { 1usize } else { 3 };
                write(&(count as u64).to_le_bytes())?;
                let record = readback.record();
                let projections = record
                    .plan()
                    .nodes()
                    .iter()
                    .filter_map(|node| match &node.operation {
                        PlanOperation::SourceOperator { descriptor, .. }
                            if matches!(
                                descriptor.kind(),
                                SourceOperatorKind::Projection { selected_row: None }
                            ) =>
                        {
                            Some((node, descriptor))
                        }
                        _ => None,
                    })
                    .filter(|(node, _)| {
                        !attention
                            || record.plan().nodes().iter().any(|consumer| {
                                matches!(
                                    &consumer.operation,
                                    PlanOperation::SourceOperator { descriptor, .. }
                                        if matches!(descriptor.kind(), SourceOperatorKind::QkRmsRotaryGated(_))
                                ) && node.outputs.iter().any(|id| consumer.inputs.contains(id))
                            })
                    })
                    .take(count)
                    .collect::<Vec<_>>();
                ensure(projections.len() == count, "reference projection coverage")?;
                for (node, descriptor) in projections {
                    ensure(
                        node.inputs.len() == 1 && node.outputs.len() == 1,
                        "projection SSA arity",
                    )?;
                    let members = descriptor.matrices();
                    ensure(!members.is_empty(), "projection source members")?;
                    let columns = members.iter().map(|m| m.rows).sum::<usize>();
                    let inner = members[0].columns;
                    let rows = descriptor.call().rows().len();
                    let input = readback.value(node.inputs[0])?;
                    let output = readback.value(node.outputs[0])?;
                    ensure(
                        input.len() == rows * inner * 2 && output.len() == rows * columns * 2,
                        "reference projection physical extent",
                    )?;
                    for value in [columns, rows, inner] {
                        write(&(value as u64).to_le_bytes())?;
                    }
                    for member in members {
                        ensure(
                            member.columns == inner,
                            "reference projection packed columns",
                        )?;
                        let matrix = record.source().matrix(&member.name, member.axes)?;
                        for first in (0..member.rows).step_by(128) {
                            let window = record
                                .source()
                                .read_rows(&matrix, first..(first + 128).min(member.rows))?;
                            write(window.as_slice())?;
                        }
                    }
                    write(input)?;
                    write(output)?;
                    println!(
                        "reference-gemm M={columns} N={rows} K={inner} input={:?} output={:?}",
                        node.inputs[0], node.outputs[0]
                    );
                }
                file.sync_all()
                    .map_err(|e| fail(format!("sync GEMM fixture: {e}")))?;
                digest = Some(
                    hash.finalize()
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>(),
                );
                Ok(())
            };
            if cached {
                session.infer_next_with(None, capture).await?;
            } else {
                session
                    .infer_prompt_with(prompt.as_slice(), None, capture)
                    .await?;
            }
            digest.ok_or_else(|| fail("reference GEMM capture absent"))
        }
        .await;
        let close = model.shutdown(None).await;
        match (run, close) {
            (Ok(digest), Ok(close)) => {
                ensure(
                    close.source_weight_windows > 0
                        && close.source_weight_windows == close.retirement_source_weight_windows,
                    "reference GEMM source retirement differs",
                )?;
                println!("REFERENCE_GEMM_CAPTURE_PASS sha256={digest}");
                Ok(())
            }
            (run, close) => Err(fail(format!(
                "reference GEMM capture failed: run={run:?}; close={close:?}"
            ))),
        }
    })
}
