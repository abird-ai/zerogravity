//! Authenticated attention operands from the unchanged native reference.
use super::*;
use sha2::{Digest, Sha256};
use std::{fs::OpenOptions, io::Write};
use zerogravity::accelerator::program::{PlanOperation, SourceOperatorKind, StateSlot};

#[test]
#[ignore = "fresh isolated release process; capture reference causal attention"]
fn capture_prefill_attention() -> Result<()> {
    capture(false)
}
#[test]
#[ignore = "fresh isolated release process; capture reference cached attention"]
fn capture_cached_attention() -> Result<()> {
    capture(true)
}

fn capture(cached: bool) -> Result<()> {
    ensure(!cfg!(debug_assertions), "release physical diagnostic only")?;
    let directory = env("ZG_QWEN35_MODEL_DIR")?;
    let native = env("ZG_QWEN35_CUDA_KERNEL_DIR")?;
    let destination = env("ZG_BF16_ATTENTION_REFERENCE")?;
    block_on(async {
        let host = qwen35_cuda::host_admission_mib(
            qwen35_cuda::HOST_MIB_MAX,
            ResourceOwner::new("qwen35-attention-reference", "operator-diagnostic")?,
        )?;
        let device = zerogravity::gpu::Device::cuda(0).map_err(|e| fail(format!("CUDA: {e}")))?;
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
            let prompt = zerogravity::qwen35::source::binding::OriginalWindowModelSource::fixed_prompt_ids();
            let mut session = model.session(None).await?;
            if cached { session.infer_prompt(prompt.as_slice(), None).await?; }
            let mut digest = None;
            let capture = |readback: zerogravity::gpu::cuda::native::TextReadback<'_>| {
                let (node, descriptor) = readback.record().plan().nodes().iter().find_map(|node| {
                    match &node.operation {
                        PlanOperation::SourceOperator { descriptor, .. }
                            if matches!(descriptor.kind(), SourceOperatorKind::FlashAttention { .. }) => Some((node, descriptor)),
                        _ => None,
                    }
                }).ok_or_else(|| fail("reference attention absent"))?;
                let SourceOperatorKind::FlashAttention { domain, geometry, scale, .. } = descriptor.kind() else { unreachable!() };
                ensure(node.inputs.len() == 1 && node.outputs.len() == 1, "attention SSA arity")?;
                let rows = descriptor.call().rows().len();
                let base = descriptor.call().rows().base();
                let capacity = base + rows;
                let query = readback.value(node.inputs[0])?;
                let expected = readback.value(node.outputs[0])?;
                let q_bytes = rows * geometry.query_heads * geometry.width * 2;
                ensure(query.len() == q_bytes && expected.len() == q_bytes, "attention Q/O extents")?;
                let (keys, values) = readback.state(StateSlot::Kv(domain), true)?;
                let values = values.ok_or_else(|| fail("attention values absent"))?;
                let elements = capacity * geometry.kv_heads * geometry.width;
                let narrow = |bytes: &[u8]| -> Result<Vec<u8>> {
                    ensure(bytes.len() >= elements * 4, "attention canonical state extent")?;
                    bytes[..elements * 4].chunks_exact(4).try_fold(Vec::with_capacity(elements * 2), |mut out, b| {
                        let bits = u32::from_le_bytes(b.try_into().unwrap());
                        ensure(bits & 0xffff == 0 && bits & 0x7f80_0000 != 0x7f80_0000, "attention BF16 state contract")?;
                        out.extend_from_slice(&((bits >> 16) as u16).to_le_bytes()); Ok(out)
                    })
                };
                let keys = narrow(keys)?; let values = narrow(values)?;
                let mut file = OpenOptions::new().write(true).create_new(true).open(&destination)
                    .map_err(|e| fail(format!("create attention fixture: {e}")))?;
                let mut hash = Sha256::new();
                let mut write = |b: &[u8]| -> Result<()> {
                    file.write_all(b).map_err(|e| fail(format!("write attention fixture: {e}")))?;
                    hash.update(b); Ok(())
                };
                write(b"ZGAT0001")?;
                for value in [rows, base, capacity, geometry.query_heads, geometry.kv_heads, geometry.width] {
                    write(&(value as u64).to_le_bytes())?;
                }
                write(&scale.to_bits().to_le_bytes())?;
                for b in [query, keys.as_slice(), values.as_slice(), expected] { write(b)?; }
                file.sync_all().map_err(|e| fail(format!("sync attention fixture: {e}")))?;
                digest = Some(format!("{:x}", hash.finalize()));
                println!("reference-attention rows={rows} base={base} geometry={geometry:?} input={:?} output={:?}", node.inputs[0], node.outputs[0]);
                Ok(())
            };
            if cached { session.infer_next_with(None, capture).await?; }
            else { session.infer_prompt_with(prompt.as_slice(), None, capture).await?; }
            digest.ok_or_else(|| fail("attention capture absent"))
        }.await;
        let close = model.shutdown(None).await;
        match (run, close) {
            (Ok(digest), Ok(close)) => {
                ensure(
                    close.source_weight_windows > 0
                        && close.source_weight_windows == close.retirement_source_weight_windows,
                    "attention source retirement differs",
                )?;
                println!("REFERENCE_ATTENTION_CAPTURE_PASS sha256={digest}");
                Ok(())
            }
            (run, close) => Err(fail(format!(
                "attention capture failed: run={run:?}; close={close:?}"
            ))),
        }
    })
}
