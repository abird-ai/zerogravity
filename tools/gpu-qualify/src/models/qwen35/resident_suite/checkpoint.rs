//! Actual CPU/file/GPU state transfers; CPU execution is a diagnostic oracle.
use super::*;
use std::path::PathBuf;
#[cfg(feature = "hardware-abort-probe")]
use zerogravity_gpu::qualification::resident::abort_upload;

const PREFIX: [u32; 3] = [11, 353, 17];

fn digest(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 || !value.is_ascii() {
        return Err(Error::Malformed(
            "checkpoint SHA256 must contain 64 hex digits".into(),
        ));
    }
    let mut result = [0; 32];
    for (byte, pair) in result.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        *byte = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16)
            .map_err(|_| Error::Malformed("invalid checkpoint SHA256".into()))?;
    }
    Ok(result)
}

/// The invoking driver independently hashes the GGUF and LE-u32 PREFIX IDs.
/// An existing input file can come from a different GPU/process; its complete
/// numerical payload must also equal independent CPU execution of PREFIX.
pub async fn qualify(
    device: Device,
    path: &Path,
    file: &Path,
    identities: (&str, &str),
    provider_cap: usize,
    device_cap: u64,
    abort: bool,
) -> Result<()> {
    let (model_sha, tokens_sha) = (digest(identities.0)?, digest(identities.1)?);
    let source = LoadedQwenModel::open(path, &path.parent().unwrap().join("config.json"))?;
    let mut cpu = source.new_session()?;
    cpu.prefill_chunk_without_logits(&PREFIX)?;
    let expected = cpu.snapshot().domain_values();
    let cpu_file = PathBuf::from(format!("{}.cpu", file.display()));
    cpu.save_cpu_checkpoint(&cpu_file, model_sha, tokens_sha)?;
    let existed = file.exists();
    if !existed {
        cpu.save_cpu_checkpoint(file, model_sha, tokens_sha)?;
    }
    let checkpoint = source.load_checkpoint(file, model_sha, tokens_sha)?;
    if checkpoint.position() != PREFIX.len() {
        return Err(Error::State(
            "checkpoint fixture frontier differs from exact token prefix".into(),
        ));
    }
    compare_domains(&expected, &checkpoint.domain_values())?;
    let resident = load_qualified(
        device.clone(),
        source.clone(),
        512,
        provider_cap,
        device_cap,
    )
    .await?;
    let baseline = resident.qualification().resident_bytes()?;
    let mut session = resident.session().await?;
    let old_packet = session
        .forward_greedy(&[42, 77])
        .await?
        .device_token
        .unwrap();
    let prior = session.snapshot();
    let prior_values = prior.read_domains().await?;
    let binding = session.binding();
    let stale_ready = session.prepare(&[19], false).await?;

    #[cfg(feature = "hardware-abort-probe")]
    if abort {
        return abort_upload(
            device,
            source.source_identity(),
            resident,
            session,
            prior,
            prior_values,
            stale_ready,
            checkpoint.position(),
            &checkpoint.domain_values(),
            device_cap,
        )
        .await;
    }
    #[cfg(not(feature = "hardware-abort-probe"))]
    if abort {
        return Err(Error::State(
            "checkpoint abort requires hardware-abort-probe".into(),
        ));
    }

    // Rejected payloads never enter the physical initialization path.
    let bytes_before = resident.qualification().resident_bytes()?;
    let uploads_before = resident.qualification().uploaded_value_bytes();
    let foreign_source = LoadedQwenModel::open(path, &path.parent().unwrap().join("config.json"))?;
    let foreign = foreign_source.load_checkpoint(file, model_sha, tokens_sha)?;
    drop(foreign_source);
    if session.import_checkpoint(&foreign).await.is_ok() {
        return Err(Error::State("foreign checkpoint source accepted".into()));
    }
    let mut beyond = expected.clone();
    for (slot, values) in &mut beyond {
        if matches!(slot, StateSlot::Kv(_)) {
            let requirement = resident
                .state_requirements()
                .iter()
                .find(|r| r.slot == *slot)
                .ok_or_else(|| Error::State("checkpoint KV fixture coverage".into()))?;
            if let StateLayout::Kv { slab, .. } = requirement.layout {
                *values = vec![0.0; 2 * slab * 513];
            }
        }
    }
    let overcapacity = source.checkpoint_from_domains(513, &beyond)?;
    if session.import_checkpoint(&overcapacity).await.is_ok() {
        return Err(Error::State("overcapacity checkpoint accepted".into()));
    }
    for (m, t) in [([0; 32], tokens_sha), (model_sha, [0; 32])] {
        if source.load_checkpoint(file, m, t).is_ok() {
            return Err(Error::State(
                "checkpoint wrong source/input digest accepted".into(),
            ));
        }
    }
    let original = std::fs::read(file)?;
    let corrupt_file = PathBuf::from(format!("{}.corrupt", file.display()));
    for variant in 0..4 {
        let mut bytes = original.clone();
        match variant {
            0 => {
                bytes.pop();
            }
            1 => bytes.push(0),
            2 => {
                let last = bytes.len() - 1;
                bytes[last] ^= 1;
            }
            _ => bytes[117..121].copy_from_slice(&f32::NAN.to_bits().to_le_bytes()),
        }
        std::fs::write(&corrupt_file, bytes)?;
        if source
            .load_checkpoint(&corrupt_file, model_sha, tokens_sha)
            .is_ok()
        {
            return Err(Error::State("corrupt/nonfinite checkpoint accepted".into()));
        }
    }
    std::fs::remove_file(corrupt_file)?;
    if session.binding() != binding
        || resident.qualification().resident_bytes()? != bytes_before
        || resident.qualification().uploaded_value_bytes() != uploads_before
    {
        return Err(Error::State(
            "rejected checkpoint mutated root/admission/uploads".into(),
        ));
    }
    compare_domains(&prior_values, &prior.read_domains().await?)?;

    let filler = resident.qualification().allocate_uninitialized(
        resident.qualification().remaining_capacity()? - resident.state_bytes() + 1,
        BufferRole::Scratch,
    )?;
    let charged = resident.qualification().resident_bytes()?;
    if session.import_checkpoint(&checkpoint).await.is_ok()
        || session.binding() != binding
        || resident.qualification().resident_bytes()? != charged
        || resident.qualification().uploaded_value_bytes() != uploads_before
    {
        return Err(Error::State(
            "checkpoint admission refusal was not atomic".into(),
        ));
    }
    drop(filler);
    session.import_checkpoint(&checkpoint).await?;
    let imported = session.binding();
    if imported.source() != binding.source()
        || imported.session() != binding.session()
        || imported.frontier() != 3
        || imported.root() != binding.root() + 1
        || imported.session_epoch() != binding.session_epoch() + 1
        || resident.qualification().uploaded_value_bytes() - uploads_before
            != resident.state_bytes() as u64
        || resident.qualification().pending_events()? != 0
    {
        return Err(Error::State(
            "checkpoint import publication/physical completion differs".into(),
        ));
    }
    if session.commit(stale_ready).is_ok()
        || session.forward_device_token(&old_packet).await.is_ok()
        || session.binding() != imported
    {
        return Err(Error::State(
            "checkpoint import failed to fence stale proposal/token".into(),
        ));
    }
    drop(old_packet);
    compare_domains(&expected, &session.snapshot().read_domains().await?)?;
    compare_domains(&prior_values, &prior.read_domains().await?)?;
    let exported = session.checkpoint(&source).await?;
    let output = PathBuf::from(format!("{}.gpu", file.display()));
    exported.save(&output, model_sha, tokens_sha)?;
    if std::fs::read(&output)? != std::fs::read(&cpu_file)? {
        return Err(Error::State(
            "GPU export changed existing CPU v1 wire bytes".into(),
        ));
    }
    let mut resumed = source.new_session()?;
    resumed.restore_cpu_checkpoint(&output, model_sha, tokens_sha)?;
    compare_domains(&expected, &resumed.snapshot().domain_values())?;
    let uploads = resident.qualification().uploaded_value_bytes();
    let mut elements = 0;
    for token in [42, 19, 77] {
        let (logits, draft, trace) = resumed.next_logits_with_layer_trace(token)?;
        let actual = session.forward(&[token], true).await?;
        exact(&logits, &actual.logits, "checkpoint resumed full logits")?;
        compare_domains(&draft.domain_values(), &actual.domains)?;
        if trace.len() != actual.observations.len() {
            return Err(Error::State("checkpoint resumed trace count".into()));
        }
        for (a, b) in trace.iter().zip(&actual.observations) {
            if a.position != b.position || a.layer != b.layer || a.stage != b.stage {
                return Err(Error::State("checkpoint resumed trace metadata".into()));
            }
            exact(&a.values, &b.values, "checkpoint resumed intermediate")?;
            elements += a.values.len();
        }
        resumed.commit(draft)?;
    }
    if resident.qualification().uploaded_value_bytes() != uploads
        || resident.qualification().pending_events()? != 0
    {
        return Err(Error::State(
            "warm checkpoint replay uploaded values or left pending events".into(),
        ));
    }
    drop(prior);
    drop(session);
    if resident.qualification().resident_bytes()? != baseline {
        return Err(Error::State("checkpoint session leaked admission".into()));
    }
    let key = resident.qualification().physical_key();
    drop(resident);
    let budget =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if budget.reserved != 0 || budget.fenced {
        return Err(Error::State(
            "checkpoint teardown leaked physical envelope".into(),
        ));
    }
    println!(
        "H21_CHECKPOINT_PASS input_existing={existed} frontier=3 domains=42 cpu_gpu_cpu=bit_exact resumed_tokens=3 full_logits=bit_exact ordered_trace_elements={elements} wire_v1=byte_exact canonical_kv_tail=exact stale_ready_token=refused source_capacity_digest_corruption=refused admission_refusal=atomic warm_value_uploads=0 final_reserved_bytes=0 output={}",
        output.display()
    );
    Ok(())
}

pub(super) async fn qualify_frontier(
    session: &ResidentSession,
    cpu: &zerogravity_model_qwen35::QwenDenseCpu,
    next: Option<u32>,
) -> Result<()> {
    let resident = session.qualification_model();
    let binding = session.binding();
    let baseline = resident.qualification().resident_bytes()?;
    let uploads = resident.qualification().uploaded_value_bytes();
    let exported = session.checkpoint(cpu.loaded_model()).await?;
    compare_domains(&cpu.snapshot().domain_values(), &exported.domain_values())?;
    let mut restored_cpu = cpu.loaded_model().new_session()?;
    restored_cpu.restore_checkpoint(&exported)?;
    if resident.qualification().uploaded_value_bytes() != uploads {
        return Err(Error::State("checkpoint export uploaded values".into()));
    }
    let mut sibling = resident.session().await?;
    if resident.qualification().uploaded_value_bytes() != uploads + resident.state_bytes() as u64 {
        return Err(Error::State(
            "checkpoint sibling initialization upload count".into(),
        ));
    }
    sibling.import_checkpoint(&cpu.checkpoint()?).await?;
    let transferred = uploads + 2 * resident.state_bytes() as u64;
    if resident.qualification().uploaded_value_bytes() != transferred {
        return Err(Error::State(
            "checkpoint sibling import upload count".into(),
        ));
    }
    compare_domains(
        &exported.domain_values(),
        &sibling.snapshot().read_domains().await?,
    )?;
    if let Some(token) = next {
        let (expected, draft) = restored_cpu.next_logits(token)?;
        let actual = sibling.forward(&[token], true).await?;
        exact(
            &expected,
            &actual.logits,
            "context checkpoint resumed logits",
        )?;
        compare_domains(&draft.domain_values(), &actual.domains)?;
    } else if sibling.prepare(&[11], false).await.is_ok() {
        return Err(Error::State(
            "full-capacity imported checkpoint advanced".into(),
        ));
    }
    drop(sibling);
    if session.binding() != binding
        || resident.qualification().resident_bytes()? != baseline
        || resident.qualification().pending_events()? != 0
        || resident.qualification().uploaded_value_bytes() != transferred
    {
        return Err(Error::State(
            "context checkpoint changed continuing root/admission".into(),
        ));
    }
    println!(
        "H21_CONTEXT_CHECKPOINT_PASS frontier={} all42_physical_domains=bit_exact cpu_gpu_cpu=exact resumed_next_token={} canonical_kv_tail=exact continuing_root_unchanged=true admission_recovered=true",
        binding.frontier(),
        next.is_some()
    );
    Ok(())
}
