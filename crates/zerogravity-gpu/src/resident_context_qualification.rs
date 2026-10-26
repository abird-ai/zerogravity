//! Original H18 context gate plus explicit768/1024 internal CUDA diagnostics.
//! Source/CPU bit agreement does not establish independent vLLM parity.
use super::*;

#[cfg(test)]
#[path = "resident_context_qualification/tests.rs"]
mod tests;

const CONTEXT: usize = 512;
const MILESTONES: [usize; 12] = [1, 32, 64, 128, 254, 255, 256, 257, 258, 384, 511, 512];

#[derive(Clone, Copy)]
struct ContextSpec {
    capacity: usize,
    milestones: &'static [usize],
    replay_start: usize,
    replay_end: usize,
}
impl ContextSpec {
    fn legacy() -> Self {
        Self {
            capacity: CONTEXT,
            milestones: &MILESTONES,
            replay_start: 254,
            replay_end: 257,
        }
    }
    fn bounded(capacity: usize) -> Result<Self> {
        let milestones: &'static [usize] = match capacity {
            768 => &[
                1, 32, 64, 128, 254, 255, 256, 257, 258, 384, 511, 512, 513, 640, 767, 768,
            ],
            1024 => &[
                1, 32, 64, 128, 254, 255, 256, 257, 258, 384, 511, 512, 513, 640, 767, 768, 769,
                896, 1023, 1024,
            ],
            _ => {
                return Err(Error::Unsupported(
                    "bounded context qualification requires768/1024".into(),
                ));
            }
        };
        Ok(Self {
            capacity,
            milestones,
            replay_start: capacity - 257,
            replay_end: capacity - 255,
        })
    }
    fn validate_ids(self, ids: &[u32], vocabulary: u64) -> Result<()> {
        if ids.len() != self.capacity
            || vocabulary != 248_320
            || ids.iter().any(|&id| u64::from(id) >= vocabulary)
        {
            return Err(Error::Malformed(
                "complete original context teacher IDs/vocabulary".into(),
            ));
        }
        Ok(())
    }
}

fn text_file(path: &Path, label: &str) -> Result<String> {
    use std::io::Read;
    const LIMIT: usize = 1024 * 1024;
    let mut text = String::new();
    std::fs::File::open(path)?
        .take(LIMIT as u64 + 1)
        .read_to_string(&mut text)?;
    if text.len() > LIMIT {
        return Err(Error::Limit(format!("{label} exceeds 1 MiB")));
    }
    Ok(text)
}

/// Binds the original official tokenizer to an independently supplied raw ID
/// list BEFORE model upload, session allocation or physical initialization.
pub(super) fn source_tokens(path: &Path, prompt_path: &Path, ids_path: &Path) -> Result<Vec<u32>> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("context source model path".into()))?;
    let prompt = text_file(prompt_path, "context prompt")?;
    let raw = text_file(ids_path, "independent context IDs")?;
    let mut supplied = Vec::with_capacity(CONTEXT);
    for (i, value) in raw.split_whitespace().enumerate() {
        if i >= CONTEXT {
            return Err(Error::Malformed(
                "independent context IDs exceed 512 tokens".into(),
            ));
        }
        supplied.push(
            value
                .parse::<u32>()
                .map_err(|e| Error::Malformed(format!("raw context token {i}: {e}")))?,
        );
    }
    if supplied.len() != CONTEXT {
        return Err(Error::Malformed(format!(
            "independent context requires exactly 512 IDs, got {}",
            supplied.len()
        )));
    }
    let tokenizer = zerogravity_model_qwen35::tokenize::QwenTokenizer::from_file(
        &parent.join("tokenizer.json"),
    )?;
    let tokens = tokenizer.encode(&prompt)?;
    if tokens.len() != CONTEXT || tokens != supplied {
        return Err(Error::State(format!(
            "official tokenizer/raw context IDs differ (encoded {}, supplied {})",
            tokens.len(),
            supplied.len()
        )));
    }
    let geometry =
        zerogravity_model_qwen35::config::ModelGeometry::from_file(&parent.join("config.json"))?;
    if tokens
        .iter()
        .any(|&id| u64::from(id) >= geometry.vocab_size)
    {
        return Err(Error::Malformed(
            "official context IDs exceed original model vocabulary".into(),
        ));
    }
    println!(
        "H18_CONTEXT_SOURCE_BINDING_PASS prompt_utf8_bytes={} official_tokenizer_ids=512 independently_supplied_ids=512 exact_ids=true before_gpu_upload=true",
        prompt.len()
    );
    Ok(tokens)
}

fn publication(before: RootBinding, after: RootBinding, tokens: usize) -> Result<()> {
    if before.source() != after.source()
        || before.session() != after.session()
        || before.root().checked_add(1) != Some(after.root())
        || before.frontier().checked_add(tokens as u64) != Some(after.frontier())
        || before.session_epoch() != after.session_epoch()
    {
        return Err(Error::State(
            "H18 chunk root/frontier publication differs".into(),
        ));
    }
    Ok(())
}
fn restoration(before: RootBinding, after: RootBinding, saved: RootBinding) -> Result<()> {
    if before.source() != after.source()
        || before.session() != after.session()
        || before.root().checked_add(1) != Some(after.root())
        || before.session_epoch().checked_add(1) != Some(after.session_epoch())
        || after.frontier() != saved.frontier()
    {
        return Err(Error::State(
            "H18 restore root/epoch/frontier differs".into(),
        ));
    }
    Ok(())
}
fn trace(expected: &[LayerObservation], actual: &[LayerObservation]) -> Result<usize> {
    if expected.len() != actual.len() {
        return Err(Error::State(format!(
            "H18 trace count {} != {}",
            expected.len(),
            actual.len()
        )));
    }
    let mut compared = 0;
    for (e, a) in expected.iter().zip(actual) {
        if e.position != a.position || e.layer != a.layer || e.stage != a.stage {
            return Err(Error::State("H18 ordered trace metadata differs".into()));
        }
        exact(
            &e.values,
            &a.values,
            &format!("H18 position {} layer {} {}", e.position, e.layer, e.stage),
        )?;
        compared += e.values.len();
    }
    Ok(compared)
}
fn chunk(read: &InferenceReadback, tokens: usize) -> Result<()> {
    if tokens == 0
        || read.state_only_steps != tokens - 1
        || read.logit_steps != 1
        || read.device_token.is_some()
        || read.chunk_bytes != read.bytes + (tokens - 1) * 4
        || read.logits.len() != 248_320
        || read.domains.len() != 42
    {
        return Err(Error::State(
            "H18 complete vocabulary/state/chunk readback receipt differs".into(),
        ));
    }
    Ok(())
}
fn quiet(resident: &Arc<ResidentModel>, uploads: u64) -> Result<()> {
    resident.service.ensure_usable()?;
    if resident.service.pending_events()? != 0 || resident.service.uploaded_value_bytes() != uploads
    {
        return Err(Error::State(
            "H18 completed context leaked events or uploaded warm values".into(),
        ));
    }
    Ok(())
}

pub(super) async fn qualify(
    device: Device,
    path: &Path,
    prompt_path: &Path,
    ids_path: &Path,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    let tokens = source_tokens(path, prompt_path, ids_path)?;
    qualify_tokens(
        device,
        path,
        tokens,
        ContextSpec::legacy(),
        provider_cap,
        device_cap,
    )
    .await
}

/// ROOT authenticates the raw corpus before this call. No UTF8 prefix decoding,
/// independent numerical-profile qualification or serving-profile expansion.
pub(super) async fn qualify_bounded(
    device: Device,
    path: &Path,
    tokens: Vec<u32>,
    capacity: usize,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    if !cfg!(feature = "cuda") || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "bounded context diagnostic requires CUDA".into(),
        ));
    }
    let spec = ContextSpec::bounded(capacity)?;
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("context source model path".into()))?;
    let geometry =
        zerogravity_model_qwen35::config::ModelGeometry::from_file(&parent.join("config.json"))?;
    spec.validate_ids(&tokens, geometry.vocab_size)?;
    println!(
        "H18_BOUNDED_CONTEXT_SCOPE capacity={capacity} original_GGUF_B1=true raw_teacher_ids=true before_gpu_upload=true numerical_scope=INTERNAL_CPU_GPU_BIT_PARITY independent_vllm_parity=NOT_ESTABLISHED serving_profile=unchanged"
    );
    qualify_tokens(device, path, tokens, spec, provider_cap, device_cap).await
}

async fn qualify_tokens(
    device: Device,
    path: &Path,
    tokens: Vec<u32>,
    spec: ContextSpec,
    provider_cap: usize,
    device_cap: u64,
) -> Result<()> {
    let capacity = spec.capacity;
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("context source model path".into()))?;
    let model = LoadedQwenModel::open(path, &parent.join("config.json"))?;
    if model.geometry().vocab_size != 248_320 {
        return Err(Error::Unsupported(
            "H18 requires the original complete 248320-token vocabulary".into(),
        ));
    }
    let full_layers = model
        .geometry()
        .layers
        .iter()
        .filter(|kind| {
            matches!(
                kind,
                zerogravity_model_qwen35::config::LayerKind::FullAttention
            )
        })
        .count();
    let boundary_launches = full_layers * (256 / 32) * 3;
    if boundary_launches != 144 {
        return Err(Error::Unsupported(
            "H18 source attention boundary geometry differs".into(),
        ));
    }
    let resident =
        load_qualified(device, model.clone(), capacity, provider_cap, device_cap).await?;
    range_qualification::qualify(&resident).await?;
    view_qualification::qualify(&resident).await?;
    alias_qualification::qualify(&resident).await?;
    let model_only_bytes = resident.service.resident_bytes()?;
    let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(model)?;
    let mut session = resident.session().await?;
    #[allow(unused_mut)]
    let mut uploads = resident.service.uploaded_value_bytes();
    let mut saved_replay = None;
    let mut saved_last = None;
    let mut final_logits = Vec::new();
    let mut previous = 0;
    let mut full_boundary = None;
    let mut compared_trace = 0usize;
    let mut total_dispatches = 0usize;
    let mut total_readback = 0usize;
    for &end in spec.milestones {
        let segment = &tokens[previous..end];
        let before = session.binding();
        let (expected, expected_trace) = cpu.prefill_transaction_with_layer_trace(segment)?;
        let actual = session.forward(segment, true).await?;
        publication(before, session.binding(), segment.len())?;
        chunk(&actual, segment.len())?;
        exact(
            &expected,
            &actual.logits,
            &format!("H18 continuous full logits at {end}"),
        )?;
        compare_domains(&cpu.snapshot().domain_values(), &actual.domains)?;
        compared_trace += trace(&expected_trace, &actual.observations)?;
        quiet(&resident, uploads)?;
        if end.is_multiple_of(256) {
            full_boundary = Some((end, actual.dispatches));
        }
        if end > 1 && (end - 1).is_multiple_of(256) {
            let (boundary, prior) = full_boundary
                .ok_or_else(|| Error::State("H18 missing physical boundary receipt".into()))?;
            if boundary + 1 != end || segment.len() != 1 {
                return Err(Error::State(
                    "H18 boundary requires successive single-token receipts".into(),
                ));
            }
            if actual.dispatches != prior + boundary_launches {
                return Err(Error::State(format!(
                    "H18 {end}th token added {} launches instead of {boundary_launches}",
                    actual.dispatches as i128 - prior as i128
                )));
            }
        }
        total_dispatches += actual.dispatches;
        total_readback += actual.chunk_bytes;
        println!(
            "H18_RESIDENT_CONTEXT_MILESTONE_PASS consumed_tokens={end} segment_tokens={} dispatches={} state_only_steps={} logit_steps=1 actual_chunk_readback_bytes={} final_step_readback_bytes={} vocabulary_logits=248320 all42_state_domains=exact canonical_kv_tails=positive_zero ordered_trace=exact root_increment=1 warm_value_uploads=0 pending_events=0",
            segment.len(),
            actual.dispatches,
            actual.state_only_steps,
            actual.chunk_bytes,
            actual.bytes
        );
        if end == spec.replay_start {
            let snapshot = session.snapshot();
            let expected = cpu.snapshot();
            compare_domains(&expected.domain_values(), &snapshot.read_domains().await?)?;
            saved_replay = Some((snapshot, expected));
        }
        if end == capacity - 1 {
            saved_last = Some((session.snapshot(), cpu.snapshot()));
        }
        if end == capacity {
            final_logits = expected;
        }
        #[cfg(feature = "portable-checkpoint")]
        if matches!(end, 256 | 257) || end == capacity {
            checkpoint::qualify_frontier(&session, &cpu, tokens.get(end).copied()).await?;
            // The sibling's explicit checkpoint transfers are diagnostic setup,
            // separate from the continuing graph's zero-upload warm execution.
            uploads = resident.service.uploaded_value_bytes();
        }
        previous = end;
    }
    alias_qualification::qualify(&resident).await?;
    let (saved_replay, cpu_replay) =
        saved_replay.ok_or_else(|| Error::State("H18 missing immutable replay snapshot".into()))?;
    let (saved_last, cpu_last) =
        saved_last.ok_or_else(|| Error::State("H18 missing final-token snapshot".into()))?;
    compare_domains(
        &cpu_replay.domain_values(),
        &saved_replay.read_domains().await?,
    )?;
    compare_domains(&cpu_last.domain_values(), &saved_last.read_domains().await?)?;
    let unchanged = session.binding();
    let admitted = resident.service.resident_bytes()?;
    for bad in [&tokens[..1], &[u32::MAX][..], &[][..]] {
        if session.prepare(bad, false).await.is_ok()
            || session.binding() != unchanged
            || resident.service.resident_bytes()? != admitted
        {
            return Err(Error::State(
                "H18 capacity/invalid/empty prepare changed root or admission".into(),
            ));
        }
        quiet(&resident, uploads)?;
    }
    println!(
        "H18_RESIDENT_CAPACITY_PASS consumed_tokens={capacity} overcapacity_prepare=refused invalid_prepare=refused empty_prepare=refused root_and_admission=unchanged"
    );

    let before = session.binding();
    session.restore(&saved_replay)?;
    restoration(before, session.binding(), saved_replay.binding)?;
    cpu.restore(&cpu_replay)?;
    let replay = &tokens[spec.replay_start..spec.replay_end];
    let (expected, expected_trace) = cpu.prefill_transaction_with_layer_trace(replay)?;
    let before = session.binding();
    let prepared = session.prepare(replay, true).await?;
    if session.binding() != before
        || before.frontier() != spec.replay_start as u64
        || prepared.ready.expected_root() != before.root()
    {
        return Err(Error::State(
            "H18 private cross-boundary Ready changed visible root".into(),
        ));
    }
    chunk(&prepared.readback, replay.len())?;
    exact(
        &expected,
        &prepared.readback.logits,
        "H18 private cross-boundary replay logits",
    )?;
    compare_domains(&cpu.snapshot().domain_values(), &prepared.readback.domains)?;
    compared_trace += trace(&expected_trace, &prepared.readback.observations)?;
    compare_domains(
        &cpu_replay.domain_values(),
        &saved_replay.read_domains().await?,
    )?;
    let actual = session.commit(prepared)?;
    publication(before, session.binding(), replay.len())?;
    let replay_dispatches = actual.dispatches;
    let replay_bytes = actual.chunk_bytes;
    drop(actual);
    let before = session.binding();
    if capacity != CONTEXT {
        let admitted = resident.service.resident_bytes()?;
        let abandoned = session
            .prepare(&tokens[spec.replay_end..spec.replay_end + 1], false)
            .await?;
        if session.binding() != before {
            return Err(Error::State(
                "H18 private abandoned Ready changed committed frontier".into(),
            ));
        }
        drop(abandoned);
        if session.binding() != before || resident.service.resident_bytes()? != admitted {
            return Err(Error::State(
                "H18 abandoned Ready changed root or retained private admission".into(),
            ));
        }
        quiet(&resident, uploads)?;
        compare_domains(
            &cpu.snapshot().domain_values(),
            &session.snapshot().read_domains().await?,
        )?;
        println!(
            "H18_BOUNDED_ABANDONED_READY_PASS capacity={capacity} committed_frontier={} completed_private_proposal_dropped=true root_unchanged=true private_admission_released=true",
            before.frontier()
        );
    }
    let stale = session
        .prepare(&tokens[spec.replay_end..spec.replay_end + 1], false)
        .await?;
    if session.binding() != before {
        return Err(Error::State(
            "H18 private Ready changed committed replay frontier".into(),
        ));
    }
    session.restore(&saved_replay)?;
    let restored = session.binding();
    restoration(before, restored, saved_replay.binding)?;
    if session.commit(stale).is_ok() || session.binding() != restored {
        return Err(Error::State(
            "H18 stale cross-boundary actual Ready published after restore".into(),
        ));
    }
    compare_domains(
        &cpu_replay.domain_values(),
        &session.snapshot().read_domains().await?,
    )?;
    quiet(&resident, uploads)?;
    if capacity == CONTEXT {
        println!(
            "H18_RESIDENT_BOUNDARY_REPLAY_PASS restored_tokens=254 replayed_valid_prefixes=255,256,257 private_ready_visible_frontier=254 published_tokens=257 root_increment=1 state_only_steps=2 logit_steps=1 dispatches={replay_dispatches} actual_chunk_readback_bytes={replay_bytes} exact_logits_and_all42_states=true ordered_trace=exact original_snapshot=immutable stale_actual_ready_after_restore=refused added_value_launches_at_257={boundary_launches}"
        );
    } else {
        println!(
            "H18_BOUNDED_BOUNDARY_REPLAY_PASS capacity={capacity} restored_tokens={} replayed_valid_prefixes={},{} private_ready_visible_frontier={} published_tokens={} root_increment=1 state_only_steps=1 logit_steps=1 dispatches={replay_dispatches} actual_chunk_readback_bytes={replay_bytes} exact_logits_and_all42_states=true ordered_trace=exact original_snapshot=immutable stale_actual_ready_after_restore=refused added_value_launches_at_boundary={boundary_launches}",
            spec.replay_start,
            spec.replay_start + 1,
            spec.replay_end,
            spec.replay_start,
            spec.replay_end
        );
    }

    let before = session.binding();
    session.restore(&saved_last)?;
    restoration(before, session.binding(), saved_last.binding)?;
    cpu.restore(&cpu_last)?;
    let expected = cpu.forward(tokens[capacity - 1])?;
    exact(
        &final_logits,
        &expected,
        "H18 CPU final-token restored replay",
    )?;
    let before = session.binding();
    let mut greedy = session
        .forward_greedy(&tokens[capacity - 1..capacity])
        .await?;
    publication(before, session.binding(), 1)?;
    if greedy.bytes != 16
        || greedy.chunk_bytes != 16
        || !greedy.logits.is_empty()
        || greedy.state_only_steps != 0
        || greedy.logit_steps != 1
    {
        return Err(Error::State(
            "H18 final-capacity greedy readback is not compact".into(),
        ));
    }
    let packet = greedy
        .device_token
        .take()
        .ok_or_else(|| Error::State("H18 missing final GPU greedy packet".into()))?;
    let best = (0..final_logits.len())
        .max_by(|a, b| {
            final_logits[*a]
                .total_cmp(&final_logits[*b])
                .then_with(|| b.cmp(a))
        })
        .ok_or_else(|| Error::State("H18 empty complete-vocabulary CPU oracle".into()))?;
    if packet.id as usize != best || packet.max_bits != final_logits[best].to_bits() {
        return Err(Error::State(
            "H18 final GPU greedy token/max bits differ".into(),
        ));
    }
    compare_domains(
        &cpu.snapshot().domain_values(),
        &session.snapshot().read_domains().await?,
    )?;
    let unchanged = session.binding();
    let admitted = resident.service.resident_bytes()?;
    if session.forward_device_token(&packet).await.is_ok()
        || session.binding() != unchanged
        || resident.service.resident_bytes()? != admitted
    {
        return Err(Error::State(
            "H18 device token consumed beyond capacity or changed admission".into(),
        ));
    }
    quiet(&resident, uploads)?;
    println!(
        "H18_RESIDENT_FINAL_GREEDY_PASS consumed_tokens={capacity} exact_token_id=true exact_max_bits=true actual_step_readback_bytes=16 final_all42_state_domains=exact device_token_beyond_capacity=refused root_and_admission=unchanged warm_value_uploads=0"
    );
    let surviving = session.snapshot();
    let expected_surviving = cpu.snapshot().domain_values();
    let key = resident.service.physical_key();
    let weak = Arc::downgrade(resident.service());
    let charged = resident.service.capacity() as u64;
    drop(greedy);
    drop(packet);
    drop(saved_replay);
    drop(saved_last);
    drop(session);
    quiet(&resident, uploads)?;
    if resident.service.resident_bytes()? != model_only_bytes + resident.state_bytes {
        return Err(Error::State(
            "H18 retired context retained unexpected state admission".into(),
        ));
    }
    drop(resident);
    if weak.upgrade().is_none()
        || zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&key)?
            .reserved
            != charged
    {
        return Err(Error::State(
            "H18 surviving snapshot lost physical charge".into(),
        ));
    }
    compare_domains(&expected_surviving, &surviving.read_domains().await?)?;
    drop(surviving);
    if weak.upgrade().is_some()
        || zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&key)?
            .reserved
            != 0
    {
        return Err(Error::State(format!(
            "H18 completed {capacity}-token teardown leaked charge"
        )));
    }
    let marker = if capacity == CONTEXT {
        "H18_RESIDENT_CONTEXT_PASS"
    } else {
        "H18_BOUNDED_RESIDENT_CONTEXT_PASS"
    };
    println!(
        "{marker} physical={key:?} genuine_continuous_tokens={capacity} milestones={} vocabulary_logits=248320 all42_state_domains=exact ordered_trace=exact compared_trace_f32={compared_trace} continuous_dispatches={total_dispatches} continuous_chunk_readback_bytes={total_readback} immutable_nonzero_snapshots=true physical_boundary_replay=true source_reuploads=0 intermediate_activation_transfers=0 warm_value_uploads=0 pending_events=0 model_only_resident_bytes={model_only_bytes} snapshot_survives_provider=true physical_charge_retained=true final_reserved_bytes=0",
        spec.milestones.len(),
    );
    Ok(())
}
