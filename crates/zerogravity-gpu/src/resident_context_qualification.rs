//! Genuine continuous 512-token resident-context qualification.
//! Text/raw IDs are independent inputs; CPU logits/state are diagnostic only.
use super::*;

const CONTEXT: usize = 512;
const MILESTONES: [usize; 12] = [1, 32, 64, 128, 254, 255, 256, 257, 258, 384, 511, 512];

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
        zerogravity_artifacts::model::ModelGeometry::from_file(&parent.join("config.json"))?;
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
        .filter(|kind| matches!(kind, zerogravity_artifacts::model::LayerKind::FullAttention))
        .count();
    let boundary_launches = full_layers * (256 / 32) * 3;
    if boundary_launches != 144 {
        return Err(Error::Unsupported(
            "H18 source attention boundary geometry differs".into(),
        ));
    }
    let resident = load_qualified(device, model.clone(), CONTEXT, provider_cap, device_cap).await?;
    range_qualification::qualify(&resident).await?;
    view_qualification::qualify(&resident).await?;
    alias_qualification::qualify(&resident).await?;
    let model_only_bytes = resident.service.resident_bytes()?;
    let mut cpu = zerogravity_model_qwen35::QwenDenseCpu::from_shared(model)?;
    let mut session = resident.session().await?;
    #[allow(unused_mut)]
    let mut uploads = resident.service.uploaded_value_bytes();
    let mut saved254 = None;
    let mut saved511 = None;
    let mut final_logits = Vec::new();
    let mut previous = 0;
    let mut full256 = None;
    let mut compared_trace = 0usize;
    let mut total_dispatches = 0usize;
    let mut total_readback = 0usize;
    for end in MILESTONES {
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
        if end == 256 {
            full256 = Some(actual.dispatches);
        }
        if end == 257 {
            let prior = full256
                .ok_or_else(|| Error::State("H18 missing physical 256-token receipt".into()))?;
            if actual.dispatches != prior + boundary_launches {
                return Err(Error::State(format!(
                    "H18 257th token added {} launches instead of {boundary_launches}",
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
        if end == 254 {
            let snapshot = session.snapshot();
            let expected = cpu.snapshot();
            compare_domains(&expected.domain_values(), &snapshot.read_domains().await?)?;
            saved254 = Some((snapshot, expected));
        }
        if end == 511 {
            saved511 = Some((session.snapshot(), cpu.snapshot()));
        }
        if end == CONTEXT {
            final_logits = expected;
        }
        #[cfg(feature = "portable-checkpoint")]
        if matches!(end, 256 | 257 | CONTEXT) {
            checkpoint::qualify_frontier(&session, &cpu, tokens.get(end).copied()).await?;
            // The sibling's explicit checkpoint transfers are diagnostic setup,
            // separate from the continuing graph's zero-upload warm execution.
            uploads = resident.service.uploaded_value_bytes();
        }
        previous = end;
    }
    alias_qualification::qualify(&resident).await?;
    let (saved254, cpu254) =
        saved254.ok_or_else(|| Error::State("H18 missing immutable 254 snapshot".into()))?;
    let (saved511, cpu511) =
        saved511.ok_or_else(|| Error::State("H18 missing 511 snapshot".into()))?;
    compare_domains(&cpu254.domain_values(), &saved254.read_domains().await?)?;
    compare_domains(&cpu511.domain_values(), &saved511.read_domains().await?)?;
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
        "H18_RESIDENT_CAPACITY_PASS consumed_tokens=512 overcapacity_prepare=refused invalid_prepare=refused empty_prepare=refused root_and_admission=unchanged"
    );

    let before = session.binding();
    session.restore(&saved254)?;
    restoration(before, session.binding(), saved254.binding)?;
    cpu.restore(&cpu254)?;
    let replay = &tokens[254..257];
    let (expected, expected_trace) = cpu.prefill_transaction_with_layer_trace(replay)?;
    let before = session.binding();
    let prepared = session.prepare(replay, true).await?;
    if session.binding() != before
        || before.frontier() != 254
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
        "H18 private 254-to-257 replay logits",
    )?;
    compare_domains(&cpu.snapshot().domain_values(), &prepared.readback.domains)?;
    compared_trace += trace(&expected_trace, &prepared.readback.observations)?;
    compare_domains(&cpu254.domain_values(), &saved254.read_domains().await?)?;
    let actual = session.commit(prepared)?;
    publication(before, session.binding(), replay.len())?;
    let replay_dispatches = actual.dispatches;
    let replay_bytes = actual.chunk_bytes;
    drop(actual);
    let before = session.binding();
    let stale = session.prepare(&tokens[257..258], false).await?;
    if session.binding() != before {
        return Err(Error::State(
            "H18 private Ready changed committed 257 frontier".into(),
        ));
    }
    session.restore(&saved254)?;
    let restored = session.binding();
    restoration(before, restored, saved254.binding)?;
    if session.commit(stale).is_ok() || session.binding() != restored {
        return Err(Error::State(
            "H18 stale cross-boundary actual Ready published after restore".into(),
        ));
    }
    compare_domains(
        &cpu254.domain_values(),
        &session.snapshot().read_domains().await?,
    )?;
    quiet(&resident, uploads)?;
    println!(
        "H18_RESIDENT_BOUNDARY_REPLAY_PASS restored_tokens=254 replayed_valid_prefixes=255,256,257 private_ready_visible_frontier=254 published_tokens=257 root_increment=1 state_only_steps=2 logit_steps=1 dispatches={replay_dispatches} actual_chunk_readback_bytes={replay_bytes} exact_logits_and_all42_states=true ordered_trace=exact original_snapshot=immutable stale_actual_ready_after_restore=refused added_value_launches_at_257={boundary_launches}"
    );

    let before = session.binding();
    session.restore(&saved511)?;
    restoration(before, session.binding(), saved511.binding)?;
    cpu.restore(&cpu511)?;
    let expected = cpu.forward(tokens[511])?;
    exact(
        &final_logits,
        &expected,
        "H18 CPU final-token restored replay",
    )?;
    let before = session.binding();
    let mut greedy = session.forward_greedy(&tokens[511..512]).await?;
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
        "H18_RESIDENT_FINAL_GREEDY_PASS consumed_tokens=512 exact_token_id=true exact_max_bits=true actual_step_readback_bytes=16 final_all42_state_domains=exact device_token_beyond_capacity=refused root_and_admission=unchanged warm_value_uploads=0"
    );
    let surviving = session.snapshot();
    let expected_surviving = cpu.snapshot().domain_values();
    let key = resident.service.physical_key();
    let weak = Arc::downgrade(resident.service());
    let charged = resident.service.capacity() as u64;
    drop(greedy);
    drop(packet);
    drop(saved254);
    drop(saved511);
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
        return Err(Error::State(
            "H18 completed 512-token teardown leaked charge".into(),
        ));
    }
    println!(
        "H18_RESIDENT_CONTEXT_PASS physical={key:?} genuine_continuous_tokens=512 milestones={} vocabulary_logits=248320 all42_state_domains=exact ordered_trace=exact compared_trace_f32={compared_trace} continuous_dispatches={total_dispatches} continuous_chunk_readback_bytes={total_readback} immutable_nonzero_snapshots=true physical_boundary_replay=true source_reuploads=0 intermediate_activation_transfers=0 warm_value_uploads=0 pending_events=0 model_only_resident_bytes={model_only_bytes} snapshot_survives_provider=true physical_charge_retained=true final_reserved_bytes=0",
        MILESTONES.len(),
    );
    Ok(())
}
