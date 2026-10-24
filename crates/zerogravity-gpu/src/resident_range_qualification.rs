//! H20 host guard equivalence over real admitted physical bindings.
use super::*;
use cubecl::client::Client;
use std::{collections::BTreeMap, hint::black_box, time::Instant};

fn bad(detail: &str) -> Error {
    Error::State(format!("H20 range qualification: {detail}"))
}
fn overlaps(a: &Handle, b: &Handle) -> bool {
    let a = a.clone().binding();
    let b = b.clone().binding();
    let ar = a.range();
    let br = b.range();
    a.service == b.service
        && a.memory.id() == b.memory.id()
        && ar.start < br.end
        && br.start < ar.end
}
/// Frozen H19 validation algorithm: diagnostic host oracle, never a launcher.
fn legacy(
    program: &ResidentProgram,
    client: &Client,
    p: RuntimeParameters,
    bindings: &[StateBinding],
) -> Result<()> {
    if p.valid_tokens
        != p.position
            .checked_add(1)
            .ok_or_else(|| bad("position overflow"))?
        || p.valid_tokens > program.plan().prefix_capacity()
        || p.position > 32768
    {
        return Err(bad("prefix"));
    }
    let mut seen = BTreeMap::new();
    let mut priors = Vec::new();
    let mut candidates = Vec::new();
    for b in bindings {
        if seen.insert(b.slot, b).is_some() {
            return Err(bad("duplicate domain"));
        }
        priors.extend(b.prior.handles());
        candidates.extend(b.candidate.handles());
    }
    if bindings.len() != program.budget().states.len() {
        return Err(bad("domain count"));
    }
    let check = |v: &DeviceBuffer, expected: usize| {
        let bytes = expected
            .checked_mul(4)
            .filter(|_| u32::try_from(expected).is_ok())
            .ok_or_else(|| bad("buffer byte/index overflow"))?;
        if v.elements != expected
            || v.handle.service != client.service_id()
            || v.handle.size_in_used() < bytes as u64
        {
            return Err(bad("binding size/service"));
        }
        Ok(())
    };
    for r in &program.budget().states {
        let b = seen.get(&r.slot).ok_or_else(|| bad("missing domain"))?;
        for state in [&b.prior, &b.candidate] {
            match (r.layout, state) {
                (StateLayout::Vector { elements }, StateBuffers::Vector(v)) => check(v, elements)?,
                (StateLayout::Kv { slab, capacity }, StateBuffers::Kv { keys, values }) => {
                    check(keys, slab * capacity)?;
                    check(values, slab * capacity)?;
                    if b.prior_tokens != p.position {
                        return Err(bad("KV prefix"));
                    }
                }
                _ => return Err(bad("typed layout")),
            }
        }
    }
    let owned = program.handles_for_attempt(&[]);
    for (i, c) in candidates.iter().enumerate() {
        if priors
            .iter()
            .chain(owned.iter())
            .chain(candidates[..i].iter())
            .any(|h| overlaps(c, h))
        {
            return Err(bad("private overlap"));
        }
    }
    if owned.iter().any(|h| h.service != client.service_id()) {
        return Err(bad("program service"));
    }
    Ok(())
}
fn first(state: &mut StateBuffers) -> &mut DeviceBuffer {
    match state {
        StateBuffers::Vector(v) => v,
        StateBuffers::Kv { keys, .. } => keys,
    }
}

pub(super) async fn qualify(resident: &Arc<ResidentModel>) -> Result<()> {
    let baseline = resident.service.resident_bytes()?;
    let session = resident.session().await?;
    let saved = session.snapshot();
    let prior = saved.read_domains().await?;
    let candidate = resident.allocate_state(false).await?;
    let valid = bind_domains(&session.domains, &candidate, 0)?;
    let binding = session.binding();
    let admitted = resident.service.resident_bytes()?;
    let uploads = resident.service.uploaded_value_bytes();
    let workspace = resident.workspace.lock().await;
    let program = &workspace.program;
    let client = resident.service.client();
    program.qualify_projection_bindings(client)?;
    let p = RuntimeParameters {
        token: 11,
        position: 0,
        valid_tokens: 1,
    };
    let (target, elements) = valid
        .iter()
        .enumerate()
        .filter_map(|(i, b)| match &b.candidate {
            StateBuffers::Vector(v) => Some((i, v.elements)),
            _ => None,
        })
        .min_by_key(|(_, n)| *n)
        .ok_or_else(|| bad("missing vector state"))?;
    let needed = elements * 4;
    let mut fixtures = vec![("valid", p, valid.clone(), true)];
    let mut add = |name, changed: Vec<StateBinding>| fixtures.push((name, p, changed, false));
    let values = program.values();
    // Choose large real originals, so alias fixtures reach the overlap guard.
    let source = workspace
        .leases
        .iter()
        .find(|(l, a)| *a == DeviceAccess::Read && l.bytes() >= needed)
        .map(|(l, _)| l.handle())
        .ok_or_else(|| bad("no large source fixture"))?;
    let activation = values
        .iter()
        .find(|v| v.elements >= elements)
        .ok_or_else(|| bad("no large activation fixture"))?
        .handle
        .clone();
    let greedy = workspace
        .greedy
        .as_ref()
        .ok_or_else(|| bad("model greedy workspace absent"))?;
    let scratch = workspace
        .leases
        .iter()
        .filter(|(_, a)| *a == DeviceAccess::Write)
        .filter(|(l, _)| !values.iter().any(|v| same_handle(&l.handle(), &v.handle)))
        .filter(|(l, _)| !greedy.iter().any(|h| same_handle(&l.handle(), h)))
        .max_by_key(|(l, _)| l.bytes())
        .filter(|(l, _)| l.bytes() >= needed)
        .map(|(l, _)| l.handle())
        .ok_or_else(|| bad("no large scratch fixture"))?;
    for (name, handle) in [
        ("source-alias", source),
        ("activation-alias", activation),
        ("scratch-alias", scratch),
    ] {
        let mut b = valid.clone();
        first(&mut b[target].candidate).handle = handle;
        add(name, b);
    }
    let mut b = valid.clone();
    b[target].candidate = b[target].prior.clone();
    add("prior-alias", b);
    let sibling = valid
        .iter()
        .enumerate()
        .find(|(i, b)| {
            *i != target
                && matches!(&b.candidate, StateBuffers::Vector(v) if v.elements == elements)
        })
        .map(|(i, _)| i)
        .ok_or_else(|| bad("no equal-layout sibling fixture"))?;
    let mut b = valid.clone();
    b[target].candidate = b[sibling].candidate.clone();
    add("sibling-candidate-alias", b);
    let mut b = valid.clone();
    b.push(b[0].clone());
    add("duplicate-domain", b);
    let mut b = valid.clone();
    b.pop();
    add("missing-domain", b);
    let kv = valid
        .iter()
        .position(|b| matches!(b.prior, StateBuffers::Kv { .. }))
        .ok_or_else(|| bad("no KV fixture"))?;
    let mut b = valid.clone();
    b[target].candidate = b[kv].candidate.clone();
    add("wrong-layout", b);
    let mut b = valid.clone();
    first(&mut b[target].candidate).elements -= 1;
    add("wrong-elements", b);
    let mut b = valid.clone();
    let v = first(&mut b[target].candidate);
    let trim = v.handle.size_in_used() - needed as u64 + 4;
    v.handle = v.handle.clone().offset_end(trim);
    add("short-physical-range", b);
    let mut b = valid.clone();
    b[kv].prior_tokens = 1;
    add("wrong-KV-prefix", b);
    struct ForeignService;
    let mut b = valid.clone();
    first(&mut b[target].candidate).handle.service.service =
        std::any::TypeId::of::<ForeignService>();
    add("foreign-service-metadata", b);
    fixtures.push((
        "wrong-runtime-prefix",
        RuntimeParameters {
            valid_tokens: 2,
            ..p
        },
        valid.clone(),
        false,
    ));
    fixtures.push((
        "over-capacity",
        RuntimeParameters {
            position: resident.capacity,
            valid_tokens: resident.capacity + 1,
            ..p
        },
        valid.clone(),
        false,
    ));
    fixtures.push((
        "position-overflow",
        RuntimeParameters {
            position: usize::MAX,
            valid_tokens: 0,
            ..p
        },
        valid.clone(),
        false,
    ));
    for (name, params, b, expected) in &fixtures {
        let old = legacy(program, client, *params, b).is_ok();
        let new = program.validate_runtime(client, *params, b).is_ok();
        if old != *expected || new != old {
            return Err(bad(&format!(
                "fixture {name}: expected={expected} legacy={old} prepared={new}"
            )));
        }
    }

    // The old CubeCL subtraction can panic or wrap for these malformed
    // coordinates. Exercise only the checked new guard, never the old oracle
    // or a GPU launch. All handles still refer to retained real allocations.
    for mode in 0..4 {
        let mut b = valid.clone();
        let handle = &mut first(&mut b[target].candidate).handle;
        match mode {
            0 => handle.offset_end = Some(handle.size() + 1),
            1 => handle.offset_start = Some(handle.size() + 1),
            2 => {
                handle.offset_start = Some(0);
                handle.offset_end = Some(handle.size());
            }
            _ => {
                handle.offset_start = Some(u64::MAX);
                handle.offset_end = Some(u64::MAX);
            }
        }
        if program.validate_runtime(client, p, &b).is_ok() {
            return Err(bad("malformed byte range accepted"));
        }
    }

    // Only host validation is timed: no launches, transfers or token throughput.
    for _ in 0..10 {
        legacy(program, client, p, &valid)?;
        program.validate_runtime(client, p, &valid)?;
    }
    let mut legacy_ns = [0u128; 5];
    let mut prepared_ns = [0u128; 5];
    let measure = |prepared: bool| -> Result<u128> {
        let start = Instant::now();
        for _ in 0..100 {
            let result = if prepared {
                black_box(program).validate_runtime(
                    black_box(client),
                    black_box(p),
                    black_box(&valid),
                )
            } else {
                legacy(
                    black_box(program),
                    black_box(client),
                    black_box(p),
                    black_box(&valid),
                )
            };
            black_box(result)?;
        }
        Ok(start.elapsed().as_nanos())
    };
    for round in 0..5 {
        if round % 2 == 0 {
            legacy_ns[round] = measure(false)?;
            prepared_ns[round] = measure(true)?;
        } else {
            prepared_ns[round] = measure(true)?;
            legacy_ns[round] = measure(false)?;
        }
    }
    let count = fixtures.len();
    drop(fixtures);
    drop(valid);
    drop(workspace);
    if session.binding() != binding
        || resident.service.resident_bytes()? != admitted
        || resident.service.uploaded_value_bytes() != uploads
        || resident.service.pending_events()? != 0
    {
        return Err(bad(
            "validator inspection changed root/admission/events/uploads",
        ));
    }
    resident.service.ensure_usable()?;
    compare_domains(&prior, &saved.read_domains().await?)?;
    drop(candidate);
    drop(saved);
    drop(session);
    if resident.service.resident_bytes()? != baseline || resident.service.pending_events()? != 0 {
        return Err(bad(
            "private/session teardown did not recover model baseline",
        ));
    }
    println!(
        "H20_RESIDENT_RANGE_GUARDS_PASS physical={:?} fixtures={count} legacy_prepared_equivalent=true malformed_ranges_refused=4 root_admission_prior_bytes=unchanged service_healthy=true private_session_bytes_released=true host_validation_only=true iterations_per_round=100 rounds=5 alternating_order=true legacy_round_ns={legacy_ns:?} prepared_round_ns={prepared_ns:?} timing_threshold=none model_throughput_claim=none",
        resident.service.physical_key()
    );
    Ok(())
}
