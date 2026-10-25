//! Model-free physical qualification: the original per-row kernels are the
//! independent bit oracle, and a host-side model of the frozen equations is
//! the second oracle. Every grouped launch must leave ALL bytes (planes,
//! native views, canaries, status) identical to the per-row launches.
//! ROOT-owned isolated release CUDA process; #[ignore]d tests run it.
//! Compilation/mocks do not establish these kernels.
use super::{gdn_state, kv};
use crate::resident_leases::{ResidentLease, ResidentService};
use crate::resident_state_transport::typed::{
    self, CONV_ELEMENTS, CONV_NATIVE_BYTES, KV_CACHE_BYTES, KV_CANONICAL_BYTES,
    PreparedTypedTransport, RECURRENT_BYTES, RECURRENT_ELEMENTS, TypedDirection, TypedLayout,
    TypedRow,
};
use cubecl::{Device, server::Handle};
use std::cell::RefCell;
use zerogravity_core::{
    Error, Result,
    accelerator::SourceIdentity,
    device_events::{BufferRole, DeviceAccess},
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};

const CAP: usize = 512 * 1024 * 1024;
const SLOT_BYTES: usize = 278_528 * 4;
const CONV: usize = CONV_ELEMENTS;
const RECURRENT: usize = RECURRENT_ELEMENTS;
const CANONICAL_WORDS: usize = KV_CANONICAL_BYTES / 4;
/// subnormal, signed zero, finite, quiet NaN, two Infs, low16-nonzero widened.
const SPECIAL: [u32; 7] = [
    0x0000_0001,
    0x8000_0000,
    0x3f00_0000,
    0x7fc1_2345,
    0x7f80_0123,
    0xff80_4567,
    0x3f80_abcd,
];

fn conv_word(row: usize, word: usize) -> u32 {
    if word % 16 < SPECIAL.len() {
        SPECIAL[(word + row * 3) % SPECIAL.len()]
    } else {
        0x3f00_0000 | (word as u32).wrapping_mul(73).wrapping_add(row as u32 + 1) & 0x007f_ffff
    }
}
fn h_word(row: usize, word: usize) -> u32 {
    conv_word(row + 1, word.wrapping_mul(31))
}
fn kv_word(row: usize, word: usize) -> u32 {
    conv_word(row + 2, word.wrapping_mul(131))
}
fn poison_word(word: usize) -> u32 {
    0xa5a5_0000 | (word as u32 & 0x0000_ffff)
}

struct Fixture {
    /// Every observed lease, in readback order.
    observed: Vec<ResidentLease>,
    /// Expected final words per observed lease.
    expected: Vec<Vec<u32>>,
    gdn: Vec<GdnCase>,
    kv: Vec<KvCase>,
}
struct GdnCase {
    kind: usize,
    /// Cached native side lives in `cache`; prefill staging uses H0/Ht.
    prefill: bool,
    /// Import: canonical source planes; export: canonical destination planes.
    planes: Vec<ResidentLease>,
    cache: ResidentLease,
    status: ResidentLease,
}
struct KvCase {
    bases: [usize; 4],
    prior_k: Vec<ResidentLease>,
    prior_v: Vec<ResidentLease>,
    candidate_k: Vec<ResidentLease>,
    candidate_v: Vec<ResidentLease>,
    cache: ResidentLease,
    positions: ResidentLease,
    seq_used: ResidentLease,
    table: ResidentLease,
    status: ResidentLease,
}

fn plane(cache: &ResidentLease, offset: usize, bytes: usize) -> Handle {
    cache
        .handle()
        .offset_start(offset as u64)
        .offset_end((cache.bytes() - offset - bytes) as u64)
}
fn conv_view(cache: &ResidentLease, row: usize) -> Handle {
    plane(cache, (1 + 4 * row) * SLOT_BYTES, CONV_NATIVE_BYTES)
}
fn h_view(cache: &ResidentLease, row: usize) -> Handle {
    plane(
        cache,
        (1 + 4 * row) * SLOT_BYTES + CONV_NATIVE_BYTES,
        RECURRENT_BYTES,
    )
}

fn upload_words(
    init: &mut crate::resident_leases::ResidentInitialization<'_>,
    words: &[u32],
) -> Result<ResidentLease> {
    init.upload_bytes(
        &words
            .iter()
            .copied()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
        BufferRole::Scratch,
    )
}
fn upload_i32(
    init: &mut crate::resident_leases::ResidentInitialization<'_>,
    values: &[i32],
) -> Result<ResidentLease> {
    init.upload_bytes(
        &values
            .iter()
            .copied()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>(),
        BufferRole::ImmutableWeights,
    )
}

pub(crate) async fn qualify(
    device: Device,
    host: &PhysicalDeviceKey,
    owner: ResourceOwner,
) -> Result<usize> {
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "grouped transport release CUDA qualification only".into(),
        ));
    }
    let arbiter = ResourceArbiter::process_shared();
    let host_ticket = arbiter.admit(
        host,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: CAP as u64,
        }],
    )?;
    let service = ResidentService::new_pinned_cuda(
        device,
        SourceIdentity::for_new_allocation()?,
        owner,
        CAP,
        5 * 1024 * 1024 * 1024,
    )?;
    let key = service.physical_key();
    let payload = (320 * 1024 * 1024) as u64;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: payload,
        }],
        None,
    )?;
    let mut init_slot = ticket.wait().await?;
    init_slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let mut fixture = Fixture {
        observed: Vec::new(),
        expected: Vec::new(),
        gdn: Vec::new(),
        kv: Vec::new(),
    };
    let observe = |fixture: &mut Fixture, lease: ResidentLease, expected: Vec<u32>| -> usize {
        let index = fixture.observed.len();
        fixture.observed.push(lease);
        fixture.expected.push(expected);
        index
    };
    // ---- GDN conv chain (kinds 0/1, cached native side) ----
    let cache_words = (0..KV_CACHE_BYTES / 4).map(poison_word).collect::<Vec<_>>();
    let conv_cache = upload_words(&mut init, &cache_words)?;
    let mut sources = Vec::new();
    let mut candidates = Vec::new();
    for row in 0..4 {
        sources.push(upload_words(
            &mut init,
            &(0..CONV).map(|w| conv_word(row, w)).collect::<Vec<_>>(),
        )?);
        candidates.push(upload_words(&mut init, &vec![0x5a5a_5a5a; CONV])?);
    }
    let status_in = upload_words(&mut init, &[0])?;
    let status_out = upload_words(&mut init, &[2])?;
    // Expected cache words: conv views = widened>>16 of the sources; every
    // other byte (H views, unused slots) stays poison.
    let mut conv_cache_expected = cache_words.clone();
    for row in 0..4 {
        let offset = (1 + 4 * row) * SLOT_BYTES / 4;
        for w in 0..CONV {
            conv_cache_expected[offset + w] = conv_word(row, w) >> 16;
        }
    }
    observe(
        &mut fixture,
        conv_cache.clone(),
        conv_cache_expected.clone(),
    );
    observe(&mut fixture, status_in.clone(), vec![1]);
    observe(&mut fixture, status_out.clone(), vec![2]);
    for (row, source) in sources.iter().enumerate() {
        observe(
            &mut fixture,
            source.clone(),
            (0..CONV).map(|w| conv_word(row, w)).collect(),
        );
    }
    for (row, candidate) in candidates.iter().enumerate() {
        let expected = (0..CONV)
            .map(|w| {
                let bits = conv_word(row, w) >> 16;
                bits << 16
            })
            .collect::<Vec<_>>();
        observe(&mut fixture, candidate.clone(), expected);
    }
    fixture.gdn.push(GdnCase {
        kind: 0,
        prefill: false,
        planes: sources.clone(),
        cache: conv_cache.clone(),
        status: status_in.clone(),
    });
    fixture.gdn.push(GdnCase {
        kind: 1,
        prefill: false,
        planes: candidates.clone(),
        cache: conv_cache.clone(),
        status: status_out.clone(),
    });
    // ---- GDN H chain: cached and prefill native sides ----
    let h_cache = upload_words(&mut init, &cache_words)?;
    let h0 = upload_words(&mut init, &vec![0x5a5a_5a5a; 4 * RECURRENT])?;
    let ht = upload_words(
        &mut init,
        &(0..4 * RECURRENT)
            .map(|w| h_word(w / RECURRENT, w % RECURRENT))
            .collect::<Vec<_>>(),
    )?;
    for prefill in [false, true] {
        let mut sources = Vec::new();
        let mut destinations = Vec::new();
        for row in 0..4 {
            sources.push(upload_words(
                &mut init,
                &(0..RECURRENT).map(|w| h_word(row, w)).collect::<Vec<_>>(),
            )?);
            destinations.push(upload_words(&mut init, &vec![0x5a5a_5a5a; RECURRENT])?);
        }
        let status_in = upload_words(&mut init, &[4])?;
        let status_out = upload_words(&mut init, &[8])?;
        // Expected native side (cached: H views; prefill: H0/Ht staging) and
        // expected export candidates follow the exact U32 transpose.
        let native = |row: usize, w: usize| -> u32 {
            let head = w / 16384;
            let major = (w / 128) % 128;
            let minor = w % 128;
            // Import writes source[head*16384 + minor*128 + major] at [w].
            h_word(row, head * 16384 + minor * 128 + major)
        };
        if !prefill {
            let mut expected_cache = cache_words.clone();
            for row in 0..4 {
                let offset = ((1 + 4 * row) * SLOT_BYTES + CONV_NATIVE_BYTES) / 4;
                for w in 0..RECURRENT {
                    expected_cache[offset + w] = native(row, w);
                }
            }
            observe(&mut fixture, h_cache.clone(), expected_cache);
        } else {
            let mut expected_h0 = vec![0x5a5a_5a5a; 4 * RECURRENT];
            for row in 0..4 {
                for w in 0..RECURRENT {
                    expected_h0[row * RECURRENT + w] = native(row, w);
                }
            }
            observe(&mut fixture, h0.clone(), expected_h0);
            observe(
                &mut fixture,
                ht.clone(),
                (0..4 * RECURRENT)
                    .map(|w| h_word(w / RECURRENT, w % RECURRENT))
                    .collect(),
            );
        }
        observe(&mut fixture, status_in.clone(), vec![4]);
        observe(&mut fixture, status_out.clone(), vec![8]);
        for (row, source) in sources.iter().enumerate() {
            observe(
                &mut fixture,
                source.clone(),
                (0..RECURRENT).map(|w| h_word(row, w)).collect(),
            );
        }
        for (row, destination) in destinations.iter().enumerate() {
            let export = |w: usize| -> u32 {
                let head = w / 16384;
                let major = (w / 128) % 128;
                let minor = w % 128;
                // Export reads the native side at [head*16384 + minor*128 +
                // major]; transposing twice yields the original word at w.
                if prefill {
                    h_word(row, head * 16384 + minor * 128 + major)
                } else {
                    h_word(row, w)
                }
            };
            let expected = (0..RECURRENT).map(export).collect::<Vec<_>>();
            observe(&mut fixture, destination.clone(), expected);
        }
        fixture.gdn.push(GdnCase {
            kind: 2,
            prefill,
            planes: sources.clone(),
            cache: if prefill { h0.clone() } else { h_cache.clone() },
            status: status_in.clone(),
        });
        fixture.gdn.push(GdnCase {
            kind: 3,
            prefill,
            planes: destinations.clone(),
            cache: if prefill { ht.clone() } else { h_cache.clone() },
            status: status_out.clone(),
        });
    }
    // ---- KV fixtures ----
    for (bases, violation) in [([47, 49, 51, 53], false), ([47, 48, 49, 50], true)] {
        let cache_words = (0..KV_CACHE_BYTES / 4)
            .map(|w| {
                if w % 9973 == 7 {
                    0x7f80_0000
                } else {
                    0x3f00_0000 | (w as u32 & 0x007f_ffff)
                }
            })
            .collect::<Vec<_>>();
        let cache = upload_words(&mut init, &cache_words)?;
        let mut prior_k = Vec::new();
        let mut prior_v = Vec::new();
        let mut candidate_k = Vec::new();
        let mut candidate_v = Vec::new();
        for (row, &base) in bases.iter().enumerate() {
            let mut words = (0..CANONICAL_WORDS)
                .map(|w| kv_word(row, w))
                .collect::<Vec<_>>();
            for word in &mut words[base * 512..] {
                *word = 0xffff_ffff;
            }
            prior_k.push(upload_words(&mut init, &words)?);
            prior_v.push(upload_words(
                &mut init,
                &(0..CANONICAL_WORDS)
                    .map(|w| kv_word(row + 4, w))
                    .collect::<Vec<_>>(),
            )?);
            candidate_k.push(upload_words(
                &mut init,
                &vec![0x5a5a_5a5a; CANONICAL_WORDS],
            )?);
            candidate_v.push(upload_words(
                &mut init,
                &vec![0x5a5a_5a5a; CANONICAL_WORDS],
            )?);
        }
        let mut positions = vec![0i32; 6 * 513 * 2];
        let mut seq_used = vec![0i32; 16];
        let mut table = vec![0i32; 16];
        for row in 0..4 {
            let end = bases[row] + 1 + usize::from(violation && row == 2) * 512;
            positions[row * 2] = bases[row] as i32;
            positions[row * 2 + 1] = 0;
            seq_used[row] = end as i32;
            table[row * 4] = (4 + 4 * row) as i32;
        }
        let positions_lease = upload_i32(&mut init, &positions)?;
        let seq_lease = upload_i32(&mut init, &seq_used)?;
        let table_lease = upload_i32(&mut init, &table)?;
        let status = upload_words(&mut init, &[0])?;
        // Expected: import writes the cache's live prefix (k>>16 at native
        // offsets), export writes candidates (native<<16 live / prior padding),
        // the guard-violation row copies nothing and ORs status.
        let mut expected_cache = cache_words.clone();
        let mut expected_status = 0u32;
        for (row, &base) in bases.iter().enumerate() {
            let end = base + 1 + usize::from(violation && row == 2) * 512;
            let block = 4 + 4 * row;
            let guard_ok = base < end && end <= 512;
            for index in 0..base * 512 {
                let position = index / 512;
                let head = (index % 512) / 256;
                let lane = index % 256;
                if kv_word(row, index) & 0xffff != 0
                    || kv_word(row + 4, index) & 0xffff != 0
                    || kv_word(row, index) & 0x7f80_0000 == 0x7f80_0000
                    || kv_word(row + 4, index) & 0x7f80_0000 == 0x7f80_0000
                {
                    expected_status |= 1;
                }
                if !guard_ok {
                    continue;
                }
                let native = block * 557_056 + head * 512 + position * 1024 + lane;
                expected_cache[native] = kv_word(row, index) >> 16;
                expected_cache[native + 256] = kv_word(row + 4, index) >> 16;
            }
            if !guard_ok {
                expected_status |= 1;
            }
        }
        // Export reads the POST-import cache; compute candidates exactly.
        let mut expected_k = vec![0x5a5a_5a5a; CANONICAL_WORDS];
        let mut expected_v = vec![0x5a5a_5a5a; CANONICAL_WORDS];
        for (row, &base) in bases.iter().enumerate() {
            let end = base + 1 + usize::from(violation && row == 2) * 512;
            let block = 4 + 4 * row;
            let guard_ok = base < end && end <= 512;
            for index in 0..512 * 512 {
                if !guard_ok {
                    continue;
                }
                let position = index / 512;
                let head = (index % 512) / 256;
                let lane = index % 256;
                let native = block * 557_056 + head * 512 + position * 1024 + lane;
                let (k, v) = if position < end {
                    (
                        (expected_cache[native] & 0xffff) << 16,
                        (expected_cache[native + 256] & 0xffff) << 16,
                    )
                } else {
                    (kv_word(row, index), kv_word(row + 4, index))
                };
                if position < end
                    && (k & 0x7f80_0000 == 0x7f80_0000 || v & 0x7f80_0000 == 0x7f80_0000)
                {
                    expected_status |= 2;
                }
                expected_k[index] = k;
                expected_v[index] = v;
            }
        }
        observe(&mut fixture, cache.clone(), expected_cache);
        observe(
            &mut fixture,
            positions_lease.clone(),
            positions.iter().copied().map(|v| v as u32).collect(),
        );
        observe(
            &mut fixture,
            seq_lease.clone(),
            seq_used.iter().copied().map(|v| v as u32).collect(),
        );
        observe(
            &mut fixture,
            table_lease.clone(),
            table.iter().copied().map(|v| v as u32).collect(),
        );
        for row in 0..4 {
            let base = bases[row];
            observe(
                &mut fixture,
                prior_k[row].clone(),
                (0..CANONICAL_WORDS)
                    .map(|w| {
                        if w >= base * 512 {
                            0xffff_ffff
                        } else {
                            kv_word(row, w)
                        }
                    })
                    .collect(),
            );
            observe(
                &mut fixture,
                prior_v[row].clone(),
                (0..CANONICAL_WORDS).map(|w| kv_word(row + 4, w)).collect(),
            );
        }
        for row in 0..4 {
            observe(&mut fixture, candidate_k[row].clone(), expected_k.clone());
            observe(&mut fixture, candidate_v[row].clone(), expected_v.clone());
        }
        observe(&mut fixture, status.clone(), vec![expected_status]);
        fixture.kv.push(KvCase {
            bases,
            prior_k,
            prior_v,
            candidate_k,
            candidate_v,
            cache,
            positions: positions_lease,
            seq_used: seq_lease,
            table: table_lease,
            status,
        });
    }
    let counts = (fixture.gdn.len(), fixture.kv.len(), fixture.observed.len());
    drop(init.complete_sync().await?);
    init_slot.finish()?;
    // ---- Two flights: per-row oracle, then grouped; both must match the
    // host model exactly (=> grouped is bit-equal to the four per-row runs).
    let mut grouped_launches = 0;
    for grouped in [false, true] {
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: payload,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let accesses = fixture
            .observed
            .iter()
            .map(|lease| (lease.clone(), DeviceAccess::Write))
            .collect::<Vec<_>>();
        let flight = service.submit(0, &accesses)?;
        let mut launches = 0;
        for case in &fixture.gdn {
            launches += launch_gdn(&service, case, grouped)?;
        }
        for case in &fixture.kv {
            launches += launch_kv(&service, case, grouped)?;
        }
        if grouped {
            grouped_launches = launches;
        } else if launches != fixture.gdn.len() * 4 + fixture.kv.len() * 7 {
            return Err(Error::State("per-row oracle launch population".into()));
        }
        let reads = fixture
            .observed
            .iter()
            .map(|lease| lease.readback(0, lease.bytes()))
            .collect::<Result<Vec<_>>>()?;
        let expected: Vec<Vec<u32>> = fixture.expected.clone();
        let seen = RefCell::new(Vec::new());
        let (ready, raw) = flight
            .complete_with_readback(&fixture.observed, &reads, |data| {
                if data.len() != fixture.observed.len() {
                    return Err(Error::State(
                        "grouped transport whole readback coverage".into(),
                    ));
                }
                for (index, bytes) in data.iter().enumerate() {
                    if bytes.len() != expected[index].len() * 4 {
                        return Err(Error::State(
                            "grouped transport readback byte extent".into(),
                        ));
                    }
                    for (word, raw) in bytes.chunks_exact(4).enumerate() {
                        let want = expected[index][word].to_le_bytes();
                        if raw != want {
                            return Err(Error::State(format!(
                                "grouped transport word differs lease={index} word={word}"
                            )));
                        }
                    }
                    seen.borrow_mut().push(bytes.to_vec());
                }
                Ok(())
            })
            .await?;
        drop(seen);
        drop(raw);
        drop(ready);
        slot.finish()?;
    }
    drop(host_ticket);
    let final_state = arbiter.snapshot(&key)?;
    if final_state.fenced || final_state.reserved != 0 {
        return Err(Error::State(
            "grouped transport final original physical reservation".into(),
        ));
    }
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "grouped transport retained lease/event".into(),
        ));
    }
    service
        .client()
        .memory_cleanup()
        .map_err(|e| Error::State(format!("grouped transport pool retirement: {e}")))?;
    Ok(counts.0 + counts.1 + grouped_launches)
}

fn launch_gdn(service: &ResidentService, case: &GdnCase, grouped: bool) -> Result<usize> {
    let elements = if case.kind < 2 { CONV } else { RECURRENT };
    let import = case.kind.is_multiple_of(2);
    let planes = |row: usize| -> (Handle, Handle) {
        match case.kind {
            0 => (case.planes[row].handle(), conv_view(&case.cache, row)),
            1 => (conv_view(&case.cache, row), case.planes[row].handle()),
            2 if case.prefill => (
                case.planes[row].handle(),
                case.cache
                    .handle()
                    .offset_start((row * RECURRENT_BYTES) as u64)
                    .offset_end((case.cache.bytes() - (row + 1) * RECURRENT_BYTES) as u64),
            ),
            2 => (case.planes[row].handle(), h_view(&case.cache, row)),
            3 if case.prefill => (
                case.cache
                    .handle()
                    .offset_start((row * RECURRENT_BYTES) as u64)
                    .offset_end((case.cache.bytes() - (row + 1) * RECURRENT_BYTES) as u64),
                case.planes[row].handle(),
            ),
            _ => (h_view(&case.cache, row), case.planes[row].handle()),
        }
    };
    if !grouped {
        let kernel = gdn_state::definition(case.kind)?;
        for row in 0..4 {
            let (source, destination) = planes(row);
            kernel.launch(
                service.client(),
                [(elements as u32).div_ceil(128), 1, 1],
                &[source, destination, case.status.handle()],
            )?;
        }
        return Ok(4);
    }
    // Collect the row planes first so the TypedRow borrows stay alive for
    // the grouped bind (the closure returns owned handles).
    let row_planes = (0..4).map(planes).collect::<Vec<_>>();
    let rows = row_planes
        .iter()
        .map(|(source, destination)| {
            if import {
                TypedRow {
                    prior: Some(source),
                    candidate: None,
                    native: destination,
                    second: None,
                    second_prior: None,
                    controls: None,
                }
            } else {
                TypedRow {
                    prior: None,
                    candidate: Some(destination),
                    native: source,
                    second: None,
                    second_prior: None,
                    controls: None,
                }
            }
        })
        .collect::<Vec<_>>();
    PreparedTypedTransport::prepare(
        service.client(),
        &typed::gdn_definition(case.kind)?,
        &rows,
        &case.status.handle(),
        if case.kind < 2 {
            TypedLayout::Conv
        } else {
            TypedLayout::RecurrentH
        },
        if import {
            TypedDirection::Import
        } else {
            TypedDirection::Export
        },
        [(elements as u32).div_ceil(128), 4, 1],
    )?
    .launch(service.client())?;
    Ok(1)
}

fn launch_kv(service: &ResidentService, case: &KvCase, grouped: bool) -> Result<usize> {
    let mut launches = 0;
    for incoming in [true, false] {
        let views = |row: usize| -> [Handle; 3] {
            [
                case.positions
                    .handle()
                    .offset_start((row * 8) as u64)
                    .offset_end((case.positions.bytes() - row * 8 - 8) as u64),
                case.seq_used
                    .handle()
                    .offset_start((row * 4) as u64)
                    .offset_end((case.seq_used.bytes() - row * 4 - 4) as u64),
                case.table
                    .handle()
                    .offset_start((row * 16) as u64)
                    .offset_end((case.table.bytes() - row * 16 - 4) as u64),
            ]
        };
        if !grouped {
            let kernel = kv::definition(incoming)?;
            for row in 0..4 {
                if incoming && case.bases[row] == 0 {
                    continue;
                }
                let count = if incoming {
                    case.bases[row] * 512
                } else {
                    512 * 512
                };
                let mut handles = vec![
                    case.prior_k[row].handle(),
                    case.prior_v[row].handle(),
                    case.cache.handle(),
                ];
                if !incoming {
                    handles.extend([
                        case.candidate_k[row].handle(),
                        case.candidate_v[row].handle(),
                    ]);
                }
                handles.extend(views(row));
                handles.push(case.status.handle());
                kernel.launch(
                    service.client(),
                    [(count as u32).div_ceil(128), 1, 1],
                    &handles,
                )?;
                launches += 1;
            }
        } else {
            // Collect per-row handles first so the TypedRow borrows stay alive
            // for the grouped bind (the closures return owned handles).
            let row_handles = (0..4)
                .map(|row| {
                    (
                        case.prior_k[row].handle(),
                        case.prior_v[row].handle(),
                        case.candidate_k[row].handle(),
                        case.candidate_v[row].handle(),
                        case.cache.handle(),
                        views(row),
                    )
                })
                .collect::<Vec<_>>();
            let rows = row_handles
                .iter()
                .map(
                    |(prior_k, prior_v, candidate_k, candidate_v, cache, view)| {
                        if incoming {
                            TypedRow {
                                prior: Some(prior_k),
                                candidate: None,
                                native: cache,
                                second: Some(prior_v),
                                second_prior: None,
                                controls: Some(view.each_ref()),
                            }
                        } else {
                            TypedRow {
                                prior: Some(prior_k),
                                candidate: Some(candidate_k),
                                native: cache,
                                second: Some(candidate_v),
                                second_prior: Some(prior_v),
                                controls: Some(view.each_ref()),
                            }
                        }
                    },
                )
                .collect::<Vec<_>>();
            let max_base = case.bases.iter().copied().max().unwrap_or(0);
            let words = if incoming { max_base * 512 } else { 512 * 512 };
            PreparedTypedTransport::prepare(
                service.client(),
                &typed::kv_definition(incoming)?,
                &rows,
                &case.status.handle(),
                TypedLayout::Kv,
                if incoming {
                    TypedDirection::Import
                } else {
                    TypedDirection::Export
                },
                [(words as u32).div_ceil(128), 4, 1],
            )?
            .launch(service.client())?;
            launches += 1;
        }
    }
    Ok(launches)
}
