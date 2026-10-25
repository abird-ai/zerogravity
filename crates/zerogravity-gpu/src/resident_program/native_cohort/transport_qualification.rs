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
use std::collections::BTreeSet;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FixtureKind {
    Historical,
    Finite(usize),
    IsolatedStatus,
}

fn finite_word(row: usize, word: usize) -> u32 {
    // Exact widened BF16, finite at every row/position.
    ((0x0080 + (word.wrapping_mul(73).wrapping_add(row * 151) % 0x7e80)) as u32) << 16
}

// Native K/V and convolution cache pointers are BF16 (u16) views, while
// the complete readback oracle is U32. Preserve the untouched neighboring
// halfword and the original byte-order of the readback words.
fn write_native_bf16(words: &mut [u32], half_index: usize, value: u16) {
    let shift = (half_index % 2) * 16;
    let slot = &mut words[half_index / 2];
    *slot = (*slot & !(0xffff << shift)) | (u32::from(value) << shift);
}

fn read_native_bf16(words: &[u32], half_index: usize) -> u16 {
    (words[half_index / 2] >> ((half_index % 2) * 16)) as u16
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
    statuses: [ResidentLease; 2],
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
    // Qualify each implementation against its own untouched source/canary
    // fixture. Reusing the per-row flight's outputs could mask a grouped
    // kernel that omitted a write entirely.
    let mut grouped_launches = 0;
    for kind in [
        FixtureKind::Historical,
        FixtureKind::Finite(0),
        FixtureKind::Finite(1),
        FixtureKind::Finite(2),
        FixtureKind::Finite(3),
        FixtureKind::IsolatedStatus,
    ] {
        let reference = qualify_mode(device.clone(), host, owner.clone(), false, kind).await?;
        let grouped = qualify_mode(device.clone(), host, owner.clone(), true, kind).await?;
        if reference.0 != grouped.0
            || reference.1 != grouped.1
            || reference.2 != grouped.2
            || reference.3 != grouped.3 * 4
        {
            return Err(Error::State(
                "grouped transport independent flight census".into(),
            ));
        }
        grouped_launches += grouped.3;
    }
    // The caller's `cases` marker counts actual grouped transport launches;
    // fixture containers and observed leases are separate census units.
    Ok(grouped_launches)
}

async fn qualify_mode(
    device: Device,
    host: &PhysicalDeviceKey,
    owner: ResourceOwner,
    grouped: bool,
    kind: FixtureKind,
) -> Result<(usize, usize, usize, usize)> {
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
    let conv_value = |row: usize, word: usize| match kind {
        FixtureKind::Historical => conv_word(row, word),
        FixtureKind::IsolatedStatus if row == 2 && word == 17 => 0x7f80_0000,
        _ => finite_word(row, word),
    };
    let h_value = |row: usize, word: usize| match kind {
        FixtureKind::Historical => h_word(row, word),
        FixtureKind::IsolatedStatus if row == 2 && word == 17 => 0x7f80_0000,
        _ => finite_word(row + 1, word.wrapping_mul(31)),
    };
    let sticky = u32::from(kind == FixtureKind::IsolatedStatus) * 16;
    // ---- GDN conv chain (kinds 0/1, cached native side) ----
    let cache_words = (0..KV_CACHE_BYTES / 4).map(poison_word).collect::<Vec<_>>();
    let conv_cache = upload_words(&mut init, &cache_words)?;
    let mut sources = Vec::new();
    let mut candidates = Vec::new();
    for row in 0..4 {
        sources.push(upload_words(
            &mut init,
            &(0..CONV).map(|w| conv_value(row, w)).collect::<Vec<_>>(),
        )?);
        candidates.push(upload_words(&mut init, &vec![0x5a5a_5a5a; CONV])?);
    }
    let status_in = upload_words(&mut init, &[sticky])?;
    let status_out = upload_words(
        &mut init,
        &[if kind == FixtureKind::Historical {
            2
        } else {
            sticky
        }],
    )?;
    // Expected cache words: conv views = widened>>16 of the sources; every
    // other byte (H views, unused slots) stays poison.
    let mut conv_cache_expected = cache_words.clone();
    for row in 0..4 {
        let half_start = (1 + 4 * row) * SLOT_BYTES / 2;
        for native_index in 0..CONV {
            let channel = native_index % 6144;
            let frame = native_index / 6144;
            write_native_bf16(
                &mut conv_cache_expected,
                half_start + native_index,
                (conv_value(row, channel * 3 + frame) >> 16) as u16,
            );
        }
    }
    observe(&mut fixture, conv_cache.clone(), conv_cache_expected);
    let conv_error = !matches!(kind, FixtureKind::Finite(_));
    observe(
        &mut fixture,
        status_in.clone(),
        vec![sticky | u32::from(conv_error)],
    );
    observe(
        &mut fixture,
        status_out.clone(),
        vec![sticky | (u32::from(conv_error) * 2)],
    );
    for (row, source) in sources.iter().enumerate() {
        observe(
            &mut fixture,
            source.clone(),
            (0..CONV).map(|w| conv_value(row, w)).collect(),
        );
    }
    for (row, candidate) in candidates.iter().enumerate() {
        let expected = (0..CONV)
            .map(|w| {
                let bits = conv_value(row, w) >> 16;
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
            .map(|w| h_value(w / RECURRENT, w % RECURRENT))
            .collect::<Vec<_>>(),
    )?;
    for prefill in [false, true] {
        let mut sources = Vec::new();
        let mut destinations = Vec::new();
        for row in 0..4 {
            sources.push(upload_words(
                &mut init,
                &(0..RECURRENT).map(|w| h_value(row, w)).collect::<Vec<_>>(),
            )?);
            destinations.push(upload_words(&mut init, &vec![0x5a5a_5a5a; RECURRENT])?);
        }
        let status_in = upload_words(
            &mut init,
            &[if kind == FixtureKind::Historical {
                4
            } else {
                sticky
            }],
        )?;
        let status_out = upload_words(
            &mut init,
            &[if kind == FixtureKind::Historical {
                8
            } else {
                sticky
            }],
        )?;
        // Expected native side (cached: H views; prefill: H0/Ht staging) and
        // expected export candidates follow the exact U32 transpose.
        let native = |row: usize, w: usize| -> u32 {
            let head = w / 16384;
            let major = (w / 128) % 128;
            let minor = w % 128;
            // Import writes source[head*16384 + minor*128 + major] at [w].
            h_value(row, head * 16384 + minor * 128 + major)
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
                    .map(|w| h_value(w / RECURRENT, w % RECURRENT))
                    .collect(),
            );
        }
        let h_error = !matches!(kind, FixtureKind::Finite(_));
        observe(
            &mut fixture,
            status_in.clone(),
            vec![sticky | (u32::from(h_error) * 4)],
        );
        observe(
            &mut fixture,
            status_out.clone(),
            vec![sticky | (u32::from(h_error) * 8)],
        );
        for (row, source) in sources.iter().enumerate() {
            observe(
                &mut fixture,
                source.clone(),
                (0..RECURRENT).map(|w| h_value(row, w)).collect(),
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
                    h_value(row, head * 16384 + minor * 128 + major)
                } else {
                    h_value(row, w)
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
    drop(cache_words);
    // ---- KV fixtures ----
    // Historical: retain both original special-value/control populations.
    // Finite: clean, then negative base, base=end, end513 and block0/5/17;
    // each rejection has one bad row and finite data. Isolated: one import low16
    // error, then one export nonfinite in the appended native position.
    // At most two KV fixtures per flight: expected arrays and complete D2H
    // working copies must also fit the unchanged host 512MiB admission.
    let variants = match kind {
        FixtureKind::Finite(batch) => batch * 2..(batch * 2 + 2).min(7),
        _ => 0..2,
    };
    for variant in variants {
        let violation = kind == FixtureKind::Historical && variant == 1;
        let bases = if violation {
            [47, 48, 49, 50]
        } else {
            [47, 49, 51, 53]
        };
        let mut control_bases = bases.map(|base| base as i64);
        let mut ends = bases.map(|base| (base + 1) as i32);
        let mut blocks = [4i32, 8, 12, 16];
        if violation {
            ends[2] += 512;
        } else if matches!(kind, FixtureKind::Finite(_)) {
            match variant {
                1 => control_bases[2] = -1,
                2 => control_bases[2] = i64::from(ends[2]),
                3 => ends[2] = 513,
                4 => blocks[2] = 0,
                5 => blocks[2] = 5,
                6 => blocks[2] = 17,
                _ => {}
            }
        }
        let kv_value = |row: usize, word: usize| {
            if kind == FixtureKind::Historical {
                kv_word(row, word)
            } else if kind == FixtureKind::IsolatedStatus
                && variant == 0
                && row == 2
                && word == bases[2] * 512 - 1
            {
                finite_word(row + 2, word.wrapping_mul(131)) | 1
            } else {
                finite_word(row + 2, word.wrapping_mul(131))
            }
        };
        let mut cache_words = (0..KV_CACHE_BYTES / 4)
            .map(|w| {
                if kind != FixtureKind::Historical {
                    (finite_word(8, w * 2) >> 16) | (finite_word(9, w * 2 + 1) & 0xffff_0000)
                } else if w % 9973 == 7 {
                    0x7f80_0000
                } else {
                    0x3f00_0000 | (w as u32 & 0x007f_ffff)
                }
            })
            .collect::<Vec<_>>();
        if kind == FixtureKind::IsolatedStatus && variant == 1 {
            // Import stops before base: this one append-side Inf reaches
            // export without setting the import status bit.
            let native = 12 * 557_056 + bases[2] * 1024 + 17;
            write_native_bf16(&mut cache_words, native, 0x7f80);
        }
        let cache = upload_words(&mut init, &cache_words)?;
        let mut prior_k = Vec::new();
        let mut prior_v = Vec::new();
        let mut candidate_k = Vec::new();
        let mut candidate_v = Vec::new();
        for (row, &base) in bases.iter().enumerate() {
            let mut words = (0..CANONICAL_WORDS)
                .map(|w| kv_value(row, w))
                .collect::<Vec<_>>();
            for word in &mut words[base * 512..] {
                *word = 0xffff_ffff;
            }
            prior_k.push(upload_words(&mut init, &words)?);
            prior_v.push(upload_words(
                &mut init,
                &(0..CANONICAL_WORDS)
                    .map(|w| kv_value(row + 4, w))
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
            let raw = control_bases[row] as u64;
            positions[row * 2] = raw as i32;
            positions[row * 2 + 1] = (raw >> 32) as i32;
            seq_used[row] = ends[row];
            table[row * 4] = blocks[row];
        }
        let positions_lease = upload_i32(&mut init, &positions)?;
        let seq_lease = upload_i32(&mut init, &seq_used)?;
        let table_lease = upload_i32(&mut init, &table)?;
        let initial_status = if kind == FixtureKind::IsolatedStatus
            || (matches!(kind, FixtureKind::Finite(_)) && variant != 0)
        {
            16
        } else {
            0
        };
        let status_in = upload_words(&mut init, &[initial_status])?;
        // Preserve the historical shared epoch. New controls observe each
        // direction independently, so neither OR can mask the other.
        let status_out = if kind == FixtureKind::Historical {
            status_in.clone()
        } else {
            upload_words(&mut init, &[initial_status])?
        };
        // Expected: import writes the cache's live prefix (k>>16 at native
        // offsets), export writes candidates (native<<16 live / prior padding),
        // the guard-violation row copies nothing and ORs status.
        let mut expected_cache = cache_words.clone();
        let mut expected_status = [initial_status; 2];
        let guard_ok = |row: usize| {
            let base = control_bases[row];
            let end = i64::from(ends[row]);
            let block = i64::from(blocks[row]);
            if base < 0 || base >= end || end > 512 || !(4..=16).contains(&block) || block % 4 != 0
            {
                return false;
            }
            let highest = block as u64 * 557_056 + 512 + (end as u64 - 1) * 1024 + 256 + 255;
            (highest + 1) * 2 <= KV_CACHE_BYTES as u64
        };
        for (row, &base) in bases.iter().enumerate() {
            if !guard_ok(row) {
                expected_status[0] |= 1;
                continue;
            }
            let block = blocks[row] as usize;
            for index in 0..base * 512 {
                let position = index / 512;
                let head = (index % 512) / 256;
                let lane = index % 256;
                if kv_value(row, index) & 0xffff != 0
                    || kv_value(row + 4, index) & 0xffff != 0
                    || kv_value(row, index) & 0x7f80_0000 == 0x7f80_0000
                    || kv_value(row + 4, index) & 0x7f80_0000 == 0x7f80_0000
                {
                    expected_status[0] |= 1;
                }
                let native = block * 557_056 + head * 512 + position * 1024 + lane;
                write_native_bf16(
                    &mut expected_cache,
                    native,
                    (kv_value(row, index) >> 16) as u16,
                );
                write_native_bf16(
                    &mut expected_cache,
                    native + 256,
                    (kv_value(row + 4, index) >> 16) as u16,
                );
            }
        }
        // Export reads the POST-import cache; compute candidates exactly.
        let mut expected_candidates = Vec::with_capacity(4);
        for (row, &base) in bases.iter().enumerate() {
            let mut expected_k = vec![0x5a5a_5a5a; CANONICAL_WORDS];
            let mut expected_v = vec![0x5a5a_5a5a; CANONICAL_WORDS];
            let end = ends[row] as usize;
            let block = blocks[row] as usize;
            if !guard_ok(row) {
                expected_status[1] |= 1;
                expected_candidates.push((expected_k, expected_v));
                continue;
            }
            for index in 0..512 * 512 {
                let position = index / 512;
                let head = (index % 512) / 256;
                let lane = index % 256;
                let native = block * 557_056 + head * 512 + position * 1024 + lane;
                let (k, v) = if position < end {
                    (
                        u32::from(read_native_bf16(&expected_cache, native)) << 16,
                        u32::from(read_native_bf16(&expected_cache, native + 256)) << 16,
                    )
                } else {
                    (
                        if index >= base * 512 {
                            0xffff_ffff
                        } else {
                            kv_value(row, index)
                        },
                        kv_value(row + 4, index),
                    )
                };
                if position < end
                    && (k & 0x7f80_0000 == 0x7f80_0000 || v & 0x7f80_0000 == 0x7f80_0000)
                {
                    expected_status[1] |= 2;
                }
                expected_k[index] = k;
                expected_v[index] = v;
            }
            expected_candidates.push((expected_k, expected_v));
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
                            kv_value(row, w)
                        }
                    })
                    .collect(),
            );
            observe(
                &mut fixture,
                prior_v[row].clone(),
                (0..CANONICAL_WORDS).map(|w| kv_value(row + 4, w)).collect(),
            );
        }
        for row in 0..4 {
            let (expected_k, expected_v) = &expected_candidates[row];
            observe(&mut fixture, candidate_k[row].clone(), expected_k.clone());
            observe(&mut fixture, candidate_v[row].clone(), expected_v.clone());
        }
        if kind == FixtureKind::Historical {
            observe(
                &mut fixture,
                status_in.clone(),
                vec![expected_status[0] | expected_status[1]],
            );
        } else {
            observe(&mut fixture, status_in.clone(), vec![expected_status[0]]);
            observe(&mut fixture, status_out.clone(), vec![expected_status[1]]);
        }
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
            statuses: [status_in, status_out],
        });
    }
    // An original flight may read immutable controls and prior canonical
    // planes but only actual kernel destinations qualify as Ready outputs.
    // In particular, read-only source/canary snapshots are observed, never
    // presented as device writes or promoted as a new StateRoot packet.
    let mut written = BTreeSet::new();
    for case in &fixture.gdn {
        written.insert(case.status.id());
        if case.kind.is_multiple_of(2) {
            written.insert(case.cache.id());
        } else {
            written.extend(case.planes.iter().map(ResidentLease::id));
        }
    }
    for case in &fixture.kv {
        written.insert(case.cache.id());
        written.extend(case.statuses.iter().map(ResidentLease::id));
        written.extend(case.candidate_k.iter().map(ResidentLease::id));
        written.extend(case.candidate_v.iter().map(ResidentLease::id));
    }
    let counts = (fixture.gdn.len(), fixture.kv.len(), fixture.observed.len());
    let resident_bytes = fixture
        .observed
        .iter()
        .map(ResidentLease::bytes)
        .sum::<usize>();
    let expected_bytes = fixture
        .expected
        .iter()
        .map(|words| words.len() * 4)
        .sum::<usize>();
    // Expected words + two full readback working copies; cache/upload oracle
    // scratch, eight candidate planes and 16MiB metadata slack. Initialization
    // completes before readback; its byte buffers cannot overlap that phase.
    let host_upper = expected_bytes
        + 2 * resident_bytes
        + 3 * KV_CACHE_BYTES
        + 8 * KV_CANONICAL_BYTES
        + 16 * 1024 * 1024;
    if resident_bytes as u64 > payload
        || host_upper > CAP
        || fixture.expected.len() != fixture.observed.len()
    {
        return Err(Error::State(
            "grouped transport fixture host/payload/readback census".into(),
        ));
    }
    drop(init.complete_sync().await?);
    init_slot.finish()?;
    // One original, independently initialized physical flight per mode.
    // Both modes compare every byte to the same deterministic host oracle.
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
        .map(|lease| {
            (
                lease.clone(),
                if written.contains(&lease.id()) {
                    DeviceAccess::Write
                } else {
                    DeviceAccess::Read
                },
            )
        })
        .collect::<Vec<_>>();
    let outputs = accesses
        .iter()
        .filter(|(_, access)| *access == DeviceAccess::Write)
        .map(|(lease, _)| lease.clone())
        .collect::<Vec<_>>();
    if outputs.len() != written.len() {
        return Err(Error::State(
            "grouped transport physical write census".into(),
        ));
    }
    let flight = service.submit(0, &accesses)?;
    let mut launches = 0;
    for case in &fixture.gdn {
        launches += launch_gdn(&service, case, grouped)?;
    }
    for case in &fixture.kv {
        launches += launch_kv(&service, case, grouped)?;
    }
    if grouped {
        if launches != fixture.gdn.len() + fixture.kv.len() * 2 {
            return Err(Error::State("grouped transport launch population".into()));
        }
    } else {
        let kv_launches = fixture
            .kv
            .iter()
            .map(|case| 4 + case.bases.iter().filter(|&&base| base != 0).count())
            .sum::<usize>();
        if launches != fixture.gdn.len() * 4 + kv_launches {
            return Err(Error::State("per-row oracle launch population".into()));
        }
    }
    let reads = fixture
        .observed
        .iter()
        .map(|lease| lease.readback(0, lease.bytes()))
        .collect::<Result<Vec<_>>>()?;
    let expected = &fixture.expected;
    let (ready, raw) = flight
        .complete_with_readback(&outputs, &reads, |data| {
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
                            "grouped transport word differs population={kind:?} grouped={grouped} lease={index} word={word}"
                        )));
                    }
                }
            }
            Ok(())
        })
        .await?;
    drop(raw);
    drop(ready);
    slot.finish()?;
    // Flight-local strong clones outlive complete_with_readback; release them
    // at the completed boundary so the ledger can reach zero below.
    drop((reads, outputs, accesses));
    // The readback completed. Release every original and observed lease
    // before asserting physical device retirement, not merely queue silence.
    drop(fixture);
    drop((
        conv_cache, sources, candidates, status_in, status_out, h_cache, h0, ht,
    ));
    let resident = service.resident_bytes()?;
    let events = service.pending_events()?;
    if resident != 0 || events != 0 {
        return Err(Error::State(format!(
            "grouped transport retained lease/event bytes={resident} pending={events}"
        )));
    }
    service
        .client()
        .memory_cleanup()
        .map_err(|e| Error::State(format!("grouped transport pool retirement: {e}")))?;
    // The service owns its original physical provider envelope until Drop.
    // Check its live zero-byte/event ledger before Drop, then check final0.
    drop(service);
    drop(host_ticket);
    let final_state = arbiter.snapshot(&key)?;
    if final_state.fenced || final_state.reserved != 0 {
        return Err(Error::State(
            "grouped transport final original physical reservation".into(),
        ));
    }
    eprintln!(
        "GROUPED_TRANSPORT_MODE_PASS population={kind:?} grouped={grouped} gdn={} kv={} observed={} launches={launches} resident_bytes={resident} pending_events={events} arbiter_reserved={} arbiter_fenced={}",
        counts.0, counts.1, counts.2, final_state.reserved, final_state.fenced
    );
    Ok((counts.0, counts.1, counts.2, launches))
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
        let status = &case.statuses[usize::from(!incoming)];
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
                handles.push(status.handle());
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
                &status.handle(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_native_bf16_preserves_neighboring_half_and_u32_canaries() {
        let mut words = [0x1234_5678u32, 0xa5a5_5a5a];
        write_native_bf16(&mut words, 0, 0x7fc1);
        assert_eq!(words, [0x1234_7fc1, 0xa5a5_5a5a]);
        write_native_bf16(&mut words, 1, 0x8000);
        assert_eq!(words, [0x8000_7fc1, 0xa5a5_5a5a]);
        write_native_bf16(&mut words, 2, 0x0001);
        assert_eq!(words, [0x8000_7fc1, 0xa5a5_0001]);
        assert_eq!(read_native_bf16(&words, 0), 0x7fc1);
        assert_eq!(read_native_bf16(&words, 1), 0x8000);
        assert_eq!(read_native_bf16(&words, 2), 0x0001);
        assert_eq!(read_native_bf16(&words, 3), 0xa5a5);
    }

    #[test]
    fn conv_native_uses_bf16_halves_and_frame_channel_transposition() {
        let mut words = vec![0xa5a5_5a5a; CONV_NATIVE_BYTES / 4 + 1];
        for native_index in 0..CONV {
            let channel = native_index % 6144;
            let frame = native_index / 6144;
            write_native_bf16(
                &mut words,
                native_index,
                (conv_word(0, channel * 3 + frame) >> 16) as u16,
            );
        }
        for native_index in [0, 1, 6143, 6144, 12288, CONV - 1] {
            let channel = native_index % 6144;
            let frame = native_index / 6144;
            assert_eq!(
                read_native_bf16(&words, native_index),
                (conv_word(0, channel * 3 + frame) >> 16) as u16,
            );
        }
        assert_eq!(words[CONV_NATIVE_BYTES / 4], 0xa5a5_5a5a);
    }

    #[test]
    fn kv_native_halfword_bounds_cover_last_cache_block() {
        let mut words = vec![0x1234_5678; KV_CACHE_BYTES / 4];
        let index = 16 * 557_056 + 512 + 511 * 1024 + 256 + 255;
        write_native_bf16(&mut words, index, 0xff80);
        assert_eq!(read_native_bf16(&words, index), 0xff80);
        assert_eq!(words.len(), KV_CACHE_BYTES / 4);
        assert!(index / 2 < words.len());
    }
}
