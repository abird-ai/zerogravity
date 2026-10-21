//! H22 physical bit-copy/certificate equivalence. No session is published.
//! Only source bits, initial guards and initial OR statuses are uploaded;
//! expected copied answers remain a host diagnostic oracle.
use super::*;
use cubecl::prelude::*;

const BOUNDS: u32 = crate::program_kernels::VIEW_OUT_OF_BOUNDS;
const NONFINITE: u32 = crate::program_kernels::NONFINITE_VALUE;
const MAX_WORDS: usize = crate::program_kernels::MAX_WORD_OPS;

struct Fixture {
    name: String,
    source: Vec<u32>,
    initial: Vec<u32>,
    source_offset: u32,
    destination_offset: u32,
    source_stride: u32,
    destination_stride: u32,
    rows: usize,
    width: usize,
    initial_status: u32,
    distinguish_untouched_nan: bool,
}
struct Expected {
    destination: Vec<u32>,
    // Legacy raw+certificate, fused copy, raw-only transport.
    status: [u32; 3],
}
struct DeviceFixture {
    fixture: Fixture,
    expected: Expected,
    source: ResidentLease,
    destination: [ResidentLease; 3],
    status: [ResidentLease; 3],
}

fn failure(detail: impl std::fmt::Display) -> Error {
    Error::State(format!("H22 View fixture: {detail}"))
}
fn finite(bits: u32) -> bool {
    bits & 0x7f80_0000 != 0x7f80_0000
}
fn guard_words(count: usize) -> Vec<u32> {
    (0..count).map(|i| 0x3f40_0000 | i as u32).collect()
}
fn finite_words(count: usize) -> Vec<u32> {
    (0..count)
        .map(|i| 0x3f00_0000 | ((i as u32 * 0x1021) & 0x007f_ffff))
        .collect()
}
fn geometry_ok(f: &Fixture) -> bool {
    f.width > 0 && f.rows > 0 && f.rows <= MAX_WORDS / f.width.max(1)
}

/// Integer-only byte-membership oracle. Invalid copies leave their initialized
/// destination unchanged; the old certificate independently visits its view.
fn oracle(f: &Fixture) -> Result<Expected> {
    let mut destination = f.initial.clone();
    let mut status = [f.initial_status; 3];
    if !geometry_ok(f) || (f.rows > 1 && (f.destination_stride as usize) < f.width) {
        status.iter_mut().for_each(|s| *s |= BOUNDS);
    } else {
        for row in 0..f.rows {
            for lane in 0..f.width {
                let src = u64::from(f.source_offset)
                    + row as u64 * u64::from(f.source_stride)
                    + lane as u64;
                let dst = u64::from(f.destination_offset)
                    + row as u64 * u64::from(f.destination_stride)
                    + lane as u64;
                if src < f.source.len() as u64 && dst < destination.len() as u64 {
                    let bits = f.source[src as usize];
                    destination[dst as usize] = bits;
                    if !finite(bits) {
                        status[1] |= NONFINITE;
                    }
                } else {
                    status.iter_mut().for_each(|s| *s |= BOUNDS);
                }
            }
        }
    }
    if !geometry_ok(f) {
        status[0] |= BOUNDS;
    } else {
        for row in 0..f.rows {
            for lane in 0..f.width {
                let dst = u64::from(f.destination_offset)
                    + row as u64 * u64::from(f.destination_stride)
                    + lane as u64;
                if dst >= destination.len() as u64 {
                    status[0] |= BOUNDS;
                } else if !finite(destination[dst as usize]) {
                    status[0] |= NONFINITE;
                }
            }
        }
    }
    if f.distinguish_untouched_nan {
        if status[0] ^ status[1] != NONFINITE || status[0] & BOUNDS == 0 || status[1] & BOUNDS == 0
        {
            return Err(failure(
                "untouched-NaN oracle did not retain its documented distinction",
            ));
        }
    } else if status[0] != status[1] {
        return Err(failure(format!(
            "{} has an unexpected legacy/fused mask distinction",
            f.name
        )));
    }
    Ok(Expected {
        destination,
        status,
    })
}

fn fixture(
    name: &str,
    source_len: usize,
    destination_len: usize,
    rows: usize,
    width: usize,
) -> Fixture {
    Fixture {
        name: name.into(),
        source: finite_words(source_len),
        initial: guard_words(destination_len),
        source_offset: 0,
        destination_offset: 0,
        source_stride: width as u32,
        destination_stride: width as u32,
        rows,
        width,
        initial_status: 0,
        distinguish_untouched_nan: false,
    }
}
fn fixtures() -> Vec<Fixture> {
    let mut cases = Vec::new();
    let patterns: [&[u32]; 3] = [
        &[
            0x7f7f_ffff,
            0xff7f_ffff,
            0x0080_0000,
            0x8080_0000,
            0x3f80_0000,
        ],
        &[0x8000_0000, 0, 1, 0x8000_0001, 0x007f_ffff, 0x807f_ffff],
        &[
            0x7fc1_2345,
            0xffc5_4321,
            0x7f81_2345,
            0xff81_2345,
            0x7f80_0000,
            0xff80_0000,
            0,
            0x8000_0000,
        ],
    ];
    for count in [1, 31, 32, 33, 255, 256, 257] {
        for (family, pattern) in patterns.iter().enumerate() {
            let mut f = fixture(
                &format!("dense-{count}-family-{family}"),
                count + 6,
                count + 8,
                1,
                count,
            );
            for (i, bits) in f.source.iter_mut().enumerate() {
                *bits = pattern[i % pattern.len()];
            }
            f.source_offset = 3;
            f.destination_offset = 4;
            if family == 0 {
                f.initial_status = 0x80;
            }
            cases.push(f);
        }
    }
    let mut f = fixture("strided-gaps-nonzero-offsets", 40, 32, 3, 4);
    f.source_offset = 3;
    f.destination_offset = 5;
    f.source_stride = 7;
    f.destination_stride = 6;
    cases.push(f);
    let mut f = fixture("broadcast-source", 12, 32, 5, 3);
    f.source_offset = 2;
    f.destination_offset = 3;
    f.source_stride = 0;
    f.destination_stride = 5;
    cases.push(f);
    let mut f = fixture("overlapping-source-windows", 16, 28, 4, 5);
    f.source_offset = 1;
    f.destination_offset = 2;
    f.source_stride = 2;
    cases.push(f);
    let mut f = fixture("source-tail-out-of-bounds", 9, 16, 3, 4);
    f.destination_offset = 2;
    cases.push(f);
    let mut f = fixture("destination-tail-out-of-bounds", 16, 9, 3, 4);
    f.destination_offset = 1;
    cases.push(f);
    let mut f = fixture("both-ranges-out-of-bounds", 12, 12, 2, 3);
    f.source_offset = 30;
    f.destination_offset = 31;
    cases.push(f);
    let mut f = fixture("u32-max-source-offset-and-stride", 12, 12, 2, 2);
    f.source_offset = u32::MAX;
    f.source_stride = u32::MAX;
    cases.push(f);
    let mut f = fixture("u32-max-destination-offset-and-stride", 12, 12, 2, 2);
    f.destination_offset = u32::MAX;
    f.destination_stride = u32::MAX;
    cases.push(f);
    let mut f = fixture("overlapping-destination-refused", 16, 16, 3, 4);
    f.destination_stride = 2;
    cases.push(f);
    cases.push(fixture("zero-rows", 8, 8, 0, 2));
    cases.push(fixture("zero-width", 8, 8, 2, 0));
    cases.push(fixture("too-many-rows", 8, 8, MAX_WORDS + 1, 1));
    cases.push(fixture("too-many-columns", 8, 8, 1, MAX_WORDS + 1));
    let mut f = fixture("invalid-copy-leaves-preexisting-NaN", 8, 8, 1, 4);
    f.source_offset = u32::MAX;
    f.initial[1] = 0x7f81_4567;
    f.distinguish_untouched_nan = true;
    cases.push(f);
    cases
}

fn word_view(handle: &Handle, count: usize) -> BufferArg {
    // SAFETY: caller binds complete admitted originals of exactly count words.
    unsafe { BufferArg::from_raw_parts(handle.clone(), count) }
}
fn check_words(bytes: &Bytes, expected: &[u32], label: &str) -> Result<()> {
    if bytes.len() != expected.len() * 4
        || bytes
            .chunks_exact(4)
            .zip(expected)
            .any(|(b, e)| u32::from_ne_bytes(b.try_into().expect("word chunk")) != *e)
    {
        return Err(failure(format!("{label}: physical word bits differ")));
    }
    Ok(())
}

pub(super) async fn qualify(resident: &Arc<ResidentModel>) -> Result<()> {
    let service = &resident.service;
    service.ensure_usable()?;
    let baseline = service.resident_bytes()?;
    let uploads = service.uploaded_value_bytes();
    let cases = fixtures();
    let count = cases.len();
    let mut input_bytes = 0usize;
    let mut scratch_bytes = 0usize;
    let checked_add = |a: usize, b| {
        a.checked_add(b)
            .ok_or_else(|| Error::Limit("H22 fixture byte overflow".into()))
    };
    for f in &cases {
        oracle(f)?;
        input_bytes = checked_add(input_bytes, f.source.len() * 4)?;
        scratch_bytes = checked_add(scratch_bytes, f.initial.len() * 12 + 12)?;
    }
    let required = checked_add(input_bytes, scratch_bytes)?;
    if required > service.remaining_capacity()? {
        return Err(Error::Limit("H22 fixture set exceeds admission".into()));
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[
            AllocationRequest {
                kind: AllocationClass::ImmutableWeights,
                bytes: input_bytes as u64,
            },
            AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: scratch_bytes as u64,
            },
        ],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    if required > service.remaining_capacity()? {
        return Err(Error::Limit(
            "H22 fixture admission changed while queued".into(),
        ));
    }
    slot.begin_submission()?;
    let mut initialization = service.begin_initialization(0)?;
    let mut device = Vec::with_capacity(count);
    for fixture in cases {
        let expected = oracle(&fixture)?;
        let source = initialization
            .upload_bytes(u32::as_bytes(&fixture.source), BufferRole::ImmutableWeights)?;
        let mut destination = Vec::new();
        let mut status = Vec::new();
        for _ in 0..3 {
            destination.push(
                initialization
                    .upload_bytes(u32::as_bytes(&fixture.initial), BufferRole::Scratch)?,
            );
            status.push(initialization.upload_bytes(
                u32::as_bytes(&[fixture.initial_status]),
                BufferRole::Scratch,
            )?);
        }
        device.push(DeviceFixture {
            fixture,
            expected,
            source,
            destination: destination
                .try_into()
                .map_err(|_| failure("destination fixture arity"))?,
            status: status
                .try_into()
                .map_err(|_| failure("status fixture arity"))?,
        });
    }
    let initialized = initialization.complete_sync().await?;
    if initialized.outputs().len() != count * 7 {
        return Err(failure("initialization original lease coverage"));
    }
    drop(initialized);
    let mut accesses = Vec::with_capacity(count * 7);
    let mut outputs = Vec::with_capacity(count * 6);
    let mut cohort = Vec::with_capacity(count * 7);
    for f in &device {
        accesses.push((f.source.clone(), DeviceAccess::Read));
        cohort.push(f.source.readback(0, f.source.bytes())?);
        for path in 0..3 {
            for l in [&f.destination[path], &f.status[path]] {
                accesses.push((l.clone(), DeviceAccess::Write));
                outputs.push(l.clone());
                cohort.push(l.readback(0, l.bytes())?);
            }
        }
    }
    let flight = service.submit(0, &accesses)?;
    for f in &device {
        let p = &f.fixture;
        let source = f.source.handle();
        let words = if geometry_ok(p) { p.rows * p.width } else { 1 };
        let grid = || CubeCount::Static((words as u32).div_ceil(32), 1, 1);
        for path in 0..3 {
            let destination = f.destination[path].handle();
            let status = f.status[path].handle();
            // SAFETY: originals are distinct, fully initialized and pinned by
            // the armed submission. Deliberately invalid logical views are
            // bounded by the helper's physical UInt64 guards before accesses.
            unsafe {
                crate::program_kernels::copy_strided_words::launch_unchecked(
                    service.client(),
                    grid(),
                    CubeDim::new_1d(32),
                    word_view(&source, p.source.len()),
                    word_view(&destination, p.initial.len()),
                    word_view(&status, 1),
                    p.source_offset,
                    p.destination_offset,
                    p.source_stride,
                    p.destination_stride,
                    p.rows,
                    p.width,
                    path == 1,
                );
                if path == 0 {
                    crate::program_kernels::certify_finite_strided::launch_unchecked(
                        service.client(),
                        grid(),
                        CubeDim::new_1d(32),
                        word_view(&destination, p.initial.len()),
                        word_view(&status, 1),
                        p.destination_offset,
                        p.destination_stride,
                        p.rows,
                        p.width,
                    );
                }
            }
        }
    }
    let (ready, read) = flight
        .complete_with_readback(&outputs, &cohort, |read| {
            for (i, f) in device.iter().enumerate() {
                check_words(&read[i * 7], &f.fixture.source, &f.fixture.name)?;
                for path in 0..3 {
                    check_words(
                        &read[i * 7 + 1 + path * 2],
                        &f.expected.destination,
                        &f.fixture.name,
                    )?;
                    check_words(
                        &read[i * 7 + 2 + path * 2],
                        &[f.expected.status[path]],
                        &f.fixture.name,
                    )?;
                }
            }
            Ok(())
        })
        .await?;
    slot.finish()?;
    let read_bytes = read.iter().map(|b| b.len()).sum::<usize>();
    let distinctions = device
        .iter()
        .filter(|f| f.fixture.distinguish_untouched_nan)
        .count();
    drop(ready);
    drop(cohort);
    drop(outputs);
    drop(accesses);
    drop(device);
    service.ensure_usable()?;
    if service.resident_bytes()? != baseline
        || service.pending_events()? != 0
        || service.uploaded_value_bytes().checked_sub(uploads) != Some(required as u64)
        || read_bytes != required
    {
        return Err(failure(
            "completed fixtures changed baseline admission/events or transfer accounting",
        ));
    }
    println!(
        "H22_RESIDENT_VIEW_FIXTURES_PASS physical={:?} cases={count} physical_paths=3 legacy_copy_and_certificate=exact fused_copy=exact raw_nonfinite_transport=exact full_source_and_destination_bits=exact untouched_guards_and_gaps=exact status_or=exact invalid_untouched_nan_mask_distinctions={distinctions} both_invalid_paths_refuse=true original_leases={} admitted_fixture_bytes={required} explicit_fixture_upload_bytes={required} actual_readback_bytes={read_bytes} dispatches={} published_roots=0 final_extra_resident_bytes=0",
        service.physical_key(),
        count * 7,
        count * 4
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixture_oracle_covers_exact_nonfinite_transport_and_the_invalid_mask_distinction() {
        let cases = fixtures();
        assert_eq!(cases.len(), 35);
        assert_eq!(
            cases.iter().filter(|f| f.distinguish_untouched_nan).count(),
            1
        );
        for f in cases {
            let expected = oracle(&f).unwrap();
            assert_eq!(
                expected.status[2] & NONFINITE,
                0,
                "{} raw transport",
                f.name
            );
            assert_eq!(expected.destination.len(), f.initial.len());
            if f.distinguish_untouched_nan {
                assert_eq!(expected.status, [BOUNDS | NONFINITE, BOUNDS, BOUNDS]);
            } else {
                assert_eq!(
                    expected.status[0], expected.status[1],
                    "{} finite certificate",
                    f.name
                );
            }
        }
    }
}
