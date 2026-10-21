//! Finite physical greedy fixtures under the existing resident owner.
//!
//! CPU total_cmp is a diagnostic oracle only. Only original input FP32 bits
//! are uploaded; every selected ID/max/status comes from actual GPU
//! kernels and physical readback. No root or session transition is published.
use crate::resident_leases::{ResidentLease, ResidentService};
use cubecl::prelude::*;
use std::sync::Arc;
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

const COUNTS: [usize; 8] = [1, 2, 255, 256, 257, 511, 512, 248_320];

struct Fixture {
    name: &'static str,
    values: Vec<f32>,
    expected_id: u32,
    expected_bits: u32,
    partial_bytes: [usize; 2],
}
struct DeviceFixture {
    fixture: Fixture,
    input: ResidentLease,
    partial: [ResidentLease; 2],
    packet: ResidentLease,
    status: ResidentLease,
}

fn oracle(values: &[f32]) -> Result<(u32, u32)> {
    let (first, rest) = values
        .split_first()
        .ok_or_else(|| Error::Malformed("empty finite greedy fixture".into()))?;
    if !first.is_finite() {
        return Err(Error::Malformed(
            "nonfinite greedy fixture oracle input".into(),
        ));
    }
    let mut id = 0u32;
    let mut best = *first;
    for (index, value) in rest.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::Malformed(
                "nonfinite greedy fixture oracle input".into(),
            ));
        }
        // Strictly greater retains the FIRST original ID for exact ties.
        if value.total_cmp(&best).is_gt() {
            best = *value;
            id = u32::try_from(index + 1)
                .map_err(|_| Error::Limit("greedy fixture index overflow".into()))?;
        }
    }
    Ok((id, best.to_bits()))
}

fn add_fixture(fixtures: &mut Vec<Fixture>, name: &'static str, values: Vec<f32>) -> Result<()> {
    let partial_bytes = crate::resident_greedy::workspace_bytes(values.len())?;
    let (expected_id, expected_bits) = oracle(&values)?;
    fixtures.push(Fixture {
        name,
        values,
        expected_id,
        expected_bits,
        partial_bytes,
    });
    Ok(())
}

fn fixtures() -> Result<Vec<Fixture>> {
    let mut fixtures = Vec::new();
    for count in COUNTS {
        let mut zeros = vec![f32::from_bits(0x8000_0000); count];
        if count > 1 {
            // An earlier -0 loses to later +0. Duplicate +0 winners preserve
            // their first ID, including across the 256-element tile boundary.
            zeros[17.min(count - 1)] = 0.0;
            zeros[count - 1] = 0.0;
        }
        add_fixture(&mut fixtures, "signed_zero", zeros)?;

        let mut positive = vec![f32::from_bits(1); count];
        positive[count - 1] = f32::from_bits(0x007f_ffff);
        add_fixture(&mut fixtures, "positive_subnormal", positive)?;

        let mut negative = vec![f32::from_bits(0x807f_ffff); count];
        negative[count - 1] = f32::from_bits(0x8000_0001);
        add_fixture(&mut fixtures, "negative_subnormal", negative)?;

        let mut duplicate = (0..count)
            .map(|i| f32::from_bits(0xbf00_0000 | (i as u32 & 0x007f_ffff)))
            .collect::<Vec<_>>();
        duplicate[17.min(count.saturating_sub(2))] = f32::from_bits(0x7f7f_ffff);
        duplicate[count - 1] = f32::from_bits(0x7f7f_ffff);
        add_fixture(&mut fixtures, "duplicate_maximum", duplicate)?;

        // Broad finite FP32 source bits, without floating conversions or
        // arithmetic which could pre-flush a subnormal or collapse zero signs.
        let mut state = 0x5135_791d_662b_8a47u64 ^ count as u64;
        let mut dense = Vec::with_capacity(count);
        for _ in 0..count {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut bits = state as u32;
            if bits & 0x7f80_0000 == 0x7f80_0000 {
                bits ^= 0x0080_0000; // Finite exponent 254; preserve sign/mantissa.
            }
            dense.push(f32::from_bits(bits));
        }
        add_fixture(&mut fixtures, "dense_finite_bits", dense)?;
    }
    Ok(fixtures)
}

fn checked_total(parts: impl IntoIterator<Item = usize>) -> Result<usize> {
    parts.into_iter().try_fold(0usize, |total, n| {
        total
            .checked_add(n)
            .ok_or_else(|| Error::Limit("finite greedy fixture byte overflow".into()))
    })
}

/// Every fixture buffer is preflighted before physical initialization. One
/// begun fair slot owns initialization and a bounded finite kernel/readback
/// batch. The submission retains every actual input/scratch/status/packet lease.
/// Nonfinite refusal fixtures belong to a separately disposable service.
pub(crate) async fn run(service: &Arc<ResidentService>) -> Result<usize> {
    service.ensure_usable()?;
    let fixtures = fixtures()?;
    let count = fixtures.len();
    let input_bytes = checked_total(fixtures.iter().map(|f| f.values.len() * 4))?;
    let scratch_bytes = checked_total(
        fixtures
            .iter()
            .map(|f| f.partial_bytes[0] + f.partial_bytes[1] + 4),
    )?;
    let output_bytes = count
        .checked_mul(12)
        .ok_or_else(|| Error::Limit("finite greedy fixture output byte overflow".into()))?;
    let required = checked_total([input_bytes, scratch_bytes, output_bytes])?;
    if required > service.remaining_capacity()? {
        return Err(Error::Limit(
            "complete finite greedy fixture set exceeds resident capacity".into(),
        ));
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
            AllocationRequest {
                kind: AllocationClass::Output,
                bytes: output_bytes as u64,
            },
        ],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    service.ensure_usable()?;
    // A queued peer can consume capacity before this worker is selected.
    // Refuse the entire fixture bundle before this attempt submits any work.
    if required > service.remaining_capacity()? {
        return Err(Error::Limit(
            "finite greedy fixture admission changed while queued".into(),
        ));
    }
    slot.begin_submission()?;
    let mut initialization = service.begin_initialization(0)?;
    let mut device = Vec::with_capacity(count);
    for fixture in fixtures {
        let input = initialization
            .upload_bytes(f32::as_bytes(&fixture.values), BufferRole::ImmutableWeights)?;
        let partial = [
            initialization.allocate_empty(fixture.partial_bytes[0], BufferRole::Scratch)?,
            initialization.allocate_empty(fixture.partial_bytes[1], BufferRole::Scratch)?,
        ];
        let packet = initialization.allocate_empty(12, BufferRole::Output)?;
        let status = initialization.allocate_empty(4, BufferRole::Scratch)?;
        device.push(DeviceFixture {
            fixture,
            input,
            partial,
            packet,
            status,
        });
    }
    let initialized = initialization.complete_sync().await?;
    if initialized.outputs().len() != count * 5 {
        return Err(Error::State(
            "finite greedy initialization lease coverage".into(),
        ));
    }
    drop(initialized); // DeviceFixture retains every original allocation.
    let mut accesses = Vec::with_capacity(count * 5);
    let mut packets = Vec::with_capacity(count);
    let mut cohort = Vec::with_capacity(count * 2);
    for f in &device {
        accesses.push((f.input.clone(), DeviceAccess::Read));
        accesses.extend(f.partial.iter().cloned().map(|p| (p, DeviceAccess::Write)));
        accesses.push((f.packet.clone(), DeviceAccess::Write));
        accesses.push((f.status.clone(), DeviceAccess::Write));
        packets.push(f.packet.clone());
        cohort.push(f.packet.readback(0, 12)?);
        cohort.push(f.status.readback(0, 4)?);
    }
    let flight = service.submit(0, &accesses)?;
    let mut dispatches = 0usize;
    for f in &device {
        // These raw views drop at the end of the iteration. Owning leases and
        // the armed submission outlive them, including every error/drop path.
        let input = f.input.handle();
        let partial0 = f.partial[0].handle();
        let partial1 = f.partial[1].handle();
        let packet = f.packet.handle();
        let status = f.status.handle();
        // SAFETY: one initialized atomic status word, admitted and retained
        // before launch, with no previous writer in this fresh fixture attempt.
        unsafe {
            crate::program_kernels::clear_step_status::launch_unchecked(
                service.client(),
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(32),
                BufferArg::from_raw_parts(status.clone(), 1),
            );
        }
        dispatches += 1 + crate::resident_greedy::launch(
            service.client(),
            &input,
            f.fixture.values.len(),
            &partial0,
            &partial1,
            &packet,
            &status,
        )?;
    }
    let (ready, read) = flight.complete_with_readback(&packets, &cohort, |read| {
        for (index, f) in device.iter().enumerate() {
            let packet = &read[index * 2];
            let status = &read[index * 2 + 1];
            if packet.len() != 12 || status.len() != 4 {
                return Err(Error::State("finite greedy physical packet/status length".into()));
            }
            let id = u32::from_ne_bytes(packet[0..4].try_into().expect("checked packet"));
            let bits = u32::from_ne_bytes(packet[4..8].try_into().expect("checked packet"));
            if packet[8..12].iter().chain(status.iter()).any(|b| *b != 0)
                || id != f.fixture.expected_id || bits != f.fixture.expected_bits {
                return Err(Error::State(format!(
                    "physical greedy fixture {} count={} differs: expected id={} bits={:08x}, got id={id} bits={bits:08x}, packet_status={:?} step_status={:?}",
                    f.fixture.name,f.fixture.values.len(),f.fixture.expected_id,f.fixture.expected_bits,&packet[8..12],status,
                )));
            }
        }
        Ok(())
    }).await?;
    slot.finish()?;
    println!(
        "H16_RESIDENT_GREEDY_FINITE_FIXTURES_PASS physical={:?} cases={count} sizes={COUNTS:?} exact_ids_and_max_bits=true total_cmp_signed_zero=true subnormals=true first_id_ties=true admitted_bytes={required} actual_readback_bytes={} dispatches={dispatches} published_roots=0",
        service.physical_key(),
        read.iter().map(|b| b.len()).sum::<usize>(),
    );
    drop(ready);
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_fixture_matrix_covers_reduction_boundaries_and_bitwise_order() {
        let fixtures = fixtures().unwrap();
        assert_eq!(fixtures.len(), COUNTS.len() * 5);
        for count in COUNTS {
            let group = fixtures
                .iter()
                .filter(|f| f.values.len() == count)
                .collect::<Vec<_>>();
            assert_eq!(group.len(), 5);
            assert!(group.iter().all(|f| f.values.iter().all(|v| v.is_finite())));
            let find = |name| *group.iter().find(|f| f.name == name).unwrap();
            let zero = find("signed_zero");
            assert_eq!(zero.expected_bits, if count == 1 { 0x8000_0000 } else { 0 });
            assert_eq!(
                zero.expected_id,
                if count == 1 {
                    0
                } else {
                    17.min(count - 1) as u32
                }
            );
            assert_eq!(find("positive_subnormal").expected_bits, 0x007f_ffff);
            assert_eq!(find("negative_subnormal").expected_bits, 0x8000_0001);
            let duplicate = find("duplicate_maximum");
            assert_eq!(
                duplicate.expected_id,
                17.min(count.saturating_sub(2)) as u32
            );
            assert_eq!(duplicate.expected_bits, 0x7f7f_ffff);
        }
        assert!(oracle(&[]).is_err());
        assert!(oracle(&[f32::NAN]).is_err());
        assert!(oracle(&[0.0, f32::INFINITY]).is_err());
    }
}
