//! Bind existing source-exact greedy kernels to already admitted buffers.
//! No allocation, readback, host logits math, event or root authority.
//! The caller retains every original handle and keeps the shared step status
//! initialized from the graph launch; this helper NEVER clears its failures.
use crate::program_kernels::{self, ARGMAX_RECORD_WORDS};
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::Error;
use zerogravity_core::Result;

const TILE: usize = 256;

/// Peak bytes in the two disjoint ping-pong allocations. The second buffer
/// only needs the second stage's record count. An unused buffer remains a
/// valid four-byte admitted allocation; no kernel reads its undefined value.
/// Excludes resident logits, the 12-byte private packet and four-byte status.
pub(crate) fn workspace_bytes(vocabulary: usize) -> Result<[usize; 2]> {
    let first = program_kernels::argmax_partial_count(vocabulary, TILE)?;
    let second = if first > 1 {
        program_kernels::argmax_partial_count(first, TILE)?
    } else {
        0
    };
    let record_bytes = ARGMAX_RECORD_WORDS
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("resident greedy record byte overflow".into()))?;
    Ok([
        first
            .checked_mul(record_bytes)
            .ok_or_else(|| Error::Limit("resident greedy workspace overflow".into()))?,
        second
            .checked_mul(record_bytes)
            .ok_or_else(|| Error::Limit("resident greedy workspace overflow".into()))?
            .max(4),
    ])
}

fn check_buffer(client: &Client, handle: &Handle, bytes: usize) -> Result<()> {
    if handle.service != client.service_id() || handle.size_in_used() < bytes as u64 {
        return Err(Error::State(
            "resident greedy buffer service/size mismatch".into(),
        ));
    }
    Ok(())
}

/// All bindings are validated before the first dispatch. `private_packet`
/// receives [original token ID, original max bits, accumulated step status].
/// Its first u32 can bind directly to the next embedding token input.
///
/// Every buffer must use a distinct original allocation, even if overlapping
/// views could otherwise appear large enough. This prevents all read/write
/// aliasing, including status/logit and final-packet/partial hazards.
pub(crate) fn launch(
    client: &Client,
    logits: &Handle,
    vocabulary: usize,
    partial0: &Handle,
    partial1: &Handle,
    private_packet: &Handle,
    status: &Handle,
) -> Result<usize> {
    let sizes = workspace_bytes(vocabulary)?;
    let logits_bytes = vocabulary
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("resident greedy logits byte overflow".into()))?;
    let buffers = [logits, partial0, partial1, private_packet, status];
    let required = [logits_bytes, sizes[0], sizes[1], 12usize, 4usize];
    for (i, (&handle, &bytes)) in buffers.iter().zip(&required).enumerate() {
        check_buffer(client, handle, bytes)?;
        let allocation = handle.clone().binding().memory.id();
        if buffers[..i]
            .iter()
            .any(|prior| (*prior).clone().binding().memory.id() == allocation)
        {
            return Err(Error::State(
                "resident greedy bindings share an allocation".into(),
            ));
        }
    }

    let partials = [partial0, partial1];
    let words = [sizes[0] / 4, sizes[1] / 4];
    let mut count = program_kernels::argmax_partial_count(vocabulary, TILE)?;
    let mut current = 0usize;
    // Validate the complete lowering before the first possible GPU command.
    // The actual dispatch loop then has no fallible geometry operation.
    let mut checked_count = count;
    let mut checked_current = current;
    while checked_count > 1 {
        let next = program_kernels::argmax_partial_count(checked_count, TILE)?;
        let destination = 1 - checked_current;
        if next * ARGMAX_RECORD_WORDS > words[destination] {
            return Err(Error::State(
                "resident greedy stage exceeds workspace".into(),
            ));
        }
        checked_count = next;
        checked_current = destination;
    }
    let mut dispatches = 1usize;
    let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
    let threads = CubeDim::new_1d(32);
    // SAFETY: bounded checked vocabulary/partial counts, exact binding byte
    // capacities, common physical service and distinct allocations above.
    // Existing step status is accumulated, never reset between graph/sampler.
    unsafe {
        program_kernels::argmax_logits::launch_unchecked(
            client,
            grid(count),
            threads,
            BufferArg::from_raw_parts(logits.clone(), vocabulary),
            BufferArg::from_raw_parts(partial0.clone(), words[0]),
            BufferArg::from_raw_parts(status.clone(), 1),
            0,
            0,
            vocabulary,
            TILE,
        );
        while count > 1 {
            let next = count.div_ceil(TILE);
            let destination = 1 - current;
            program_kernels::argmax_partials::launch_unchecked(
                client,
                grid(next),
                threads,
                BufferArg::from_raw_parts(partials[current].clone(), words[current]),
                BufferArg::from_raw_parts(partials[destination].clone(), words[destination]),
                BufferArg::from_raw_parts(status.clone(), 1),
                0,
                0,
                count,
                vocabulary,
                TILE,
            );
            count = next;
            current = destination;
            dispatches += 1;
        }
        program_kernels::finish_argmax::launch_unchecked(
            client,
            grid(1),
            threads,
            BufferArg::from_raw_parts(partials[current].clone(), words[current]),
            BufferArg::from_raw_parts(private_packet.clone(), 3),
            BufferArg::from_raw_parts(status.clone(), 1),
            0,
            vocabulary,
        );
    }
    Ok(dispatches + 1)
}
