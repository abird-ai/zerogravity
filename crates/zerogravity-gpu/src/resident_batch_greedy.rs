//! Source-exact native row reductions over one resident batch vocabulary.
//!
//! The existing physical owner retains every lease and validates completion.
//! This module never clears graph status, allocates buffers or publishes roots.
//! Terminal records are transported into independent original packet leases.

use crate::program_kernels::{self, ARGMAX_RECORD_WORDS};
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::{Error, Result};

const TILE: usize = 256;

pub(crate) struct GreedySubmission {
    pub dispatches: usize,
    /// Actual native sampler stages; excludes per-row packet transport.
    pub native_dispatches: usize,
}

/// Two original per-row reduction workspaces plus dense terminal records.
/// External independent 12-byte output packets remain separately admitted.
pub(crate) fn workspace_bytes(vocabulary: usize, rows: usize) -> Result<[usize; 3]> {
    if !matches!(rows, 2 | 4 | 8) {
        return Err(Error::Unsupported(
            "native greedy rows must be 2/4/8".into(),
        ));
    }
    let original = crate::resident_greedy::workspace_bytes(vocabulary)?;
    let scale = |bytes: usize| {
        bytes
            .checked_mul(rows)
            .filter(|n| u32::try_from(n / 4).is_ok())
            .ok_or_else(|| Error::Limit("native greedy workspace overflow".into()))
    };
    Ok([scale(original[0])?, scale(original[1])?, scale(12)?])
}

fn check_buffer(client: &Client, handle: &Handle, bytes: usize) -> Result<()> {
    if handle.service != client.service_id() || handle.size_in_used() < bytes as u64 {
        return Err(Error::State(
            "native greedy buffer service/size mismatch".into(),
        ));
    }
    Ok(())
}

/// Check the complete closed reduction/transport before the enclosing launch.
/// Every binding names a distinct allocation, including independent packets.
pub(crate) fn validate(
    client: &Client,
    logits: &Handle,
    vocabulary: usize,
    rows: usize,
    workspace: &[Handle; 3],
    private_packets: &[Handle],
    status: &Handle,
) -> Result<()> {
    let sizes = workspace_bytes(vocabulary, rows)?;
    if private_packets.len() != rows {
        return Err(Error::Malformed("native greedy packet row count".into()));
    }
    let logits_bytes = vocabulary
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(4))
        .filter(|n| u32::try_from(n / 4).is_ok())
        .ok_or_else(|| Error::Limit("native greedy logits overflow".into()))?;
    let mut buffers = vec![(logits, logits_bytes), (status, 4)];
    buffers.extend(workspace.iter().zip(sizes));
    buffers.extend(private_packets.iter().map(|packet| (packet, 12)));
    for (index, &(handle, bytes)) in buffers.iter().enumerate() {
        check_buffer(client, handle, bytes)?;
        let allocation = handle.clone().binding().memory.id();
        if buffers[..index]
            .iter()
            .any(|(prior, _)| (*prior).clone().binding().memory.id() == allocation)
        {
            return Err(Error::State(
                "native greedy bindings share an allocation".into(),
            ));
        }
    }
    let strides = [sizes[0] / 4 / rows, sizes[1] / 4 / rows];
    let mut count = program_kernels::argmax_partial_count(vocabulary, TILE)?;
    let mut current = 0;
    while count > 1 {
        let next = program_kernels::argmax_partial_count(count, TILE)?;
        if next * ARGMAX_RECORD_WORDS > strides[1 - current] {
            return Err(Error::State(
                "native greedy reduction exceeds workspace".into(),
            ));
        }
        count = next;
        current = 1 - current;
    }
    Ok(())
}

/// One physical dispatch per original reduction stage covers every row.
/// Source IDs remain local to each vocabulary; ties use the lowest original ID.
pub(crate) fn launch(
    client: &Client,
    logits: &Handle,
    vocabulary: usize,
    rows: usize,
    workspace: &[Handle; 3],
    private_packets: &[Handle],
    status: &Handle,
) -> Result<GreedySubmission> {
    validate(
        client,
        logits,
        vocabulary,
        rows,
        workspace,
        private_packets,
        status,
    )?;
    let sizes = workspace_bytes(vocabulary, rows)?;
    let strides = [sizes[0] / 4 / rows, sizes[1] / 4 / rows];
    let words = [sizes[0] / 4, sizes[1] / 4];
    let mut count = program_kernels::argmax_partial_count(vocabulary, TILE)?;
    let mut current = 0;
    let mut native_dispatches = 0;
    let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
    let threads = CubeDim::new_1d(32);
    // SAFETY: all layouts, complete stage capacities, source service and
    // distinct physical bindings pass the closed preflight above.
    unsafe {
        batch_argmax_logits::launch_unchecked(
            client,
            grid(rows * count),
            threads,
            BufferArg::from_raw_parts(logits.clone(), rows * vocabulary),
            BufferArg::from_raw_parts(workspace[0].clone(), words[0]),
            BufferArg::from_raw_parts(status.clone(), 1),
            vocabulary,
            rows,
            strides[0],
        );
        native_dispatches += 1;
        while count > 1 {
            let next = count.div_ceil(TILE);
            let destination = 1 - current;
            batch_argmax_partials::launch_unchecked(
                client,
                grid(rows * next),
                threads,
                BufferArg::from_raw_parts(workspace[current].clone(), words[current]),
                BufferArg::from_raw_parts(workspace[destination].clone(), words[destination]),
                BufferArg::from_raw_parts(status.clone(), 1),
                count,
                vocabulary,
                rows,
                strides[current],
                strides[destination],
            );
            native_dispatches += 1;
            count = next;
            current = destination;
        }
        batch_finish_argmax::launch_unchecked(
            client,
            grid(rows),
            threads,
            BufferArg::from_raw_parts(workspace[current].clone(), words[current]),
            BufferArg::from_raw_parts(workspace[2].clone(), rows * ARGMAX_RECORD_WORDS),
            BufferArg::from_raw_parts(status.clone(), 1),
            vocabulary,
            rows,
            strides[current],
        );
        native_dispatches += 1;
        for (row, packet) in private_packets.iter().enumerate() {
            program_kernels::copy_strided_words::launch_unchecked(
                client,
                grid(ARGMAX_RECORD_WORDS),
                threads,
                BufferArg::from_raw_parts(workspace[2].clone(), rows * ARGMAX_RECORD_WORDS),
                BufferArg::from_raw_parts(packet.clone(), ARGMAX_RECORD_WORDS),
                BufferArg::from_raw_parts(status.clone(), 1),
                (row * ARGMAX_RECORD_WORDS) as u32,
                0,
                ARGMAX_RECORD_WORDS as u32,
                ARGMAX_RECORD_WORDS as u32,
                1,
                ARGMAX_RECORD_WORDS,
                false,
            );
        }
    }
    Ok(GreedySubmission {
        dispatches: native_dispatches + rows,
        native_dispatches,
    })
}

#[cube(launch_unchecked)]
fn batch_argmax_logits(
    source: &[u32],
    partials: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] count: usize,
    #[comptime] rows: usize,
    #[comptime] destination_stride: usize,
) {
    let absolute = ABSOLUTE_POS;
    let groups = count.div_ceil(TILE);
    if count == 0usize || count > program_kernels::MAX_WORD_OPS {
        if absolute == 0usize {
            status[0].fetch_or(program_kernels::INVALID_REDUCTION);
        }
    } else if absolute < rows * groups {
        let row = absolute / groups;
        let group = absolute % groups;
        let mut found = false;
        let mut best_id = 0xffffffffu32;
        let mut best_bits = 0u32;
        let mut best_key = 0u32;
        let mut flags = 0u32;
        for lane in 0..TILE {
            let id = group * TILE + lane;
            if id < count {
                let src = u64::cast_from(row) * u64::cast_from(count) + u64::cast_from(id);
                if src < u64::cast_from(source.len()) {
                    let bits = source[usize::cast_from(src)];
                    if (bits & 0x7f800000u32) == 0x7f800000u32 {
                        flags |= program_kernels::NONFINITE_VALUE;
                    } else {
                        let key = program_kernels::finite_sortable_key(bits);
                        let token = u32::cast_from(id);
                        if !found || key > best_key || (key == best_key && token < best_id) {
                            found = true;
                            best_id = token;
                            best_bits = bits;
                            best_key = key;
                        }
                    }
                } else {
                    flags |= program_kernels::VIEW_OUT_OF_BOUNDS;
                }
            }
        }
        if !found {
            flags |= program_kernels::INVALID_REDUCTION;
        }
        let dst = u64::cast_from(row) * u64::cast_from(destination_stride)
            + u64::cast_from(group) * u64::cast_from(ARGMAX_RECORD_WORDS);
        if dst + 2u64 < u64::cast_from(partials.len()) {
            let d = usize::cast_from(dst);
            partials[d] = best_id;
            partials[d + 1] = best_bits;
            partials[d + 2] = flags;
        } else {
            flags |= program_kernels::VIEW_OUT_OF_BOUNDS;
        }
        if flags != 0u32 {
            status[0].fetch_or(flags);
        }
    }
}

#[cube(launch_unchecked)]
fn batch_argmax_partials(
    source: &[u32],
    partials: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] count: usize,
    #[comptime] vocabulary: usize,
    #[comptime] rows: usize,
    #[comptime] source_stride: usize,
    #[comptime] destination_stride: usize,
) {
    let absolute = ABSOLUTE_POS;
    let groups = count.div_ceil(TILE);
    if count == 0usize
        || count > program_kernels::MAX_WORD_OPS
        || vocabulary == 0usize
        || vocabulary > program_kernels::MAX_WORD_OPS
    {
        if absolute == 0usize {
            status[0].fetch_or(program_kernels::INVALID_REDUCTION);
        }
    } else if absolute < rows * groups {
        let row = absolute / groups;
        let group = absolute % groups;
        let mut found = false;
        let mut best_id = 0xffffffffu32;
        let mut best_bits = 0u32;
        let mut best_key = 0u32;
        let mut flags = 0u32;
        for lane in 0..TILE {
            let item = group * TILE + lane;
            if item < count {
                let src = u64::cast_from(row) * u64::cast_from(source_stride)
                    + u64::cast_from(item) * u64::cast_from(ARGMAX_RECORD_WORDS);
                if src + 2u64 < u64::cast_from(source.len()) {
                    let s = usize::cast_from(src);
                    let token = source[s];
                    let bits = source[s + 1];
                    flags |= source[s + 2];
                    if token >= u32::cast_from(vocabulary) {
                        flags |= program_kernels::INVALID_REDUCTION;
                    } else if (bits & 0x7f800000u32) == 0x7f800000u32 {
                        flags |= program_kernels::NONFINITE_VALUE;
                    } else {
                        let key = program_kernels::finite_sortable_key(bits);
                        if !found || key > best_key || (key == best_key && token < best_id) {
                            found = true;
                            best_id = token;
                            best_bits = bits;
                            best_key = key;
                        }
                    }
                } else {
                    flags |= program_kernels::VIEW_OUT_OF_BOUNDS;
                }
            }
        }
        if !found {
            flags |= program_kernels::INVALID_REDUCTION;
        }
        let dst = u64::cast_from(row) * u64::cast_from(destination_stride)
            + u64::cast_from(group) * u64::cast_from(ARGMAX_RECORD_WORDS);
        if dst + 2u64 < u64::cast_from(partials.len()) {
            let d = usize::cast_from(dst);
            partials[d] = best_id;
            partials[d + 1] = best_bits;
            partials[d + 2] = flags;
        } else {
            flags |= program_kernels::VIEW_OUT_OF_BOUNDS;
        }
        if flags != 0u32 {
            status[0].fetch_or(flags);
        }
    }
}

#[cube(launch_unchecked)]
fn batch_finish_argmax(
    source: &[u32],
    output: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] vocabulary: usize,
    #[comptime] rows: usize,
    #[comptime] source_stride: usize,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        let mut flags = status[0].load();
        let mut token = 0xffffffffu32;
        let mut bits = 0u32;
        let src = u64::cast_from(row) * u64::cast_from(source_stride);
        if src + 2u64 < u64::cast_from(source.len()) {
            let s = usize::cast_from(src);
            token = source[s];
            bits = source[s + 1];
            flags |= source[s + 2];
            if vocabulary == 0usize
                || vocabulary > program_kernels::MAX_WORD_OPS
                || token >= u32::cast_from(vocabulary)
            {
                flags |= program_kernels::INVALID_REDUCTION;
            }
            if (bits & 0x7f800000u32) == 0x7f800000u32 {
                flags |= program_kernels::NONFINITE_VALUE;
            }
        } else {
            flags |= program_kernels::VIEW_OUT_OF_BOUNDS;
        }
        if flags != 0u32 {
            token = 0xffffffffu32;
            status[0].fetch_or(flags);
        }
        let dst = row * ARGMAX_RECORD_WORDS;
        if dst + 2usize < output.len() {
            output[dst] = token;
            output[dst + 1] = bits;
            output[dst + 2] = flags;
        } else {
            status[0].fetch_or(program_kernels::VIEW_OUT_OF_BOUNDS);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trained_vocabulary_has_complete_original_row_workspaces() {
        for rows in [2usize, 4, 8] {
            assert_eq!(
                workspace_bytes(248_320, rows).unwrap(),
                [11_640 * rows, 48 * rows, 12 * rows]
            );
        }
        assert_eq!(workspace_bytes(1, 2).unwrap(), [24, 8, 24]);
    }

    #[test]
    fn refuse_unbounded_or_empty_native_greedy_geometry() {
        for rows in [0, 1, 3, 9, usize::MAX] {
            assert!(workspace_bytes(248_320, rows).is_err());
        }
        for vocabulary in [0, program_kernels::MAX_WORD_OPS + 1, usize::MAX] {
            assert!(workspace_bytes(vocabulary, 2).is_err());
        }
    }
}
