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

/// Ping-pong scratch plus dense terminal records for continuing an already
/// source-exact greedy reduction from a row of three-word records.
#[allow(dead_code)]
pub(crate) fn continuation_workspace_bytes(source_count: usize, rows: usize) -> Result<[usize; 3]> {
    if !matches!(rows, 2 | 4 | 8)
        || source_count == 0
        || source_count > program_kernels::MAX_WORD_OPS
    {
        return Err(Error::Unsupported(
            "native greedy continuation geometry".into(),
        ));
    }
    let mut maxima = [1usize, 1usize];
    let mut count = source_count;
    let mut destination = 0usize;
    while count > 1 {
        let next = program_kernels::argmax_partial_count(count, TILE)?;
        maxima[destination] = maxima[destination].max(next);
        count = next;
        destination ^= 1;
    }
    let bytes = |records: usize| {
        records
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(ARGMAX_RECORD_WORDS * 4))
            .filter(|n| u32::try_from(n / 4).is_ok())
            .ok_or_else(|| Error::Limit("native greedy continuation workspace overflow".into()))
    };
    Ok([bytes(maxima[0])?, bytes(maxima[1])?, bytes(1)?])
}

fn check_buffer(client: &Client, handle: &Handle, bytes: usize) -> Result<()> {
    if handle.service != client.service_id() || handle.size_in_used() < bytes as u64 {
        return Err(Error::State(
            "native greedy buffer service/size mismatch".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum LogitLayout {
    F32,
    #[allow(dead_code)]
    PackedBf16NumericZero,
}

impl LogitLayout {
    fn bytes(self, vocabulary: usize, rows: usize) -> Result<usize> {
        if matches!(self, Self::PackedBf16NumericZero) && !vocabulary.is_multiple_of(2) {
            return Err(Error::Unsupported(
                "native packed BF16 greedy requires even vocabulary".into(),
            ));
        }
        vocabulary
            .checked_mul(rows)
            .and_then(|n| {
                n.checked_mul(match self {
                    Self::F32 => 4,
                    Self::PackedBf16NumericZero => 2,
                })
            })
            .filter(|n| u32::try_from(n.div_ceil(4)).is_ok())
            .ok_or_else(|| Error::Limit("native greedy logits overflow".into()))
    }
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
    validate_layout(
        client,
        logits,
        vocabulary,
        rows,
        workspace,
        private_packets,
        status,
        LogitLayout::F32,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_layout(
    client: &Client,
    logits: &Handle,
    vocabulary: usize,
    rows: usize,
    workspace: &[Handle; 3],
    private_packets: &[Handle],
    status: &Handle,
    layout: LogitLayout,
) -> Result<()> {
    let sizes = workspace_bytes(vocabulary, rows)?;
    if private_packets.len() != rows {
        return Err(Error::Malformed("native greedy packet row count".into()));
    }
    let logits_bytes = layout.bytes(vocabulary, rows)?;
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
    launch_layout(
        client,
        logits,
        vocabulary,
        rows,
        workspace,
        private_packets,
        status,
        LogitLayout::F32,
    )
}

#[allow(dead_code)]
pub(crate) fn launch_bf16_numeric_zero(
    client: &Client,
    logits: &Handle,
    vocabulary: usize,
    rows: usize,
    workspace: &[Handle; 3],
    private_packets: &[Handle],
    status: &Handle,
) -> Result<GreedySubmission> {
    launch_layout(
        client,
        logits,
        vocabulary,
        rows,
        workspace,
        private_packets,
        status,
        LogitLayout::PackedBf16NumericZero,
    )
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn launch_partials_numeric_zero(
    client: &Client,
    source: &Handle,
    source_count: usize,
    source_stride: usize,
    vocabulary: usize,
    rows: usize,
    workspace: &[Handle; 3],
    private_packets: &[Handle],
    status: &Handle,
) -> Result<GreedySubmission> {
    let sizes = continuation_workspace_bytes(source_count, rows)?;
    let required_source_stride = source_count
        .checked_mul(ARGMAX_RECORD_WORDS)
        .ok_or_else(|| Error::Limit("native greedy continuation source stride".into()))?;
    if source_stride < required_source_stride || private_packets.len() != rows {
        return Err(Error::Malformed(
            "native greedy continuation source/packet geometry".into(),
        ));
    }
    let source_bytes = source_stride
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Limit("native greedy continuation source bytes".into()))?;
    let mut buffers = vec![(source, source_bytes), (status, 4)];
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
                "native greedy continuation bindings share an allocation".into(),
            ));
        }
    }
    let scratch_strides = [sizes[0] / 4 / rows, sizes[1] / 4 / rows];
    let scratch_words = [sizes[0] / 4, sizes[1] / 4];
    let mut count = source_count;
    let mut current_source = source.clone();
    let mut current_words = source_bytes / 4;
    let mut current_stride = source_stride;
    let mut destination = 0usize;
    let mut native_dispatches = 0usize;
    let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
    let threads = CubeDim::new_1d(32);
    unsafe {
        while count > 1 {
            let next = count.div_ceil(TILE);
            batch_argmax_partials::launch_unchecked(
                client,
                grid(rows * next),
                threads,
                BufferArg::from_raw_parts(current_source, current_words),
                BufferArg::from_raw_parts(
                    workspace[destination].clone(),
                    scratch_words[destination],
                ),
                BufferArg::from_raw_parts(status.clone(), 1),
                count,
                vocabulary,
                rows,
                current_stride,
                scratch_strides[destination],
                true,
            );
            current_source = workspace[destination].clone();
            current_words = scratch_words[destination];
            current_stride = scratch_strides[destination];
            count = next;
            destination ^= 1;
            native_dispatches += 1;
        }
        batch_finish_argmax::launch_unchecked(
            client,
            grid(rows),
            threads,
            BufferArg::from_raw_parts(current_source, current_words),
            BufferArg::from_raw_parts(workspace[2].clone(), sizes[2] / 4),
            BufferArg::from_raw_parts(status.clone(), 1),
            vocabulary,
            rows,
            current_stride,
        );
        native_dispatches += 1;
        for (row, packet) in private_packets.iter().enumerate() {
            program_kernels::copy_strided_words::launch_unchecked(
                client,
                grid(ARGMAX_RECORD_WORDS),
                threads,
                BufferArg::from_raw_parts(workspace[2].clone(), sizes[2] / 4),
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

#[allow(clippy::too_many_arguments)]
fn launch_layout(
    client: &Client,
    logits: &Handle,
    vocabulary: usize,
    rows: usize,
    workspace: &[Handle; 3],
    private_packets: &[Handle],
    status: &Handle,
    layout: LogitLayout,
) -> Result<GreedySubmission> {
    validate_layout(
        client,
        logits,
        vocabulary,
        rows,
        workspace,
        private_packets,
        status,
        layout,
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
        match layout {
            LogitLayout::F32 => batch_argmax_logits::launch_unchecked(
                client,
                grid(rows * count),
                threads,
                BufferArg::from_raw_parts(logits.clone(), rows * vocabulary),
                BufferArg::from_raw_parts(workspace[0].clone(), words[0]),
                BufferArg::from_raw_parts(status.clone(), 1),
                vocabulary,
                rows,
                strides[0],
            ),
            LogitLayout::PackedBf16NumericZero => {
                batch_argmax_bf16_logits_numeric_zero::launch_unchecked(
                    client,
                    grid(rows * count),
                    threads,
                    BufferArg::from_raw_parts(logits.clone(), rows * vocabulary.div_ceil(2)),
                    BufferArg::from_raw_parts(workspace[0].clone(), words[0]),
                    BufferArg::from_raw_parts(status.clone(), 1),
                    vocabulary,
                    rows,
                    strides[0],
                )
            }
        }
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
                matches!(layout, LogitLayout::PackedBf16NumericZero),
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
fn batch_argmax_bf16_logits_numeric_zero(
    source: &[u32],
    partials: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] count: usize,
    #[comptime] rows: usize,
    #[comptime] destination_stride: usize,
) {
    let absolute = ABSOLUTE_POS;
    let groups = count.div_ceil(TILE);
    if count == 0usize || count > program_kernels::MAX_WORD_OPS || !count.is_multiple_of(2usize) {
        if absolute == 0usize {
            status[0].fetch_or(program_kernels::INVALID_REDUCTION);
        }
    } else if absolute < rows * groups {
        let row = absolute / groups;
        let group = absolute % groups;
        let row_words = count.div_ceil(2usize);
        let mut found = false;
        let mut best_id = 0xffffffffu32;
        let mut best_bits = 0u32;
        let mut best_key = 0u32;
        let mut flags = 0u32;
        for lane in 0..TILE {
            let id = group * TILE + lane;
            if id < count {
                let word =
                    u64::cast_from(row) * u64::cast_from(row_words) + u64::cast_from(id / 2usize);
                if word < u64::cast_from(source.len()) {
                    let shift = u32::cast_from(id % 2usize) * 16u32;
                    let bf16 = (source[usize::cast_from(word)] >> shift) & 0xffffu32;
                    if (bf16 & 0x7f80u32) == 0x7f80u32 {
                        flags |= program_kernels::NONFINITE_VALUE;
                    } else {
                        let bits = bf16 << 16u32;
                        let key = program_kernels::finite_numeric_sortable_key(bits);
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
    #[comptime] numeric_zero_ties: bool,
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
                        let mut key = program_kernels::finite_sortable_key(bits);
                        if comptime!(numeric_zero_ties) {
                            key = program_kernels::finite_numeric_sortable_key(bits);
                        }
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

    #[cfg(all(
        feature = "qualification",
        feature = "cuda-cublas",
        target_os = "linux"
    ))]
    #[test]
    #[ignore = "actual NVIDIA packed-BF16 B8 greedy versus independent numeric-zero oracle"]
    fn bf16_numeric_zero_b8_on_nvidia() -> Result<()> {
        use crate::resident_leases::ResidentLease;
        use zerogravity_core::device_events::{BufferRole, DeviceAccess};

        fn key(bits: u32) -> u32 {
            if bits & 0x7fff_ffff == 0 {
                0x8000_0000
            } else if bits & 0x8000_0000 != 0 {
                !bits
            } else {
                bits ^ 0x8000_0000
            }
        }
        fn expected(row: &[u16]) -> (u32, u32) {
            row.iter()
                .enumerate()
                .map(|(id, &bf16)| (id as u32, u32::from(bf16) << 16))
                .max_by(|(a_id, a_bits), (b_id, b_bits)| {
                    key(*a_bits).cmp(&key(*b_bits)).then_with(|| b_id.cmp(a_id))
                })
                .unwrap()
        }

        futures_lite::future::block_on(async {
            const ROWS: usize = 8;
            const VOCABULARY: usize = 512;
            let diagnostic = crate::cuda::test_support::CudaDiagnostic::new_on(
                "batch-bf16-greedy",
                cubecl::Device::cuda(0)
                    .map_err(|error| Error::State(format!("CUDA device 0 unavailable: {error}")))?,
            )?;
            let service = diagnostic.service.clone();
            let client = service.client().clone();
            let mut rows = vec![vec![0xbf80u16; VOCABULARY]; ROWS];
            rows[0][3] = 0x8000;
            rows[0][5] = 0x0000;
            rows[1][17] = 0x4080;
            rows[1][256] = 0x4080;
            rows[2][511] = 0x0001;
            rows[3][7] = 0x3f80;
            rows[4][31] = 0x4000;
            rows[5][63] = 0x4040;
            rows[6][127] = 0x40a0;
            rows[7][255] = 0x40c0;
            let expected = rows.iter().map(|row| expected(row)).collect::<Vec<_>>();
            let packed = rows
                .iter()
                .flat_map(|row| row.iter().flat_map(|word| word.to_ne_bytes()))
                .collect::<Vec<_>>();
            let sizes = workspace_bytes(VOCABULARY, ROWS)?;

            let (input, workspace, packets, status) = {
                let mut init = service.begin_initialization(0)?;
                let input = init.upload_bytes(&packed, BufferRole::ImmutableWeights)?;
                let workspace = sizes
                    .into_iter()
                    .map(|bytes| init.allocate_empty(bytes, BufferRole::Scratch))
                    .collect::<Result<Vec<_>>>()?;
                let packets = (0..ROWS)
                    .map(|_| init.allocate_empty(12, BufferRole::Output))
                    .collect::<Result<Vec<_>>>()?;
                let status = init.upload_zeroed(4, BufferRole::Scratch)?;
                drop(init.complete_sync().await?);
                (input, workspace, packets, status)
            };
            let workspace_handles: [Handle; 3] =
                std::array::from_fn(|index| workspace[index].handle());
            let packet_handles = packets
                .iter()
                .map(ResidentLease::handle)
                .collect::<Vec<_>>();
            let mut accesses = vec![
                (input.clone(), DeviceAccess::Read),
                (status.clone(), DeviceAccess::Write),
            ];
            accesses.extend(
                workspace
                    .iter()
                    .cloned()
                    .map(|lease| (lease, DeviceAccess::Write)),
            );
            accesses.extend(
                packets
                    .iter()
                    .cloned()
                    .map(|lease| (lease, DeviceAccess::Write)),
            );
            let flight = service.submit(0, &accesses)?;
            let receipt = launch_bf16_numeric_zero(
                &client,
                &input.handle(),
                VOCABULARY,
                ROWS,
                &workspace_handles,
                &packet_handles,
                &status.handle(),
            )?;
            if receipt.native_dispatches != 3 || receipt.dispatches != 3 + ROWS {
                return Err(Error::State(
                    "packed-BF16 B8 greedy physical dispatch count".into(),
                ));
            }
            let mut reads = vec![status.readback(0, 4)?];
            reads.extend(
                packets
                    .iter()
                    .map(|packet| packet.readback(0, 12))
                    .collect::<Result<Vec<_>>>()?,
            );
            let (ready, readback) = flight
                .complete_with_readback(&packets, &reads, |read| {
                    if &read[0][..] != 0u32.to_ne_bytes().as_slice() {
                        return Err(Error::State("packed-BF16 B8 greedy status".into()));
                    }
                    for (packet, &(id, bits)) in read[1..].iter().zip(&expected) {
                        let actual_id = u32::from_ne_bytes(packet[0..4].try_into().unwrap());
                        let actual_bits = u32::from_ne_bytes(packet[4..8].try_into().unwrap());
                        let flags = u32::from_ne_bytes(packet[8..12].try_into().unwrap());
                        if (actual_id, actual_bits, flags) != (id, bits, 0) {
                            return Err(Error::State(
                                "packed-BF16 B8 greedy packet differs".into(),
                            ));
                        }
                    }
                    Ok(())
                })
                .await?;
            drop(reads);
            drop(readback);
            drop(ready);
            drop(accesses);
            drop(packet_handles);
            drop(workspace_handles);
            drop(input);
            drop(workspace);
            drop(packets);
            drop(status);
            client
                .memory_cleanup()
                .map_err(|error| Error::State(format!("batch BF16 greedy cleanup: {error}")))?;
            let resident_bytes = service.resident_bytes()?;
            let events = service.model_events()?;
            if resident_bytes != 0 || events != 0 {
                return Err(Error::State(format!(
                    "batch BF16 greedy left resident bytes/events bytes={resident_bytes} events={events}"
                )));
            }
            service.close_model_namespace()?;
            println!("batch-bf16-greedy B8 PASS");
            Ok(())
        })
    }
}
