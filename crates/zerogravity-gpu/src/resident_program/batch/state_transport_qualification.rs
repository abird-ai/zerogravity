//! Original per-row physical kernels retained as the independent bit oracle.
use super::*;
use crate::resident_state_transport::{Direction, Row, Shape};

/// Call only after both new/old complete physical bindings pass preflight.
/// This is the original row loop and original kernels, never host expected math.
pub(crate) fn launch_original(
    client: &Client,
    rows: &[Row<'_>],
    dense_first: &Handle,
    dense_second: Option<&Handle>,
    shape: Shape,
    direction: Direction,
    status: &Handle,
) -> Result<usize> {
    let (words, total) = crate::resident_state_transport::geometry(shape, rows.len())?;
    // Same real views as candidate preflight; no device fallback/hidden upload.
    PreparedTransport::prepare(client, rows, dense_first, dense_second, shape, direction)?;
    for (row, private) in rows.iter().enumerate() {
        // SAFETY: actual preflight covers every original visible range and
        // the outer original flight retains all physical leases/status.
        unsafe {
            match shape {
                Shape::Vector { .. } => {
                    let (input, output, input_words, output_words, source, destination) =
                        if direction == Direction::Scatter {
                            (dense_first, private.first, total, words, row * words, 0)
                        } else {
                            (private.first, dense_first, words, total, 0, row * words)
                        };
                    crate::program_kernels::copy_strided_words::launch_unchecked(
                        client,
                        grid(words),
                        threads(),
                        arg(input, input_words),
                        arg(output, output_words),
                        arg(status, 1),
                        source as u32,
                        destination as u32,
                        words as u32,
                        words as u32,
                        1,
                        words,
                        false,
                    );
                }
                Shape::Kv { slab, capacity } => transport_kv::launch_unchecked(
                    client,
                    grid(words),
                    threads(),
                    arg(private.first, words),
                    arg(private.second.expect("checked original V"), words),
                    arg(dense_first, total),
                    arg(dense_second.expect("checked original denseV"), total),
                    row,
                    slab,
                    capacity,
                    rows.len(),
                    direction == Direction::Scatter,
                ),
            }
        }
    }
    Ok(rows.len())
}

mod physical;

// Typed certification has a separate differential/status/unknown qualifier.
mod certified;
