//! Closed learned BF16 layouts; original actor/init still owns actual copies.
use super::OwnedCudaQueue;
use cubecl_core::server::{Handle, ServerError};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceWeightLayout {
    GdnQkvz,
    GdnBa,
    GdnOutput,
    AttentionQkv,
    AttentionOutput,
    MlpGateUp,
    MlpDown,
    Qwen2bGdnQkvz,
    Qwen2bGdnBa,
    Qwen2bGdnOutput,
    Qwen2bAttentionQkv,
    Qwen2bAttentionOutput,
    Qwen2bMlpGateUp,
    Qwen2bMlpDown,
    Vocabulary08,
    Vocabulary2b,
}
impl SourceWeightLayout {
    pub const fn columns(self) -> usize {
        match self {
            Self::GdnOutput | Self::AttentionOutput => 2048,
            Self::MlpDown => 3584,
            Self::Qwen2bGdnQkvz
            | Self::Qwen2bGdnBa
            | Self::Qwen2bGdnOutput
            | Self::Qwen2bAttentionQkv
            | Self::Qwen2bAttentionOutput
            | Self::Qwen2bMlpGateUp
            | Self::Vocabulary2b => 2048,
            Self::Qwen2bMlpDown => 6144,
            _ => 1024,
        }
    }
    pub const fn rows(self) -> usize {
        match self {
            Self::Vocabulary08 | Self::Vocabulary2b => 248320,
            Self::GdnQkvz | Self::Qwen2bGdnQkvz => 8192,
            Self::GdnBa | Self::Qwen2bGdnBa => 32,
            Self::AttentionQkv | Self::Qwen2bAttentionQkv => 5120,
            Self::MlpGateUp => 7168,
            Self::Qwen2bMlpGateUp => 12288,
            Self::Qwen2bMlpDown | Self::Qwen2bGdnOutput | Self::Qwen2bAttentionOutput => 2048,
            _ => 1024,
        }
    }
    pub const fn bytes(self) -> usize {
        self.rows() * self.columns() * 2
    }
    pub(crate) fn window(self, offset: usize, bytes: usize) -> Result<(), ServerError> {
        let row = self.columns() * 2;
        if bytes == 0
            || bytes > 1024 * 1024
            || !offset.is_multiple_of(row)
            || !bytes.is_multiple_of(row)
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > self.bytes())
        {
            return Err(ServerError::graph_state(
                "closed original .8B learned weight window",
            ));
        }
        Ok(())
    }
}
impl OwnedCudaQueue {
    /// # Safety
    /// Same admitted immutable original source window is retained by the armed
    /// initialization BEFORE enqueue through real sync_buffers or quarantine.
    /// Destination is its complete uninitialized exclusive immutable-W lease;
    /// the closed model coverage owner validates source/member/offset/order.
    pub unsafe fn enqueue_vllm08_weight_window<T>(
        &self,
        destination: Handle,
        layout: SourceWeightLayout,
        offset: usize,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        layout.window(offset, source.as_ref().as_ref().len())?;
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.upload_owned_vllm08_weight_window(
                    stream,
                    destination,
                    layout,
                    offset,
                    source,
                )
            })
            .map_err(|error| {
                ServerError::graph_state(format!("source upload original actor: {error:?}"))
            })?
    }
    /// # Safety
    /// Exact admitted TensorWindow is installed on the actual armed flight's
    /// owner before enqueue; it stays there until this head's real checkpoint.
    #[cfg(all(feature = "owned-cubin", target_os = "linux"))]
    pub unsafe fn enqueue_streamed_head_window<T>(
        &self,
        destination: Handle,
        layout: SourceWeightLayout,
        source: Arc<T>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        crate::owned_command::cubin::head_policy::streamed::geometry(layout)?;
        if source.as_ref().as_ref().len() != 1024 * 1024 {
            return Err(ServerError::graph_state(
                "streamed head exact1MiB source window",
            ));
        }
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.upload_owned_original_window(stream, destination, 1024 * 1024, 0, source)
            })
            .map_err(|e| ServerError::graph_state(format!("streamed head H2D actor: {e:?}")))?
    }
    /// Real original-stream barrier after H2D+selected native consumption.
    /// # Safety
    /// Original armed flight retains complete W/Y/source/program owners before
    /// either operation. Success only credits reusable storage, never Ready.
    #[cfg(all(feature = "owned-cubin", target_os = "linux"))]
    pub unsafe fn checkpoint_streamed_head<T>(
        &self,
        weight: Handle,
        output: Handle,
        layout: SourceWeightLayout,
        ordinal: usize,
        source: Arc<T>,
        verification: Option<Arc<std::sync::Mutex<Vec<u8>>>>,
    ) -> Result<(), ServerError>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        let window = crate::owned_command::cubin::head_policy::streamed::Window {
            layout,
            input_rows: 1,
            ordinal,
        };
        window.validate()?;
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.checkpoint_owned_streamed_head(
                    stream,
                    weight,
                    output,
                    window.output_range()?,
                    source,
                    verification,
                )
            })
            .map_err(|e| {
                ServerError::graph_state(format!("streamed head checkpoint actor: {e:?}"))
            })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn closed_layout_extents_and_real_down_tail_are_exact() {
        for (layout, bytes) in [
            (SourceWeightLayout::GdnQkvz, 16777216),
            (SourceWeightLayout::GdnBa, 65536),
            (SourceWeightLayout::GdnOutput, 4194304),
            (SourceWeightLayout::AttentionQkv, 10485760),
            (SourceWeightLayout::AttentionOutput, 4194304),
            (SourceWeightLayout::MlpGateUp, 14680064),
            (SourceWeightLayout::MlpDown, 7340032),
        ] {
            assert_eq!(layout.bytes(), bytes);
            let row = layout.columns() * 2;
            for (offset, size) in [
                (1, row),
                (0, 0),
                (0, row - 1),
                (bytes, row),
                (usize::MAX - row + 1, row),
                (0, 1024 * 1024 + row),
            ] {
                assert!(layout.window(offset, size).is_err());
            }
        }
        // K3584 is not a divisor of1MiB. Never round a copy into source padding.
        let layout = SourceWeightLayout::MlpDown;
        let stride = 146 * 7168;
        for index in 0..7 {
            layout.window(index * stride, stride).unwrap();
        }
        layout.window(7 * stride, 2 * 7168).unwrap();
        assert_eq!(7 * stride + 2 * 7168, layout.bytes());
    }
    #[test]
    fn qwen2b_distinct_layouts_preserve_exact_rows_and_reject_padding() {
        let gate = SourceWeightLayout::Qwen2bMlpGateUp;
        let down = SourceWeightLayout::Qwen2bMlpDown;
        assert_eq!(
            (gate.rows(), gate.columns(), gate.bytes()),
            (12288, 2048, 50331648)
        );
        assert_eq!(
            (down.rows(), down.columns(), down.bytes()),
            (2048, 6144, 25165824)
        );
        assert_ne!(gate, SourceWeightLayout::MlpGateUp);
        assert_ne!(down, SourceWeightLayout::MlpDown);
        let chunk = 85 * 12288;
        for i in 0..24 {
            down.window(i * chunk, chunk).unwrap();
        }
        down.window(24 * chunk, 8 * 12288).unwrap();
        assert_eq!(24 * chunk + 8 * 12288, down.bytes());
        for (offset, size) in [
            (24 * chunk, 9 * 12288),
            (0, 86 * 12288),
            (1, 12288),
            (down.bytes(), 12288),
        ] {
            assert!(down.window(offset, size).is_err());
        }
        assert_eq!(SourceWeightLayout::MlpGateUp.bytes(), 14680064);
        assert_eq!(SourceWeightLayout::MlpDown.bytes(), 7340032);
    }
    #[test]
    fn text_2b_learned_layouts_preserve_real_widths_and_window_bounds() {
        for (layout, rows, bytes) in [
            (SourceWeightLayout::Qwen2bGdnQkvz, 8192, 33554432),
            (SourceWeightLayout::Qwen2bGdnBa, 32, 131072),
            (SourceWeightLayout::Qwen2bGdnOutput, 2048, 8388608),
            (SourceWeightLayout::Qwen2bAttentionQkv, 5120, 20971520),
            (SourceWeightLayout::Qwen2bAttentionOutput, 2048, 8388608),
        ] {
            assert_eq!(
                (layout.rows(), layout.columns(), layout.bytes()),
                (rows, 2048, bytes)
            );
            layout.window(0, 4096).unwrap();
            layout.window(bytes - 4096, 4096).unwrap();
            for (offset, len) in [(1, 4096), (0, 4095), (bytes, 4096), (0, 1048576 + 4096)] {
                assert!(layout.window(offset, len).is_err());
            }
        }
    }

    #[test]
    fn text_tied_tables_have_whole_source_rows_and_exact_bounded_windows() {
        for (layout, bytes, windows) in [
            (SourceWeightLayout::Vocabulary08, 508559360, 485),
            (SourceWeightLayout::Vocabulary2b, 1017118720, 970),
        ] {
            assert_eq!(layout.rows(), 248320);
            assert_eq!(layout.bytes(), bytes);
            assert_eq!(bytes / 1048576, windows);
            assert_eq!(bytes % 1048576, 0);
            for offset in (0..bytes).step_by(1048576) {
                layout.window(offset, 1048576).unwrap();
            }
            assert!(layout.window(bytes, layout.columns() * 2).is_err());
            assert!(layout.window(bytes - 1, 1).is_err());
        }
    }
}

/// An original-actor CUDA event owner. Non-Clone: each checked generation has
/// one reader; re-record consumes a mark whose previous generation retired.
/// This is transfer retirement, never initialization or Ready authority.
#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
pub struct OwnedCudaTransferRetirement {
    actor: cubecl_common::device_handle::DeviceHandle<crate::compute::CudaServer>,
    stream: cubecl_environment::stream::StreamId,
    event: usize,
    generation: u64,
    retired: bool,
}

#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
impl OwnedCudaTransferRetirement {
    pub fn ready(&mut self) -> Result<bool, ServerError> {
        if self.retired {
            return Ok(true);
        }
        let event = self.event;
        let stream = self.stream;
        let ready = self
            .actor
            .submit_blocking(move |server| {
                server.query_owned_source_retirement(stream, event, false)
            })
            .map_err(|e| {
                ServerError::graph_state(format!("transfer query original actor: {e:?}"))
            })??;
        self.retired = ready;
        Ok(ready)
    }

    /// Only waits for an already-recorded event. Every command it orders was
    /// submitted before this call; no later actor task is needed for progress.
    pub fn wait(&mut self) -> Result<(), ServerError> {
        if self.retired {
            return Ok(());
        }
        let event = self.event;
        let stream = self.stream;
        self.actor
            .submit_blocking(move |server| {
                server.query_owned_source_retirement(stream, event, true)
            })
            .map_err(|e| {
                ServerError::graph_state(format!("transfer wait original actor: {e:?}"))
            })??;
        // Caller must obtain an actual successful query before owner release.
        Ok(())
    }
}

#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
impl Drop for OwnedCudaTransferRetirement {
    fn drop(&mut self) {
        let event = self.event;
        // CUDA event destruction does not cancel stream work. Armed-flight
        // source owners must outlive the mark until query success/quarantine.
        let stream = self.stream;
        let _ = self
            .actor
            .submit_blocking(move |server| server.destroy_owned_source_retirement(stream, event));
    }
}

#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
impl OwnedCudaQueue {
    /// # Safety
    /// The original flight already retains exact W/Y/program and immutable
    /// host source BEFORE H2D/native enqueue. Same stream orders that native
    /// consumer before this mark and the next W overwrite. Keep those owners
    /// until successful query or original uncertain-work quarantine.
    pub unsafe fn mark_streamed_head_retirement(
        &self,
        weight: Handle,
        output: Handle,
        layout: SourceWeightLayout,
        ordinal: usize,
        reusable: Option<OwnedCudaTransferRetirement>,
    ) -> Result<OwnedCudaTransferRetirement, ServerError> {
        let range = crate::owned_command::cubin::head_policy::streamed::Window {
            layout,
            input_rows: 1,
            ordinal,
        }
        .output_range()?;
        let generation = if let Some(mark) = &reusable {
            next_retirement_generation(
                mark.actor.service_id() == self.actor.service_id(),
                mark.stream == self.stream,
                mark.retired,
                mark.generation,
            )?
        } else {
            1
        };
        let previous = reusable.as_ref().map(|mark| mark.event);
        let stream = self.stream;
        let event = self
            .actor
            .submit_blocking(move |server| {
                server.record_owned_source_retirement(
                    stream,
                    [weight, output],
                    [1024 * 1024, 248320 * 2],
                    range,
                    previous,
                )
            })
            .map_err(|e| {
                ServerError::graph_state(format!("transfer mark original actor: {e:?}"))
            })??;
        let mut mark = match reusable {
            Some(mark) => mark,
            None => OwnedCudaTransferRetirement {
                actor: self.actor.clone(),
                stream,
                event,
                generation,
                retired: true,
            },
        };
        mark.generation = generation;
        mark.retired = false;
        Ok(mark)
    }
}

#[cfg(all(feature = "owned-cubin", target_os = "linux"))]
fn next_retirement_generation(
    same_service: bool,
    same_stream: bool,
    retired: bool,
    previous: u64,
) -> Result<u64, ServerError> {
    if !same_service || !same_stream || !retired || previous == 0 {
        return Err(ServerError::graph_state(
            "transfer mark wrong service/stream or pending generation",
        ));
    }
    previous
        .checked_add(1)
        .ok_or_else(|| ServerError::graph_state("transfer generation exhausted"))
}

#[cfg(all(test, feature = "owned-cubin", target_os = "linux"))]
mod retirement_tests {
    use super::*;
    #[test]
    fn event_generation_reuse_requires_same_original_retired_owner() {
        assert_eq!(next_retirement_generation(true, true, true, 1).unwrap(), 2);
        assert_eq!(
            next_retirement_generation(true, true, true, 970).unwrap(),
            971
        );
        for (service, stream, retired, previous) in [
            (false, true, true, 1),
            (true, false, true, 1),
            (true, true, false, 1),
            (true, true, true, 0),
            (true, true, true, u64::MAX),
        ] {
            assert!(next_retirement_generation(service, stream, retired, previous).is_err());
        }
    }
}
