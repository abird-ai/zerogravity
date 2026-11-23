//! FlashAttention pointer resolution and launch stay inside the original CUDA actor.

use super::*;
use crate::owned_command::{Bf16FlashAttention, Bf16FlashAttentionBindings};

unsafe extern "C" {
    fn zg_fa2_bf16_hdim256_causal(
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        out: *mut c_void,
        lse: *mut c_void,
        rows: i32,
        base: i32,
        capacity: i32,
        query_heads: i32,
        kv_heads: i32,
        scale: f32,
        stream: CUstream,
    ) -> i32;
}

fn failure(message: impl Into<String>) -> ServerError {
    ServerError::graph_state(message)
}

impl CudaServer {
    pub(crate) fn enqueue_owned_flash_attention(
        &mut self,
        stream: StreamId,
        operation: Bf16FlashAttention,
        bindings: Bf16FlashAttentionBindings,
    ) -> Result<(), ServerError> {
        let sizes = operation.sizes()?;
        self.streams.ensure_written(
            [
                &bindings.query.clone().binding(),
                &bindings.keys.clone().binding(),
                &bindings.values.clone().binding(),
            ]
            .into_iter(),
        )?;

        let mut written = self.write_set();
        written.push(bindings.output.clone().binding());
        written.push(bindings.softmax_lse.clone().binding());

        ExecuteScope::over(self, stream, written)
            .execute(move |server| {
                let native = bindings
                    .handles()
                    .map(|handle| handle.clone().binding());
                let mut command = server.command(stream, native.iter())?;
                if command.stream().capturing.is_recording() {
                    return Err(failure(
                        "FlashAttention provider graph capture is not qualified",
                    ));
                }

                let mut resources = [(0_u64, 0_u64); 5];
                for (index, (binding, bytes)) in native.into_iter().zip(sizes).enumerate() {
                    let resource = command.resource(binding)?;
                    let alignment = if index == 4 { 4 } else { 16 };
                    if resource.size != bytes as u64
                        || resource.ptr == 0
                        || !resource.ptr.is_multiple_of(alignment)
                    {
                        return Err(failure(format!(
                            "FlashAttention binding {index} native extent/alignment differs"
                        )));
                    }
                    resources[index] = (
                        resource.ptr,
                        resource
                            .ptr
                            .checked_add(resource.size)
                            .ok_or_else(|| failure("FlashAttention binding address overflow"))?,
                    );
                }

                for left in 0..resources.len() {
                    for right in left + 1..resources.len() {
                        if resources[left].0 < resources[right].1
                            && resources[right].0 < resources[left].1
                        {
                            return Err(failure("FlashAttention bindings alias"));
                        }
                    }
                }

                let status = unsafe {
                    zg_fa2_bf16_hdim256_causal(
                        resources[0].0 as *mut c_void,
                        resources[1].0 as *mut c_void,
                        resources[2].0 as *mut c_void,
                        resources[3].0 as *mut c_void,
                        resources[4].0 as *mut c_void,
                        i32::try_from(operation.rows())
                            .map_err(|_| failure("FlashAttention rows exceed i32"))?,
                        i32::try_from(operation.base())
                            .map_err(|_| failure("FlashAttention base exceeds i32"))?,
                        i32::try_from(operation.capacity())
                            .map_err(|_| failure("FlashAttention capacity exceeds i32"))?,
                        i32::try_from(operation.query_heads())
                            .map_err(|_| failure("FlashAttention query heads exceed i32"))?,
                        i32::try_from(operation.kv_heads())
                            .map_err(|_| failure("FlashAttention KV heads exceed i32"))?,
                        operation.scale(),
                        command.stream().sys,
                    )
                };
                if status != 0 {
                    return Err(failure(format!(
                        "FlashAttention FA2 launch returned CUDA status {status}"
                    )));
                }
                Ok(())
            })
            .into_result()
    }
}
