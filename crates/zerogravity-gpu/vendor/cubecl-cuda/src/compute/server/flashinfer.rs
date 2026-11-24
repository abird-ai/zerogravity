//! FlashInfer pointer resolution and launch stay inside the original CUDA actor.

use super::*;
use crate::owned_command::{Bf16FlashInferDecode, Bf16FlashInferDecodeBindings};

unsafe extern "C" {
    fn zg_flashinfer_bf16_hdim256_decode(
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        out: *mut c_void,
        tmp: *mut c_void,
        kv_len: i32,
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
    pub(crate) fn enqueue_owned_flashinfer_decode(
        &mut self,
        stream: StreamId,
        operation: Bf16FlashInferDecode,
        bindings: Bf16FlashInferDecodeBindings,
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
        written.push(bindings.scratch.clone().binding());

        ExecuteScope::over(self, stream, written)
            .execute(move |server| {
                let native = bindings
                    .handles()
                    .map(|handle| handle.clone().binding());
                let mut command = server.command(stream, native.iter())?;
                if command.stream().capturing.is_recording() {
                    return Err(failure(
                        "FlashInfer decode graph capture is not qualified",
                    ));
                }

                let mut resources = [(0_u64, 0_u64); 5];
                for (index, (binding, bytes)) in native.into_iter().zip(sizes).enumerate() {
                    let resource = command.resource(binding)?;
                    if resource.size != bytes as u64
                        || resource.ptr == 0
                        || !resource.ptr.is_multiple_of(16)
                    {
                        return Err(failure(format!(
                            "FlashInfer binding {index} native extent/alignment differs"
                        )));
                    }
                    resources[index] = (
                        resource.ptr,
                        resource
                            .ptr
                            .checked_add(resource.size)
                            .ok_or_else(|| failure("FlashInfer binding address overflow"))?,
                    );
                }

                for left in 0..resources.len() {
                    for right in left + 1..resources.len() {
                        if resources[left].0 < resources[right].1
                            && resources[right].0 < resources[left].1
                        {
                            return Err(failure("FlashInfer bindings alias"));
                        }
                    }
                }

                let status = unsafe {
                    zg_flashinfer_bf16_hdim256_decode(
                        resources[0].0 as *mut c_void,
                        resources[1].0 as *mut c_void,
                        resources[2].0 as *mut c_void,
                        resources[3].0 as *mut c_void,
                        resources[4].0 as *mut c_void,
                        i32::try_from(operation.kv_len())
                            .map_err(|_| failure("FlashInfer KV length exceeds i32"))?,
                        i32::try_from(operation.query_heads())
                            .map_err(|_| failure("FlashInfer query heads exceed i32"))?,
                        i32::try_from(operation.kv_heads())
                            .map_err(|_| failure("FlashInfer KV heads exceed i32"))?,
                        operation.scale(),
                        command.stream().sys,
                    )
                };
                if status != 0 {
                    return Err(failure(format!(
                        "FlashInfer decode launch returned CUDA status {status}"
                    )));
                }
                Ok(())
            })
            .into_result()
    }
}
