//! Pointer resolution stays inside the original CudaServer actor.
use super::*;
use crate::owned_command::{Bf16GemmBindings, Bf16GemmInspection, Bf16GemmOperation};

impl CudaServer {
    pub(crate) fn inspect_owned_bf16_gemm(
        &mut self,
        stream: StreamId,
        operation: Bf16GemmOperation,
        bindings: Bf16GemmBindings,
    ) -> Result<Bf16GemmInspection, ServerError> {
        self.streams.ensure_written(
            [
                &bindings.weights.clone().binding(),
                &bindings.input.clone().binding(),
            ]
            .into_iter(),
        )?;
        let handles = bindings.handles();
        let native = handles.map(|handle| handle.clone().binding());
        let mut command = self.command(stream, native.iter())?;
        let mut extents = [(0_u64, 0_u64); 4];
        for (index, (binding, bytes)) in native
            .into_iter()
            .zip(operation.sizes()?)
            .enumerate()
        {
            let resource = command.resource(binding)?;
            if resource.size != bytes as u64
                || resource.ptr % (if index == 3 { 256 } else { 16 }) != 0
            {
                return Err(ServerError::graph_state(
                    "cuBLAS exact actual BF16/workspace extent/alignment",
                ));
            }
            extents[index] = (
                resource.ptr,
                resource
                    .ptr
                    .checked_add(resource.size)
                    .ok_or_else(|| ServerError::graph_state("cuBLAS binding address overflow"))?,
            );
        }
        for left in 0..extents.len() {
            for right in left + 1..extents.len() {
                if extents[left].0 < extents[right].1 && extents[right].0 < extents[left].1 {
                    return Err(ServerError::graph_state(
                        "cuBLAS immutable/input/output/workspace actual alias",
                    ));
                }
            }
        }
        Ok(Bf16GemmInspection {
            operation,
            bytes: operation.sizes()?,
            provider_allocations_admitted: false,
        })
    }
}
