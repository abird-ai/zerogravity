//! Keep native BLAS on the original actor/stream and normal write scope.
use super::*;
use crate::{RocBlasGemm, prepared_native::failure, rocblas::Provider};
use std::ffi::c_void;

impl HipServer {
    pub(crate) fn prepare_rocblas(&mut self, key: u64) -> Result<(), ServerError> {
        self.require_native(key)?;
        let prepared = self
            .ctx
            .native
            .set_mut(key)
            .ok_or_else(|| failure("rocBLAS native set absent"))?;
        if prepared.rocblas.is_some() {
            return Err(failure("rocBLAS already attached"));
        }
        // Symbol discovery is cold. No foreign workspace allocator or stream.
        prepared.rocblas = Some(Provider::open()?);
        let result = prepared.rocblas.as_mut().unwrap().initialize();
        if result.is_err() {
            self.ctx.native.fence_all();
        }
        result
    }

    pub(crate) fn enqueue_rocblas(
        &mut self,
        key: u64,
        stream: StreamId,
        op: RocBlasGemm,
        handles: [Handle; 4],
    ) -> Result<(), ServerError> {
        self.require_native(key)?;
        self.ctx.native.require_stream(stream)?;
        let prepared = self
            .ctx
            .native
            .set(key)
            .ok_or_else(|| failure("rocBLAS native set absent"))?;
        let provider = prepared
            .rocblas
            .as_ref()
            .filter(|p| p.handle != 0)
            .ok_or_else(|| failure("rocBLAS provider not prepared"))?;
        let (api, handle) = (provider.api, provider.handle);
        let extents = op.extents()?;
        let service = self.utilities.service;
        for (i, h) in handles.iter().enumerate() {
            let bytes = h
                .size()
                .checked_sub(h.offset_start.unwrap_or(0))
                .and_then(|n| n.checked_sub(h.offset_end.unwrap_or(0)));
            if h.service != service || bytes != Some(extents[i] as u64) {
                return Err(failure("rocBLAS original binding extent/service"));
            }
        }
        let bindings = handles.map(Handle::binding);
        let mut written = self.write_set();
        written.extend([bindings[2].clone(), bindings[3].clone()]);
        ExecuteScope::over(self, stream, written)
            .execute(move |server| {
                server.streams.ensure_written(bindings[..2].iter())?;
                let mut command = server.command(stream, bindings.iter())?;
                if command.stream().capturing.is_recording() {
                    return Err(failure("rocBLAS graph capture is not admitted"));
                }
                let mut ptrs = [std::ptr::null_mut(); 4];
                for (i, b) in bindings.iter().enumerate() {
                    let actual = command.resource(b.clone())?;
                    if actual.size != extents[i] as u64 {
                        return Err(failure("rocBLAS physical view extent"));
                    }
                    ptrs[i] = actual.ptr as *mut c_void;
                    let start = ptrs[i] as usize;
                    let end = start
                        .checked_add(extents[i])
                        .ok_or_else(|| failure("rocBLAS physical range"))?;
                    if start == 0
                        || start % if i == 3 { 256 } else { op.element_bytes() } != 0
                        || (0..i).any(|j| {
                            let other = ptrs[j] as usize;
                            other
                                .checked_add(extents[j])
                                .is_none_or(|n| start < n && other < end)
                        })
                    {
                        return Err(failure("rocBLAS physical bindings overlap/alignment"));
                    }
                }
                unsafe { api.enqueue(handle, command.stream().sys, op, ptrs) }
            })
            .into_result()
    }
}
