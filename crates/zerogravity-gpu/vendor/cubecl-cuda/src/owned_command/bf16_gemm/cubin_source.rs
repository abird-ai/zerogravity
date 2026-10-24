//! Only exact raw bytes of an already verified retained provider FD.
use super::*;
use crate::owned_command::cubin::{CubinProvider, VerifiedCubin};
use std::any::Any;
use std::os::unix::fs::FileExt;

impl Bf16BlasSources {
    /// No decompression, unrelated owner or caller-asserted derived membership.
    ///
    /// # Safety
    /// `admission_owner` holds original host image admission (max32MiB) AND
    /// this original queue/service's existing native-module budget, admitted
    /// before allocation. It must retain both until actual unload or original
    /// quarantine, never substitute a second arbiter/charge. The sealed image
    /// retains it through partial load, token drop and unload errors.
    /// Expected hash is the pinned actual module hash, not arbitrary ELF qualification.
    pub unsafe fn cubin(
        self: &Arc<Self>,
        admission_owner: Arc<dyn Any + Send + Sync>,
        provider: CubinProvider,
        offset: u64,
        bytes: usize,
        expected: [u8; 32],
    ) -> Result<Arc<VerifiedCubin>, ServerError> {
        self.require_queue(&self.queue)?;
        let file = match provider {
            CubinProvider::Cublas => &self.cublas,
            CubinProvider::CublasLt => &self.lt,
        };
        if offset
            .checked_add(bytes as u64)
            .is_none_or(|end| end > file.stamp[2])
        {
            return Err(failure("cubin raw provider interval outside original FD"));
        }
        VerifiedCubin::read_verified(self.clone(), admission_owner, bytes, expected, |target| {
            file.unchanged()?;
            let mut read = 0;
            while read < target.len() {
                let n = file
                    .file
                    .read_at(&mut target[read..], offset + read as u64)
                    .map_err(|e| failure(format!("cubin original FD read: {e}")))?;
                if n == 0 {
                    return Err(failure("cubin original FD premature EOF"));
                }
                read += n;
            }
            file.unchanged()
        })
    }
}
