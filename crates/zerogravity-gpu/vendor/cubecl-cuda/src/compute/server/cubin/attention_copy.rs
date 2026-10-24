//! Exact original N1 FA output copy, SAME actor/context/stream/access scope.
use super::*;

fn ranges(source: u64, destination: u64) -> Result<(), ServerError> {
    let source_end = source
        .checked_add(4096)
        .ok_or_else(|| failure("attention source copy range overflow"))?;
    let destination_end = destination
        .checked_add(4096)
        .ok_or_else(|| failure("attention destination copy range overflow"))?;
    if source == 0
        || destination == 0
        || !source.is_multiple_of(16)
        || !destination.is_multiple_of(16)
        || (source < destination_end && destination < source_end)
    {
        return Err(failure(
            "attention original copy nonalias aligned4096B ranges",
        ));
    }
    Ok(())
}
impl CudaServer {
    pub(crate) fn copy_owned_attention_output(
        &mut self,
        stream: StreamId,
        set_key: u64,
        key: u64,
        source: Handle,
        destination: Handle,
    ) -> Result<(), ServerError> {
        let set = self
            .cubin_set
            .as_ref()
            .ok_or_else(|| failure("original attention set absent"))?;
        set.require_active(set_key)?;
        if !matches!(
            set.policy,
            CubinSetPolicy::ShortHf08
                | CubinSetPolicy::ShortHf08Cohort4
                | CubinSetPolicy::ShortHf2B
                | CubinSetPolicy::ShortHf2BCachedCohort4
                | CubinSetPolicy::FirstAttentionCaptured
                | CubinSetPolicy::FirstAttentionComputedGate
                | CubinSetPolicy::FullQwen08
                | CubinSetPolicy::Canonical47Natural
        ) {
            return Err(failure("attention copy outside exact captured-leaf set"));
        }
        let record = set
            .records
            .get(&key)
            .filter(|r| r.ready)
            .ok_or_else(|| failure("attention final N1 record absent"))?;
        let image = set
            .modules
            .get(&record.image)
            .filter(|m| m.ready)
            .ok_or_else(|| failure("attention original image module absent"))?;
        if matches!(
            set.policy,
            CubinSetPolicy::ShortHf08
                | CubinSetPolicy::ShortHf08Cohort4
                | CubinSetPolicy::ShortHf2B
                | CubinSetPolicy::ShortHf2BCachedCohort4
        ) {
            crate::owned_command::cubin::short_hf_policy::require(
                set.policy,
                &image.image,
                &record.manifest,
            )?;
            crate::owned_command::cubin::short_hf_policy::require_n1_copy(
                &image.image,
                &record.manifest,
            )?;
        } else if set.policy == CubinSetPolicy::Canonical47Natural {
            crate::owned_command::cubin::canonical47_natural_policy::require_n1_copy(
                &image.image,
                &record.manifest,
            )?;
        } else {
            crate::owned_command::cubin::attention_policy::require_n1_copy(
                &image.image,
                &record.manifest,
            )?;
        }
        let original_service = source.service == self.utilities.service
            && destination.service == self.utilities.service;
        let native = [source.binding(), destination.binding()];
        let mut written = self.write_set();
        written.push(native[1].clone());
        launch_scope(self, stream, key, std::iter::once(&native[0]), written)
            .execute(move |server| {
                if !original_service {
                    return Err(failure("attention original copy service differs"));
                }
                let mut command = server.command(stream, native.iter())?;
                if command.stream().capturing.is_recording() {
                    return Err(failure("attention copy graph capture unsupported"));
                }
                let source = command.resource(native[0].clone())?;
                let destination = command.resource(native[1].clone())?;
                if source.size != 4096 || destination.size != 4096 {
                    return Err(failure("attention original copy exact resource extents"));
                }
                ranges(source.ptr, destination.ptr)?;
                // Full original resources/context/stream and nonalias ranges were
                // validated before enqueue. Failure keeps original output taint;
                // caller's armed flight follows the same uncertain quarantine.
                status(
                    unsafe {
                        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                            destination.ptr,
                            source.ptr,
                            4096,
                            command.stream().sys,
                        )
                    },
                    "attention4096B cuMemcpyDtoDAsync_v2",
                )
            })
            .into_result()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn copy_ranges_refuse_alias_overflow_and_misaligned_addresses() {
        assert!(ranges(0x1000, 0x2000).is_ok());
        for (a, b) in [
            (0, 0x2000),
            (0x1000, 0),
            (0x1000, 0x1000),
            (0x1000, 0x1ff0),
            (0x1001, 0x3000),
            (u64::MAX - 15, 0x1000),
            (0x1000, u64::MAX - 15),
        ] {
            assert!(ranges(a, b).is_err());
        }
    }
}
