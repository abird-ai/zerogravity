//! Exact original N1 FA output copy, SAME actor/context/stream/access scope.
use super::*;

#[cfg(test)]
fn ranges(source: u64, destination: u64) -> Result<(), ServerError> {
    super::super::owned_transfer::require_copy_ranges(source, destination, 4096, 16)
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
            CubinSetPolicy::Text08
                | CubinSetPolicy::Text08FreshPrefill
                | CubinSetPolicy::Text08Cohort4
                | CubinSetPolicy::Text2B
                | CubinSetPolicy::Text2BCachedCohort4
                | CubinSetPolicy::FirstAttentionCaptured
                | CubinSetPolicy::FirstAttentionComputedGate
                | CubinSetPolicy::FullQwen800M
                | CubinSetPolicy::FixedPromptNatural
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
            CubinSetPolicy::Text08
                | CubinSetPolicy::Text08FreshPrefill
                | CubinSetPolicy::Text08Cohort4
                | CubinSetPolicy::Text2B
                | CubinSetPolicy::Text2BCachedCohort4
        ) {
            crate::owned_command::cubin::text_policy::require(
                set.policy,
                &image.image,
                &record.manifest,
            )?;
            crate::owned_command::cubin::text_policy::require_n1_copy(
                &image.image,
                &record.manifest,
            )?;
        } else if set.policy == CubinSetPolicy::FixedPromptNatural {
            crate::owned_command::cubin::fixed_prompt_natural_policy::require_n1_copy(
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
                super::super::owned_transfer::copy_exact(
                    &mut command,
                    native[0].clone(),
                    native[1].clone(),
                    4096,
                    16,
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
