//! Phased user text preparation; expected archive never supplies input IDs.
use super::io::*;
use serde_json::{Value, json};
use std::path::Path;
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
use zerogravity_model_gemma4::text::{
    OriginalGemma4Text,
    tokenizer::{Gemma4PreparedText, Gemma4TextTokenizer, Gemma4TokenizerFiles},
};

pub(super) struct Input {
    prepared: Gemma4PreparedText,
    phases: Value,
}
impl Input {
    pub fn prepare(directory: &Path, text: &str, host: &TransferAdmission) -> Proof<Self> {
        let before = host.arbiter.snapshot(&host.resource)?.reserved;
        let control = host.arbiter.admit(
            &host.resource,
            host.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 4 * 1024 * 1024,
            }],
        )?;
        let tokenizer = Gemma4TextTokenizer::prepare(
            Gemma4TokenizerFiles {
                tokenizer: open(&directory.join("tokenizer.json"))?,
                tokenizer_config: open(&directory.join("tokenizer_config.json"))?,
                generation_config: open(&directory.join("generation_config.json"))?,
                model_config: open(&directory.join("config.json"))?,
            },
            host,
            None,
        )?;
        let constructed = host.arbiter.snapshot(&host.resource)?.reserved;
        let prepared = tokenizer.into_plain_text(text, None)?;
        let retired = host.arbiter.snapshot(&host.resource)?.reserved;
        need(
            constructed == before + (4 + 64 + 8) * 1024 * 1024
                && retired == before + 4 * 1024 * 1024 + 65536,
            "parsed tokenizer/source owners retire before any learned model load",
        )?;
        drop(control);
        let ready = host.arbiter.snapshot(&host.resource)?.reserved;
        need(
            ready == before + 65536,
            "only bounded prepared IDs/source survive",
        )?;
        Ok(Self {
            prepared,
            phases: json!({"input":"actual user plain text","text":text,
                "before_reserved":before,"constructed_reserved":constructed,
                "after_tokenizer_retirement_reserved":retired,"model_phase_input_reserved":ready,
                "tokenizer_construction_admission_upper":128*1024*1024,
                "tokenizer_retired_before_model":true,"expected_ids_used_as_operands":false}),
        })
    }
    pub fn require_model(&self, model: &OriginalGemma4Text) -> Proof<()> {
        self.prepared.require_model(model)?;
        need(
            self.prepared.eos_ids() == [1, 106, 50],
            "authentic source stop metadata",
        )
    }
    pub fn ids(&self) -> &[u32] {
        self.prepared.ids()
    }
    pub fn after(&self) -> Proof<()> {
        Ok(self.prepared.verify_source(None)?)
    }
    pub fn report(&self) -> Value {
        let mut report = self.phases.clone();
        report["actual_prepared_ids"] = json!(self.ids());
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    #[test]
    fn real_phase_pressure_refuses_before_opening_source_and_releases_control() {
        let host = TransferAdmission {
            arbiter: ResourceArbiter::new(),
            resource: PhysicalDeviceKey::new("host", "plain text fixture").unwrap(),
            owner: ResourceOwner::new("Gemma", "phased input").unwrap(),
        };
        host.arbiter
            .register(host.resource.clone(), 4 * 1024 * 1024 - 1)
            .unwrap();
        assert!(Input::prepare(Path::new("/no-such-Gemma-source"), "real text", &host).is_err());
        assert_eq!(host.arbiter.snapshot(&host.resource).unwrap().reserved, 0);
    }
}
