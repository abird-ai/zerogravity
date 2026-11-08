//! Original source plan + resident owners on the existing engine worker.
use crate::{
    gpu::{SourceCallModel, SourceCallSession},
    identity::SourceIdentity,
    qwen35::plan::OrdinaryQwenPlan,
    resource_arbiter::{AllocationClass, GlobalResourceLease},
};
use std::sync::{Arc, Mutex};
use zerogravity_engine::{
    model_instance::{PreparedModel, TextTokenizer},
    router::{ServiceError, pending::ServingHost},
};
mod execute;
mod load;
pub(crate) use load::{SourceBackend, load_router};

/// One cold raw-text source, shared by aliases. No native cross-session batching or warm
/// runtime reconfiguration. Host/provider/context bounds remain backend-owned.
#[derive(Clone, Copy, Debug)]
pub struct QwenSourceProfile {
    pub active: usize,
    pub context: usize,
    pub output: usize,
    pub max_pending: Option<usize>,
}
impl QwenSourceProfile {
    pub(crate) fn validate(self) -> Result<(), ServiceError> {
        use crate::qwen35::plan::{ORDINARY_SOURCE_MAX_CONTEXT, ORDINARY_SOURCE_MIN_CONTEXT};
        if !(1..=zerogravity_engine::router::MAX_ACTIVE_PER_WORKER).contains(&self.active)
            || !(ORDINARY_SOURCE_MIN_CONTEXT..=ORDINARY_SOURCE_MAX_CONTEXT).contains(&self.context)
            || self.output == 0
            || self.output > self.context
            || self.output > zerogravity_engine::router::MAX_OUTPUT_TOKENS
            || self.max_pending == Some(0)
        {
            return Err(ServiceError::new(
                400,
                "invalid_source_profile",
                "source GPU requires active=1..8, context=512..32768, output=1..512, positive pending bound",
            ));
        }
        // Keep every shared worker refusal before source IO or device creation.
        zerogravity_engine::router::validate_config(
            1,
            1,
            self.active,
            self.context,
            self.output,
            self.max_pending,
        )
    }
}
struct Source {
    plan: OrdinaryQwenPlan,
    metadata: crate::qwen35_source::Metadata,
    _snapshot: Arc<crate::cache::SnapshotLease>,
    // Source/canonical metadata and plan allocation, not device/root storage.
    _control: GlobalResourceLease,
}
struct Model {
    resident: Mutex<Option<Arc<SourceCallModel>>>,
    source: Arc<Source>,
    tokenizer: Arc<dyn TextTokenizer>,
    max_token_bytes: usize,
    _metadata: Arc<zerogravity_engine::router::pending::ServingCharge>,
}
impl Model {
    fn resident(&self) -> Result<Arc<SourceCallModel>, ServiceError> {
        self.resident
            .lock()
            .map_err(|_| poison())?
            .as_ref()
            .cloned()
            .ok_or_else(|| ServiceError::new(503, "model_retired", "original source model retired"))
    }
}
fn poison() -> ServiceError {
    ServiceError::new(500, "source_owner", "original model owner lock poisoned")
}
impl PreparedModel for Model {
    fn validate_chat(&self) -> Result<(), ServiceError> {
        Err(ServiceError::new(
            400,
            "unsupported_chat",
            "original source GPU currently accepts raw text only",
        ))
    }
    fn validate_request(
        &self,
        request: &zerogravity_engine::router::Request,
    ) -> Result<(), ServiceError> {
        if request.chat {
            self.validate_chat()
        } else {
            Ok(())
        }
    }
    fn source(&self) -> SourceIdentity {
        self.source.plan.source_seal().source_identity()
    }
    fn tokenizer(&self) -> Arc<dyn TextTokenizer> {
        self.tokenizer.clone()
    }
    fn max_context(&self) -> usize {
        self.source.plan.context_capacity_tokens()
    }
    fn max_token_bytes(&self) -> usize {
        self.max_token_bytes
    }
    fn vocabulary_size(&self) -> u64 {
        self.source.plan.model_profile().vocabulary_size() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_profile_refuses_unsupported_limits_before_io() {
        let good = QwenSourceProfile {
            active: 2,
            context: 514,
            output: 2,
            max_pending: Some(4),
        };
        assert!(good.validate().is_ok());
        for context in [8192, 16384, 32768] {
            assert!(QwenSourceProfile { context, ..good }.validate().is_ok());
        }
        for bad in [
            QwenSourceProfile { active: 0, ..good },
            QwenSourceProfile { active: 9, ..good },
            QwenSourceProfile {
                context: 511,
                ..good
            },
            QwenSourceProfile {
                context: 32769,
                ..good
            },
            QwenSourceProfile {
                context: 16384,
                output: 513,
                ..good
            },
            QwenSourceProfile {
                output: 515,
                ..good
            },
            QwenSourceProfile {
                max_pending: Some(0),
                ..good
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }
}
