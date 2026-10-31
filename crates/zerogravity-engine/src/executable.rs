//! Prepared resources retain one original semantic program; no second model IR.
use std::sync::Arc;
use zerogravity_core::{provider::PreparedExecutable, semantic::ForwardPlan};
pub struct ExecutableBundle {
    prepared: Arc<dyn PreparedExecutable>,
}
impl ExecutableBundle {
    pub fn program(&self) -> &ForwardPlan {
        self.prepared.program()
    }
}
