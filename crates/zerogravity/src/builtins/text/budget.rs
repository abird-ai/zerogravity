use super::*;
pub(super) fn session_budget(model: &LoadedQwenModel, context: usize) -> Result<u64, ServiceError> {
    let mut bytes = 0u64;
    for spec in model.state_specs() {
        let count = match *spec {
            StateSpec::Delta { conv, state } => {
                conv.checked_add(state).and_then(|n| n.checked_mul(8))
            }
            StateSpec::Full { kv_width } => kv_width
                .checked_mul(context)
                .and_then(|n| n.checked_mul(16)),
        }
        .ok_or_else(|| {
            ServiceError::new(
                400,
                "state_budget",
                "source session state capacity overflow",
            )
        })?;
        bytes = bytes
            .checked_add(count as u64)
            .ok_or_else(|| ServiceError::new(400, "state_budget", "total state budget overflow"))?;
    }
    // Activation/program/logits scratch envelope is conservative, not
    // a claim of actual driver/device memory.
    bytes = bytes
        .checked_add(32 * 1024 * 1024)
        .ok_or_else(|| ServiceError::new(400, "state_budget", "scratch envelope overflow"))?;
    Ok(bytes)
}
