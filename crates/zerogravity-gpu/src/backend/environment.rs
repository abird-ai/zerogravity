use super::contract::Provider;
use zerogravity_core::{Error, Result};

pub(super) fn require_runtime(
    provider: Provider,
    hip_available: impl FnOnce() -> bool,
) -> Result<()> {
    if provider == Provider::Hip && !hip_available() {
        return Err(Error::Unsupported(
            "resident HIP runtime/RTC libraries are unavailable; set HIP_PATH/ROCM_PATH to the approved SDK before the first runtime lookup"
                .into(),
        ));
    }
    Ok(())
}

pub(super) fn hip_available() -> bool {
    #[cfg(feature = "hip")]
    {
        cubecl::hip::is_runtime_available()
    }
    #[cfg(not(feature = "hip"))]
    {
        false
    }
}
